"""Generation for the MLX Qwen3-TTS runtime adapter.

The call made to the model, the typed failures around it, and the lazy
mlx-audio import. Sample-to-WAV conversion lives in
`test_mlx_qwen_audio_conversion.py`; config and path validation in
`test_mlx_qwen_runtime_config.py`.

`mlx_audio` is not installed here and is never imported by these tests: the
loader is injected. Nothing below proves the real model works — see
`test_no_real_model_is_claimed`.
"""

import io
import json
import wave
from pathlib import Path

import pytest

from voiceclonegpt.synthesis import mlx_qwen_runtime
from voiceclonegpt.synthesis.mlx_qwen_runtime import (
    MlxQwenRuntime,
    SynthesisError,
)


class FakeResult:
    def __init__(self, audio):
        self.audio = audio


class FakeModel:
    """Stands in for an mlx_audio TTS model."""

    def __init__(self, audio=None):
        self.audio = [0.0, 0.5, -0.5, 1.0] if audio is None else audio
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield FakeResult(self.audio)


def fake_loader(model=None):
    """Build a loader that records the locator it was given."""
    seen = []

    def loader(locator):
        seen.append(locator)
        return model if model is not None else FakeModel()

    loader.seen = seen
    return loader


@pytest.fixture
def bundle(tmp_path):
    """A runtime directory with a config, a model dir, and a reference clip."""
    root = tmp_path / "runtime"
    (root / "model").mkdir(parents=True)
    (root / "model" / "weights.safetensors").write_bytes(b"")
    (root / "ref").mkdir()
    (root / "ref" / "neutral.wav").write_bytes(b"RIFF....WAVE")

    config = {
        "model_locator": "model",
        "ref_audio": "ref/neutral.wav",
        "ref_text": "This is the neutral reading.",
        "sample_rate": 24000,
    }
    path = root / "runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def write_config(path, **overrides):
    """Write a config with fields replaced or removed (value `None` deletes)."""
    config = json.loads(path.read_text())
    for key, value in overrides.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def read_wav(payload):
    with wave.open(io.BytesIO(payload), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        return {
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frame_rate": handle.getframerate(),
            "frames": handle.getnframes(),
            "raw": frames,
        }


class TestSynthesis:
    """Generation goes through the model and returns real WAV bytes."""

    def _handle(self, bundle, model):
        return MlxQwenRuntime(loader=fake_loader(model)).load(bundle, {})

    def test_passes_reference_audio_and_text(self, bundle):
        model = FakeModel()
        runtime = MlxQwenRuntime(loader=fake_loader(model))
        handle = runtime.load(bundle, {})

        runtime.synthesize(handle, "Hello there.")

        call = model.calls[0]
        assert call["text"] == "Hello there."
        assert call["ref_audio"] == str((bundle.parent / "ref/neutral.wav").resolve())
        assert call["ref_text"] == "This is the neutral reading."

    def test_returns_a_valid_wav(self, bundle):
        model = FakeModel()
        runtime = MlxQwenRuntime(loader=fake_loader(model))
        handle = runtime.load(bundle, {})

        info = read_wav(runtime.synthesize(handle, "Hello."))

        assert info["channels"] == 1
        assert info["sample_width"] == 2
        assert info["frame_rate"] == 24000
        assert info["frames"] == 4

    def test_uses_the_configured_sample_rate(self, bundle):
        write_config(bundle, sample_rate=16000)
        model = FakeModel()
        runtime = MlxQwenRuntime(loader=fake_loader(model))
        handle = runtime.load(bundle, {})

        assert read_wav(runtime.synthesize(handle, "Hi."))["frame_rate"] == 16000

    def test_conversion_is_deterministic(self, bundle):
        model = FakeModel()
        runtime = MlxQwenRuntime(loader=fake_loader(model))
        handle = runtime.load(bundle, {})

        assert runtime.synthesize(handle, "Same.") == runtime.synthesize(
            handle, "Same."
        )

    def test_only_the_first_result_is_consumed(self, bundle):
        class TwoResults(FakeModel):
            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield FakeResult([0.0, 0.0])
                raise AssertionError("second result must not be requested")

        runtime = MlxQwenRuntime(loader=fake_loader(TwoResults()))
        handle = runtime.load(bundle, {})

        assert read_wav(runtime.synthesize(handle, "Hi."))["frames"] == 2

    def test_accepts_array_like_audio(self, bundle):
        """mlx/numpy arrays expose `.tolist()`."""

        class ArrayLike:
            def tolist(self):
                return [0.25, -0.25]

        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel(ArrayLike())))
        handle = runtime.load(bundle, {})

        assert read_wav(runtime.synthesize(handle, "Hi."))["frames"] == 2

    def test_audio_is_never_tested_for_truthiness(self, bundle):
        """An mx.array raises on `bool()` — the adapter must not ask.

        numpy and mlx arrays with more than one element raise
        "truth value of an array is ambiguous" from `__bool__`. Any
        `audio or []` / `if audio:` on the model output crashes on real
        output while passing against list-based fakes.
        """

        class AmbiguousArray:
            def __bool__(self):
                raise ValueError(
                    "The truth value of an array with more than one element "
                    "is ambiguous"
                )

            def tolist(self):
                return [0.25, -0.25, 0.75]

        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel(AmbiguousArray())))
        handle = runtime.load(bundle, {})

        assert read_wav(runtime.synthesize(handle, "Hi."))["frames"] == 3

    def test_missing_audio_attribute_is_a_typed_error(self, bundle):
        """A result object without `.audio` must not become a crash."""

        class NoAudio:
            pass

        class Model(FakeModel):
            def generate(self, **kwargs):
                self.calls.append(kwargs)
                yield NoAudio()

        runtime = MlxQwenRuntime(loader=fake_loader(Model()))
        handle = runtime.load(bundle, {})

        with pytest.raises(SynthesisError, match="no audio"):
            runtime.synthesize(handle, "Hi.")


class TestSynthesisFailures:
    """Every failure is typed; none leaks a provider exception."""

    def _runtime_and_handle(self, bundle, model):
        runtime = MlxQwenRuntime(loader=fake_loader(model))
        return runtime, runtime.load(bundle, {})

    def test_no_results_is_an_error(self, bundle):
        class Empty(FakeModel):
            def generate(self, **kwargs):
                return iter(())

        runtime, handle = self._runtime_and_handle(bundle, Empty())

        with pytest.raises(SynthesisError, match="no audio"):
            runtime.synthesize(handle, "Hi.")

    def test_empty_audio_is_an_error(self, bundle):
        runtime, handle = self._runtime_and_handle(bundle, FakeModel([]))

        with pytest.raises(SynthesisError, match="no audio"):
            runtime.synthesize(handle, "Hi.")

    def test_empty_text_is_rejected_before_generation(self, bundle):
        model = FakeModel()
        runtime, handle = self._runtime_and_handle(bundle, model)

        with pytest.raises(SynthesisError, match="text"):
            runtime.synthesize(handle, "   ")

        assert model.calls == []

    def test_model_failure_is_wrapped(self, bundle):
        class Broken(FakeModel):
            def generate(self, **kwargs):
                raise RuntimeError("mlx kernel exploded")

        runtime, handle = self._runtime_and_handle(bundle, Broken())

        with pytest.raises(SynthesisError, match="generation failed") as excinfo:
            runtime.synthesize(handle, "Hi.")

        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_non_numeric_samples_are_rejected(self, bundle):
        runtime, handle = self._runtime_and_handle(bundle, FakeModel(["a", "b"]))

        with pytest.raises(SynthesisError, match="numeric"):
            runtime.synthesize(handle, "Hi.")

    def test_oversized_audio_is_rejected(self, bundle, monkeypatch):
        monkeypatch.setattr(mlx_qwen_runtime, "MAX_OUTPUT_SAMPLES", 3)
        runtime, handle = self._runtime_and_handle(bundle, FakeModel([0.0] * 10))

        with pytest.raises(SynthesisError, match="too long"):
            runtime.synthesize(handle, "Hi.")


class TestDependencySeam:
    """mlx_audio is optional and imported lazily."""

    def test_missing_mlx_audio_raises_a_clear_import_error(self, bundle, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("mlx_audio"):
                raise ImportError("No module named 'mlx_audio'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="mlx-audio"):
            MlxQwenRuntime().load(bundle, {})

    def test_module_does_not_import_mlx_at_module_scope(self):
        source = Path(mlx_qwen_runtime.__file__).read_text(encoding="utf-8")
        imports = [
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        ]

        assert not any("mlx" in line for line in imports)

    def test_imports_no_numpy_or_soundfile(self):
        source = Path(mlx_qwen_runtime.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("numpy", "soundfile", "librosa", "torch", "requests"):
            assert banned not in imports

    def test_no_real_model_is_claimed(self):
        """This suite proves plumbing, not audio quality.

        Every test above injects a fake loader and a fake model. Nothing here
        has loaded Qwen3-TTS or produced speech; that requires `mlx-audio`
        installed and a locally downloaded model, and must be verified by
        listening. See the module CONTEXT.
        """
        assert mlx_qwen_runtime.IS_VERIFIED_AGAINST_REAL_MODEL is False

