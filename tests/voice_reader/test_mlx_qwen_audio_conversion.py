"""Audio conversion for the MLX Qwen3-TTS runtime adapter.

Everything between "the model handed us some samples" and "here are WAV bytes":
int16 conversion and clamping, bounded materialization of untrusted output,
refusal of non-finite values, and wrapping of provider-side `.tolist()` and
iteration failures.

Generation and the dependency seam live in `test_mlx_qwen_runtime.py`; config
and path validation in `test_mlx_qwen_runtime_config.py`. `mlx_audio` is never
imported here.
"""

import io
import json
import wave

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


class TestSampleConversion:
    """Float samples become int16 PCM, clamped, never wrapped."""

    def _samples(self, bundle, audio):
        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel(audio)))
        handle = runtime.load(bundle, {})
        info = read_wav(runtime.synthesize(handle, "Hi."))
        return list(
            int.from_bytes(info["raw"][i : i + 2], "little", signed=True)
            for i in range(0, len(info["raw"]), 2)
        )

    def test_silence_maps_to_zero(self, bundle):
        assert self._samples(bundle, [0.0, 0.0]) == [0, 0]

    def test_full_scale_maps_to_the_int16_limits(self, bundle):
        assert self._samples(bundle, [1.0, -1.0]) == [32767, -32767]

    def test_out_of_range_values_clamp_rather_than_wrap(self, bundle):
        """A wrapped sample is a loud click; clamping is the safe failure."""
        assert self._samples(bundle, [9.0, -9.0, 1.5]) == [32767, -32767, 32767]

    def test_integer_samples_are_treated_as_full_scale_units(self, bundle):
        assert self._samples(bundle, [1, 0, -1]) == [32767, 0, -32767]

    def test_midscale_value(self, bundle):
        assert self._samples(bundle, [0.5]) == [16384]


class TestBoundedMaterialization:
    """Model output is untrusted: never materialize it unbounded."""

    def test_endless_audio_is_cut_off_not_consumed_forever(self, bundle, monkeypatch):
        """A generator that never stops must not hang or exhaust memory."""
        monkeypatch.setattr(mlx_qwen_runtime, "MAX_OUTPUT_SAMPLES", 4)
        produced = []

        def endless():
            while True:
                produced.append(1)
                yield 0.0

        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel(endless())))
        handle = runtime.load(bundle, {})

        with pytest.raises(SynthesisError, match="too long"):
            runtime.synthesize(handle, "Hi.")

        # One past the cap is enough to know it is over; never more.
        assert len(produced) <= 5

    def test_oversized_finite_audio_is_rejected(self, bundle, monkeypatch):
        monkeypatch.setattr(mlx_qwen_runtime, "MAX_OUTPUT_SAMPLES", 3)
        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel([0.0] * 50)))
        handle = runtime.load(bundle, {})

        with pytest.raises(SynthesisError, match="too long"):
            runtime.synthesize(handle, "Hi.")

    def test_audio_at_the_cap_is_accepted(self, bundle, monkeypatch):
        monkeypatch.setattr(mlx_qwen_runtime, "MAX_OUTPUT_SAMPLES", 4)
        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel([0.0] * 4)))
        handle = runtime.load(bundle, {})

        assert read_wav(runtime.synthesize(handle, "Hi."))["frames"] == 4


class TestHostileSampleValues:
    """Non-finite samples must be refused, not crash the converter."""

    def _runtime(self, bundle, audio):
        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel(audio)))
        return runtime, runtime.load(bundle, {})

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")]
    )
    def test_non_finite_samples_are_rejected(self, bundle, value):
        runtime, handle = self._runtime(bundle, [0.0, value, 0.0])

        with pytest.raises(SynthesisError, match="finite"):
            runtime.synthesize(handle, "Hi.")

    def test_nan_does_not_leak_a_value_error(self, bundle):
        """int(round(nan)) raises ValueError; that must not escape."""
        runtime, handle = self._runtime(bundle, [float("nan")])

        with pytest.raises(SynthesisError):
            runtime.synthesize(handle, "Hi.")

    def test_infinity_does_not_leak_an_overflow_error(self, bundle):
        """int(round(inf)) raises OverflowError; that must not escape."""
        runtime, handle = self._runtime(bundle, [float("inf")])

        with pytest.raises(SynthesisError):
            runtime.synthesize(handle, "Hi.")


class TestAudioConversionFailures:
    """`.tolist()` and iteration are provider code; wrap their failures."""

    def _runtime(self, bundle, audio):
        runtime = MlxQwenRuntime(loader=fake_loader(FakeModel(audio)))
        return runtime, runtime.load(bundle, {})

    def test_tolist_failure_is_wrapped(self, bundle):
        class BrokenArray:
            def tolist(self):
                raise RuntimeError("mlx eval failed")

        runtime, handle = self._runtime(bundle, BrokenArray())

        with pytest.raises(SynthesisError, match="audio") as excinfo:
            runtime.synthesize(handle, "Hi.")

        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_iteration_failure_is_wrapped(self, bundle):
        def exploding():
            yield 0.0
            raise RuntimeError("stream died mid-generation")

        runtime, handle = self._runtime(bundle, exploding())

        with pytest.raises(SynthesisError, match="audio") as excinfo:
            runtime.synthesize(handle, "Hi.")

        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_non_iterable_audio_is_wrapped(self, bundle):
        runtime, handle = self._runtime(bundle, 42)

        with pytest.raises(SynthesisError, match="audio"):
            runtime.synthesize(handle, "Hi.")

