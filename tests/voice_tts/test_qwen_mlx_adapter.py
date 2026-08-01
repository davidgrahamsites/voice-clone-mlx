"""The Qwen3-TTS adapter, which delegates to `synthesis.mlx_qwen_runtime`.

The adapter under test owns exactly two things the runtime does not: writing
the returned WAV bytes to a path, and measuring what was written. Everything
else — the lazy mlx-audio import, config validation, path confinement, sample
conversion — belongs to the runtime and is tested there, not restated here.

mlx-audio is not installed in this workspace and nothing here installs it.
"""

import importlib.util
import io
import json
import wave

import pytest

from voiceclonemlx.synthesis.mlx_qwen_runtime import RuntimeConfigError
from voiceclonemlx.tts import qwen_mlx
from voiceclonemlx.tts.backend import SynthesisError, SynthesisResult, TTSBackend

REF_TEXT = "This is the neutral reading."


def wav_bytes(frames: int, sample_rate: int = 24_000) -> bytes:
    """A valid mono PCM16 WAV, standing in for the runtime's output."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


class FakeRuntime:
    """Stands in for `MlxQwenRuntime`: the same `load` / `synthesize` shape."""

    def __init__(self, audio=None, ref_audio="ref.wav", load_error=None):
        self.load_calls = []
        self.synthesize_calls = []
        self._audio = wav_bytes(2_400) if audio is None else audio
        self._ref_audio = ref_audio
        self._load_error = load_error

    def load(self, artifact_path, manifest):
        self.load_calls.append((artifact_path, manifest))
        if self._load_error is not None:
            raise self._load_error
        return {"runtime": "mlx_qwen", "config": {"ref_audio": self._ref_audio}}

    def synthesize(self, handle, text):
        self.synthesize_calls.append((handle, text))
        return self._audio


@pytest.fixture
def config_path(tmp_path):
    """A runtime config the real `MlxQwenRuntime` would accept."""
    (tmp_path / "model").mkdir()
    (tmp_path / "ref.wav").write_bytes(wav_bytes(240))
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_locator": "model",
                "ref_audio": "ref.wav",
                "ref_text": REF_TEXT,
                "sample_rate": 24_000,
            }
        ),
        encoding="utf-8",
    )
    return path


def make_backend(runtime, config_path, **kwargs):
    outcome = qwen_mlx.create_qwen_mlx_backend(config_path, runtime=runtime, **kwargs)
    assert outcome.available, outcome.reason
    return outcome.backend


class TestItStaysThin:
    def test_importing_the_module_does_not_import_mlx_audio(self):
        import sys

        assert "mlx_audio" not in sys.modules

    def test_the_module_never_imports_mlx_audio(self):
        source = open(qwen_mlx.__file__, encoding="utf-8").read()
        for line in source.splitlines():
            if line.startswith(("import ", "from ")):
                assert "mlx_audio" not in line

    def test_audio_conversion_is_not_re_implemented_here(self):
        for owned_by_the_runtime in ("samples_to_wav", "_to_sample_list"):
            assert not hasattr(qwen_mlx, owned_by_the_runtime), (
                f"{owned_by_the_runtime} belongs to synthesis.mlx_qwen_runtime"
            )

    def test_the_adapter_is_not_claimed_to_be_verified(self):
        assert qwen_mlx.IS_VERIFIED_AGAINST_REAL_MODEL is False

    def test_mlx_audio_is_genuinely_absent_here(self):
        assert importlib.util.find_spec("mlx_audio") is None


class TestFactory:
    def test_an_injected_runtime_yields_an_available_backend(self, config_path):
        outcome = qwen_mlx.create_qwen_mlx_backend(config_path, runtime=FakeRuntime())

        assert outcome.available is True
        assert outcome.reason is None
        assert isinstance(outcome.backend, TTSBackend)

    def test_the_config_is_handed_straight_to_the_runtime(self, config_path):
        runtime = FakeRuntime()
        manifest = {"bundle_schema_version": "1.0.0"}

        qwen_mlx.create_qwen_mlx_backend(config_path, manifest=manifest, runtime=runtime)

        assert runtime.load_calls == [(config_path, manifest)]

    def test_missing_mlx_audio_is_reported_not_raised(self, config_path):
        runtime = FakeRuntime(
            load_error=ImportError(
                "Running the Qwen3-TTS runtime requires the mlx-audio package."
            )
        )

        outcome = qwen_mlx.create_qwen_mlx_backend(config_path, runtime=runtime)

        assert outcome.available is False
        assert outcome.backend is None
        assert "mlx-audio" in outcome.reason

    def test_a_refused_config_is_reported_not_raised(self, config_path):
        runtime = FakeRuntime(load_error=RuntimeConfigError("ref_audio is not a file"))

        outcome = qwen_mlx.create_qwen_mlx_backend(config_path, runtime=runtime)

        assert outcome.available is False
        assert outcome.backend is None
        assert "ref_audio is not a file" in outcome.reason

    def test_the_real_runtime_is_unavailable_in_this_workspace(self, config_path):
        """The whole delegation path, with nothing faked. mlx-audio is absent."""
        outcome = qwen_mlx.create_qwen_mlx_backend(config_path)

        assert outcome.available is False
        assert outcome.backend is None
        assert "mlx-audio" in outcome.reason

    def test_a_bad_config_is_reported_through_the_real_runtime(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{ not json", encoding="utf-8")

        outcome = qwen_mlx.create_qwen_mlx_backend(path)

        assert outcome.available is False
        assert outcome.reason

    def test_outcome_is_frozen(self, config_path):
        outcome = qwen_mlx.create_qwen_mlx_backend(config_path, runtime=FakeRuntime())

        with pytest.raises(Exception):
            outcome.available = False


class TestOutcomeInvariant:
    """An outcome must be consistent however it was built, not just via the
    factory. A caller that trusts `available` must find the matching field
    populated and the other one empty."""

    def test_an_available_outcome_carries_a_backend_and_no_reason(self):
        backend = object()

        outcome = qwen_mlx.BackendOutcome(available=True, backend=backend)

        assert outcome.backend is backend
        assert outcome.reason is None

    def test_an_unavailable_outcome_carries_a_reason_and_no_backend(self):
        outcome = qwen_mlx.BackendOutcome(available=False, reason="no mlx-audio")

        assert outcome.reason == "no mlx-audio"
        assert outcome.backend is None

    def test_available_without_a_backend_is_refused(self):
        with pytest.raises(ValueError, match="must carry a backend"):
            qwen_mlx.BackendOutcome(available=True)

    def test_available_with_a_reason_is_refused(self):
        with pytest.raises(ValueError, match="must not carry a reason"):
            qwen_mlx.BackendOutcome(
                available=True, backend=object(), reason="it worked, but"
            )

    def test_unavailable_without_a_reason_is_refused(self):
        with pytest.raises(ValueError, match="must say why"):
            qwen_mlx.BackendOutcome(available=False)

    def test_unavailable_with_a_backend_is_refused(self):
        with pytest.raises(ValueError, match="must not carry a backend"):
            qwen_mlx.BackendOutcome(
                available=False, backend=object(), reason="broken anyway"
            )

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t", " "])
    def test_a_blank_reason_is_refused(self, blank):
        """A reason exists to be read by a person. Whitespace tells them
        nothing while still passing an `is not None` check."""
        with pytest.raises(ValueError, match="must not be blank"):
            qwen_mlx.BackendOutcome(available=False, reason=blank)

    @pytest.mark.parametrize("not_a_string", [42, 0, ["no mlx-audio"], object()])
    def test_a_non_string_reason_is_refused(self, not_a_string):
        with pytest.raises(ValueError, match="must be a string"):
            qwen_mlx.BackendOutcome(available=False, reason=not_a_string)

    def test_the_refusal_names_the_offending_value(self):
        with pytest.raises(ValueError) as exc:
            qwen_mlx.BackendOutcome(available=False, reason="   ")

        assert "'   '" in str(exc.value)


class TestSynthesize:
    def test_the_runtime_receives_the_handle_and_text(self, tmp_path, config_path):
        runtime = FakeRuntime()

        make_backend(runtime, config_path).synthesize(
            "Hello there.", None, tmp_path / "u1.wav"
        )

        assert len(runtime.synthesize_calls) == 1
        handle, text = runtime.synthesize_calls[0]
        assert text == "Hello there."
        assert handle["runtime"] == "mlx_qwen"

    def test_the_out_path_is_returned_and_written(self, tmp_path, config_path):
        out_path = tmp_path / "u1.wav"

        result = make_backend(FakeRuntime(), config_path).synthesize(
            "Hello.", None, out_path
        )

        assert isinstance(result, SynthesisResult)
        assert result.out_path == out_path
        assert result.backend_name == qwen_mlx.BACKEND_NAME
        assert out_path.read_bytes() == wav_bytes(2_400)

    def test_duration_is_measured_from_the_written_wav(self, tmp_path, config_path):
        runtime = FakeRuntime(audio=wav_bytes(12_000))

        result = make_backend(runtime, config_path).synthesize(
            "Hello.", None, tmp_path / "u1.wav"
        )

        assert result.duration_s == pytest.approx(0.5)


class TestTheReferenceClipIsFixedAtLoadTime:
    """The runtime's clip comes from its config; a caller cannot swap it."""

    def test_the_configured_clip_is_accepted(self, tmp_path, config_path):
        ref_audio = config_path.parent / "ref.wav"
        runtime = FakeRuntime(ref_audio=str(ref_audio))

        result = make_backend(runtime, config_path).synthesize(
            "Hello.", ref_audio, tmp_path / "u1.wav"
        )

        assert result.out_path.is_file()

    def test_omitting_the_clip_uses_the_configured_one(self, tmp_path, config_path):
        runtime = FakeRuntime()

        make_backend(runtime, config_path).synthesize("Hi.", None, tmp_path / "u1.wav")

        assert len(runtime.synthesize_calls) == 1

    def test_a_different_clip_is_refused_rather_than_ignored(
        self, tmp_path, config_path
    ):
        other = tmp_path / "other.wav"
        other.write_bytes(wav_bytes(240))
        runtime = FakeRuntime(ref_audio=str(config_path.parent / "ref.wav"))

        with pytest.raises(SynthesisError) as exc:
            make_backend(runtime, config_path).synthesize(
                "Hi.", other, tmp_path / "u1.wav"
            )

        assert "reference clip" in str(exc.value)
        assert runtime.synthesize_calls == []


class TestRefusals:
    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_text_is_refused_before_the_runtime(self, tmp_path, config_path, text):
        runtime = FakeRuntime()

        with pytest.raises(SynthesisError):
            make_backend(runtime, config_path).synthesize(text, None, tmp_path / "u1.wav")

        assert runtime.synthesize_calls == []

    def test_a_missing_output_directory_is_refused(self, tmp_path, config_path):
        runtime = FakeRuntime()

        with pytest.raises(SynthesisError):
            make_backend(runtime, config_path).synthesize(
                "Hi.", None, tmp_path / "no" / "u1.wav"
            )

        assert runtime.synthesize_calls == []

    def test_a_runtime_failure_becomes_a_synthesis_error(self, tmp_path, config_path):
        from voiceclonemlx.synthesis.mlx_qwen_runtime import (
            SynthesisError as RuntimeSynthesisError,
        )

        class Broken(FakeRuntime):
            def synthesize(self, handle, text):
                raise RuntimeSynthesisError("model produced no audio")

        with pytest.raises(SynthesisError) as exc:
            make_backend(Broken(), config_path).synthesize(
                "Hi.", None, tmp_path / "u1.wav"
            )

        assert "model produced no audio" in str(exc.value)

    def test_nothing_is_written_when_generation_fails(self, tmp_path, config_path):
        class Broken(FakeRuntime):
            def synthesize(self, handle, text):
                raise RuntimeError("mlx exploded")

        out_path = tmp_path / "u1.wav"

        with pytest.raises(SynthesisError):
            make_backend(Broken(), config_path).synthesize("Hi.", None, out_path)

        assert not out_path.exists()

    def test_unusable_bytes_are_refused_rather_than_left_on_disk(
        self, tmp_path, config_path
    ):
        runtime = FakeRuntime(audio=b"not a wav")
        out_path = tmp_path / "u1.wav"

        with pytest.raises(SynthesisError):
            make_backend(runtime, config_path).synthesize("Hi.", None, out_path)

        assert not out_path.exists()
