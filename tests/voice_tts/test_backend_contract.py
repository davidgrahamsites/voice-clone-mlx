"""The TTS backend seam: the protocol, the result, and the stub backend."""

import io
import sys
import wave

import pytest

from voiceclonegpt.tts.backend import (
    FakeBackend,
    SynthesisError,
    SynthesisResult,
    TTSBackend,
)


@pytest.fixture
def ref_audio(tmp_path):
    """A reference clip on disk. FakeBackend never reads it."""
    path = tmp_path / "ref.wav"
    path.write_bytes(b"not really audio")
    return path


class TestProtocol:
    def test_fake_backend_satisfies_the_protocol(self):
        assert isinstance(FakeBackend(), TTSBackend)

    def test_an_object_without_synthesize_does_not(self):
        class Empty:
            pass

        assert not isinstance(Empty(), TTSBackend)

    def test_backend_name_is_not_a_voice_model(self):
        assert FakeBackend.IS_VOICE_MODEL is False


class TestSynthesisResult:
    def test_result_fields_are_populated(self, tmp_path, ref_audio):
        out_path = tmp_path / "u1.wav"

        result = FakeBackend().synthesize("Hello there.", ref_audio, out_path)

        assert isinstance(result, SynthesisResult)
        assert result.out_path == out_path
        assert result.backend_name == "fake"
        assert result.duration_s > 0

    def test_result_is_frozen(self, tmp_path, ref_audio):
        result = FakeBackend().synthesize("Hello.", ref_audio, tmp_path / "u1.wav")

        with pytest.raises(Exception):
            result.duration_s = 99.0


class TestStubWav:
    def test_a_readable_wav_is_written(self, tmp_path, ref_audio):
        out_path = tmp_path / "u1.wav"

        FakeBackend().synthesize("Hello there.", ref_audio, out_path)

        assert out_path.is_file()
        with wave.open(str(out_path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() > 0
            assert handle.getnframes() > 0

    def test_reported_duration_matches_the_file(self, tmp_path, ref_audio):
        out_path = tmp_path / "u1.wav"

        result = FakeBackend().synthesize("A longer line of text.", ref_audio, out_path)

        with wave.open(str(out_path), "rb") as handle:
            actual = handle.getnframes() / handle.getframerate()
        assert result.duration_s == pytest.approx(actual, abs=0.01)

    def test_the_stub_is_silence(self, tmp_path, ref_audio):
        out_path = tmp_path / "u1.wav"

        FakeBackend().synthesize("Hello there.", ref_audio, out_path)

        with wave.open(str(out_path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
        assert set(frames) == {0}

    def test_longer_text_gives_a_longer_file(self, tmp_path, ref_audio):
        short = FakeBackend().synthesize("Hi.", ref_audio, tmp_path / "a.wav")
        long = FakeBackend().synthesize("Hi. " * 40, ref_audio, tmp_path / "b.wav")

        assert long.duration_s > short.duration_s

    def test_duration_is_clamped(self, tmp_path, ref_audio):
        """Observed behaviour, not a constant: past the cap, more text is no
        longer more audio. What the cap actually is belongs to the stub."""
        huge = FakeBackend().synthesize("x" * 200_000, ref_audio, tmp_path / "a.wav")
        huger = FakeBackend().synthesize("x" * 400_000, ref_audio, tmp_path / "b.wav")

        assert huger.duration_s == huge.duration_s
        assert 0 < huge.duration_s < 200_000

    def test_output_is_deterministic(self, tmp_path, ref_audio):
        FakeBackend().synthesize("Same text.", ref_audio, tmp_path / "a.wav")
        FakeBackend().synthesize("Same text.", ref_audio, tmp_path / "b.wav")

        assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


class TestRefusals:
    @pytest.mark.parametrize("text", ["", "   ", None, 7])
    def test_empty_or_non_string_text_is_refused(self, tmp_path, ref_audio, text):
        out_path = tmp_path / "u1.wav"

        with pytest.raises(SynthesisError):
            FakeBackend().synthesize(text, ref_audio, out_path)

        assert not out_path.exists()

    def test_a_missing_output_directory_is_refused(self, tmp_path, ref_audio):
        with pytest.raises(SynthesisError):
            FakeBackend().synthesize("Hello.", ref_audio, tmp_path / "no" / "u1.wav")


class TestNoGodObject:
    """The seam defines a shape and writes a file. Nothing else."""

    #: The only in-package import allowed here: the stub audio it reuses
    #: rather than re-implements.
    ALLOWED_PACKAGE_IMPORTS = {"voiceclonegpt.synthesis.null_runtime"}

    def test_the_module_imports_only_stdlib_and_the_stub_it_reuses(self):
        import ast

        import voiceclonegpt.tts.backend as backend

        tree = ast.parse(open(backend.__file__, encoding="utf-8").read())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")

        for name in imported:
            if name in self.ALLOWED_PACKAGE_IMPORTS:
                continue
            root = name.split(".")[0]
            assert root != "voiceclonegpt", f"backend.py should not import {name}"
            assert root in sys.stdlib_module_names, f"backend.py should not import {name}"

    def test_the_backend_offers_no_model_or_queue_management(self):
        for absent in ("load", "unload", "enqueue", "queue", "warm_up"):
            assert not hasattr(FakeBackend, absent)

    def test_the_silence_settings_have_exactly_one_home(self):
        """The stub's settings are the stub's business.

        Not copied here, and not mirrored here either: even a pointer at
        another module's constant breaks when that module reorganizes its
        settings, which is exactly how this seam broke once.
        """
        for owned_by_the_stub in (
            "SAMPLE_RATE",
            "MAX_SECONDS",
            "MIN_SECONDS",
            "CHARS_PER_SECOND",
            "SAMPLE_WIDTH_BYTES",
            "CHANNELS",
        ):
            assert not hasattr(FakeBackend, owned_by_the_stub), (
                f"{owned_by_the_stub} belongs to the stub; do not restate or "
                f"mirror it on the backend"
            )

    def test_only_synthesize_is_required_of_a_stub(self, tmp_path, ref_audio):
        """A stub with no attributes at all still works — pure duck typing."""

        class BareStub:
            def synthesize(self, handle, text):
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(16_000)
                    out.writeframes(b"\x00\x00" * 16_000)
                return buffer.getvalue()

        result = FakeBackend(stub=BareStub()).synthesize(
            "Hello.", ref_audio, tmp_path / "u1.wav"
        )

        assert result.duration_s == pytest.approx(1.0)

    def test_the_stub_audio_source_is_injectable(self, tmp_path, ref_audio):
        class LoudStub:
            def synthesize(self, handle, text):
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(8_000)
                    out.writeframes(b"\x01\x02" * 8_000)
                return buffer.getvalue()

        result = FakeBackend(stub=LoudStub()).synthesize(
            "Hello.", ref_audio, tmp_path / "u1.wav"
        )

        assert result.duration_s == pytest.approx(1.0)
