"""Test generating utterance audio for a script manifest.

Generation, resume, failure handling, and the rules this module borrows rather
than restates. Where the audio lands and why the writer cannot be bypassed
live in `test_manifest_roundtrip_writer.py`.

Orchestration only: the synthesis callable is injected, row validation comes
from `load_manifest`, resume comes from `find_audio`, and the write is
delegated to `ReaderSynthesisSession`. Nothing here loads a model or reaches a
network.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from voiceclonemlx.reader_app import manifest_roundtrip
from voiceclonemlx.reader_app.manifest_roundtrip import (
    ManifestRoundTripError,
    RoundTripResult,
    generate_missing_takes,
)

WAV = b"RIFF$\x00\x00\x00WAVEfake"


def write_manifest(directory, *rows):
    path = directory / "script_session_1.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def row(utterance_id, text="A line.", style="neutral"):
    return {"id": utterance_id, "text": text, "style": style}


def recording_synth(audio=WAV):
    """A synthesize callable that records the text it was asked for."""
    seen = []

    def synthesize(text):
        seen.append(text)
        return audio

    synthesize.seen = seen
    return synthesize


@pytest.fixture
def manifest(tmp_path):
    return write_manifest(tmp_path, row("NEUTRAL-1"), row("NEUTRAL-2"))


class TestGeneration:
    """Each utterance becomes a WAV beside the manifest."""

    def test_writes_one_file_per_utterance(self, manifest):
        generate_missing_takes(manifest, recording_synth())

        assert (manifest.parent / "NEUTRAL-1.wav").is_file()
        assert (manifest.parent / "NEUTRAL-2.wav").is_file()

    def test_written_bytes_are_what_synthesis_returned(self, manifest):
        generate_missing_takes(manifest, recording_synth(b"RIFFexact"))

        assert (manifest.parent / "NEUTRAL-1.wav").read_bytes() == b"RIFFexact"

    def test_synthesize_receives_the_row_text_verbatim(self, tmp_path):
        path = write_manifest(tmp_path, row("N-1", text="  Spaced  text.  "))
        synth = recording_synth()

        generate_missing_takes(path, synth)

        assert synth.seen == ["  Spaced  text.  "]

    def test_utterances_are_processed_in_manifest_order(self, tmp_path):
        path = write_manifest(
            tmp_path, row("N-1", text="first"), row("N-2", text="second")
        )
        synth = recording_synth()

        generate_missing_takes(path, synth)

        assert synth.seen == ["first", "second"]

    def test_result_lists_generated_ids(self, manifest):
        result = generate_missing_takes(manifest, recording_synth())

        assert result.generated == ("NEUTRAL-1", "NEUTRAL-2")
        assert result.skipped == ()

    def test_result_records_the_manifest(self, manifest):
        result = generate_missing_takes(manifest, recording_synth())

        assert result.manifest_path == str(manifest)

    def test_an_empty_manifest_generates_nothing(self, tmp_path):
        path = write_manifest(tmp_path)
        synth = recording_synth()

        result = generate_missing_takes(path, synth)

        assert result.generated == ()
        assert synth.seen == []


class TestResume:
    """Resume is defined by what Reader can already find."""

    def test_existing_wav_is_skipped(self, manifest):
        (manifest.parent / "NEUTRAL-1.wav").write_bytes(b"RIFFalready")
        synth = recording_synth()

        result = generate_missing_takes(manifest, synth)

        assert result.skipped == ("NEUTRAL-1",)
        assert result.generated == ("NEUTRAL-2",)

    def test_an_existing_take_is_not_regenerated(self, manifest):
        (manifest.parent / "NEUTRAL-1.wav").write_bytes(b"RIFFalready")

        generate_missing_takes(manifest, recording_synth(b"RIFFnew"))

        assert (manifest.parent / "NEUTRAL-1.wav").read_bytes() == b"RIFFalready"

    def test_synthesis_is_not_called_for_an_existing_take(self, manifest):
        (manifest.parent / "NEUTRAL-1.wav").write_bytes(b"")
        synth = recording_synth()

        generate_missing_takes(manifest, synth)

        assert len(synth.seen) == 1

    @pytest.mark.parametrize("suffix", [".m4a", ".mp3", ".aiff"])
    def test_any_format_reader_can_find_counts_as_done(self, manifest, suffix):
        """A hand-placed recording is honoured, not overwritten with a fake."""
        (manifest.parent / f"NEUTRAL-1{suffix}").write_bytes(b"real audio")

        result = generate_missing_takes(manifest, recording_synth())

        assert "NEUTRAL-1" in result.skipped
        assert (manifest.parent / f"NEUTRAL-1{suffix}").read_bytes() == b"real audio"

    def test_a_second_run_generates_nothing(self, manifest):
        generate_missing_takes(manifest, recording_synth())
        synth = recording_synth()

        result = generate_missing_takes(manifest, synth)

        assert result.generated == ()
        assert result.skipped == ("NEUTRAL-1", "NEUTRAL-2")
        assert synth.seen == []


class TestFailure:
    """A failure stops the run without losing what already succeeded."""

    def test_synthesis_error_is_wrapped_and_names_the_utterance(self, manifest):
        def broken(text):
            raise RuntimeError("engine exploded")

        with pytest.raises(ManifestRoundTripError, match="NEUTRAL-1") as excinfo:
            generate_missing_takes(manifest, broken)

        assert excinfo.value.__cause__ is not None

    def test_earlier_takes_survive_a_later_failure(self, manifest):
        def fails_on_second(text):
            if len(getattr(fails_on_second, "calls", [])) >= 1:
                raise RuntimeError("boom")
            fails_on_second.calls = getattr(fails_on_second, "calls", []) + [text]
            return WAV

        with pytest.raises(ManifestRoundTripError):
            generate_missing_takes(manifest, fails_on_second)

        assert (manifest.parent / "NEUTRAL-1.wav").is_file()
        assert not (manifest.parent / "NEUTRAL-2.wav").exists()

    def test_a_rerun_resumes_after_a_failure(self, manifest):
        def fails_on_second(text):
            if getattr(fails_on_second, "done", False):
                raise RuntimeError("boom")
            fails_on_second.done = True
            return WAV

        with pytest.raises(ManifestRoundTripError):
            generate_missing_takes(manifest, fails_on_second)

        result = generate_missing_takes(manifest, recording_synth())

        assert result.skipped == ("NEUTRAL-1",)
        assert result.generated == ("NEUTRAL-2",)

    @pytest.mark.parametrize("audio", [b"", None, "not bytes", 42])
    def test_unusable_audio_is_refused_and_writes_nothing(self, manifest, audio):
        with pytest.raises(ManifestRoundTripError):
            generate_missing_takes(manifest, recording_synth(audio))

        assert not (manifest.parent / "NEUTRAL-1.wav").exists()

    def test_no_temp_file_survives_a_failure(self, manifest):
        def broken(text):
            raise RuntimeError("boom")

        with pytest.raises(ManifestRoundTripError):
            generate_missing_takes(manifest, broken)

        leftovers = [
            p.name for p in manifest.parent.iterdir() if "partial" in p.name
        ]
        assert leftovers == []


class TestInputValidation:
    """Bad inputs surface as they are, or as a typed error."""

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        """Left to `load_manifest`, unchanged."""
        with pytest.raises(FileNotFoundError):
            generate_missing_takes(tmp_path / "absent.jsonl", recording_synth())

    def test_malformed_row_raises_from_load_manifest(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "A"}\n', encoding="utf-8")

        with pytest.raises(ValueError):
            generate_missing_takes(path, recording_synth())

    @pytest.mark.parametrize("value", [None, 42, "synthesize", []])
    def test_non_callable_synthesize_is_refused(self, manifest, value):
        with pytest.raises(ManifestRoundTripError, match="synthesize"):
            generate_missing_takes(manifest, value)

    def test_nothing_is_generated_when_synthesize_is_invalid(self, manifest):
        with pytest.raises(ManifestRoundTripError):
            generate_missing_takes(manifest, None)

        assert not (manifest.parent / "NEUTRAL-1.wav").exists()


class TestResultContract:
    """The result is a small immutable value."""

    def test_result_is_frozen(self, manifest):
        result = generate_missing_takes(manifest, recording_synth())

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.generated = ()

    def test_generated_and_skipped_are_tuples(self, manifest):
        result = generate_missing_takes(manifest, recording_synth())

        assert isinstance(result.generated, tuple)
        assert isinstance(result.skipped, tuple)

    def test_result_is_constructible_directly(self):
        result = RoundTripResult(
            generated=("a",), skipped=(), manifest_path="m.jsonl"
        )

        assert result.generated == ("a",)


class TestSeamIsOrchestrationOnly:
    """Rules that already have a home are not re-implemented here."""

    def _source(self):
        return Path(manifest_roundtrip.__file__).read_text(encoding="utf-8")

    def test_reuses_the_existing_reader_contracts(self):
        source = self._source()

        assert "load_manifest" in source
        assert "ReaderSynthesisSession" in source

    def test_does_not_reimplement_id_safety_or_discovery(self):
        source = self._source()

        assert "def " + "is_safe_utterance_id" not in source
        assert "def " + "find_audio" not in source
        assert "AUDIO_SUFFIXES = " not in source

    def test_does_not_reimplement_atomic_writing(self):
        source = self._source()

        assert "mkstemp" not in source
        assert "os.replace" not in source

    def test_imports_nothing_dangerous(self):
        source = self._source()
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )
        imports = imports.replace("voiceclonemlx.", "")

        for banned in ("mlx", "urllib", "requests", "socket", "subprocess",
                       "tkinter", "datetime", "time", "random"):
            assert banned not in imports

    def test_imports_no_studio_app(self):
        assert "studio_app" not in self._source()
