"""Test where audio is written, and that the writer cannot be bypassed.

Placement beside the manifest, and the guarantees the write carries:
confinement, symlink refusal, byte validation, atomic replacement. All of them
come from `ReaderSynthesisSession`, and none of them is optional — there is no
public `writer` argument to substitute.

Generation, resume, failure handling, and the borrowed-rule checks live in
`test_manifest_roundtrip.py`.
"""

import inspect
import json

import pytest

from voiceclonegpt.reader_app import manifest_roundtrip
from voiceclonegpt.reader_app.manifest_roundtrip import (
    ManifestRoundTripError,
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


class TestPlacement:
    """Audio lands beside the manifest, and nowhere else."""

    def test_output_is_the_manifest_parent(self, tmp_path):
        nested = tmp_path / "runs" / "session-1"
        nested.mkdir(parents=True)
        path = write_manifest(nested, row("N-1"))

        generate_missing_takes(path, recording_synth())

        assert (nested / "N-1.wav").is_file()

    def test_nothing_is_written_elsewhere(self, tmp_path):
        nested = tmp_path / "runs"
        nested.mkdir()
        path = write_manifest(nested, row("N-1"))

        generate_missing_takes(path, recording_synth())

        assert [p.name for p in tmp_path.iterdir()] == ["runs"]

    def test_a_symlinked_directory_resolves_to_the_real_one(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        path = write_manifest(real, row("N-1"))
        link = tmp_path / "link"
        link.symlink_to(real)

        generate_missing_takes(link / path.name, recording_synth())

        assert (real / "N-1.wav").is_file()

    def test_no_temp_file_survives(self, manifest):
        generate_missing_takes(manifest, recording_synth())

        leftovers = [
            p.name for p in manifest.parent.iterdir() if "partial" in p.name
        ]
        assert leftovers == []

    def test_the_manifest_itself_is_untouched(self, manifest):
        before = manifest.read_bytes()

        generate_missing_takes(manifest, recording_synth())

        assert manifest.read_bytes() == before


class TestWriterIsNotBypassable:
    """The session writer is the only way bytes reach disk.

    An injectable `writer=` would let a caller substitute something that
    skips confinement, symlink refusal, byte validation, and atomic
    replacement — every guarantee the write is supposed to carry. So there is
    no such parameter, and tests observe the private boundary instead.
    """

    def test_there_is_no_public_writer_parameter(self):
        parameters = inspect.signature(generate_missing_takes).parameters

        assert "writer" not in parameters

    def test_passing_a_writer_is_rejected(self, manifest):
        with pytest.raises(TypeError):
            generate_missing_takes(
                manifest, recording_synth(), writer=lambda text, name: None
            )

    def test_the_session_writer_is_asked_for_each_missing_take(
        self, manifest, monkeypatch
    ):
        calls = []

        def fake_default_writer(output_dir, synthesize):
            def write(text, output_name):
                calls.append((text, output_name))
                return output_dir / output_name

            return write

        monkeypatch.setattr(
            manifest_roundtrip, "_default_writer", fake_default_writer
        )

        generate_missing_takes(manifest, recording_synth())

        assert [name for _, name in calls] == ["NEUTRAL-1.wav", "NEUTRAL-2.wav"]

    def test_the_session_writer_receives_the_row_text(
        self, manifest, monkeypatch
    ):
        captured = []

        def fake_default_writer(output_dir, synthesize):
            def write(text, output_name):
                captured.append(text)
                return output_dir / output_name

            return write

        monkeypatch.setattr(
            manifest_roundtrip, "_default_writer", fake_default_writer
        )

        generate_missing_takes(manifest, recording_synth())

        assert captured == ["A line.", "A line."]

    def test_the_writer_is_built_from_a_reader_synthesis_session(
        self, manifest, monkeypatch
    ):
        """The production path must construct the real session."""
        built = []

        real_session = manifest_roundtrip.ReaderSynthesisSession

        def recording_session(**kwargs):
            built.append(kwargs)
            return real_session(**kwargs)

        monkeypatch.setattr(
            manifest_roundtrip, "ReaderSynthesisSession", recording_session
        )

        generate_missing_takes(manifest, recording_synth())

        assert len(built) == 1
        assert built[0]["output_dir"] == manifest.resolve().parent

    def test_the_session_confinement_still_applies(self, tmp_path):
        """Proof the real writer is in the path: its refusals still bite.

        A symlink planted where a take would be written is refused by the
        session, and the file it points at is left untouched.

        The victim must sit *outside* the manifest directory: a link to a file
        inside it resolves cleanly, so `find_audio` would treat the take as
        already present and skip it rather than exercising the write.
        """
        victim = tmp_path / "victim.wav"
        victim.write_bytes(b"ORIGINAL")

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        path = write_manifest(run_dir, row("N-1"))
        (run_dir / "N-1.wav").symlink_to(victim)

        with pytest.raises(ManifestRoundTripError, match="N-1"):
            generate_missing_takes(path, recording_synth())

        assert victim.read_bytes() == b"ORIGINAL"

    def test_default_writer_produces_reader_discoverable_names(self, manifest):
        """The whole point: `find_audio` must locate what was written."""
        from voiceclonegpt.reader_app.core import find_audio

        generate_missing_takes(manifest, recording_synth())

        assert find_audio(manifest.parent, "NEUTRAL-1") is not None
        assert find_audio(manifest.parent, "NEUTRAL-2") is not None

    def test_generated_takes_are_visible_to_load_manifest(self, manifest):
        """A Reader reloading the manifest now sees audio for every take."""
        from voiceclonegpt.reader_app.core import load_manifest

        generate_missing_takes(manifest, recording_synth())

        assert all(take.has_audio for take in load_manifest(manifest))
