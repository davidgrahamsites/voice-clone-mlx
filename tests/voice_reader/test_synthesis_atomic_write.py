"""Atomic output tests for the Reader's confined writer."""

import os
from pathlib import Path

import pytest

from voiceclonegpt.reader_app import synthesis_session
from voiceclonegpt.reader_app.synthesis_session import (
    ReaderSynthesisError,
    ReaderSynthesisSession,
)

WAV = b"RIFF$\x00\x00\x00WAVEfake"


def fake_round_trip(audio=WAV):
    def round_trip(bundle_dir, *, runtime_id, text, runtime):
        return audio

    return round_trip


@pytest.fixture
def bundle_dir(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    return path


@pytest.fixture
def output_dir(tmp_path):
    path = tmp_path / "out"
    path.mkdir()
    return path


def make_session(bundle_dir, output_dir, round_trip=None):
    return ReaderSynthesisSession(
        bundle_dir=bundle_dir,
        runtime_id="mlx",
        runtime="fake-runtime",
        output_dir=output_dir,
        round_trip=round_trip or fake_round_trip(),
    )


class TestAtomicWrite:
    """A reader must never observe a half-written take."""

    def test_no_temp_file_survives_a_successful_write(self, bundle_dir, output_dir):
        make_session(bundle_dir, output_dir).synthesize("Hi.", "a.wav")

        assert [p.name for p in output_dir.iterdir()] == ["a.wav"]

    def test_a_failed_write_leaves_the_previous_take_intact(
        self, bundle_dir, output_dir, monkeypatch
    ):
        make_session(bundle_dir, output_dir).synthesize("Hi.", "a.wav")

        def broken_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(synthesis_session.os, "replace", broken_replace)
        session = make_session(bundle_dir, output_dir, fake_round_trip(b"RIFFnew"))

        with pytest.raises(ReaderSynthesisError, match="could not write"):
            session.synthesize("Hi.", "a.wav")

        assert (output_dir / "a.wav").read_bytes() == WAV
        assert [p.name for p in output_dir.iterdir()] == ["a.wav"]

    def test_a_planted_partial_symlink_cannot_redirect_the_write(
        self, bundle_dir, output_dir, tmp_path
    ):
        """A predictable temp path must not be followed."""
        victim = tmp_path / "victim.wav"
        victim.write_bytes(b"ORIGINAL")
        (output_dir / "a.wav.partial").symlink_to(victim)

        make_session(bundle_dir, output_dir, fake_round_trip(b"RIFFattack")).synthesize(
            "Hi.", "a.wav"
        )

        assert victim.read_bytes() == b"ORIGINAL"
        assert (output_dir / "a.wav").read_bytes() == b"RIFFattack"

    def test_the_temp_path_is_not_predictable(self, bundle_dir, output_dir):
        seen = []
        real_replace = os.replace

        def recording_replace(src, dst):
            seen.append(Path(src).name)
            return real_replace(src, dst)

        original = synthesis_session.os.replace
        synthesis_session.os.replace = recording_replace
        try:
            session = make_session(bundle_dir, output_dir)
            session.synthesize("Hi.", "a.wav")
            session.synthesize("Hi.", "a.wav")
        finally:
            synthesis_session.os.replace = original

        assert seen[0] != "a.wav.partial"
        assert seen[0] != seen[1]

    def test_a_planted_partial_file_does_not_block_the_write(
        self, bundle_dir, output_dir
    ):
        (output_dir / "a.wav.partial").write_bytes(b"stale")

        path = make_session(bundle_dir, output_dir).synthesize("Hi.", "a.wav")

        assert path.read_bytes() == WAV

    def test_the_temp_file_is_a_sibling(self, bundle_dir, output_dir):
        seen = []
        real_replace = os.replace

        def recording_replace(src, dst):
            seen.append((Path(src).parent, Path(dst).parent))
            return real_replace(src, dst)

        original = synthesis_session.os.replace
        synthesis_session.os.replace = recording_replace
        try:
            make_session(bundle_dir, output_dir).synthesize("Hi.", "a.wav")
        finally:
            synthesis_session.os.replace = original

        assert seen and seen[0][0] == seen[0][1]
