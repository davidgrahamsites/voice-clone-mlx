"""Test the Reader synthesis session.

`voiceclonegpt.shared.roundtrip` is not present in this worktree (it lives on
the model-roundtrip branch), so the round-trip callable is injected. The
default path lazily imports it — see `TestRoundTripSeam`.

No network, no model, no Tk.
"""

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
    """Build a round-trip callable that records how it was called."""
    calls = []

    def round_trip(bundle_dir, *, runtime_id, text, runtime):
        calls.append(
            {
                "bundle_dir": bundle_dir,
                "runtime_id": runtime_id,
                "text": text,
                "runtime": runtime,
            }
        )
        return audio

    round_trip.calls = calls
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


def make_session(
    bundle_dir,
    output_dir,
    round_trip=None,
    runtime="fake-runtime",
    audio_encoder=None,
):
    return ReaderSynthesisSession(
        bundle_dir=bundle_dir,
        runtime_id="mlx",
        runtime=runtime,
        output_dir=output_dir,
        round_trip=round_trip or fake_round_trip(),
        audio_encoder=audio_encoder,
    )


class TestSynthesis:
    """The happy path writes exactly what the round trip returned."""

    def test_writes_the_returned_bytes(self, bundle_dir, output_dir):
        session = make_session(bundle_dir, output_dir)

        path = session.synthesize("Hello from my voice.", "take-1.wav")

        assert path == output_dir / "take-1.wav"
        assert path.read_bytes() == WAV

    def test_passes_the_exact_bundle_and_runtime(self, bundle_dir, output_dir):
        round_trip = fake_round_trip()
        session = make_session(bundle_dir, output_dir, round_trip, runtime="R")

        session.synthesize("Hi.", "a.wav")

        call = round_trip.calls[0]
        assert call["bundle_dir"] == bundle_dir
        assert call["runtime_id"] == "mlx"
        assert call["text"] == "Hi."
        assert call["runtime"] == "R"

    def test_calls_the_round_trip_once_per_synthesis(self, bundle_dir, output_dir):
        round_trip = fake_round_trip()
        session = make_session(bundle_dir, output_dir, round_trip)

        session.synthesize("One.", "one.wav")
        session.synthesize("Two.", "two.wav")

        assert len(round_trip.calls) == 2

    def test_returns_an_absolute_path(self, bundle_dir, output_dir):
        session = make_session(bundle_dir, output_dir)

        assert session.synthesize("Hi.", "a.wav").is_absolute()

    def test_overwrites_a_previous_take(self, bundle_dir, output_dir):
        make_session(bundle_dir, output_dir).synthesize("Hi.", "a.wav")
        session = make_session(
            bundle_dir, output_dir, fake_round_trip(b"RIFFsecond")
        )

        assert session.synthesize("Hi.", "a.wav").read_bytes() == b"RIFFsecond"

    def test_writes_mp3_through_the_injected_encoder(self, bundle_dir, output_dir):
        calls = []

        def encoder(wav, output_format):
            calls.append((wav, output_format))
            return b"ID3-mp3"

        session = make_session(
            bundle_dir,
            output_dir,
            audio_encoder=encoder,
        )

        path = session.synthesize("Hello from my voice.", "take-1.mp3")

        assert path == output_dir / "take-1.mp3"
        assert path.read_bytes() == b"ID3-mp3"
        assert calls == [(WAV, "mp3")]


class TestInputValidation:
    """Empty text or name is refused before anything is called."""

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_text_is_rejected(self, bundle_dir, output_dir, text):
        round_trip = fake_round_trip()
        session = make_session(bundle_dir, output_dir, round_trip)

        with pytest.raises(ReaderSynthesisError, match="text"):
            session.synthesize(text, "a.wav")

        assert round_trip.calls == []

    def test_non_string_text_is_rejected(self, bundle_dir, output_dir):
        session = make_session(bundle_dir, output_dir)

        with pytest.raises(ReaderSynthesisError, match="text"):
            session.synthesize(None, "a.wav")

    def test_unsupported_output_format_is_rejected_before_generation(
        self, bundle_dir, output_dir
    ):
        round_trip = fake_round_trip()
        session = make_session(bundle_dir, output_dir, round_trip)

        with pytest.raises(ReaderSynthesisError, match="wav|mp3"):
            session.synthesize("Hi.", "take-1.flac")

        assert round_trip.calls == []

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_empty_name_is_rejected(self, bundle_dir, output_dir, name):
        round_trip = fake_round_trip()
        session = make_session(bundle_dir, output_dir, round_trip)

        with pytest.raises(ReaderSynthesisError, match="name"):
            session.synthesize("Hi.", name)

        assert round_trip.calls == []


class TestOutputConfinement:
    """`output_name` is a file name, never a path."""

    @pytest.mark.parametrize(
        "name",
        [
            "../escape.wav",
            "../../etc/passwd",
            "sub/nested.wav",
            "sub\\nested.wav",
            "/etc/passwd",
            "..",
            ".",
            "with\x00null.wav",
        ],
    )
    def test_unsafe_names_are_rejected(self, bundle_dir, output_dir, name):
        round_trip = fake_round_trip()
        session = make_session(bundle_dir, output_dir, round_trip)

        with pytest.raises(ReaderSynthesisError, match="name"):
            session.synthesize("Hi.", name)

        assert round_trip.calls == [], "nothing may be generated for a bad name"

    def test_traversal_cannot_write_outside(self, bundle_dir, output_dir, tmp_path):
        session = make_session(bundle_dir, output_dir)

        with pytest.raises(ReaderSynthesisError):
            session.synthesize("Hi.", "../pwned.wav")

        assert not (tmp_path / "pwned.wav").exists()

    def test_symlink_target_escaping_the_output_dir_is_refused(
        self, bundle_dir, output_dir, tmp_path
    ):
        """A pre-placed symlink must not redirect the write."""
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"original")
        (output_dir / "take-1.wav").symlink_to(outside)

        session = make_session(bundle_dir, output_dir)

        with pytest.raises(ReaderSynthesisError, match="symlink|name"):
            session.synthesize("Hi.", "take-1.wav")

        assert outside.read_bytes() == b"original"

    def test_symlinked_output_dir_is_resolved(self, bundle_dir, tmp_path):
        real = tmp_path / "real_out"
        real.mkdir()
        link = tmp_path / "link_out"
        link.symlink_to(real)

        path = make_session(bundle_dir, link).synthesize("Hi.", "a.wav")

        assert path.resolve().parent == real.resolve()

    def test_missing_output_dir_is_rejected(self, bundle_dir, tmp_path):
        session = make_session(bundle_dir, tmp_path / "absent")

        with pytest.raises(ReaderSynthesisError, match="output"):
            session.synthesize("Hi.", "a.wav")

    def test_output_dir_that_is_a_file_is_rejected(self, bundle_dir, tmp_path):
        target = tmp_path / "afile"
        target.write_bytes(b"")
        session = make_session(bundle_dir, target)

        with pytest.raises(ReaderSynthesisError, match="output"):
            session.synthesize("Hi.", "a.wav")


class TestRoundTripFailures:
    """Provider failures surface as one typed error."""

    def test_round_trip_error_is_wrapped(self, bundle_dir, output_dir):
        def failing(bundle_dir, *, runtime_id, text, runtime):
            raise RuntimeError("runtime 'mlx' failed: loader unavailable")

        session = make_session(bundle_dir, output_dir, failing)

        with pytest.raises(ReaderSynthesisError, match="synthesis failed") as exc:
            session.synthesize("Hi.", "a.wav")

        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_no_file_is_left_behind_after_a_failure(self, bundle_dir, output_dir):
        def failing(bundle_dir, *, runtime_id, text, runtime):
            raise RuntimeError("boom")

        session = make_session(bundle_dir, output_dir, failing)

        with pytest.raises(ReaderSynthesisError):
            session.synthesize("Hi.", "a.wav")

        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize("audio", [b"", None, "not bytes", 42])
    def test_non_bytes_audio_is_rejected(self, bundle_dir, output_dir, audio):
        session = make_session(bundle_dir, output_dir, fake_round_trip(audio))

        with pytest.raises(ReaderSynthesisError, match="audio"):
            session.synthesize("Hi.", "a.wav")

        assert list(output_dir.iterdir()) == []


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
        """The temp path must not be guessable or followed.

        A predictable `<target>.partial` written with `write_bytes` follows a
        symlink planted there, so anything that can create a file in the
        output directory can overwrite an arbitrary file elsewhere.
        """
        victim = tmp_path / "victim.wav"
        victim.write_bytes(b"ORIGINAL")
        (output_dir / "a.wav.partial").symlink_to(victim)

        session = make_session(
            bundle_dir, output_dir, fake_round_trip(b"RIFFattack")
        )
        session.synthesize("Hi.", "a.wav")

        assert victim.read_bytes() == b"ORIGINAL"
        assert (output_dir / "a.wav").read_bytes() == b"RIFFattack"

    def test_the_temp_path_is_not_the_predictable_partial_name(
        self, bundle_dir, output_dir
    ):
        """Exclusive creation means a fresh, unguessable name each time."""
        seen = []
        real_replace = os.replace

        def recording_replace(src, dst):
            seen.append(Path(src).name)
            return real_replace(src, dst)

        session = make_session(bundle_dir, output_dir)
        original = synthesis_session.os.replace
        synthesis_session.os.replace = recording_replace
        try:
            session.synthesize("Hi.", "a.wav")
            session.synthesize("Hi.", "a.wav")
        finally:
            synthesis_session.os.replace = original

        assert seen[0] != "a.wav.partial"
        assert seen[0] != seen[1], "temp names must not repeat"

    def test_a_planted_partial_file_does_not_block_the_write(
        self, bundle_dir, output_dir
    ):
        """A leftover or hostile `.partial` must not break a new take."""
        (output_dir / "a.wav.partial").write_bytes(b"stale")

        path = make_session(bundle_dir, output_dir).synthesize("Hi.", "a.wav")

        assert path.read_bytes() == WAV

    def test_the_temp_file_is_a_sibling(self, bundle_dir, output_dir):
        """Same directory, so `os.replace` stays on one filesystem."""
        seen = []
        real_replace = os.replace

        def recording_replace(src, dst):
            seen.append((Path(src).parent, Path(dst).parent))
            return real_replace(src, dst)

        session = make_session(bundle_dir, output_dir)
        original = synthesis_session.os.replace
        synthesis_session.os.replace = recording_replace
        try:
            session.synthesize("Hi.", "a.wav")
        finally:
            synthesis_session.os.replace = original

        assert seen and seen[0][0] == seen[0][1]
