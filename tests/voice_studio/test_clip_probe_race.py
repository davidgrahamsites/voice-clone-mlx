"""Test what the clip probe does when the file changes underneath it.

Every check here is about *time*: the clip swapped, relinked, grown, or
rewritten between the moment the path is approved and the moment the bytes are
read. Static refusals live in `test_clip_probe_safety.py`, measuring a valid
clip in `test_clip_probe.py`.

Split out when the safety module outgrew the ICM size threshold; the classes
below moved verbatim. `write_wav` is duplicated rather than shared — a
`conftest.py` fixture named that generically is what gets silently shadowed by
a module-local one later.
"""

import os
import wave
from pathlib import Path

import pytest

from voiceclonemlx.dataset import clip_probe
from voiceclonemlx.dataset.clip_probe import (
    MAX_CLIP_BYTES,
    ClipProbeError,
    probe_clip,
)


def write_wav(path, *, frames=240, rate=24000, channels=1, width=2):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * (frames * channels * width))
    return path


def during_digest(monkeypatch, mutate):
    """Run `mutate(path)` inside the read window, where a real race lives."""
    real_digest = clip_probe._digest

    def wrapped(handle, max_bytes, path):
        result = real_digest(handle, max_bytes, path)
        mutate(path)
        return result

    monkeypatch.setattr(clip_probe, "_digest", wrapped)


class TestOneFileOnly:
    """Digest, properties, and size must describe the same bytes."""

    def test_the_clip_is_opened_exactly_once(self, tmp_path, monkeypatch):
        clip = write_wav(tmp_path / "c.wav")
        opens = []
        real_open = clip_probe.os.open

        def counting_open(path, *args, **kwargs):
            opens.append(str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(clip_probe.os, "open", counting_open)
        probe_clip(clip)

        assert opens.count(str(clip.resolve())) == 1

    def test_the_header_is_not_reopened_by_name(self, tmp_path, monkeypatch):
        """`wave` gets the open file, never the path again."""
        clip = write_wav(tmp_path / "c.wav")
        seen = []
        real_wave_open = clip_probe.wave.open

        def recording_open(source, *args, **kwargs):
            seen.append(source)
            return real_wave_open(source, *args, **kwargs)

        monkeypatch.setattr(clip_probe.wave, "open", recording_open)
        probe_clip(clip)

        assert seen and not any(isinstance(s, (str, Path)) for s in seen)

    def test_replacing_the_file_midway_is_detected(self, tmp_path, monkeypatch):
        """Mutation inside the read window is the only real race."""
        clip = write_wav(tmp_path / "c.wav", frames=24000)

        def swap(path):
            other = write_wav(tmp_path / "other.wav", frames=48000)
            path.write_bytes(other.read_bytes())

        during_digest(monkeypatch, swap)

        with pytest.raises(ClipProbeError, match="changed while being measured"):
            probe_clip(clip)

    def test_a_renamed_replacement_is_refused(self, tmp_path, monkeypatch):
        """A true digest of bytes the path no longer names is still wrong."""
        clip = write_wav(tmp_path / "c.wav", frames=24000)

        def rename_over(path):
            other = write_wav(tmp_path / "other.wav", frames=48000)
            os.replace(other, path)

        during_digest(monkeypatch, rename_over)

        with pytest.raises(ClipProbeError, match="replaced|changed"):
            probe_clip(clip)

    def test_the_name_check_rejects_a_replacement_on_its_own(self, tmp_path):
        """The direct guarantee, not resting on a side effect.

        `os.replace` also moves the old inode's ctime here, so the identity
        """
        clip = write_wav(tmp_path / "c.wav", frames=24000)

        with clip.open("rb") as handle:
            other = write_wav(tmp_path / "other.wav", frames=48000)
            os.replace(other, clip)

            with pytest.raises(ClipProbeError, match="no longer names"):
                clip_probe._check_still_named(handle, clip)

    def test_probe_clip_calls_the_name_check(self, tmp_path, monkeypatch):
        """Pins the call site: with identity frozen, only `lstat` can refuse."""
        clip = write_wav(tmp_path / "c.wav", frames=2400)
        frozen = (1, 2, clip.stat().st_size, 4, 5)
        monkeypatch.setattr(clip_probe, "_identity", lambda handle: frozen)

        def rename_over(path):
            other = write_wav(tmp_path / "other.wav", frames=2400)
            os.replace(other, path)

        during_digest(monkeypatch, rename_over)

        with pytest.raises(ClipProbeError, match="no longer names"):
            probe_clip(clip)

    def test_the_name_check_passes_for_an_untouched_clip(self, tmp_path):
        clip = write_wav(tmp_path / "c.wav")
        with clip.open("rb") as handle:
            assert clip_probe._check_still_named(handle, clip) is None

    def test_the_name_check_refuses_a_symlink_swapped_over_the_path(self, tmp_path):
        """`lstat`, not `stat`: a link dropped in place is a mismatch."""
        clip = write_wav(tmp_path / "c.wav")
        other = write_wav(tmp_path / "other.wav", frames=48000)

        with clip.open("rb") as handle:
            clip.unlink()
            os.symlink(other, clip)

            with pytest.raises(ClipProbeError, match="no longer names"):
                clip_probe._check_still_named(handle, clip)


class TestSymlinkRace:
    """The name can become a link between the check and the open."""

    def test_a_path_swapped_to_a_symlink_just_before_open_is_refused(self, tmp_path):
        """Staged after `_resolved_clip` approved the path — exactly the moment"""
        clip = write_wav(tmp_path / "c.wav")
        secret = write_wav(tmp_path / "secret.wav", frames=48000)
        real_resolved = clip_probe._resolved_clip

        def swapping_resolve(path):
            resolved = real_resolved(path)
            resolved.unlink()
            resolved.symlink_to(secret)
            return resolved

        monkey = pytest.MonkeyPatch()
        monkey.setattr(clip_probe, "_resolved_clip", swapping_resolve)
        try:
            with pytest.raises(ClipProbeError, match="symlink"):
                probe_clip(clip)
        finally:
            monkey.undo()

    def test_no_follow_is_actually_requested(self, tmp_path, monkeypatch):
        """Pins the flag: without it the refusal above is accidental."""
        clip = write_wav(tmp_path / "c.wav")
        flags = []
        real_open = clip_probe.os.open

        def recording_open(path, flag, *args, **kwargs):
            flags.append(flag)
            return real_open(path, flag, *args, **kwargs)

        monkeypatch.setattr(clip_probe.os, "open", recording_open)
        probe_clip(clip)

        assert flags and all(f & clip_probe._NOFOLLOW for f in flags)

    def test_a_directory_swapped_in_is_refused_from_the_descriptor(self, tmp_path):
        """Regular-file status is asked of the open file, not of the name."""
        clip = write_wav(tmp_path / "c.wav")

        monkey = pytest.MonkeyPatch()
        monkey.setattr(
            clip_probe.stat_module, "S_ISREG", lambda mode: False
        )
        try:
            with pytest.raises(ClipProbeError, match="regular file"):
                probe_clip(clip)
        finally:
            monkey.undo()


class TestGrowthDuringProbe:
    """A cap checked once is a claim about the past."""

    class Endless:
        """A handle that never stops yielding bytes."""

        def seek(self, *args): return 0
        def read(self, size): return b"\x00" * size

    def test_a_clip_that_grows_past_the_cap_is_refused(self, tmp_path):
        with pytest.raises(ClipProbeError, match="too large|exceeded"):
            clip_probe._digest(self.Endless(), 1000, tmp_path / "c.wav")

    def test_the_streaming_cap_names_the_limit(self, tmp_path):
        with pytest.raises(ClipProbeError, match="4096"):
            clip_probe._digest(self.Endless(), 4096, tmp_path / "c.wav")

    def test_appending_during_the_probe_is_detected(self, tmp_path, monkeypatch):
        """Growth in the read window changes size and mtime, so it is caught."""
        clip = write_wav(tmp_path / "c.wav")

        def grow(path):
            with open(path, "ab") as grower:
                grower.write(b"\x00" * 4096)

        during_digest(monkeypatch, grow)

        with pytest.raises(ClipProbeError, match="changed while being measured"):
            probe_clip(clip)

    def test_a_same_size_rewrite_with_restored_mtime_is_detected(
        self, tmp_path, monkeypatch
    ):
        """Only `st_ctime_ns` sees this: same length, mtime put back."""

        clip = write_wav(tmp_path / "c.wav", frames=2400)
        original = clip.stat()

        def rewrite(path):
            other = write_wav(tmp_path / "other.wav", frames=2400)
            replacement = other.read_bytes()
            assert len(replacement) == original.st_size, "fixture must match size"
            path.write_bytes(replacement)
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            assert path.stat().st_mtime_ns == original.st_mtime_ns

        during_digest(monkeypatch, rewrite)

        with pytest.raises(ClipProbeError, match="changed while being measured"):
            probe_clip(clip)

    def test_an_unchanged_clip_still_measures(self, tmp_path):
        """The identity check must not reject an ordinary, quiet file."""
        clip = write_wav(tmp_path / "c.wav")

        assert probe_clip(clip).byte_size == clip.stat().st_size


class TestRaceFailuresAreChained:
    """A failed `stat`, `open`, or read surfaces as `ClipProbeError`."""

    def _boom(self, *args, **kwargs):
        raise OSError("boom")

    @pytest.mark.parametrize(
        "target,attr,message",
        [
            ("os", "open", "could not be opened"),
            ("path", "is_file", "could not be inspected"),
            ("os", "fstat", "state could not be read"),
        ],
    )
    def test_a_failed_call_is_chained(self, tmp_path, monkeypatch, target, attr, message):
        clip = write_wav(tmp_path / "c.wav")
        owner = Path if target == "path" else clip_probe.os

        monkeypatch.setattr(owner, attr, self._boom)

        with pytest.raises(ClipProbeError, match=message) as excinfo:
            probe_clip(clip)

        assert isinstance(excinfo.value.__cause__, OSError)

    def test_a_failed_read_is_chained(self, tmp_path):
        clip = write_wav(tmp_path / "c.wav")

        class Failing:
            def seek(self, *args): return 0
            def read(self, size): raise OSError("read error")

        with pytest.raises(ClipProbeError, match="could not be read") as excinfo:
            clip_probe._digest(Failing(), MAX_CLIP_BYTES, clip)

        assert isinstance(excinfo.value.__cause__, OSError)
