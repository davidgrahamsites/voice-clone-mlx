"""Test what the clip probe refuses.

Path safety, the byte cap, malformed audio, and purity. Races — a clip swapped
or grown mid-probe — live in `test_clip_probe_race.py`; measuring a valid clip
and the measurement contract in `test_clip_probe.py`.

A digest and a duration reach a dataset manifest as fact, so every refusal here
exists because the alternative is a plausible number recorded for something
never really measured.
"""

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


class TestPathSafety:
    """A clip is a real regular file, reached without following a link."""

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(ClipProbeError, match="not found|no such"):
            probe_clip(tmp_path / "absent.wav")

    def test_a_directory_is_refused(self, tmp_path):
        (tmp_path / "clips").mkdir()

        with pytest.raises(ClipProbeError, match="file"):
            probe_clip(tmp_path / "clips")

    def test_a_symlink_is_refused(self, tmp_path):
        """Refused rather than followed: what is measured must be what is named."""
        real = write_wav(tmp_path / "real.wav")
        link = tmp_path / "link.wav"
        link.symlink_to(real)

        with pytest.raises(ClipProbeError, match="symlink"):
            probe_clip(link)

    def test_a_symlinked_parent_is_still_measured_by_its_own_path(self, tmp_path):
        """Only the clip itself is checked for being a link, not its ancestors."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        clip = write_wav(real_dir / "c.wav")
        (tmp_path / "alias").symlink_to(real_dir)

        assert probe_clip(tmp_path / "alias" / "c.wav").byte_size == clip.stat().st_size

    @pytest.mark.parametrize("value", [None, 42, [], {}, b"clip.wav"])
    def test_a_non_path_is_refused(self, value):
        with pytest.raises(ClipProbeError):
            probe_clip(value)

    def test_an_empty_path_is_refused(self):
        with pytest.raises(ClipProbeError):
            probe_clip("")


class TestByteCap:
    """The cap is enforced from the file size, before anything is read."""

    def test_a_file_over_the_cap_is_refused(self, tmp_path):
        clip = write_wav(tmp_path / "big.wav", frames=5000)

        with pytest.raises(ClipProbeError, match="too large|bytes"):
            probe_clip(clip, max_bytes=100)

    def test_the_cap_names_the_limit(self, tmp_path):
        clip = write_wav(tmp_path / "big.wav", frames=5000)

        with pytest.raises(ClipProbeError, match="100"):
            probe_clip(clip, max_bytes=100)

    def test_nothing_is_hashed_when_the_cap_is_exceeded(self, tmp_path, monkeypatch):
        """Refused on size before a byte is hashed. The file *is* opened first:"""
        clip = write_wav(tmp_path / "big.wav", frames=5000)

        def explode(*args, **kwargs):
            raise AssertionError("the clip was hashed despite exceeding the cap")

        monkeypatch.setattr(clip_probe.hashlib, "sha256", explode)

        with pytest.raises(ClipProbeError, match="too large"):
            probe_clip(clip, max_bytes=100)

    def test_a_file_at_exactly_the_cap_is_accepted(self, tmp_path):
        clip = write_wav(tmp_path / "c.wav")

        assert probe_clip(clip, max_bytes=clip.stat().st_size) is not None

    @pytest.mark.parametrize("bad", [0, -1, -1024])
    def test_a_non_positive_cap_is_refused(self, tmp_path, bad):
        clip = write_wav(tmp_path / "c.wav")

        with pytest.raises(ClipProbeError, match="max_bytes"):
            probe_clip(clip, max_bytes=bad)

    @pytest.mark.parametrize("bad", [1.5, "1024", None, [], True])
    def test_a_non_integer_cap_is_refused(self, tmp_path, bad):
        """`True` included: `bool` is an `int` subclass."""
        clip = write_wav(tmp_path / "c.wav")

        with pytest.raises(ClipProbeError, match="max_bytes"):
            probe_clip(clip, max_bytes=bad)

    def test_the_default_cap_applies_without_an_argument(self, tmp_path):
        clip = write_wav(tmp_path / "c.wav")

        assert probe_clip(clip).byte_size < MAX_CLIP_BYTES


class TestUnreadableAudio:
    """Anything `wave` cannot read is a refusal, not a guess."""

    def test_an_empty_file_is_refused(self, tmp_path):
        clip = tmp_path / "empty.wav"
        clip.write_bytes(b"")

        with pytest.raises(ClipProbeError):
            probe_clip(clip)

    def test_non_wav_bytes_are_refused(self, tmp_path):
        clip = tmp_path / "fake.wav"
        clip.write_bytes(b"not audio at all, just text pretending")

        with pytest.raises(ClipProbeError, match="WAV|wav"):
            probe_clip(clip)

    def test_a_truncated_header_is_refused(self, tmp_path):
        full = write_wav(tmp_path / "full.wav").read_bytes()
        clip = tmp_path / "cut.wav"
        clip.write_bytes(full[:20])

        with pytest.raises(ClipProbeError):
            probe_clip(clip)

    def test_the_underlying_error_is_chained(self, tmp_path):
        clip = tmp_path / "fake.wav"
        clip.write_bytes(b"nonsense")

        with pytest.raises(ClipProbeError) as excinfo:
            probe_clip(clip)

        assert excinfo.value.__cause__ is not None

    def test_a_zero_frame_rate_is_refused_rather_than_divided_by(self, tmp_path):
        clip = write_wav(tmp_path / "c.wav")

        class ZeroRate:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def getframerate(self): return 0
            def getnchannels(self): return 1
            def getsampwidth(self): return 2
            def getnframes(self): return 100

        monkey = pytest.MonkeyPatch()
        monkey.setattr(clip_probe.wave, "open", lambda *a, **k: ZeroRate())
        try:
            with pytest.raises(ClipProbeError, match="frame rate|rate"):
                probe_clip(clip)
        finally:
            monkey.undo()


class TestSeamIsPure:
    """Stdlib only: no backend, no model, no process, no network, no clock."""

    def _imports(self):
        source = Path(clip_probe.__file__).read_text(encoding="utf-8")
        return " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

    @pytest.mark.parametrize(
        "banned",
        ["mlx", "qwen", "torch", "numpy", "subprocess", "socket", "urllib",
         "requests", "random", "tkinter"],
    )
    def test_no_dangerous_import(self, banned):
        assert banned not in self._imports()

    def test_no_clock_import(self):
        """Needles built at runtime so this cannot match its own source."""
        for name in ("time", "date" + "time"):
            assert f"import {name}" not in self._imports()

    def _code(self):
        """Executable lines only — docstrings and comments are prose, not logic."""
        import ast

        source = Path(clip_probe.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    source = source.replace(doc, "")

        return "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )

    def test_nothing_is_written(self):
        code = self._code()

        for needle in ("write_text", "write_bytes", "mkdir", "unlink", "replace("):
            assert needle not in code

    def test_the_whole_file_is_never_read_at_once(self):
        """`read_bytes` would defeat the cap it is meant to enforce."""
        assert "read_bytes" not in self._code()

    def test_no_synthesis_layer_is_imported(self):
        imports = self._imports()

        for layer in ("tts", "synthesis", "reader_app", "studio_app"):
            assert layer not in imports
