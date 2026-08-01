"""Measure a produced clip: its checksum and its audio properties.

One job: read a WAV already on disk and report the two things
`build_dataset_row` requires and cannot compute for itself — a SHA-256 digest
and the four audio properties. It synthesizes nothing, loads no model, imports
no backend, starts no process, touches no network, and writes no file.

`dataset_rows` takes the digest as a given because hashing means reading a file
and an artifact contract should not be a file-touching utility. This seam reads,
so the number is produced by code rather than typed by hand.

**Measured, never assumed**, from **one descriptor**: the clip is opened once
with `O_NOFOLLOW`, and size, digest, and header all come from that open file.
The name is re-checked afterwards, because a path can be renamed away from the
bytes that were read.

Stdlib only — `wave`, `hashlib`, `os`, `stat`, `errno`, `pathlib`.
"""

import errno
import hashlib
import os
import stat as stat_module
import wave
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

#: Refuse anything larger before reading a byte. A clip is one utterance, not
#: a session: a file this size is a mistake worth surfacing.
MAX_CLIP_BYTES = 64 * 1024 * 1024

#: Read size for the digest. The file is streamed, never loaded whole, so the
#: cap above bounds the work rather than the memory after the fact.
CHUNK_BYTES = 1024 * 1024

#: Exactly the keys `dataset_row_schema.AUDIO_FIELDS` requires, so the result
#: feeds `build_dataset_row` with no translation step to get wrong.
AUDIO_PROPERTY_KEYS = ("sample_rate_hz", "channels", "bit_depth", "duration_s")

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")

#: Bits per byte, for turning `wave`'s sample width into a bit depth.
_BITS_PER_BYTE = 8

#: Refuses a final-component symlink at open time, closing the gap between
#: "this is not a link" and "this is the file I opened". See `_open_no_follow`.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_HAS_NOFOLLOW = _NOFOLLOW != 0


class ClipProbeError(ValueError):
    """A clip cannot be measured."""


def _checked_digest(value: Any) -> str:
    """A lowercase 64-hex SHA-256 digest."""
    if not isinstance(value, str):
        raise ClipProbeError(f"clip_sha256 must be a string: {value!r}")

    if len(value) != _SHA256_LENGTH or not set(value) <= _HEX_DIGITS:
        raise ClipProbeError(
            f"clip_sha256 must be {_SHA256_LENGTH} lowercase hex characters: "
            f"{value!r}"
        )

    return value


def _checked_count(value: Any, field: str) -> int:
    """A positive integer. `bool` is excluded — it is an `int` subclass."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClipProbeError(f"{field} must be an integer: {value!r}")

    if value <= 0:
        raise ClipProbeError(f"{field} must be positive: {value!r}")

    return value


def _checked_duration(value: Any) -> float:
    """A finite, strictly positive number of seconds.

    Matches `dataset_row_schema`'s `duration_s > 0`: a zero-frame WAV is real,
    but no dataset row could be built from it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClipProbeError(f"duration_s must be a number: {value!r}")

    # `!=` catches NaN, which compares false against everything including itself.
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:
        raise ClipProbeError(
            f"duration_s must be finite and positive: {value!r}"
        )

    return float(value)


def _checked_properties(properties: Any) -> dict:
    """Exactly the four contract keys, each a usable value."""
    if not isinstance(properties, Mapping):
        raise ClipProbeError(
            f"audio_properties must be a mapping: {type(properties).__name__}"
        )

    if set(properties) != set(AUDIO_PROPERTY_KEYS):
        expected = ", ".join(AUDIO_PROPERTY_KEYS)
        raise ClipProbeError(
            f"audio_properties must have exactly ({expected}): "
            f"{sorted(properties)}"
        )

    checked = {
        key: _checked_count(properties[key], key)
        for key in AUDIO_PROPERTY_KEYS
        if key != "duration_s"
    }
    checked["duration_s"] = _checked_duration(properties["duration_s"])

    return {key: checked[key] for key in AUDIO_PROPERTY_KEYS}


@dataclass(frozen=True)
class ClipMeasurement:
    """What was measured from one clip.

    Validated in `__post_init__`, so a hand-built measurement faces the same
    checks `probe_clip` produces: a digest recorded here becomes fact in a
    dataset manifest.
    """

    clip_sha256: str
    audio_properties: Mapping[str, Any]
    byte_size: int

    def __post_init__(self) -> None:
        set_field = object.__setattr__

        set_field(self, "clip_sha256", _checked_digest(self.clip_sha256))
        set_field(
            self,
            "audio_properties",
            # Copied then frozen: a caller mutating their mapping afterwards
            # cannot change what an existing measurement says.
            MappingProxyType(_checked_properties(self.audio_properties)),
        )

        if isinstance(self.byte_size, bool) or not isinstance(self.byte_size, int):
            raise ClipProbeError(f"byte_size must be an integer: {self.byte_size!r}")

        if self.byte_size < 0:
            raise ClipProbeError(
                f"byte_size must not be negative: {self.byte_size!r}"
            )


def _resolved_clip(path: Any) -> Path:
    """The clip as an existing path, or raise."""
    if isinstance(path, Path):
        candidate = path
    elif isinstance(path, str) and path.strip():
        candidate = Path(path)
    else:
        raise ClipProbeError(f"clip path must be a non-empty path or string: {path!r}")

    if candidate.is_symlink():
        raise ClipProbeError(f"clip path is a symlink: {candidate}")

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ClipProbeError(f"clip not found: {candidate}") from exc

    try:
        is_file = resolved.is_file()
    except OSError as exc:
        raise ClipProbeError(f"clip could not be inspected: {resolved}") from exc

    if not is_file:
        raise ClipProbeError(f"clip path is not a file: {resolved}")

    return resolved


def _identity(handle) -> tuple:
    """The open file's identity and state, from the descriptor.

    `os.fstat` asks about *this* open file, not whatever answers to the path
    now. Comparing before and after catches a clip replaced mid-probe.
    """
    try:
        status = os.fstat(handle.fileno())
    except OSError as exc:
        raise ClipProbeError(f"clip state could not be read: {exc}") from exc

    # `st_ctime_ns` matters: a same-length rewrite can leave size identical and
    # mtime restorable, but change-time still moves.
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _digest(handle, max_bytes: int, path: Path) -> str:
    """Stream the open file into SHA-256, stopping if it outgrows the cap.

    Chunked and counted as it goes: a size checked once is a claim about the
    past, and a growing file would outrun it.
    """
    digest = hashlib.sha256()
    total = 0

    try:
        handle.seek(0)
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break

            total += len(chunk)
            if total > max_bytes:
                raise ClipProbeError(
                    f"clip is too large: exceeded max_bytes={max_bytes} "
                    f"while reading {path}"
                )

            digest.update(chunk)
    except OSError as exc:
        raise ClipProbeError(f"clip could not be read: {path}") from exc

    return digest.hexdigest()


def _properties(handle, path: Path) -> dict:
    """Read the four audio properties from the same open file.

    Reopening by name would let the properties describe a different file than
    the digest covered. `duration_s` is frames over frame rate; frames count
    across channels, so stereo is not twice as long as mono.
    """
    try:
        handle.seek(0)
        reader = wave.open(handle, "rb")
        frame_rate = reader.getframerate()
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frames = reader.getnframes()
    except (OSError, wave.Error, EOFError) as exc:
        raise ClipProbeError(f"not a readable WAV: {path}") from exc

    if not frame_rate:
        raise ClipProbeError(f"WAV declares no frame rate: {path}")

    if not frames:
        raise ClipProbeError(
            f"clip has no audio frames, so its duration would be zero and no "
            f"dataset row could be built from it: {path}"
        )

    return {
        "sample_rate_hz": frame_rate,
        "channels": channels,
        "bit_depth": sample_width * _BITS_PER_BYTE,
        "duration_s": frames / frame_rate,
    }


def _open_no_follow(clip: Path):
    """Open the clip, refusing a final-component symlink at open time.

    Checking `is_symlink()` then opening by name leaves a window: the name can
    become a link in between. `O_NOFOLLOW` closes it in the kernel. Without the
    flag, `lstat` is compared against the descriptor and a mismatch refused —
    detection, not prevention; stated, not glossed.
    """
    try:
        descriptor = os.open(str(clip), os.O_RDONLY | _NOFOLLOW)
    except OSError as exc:
        if _HAS_NOFOLLOW and exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ClipProbeError(
                f"clip path is a symlink: {clip}"
            ) from exc
        raise ClipProbeError(f"clip could not be opened: {clip}") from exc

    try:
        handle = os.fdopen(descriptor, "rb")
    except OSError as exc:
        os.close(descriptor)
        raise ClipProbeError(f"clip could not be opened: {clip}") from exc

    if not _HAS_NOFOLLOW:
        try:
            named = os.lstat(str(clip))
            opened = os.fstat(handle.fileno())
        except OSError as exc:
            handle.close()
            raise ClipProbeError(f"clip could not be inspected: {clip}") from exc

        if stat_module.S_ISLNK(named.st_mode) or (
            named.st_dev, named.st_ino
        ) != (opened.st_dev, opened.st_ino):
            handle.close()
            raise ClipProbeError(
                f"clip path is a symlink or was replaced while opening: {clip}"
            )

    return handle


def _check_regular_file(handle, clip: Path) -> None:
    """Confirm from the descriptor that this is a regular file, so a directory
    or device that replaced the path cannot be measured as a clip."""
    try:
        mode = os.fstat(handle.fileno()).st_mode
    except OSError as exc:
        raise ClipProbeError(f"clip state could not be read: {exc}") from exc

    if not stat_module.S_ISREG(mode):
        raise ClipProbeError(f"clip path is not a regular file: {clip}")


def _check_still_named(handle, clip: Path) -> None:
    """Confirm the path still names the file that was measured.

    The descriptor keeps its own inode alive, so an `os.replace` onto the path
    leaves the read intact: a true digest of bytes the manifest's path no
    longer refers to. `lstat` of the name against `fstat` of the descriptor
    catches that — `lstat`, so a symlink dropped in place is a mismatch rather
    than something to follow.
    """
    try:
        named = os.lstat(str(clip))
        opened = os.fstat(handle.fileno())
    except OSError as exc:
        raise ClipProbeError(f"clip could not be re-checked: {clip}") from exc

    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise ClipProbeError(
            f"clip was replaced while being measured; the path no longer "
            f"names the file that was read: {clip}"
        )


def probe_clip(path, *, max_bytes: int = MAX_CLIP_BYTES) -> ClipMeasurement:
    """Measure one clip's digest and audio properties."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise ClipProbeError(f"max_bytes must be an integer: {max_bytes!r}")

    if max_bytes <= 0:
        raise ClipProbeError(f"max_bytes must be positive: {max_bytes!r}")

    clip = _resolved_clip(path)

    handle = _open_no_follow(clip)

    with handle:
        # Every fact below comes from this descriptor, never the name again.
        before = _identity(handle)
        byte_size = before[2]

        _check_regular_file(handle, clip)

        if byte_size > max_bytes:
            raise ClipProbeError(
                f"clip is too large: {byte_size} bytes exceeds "
                f"max_bytes={max_bytes}"
            )

        clip_sha256 = _digest(handle, max_bytes, clip)
        audio_properties = _properties(handle, clip)

        # A changed size, mtime, ctime, or inode means the bytes just hashed
        # are no longer the bytes on disk.
        if _identity(handle) != before:
            raise ClipProbeError(
                f"clip changed while being measured: {clip}"
            )

        _check_still_named(handle, clip)

    return ClipMeasurement(
        clip_sha256=clip_sha256,
        audio_properties=audio_properties,
        byte_size=byte_size,
    )
