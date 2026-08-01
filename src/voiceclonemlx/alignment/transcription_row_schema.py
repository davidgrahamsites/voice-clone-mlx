"""What a transcription manifest field may be.

The vocabulary layer for `transcription_manifest`: the schema version, the
exact key set, the error type, and one validator per field kind. It holds no
row, builds no manifest, and reads nothing — it only answers "is this value
allowed here".

Split from `transcription_manifest.py` on the same seam the dataset modules
use: *what a field may be* here, *what a row does* there. Callers import from
`transcription_manifest`, which re-exports every public name below.
"""

import math
from typing import Any

#: The only schema this module writes or accepts. Validated like every other
#: field: without that, a row could claim a schema whose rules were never
#: applied — every value checked except the label saying which checks ran.
TRANSCRIPTION_SCHEMA_VERSION = "1"

#: Keys always present on a serialized row.
REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "master_audio",
        "segment_id",
        "start_s",
        "end_s",
        "text",
        "transcriber",
        "transcriber_version",
    }
)

#: Written only when set, so its absence is meaningful rather than a null.
OPTIONAL_KEYS = frozenset({"language"})

ROW_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS

#: Refused anywhere in `master_audio`. A manifest names a local recording; a
#: scheme would make it a place to reach, which is a different kind of thing.
REMOTE_MARKERS = ("://",)

#: Refused in a `segment_id`. An id labels a row and must never be usable as a
#: path fragment by something that later joins it to a directory.
ID_SEPARATORS = ("/", "\\", "..", "\0")

#: Zero-padding for positional ids. Wide enough that ordinary sessions sort
#: lexicographically; longer transcripts simply produce longer ids rather than
#: colliding.
ID_DIGITS = 4


class TranscriptionRowError(ValueError):
    """A transcription row or manifest is not usable."""


def checked_label(value: Any, field: str) -> str:
    """A non-blank string, returned unchanged.

    Used for free-text identity fields — the transcriber and its version. They
    are recorded exactly as supplied and never parsed: this module has no
    business deciding what a version number looks like.
    """
    if not isinstance(value, str) or not value.strip():
        raise TranscriptionRowError(f"{field} must be a non-empty string: {value!r}")
    return value


def checked_optional_label(value: Any, field: str):
    """A non-blank string or None."""
    if value is None:
        return None
    return checked_label(value, field)


def checked_relative_path(value: Any, field: str) -> str:
    """A relative, local, traversal-free path, returned unchanged.

    Refuses absolute paths, `..` in any position, and anything carrying a URL
    scheme. The string is not resolved and no filesystem is touched — this is a
    statement about the recorded value, not about a file that must exist.
    """
    if not isinstance(value, str) or not value.strip():
        raise TranscriptionRowError(f"{field} must be a non-empty string: {value!r}")

    for marker in REMOTE_MARKERS:
        if marker in value:
            raise TranscriptionRowError(f"{field} must be local: {value!r}")

    if value.startswith(("/", "\\")):
        raise TranscriptionRowError(f"{field} must be relative: {value!r}")

    # Checked on both separators: a manifest written on one platform is read on
    # another, and `..` escapes just as well either way.
    parts = value.replace("\\", "/").split("/")
    if ".." in parts:
        raise TranscriptionRowError(f"{field} must not traverse upward: {value!r}")

    return value


def checked_segment_id(value: Any, field: str = "segment_id") -> str:
    """A non-blank id with nothing that could act as a path."""
    if not isinstance(value, str) or not value.strip():
        raise TranscriptionRowError(f"{field} must be a non-empty string: {value!r}")

    for separator in ID_SEPARATORS:
        if separator in value:
            raise TranscriptionRowError(
                f"{field} must not contain {separator!r}: {value!r}"
            )

    return value


def checked_time(value: Any, field: str) -> float:
    """A real, finite, non-negative number as a float.

    `bool` is excluded on purpose: it is an `int` subclass, so `True` would
    otherwise pass as the number 1. This mirrors `whisper_json._valid_time`;
    the rule is the same because the values are the same values.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptionRowError(f"{field} must be a number: {value!r}")

    if not math.isfinite(value) or value < 0:
        raise TranscriptionRowError(
            f"{field} must be finite and non-negative: {value!r}"
        )

    return float(value)


def checked_text(value: Any, field: str = "text") -> str:
    """Non-blank text, stored exactly as produced.

    Emptiness is judged on a stripped copy; the value kept is unchanged.
    Whisper's spacing is evidence about what was said and how it was segmented,
    and alignment compares it against expected script text — normalizing here
    would quietly change that comparison.
    """
    if not isinstance(value, str) or not value.strip():
        raise TranscriptionRowError(f"{field} must be non-blank text: {value!r}")
    return value


def checked_schema_version(value: Any, field: str = "schema_version") -> str:
    """Exactly the version this module implements."""
    if value != TRANSCRIPTION_SCHEMA_VERSION:
        raise TranscriptionRowError(
            f"{field} must be {TRANSCRIPTION_SCHEMA_VERSION!r}: {value!r}"
        )
    return value


def checked_keys(keys, line_no: int) -> None:
    """Exactly the v1 keys: no unknown, no missing required."""
    present = set(keys)

    unknown = sorted(present - ROW_KEYS)
    if unknown:
        raise TranscriptionRowError(
            f"line {line_no}: unknown keys: {', '.join(unknown)}"
        )

    missing = sorted(REQUIRED_KEYS - present)
    if missing:
        raise TranscriptionRowError(
            f"line {line_no}: missing keys: {', '.join(missing)}"
        )


def derive_segment_id(master_audio: str, index: int) -> str:
    """The id for the segment at `index`, derived from position.

    Deterministic by construction: the same transcript and the same master
    audio always produce the same ids, so a manifest can be regenerated and
    compared rather than trusted. Because it is derived, a parser can recompute
    what an id *should* be and refuse a manifest whose rows were renamed or
    reordered.

    The stem is used rather than the whole path so an id stays a label — a
    directory in it would make ids look like paths, which `checked_segment_id`
    refuses for good reason.
    """
    checked_relative_path(master_audio, "master_audio")

    stem = master_audio.replace("\\", "/").rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]

    if not stem.strip():
        raise TranscriptionRowError(
            f"master_audio has no usable name: {master_audio!r}"
        )

    return checked_segment_id(f"{stem}-{index:0{ID_DIGITS}d}")
