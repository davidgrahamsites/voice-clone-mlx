"""Field constants and validators for DatasetManifest v1 rows.

The vocabulary of the artifact: its schema version, split set, audio-property
key set, exact row/document key sets, typed error, and one validator per field
kind. It builds no rows and serializes nothing — `dataset_rows.py` does that
and re-exports the public names from here, so callers keep importing from one
place.

Split out when the row module crossed the ICM size threshold; the seam is
"what a field may be" versus "what a row does".

Pure: no audio, no hashing, no clock, no filesystem, no network.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType

from voiceclonegpt.shared.styles import CANONICAL_STYLES

#: The only `ClipDecision.status` that admits a clip. The rule that produces
#: it lives in `alignment/overlap_gate.py`; this is only the value recorded.
ACCEPT_STATUS = "accept"

DATASET_SCHEMA_VERSION = "1.0.0"

#: The only splits a row may belong to.
SPLITS = ("train", "validation", "test")

#: Exactly the audio properties a clip must declare.
AUDIO_INTEGER_FIELDS = ("sample_rate_hz", "channels", "bit_depth")
AUDIO_FIELDS = AUDIO_INTEGER_FIELDS + ("duration_s",)

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")

_ROW_FIELDS = (
    "schema_version",
    "utterance_id",
    "text",
    "style",
    "clip_path",
    "clip_sha256",
    "audio_properties",
    "session_id",
    "split",
    "alignment_master_audio",
    "accepted_by",
    "accepted_at",
    "clip_decision_status",
    "clip_decision_reason",
)

_DOCUMENT_FIELDS = ("schema_version", "rows")


class DatasetRowError(ValueError):
    """A row is not admissible, or a manifest is unusable."""


def _text(value, field: str) -> str:
    """Return a non-blank string, or refuse."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise DatasetRowError(f"{field} must be a string: {value!r}")
    if not value.strip():
        raise DatasetRowError(f"{field} must not be blank")
    return value


def _relative_path(value, field: str) -> str:
    """Return a manifest-safe relative path.

    Absolute paths are forbidden inside a manifest: an artifact pointing
    outside its own run is not portable evidence.
    """
    path = _text(value, field)

    if path.startswith("/") or path.startswith("\\") or ":" in path:
        raise DatasetRowError(f"{field} must be relative: {value!r}")
    if ".." in path.replace("\\", "/").split("/"):
        raise DatasetRowError(f"{field} must not escape the run: {value!r}")

    return path


def _sha256(value) -> str:
    """Return a lowercase 64-character hex digest.

    The digest is **supplied by the caller**. This module never computes one:
    hashing means reading the clip, and that would make an artifact contract
    into a file-touching utility.
    """
    digest = _text(value, "clip_sha256")

    if len(digest) != _SHA256_LENGTH or not set(digest) <= _HEX_DIGITS:
        raise DatasetRowError(
            f"clip_sha256 must be {_SHA256_LENGTH} lowercase hex characters: "
            f"{value!r}"
        )

    return digest


def _split(value) -> str:
    """Return one of the known splits."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise DatasetRowError(f"split must be a string: {value!r}")
    if value not in SPLITS:
        raise DatasetRowError(
            f"split must be one of {', '.join(SPLITS)}: {value!r}"
        )
    return value


def _audio_properties(value) -> Mapping:
    """Validate and freeze the audio properties.

    Exactly four keys, no more and no fewer. A copy is taken, so a caller
    mutating their dict afterwards cannot change an existing row.
    """
    if not isinstance(value, Mapping):
        raise DatasetRowError(
            f"audio_properties must be a mapping: {value!r}"
        )

    present = set(value)
    expected = set(AUDIO_FIELDS)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise DatasetRowError(
            f"audio_properties is missing: {', '.join(missing)}"
        )
    if unexpected:
        raise DatasetRowError(
            f"audio_properties has unexpected key(s): {', '.join(unexpected)}"
        )

    checked = {}
    for field in AUDIO_INTEGER_FIELDS:
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, int):
            raise DatasetRowError(
                f"audio_properties.{field} must be an integer: {number!r}"
            )
        if number <= 0:
            raise DatasetRowError(
                f"audio_properties.{field} must be positive: {number!r}"
            )
        checked[field] = number

    duration = value["duration_s"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise DatasetRowError(
            f"audio_properties.duration_s must be a number: {duration!r}"
        )
    if not math.isfinite(duration) or duration <= 0:
        raise DatasetRowError(
            f"audio_properties.duration_s must be finite and positive: "
            f"{duration!r}"
        )
    checked["duration_s"] = float(duration)

    return MappingProxyType(checked)


def _style(value) -> str:
    """Return a canonical style.

    Taken from `shared.styles`, which is the one home for the list. Aliases are
    not resolved here: by the time a row reaches the dataset the style was
    normalized upstream, and a manifest records canonical values only.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise DatasetRowError(f"style must be a string: {value!r}")
    if value not in CANONICAL_STYLES:
        raise DatasetRowError(
            f"style must be canonical: {value!r}; known styles: "
            f"{', '.join(sorted(CANONICAL_STYLES))}"
        )
    return value


def _accept_status(value) -> str:
    """Return the gate's status, admitting only `accept`."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise DatasetRowError(
            f"clip_decision_status must be a string: {value!r}"
        )
    if value != ACCEPT_STATUS:
        raise DatasetRowError(
            f"clip_decision_status must be {ACCEPT_STATUS!r}; a clip the gate "
            f"did not accept cannot enter the dataset: {value!r}"
        )
    return value


def _schema_version(value) -> str:
    """Return the supported schema version, or refuse.

    Checked on every row, however it was built. Without this, direct
    construction could mint a row labelled with a version whose rules were
    never applied — every other field is validated, so the label must be too.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise DatasetRowError(f"schema_version must be a string: {value!r}")
    if value != DATASET_SCHEMA_VERSION:
        raise DatasetRowError(
            f"unsupported schema_version: {value!r}; this module implements "
            f"{DATASET_SCHEMA_VERSION!r}"
        )
    return value
