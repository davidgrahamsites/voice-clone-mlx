"""Field constants and validators for AlignmentManifest v1 rows.

The vocabulary of the artifact: its schema version, review states, exact key
set, typed error, and one validator per field kind. It builds no rows and
serializes nothing — `alignment_rows.py` does that and re-exports the public
names from here, so callers keep importing from one place.

Split out because the row module crossed the size threshold; the seam is
"what a field may be" versus "what a row does".

Pure: no clock, no filesystem, no audio, no model, no network.
"""

import math
from typing import Optional, Sequence, Tuple

from voiceclonemlx.shared.styles import CANONICAL_STYLES

ALIGNMENT_SCHEMA_VERSION = "1.0.0"

REVIEW_PENDING = "pending"
REVIEW_ACCEPTED = "accepted"
REVIEW_REJECTED = "rejected"

REVIEW_STATES = (REVIEW_PENDING, REVIEW_ACCEPTED, REVIEW_REJECTED)

#: States that record who decided, and when.
_DECIDED_STATES = (REVIEW_ACCEPTED, REVIEW_REJECTED)

_FIELDS = (
    "schema_version",
    "utterance_id",
    "expected_text",
    "observed_text",
    "style",
    "start_s",
    "end_s",
    "segment_ids",
    "master_audio",
    "confidence",
    "mismatch_reasons",
    "review_state",
    "reviewed_by",
    "reviewed_at",
)


class AlignmentRowError(ValueError):
    """A row is unusable, or a review transition is not allowed."""


def _text(value, field: str, *, allow_empty: bool = False) -> str:
    """Return a string field, preserving it exactly."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise AlignmentRowError(f"{field} must be a string: {value!r}")
    if not allow_empty and not value.strip():
        raise AlignmentRowError(f"{field} must not be blank")
    return value


def _time(value, field: str) -> float:
    """Return a finite, non-negative time.

    `bool` is excluded: it is an `int` subclass, so `True` would pass as 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentRowError(f"{field} must be a number: {value!r}")
    if not math.isfinite(value) or value < 0:
        raise AlignmentRowError(
            f"{field} must be finite and non-negative: {value!r}"
        )
    return float(value)


def _style(value) -> str:
    """Return a canonical style.

    Aliases are *not* resolved here. Normalization belongs upstream, at the
    point the marker was recognized; a manifest records canonical styles only,
    so a reviewer never has to wonder which spelling meant what.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise AlignmentRowError(f"style must be a string: {value!r}")
    if value not in CANONICAL_STYLES:
        raise AlignmentRowError(
            f"style must be canonical: {value!r}; known styles: "
            f"{', '.join(sorted(CANONICAL_STYLES))}"
        )
    return value


def _string_tuple(value, field: str) -> Tuple[str, ...]:
    """Return a tuple of non-blank strings."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlignmentRowError(f"{field} must be a sequence of strings: {value!r}")

    entries = tuple(value)
    for entry in entries:
        if isinstance(entry, bool) or not isinstance(entry, str) or not entry.strip():
            raise AlignmentRowError(
                f"{field} entries must be non-blank strings: {entry!r}"
            )
    return entries


def _relative_reference(value) -> str:
    """Return a manifest-safe relative path.

    The lifecycle contract forbids absolute paths inside a manifest: an
    artifact that points outside its own run is not portable evidence.
    """
    reference = _text(value, "master_audio")

    if reference.startswith("/") or reference.startswith("\\"):
        raise AlignmentRowError(f"master_audio must be relative: {value!r}")
    if ":" in reference:
        raise AlignmentRowError(f"master_audio must be relative: {value!r}")
    if ".." in reference.replace("\\", "/").split("/"):
        raise AlignmentRowError(
            f"master_audio must not escape the run: {value!r}"
        )

    return reference


def _confidence(value) -> Optional[float]:
    """Return a confidence in [0, 1], or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentRowError(f"confidence must be a number or None: {value!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AlignmentRowError(f"confidence must be between 0 and 1: {value!r}")
    return float(value)

