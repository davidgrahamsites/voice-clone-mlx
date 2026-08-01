"""Decide whether a diarized clip is clean enough to enter the dataset.

One job: given diarization segments, say whether the clip is a single-speaker
recording of the target voice. Data in, verdict out — it decodes no audio, runs
no model, opens no file, and reaches no network. The diarizer that produced the
segments is somebody else's problem; so is what happens to a rejected clip.

The rule is deliberately unforgiving: **one overlap, or one syllable of another
voice anywhere in the clip, rejects the whole clip.** Source separation cannot
promote a mixed-speaker recording into training data, and a model trained on
someone else's voice fragments is not the user's voice.

Standard library only.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple

#: The clip contained no segments at all.
REASON_EMPTY = "empty"

#: A segment's times or speaker could not be used.
REASON_MALFORMED = "malformed"

#: A voice other than the target appears somewhere in the clip.
REASON_NON_TARGET_SPEAKER = "non_target_speaker"

#: Two segments cover the same instant.
REASON_OVERLAP = "overlap"

REASON_CODES = (
    REASON_EMPTY,
    REASON_MALFORMED,
    REASON_NON_TARGET_SPEAKER,
    REASON_OVERLAP,
)


@dataclass(frozen=True)
class NormalizedSegment:
    """One accepted span of the target speaker, times as floats."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class AcceptanceResult:
    """The verdict on one clip.

    `segments` is populated **only** when `accepted` is True, so a caller
    cannot accidentally use the spans of a clip that was refused.
    """

    accepted: bool
    reasons: Tuple[str, ...] = ()
    segments: Tuple[NormalizedSegment, ...] = ()


def _read(segment: Any, key: str):
    """Read a field from a mapping or an object, or return None."""
    if hasattr(segment, "get"):
        try:
            return segment.get(key)
        except Exception:
            return None
    return getattr(segment, key, None)


def _valid_time(value) -> bool:
    """True for a real, finite, non-negative number.

    `bool` is excluded on purpose: it is an `int` subclass, so `True` would
    otherwise pass as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _normalize(segment: Any):
    """Return a `NormalizedSegment`, or None if the segment is unusable."""
    start = _read(segment, "start")
    end = _read(segment, "end")
    speaker = _read(segment, "speaker")

    if not _valid_time(start) or not _valid_time(end):
        return None

    if float(start) >= float(end):
        return None

    if not isinstance(speaker, str) or not speaker.strip():
        return None

    return NormalizedSegment(start=float(start), end=float(end), speaker=speaker)


def validate_single_speaker_segments(
    segments: Iterable[Any], *, target_speaker: str
) -> AcceptanceResult:
    """Judge one clip's diarization segments.

    Args:
        segments: Iterable of mappings or objects with `start`, `end`, and
            `speaker`. Never mutated, and never reordered in place.
        target_speaker: The exact speaker label the clip must contain, and
            only contain. Matched exactly — no case folding, no stripping.

    Returns:
        An `AcceptanceResult`. Reasons are sorted and unique. Gaps between
        segments are silence and are **not** a reason to reject.

    Raises:
        ValueError: If `target_speaker` is not a non-empty string. That is a
            caller error — the target is the question being asked, not part of
            the data being judged — so it is not reported as a reason code.
    """
    if not isinstance(target_speaker, str) or not target_speaker.strip():
        raise ValueError(
            f"target_speaker must be a non-empty string: {target_speaker!r}"
        )

    original = list(segments)

    if not original:
        return AcceptanceResult(accepted=False, reasons=(REASON_EMPTY,))

    normalized = [_normalize(segment) for segment in original]

    # Unusable times make overlap unknowable, so this is reported alone rather
    # than guessing at further reasons from numbers we do not trust.
    if any(segment is None for segment in normalized):
        return AcceptanceResult(accepted=False, reasons=(REASON_MALFORMED,))

    # `sorted` builds a new list: the caller's sequence is left untouched.
    ordered = sorted(normalized, key=lambda s: (s.start, s.end))

    reasons = set()

    if any(segment.speaker != target_speaker for segment in ordered):
        reasons.add(REASON_NON_TARGET_SPEAKER)

    for earlier, later in zip(ordered, ordered[1:]):
        # Touching boundaries (`earlier.end == later.start`) are adjacency,
        # not overlap.
        if later.start < earlier.end:
            reasons.add(REASON_OVERLAP)
            break

    if reasons:
        return AcceptanceResult(accepted=False, reasons=tuple(sorted(reasons)))

    return AcceptanceResult(accepted=True, segments=tuple(ordered))

