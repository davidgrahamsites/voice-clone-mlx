"""Decide whether a diarized candidate clip is safe for dataset admission."""

import math
from dataclasses import dataclass
from typing import Sequence


class OverlapGateError(ValueError):
    """A span or speaker id was malformed, so no decision can be made."""


def _checked_time(value, field: str) -> float:
    """Coerce a span endpoint, or refuse.

    A malformed endpoint is not a value to work around. `NaN` compares False
    against everything, so an intersection test would silently miss a second
    voice and the gate would accept mixed audio.

    Raises:
        OverlapGateError: The value is not a finite, non-negative number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OverlapGateError(f"{field} must be a number: {value!r}")
    if not math.isfinite(value):
        raise OverlapGateError(f"{field} must be finite: {value!r}")
    if value < 0:
        raise OverlapGateError(f"{field} must not be negative: {value!r}")
    return value


def _checked_speaker(value, field: str) -> str:
    """Coerce a speaker id, or refuse."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise OverlapGateError(f"{field} must be a string: {value!r}")
    if not value.strip():
        raise OverlapGateError(f"{field} must not be blank")
    return value


@dataclass(frozen=True)
class SpeakerTurn:
    """A provider-neutral diarization span."""

    start_s: float
    end_s: float
    speaker_id: str
    overlap: bool = False

    def __post_init__(self) -> None:
        _checked_time(self.start_s, "start_s")
        _checked_time(self.end_s, "end_s")
        if not self.start_s < self.end_s:
            raise OverlapGateError(
                f"turn must advance: {self.start_s!r} -> {self.end_s!r}"
            )
        _checked_speaker(self.speaker_id, "speaker_id")
        if not isinstance(self.overlap, bool):
            raise OverlapGateError(f"overlap must be a bool: {self.overlap!r}")


#: The two statuses a `ClipDecision` can carry. Named here so a caller asking
#: "was this accepted?" does not have to spell the string a second time.
ACCEPT_STATUS = "accept"
REJECT_STATUS = "reject"


@dataclass(frozen=True)
class ClipDecision:
    """The only decisions the dataset builder needs from this gate."""

    status: str
    reason: str

    @classmethod
    def accept(cls, reason: str) -> "ClipDecision":
        return cls(ACCEPT_STATUS, reason)

    @classmethod
    def reject(cls, reason: str) -> "ClipDecision":
        return cls(REJECT_STATUS, reason)

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPT_STATUS


def decide_clip(
    *,
    clip_start_s: float,
    clip_end_s: float,
    target_speaker: str,
    turns: list[SpeakerTurn],
) -> ClipDecision:
    """Reject the entire clip if any second voice touches its time range.

    The gate intentionally does not attempt source separation. A clip is
    accepted only when at least one target turn intersects it and every
    intersecting turn belongs to the enrolled speaker with no overlap flag.

    Raises:
        OverlapGateError: A bound is not a finite, non-negative, advancing
            span; the target speaker is blank or not a string; or `turns` is
            not a sequence of `SpeakerTurn`.
    """

    _checked_time(clip_start_s, "clip_start_s")
    _checked_time(clip_end_s, "clip_end_s")
    if not clip_start_s < clip_end_s:
        raise OverlapGateError(
            f"clip must advance: {clip_start_s!r} -> {clip_end_s!r}"
        )
    _checked_speaker(target_speaker, "target_speaker")

    if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
        raise OverlapGateError(f"turns must be a sequence: {turns!r}")
    for turn in turns:
        if not isinstance(turn, SpeakerTurn):
            raise OverlapGateError(f"turns must contain SpeakerTurn: {turn!r}")

    intersecting = [
        turn
        for turn in turns
        if turn.end_s > clip_start_s and turn.start_s < clip_end_s
    ]
    if any(turn.overlap for turn in intersecting):
        return ClipDecision.reject("overlap_detected")

    speakers = {turn.speaker_id for turn in intersecting}
    if target_speaker not in speakers:
        return ClipDecision.reject("target_not_detected")
    if speakers != {target_speaker}:
        return ClipDecision.reject("other_speaker_detected")
    return ClipDecision.accept("target_only")
