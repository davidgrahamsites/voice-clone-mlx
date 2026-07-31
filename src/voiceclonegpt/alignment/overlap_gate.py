"""Decide whether a diarized candidate clip is safe for dataset admission."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    """A provider-neutral diarization span."""

    start_s: float
    end_s: float
    speaker_id: str
    overlap: bool = False


@dataclass(frozen=True)
class ClipDecision:
    """The only decisions the dataset builder needs from this gate."""

    status: str
    reason: str

    @classmethod
    def accept(cls, reason: str) -> "ClipDecision":
        return cls("accept", reason)

    @classmethod
    def reject(cls, reason: str) -> "ClipDecision":
        return cls("reject", reason)


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
    """

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
