"""Run injected local speaker backends and emit overlap-gate evidence.

This module owns orchestration and validation, not a model implementation.
Callers inject one local diarizer and one local speaker verifier. The runtime
calls each at most once, never retries or falls back, imports no backend, and
does not read audio itself. Existing local files are resolved before either
backend runs so a provider cannot receive a remote locator by accident.

Anonymous diarizer labels become the enrolled ``owner_id`` only when the
verifier's cosine-similarity score meets the caller's calibrated threshold.
All other labels remain visibly non-owner. Provider overlap flags are kept,
and intersecting time ranges are also marked as overlap before the immutable
``SpeakerTurn`` evidence reaches the existing admission gate.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, Tuple

from .overlap_gate import SpeakerTurn

NON_OWNER_PREFIX = "non_owner:"
MAX_DURATION_S = 600.0
MAX_ENROLLMENT_FILES = 16
MAX_TIMEOUT_S = 300.0
MAX_TURNS = 512


class SpeakerRuntimeError(ValueError):
    """The requested local speaker analysis cannot safely run."""


def _local_file(value, field: str) -> Path:
    """Resolve one existing local file without opening it."""
    if isinstance(value, str) and "://" in value:
        raise SpeakerRuntimeError(f"{field} must be a local path")
    if isinstance(value, bool):
        raise SpeakerRuntimeError(f"{field} must be a path")
    try:
        path = Path(value).resolve(strict=True)
    except (TypeError, OSError, RuntimeError) as exc:
        raise SpeakerRuntimeError(f"{field} must be an existing local file") from exc
    if not path.is_file():
        raise SpeakerRuntimeError(f"{field} must be an existing local file")
    return path


@dataclass(frozen=True)
class RuntimeProvenance:
    """Caller-declared identity of the local models and enrollment set."""

    diarizer: str
    diarizer_version: str
    verifier: str
    verifier_version: str
    enrollment_id: str

    def __post_init__(self) -> None:
        for field in (
            "diarizer",
            "diarizer_version",
            "verifier",
            "verifier_version",
            "enrollment_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise SpeakerRuntimeError(f"{field} must be a non-blank string")


@dataclass(frozen=True)
class SpeakerScore:
    """One anonymous diarized speaker's similarity to the enrollment."""

    speaker_id: str
    owner_similarity: float


@dataclass(frozen=True)
class SpeakerEvidence:
    """Ordered, auditable owner evidence consumable by the existing gate."""

    audio_path: str
    enrollment_paths: Tuple[str, ...]
    provenance: RuntimeProvenance
    owner_id: str
    owner_threshold: float
    duration_s: float
    scores: Tuple[SpeakerScore, ...]
    turns: Tuple[SpeakerTurn, ...]


def run_speaker_runtime(
    audio_path,
    enrollment_paths,
    *,
    owner_id: str,
    owner_threshold: float,
    duration_s: float,
    provenance: RuntimeProvenance,
    diarizer: Callable,
    verifier: Callable,
    timeout_s: float = 120.0,
) -> SpeakerEvidence:
    """Run one local diarizer and verifier, then label owner evidence.

    ``diarizer`` receives ``(absolute_audio_path, timeout_s=...)`` and returns
    a bounded sequence of provider-neutral ``SpeakerTurn`` values carrying
    anonymous speaker ids. ``verifier`` receives ``(absolute_audio_path,
    absolute_enrollment_paths, turns, timeout_s=...)`` and returns exactly one
    cosine-similarity score in ``[-1, 1]`` per anonymous speaker id.

    The injected backends own actual audio/model work and must honor the
    supplied timeout. This seam calls each backend once, serially, with no
    retry, fallback, source separation, file writes, or network operation.

    Raises:
        SpeakerRuntimeError: An input, backend result, or resource bound is
            unusable. Backend exceptions propagate unchanged and are not
            retried.
    """
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(duration_s)
        or duration_s <= 0
        or duration_s > MAX_DURATION_S
    ):
        raise SpeakerRuntimeError(
            f"duration_s must be positive and at most {MAX_DURATION_S}: {duration_s!r}"
        )
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
        or timeout_s > MAX_TIMEOUT_S
    ):
        raise SpeakerRuntimeError(
            f"timeout_s must be positive and at most {MAX_TIMEOUT_S}: {timeout_s!r}"
        )
    if (
        isinstance(owner_threshold, bool)
        or not isinstance(owner_threshold, (int, float))
        or not math.isfinite(owner_threshold)
        or not -1.0 <= owner_threshold <= 1.0
    ):
        raise SpeakerRuntimeError(
            f"owner_threshold must be finite and between -1 and 1: {owner_threshold!r}"
        )
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise SpeakerRuntimeError("owner_id must be a non-blank string")
    if not isinstance(provenance, RuntimeProvenance):
        raise SpeakerRuntimeError("provenance must be RuntimeProvenance")
    if (
        isinstance(enrollment_paths, (str, bytes))
        or not isinstance(enrollment_paths, Sequence)
        or not enrollment_paths
        or len(enrollment_paths) > MAX_ENROLLMENT_FILES
    ):
        raise SpeakerRuntimeError(
            f"enrollment_paths must contain 1 to {MAX_ENROLLMENT_FILES} local files"
        )
    audio = _local_file(audio_path, "audio_path")
    enrollment = tuple(
        str(_local_file(path, "enrollment_path")) for path in enrollment_paths
    )
    raw_turns: Sequence[SpeakerTurn] = diarizer(str(audio), timeout_s=timeout_s)
    if isinstance(raw_turns, (str, bytes)) or not isinstance(raw_turns, Sequence):
        raise SpeakerRuntimeError("diarizer turns must be a sequence")
    if len(raw_turns) > MAX_TURNS:
        raise SpeakerRuntimeError(f"diarizer turns exceed the {MAX_TURNS} turn cap")
    raw_turns = tuple(raw_turns)
    if any(not isinstance(turn, SpeakerTurn) for turn in raw_turns):
        raise SpeakerRuntimeError("diarizer turns must contain SpeakerTurn values")
    if any(turn.end_s > duration_s for turn in raw_turns):
        raise SpeakerRuntimeError("diarizer turn exceeds the declared audio duration")
    raw_scores: Mapping[str, float] = verifier(
        str(audio), enrollment, tuple(raw_turns), timeout_s=timeout_s
    )
    speaker_ids = {turn.speaker_id for turn in raw_turns}
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != speaker_ids:
        raise SpeakerRuntimeError(
            "verifier scores must contain exactly the diarized speaker ids"
        )
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not -1.0 <= score <= 1.0
        for score in raw_scores.values()
    ):
        raise SpeakerRuntimeError(
            "verifier scores must be finite numbers between -1 and 1"
        )
    labelled = tuple(
        SpeakerTurn(
            turn.start_s,
            turn.end_s,
            owner_id
            if raw_scores[turn.speaker_id] >= owner_threshold
            else f"{NON_OWNER_PREFIX}{turn.speaker_id}",
            turn.overlap,
        )
        for turn in raw_turns
    )
    ordered = tuple(sorted(labelled, key=lambda turn: (turn.start_s, turn.end_s)))
    turns = tuple(
        SpeakerTurn(
            turn.start_s,
            turn.end_s,
            turn.speaker_id,
            turn.overlap
            or any(
                other is not turn
                and other.end_s > turn.start_s
                and other.start_s < turn.end_s
                for other in ordered
            ),
        )
        for turn in ordered
    )
    return SpeakerEvidence(
        audio_path=str(audio),
        enrollment_paths=enrollment,
        provenance=provenance,
        owner_id=owner_id,
        owner_threshold=float(owner_threshold),
        duration_s=float(duration_s),
        scores=tuple(
            SpeakerScore(speaker_id, float(raw_scores[speaker_id]))
            for speaker_id in sorted(speaker_ids)
        ),
        turns=turns,
    )
