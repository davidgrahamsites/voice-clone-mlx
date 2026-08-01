"""Compose reviewed free-speech evidence into the existing dataset contract.

This application-layer module coordinates existing domain seams.  It owns no
speaker, transcription, alignment, split, clip-measurement, or dataset policy.
All runtime/file boundaries are injected so importing the module performs no
work and deleting it leaves every producer and consumer importable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from voiceclonemlx.alignment.alignment_rows import accept_row, build_alignment_row
from voiceclonemlx.alignment.free_speech_plan import (
    FreeSpeechCandidate,
    plan_free_speech,
)
from voiceclonemlx.alignment.transcription_manifest import parse_transcription_jsonl
from voiceclonemlx.alignment.whisper_runner import run_whisper_plan
from voiceclonemlx.dataset.clip_probe import ClipMeasurement
from voiceclonemlx.dataset.dataset_rows import build_dataset_row, rows_to_json
from voiceclonemlx.dataset.split_plan import SplitPlan, split_for
from voiceclonemlx.shared.styles import CANONICAL_STYLES


class DatasetPipelineError(ValueError):
    """The supplied evidence cannot safely form a dataset."""


@dataclass(frozen=True)
class RuntimeCandidate:
    """One stable candidate identity around the existing planner input."""

    candidate_id: str
    style: str
    clip_path: str
    candidate: FreeSpeechCandidate


@dataclass(frozen=True)
class SpeakerRuntimeEvidence:
    """Provider-neutral speaker evidence bound to one recorded master."""

    session_id: str
    master_audio: str
    master_sha256: str
    runtime_id: str
    runtime_version: str
    target_speaker: str
    candidates: tuple[RuntimeCandidate, ...]


@dataclass(frozen=True)
class ReviewCandidate:
    """An owner-only transcript awaiting an attributed human decision."""

    candidate_id: str
    style: str
    clip_path: str
    observed_text: str
    source_start_s: float
    source_end_s: float
    start_s: float
    end_s: float
    segment_ids: tuple[str, ...]
    clip_decision: Any


@dataclass(frozen=True)
class RejectedCandidate:
    """A candidate retained with the stage and reason that refused it."""

    candidate_id: str
    clip_start_s: float
    clip_end_s: float
    stage: str
    reason: str


@dataclass(frozen=True)
class DatasetReviewBundle:
    """Prepared evidence that cannot enter a dataset without human review."""

    session_id: str
    master_audio: str
    master_sha256: str
    review_candidates: tuple[ReviewCandidate, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]


@dataclass(frozen=True)
class HumanReview:
    """One attributed decision made outside the composer."""

    accepted: bool
    actor: str
    at: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise DatasetPipelineError("human review accepted must be a bool")
        _text(self.actor, "human review actor")
        _text(self.at, "human review time")
        if self.accepted and self.reason is not None:
            raise DatasetPipelineError("an accepted human review must not give a rejection reason")
        if not self.accepted:
            _text(self.reason, "human review rejection reason")


@dataclass(frozen=True)
class DatasetComposition:
    """The accepted dataset document plus every retained rejection."""

    dataset_json: str
    rejected_candidates: tuple[RejectedCandidate, ...]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetPipelineError(f"{field} must be non-blank text")
    return value


def _relative_path(value: object, field: str) -> str:
    path = _text(value, field)
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or "\\" in path
        or "://" in path
        or ".." in parsed.parts
        or path in {".", ".."}
    ):
        raise DatasetPipelineError(f"{field} must be a portable relative path")
    return path


def _sha256(value: object, field: str) -> str:
    digest = _text(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DatasetPipelineError(f"{field} must be a lowercase SHA-256 checksum")
    return digest


def _session_values(session_manifest: Mapping[str, object]) -> tuple[str, str, str]:
    if not isinstance(session_manifest, Mapping):
        raise DatasetPipelineError("session_manifest must be a mapping")
    session_id = _text(session_manifest.get("session_id"), "session_id")
    if session_manifest.get("schema_version") != "1.0.0":
        raise DatasetPipelineError("recording session schema_version must be '1.0.0'")
    master_audio = _relative_path(session_manifest.get("master_audio"), "master_audio")
    if session_manifest.get("status") != "captured":
        raise DatasetPipelineError("recording session must have status 'captured'")
    masters = session_manifest.get("masters")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise DatasetPipelineError("recording session masters must be a sequence")
    matching = [item for item in masters if isinstance(item, Mapping) and item.get("path") == master_audio]
    if len(matching) != 1:
        raise DatasetPipelineError("recording master_audio must name exactly one master")
    master_sha256 = _sha256(matching[0].get("sha256"), "master_sha256")
    return session_id, master_audio, master_sha256


def prepare_free_speech_dataset(
    session_manifest: Mapping[str, object],
    speaker_evidence: SpeakerRuntimeEvidence,
    *,
    model_path: Any,
    output_dir: Any,
    transcriber_version: str,
    planner: Callable[..., Sequence[Any]] = plan_free_speech,
    runner: Callable[..., str] = run_whisper_plan,
) -> DatasetReviewBundle:
    """Plan and transcribe only owner-only candidates for human review."""
    session_id, master_audio, master_sha256 = _session_values(session_manifest)
    if not isinstance(speaker_evidence, SpeakerRuntimeEvidence):
        raise DatasetPipelineError("speaker_evidence must be SpeakerRuntimeEvidence")
    if (
        speaker_evidence.session_id != session_id
        or speaker_evidence.master_audio != master_audio
        or speaker_evidence.master_sha256 != master_sha256
    ):
        raise DatasetPipelineError("speaker evidence does not match recording provenance")
    _text(speaker_evidence.runtime_id, "runtime_id")
    _text(speaker_evidence.runtime_version, "runtime_version")
    _text(speaker_evidence.target_speaker, "target_speaker")
    candidate_ids = []
    for item in speaker_evidence.candidates:
        if not isinstance(item, RuntimeCandidate):
            raise DatasetPipelineError("speaker evidence candidates must be RuntimeCandidate")
        candidate_ids.append(_text(item.candidate_id, "candidate_id"))
        if item.style not in CANONICAL_STYLES:
            raise DatasetPipelineError("candidate style must be canonical")
        _relative_path(item.clip_path, "candidate clip_path")
        if not isinstance(item.candidate, FreeSpeechCandidate):
            raise DatasetPipelineError("runtime candidate must carry FreeSpeechCandidate")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DatasetPipelineError("candidate_id values must be unique")

    planned = tuple(
        planner(
            tuple(item.candidate for item in speaker_evidence.candidates),
            target_speaker=speaker_evidence.target_speaker,
            model_path=model_path,
            output_dir=output_dir,
        )
    )
    if len(planned) != len(speaker_evidence.candidates):
        raise DatasetPipelineError("planner must return one result per candidate")

    review_candidates = []
    rejected_candidates = []
    for source, result in zip(speaker_evidence.candidates, planned):
        if not result.decision.accepted:
            rejected_candidates.append(
                RejectedCandidate(
                    candidate_id=source.candidate_id,
                    clip_start_s=source.candidate.clip_start_s,
                    clip_end_s=source.candidate.clip_end_s,
                    stage="speaker_gate",
                    reason=result.decision.reason,
                )
            )
            continue
        if result.whisper_plan is None:
            raise DatasetPipelineError("accepted candidate has no Whisper plan")
        transcript = parse_transcription_jsonl(
            runner(
                result.whisper_plan,
                master_audio=source.clip_path,
                transcriber_version=transcriber_version,
            )
        )
        if not transcript:
            raise DatasetPipelineError("accepted candidate produced no transcript rows")
        if any(row.master_audio != source.clip_path for row in transcript):
            raise DatasetPipelineError("transcript does not match candidate clip provenance")
        review_candidates.append(
            ReviewCandidate(
                candidate_id=source.candidate_id,
                style=source.style,
                clip_path=source.clip_path,
                observed_text="".join(row.text for row in transcript),
                source_start_s=source.candidate.clip_start_s,
                source_end_s=source.candidate.clip_end_s,
                start_s=transcript[0].start_s,
                end_s=transcript[-1].end_s,
                segment_ids=tuple(row.segment_id for row in transcript),
                clip_decision=result.decision,
            )
        )

    return DatasetReviewBundle(
        session_id=session_id,
        master_audio=master_audio,
        master_sha256=master_sha256,
        review_candidates=tuple(review_candidates),
        rejected_candidates=tuple(rejected_candidates),
    )


def finalize_free_speech_dataset(
    bundle: DatasetReviewBundle,
    *,
    reviews: Mapping[str, HumanReview],
    measurements: Mapping[str, ClipMeasurement],
    split_plan: SplitPlan,
    row_builder: Callable[..., Any] = build_dataset_row,
    serializer: Callable[[Sequence[Any]], str] = rows_to_json,
) -> DatasetComposition:
    """Admit only measured candidates carrying explicit human acceptance."""
    if not isinstance(bundle, DatasetReviewBundle):
        raise DatasetPipelineError("bundle must be a DatasetReviewBundle")
    if not isinstance(reviews, Mapping):
        raise DatasetPipelineError("reviews must be a mapping")
    if not isinstance(measurements, Mapping):
        raise DatasetPipelineError("measurements must be a mapping")

    expected_ids = {candidate.candidate_id for candidate in bundle.review_candidates}
    if set(reviews) != expected_ids:
        raise DatasetPipelineError("every review candidate requires exactly one human review")
    for review in reviews.values():
        if not isinstance(review, HumanReview):
            raise DatasetPipelineError("reviews must contain HumanReview values")

    accepted_ids = {
        candidate_id for candidate_id, review in reviews.items() if review.accepted
    }
    if set(measurements) != accepted_ids:
        raise DatasetPipelineError("every accepted candidate requires exactly one measurement")
    for measurement in measurements.values():
        if not isinstance(measurement, ClipMeasurement):
            raise DatasetPipelineError("measurements must contain ClipMeasurement values")

    # Resolve the session split before constructing any row. An unplanned
    # session is a provenance failure, not an excuse to default to training.
    split = split_for(split_plan, bundle.session_id)
    rows = []
    rejected = list(bundle.rejected_candidates)
    for candidate in bundle.review_candidates:
        review = reviews[candidate.candidate_id]
        if not review.accepted:
            rejected.append(
                RejectedCandidate(
                    candidate_id=candidate.candidate_id,
                    clip_start_s=candidate.source_start_s,
                    clip_end_s=candidate.source_end_s,
                    stage="human_review",
                    reason=review.reason or "human_rejected",
                )
            )
            continue

        pending = build_alignment_row(
            utterance_id=candidate.candidate_id,
            expected_text=candidate.observed_text,
            observed_text=candidate.observed_text,
            style=candidate.style,
            start_s=candidate.start_s,
            end_s=candidate.end_s,
            segment_ids=candidate.segment_ids,
            master_audio=candidate.clip_path,
        )
        accepted = accept_row(pending, actor=review.actor, at=review.at)
        measurement = measurements[candidate.candidate_id]
        rows.append(
            row_builder(
                accepted_row=accepted,
                clip_path=candidate.clip_path,
                clip_sha256=measurement.clip_sha256,
                audio_properties=measurement.audio_properties,
                session_id=bundle.session_id,
                split=split,
                clip_decision=candidate.clip_decision,
            )
        )

    return DatasetComposition(
        dataset_json=serializer(rows),
        rejected_candidates=tuple(rejected),
    )
