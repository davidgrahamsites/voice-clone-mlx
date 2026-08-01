"""Evaluate one learned source checkpoint without promoting it."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

_LEARNED_KINDS = frozenset(("fine_tuned_full", "fine_tuned_adapter"))
_LOWER_HEX = frozenset("0123456789abcdef")


class InvalidSourceEvaluation(ValueError):
    """Evaluation evidence is incomplete, contradictory, or malformed."""


class SourceEvaluationError(RuntimeError):
    """The injected evaluator failed during its single attempt."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class EvaluationThreshold:
    name: str
    minimum: float
    mandatory: bool


@dataclass(frozen=True)
class EvaluationMeasurement:
    name: str
    value: float


@dataclass(frozen=True)
class EvaluationPreview:
    acceptance_item_id: str
    artifact_path: str
    artifact_sha256: str


@dataclass(frozen=True)
class SourceEvaluationRequest:
    evaluation_id: str
    training_run_id: str
    training_manifest_sha256: str
    dataset_manifest_id: str
    dataset_sha256: str
    backend_id: str
    backend_revision: str
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    training_config_sha256: str
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_artifact_kind: str
    checkpoint_declared_sha256: str
    checkpoint_verified_sha256: str
    acceptance_manifest_id: str
    acceptance_manifest_sha256: str
    acceptance_item_ids: tuple[str, ...]
    evaluation_config_sha256: str
    thresholds: tuple[EvaluationThreshold, ...]


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_id: str
    evaluator_revision: str
    measurements: tuple[EvaluationMeasurement, ...]
    previews: tuple[EvaluationPreview, ...]


@dataclass(frozen=True)
class SourceEvaluationReport:
    evaluation_id: str
    training_run_id: str
    training_manifest_sha256: str
    dataset_manifest_id: str
    dataset_sha256: str
    backend_id: str
    backend_revision: str
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    training_config_sha256: str
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_artifact_kind: str
    checkpoint_sha256: str
    acceptance_manifest_id: str
    acceptance_manifest_sha256: str
    evaluation_config_sha256: str
    evaluator_id: str
    evaluator_revision: str
    thresholds: tuple[EvaluationThreshold, ...]
    measurements: tuple[EvaluationMeasurement, ...]
    previews: tuple[EvaluationPreview, ...]
    mandatory_thresholds_passed: bool
    decision: str


class SourceEvaluator(Protocol):
    def evaluate(self, request: SourceEvaluationRequest) -> EvaluationResult: ...


def evaluate_source(
    request: SourceEvaluationRequest,
    evaluator: SourceEvaluator,
) -> SourceEvaluationReport:
    """Score one checkpoint and leave acceptance to a later human gate."""
    for field in (
        "evaluation_id",
        "training_run_id",
        "dataset_manifest_id",
        "backend_id",
        "backend_revision",
        "base_model_id",
        "base_model_revision",
        "checkpoint_id",
        "checkpoint_path",
        "acceptance_manifest_id",
    ):
        if not _is_text(getattr(request, field)):
            raise InvalidSourceEvaluation(f"{field} must be non-blank text.")
    checksums = (
        request.training_manifest_sha256,
        request.dataset_sha256,
        request.base_model_sha256,
        request.training_config_sha256,
        request.checkpoint_declared_sha256,
        request.checkpoint_verified_sha256,
        request.acceptance_manifest_sha256,
        request.evaluation_config_sha256,
    )
    if not all(_is_sha256(value) for value in checksums):
        raise InvalidSourceEvaluation("Every checksum must be lowercase SHA-256.")
    if request.checkpoint_declared_sha256 != request.checkpoint_verified_sha256:
        raise InvalidSourceEvaluation("The checkpoint checksum does not match verified bytes.")
    if request.checkpoint_artifact_kind not in _LEARNED_KINDS:
        raise InvalidSourceEvaluation("The checkpoint must be a learned artifact.")
    if not request.acceptance_item_ids or any(
        not _is_text(item) for item in request.acceptance_item_ids
    ):
        raise InvalidSourceEvaluation("At least one named acceptance item is required.")
    if len(set(request.acceptance_item_ids)) != len(request.acceptance_item_ids):
        raise InvalidSourceEvaluation("Acceptance item ids must be unique.")
    threshold_names = [item.name for item in request.thresholds]
    if (
        not request.thresholds
        or any(not _is_text(item.name) for item in request.thresholds)
        or any(not _is_finite_number(item.minimum) for item in request.thresholds)
        or len(set(threshold_names)) != len(threshold_names)
    ):
        raise InvalidSourceEvaluation("Evaluation thresholds must be named and unique.")
    try:
        result = evaluator.evaluate(request)
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        raise SourceEvaluationError(f"Source evaluator failed: {reason}") from exc
    if not isinstance(result, EvaluationResult):
        raise InvalidSourceEvaluation("The evaluator must return an EvaluationResult.")
    if not _is_text(result.evaluator_id) or not _is_text(result.evaluator_revision):
        raise InvalidSourceEvaluation("Evaluator identity and revision are required.")
    measurement_names = [item.name for item in result.measurements]
    if set(measurement_names) != set(threshold_names) or len(measurement_names) != len(
        set(measurement_names)
    ):
        raise InvalidSourceEvaluation("Measurements must match every threshold exactly.")
    if any(not _is_finite_number(item.value) for item in result.measurements):
        raise InvalidSourceEvaluation("Every measurement must be finite.")
    preview_ids = [item.acceptance_item_id for item in result.previews]
    if set(preview_ids) != set(request.acceptance_item_ids) or len(preview_ids) != len(
        set(preview_ids)
    ):
        raise InvalidSourceEvaluation("Previews must match every acceptance item exactly.")
    if any(
        not _is_text(item.artifact_path) or not _is_sha256(item.artifact_sha256)
        for item in result.previews
    ):
        raise InvalidSourceEvaluation("Every preview needs a path and lowercase SHA-256.")
    measured = {item.name: item.value for item in result.measurements}
    passed = all(
        not threshold.mandatory
        or measured.get(threshold.name, float("-inf")) >= threshold.minimum
        for threshold in request.thresholds
    )
    return SourceEvaluationReport(
        evaluation_id=request.evaluation_id,
        training_run_id=request.training_run_id,
        training_manifest_sha256=request.training_manifest_sha256,
        dataset_manifest_id=request.dataset_manifest_id,
        dataset_sha256=request.dataset_sha256,
        backend_id=request.backend_id,
        backend_revision=request.backend_revision,
        base_model_id=request.base_model_id,
        base_model_revision=request.base_model_revision,
        base_model_sha256=request.base_model_sha256,
        training_config_sha256=request.training_config_sha256,
        checkpoint_id=request.checkpoint_id,
        checkpoint_path=request.checkpoint_path,
        checkpoint_artifact_kind=request.checkpoint_artifact_kind,
        checkpoint_sha256=request.checkpoint_verified_sha256,
        acceptance_manifest_id=request.acceptance_manifest_id,
        acceptance_manifest_sha256=request.acceptance_manifest_sha256,
        evaluation_config_sha256=request.evaluation_config_sha256,
        evaluator_id=result.evaluator_id,
        evaluator_revision=result.evaluator_revision,
        thresholds=request.thresholds,
        measurements=result.measurements,
        previews=result.previews,
        mandatory_thresholds_passed=passed,
        decision="pending_human_review",
    )
