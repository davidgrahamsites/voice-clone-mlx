"""Coordinate one training run without knowing or calling a provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

TRAINED_ARTIFACT_KINDS = frozenset(("fine_tuned_full", "fine_tuned_adapter"))
_LOWER_HEX = frozenset("0123456789abcdef")


class InvalidTrainingRequest(ValueError):
    """The run cannot safely cross the provider boundary."""


class InvalidTrainingResult(ValueError):
    """The provider did not return a usable learned artifact."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidTrainingRequest(f"{field} must be non-blank text.")


@dataclass(frozen=True)
class DatasetEvidence:
    """The reviewed dataset identity and independently observed checksum."""

    manifest_id: str
    declared_sha256: str
    verified_sha256: str
    review_state: str


@dataclass(frozen=True)
class TrainedCheckpoint:
    """One immutable learned artifact returned by a training provider."""

    checkpoint_id: str
    artifact_path: str
    artifact_sha256: str
    artifact_kind: str


@dataclass(frozen=True)
class TrainingRequest:
    """Provider-neutral inputs for one bounded training run."""

    run_id: str
    dataset: DatasetEvidence
    backend_id: str
    backend_version: str
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    training_config_sha256: str
    artifact_kind: str
    checkpoints: tuple[TrainedCheckpoint, ...] = ()


@dataclass(frozen=True)
class TrainingRunManifest:
    """The durable outcome a caller can serialize into its run folder."""

    run_id: str
    dataset_manifest_id: str
    dataset_sha256: str
    backend_id: str
    backend_version: str
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    training_config_sha256: str
    artifact_kind: str
    checkpoints: tuple[TrainedCheckpoint, ...]
    status: str
    failure_reason: str | None = None


class TrainingRunError(RuntimeError):
    """One provider attempt failed; its resumable manifest remains available."""

    def __init__(self, message: str, manifest: TrainingRunManifest):
        super().__init__(message)
        self.manifest = manifest


class TrainingProvider(Protocol):
    """External boundary implemented by one explicitly selected provider."""

    def train(
        self,
        request: TrainingRequest,
        *,
        resume_from: TrainedCheckpoint | None,
    ) -> TrainedCheckpoint: ...


def _manifest_for(
    request: TrainingRequest,
    *,
    checkpoints: tuple[TrainedCheckpoint, ...],
    status: str,
    failure_reason: str | None = None,
) -> TrainingRunManifest:
    return TrainingRunManifest(
        run_id=request.run_id,
        dataset_manifest_id=request.dataset.manifest_id,
        dataset_sha256=request.dataset.verified_sha256,
        backend_id=request.backend_id,
        backend_version=request.backend_version,
        base_model_id=request.base_model_id,
        base_model_revision=request.base_model_revision,
        base_model_sha256=request.base_model_sha256,
        training_config_sha256=request.training_config_sha256,
        artifact_kind=request.artifact_kind,
        checkpoints=checkpoints,
        status=status,
        failure_reason=failure_reason,
    )


def execute_training(
    request: TrainingRequest,
    trainer: TrainingProvider,
) -> TrainingRunManifest:
    """Run one provider attempt and return its completed manifest."""
    for field in (
        "run_id",
        "backend_id",
        "backend_version",
        "base_model_id",
        "base_model_revision",
    ):
        _require_text(getattr(request, field), field)
    if request.dataset.review_state != "accepted":
        raise InvalidTrainingRequest("The dataset must be accepted before training.")
    if request.dataset.declared_sha256 != request.dataset.verified_sha256:
        raise InvalidTrainingRequest("The dataset checksum does not match verified bytes.")
    checksums = (
        request.dataset.declared_sha256,
        request.dataset.verified_sha256,
        request.base_model_sha256,
        request.training_config_sha256,
    )
    if not all(_is_sha256(value) for value in checksums):
        raise InvalidTrainingRequest("Every input checksum must be lowercase SHA-256.")
    if request.artifact_kind not in TRAINED_ARTIFACT_KINDS:
        raise InvalidTrainingRequest(
            "artifact_kind must be fine_tuned_full or fine_tuned_adapter."
        )
    try:
        checkpoint = trainer.train(
            request,
            resume_from=request.checkpoints[-1] if request.checkpoints else None,
        )
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        failed = _manifest_for(
            request,
            checkpoints=request.checkpoints,
            status="failed",
            failure_reason=reason,
        )
        raise TrainingRunError(f"Training provider failed: {reason}", failed) from exc
    if not isinstance(checkpoint, TrainedCheckpoint):
        raise InvalidTrainingResult("The provider must return a TrainedCheckpoint.")
    if (
        not isinstance(checkpoint.checkpoint_id, str)
        or not checkpoint.checkpoint_id.strip()
    ):
        raise InvalidTrainingResult("The trained checkpoint id is missing.")
    if (
        not isinstance(checkpoint.artifact_path, str)
        or not checkpoint.artifact_path.strip()
    ):
        raise InvalidTrainingResult("The trained checkpoint artifact path is missing.")
    if checkpoint.artifact_kind != request.artifact_kind:
        raise InvalidTrainingResult(
            "The trained checkpoint artifact kind does not match the request."
        )
    if not _is_sha256(checkpoint.artifact_sha256):
        raise InvalidTrainingResult("The trained checkpoint checksum is not valid SHA-256.")
    if any(
        item.checkpoint_id == checkpoint.checkpoint_id for item in request.checkpoints
    ):
        raise InvalidTrainingResult(
            f"Checkpoint id {checkpoint.checkpoint_id!r} already exists; "
            "checkpoints are append-only."
        )
    return _manifest_for(
        request,
        checkpoints=request.checkpoints + (checkpoint,),
        status="completed",
    )
