"""Build immutable source-release metadata after explicit human approval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

_LEARNED_KINDS = frozenset(("fine_tuned_full", "fine_tuned_adapter"))
_LOWER_HEX = frozenset("0123456789abcdef")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class InvalidSourcePromotion(ValueError):
    """The checkpoint lacks exact evidence for explicit human promotion."""


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _require_utc(value: object, field: str) -> None:
    if not _is_text(value):
        raise InvalidSourcePromotion(f"{field} must be a named UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidSourcePromotion(f"{field} must be a valid UTC timestamp.") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise InvalidSourcePromotion(f"{field} must include the UTC offset.")


@dataclass(frozen=True)
class ReviewedPreview:
    acceptance_item_id: str
    preview_sha256: str
    listened: bool


@dataclass(frozen=True)
class HumanApproval:
    decision: str
    approver_name: str
    approved_at: str
    checkpoint_sha256: str
    acceptance_manifest_sha256: str
    evaluation_config_sha256: str
    evaluation_report_sha256: str


@dataclass(frozen=True)
class PromotionRequest:
    source_release_id: str
    voice_id: str
    model_version: str
    created_at: str
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
    checkpoint_format: str
    checkpoint_revision: str
    checkpoint_declared_sha256: str
    checkpoint_verified_sha256: str
    evaluation_id: str
    evaluation_report_declared_sha256: str
    evaluation_report_verified_sha256: str
    acceptance_manifest_id: str
    acceptance_manifest_sha256: str
    evaluation_config_sha256: str
    mandatory_thresholds_passed: bool
    reviewed_previews: tuple[ReviewedPreview, ...]
    approval: HumanApproval
    license_layers_sha256: str
    reference_library_id: str
    reference_library_sha256: str
    existing_source_release_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceModelRelease:
    source_release_id: str
    voice_id: str
    model_version: str
    created_at: str
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
    copy_from_checkpoint_path: str
    checkpoint_artifact_kind: str
    checkpoint_format: str
    checkpoint_revision: str
    checkpoint_sha256: str
    evaluation_id: str
    evaluation_report_sha256: str
    acceptance_manifest_id: str
    acceptance_manifest_sha256: str
    evaluation_config_sha256: str
    reviewed_previews: tuple[ReviewedPreview, ...]
    approval: HumanApproval
    license_layers_sha256: str
    reference_library_id: str
    reference_library_sha256: str


def build_source_release(request: PromotionRequest) -> SourceModelRelease:
    """Return release metadata only; the caller owns copying and persistence."""
    for field in (
        "source_release_id",
        "voice_id",
        "training_run_id",
        "dataset_manifest_id",
        "backend_id",
        "backend_revision",
        "base_model_id",
        "base_model_revision",
        "checkpoint_id",
        "checkpoint_path",
        "checkpoint_format",
        "checkpoint_revision",
        "evaluation_id",
        "acceptance_manifest_id",
        "reference_library_id",
    ):
        if not _is_text(getattr(request, field)):
            raise InvalidSourcePromotion(f"{field} must be non-blank text.")
    if not _is_text(request.model_version) or not _SEMVER.fullmatch(request.model_version):
        raise InvalidSourcePromotion("model_version must be MAJOR.MINOR.PATCH.")
    _require_utc(request.created_at, "created_at")
    checksums = (
        request.training_manifest_sha256,
        request.dataset_sha256,
        request.base_model_sha256,
        request.training_config_sha256,
        request.checkpoint_declared_sha256,
        request.checkpoint_verified_sha256,
        request.evaluation_report_declared_sha256,
        request.evaluation_report_verified_sha256,
        request.acceptance_manifest_sha256,
        request.evaluation_config_sha256,
        request.license_layers_sha256,
        request.reference_library_sha256,
    )
    if not all(_is_sha256(value) for value in checksums):
        raise InvalidSourcePromotion("Every provenance value must be lowercase SHA-256.")
    if request.checkpoint_declared_sha256 != request.checkpoint_verified_sha256:
        raise InvalidSourcePromotion("The checkpoint checksum does not match verified bytes.")
    if (
        request.evaluation_report_declared_sha256
        != request.evaluation_report_verified_sha256
    ):
        raise InvalidSourcePromotion(
            "The evaluation report checksum does not match verified bytes."
        )
    if request.checkpoint_artifact_kind not in _LEARNED_KINDS:
        raise InvalidSourcePromotion("Only a learned checkpoint may be promoted.")
    if not request.mandatory_thresholds_passed:
        raise InvalidSourcePromotion("Every mandatory evaluation threshold must pass.")
    if not request.reviewed_previews:
        raise InvalidSourcePromotion("At least one reviewed preview is required.")
    preview_ids = [item.acceptance_item_id for item in request.reviewed_previews]
    if (
        len(preview_ids) != len(set(preview_ids))
        or any(not _is_text(item.acceptance_item_id) for item in request.reviewed_previews)
        or any(not _is_sha256(item.preview_sha256) for item in request.reviewed_previews)
    ):
        raise InvalidSourcePromotion("Reviewed previews need unique ids and SHA-256 values.")
    if any(item.listened is not True for item in request.reviewed_previews):
        raise InvalidSourcePromotion("A human must listen to every preview.")
    if request.source_release_id in request.existing_source_release_ids:
        raise InvalidSourcePromotion("The source release already exists and cannot be overwritten.")
    approval = request.approval
    if approval.decision != "accepted":
        raise InvalidSourcePromotion("Human approval must explicitly be accepted.")
    if not _is_text(approval.approver_name):
        raise InvalidSourcePromotion("A named human approver is required.")
    _require_utc(approval.approved_at, "approval")
    bindings = (
        ("checkpoint", approval.checkpoint_sha256, request.checkpoint_verified_sha256),
        (
            "acceptance manifest",
            approval.acceptance_manifest_sha256,
            request.acceptance_manifest_sha256,
        ),
        (
            "evaluation configuration",
            approval.evaluation_config_sha256,
            request.evaluation_config_sha256,
        ),
        (
            "evaluation report",
            approval.evaluation_report_sha256,
            request.evaluation_report_verified_sha256,
        ),
    )
    for field, approved, actual in bindings:
        if not _is_sha256(approved) or approved != actual:
            raise InvalidSourcePromotion(f"Human approval does not bind the exact {field}.")
    return SourceModelRelease(
        source_release_id=request.source_release_id,
        voice_id=request.voice_id,
        model_version=request.model_version,
        created_at=request.created_at,
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
        copy_from_checkpoint_path=request.checkpoint_path,
        checkpoint_artifact_kind=request.checkpoint_artifact_kind,
        checkpoint_format=request.checkpoint_format,
        checkpoint_revision=request.checkpoint_revision,
        checkpoint_sha256=request.checkpoint_verified_sha256,
        evaluation_id=request.evaluation_id,
        evaluation_report_sha256=request.evaluation_report_verified_sha256,
        acceptance_manifest_id=request.acceptance_manifest_id,
        acceptance_manifest_sha256=request.acceptance_manifest_sha256,
        evaluation_config_sha256=request.evaluation_config_sha256,
        reviewed_previews=request.reviewed_previews,
        approval=request.approval,
        license_layers_sha256=request.license_layers_sha256,
        reference_library_id=request.reference_library_id,
        reference_library_sha256=request.reference_library_sha256,
    )
