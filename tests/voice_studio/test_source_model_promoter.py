"""Public contract tests for pure, human-gated source promotion."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from voiceclonegpt.training.source_model_promoter import (
    HumanApproval,
    InvalidSourcePromotion,
    PromotionRequest,
    ReviewedPreview,
    build_source_release,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def approval(**overrides: object) -> HumanApproval:
    fields = {
        "decision": "accepted",
        "approver_name": "Local Owner",
        "approved_at": "2026-08-01T14:00:00+00:00",
        "checkpoint_sha256": SHA_E,
        "acceptance_manifest_sha256": SHA_F,
        "evaluation_config_sha256": SHA_A,
        "evaluation_report_sha256": SHA_B,
    }
    fields.update(overrides)
    return HumanApproval(**fields)  # type: ignore[arg-type]


def request(**overrides: object) -> PromotionRequest:
    fields = {
        "source_release_id": "voice-001@0.1.0-source",
        "voice_id": "voice-001",
        "model_version": "0.1.0",
        "created_at": "2026-08-01T14:01:00+00:00",
        "training_run_id": "run-001",
        "training_manifest_sha256": SHA_C,
        "dataset_manifest_id": "dataset-001",
        "dataset_sha256": SHA_D,
        "backend_id": "qwen3-tts",
        "backend_revision": "backend-revision",
        "base_model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "base_model_revision": "base-revision",
        "base_model_sha256": SHA_A,
        "training_config_sha256": SHA_C,
        "checkpoint_id": "checkpoint-001",
        "checkpoint_path": "checkpoints/checkpoint-001",
        "checkpoint_artifact_kind": "fine_tuned_adapter",
        "checkpoint_format": "safetensors-adapter",
        "checkpoint_revision": "checkpoint-revision",
        "checkpoint_declared_sha256": SHA_E,
        "checkpoint_verified_sha256": SHA_E,
        "evaluation_id": "evaluation-001",
        "evaluation_report_declared_sha256": SHA_B,
        "evaluation_report_verified_sha256": SHA_B,
        "acceptance_manifest_id": "acceptance-001",
        "acceptance_manifest_sha256": SHA_F,
        "evaluation_config_sha256": SHA_A,
        "mandatory_thresholds_passed": True,
        "reviewed_previews": (ReviewedPreview("prompt-001", SHA_D, True),),
        "approval": approval(),
        "license_layers_sha256": SHA_C,
        "reference_library_id": "references-001",
        "reference_library_sha256": SHA_F,
        "existing_source_release_ids": (),
    }
    fields.update(overrides)
    return PromotionRequest(**fields)  # type: ignore[arg-type]


def test_explicit_listening_approval_builds_immutable_source_release_metadata():
    release = build_source_release(request())

    assert release.source_release_id == "voice-001@0.1.0-source"
    assert release.checkpoint_sha256 == SHA_E
    assert release.evaluation_report_sha256 == SHA_B
    assert release.approval.approver_name == "Local Owner"
    assert release.copy_from_checkpoint_path == "checkpoints/checkpoint-001"
    with pytest.raises(FrozenInstanceError):
        release.model_version = "9.9.9"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"mandatory_thresholds_passed": False}, "threshold"),
        ({"reviewed_previews": ()}, "preview"),
        ({"reviewed_previews": (ReviewedPreview("prompt-001", SHA_D, False),)}, "listen"),
        ({"approval": approval(decision="rejected")}, "accepted"),
        ({"approval": approval(approver_name=" ")}, "approver"),
        ({"approval": approval(approved_at="2026-08-01T14:00:00")}, "UTC"),
        ({"approval": approval(checkpoint_sha256=SHA_D)}, "checkpoint"),
        ({"approval": approval(evaluation_report_sha256=SHA_C)}, "evaluation report"),
        ({"checkpoint_verified_sha256": SHA_D}, "checkpoint checksum"),
        ({"evaluation_report_verified_sha256": SHA_C}, "evaluation report checksum"),
        ({"checkpoint_artifact_kind": "reference_clone"}, "learned"),
        ({"existing_source_release_ids": ("voice-001@0.1.0-source",)}, "already exists"),
    ),
)
def test_promotion_fails_closed_without_exact_human_bound_evidence(overrides, message):
    with pytest.raises(InvalidSourcePromotion, match=message):
        build_source_release(request(**overrides))


def test_release_preserves_exact_training_evaluation_license_and_reference_lineage():
    release = build_source_release(request())

    assert release.training_manifest_sha256 == SHA_C
    assert release.dataset_sha256 == SHA_D
    assert release.base_model_revision == "base-revision"
    assert release.base_model_sha256 == SHA_A
    assert release.training_config_sha256 == SHA_C
    assert release.acceptance_manifest_sha256 == SHA_F
    assert release.evaluation_config_sha256 == SHA_A
    assert release.license_layers_sha256 == SHA_C
    assert release.reference_library_sha256 == SHA_F


def test_module_is_pure_and_has_no_evaluation_conversion_or_publication_dependency():
    import voiceclonegpt.training.source_model_promoter as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = {
        name
        for node in ast.walk(ast.parse(source))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert imported <= {"__future__", "dataclasses", "datetime", "re"}
    assert not any(
        word in source
        for word in (
            "source_model_evaluator",
            "runtime_variant_converter",
            "bundle_publisher",
            "pathlib",
            "subprocess",
            "requests",
        )
    )
