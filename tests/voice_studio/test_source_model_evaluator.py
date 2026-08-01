"""Public contract tests for source-checkpoint evaluation."""

import ast
from pathlib import Path

import pytest

from voiceclonemlx.training.source_model_evaluator import (
    EvaluationMeasurement,
    EvaluationPreview,
    EvaluationResult,
    EvaluationThreshold,
    SourceEvaluationRequest,
    InvalidSourceEvaluation,
    SourceEvaluationError,
    evaluate_source,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def request(**overrides: object) -> SourceEvaluationRequest:
    fields = {
        "evaluation_id": "evaluation-001",
        "training_run_id": "run-001",
        "training_manifest_sha256": SHA_A,
        "dataset_manifest_id": "dataset-001",
        "dataset_sha256": SHA_B,
        "backend_id": "qwen3-tts",
        "backend_revision": "backend-revision",
        "base_model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "base_model_revision": "base-revision",
        "base_model_sha256": SHA_C,
        "training_config_sha256": SHA_D,
        "checkpoint_id": "checkpoint-001",
        "checkpoint_path": "checkpoints/checkpoint-001",
        "checkpoint_artifact_kind": "fine_tuned_adapter",
        "checkpoint_declared_sha256": SHA_E,
        "checkpoint_verified_sha256": SHA_E,
        "acceptance_manifest_id": "acceptance-001",
        "acceptance_manifest_sha256": SHA_F,
        "acceptance_item_ids": ("prompt-001",),
        "evaluation_config_sha256": SHA_A,
        "thresholds": (EvaluationThreshold("intelligibility", 0.8, True),),
    }
    fields.update(overrides)
    return SourceEvaluationRequest(**fields)  # type: ignore[arg-type]


class PassingEvaluator:
    def evaluate(self, source_request: SourceEvaluationRequest) -> EvaluationResult:
        assert source_request == request()
        return EvaluationResult(
            evaluator_id="official-source-runtime",
            evaluator_revision="revision-123",
            measurements=(EvaluationMeasurement("intelligibility", 0.9),),
            previews=(EvaluationPreview("prompt-001", "previews/001.wav", SHA_B),),
        )


def test_valid_evidence_is_scored_once_without_automatic_acceptance():
    report = evaluate_source(request(), PassingEvaluator())

    assert report.decision == "pending_human_review"
    assert report.mandatory_thresholds_passed is True
    assert report.checkpoint_sha256 == SHA_E
    assert report.acceptance_manifest_sha256 == SHA_F
    assert report.previews[0].acceptance_item_id == "prompt-001"


class MustNotEvaluate:
    def evaluate(self, source_request):
        raise AssertionError("evaluator must not be called")


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"checkpoint_verified_sha256": SHA_D}, "checkpoint checksum"),
        ({"checkpoint_artifact_kind": "reference_clone"}, "learned"),
        ({"acceptance_item_ids": ()}, "acceptance item"),
        ({"training_manifest_sha256": "bad"}, "SHA-256"),
    ),
)
def test_invalid_provenance_is_refused_before_evaluator(overrides, message):
    with pytest.raises(InvalidSourceEvaluation, match=message):
        evaluate_source(request(**overrides), MustNotEvaluate())


def test_non_finite_measurement_is_refused():
    class InvalidEvaluator:
        def evaluate(self, source_request):
            return EvaluationResult(
                evaluator_id="source-runtime",
                evaluator_revision="revision-123",
                measurements=(EvaluationMeasurement("intelligibility", float("nan")),),
                previews=(EvaluationPreview("prompt-001", "previews/001.wav", SHA_B),),
            )

    with pytest.raises(InvalidSourceEvaluation, match="finite"):
        evaluate_source(request(), InvalidEvaluator())


def test_non_numeric_threshold_is_refused_as_invalid_evidence():
    with pytest.raises(InvalidSourceEvaluation, match="threshold"):
        evaluate_source(
            request(thresholds=(EvaluationThreshold("intelligibility", "high", True),)),  # type: ignore[arg-type]
            MustNotEvaluate(),
        )


def test_failed_threshold_remains_pending_and_cannot_promote_itself():
    class BelowThresholdEvaluator:
        def evaluate(self, source_request):
            return EvaluationResult(
                evaluator_id="source-runtime",
                evaluator_revision="revision-123",
                measurements=(EvaluationMeasurement("intelligibility", 0.7),),
                previews=(EvaluationPreview("prompt-001", "previews/001.wav", SHA_B),),
            )

    report = evaluate_source(request(), BelowThresholdEvaluator())

    assert report.mandatory_thresholds_passed is False
    assert report.decision == "pending_human_review"


def test_evaluator_failure_has_one_attempt_and_preserves_cause():
    class FailingEvaluator:
        calls = 0

        def evaluate(self, source_request):
            self.calls += 1
            raise RuntimeError("source runtime stopped")

    evaluator = FailingEvaluator()
    with pytest.raises(SourceEvaluationError, match="source runtime stopped") as caught:
        evaluate_source(request(), evaluator)

    assert evaluator.calls == 1
    assert caught.value.__cause__.__class__ is RuntimeError


def test_every_frozen_acceptance_item_requires_one_checksum_bound_preview():
    class MissingPreviewEvaluator:
        def evaluate(self, source_request):
            return EvaluationResult(
                evaluator_id="source-runtime",
                evaluator_revision="revision-123",
                measurements=(EvaluationMeasurement("intelligibility", 0.9),),
                previews=(),
            )

    with pytest.raises(InvalidSourceEvaluation, match="every acceptance item"):
        evaluate_source(request(), MissingPreviewEvaluator())


def test_module_has_no_filesystem_model_audio_gpu_or_downstream_dependency():
    import voiceclonemlx.training.source_model_evaluator as module

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
    assert imported <= {"__future__", "dataclasses", "math", "typing"}
    assert not any(
        word in source
        for word in (
            "runtime_variant_converter",
            "bundle_publisher",
            "source_model_promoter",
            "subprocess",
            "requests",
        )
    )
