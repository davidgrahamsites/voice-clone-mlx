"""Public contract tests for one provider-neutral remote training run."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from voiceclonegpt.training.remote_training_run import (
    DatasetEvidence,
    InvalidTrainingRequest,
    InvalidTrainingResult,
    TrainedCheckpoint,
    TrainingRequest,
    TrainingRunError,
    execute_training,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def accepted_request(**overrides: object) -> TrainingRequest:
    fields = {
        "run_id": "run-001",
        "dataset": DatasetEvidence(
            manifest_id="dataset-001",
            declared_sha256=SHA_A,
            verified_sha256=SHA_A,
            review_state="accepted",
        ),
        "backend_id": "qwen3-tts",
        "backend_version": "1.2.3",
        "base_model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "base_model_revision": "revision-123",
        "base_model_sha256": SHA_B,
        "training_config_sha256": SHA_C,
        "artifact_kind": "fine_tuned_adapter",
    }
    fields.update(overrides)
    return TrainingRequest(**fields)  # type: ignore[arg-type]


class SuccessfulTrainer:
    def train(self, request: TrainingRequest, *, resume_from: TrainedCheckpoint | None):
        return TrainedCheckpoint(
            checkpoint_id="checkpoint-001",
            artifact_path="checkpoints/checkpoint-001",
            artifact_sha256=SHA_D,
            artifact_kind=request.artifact_kind,
        )


class MustNotRun:
    def train(self, request, *, resume_from):
        raise AssertionError("provider must not be called")


def test_completed_run_binds_inputs_and_fine_tuned_checkpoint():
    request = accepted_request()

    manifest = execute_training(request, SuccessfulTrainer())

    assert manifest.status == "completed"
    assert manifest.run_id == "run-001"
    assert manifest.dataset_manifest_id == "dataset-001"
    assert manifest.dataset_sha256 == SHA_A
    assert manifest.base_model_sha256 == SHA_B
    assert manifest.training_config_sha256 == SHA_C
    assert manifest.checkpoints == (
        TrainedCheckpoint(
            checkpoint_id="checkpoint-001",
            artifact_path="checkpoints/checkpoint-001",
            artifact_sha256=SHA_D,
            artifact_kind="fine_tuned_adapter",
        ),
    )


def test_unaccepted_dataset_is_refused_before_provider_call():
    dataset = DatasetEvidence(
        manifest_id="dataset-001",
        declared_sha256=SHA_A,
        verified_sha256=SHA_A,
        review_state="pending",
    )

    with pytest.raises(InvalidTrainingRequest, match="accepted"):
        execute_training(accepted_request(dataset=dataset), MustNotRun())


def test_dataset_checksum_mismatch_is_refused_before_provider_call():
    dataset = DatasetEvidence(
        manifest_id="dataset-001",
        declared_sha256=SHA_A,
        verified_sha256=SHA_B,
        review_state="accepted",
    )

    with pytest.raises(InvalidTrainingRequest, match="checksum"):
        execute_training(accepted_request(dataset=dataset), MustNotRun())


def test_reference_clone_cannot_claim_the_trained_model_path():
    with pytest.raises(InvalidTrainingRequest, match="fine_tuned"):
        execute_training(
            accepted_request(artifact_kind="reference_clone"),
            MustNotRun(),
        )


class EmptyArtifactTrainer:
    def train(self, request, *, resume_from):
        return TrainedCheckpoint(
            checkpoint_id="checkpoint-001",
            artifact_path="   ",
            artifact_sha256=SHA_D,
            artifact_kind=request.artifact_kind,
        )


def test_success_without_a_nonempty_artifact_is_refused():
    with pytest.raises(InvalidTrainingResult, match="artifact path"):
        execute_training(accepted_request(), EmptyArtifactTrainer())


def test_resume_uses_latest_checkpoint_and_preserves_history():
    first = TrainedCheckpoint(
        checkpoint_id="checkpoint-001",
        artifact_path="checkpoints/checkpoint-001",
        artifact_sha256=SHA_C,
        artifact_kind="fine_tuned_adapter",
    )
    second = TrainedCheckpoint(
        checkpoint_id="checkpoint-002",
        artifact_path="checkpoints/checkpoint-002",
        artifact_sha256=SHA_D,
        artifact_kind="fine_tuned_adapter",
    )

    class ResumingTrainer:
        def train(self, request, *, resume_from):
            assert resume_from == first
            return second

    manifest = execute_training(
        accepted_request(checkpoints=(first,)),
        ResumingTrainer(),
    )

    assert manifest.checkpoints == (first, second)


def test_provider_failure_is_wrapped_once_and_preserves_checkpoint_history():
    prior = TrainedCheckpoint(
        checkpoint_id="checkpoint-001",
        artifact_path="checkpoints/checkpoint-001",
        artifact_sha256=SHA_D,
        artifact_kind="fine_tuned_adapter",
    )

    class FailingTrainer:
        calls = 0

        def train(self, request, *, resume_from):
            self.calls += 1
            raise RuntimeError("remote machine stopped")

    trainer = FailingTrainer()
    with pytest.raises(TrainingRunError, match="remote machine stopped") as caught:
        execute_training(accepted_request(checkpoints=(prior,)), trainer)

    assert trainer.calls == 1
    assert caught.value.manifest.status == "failed"
    assert caught.value.manifest.checkpoints == (prior,)
    assert caught.value.__cause__.__class__ is RuntimeError


def test_provider_cannot_return_a_different_artifact_kind():
    class WrongKindTrainer:
        def train(self, request, *, resume_from):
            return TrainedCheckpoint(
                checkpoint_id="checkpoint-001",
                artifact_path="checkpoints/checkpoint-001",
                artifact_sha256=SHA_D,
                artifact_kind="fine_tuned_full",
            )

    with pytest.raises(InvalidTrainingResult, match="artifact kind"):
        execute_training(accepted_request(), WrongKindTrainer())


def test_provider_cannot_return_an_invalid_artifact_checksum():
    class InvalidChecksumTrainer:
        def train(self, request, *, resume_from):
            return TrainedCheckpoint(
                checkpoint_id="checkpoint-001",
                artifact_path="checkpoints/checkpoint-001",
                artifact_sha256="not-a-checksum",
                artifact_kind=request.artifact_kind,
            )

    with pytest.raises(InvalidTrainingResult, match="checksum"):
        execute_training(accepted_request(), InvalidChecksumTrainer())


@pytest.mark.parametrize("field", ("dataset", "base_model_sha256", "training_config_sha256"))
def test_invalid_input_checksums_are_refused_before_provider_call(field):
    value = (
        DatasetEvidence(
            manifest_id="dataset-001",
            declared_sha256="bad",
            verified_sha256="bad",
            review_state="accepted",
        )
        if field == "dataset"
        else "bad"
    )

    with pytest.raises(InvalidTrainingRequest, match="checksum"):
        execute_training(accepted_request(**{field: value}), MustNotRun())


def test_module_has_no_app_provider_network_or_filesystem_dependency():
    import voiceclonegpt.training.remote_training_run as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }

    assert imported <= {"__future__", "dataclasses", "typing"}
    assert not any(
        word in source
        for word in ("studio_app", "reader_app", "publishing", "conversion")
    )


def test_provider_must_return_a_checkpoint_record():
    class WrongResultTrainer:
        def train(self, request, *, resume_from):
            return object()

    with pytest.raises(InvalidTrainingResult, match="TrainedCheckpoint"):
        execute_training(accepted_request(), WrongResultTrainer())


@pytest.mark.parametrize(
    "field",
    (
        "run_id",
        "backend_id",
        "backend_version",
        "base_model_id",
        "base_model_revision",
    ),
)
def test_blank_required_identity_is_refused_before_provider_call(field):
    with pytest.raises(InvalidTrainingRequest, match=field):
        execute_training(accepted_request(**{field: "  "}), MustNotRun())


def test_provider_cannot_return_a_blank_checkpoint_identity():
    class BlankIdentityTrainer:
        def train(self, request, *, resume_from):
            return TrainedCheckpoint(
                checkpoint_id=" ",
                artifact_path="checkpoints/checkpoint-001",
                artifact_sha256=SHA_D,
                artifact_kind=request.artifact_kind,
            )

    with pytest.raises(InvalidTrainingResult, match="checkpoint id"):
        execute_training(accepted_request(), BlankIdentityTrainer())


def test_resume_cannot_overwrite_an_existing_checkpoint_identity():
    prior = TrainedCheckpoint(
        checkpoint_id="checkpoint-001",
        artifact_path="checkpoints/checkpoint-001",
        artifact_sha256=SHA_C,
        artifact_kind="fine_tuned_adapter",
    )

    with pytest.raises(InvalidTrainingResult, match="already exists"):
        execute_training(
            accepted_request(checkpoints=(prior,)),
            SuccessfulTrainer(),
        )
