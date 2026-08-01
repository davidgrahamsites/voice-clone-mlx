"""Command and recipe contract tests for the bounded CUDA command provider."""

from voiceclonemlx.training.cuda_command_provider import (
    F5_TTS_V1_RECIPE,
    QWEN3_TTS_06B_RECIPE,
    CommandResult,
    CudaCommandTrainingProvider,
    PreflightRejected,
    RecipeMismatch,
)
from voiceclonemlx.training.remote_training_run import TrainedCheckpoint

from cuda_command_provider_test_support import (
    SHA_C,
    SHA_D,
    RecordingRunner,
    approved,
    manifest,
    request,
)


def test_qwen_request_maps_to_one_exact_command_and_checkpoint():
    runner = RecordingRunner(CommandResult(0, 4096, "checkpoint-001", "/workspace/run/checkpoints/checkpoint-001", SHA_D, "fine_tuned_adapter"))
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    checkpoint = provider.train(request(), resume_from=None)

    assert len(runner.plans) == 1
    assert runner.plans[0].argv == (
        "python", "-m", "qwen_finetune", "--run-id", "run-001",
        "--dataset-manifest", "/workspace/run/dataset.json", "--training-config",
        "/workspace/run/training.json", "--base-model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "--base-revision", "revision-123", "--checkpoint-directory", "/workspace/run/checkpoints",
    )
    assert runner.plans[0].timeout_seconds == 3600
    assert runner.plans[0].max_output_bytes == 1_000_000
    assert runner.plans[0].recipe_id == QWEN3_TTS_06B_RECIPE
    assert runner.plans[0].recipe_revision == "recipe-commit-123"
    assert runner.plans[0].backend_id == "qwen3-tts"
    assert checkpoint.checkpoint_id == "checkpoint-001"
    assert checkpoint.artifact_path == "/workspace/run/checkpoints/checkpoint-001"
    assert checkpoint.artifact_sha256 == SHA_D


def test_denied_preflight_never_reaches_runner():
    runner = RecordingRunner(CommandResult(0, 0, "unused", "unused", SHA_D, "fine_tuned_adapter"))
    from voiceclonemlx.training.cost_preflight import PreflightDecision
    denied = PreflightDecision(False, ("The hard cost cap was exceeded.",), 99.0)
    provider = CudaCommandTrainingProvider(manifest(), denied, runner)
    import pytest
    with pytest.raises(PreflightRejected, match="hard cost cap"):
        provider.train(request(), resume_from=None)
    assert runner.plans == []


def test_qwen_recipe_never_substitutes_an_f5_request():
    import pytest
    runner = RecordingRunner(CommandResult(0, 0, "unused", "unused", SHA_D, "fine_tuned_adapter"))
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)
    with pytest.raises(RecipeMismatch, match="qwen3-tts"):
        provider.train(request(backend_id="f5-tts", base_model_id="SWivid/F5-TTS_v1"), resume_from=None)
    assert runner.plans == []


def test_explicit_f5_recipe_maps_only_the_f5_request():
    runner = RecordingRunner(CommandResult(0, 10, "f5-checkpoint-001", "/workspace/run/checkpoints/f5-checkpoint-001", SHA_D, "fine_tuned_full"))
    provider = CudaCommandTrainingProvider(manifest(recipe_id=F5_TTS_V1_RECIPE, backend_id="f5-tts", argv_prefix=("python", "-m", "f5_train")), approved(), runner)
    checkpoint = provider.train(request(backend_id="f5-tts", base_model_id="SWivid/F5-TTS_v1", artifact_kind="fine_tuned_full"), resume_from=None)
    assert runner.plans[0].argv[:3] == ("python", "-m", "f5_train")
    assert checkpoint.artifact_kind == "fine_tuned_full"


def test_resume_checkpoint_maps_to_one_explicit_argument():
    runner = RecordingRunner(CommandResult(0, 10, "checkpoint-002", "/workspace/run/checkpoints/checkpoint-002", SHA_D, "fine_tuned_adapter"))
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)
    prior = TrainedCheckpoint("checkpoint-001", "/workspace/run/checkpoints/checkpoint-001", SHA_C, "fine_tuned_adapter")
    provider.train(request(), resume_from=prior)
    assert runner.plans[0].argv[-2:] == ("--resume-from", "/workspace/run/checkpoints/checkpoint-001")
