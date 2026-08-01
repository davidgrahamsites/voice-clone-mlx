"""Shared fixtures for CUDA command-provider contract tests."""

from voiceclonemlx.training.cost_preflight import PreflightDecision
from voiceclonemlx.training.cuda_command_provider import (
    F5_TTS_V1_RECIPE,
    QWEN3_TTS_06B_RECIPE,
    CommandResult,
    CudaCommandTrainingProvider,
    TrainingCommandManifest,
)
from voiceclonemlx.training.remote_training_run import DatasetEvidence, TrainingRequest

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def request(**overrides: object) -> TrainingRequest:
    fields = {
        "run_id": "run-001",
        "dataset": DatasetEvidence("dataset-001", SHA_A, SHA_A, "accepted"),
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


def manifest(**overrides: object) -> TrainingCommandManifest:
    fields = {
        "recipe_id": QWEN3_TTS_06B_RECIPE,
        "recipe_revision": "recipe-commit-123",
        "backend_id": "qwen3-tts",
        "argv_prefix": ("python", "-m", "qwen_finetune"),
        "working_directory": "/workspace/qwen",
        "dataset_manifest_path": "/workspace/run/dataset.json",
        "training_config_path": "/workspace/run/training.json",
        "checkpoint_directory": "/workspace/run/checkpoints",
        "environment": (("CUDA_VISIBLE_DEVICES", "0"),),
        "target_id": "local-cuda-0",
        "timeout_seconds": 3600,
        "max_output_bytes": 1_000_000,
    }
    fields.update(overrides)
    return TrainingCommandManifest(**fields)  # type: ignore[arg-type]


class RecordingRunner:
    def __init__(self, result: CommandResult):
        self.plans = []
        self.result = result

    def run(self, plan):
        self.plans.append(plan)
        return self.result


def approved() -> PreflightDecision:
    return PreflightDecision(True, (), 12.5)


def command_result(**overrides: object) -> CommandResult:
    fields = {
        "exit_code": 0,
        "captured_output_bytes": 10,
        "checkpoint_id": "checkpoint-001",
        "checkpoint_path": "/workspace/run/checkpoints/checkpoint-001",
        "checkpoint_sha256": SHA_D,
        "artifact_kind": "fine_tuned_adapter",
    }
    fields.update(overrides)
    return CommandResult(**fields)  # type: ignore[arg-type]
