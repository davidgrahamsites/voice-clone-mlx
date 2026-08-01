"""Public contract tests for the bounded CUDA command provider."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from voiceclonegpt.training.cost_preflight import PreflightDecision
from voiceclonegpt.training.cuda_command_provider import (
    F5_TTS_V1_RECIPE,
    QWEN3_TTS_06B_RECIPE,
    CommandResult,
    CommandExecutionFailed,
    CommandOutputLimitExceeded,
    CommandRunnerStopped,
    CommandSafetyStop,
    CommandTimedOut,
    CommandCancelled,
    ConcurrentTraining,
    InvalidCommandResult,
    InvalidCommandManifest,
    CudaCommandTrainingProvider,
    PreflightRejected,
    RecipeMismatch,
    TrainingCommandManifest,
)
from voiceclonegpt.training.remote_training_run import (
    DatasetEvidence,
    TrainedCheckpoint,
    TrainingRequest,
)


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


def test_qwen_request_maps_to_one_exact_command_and_checkpoint():
    runner = RecordingRunner(
        CommandResult(
            exit_code=0,
            captured_output_bytes=4096,
            checkpoint_id="checkpoint-001",
            checkpoint_path="/workspace/run/checkpoints/checkpoint-001",
            checkpoint_sha256=SHA_D,
            artifact_kind="fine_tuned_adapter",
        )
    )
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    checkpoint = provider.train(request(), resume_from=None)

    assert len(runner.plans) == 1
    assert runner.plans[0].argv == (
        "python",
        "-m",
        "qwen_finetune",
        "--run-id",
        "run-001",
        "--dataset-manifest",
        "/workspace/run/dataset.json",
        "--training-config",
        "/workspace/run/training.json",
        "--base-model",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "--base-revision",
        "revision-123",
        "--checkpoint-directory",
        "/workspace/run/checkpoints",
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
    runner = RecordingRunner(
        CommandResult(0, 0, "unused", "unused", SHA_D, "fine_tuned_adapter")
    )
    denied = PreflightDecision(False, ("The hard cost cap was exceeded.",), 99.0)
    provider = CudaCommandTrainingProvider(manifest(), denied, runner)

    with pytest.raises(PreflightRejected, match="hard cost cap"):
        provider.train(request(), resume_from=None)

    assert runner.plans == []


def test_qwen_recipe_never_substitutes_an_f5_request():
    runner = RecordingRunner(
        CommandResult(0, 0, "unused", "unused", SHA_D, "fine_tuned_adapter")
    )
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(RecipeMismatch, match="qwen3-tts"):
        provider.train(
            request(
                backend_id="f5-tts",
                base_model_id="SWivid/F5-TTS_v1",
            ),
            resume_from=None,
        )

    assert runner.plans == []


def test_explicit_f5_recipe_maps_only_the_f5_request():
    runner = RecordingRunner(
        CommandResult(
            0,
            10,
            "f5-checkpoint-001",
            "/workspace/run/checkpoints/f5-checkpoint-001",
            SHA_D,
            "fine_tuned_full",
        )
    )
    provider = CudaCommandTrainingProvider(
        manifest(
            recipe_id=F5_TTS_V1_RECIPE,
            backend_id="f5-tts",
            argv_prefix=("python", "-m", "f5_train"),
        ),
        approved(),
        runner,
    )

    checkpoint = provider.train(
        request(
            backend_id="f5-tts",
            base_model_id="SWivid/F5-TTS_v1",
            artifact_kind="fine_tuned_full",
        ),
        resume_from=None,
    )

    assert runner.plans[0].argv[:3] == ("python", "-m", "f5_train")
    assert checkpoint.artifact_kind == "fine_tuned_full"


def test_resume_checkpoint_maps_to_one_explicit_argument():
    runner = RecordingRunner(
        CommandResult(
            0,
            10,
            "checkpoint-002",
            "/workspace/run/checkpoints/checkpoint-002",
            SHA_D,
            "fine_tuned_adapter",
        )
    )
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)
    prior = TrainedCheckpoint(
        "checkpoint-001",
        "/workspace/run/checkpoints/checkpoint-001",
        SHA_C,
        "fine_tuned_adapter",
    )

    provider.train(request(), resume_from=prior)

    assert runner.plans[0].argv[-2:] == (
        "--resume-from",
        "/workspace/run/checkpoints/checkpoint-001",
    )


@pytest.mark.parametrize("exit_code", (1, 137))
def test_nonzero_exit_is_a_typed_failure_without_retry(exit_code):
    runner = RecordingRunner(
        CommandResult(
            exit_code,
            100,
            "checkpoint-001",
            "/workspace/run/checkpoints/checkpoint-001",
            SHA_D,
            "fine_tuned_adapter",
        )
    )
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(CommandExecutionFailed, match=str(exit_code)):
        provider.train(request(), resume_from=None)

    assert len(runner.plans) == 1


def test_output_over_cap_is_refused_after_one_call():
    runner = RecordingRunner(
        CommandResult(
            0,
            1_000_001,
            "checkpoint-001",
            "/workspace/run/checkpoints/checkpoint-001",
            SHA_D,
            "fine_tuned_adapter",
        )
    )
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(CommandOutputLimitExceeded, match="1,000,000"):
        provider.train(request(), resume_from=None)

    assert len(runner.plans) == 1


def test_runner_timeout_is_typed_and_not_retried():
    class TimeoutRunner:
        calls = 0

        def run(self, plan):
            self.calls += 1
            raise CommandRunnerStopped("timeout", "hard limit reached")

    runner = TimeoutRunner()
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(CommandTimedOut, match="hard limit reached"):
        provider.train(request(), resume_from=None)

    assert runner.calls == 1


@pytest.mark.parametrize(
    ("kind", "error_type"),
    (
        ("cancelled", CommandCancelled),
        ("authentication", CommandSafetyStop),
        ("quota", CommandSafetyStop),
        ("rate_limit", CommandSafetyStop),
        ("abuse", CommandSafetyStop),
        ("repeated_error", CommandSafetyStop),
    ),
)
def test_stop_signals_are_typed_and_never_retried(kind, error_type):
    class StoppedRunner:
        calls = 0

        def run(self, plan):
            self.calls += 1
            raise CommandRunnerStopped(kind, "stop now")

    runner = StoppedRunner()
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(error_type, match=kind):
        provider.train(request(), resume_from=None)

    assert runner.calls == 1


def test_one_provider_instance_allows_only_one_in_flight_command():
    result = CommandResult(
        0,
        10,
        "checkpoint-001",
        "/workspace/run/checkpoints/checkpoint-001",
        SHA_D,
        "fine_tuned_adapter",
    )

    class ReentrantRunner:
        calls = 0
        nested_error = None
        provider = None

        def run(self, plan):
            self.calls += 1
            if self.calls == 1:
                try:
                    self.provider.train(request(), resume_from=None)
                except Exception as exc:
                    self.nested_error = exc
            return result

    runner = ReentrantRunner()
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)
    runner.provider = provider

    provider.train(request(), resume_from=None)

    assert runner.calls == 1
    assert isinstance(runner.nested_error, ConcurrentTraining)


def test_runner_must_return_a_command_result():
    class MalformedRunner:
        def run(self, plan):
            return object()

    provider = CudaCommandTrainingProvider(manifest(), approved(), MalformedRunner())

    with pytest.raises(InvalidCommandResult, match="CommandResult"):
        provider.train(request(), resume_from=None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("timeout_seconds", 0),
        ("timeout_seconds", True),
        ("max_output_bytes", 0),
        ("max_output_bytes", True),
        ("argv_prefix", "python -m train"),
        ("argv_prefix", ()),
    ),
)
def test_command_limits_and_argv_are_validated(field, value):
    with pytest.raises(InvalidCommandManifest, match=field):
        manifest(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("exit_code", True),
        ("captured_output_bytes", -1),
        ("checkpoint_id", " "),
        ("checkpoint_path", ""),
        ("checkpoint_sha256", "bad"),
        ("artifact_kind", "reference_clone"),
    ),
)
def test_malformed_runner_fields_are_typed_failures(field, value):
    provider = CudaCommandTrainingProvider(
        manifest(),
        approved(),
        RecordingRunner(command_result(**{field: value})),
    )

    with pytest.raises(InvalidCommandResult, match=field):
        provider.train(request(), resume_from=None)


def test_unexpected_runner_error_is_wrapped_once_with_its_cause():
    class BrokenRunner:
        calls = 0

        def run(self, plan):
            self.calls += 1
            raise RuntimeError("runner transport broke")

    runner = BrokenRunner()
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(CommandExecutionFailed, match="transport broke") as caught:
        provider.train(request(), resume_from=None)

    assert runner.calls == 1
    assert caught.value.__cause__.__class__ is RuntimeError


def test_module_has_no_shell_network_vendor_or_app_dependency():
    import voiceclonegpt.training.cuda_command_provider as module

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

    assert imported <= {
        "__future__",
        "dataclasses",
        "threading",
        "typing",
        "voiceclonegpt.training.cost_preflight",
        "voiceclonegpt.training.remote_training_run",
    }
    assert "shell" not in source
    assert not any(
        word in source
        for word in (
            "subprocess",
            "socket",
            "requests",
            "boto",
            "runpod",
            "vastai",
            "studio_app",
            "reader_app",
        )
    )


@pytest.mark.parametrize(
    "field",
    (
        "recipe_id",
        "recipe_revision",
        "backend_id",
        "working_directory",
        "dataset_manifest_path",
        "training_config_path",
        "checkpoint_directory",
        "target_id",
    ),
)
def test_command_manifest_requires_all_identity_and_path_text(field):
    with pytest.raises(InvalidCommandManifest, match=field):
        manifest(**{field: "  "})


@pytest.mark.parametrize(
    "environment",
    (
        "CUDA_VISIBLE_DEVICES=0",
        (("", "0"),),
        (("CUDA_VISIBLE_DEVICES", 0),),
        (("CUDA_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", "1")),
    ),
)
def test_environment_allowlist_is_explicit_and_unambiguous(environment):
    with pytest.raises(InvalidCommandManifest, match="environment"):
        manifest(environment=environment)


def test_inconsistent_preflight_never_reaches_runner():
    runner = RecordingRunner(command_result())
    inconsistent = PreflightDecision(True, ("approval is contradictory",), 12.5)
    provider = CudaCommandTrainingProvider(manifest(), inconsistent, runner)

    with pytest.raises(PreflightRejected, match="contradictory"):
        provider.train(request(), resume_from=None)

    assert runner.plans == []


def test_qwen_recipe_refuses_a_different_qwen_base_model():
    runner = RecordingRunner(command_result())
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)

    with pytest.raises(RecipeMismatch, match="base model"):
        provider.train(
            request(base_model_id="Qwen/Some-Other-Model"),
            resume_from=None,
        )

    assert runner.plans == []
