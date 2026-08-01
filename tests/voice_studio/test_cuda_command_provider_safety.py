"""Failure, stop, concurrency, purity, and refusal tests for the CUDA provider."""

import ast
from pathlib import Path

import pytest

from voiceclonegpt.training.cuda_command_provider import (
    CommandCancelled, CommandExecutionFailed, CommandOutputLimitExceeded,
    CommandRunnerStopped, CommandSafetyStop, CommandTimedOut, ConcurrentTraining,
    CudaCommandTrainingProvider, RecipeMismatch,
)
from cuda_command_provider_test_support import SHA_D, RecordingRunner, approved, command_result, manifest, request


@pytest.mark.parametrize("exit_code", (1, 137))
def test_nonzero_exit_is_a_typed_failure_without_retry(exit_code):
    runner = RecordingRunner(command_result(exit_code=exit_code, captured_output_bytes=100))
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)
    with pytest.raises(CommandExecutionFailed, match=str(exit_code)):
        provider.train(request(), resume_from=None)
    assert len(runner.plans) == 1


def test_output_over_cap_is_refused_after_one_call():
    runner = RecordingRunner(command_result(captured_output_bytes=1_000_001))
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


@pytest.mark.parametrize(("kind", "error_type"), (("cancelled", CommandCancelled), ("output_limit", CommandOutputLimitExceeded), ("authentication", CommandSafetyStop), ("quota", CommandSafetyStop), ("rate_limit", CommandSafetyStop), ("abuse", CommandSafetyStop), ("repeated_error", CommandSafetyStop)))
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
    result = command_result()
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
    imported = {name for node in ast.walk(tree) for name in ([alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])}
    assert imported <= {"__future__", "dataclasses", "threading", "typing", "voiceclonegpt.training.cost_preflight", "voiceclonegpt.training.remote_training_run"}
    assert "shell" not in source
    assert not any(word in source for word in ("subprocess", "socket", "requests", "boto", "runpod", "vastai", "studio_app", "reader_app"))


def test_inconsistent_preflight_never_reaches_runner():
    from voiceclonegpt.training.cost_preflight import PreflightDecision
    from voiceclonegpt.training.cuda_command_provider import PreflightRejected
    runner = RecordingRunner(command_result())
    provider = CudaCommandTrainingProvider(manifest(), PreflightDecision(True, ("approval is contradictory",), 12.5), runner)
    with pytest.raises(PreflightRejected, match="contradictory"):
        provider.train(request(), resume_from=None)
    assert runner.plans == []


def test_qwen_recipe_refuses_a_different_qwen_base_model():
    runner = RecordingRunner(command_result())
    provider = CudaCommandTrainingProvider(manifest(), approved(), runner)
    with pytest.raises(RecipeMismatch, match="base model"):
        provider.train(request(base_model_id="Qwen/Some-Other-Model"), resume_from=None)
    assert runner.plans == []
