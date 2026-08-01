"""Public contract tests for the local bounded training command runner."""

from __future__ import annotations

from voiceclonegpt.training.cuda_command_provider import CommandPlan, CommandResult
import pytest

from voiceclonegpt.training.local_command_runner import (
    CancellationToken,
    LocalCommandCancelled,
    LocalCommandConcurrencyError,
    LocalCommandOutputLimitExceeded,
    LocalCommandRunner,
    LocalCommandRunnerError,
    LocalCommandTimedOut,
    LocalCommandValidationError,
)


class FakeProcess:
    def __init__(
        self, chunks: tuple[bytes, ...] = (b"trained",), exit_code: int = 0
    ) -> None:
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False
        self._chunks = list(chunks)

    def read_chunk(self, timeout: float):
        return self._chunks.pop(0) if self._chunks else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class RecordingProcessFactory:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.calls = []

    def start(self, argv, *, cwd, env):
        self.calls.append((argv, cwd, env))
        return self.process


class RecordingResultMapper:
    def __init__(self) -> None:
        self.calls = []

    def result_for(self, command_plan, captured_output):
        self.calls.append((command_plan, captured_output))
        return CommandResult(
            0,
            len(captured_output),
            "checkpoint-001",
            "/workspace/run/checkpoints/checkpoint-001",
            "d" * 64,
            "fine_tuned_adapter",
        )


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def monotonic(self) -> float:
        return next(self._values)


class Cancelled:
    def is_cancelled(self) -> bool:
        return True


def plan(**overrides: object) -> CommandPlan:
    fields: dict[str, object] = {
        "recipe_id": "qwen3-tts-0.6b-base-official-cuda",
        "recipe_revision": "recipe-commit-123",
        "backend_id": "qwen3-tts",
        "argv": ("python", "-m", "qwen_finetune"),
        "working_directory": "/workspace/qwen",
        "environment": (("CUDA_VISIBLE_DEVICES", "0"),),
        "target_id": "cuda-host-01",
        "timeout_seconds": 60,
        "max_output_bytes": 1024,
    }
    fields.update(overrides)
    return CommandPlan(**fields)  # type: ignore[arg-type]


def test_runs_one_safe_argv_with_only_allowed_environment() -> None:
    factory = RecordingProcessFactory(FakeProcess())
    mapper = RecordingResultMapper()
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=factory,
        result_mapper=mapper,
    )

    result = runner.run(plan())

    assert result.exit_code == 0
    assert result.captured_output_bytes == 7
    assert factory.calls == [
        (
            ("python", "-m", "qwen_finetune"),
            "/workspace/qwen",
            {"CUDA_VISIBLE_DEVICES": "0"},
        )
    ]
    assert mapper.calls == [(plan(), b"trained")]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("argv", "python -m train", "argv"),
        ("working_directory", "/outside", "working directory"),
        ("working_directory", "/workspace/../outside", "working directory"),
        ("environment", (("UNSAFE", "1"),), "environment"),
        ("environment", "CUDA_VISIBLE_DEVICES=0", "environment"),
        ("environment", (("CUDA_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", "1")), "environment"),
        ("target_id", "another-host", "target"),
    ),
)
def test_refuses_unsafe_plan_before_launch(field, value, message) -> None:
    factory = RecordingProcessFactory(FakeProcess())
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=factory,
        result_mapper=RecordingResultMapper(),
    )

    with pytest.raises(LocalCommandValidationError, match=message):
        runner.run(plan(**{field: value}))

    assert factory.calls == []


def test_cancellation_terminates_before_reading_output() -> None:
    process = FakeProcess()
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=RecordingProcessFactory(process),
        result_mapper=RecordingResultMapper(),
    )

    with pytest.raises(LocalCommandCancelled):
        runner.run(plan(), cancel=Cancelled())

    assert process.terminated is True


def test_output_cap_terminates_the_single_process() -> None:
    process = FakeProcess((b"123", b"456"))
    factory = RecordingProcessFactory(process)
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=factory,
        result_mapper=RecordingResultMapper(),
    )

    with pytest.raises(LocalCommandOutputLimitExceeded):
        runner.run(plan(max_output_bytes=5))

    assert process.terminated is True
    assert len(factory.calls) == 1


def test_uses_the_process_exit_code_and_observed_output_size() -> None:
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=RecordingProcessFactory(FakeProcess(exit_code=17)),
        result_mapper=RecordingResultMapper(),
    )

    result = runner.run(plan())

    assert result.exit_code == 17
    assert result.captured_output_bytes == 7


def test_timeout_terminates_the_single_process() -> None:
    process = FakeProcess((b"still running",))
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=RecordingProcessFactory(process),
        result_mapper=RecordingResultMapper(),
        clock=FakeClock(0, 0, 61),
    )

    with pytest.raises(LocalCommandTimedOut):
        runner.run(plan(timeout_seconds=60))

    assert process.terminated is True


def test_allows_only_one_in_flight_local_command() -> None:
    class ReentrantProcess(FakeProcess):
        runner = None
        nested_error = None

        def read_chunk(self, timeout: float):
            if self.runner is not None:
                try:
                    self.runner.run(plan())
                except Exception as error:
                    self.nested_error = error
                self.runner = None
            return super().read_chunk(timeout)

    process = ReentrantProcess()
    runner = LocalCommandRunner(
        target_id="cuda-host-01",
        working_directory_roots=("/workspace",),
        environment_allowlist=frozenset(("CUDA_VISIBLE_DEVICES",)),
        process_factory=RecordingProcessFactory(process),
        result_mapper=RecordingResultMapper(),
    )
    process.runner = runner

    runner.run(plan())

    assert isinstance(process.nested_error, LocalCommandConcurrencyError)
