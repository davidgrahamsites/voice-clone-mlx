"""Execute one approved local training command through injected process seams."""

from __future__ import annotations

from dataclasses import replace
from os.path import commonpath, isabs, realpath
from threading import Lock
from time import monotonic
from typing import Protocol

from voiceclonemlx.training.cuda_command_provider import (
    CommandPlan,
    CommandResult,
    CommandRunnerStopped,
)


class LocalCommandRunnerError(RuntimeError):
    """The local runner refused or could not execute a command plan."""


class LocalCommandValidationError(LocalCommandRunnerError):
    """The immutable command plan does not satisfy this host's policy."""


class LocalCommandConcurrencyError(LocalCommandRunnerError):
    """This runner already has its one allowed command in flight."""


class LocalCommandTimedOut(CommandRunnerStopped, LocalCommandRunnerError):
    """The command reached its approved hard time limit."""

    def __init__(self) -> None:
        super().__init__("timeout", "The local training command reached its hard time limit.")


class LocalCommandCancelled(CommandRunnerStopped, LocalCommandRunnerError):
    """The caller cancelled the command before it completed."""

    def __init__(self) -> None:
        super().__init__("cancelled", "The local training command was cancelled.")


class LocalCommandOutputLimitExceeded(CommandRunnerStopped, LocalCommandRunnerError):
    """The command exceeded its approved captured-output cap."""

    def __init__(self) -> None:
        super().__init__("output_limit", "The local training command exceeded its captured-output cap.")


class Process(Protocol):
    exit_code: int

    def read_chunk(self, timeout: float) -> bytes | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessFactory(Protocol):
    def start(
        self, argv: tuple[str, ...], *, cwd: str, env: dict[str, str]
    ) -> Process: ...


class ResultMapper(Protocol):
    def result_for(self, plan: CommandPlan, captured_output: bytes) -> CommandResult: ...


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...


class _SystemClock:
    def monotonic(self) -> float:
        return monotonic()


def _is_under(path: str, root: str) -> bool:
    resolved_path = realpath(path)
    resolved_root = realpath(root)
    return commonpath((resolved_path, resolved_root)) == resolved_root


class LocalCommandRunner:
    """Run one plan with an injected process factory and result mapper."""

    def __init__(
        self,
        *,
        target_id: str,
        working_directory_roots: tuple[str, ...],
        environment_allowlist: frozenset[str],
        process_factory: ProcessFactory,
        result_mapper: ResultMapper,
        clock: Clock | None = None,
    ) -> None:
        self._target_id = target_id
        self._roots = working_directory_roots
        self._environment_allowlist = environment_allowlist
        self._process_factory = process_factory
        self._result_mapper = result_mapper
        self._clock = clock or _SystemClock()
        self._lock = Lock()

    def run(
        self, plan: CommandPlan, *, cancel: CancellationToken | None = None
    ) -> CommandResult:
        if not self._lock.acquire(blocking=False):
            raise LocalCommandConcurrencyError(
                "One local training command is already running."
            )
        try:
            self._validate(plan)
            process = self._process_factory.start(
                plan.argv, cwd=plan.working_directory, env=dict(plan.environment)
            )
            return self._collect(process, plan, cancel)
        finally:
            self._lock.release()

    def _validate(self, plan: CommandPlan) -> None:
        if plan.target_id != self._target_id:
            raise LocalCommandValidationError("The command target is not this CUDA host.")
        if (
            not isinstance(plan.argv, tuple)
            or not plan.argv
            or any(not isinstance(arg, str) or not arg for arg in plan.argv)
        ):
            raise LocalCommandValidationError("argv must be a nonempty argument tuple.")
        if not isabs(plan.working_directory) or not any(
            _is_under(plan.working_directory, root) for root in self._roots
        ):
            raise LocalCommandValidationError("The working directory is not an approved host path.")
        if (
            not isinstance(plan.environment, tuple)
            or any(
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
                or not isinstance(entry[1], str)
                for entry in plan.environment
            )
        ):
            raise LocalCommandValidationError(
                "The command environment must be an explicit key/value tuple."
            )
        environment_keys = tuple(key for key, _ in plan.environment)
        if len(environment_keys) != len(set(environment_keys)):
            raise LocalCommandValidationError(
                "The command environment must not repeat a key."
            )
        if any(key not in self._environment_allowlist for key in environment_keys):
            raise LocalCommandValidationError("The command environment contains a disallowed key.")

    def _collect(
        self,
        process: Process,
        plan: CommandPlan,
        cancel: CancellationToken | None,
    ) -> CommandResult:
        output = bytearray()
        deadline = self._clock.monotonic() + plan.timeout_seconds
        while True:
            if cancel is not None and cancel.is_cancelled():
                process.terminate()
                raise LocalCommandCancelled()
            remaining = deadline - self._clock.monotonic()
            if remaining <= 0:
                process.terminate()
                raise LocalCommandTimedOut()
            chunk = process.read_chunk(min(remaining, 0.1))
            if chunk is None:
                result = self._result_mapper.result_for(plan, bytes(output))
                return replace(
                    result,
                    exit_code=process.exit_code,
                    captured_output_bytes=len(output),
                )
            output.extend(chunk)
            if len(output) > plan.max_output_bytes:
                process.terminate()
                raise LocalCommandOutputLimitExceeded()
