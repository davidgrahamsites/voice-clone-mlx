"""Run one explicit CUDA training command through an injected runner."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from voiceclonegpt.training.cost_preflight import PreflightDecision
from voiceclonegpt.training.remote_training_run import TrainedCheckpoint, TrainingRequest

QWEN3_TTS_06B_RECIPE = "qwen3-tts-0.6b-base-official-cuda"
F5_TTS_V1_RECIPE = "f5-tts-v1-official-cuda"
_RECIPE_IDENTITIES = {
    QWEN3_TTS_06B_RECIPE: (
        "qwen3-tts",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ),
    F5_TTS_V1_RECIPE: ("f5-tts", "SWivid/F5-TTS_v1"),
}
_LOWER_HEX = frozenset("0123456789abcdef")


class CudaCommandProviderError(RuntimeError):
    """The bounded command provider refused or failed one request."""


class PreflightRejected(CudaCommandProviderError):
    """The approved cost and shutdown gate did not permit execution."""


class RecipeMismatch(CudaCommandProviderError):
    """The explicit recipe and request name different model families."""


class InvalidCommandManifest(CudaCommandProviderError):
    """The frozen command description is unsafe or incomplete."""


class CommandExecutionFailed(CudaCommandProviderError):
    """The one bounded command attempt did not complete successfully."""


class CommandOutputLimitExceeded(CudaCommandProviderError):
    """The runner reported more captured output than the approved cap."""


class CommandTimedOut(CommandExecutionFailed):
    """The runner stopped at the command plan's hard time limit."""


class CommandCancelled(CommandExecutionFailed):
    """The caller cancelled the one in-flight command."""


class CommandSafetyStop(CommandExecutionFailed):
    """An authentication, quota, rate, abuse, or repeated-error signal stopped work."""


class ConcurrentTraining(CudaCommandProviderError):
    """This provider already has its one allowed command in flight."""


class InvalidCommandResult(CudaCommandProviderError):
    """The runner returned data that cannot identify a checkpoint."""


class CommandRunnerStopped(RuntimeError):
    """Typed stop signal emitted by an injected command runner."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class TrainingCommandManifest:
    recipe_id: str
    recipe_revision: str
    backend_id: str
    argv_prefix: tuple[str, ...]
    working_directory: str
    dataset_manifest_path: str
    training_config_path: str
    checkpoint_directory: str
    environment: tuple[tuple[str, str], ...]
    target_id: str
    timeout_seconds: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "recipe_id",
            "recipe_revision",
            "backend_id",
            "working_directory",
            "dataset_manifest_path",
            "training_config_path",
            "checkpoint_directory",
            "target_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise InvalidCommandManifest(f"{field} must be non-blank text.")
        if (
            not isinstance(self.argv_prefix, tuple)
            or not self.argv_prefix
            or any(not isinstance(item, str) or not item for item in self.argv_prefix)
        ):
            raise InvalidCommandManifest(
                "argv_prefix must be a nonempty tuple of nonempty strings."
            )
        if not isinstance(self.environment, tuple):
            raise InvalidCommandManifest("environment must be an explicit tuple.")
        environment_keys = []
        for entry in self.environment:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0].strip()
                or not isinstance(entry[1], str)
            ):
                raise InvalidCommandManifest(
                    "environment entries must be non-blank key and string value pairs."
                )
            environment_keys.append(entry[0])
        if len(environment_keys) != len(set(environment_keys)):
            raise InvalidCommandManifest("environment keys must not be duplicated.")
        for field in ("timeout_seconds", "max_output_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidCommandManifest(f"{field} must be a positive integer.")


@dataclass(frozen=True)
class CommandPlan:
    recipe_id: str
    recipe_revision: str
    backend_id: str
    argv: tuple[str, ...]
    working_directory: str
    environment: tuple[tuple[str, str], ...]
    target_id: str
    timeout_seconds: int
    max_output_bytes: int
    concurrency_limit: int = 1


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    captured_output_bytes: int
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    artifact_kind: str


class CommandRunner(Protocol):
    def run(self, plan: CommandPlan) -> CommandResult: ...


def _check_preflight(preflight: PreflightDecision) -> None:
    if not preflight.approved or preflight.reasons:
        reasons = "; ".join(preflight.reasons) or "preflight was not approved"
        raise PreflightRejected(f"CUDA training preflight rejected: {reasons}")


def _check_recipe(command: TrainingCommandManifest, request: TrainingRequest) -> None:
    identity = _RECIPE_IDENTITIES.get(command.recipe_id)
    backend = identity[0] if identity else None
    if backend is None or command.backend_id != backend or request.backend_id != backend:
        raise RecipeMismatch(
            "Recipe, command manifest, and request must name one exact backend; "
            f"got recipe {command.recipe_id!r}, command backend "
            f"{command.backend_id!r}, request backend {request.backend_id!r}."
        )
    expected_base_model = identity[1]
    if request.base_model_id != expected_base_model:
        raise RecipeMismatch(
            f"Recipe {command.recipe_id!r} requires base model "
            f"{expected_base_model!r}, got {request.base_model_id!r}."
        )


def _build_plan(
    command: TrainingCommandManifest,
    request: TrainingRequest,
    resume_from: TrainedCheckpoint | None,
) -> CommandPlan:
    argv = command.argv_prefix + (
        "--run-id",
        request.run_id,
        "--dataset-manifest",
        command.dataset_manifest_path,
        "--training-config",
        command.training_config_path,
        "--base-model",
        request.base_model_id,
        "--base-revision",
        request.base_model_revision,
        "--checkpoint-directory",
        command.checkpoint_directory,
    )
    if resume_from is not None:
        argv += ("--resume-from", resume_from.artifact_path)
    return CommandPlan(
        recipe_id=command.recipe_id,
        recipe_revision=command.recipe_revision,
        backend_id=command.backend_id,
        argv=argv,
        working_directory=command.working_directory,
        environment=command.environment,
        target_id=command.target_id,
        timeout_seconds=command.timeout_seconds,
        max_output_bytes=command.max_output_bytes,
    )


def _translate_runner_stop(exc: CommandRunnerStopped) -> CommandExecutionFailed:
    if exc.kind == "timeout":
        return CommandTimedOut(f"CUDA training command timed out: {exc}")
    if exc.kind == "cancelled":
        return CommandCancelled(f"CUDA training command stopped (cancelled): {exc}")
    if exc.kind == "output_limit":
        return CommandOutputLimitExceeded(
            f"CUDA training command stopped (output_limit): {exc}"
        )
    if exc.kind in {
        "authentication",
        "quota",
        "rate_limit",
        "abuse",
        "repeated_error",
    }:
        return CommandSafetyStop(
            f"CUDA training command stopped ({exc.kind}): {exc}"
        )
    return CommandExecutionFailed(
        f"CUDA training command stopped ({exc.kind}): {exc}"
    )


def _run_once(runner: CommandRunner, lock: Lock, plan: CommandPlan) -> CommandResult:
    if not lock.acquire(blocking=False):
        raise ConcurrentTraining(
            "This CUDA training provider already has one command in flight."
        )
    try:
        try:
            return runner.run(plan)
        except CommandRunnerStopped as exc:
            raise _translate_runner_stop(exc) from exc
        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            raise CommandExecutionFailed(
                f"CUDA training runner failed: {reason}"
            ) from exc
    finally:
        lock.release()


def _checkpoint_from_result(
    result: CommandResult,
    request: TrainingRequest,
    max_output_bytes: int,
) -> TrainedCheckpoint:
    if not isinstance(result, CommandResult):
        raise InvalidCommandResult("The runner must return a CommandResult.")
    if isinstance(result.exit_code, bool) or not isinstance(result.exit_code, int):
        raise InvalidCommandResult("exit_code must be an integer.")
    if (
        isinstance(result.captured_output_bytes, bool)
        or not isinstance(result.captured_output_bytes, int)
        or result.captured_output_bytes < 0
    ):
        raise InvalidCommandResult(
            "captured_output_bytes must be a non-negative integer."
        )
    for field in ("checkpoint_id", "checkpoint_path"):
        value = getattr(result, field)
        if not isinstance(value, str) or not value.strip():
            raise InvalidCommandResult(f"{field} must be non-blank text.")
    if (
        not isinstance(result.checkpoint_sha256, str)
        or len(result.checkpoint_sha256) != 64
        or any(character not in _LOWER_HEX for character in result.checkpoint_sha256)
    ):
        raise InvalidCommandResult(
            "checkpoint_sha256 must be a lowercase SHA-256 checksum."
        )
    if result.artifact_kind != request.artifact_kind:
        raise InvalidCommandResult(
            "artifact_kind must exactly match the training request."
        )
    if result.exit_code != 0:
        raise CommandExecutionFailed(
            f"CUDA training command exited with status {result.exit_code}."
        )
    if result.captured_output_bytes > max_output_bytes:
        raise CommandOutputLimitExceeded(
            "CUDA training command exceeded the captured-output cap of "
            f"{max_output_bytes:,} bytes."
        )
    return TrainedCheckpoint(
        checkpoint_id=result.checkpoint_id,
        artifact_path=result.checkpoint_path,
        artifact_sha256=result.checkpoint_sha256,
        artifact_kind=result.artifact_kind,
    )


class CudaCommandTrainingProvider:
    """Translate one training request into one bounded runner call."""

    def __init__(
        self,
        command: TrainingCommandManifest,
        preflight: PreflightDecision,
        runner: CommandRunner,
    ):
        self._command = command
        self._preflight = preflight
        self._runner = runner
        self._run_lock = Lock()

    def train(
        self,
        request: TrainingRequest,
        *,
        resume_from: TrainedCheckpoint | None,
    ) -> TrainedCheckpoint:
        _check_preflight(self._preflight)
        _check_recipe(self._command, request)
        plan = _build_plan(self._command, request, resume_from)
        result = _run_once(self._runner, self._run_lock, plan)
        return _checkpoint_from_result(
            result,
            request,
            self._command.max_output_bytes,
        )
