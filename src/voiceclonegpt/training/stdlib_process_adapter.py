"""Optional standard-library host adapters for one bounded training command."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from os.path import commonpath, isabs, realpath
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, Protocol

from voiceclonegpt.training.cuda_command_provider import CommandPlan, CommandResult


class ResultMappingError(ValueError):
    """Captured output did not contain one approved checkpoint result."""


class _Pipe(Protocol):
    def read(self, size: int) -> bytes: ...


class _Popen(Protocol):
    stdout: _Pipe | None
    pid: int

    def poll(self) -> int | None: ...


class _StdlibProcess:
    def __init__(self, process: _Popen, group_signaler: Callable[[int, int], None]) -> None:
        if process.stdout is None:
            raise RuntimeError("The training process did not provide a captured-output pipe.")
        self._process = process
        self._group_signaler = group_signaler
        self._chunks: Queue[bytes | object] = Queue()
        self._eof = object()
        Thread(target=self._pump, args=(process.stdout,), daemon=True).start()

    @property
    def exit_code(self) -> int:
        result = self._process.poll()
        if isinstance(result, bool) or not isinstance(result, int):
            raise RuntimeError("The training process ended without an exit status.")
        return result

    def _pump(self, pipe: _Pipe) -> None:
        while True:
            chunk = pipe.read(65_536)
            if not chunk:
                self._chunks.put(self._eof)
                return
            self._chunks.put(chunk)

    def read_chunk(self, timeout: float) -> bytes | None:
        try:
            item = self._chunks.get(timeout=timeout)
        except Empty:
            return b""
        return None if item is self._eof else item  # type: ignore[return-value]

    def terminate(self) -> None:
        self._group_signaler(self._process.pid, signal.SIGTERM)

    def kill(self) -> None:
        self._group_signaler(self._process.pid, signal.SIGKILL)


class StdlibProcessFactory:
    """Start an argument-vector command in a new process group.

    The caller's runner owns timeouts, cancellation, output limits, and retry policy.
    """

    def __init__(
        self,
        *,
        popen_factory: Callable[..., _Popen] = subprocess.Popen,
        group_signaler: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self._popen_factory = popen_factory
        self._group_signaler = group_signaler

    def start(
        self, argv: tuple[str, ...], *, cwd: str, env: dict[str, str]
    ) -> _StdlibProcess:
        process = self._popen_factory(
            argv,
            cwd=cwd,
            env=env,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return _StdlibProcess(process, self._group_signaler)


_FIELDS = frozenset(
    ("checkpoint_id", "checkpoint_path", "checkpoint_sha256", "artifact_kind")
)
_LOWER_HEX = frozenset("0123456789abcdef")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultMappingError("The checkpoint result repeats a field.")
        result[key] = value
    return result


def _is_under(path: str, root: str) -> bool:
    return commonpath((realpath(path), realpath(root))) == realpath(root)


class JsonLineResultMapper:
    """Map one strict, generic JSON checkpoint record into a command result."""

    def __init__(self, *, checkpoint_directory: str, artifact_kind: str) -> None:
        if not isabs(checkpoint_directory) or not artifact_kind:
            raise ValueError("checkpoint_directory and artifact_kind must be approved values.")
        self._checkpoint_directory = checkpoint_directory
        self._artifact_kind = artifact_kind

    def result_for(self, plan: CommandPlan, captured_output: bytes) -> CommandResult:
        try:
            record = json.loads(captured_output.decode("utf-8"), object_pairs_hook=_strict_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ResultMappingError) as exc:
            raise ResultMappingError("The captured output must be one strict UTF-8 JSON record.") from exc
        if not isinstance(record, dict) or frozenset(record) != _FIELDS:
            raise ResultMappingError("The checkpoint result has missing or unapproved fields.")
        checkpoint_id = record["checkpoint_id"]
        checkpoint_path = record["checkpoint_path"]
        checksum = record["checkpoint_sha256"]
        artifact_kind = record["artifact_kind"]
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ResultMappingError("checkpoint_id must be non-blank text.")
        if (
            not isinstance(checkpoint_path, str)
            or not isabs(checkpoint_path)
            or not _is_under(checkpoint_path, self._checkpoint_directory)
            or realpath(checkpoint_path) == realpath(self._checkpoint_directory)
        ):
            raise ResultMappingError("checkpoint_path must be inside the approved checkpoint directory.")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in _LOWER_HEX for character in checksum)
        ):
            raise ResultMappingError("checkpoint_sha256 must be a lowercase SHA-256 checksum.")
        if artifact_kind != self._artifact_kind:
            raise ResultMappingError("artifact_kind does not match the approved training artifact.")
        return CommandResult(
            exit_code=0,
            captured_output_bytes=len(captured_output),
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checksum,
            artifact_kind=artifact_kind,
        )
