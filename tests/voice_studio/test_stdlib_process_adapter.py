"""Fake-driven contract tests for the optional standard-library host adapter."""

from __future__ import annotations

import json

import pytest

from voiceclonegpt.training.cuda_command_provider import CommandPlan
from voiceclonegpt.training.stdlib_process_adapter import (
    JsonLineResultMapper,
    ResultMappingError,
    StdlibProcessFactory,
)


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


class FakePipe:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = list(chunks)

    def read(self, size: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


class FakePopen:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.stdout = FakePipe(chunks)
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode


def test_factory_starts_one_new_process_group_with_an_explicit_argv() -> None:
    calls = []
    fake = FakePopen((b"first", b"second"))

    def popen_factory(argv, **kwargs):
        calls.append((argv, kwargs))
        return fake

    factory = StdlibProcessFactory(popen_factory=popen_factory, group_signaler=lambda *_: None)

    process = factory.start(plan().argv, cwd=plan().working_directory, env={"CUDA_VISIBLE_DEVICES": "0"})

    assert calls[0][0] == ("python", "-m", "qwen_finetune")
    assert calls[0][1]["cwd"] == "/workspace/qwen"
    assert calls[0][1]["env"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["shell"] is False
    assert process.read_chunk(1) == b"first"
    assert process.read_chunk(1) == b"second"
    assert process.read_chunk(1) is None


def test_terminate_and_kill_signal_the_entire_process_group() -> None:
    signals = []
    fake = FakePopen(())
    fake.pid = 321
    factory = StdlibProcessFactory(
        popen_factory=lambda *_args, **_kwargs: fake,
        group_signaler=lambda pid, signal: signals.append((pid, signal)),
    )

    process = factory.start(plan().argv, cwd=plan().working_directory, env={})
    process.terminate()
    process.kill()

    assert signals[0][0] == 321
    assert signals[1][0] == 321
    assert signals[0][1] != signals[1][1]


def result_json(**overrides: object) -> bytes:
    fields: dict[str, object] = {
        "checkpoint_id": "checkpoint-001",
        "checkpoint_path": "/workspace/run/checkpoints/checkpoint-001",
        "checkpoint_sha256": "d" * 64,
        "artifact_kind": "fine_tuned_adapter",
    }
    fields.update(overrides)
    return json.dumps(fields, separators=(",", ":")).encode("utf-8")


def test_result_mapper_accepts_one_strict_checkpoint_record() -> None:
    output = result_json()
    mapper = JsonLineResultMapper(
        checkpoint_directory="/workspace/run/checkpoints",
        artifact_kind="fine_tuned_adapter",
    )

    result = mapper.result_for(plan(), output)

    assert result.checkpoint_id == "checkpoint-001"
    assert result.checkpoint_path == "/workspace/run/checkpoints/checkpoint-001"
    assert result.checkpoint_sha256 == "d" * 64
    assert result.artifact_kind == "fine_tuned_adapter"
    assert result.captured_output_bytes == len(output)


@pytest.mark.parametrize(
    "output",
    (
        b"not json",
        result_json(extra="field"),
        result_json(checkpoint_path="/workspace/elsewhere/checkpoint-001"),
        result_json(checkpoint_sha256="not-a-checksum"),
        result_json(artifact_kind="fine_tuned_full"),
        result_json() + b"\n" + result_json(),
    ),
)
def test_result_mapper_refuses_malformed_or_unapproved_checkpoint_records(output: bytes) -> None:
    mapper = JsonLineResultMapper(
        checkpoint_directory="/workspace/run/checkpoints",
        artifact_kind="fine_tuned_adapter",
    )

    with pytest.raises(ResultMappingError):
        mapper.result_for(plan(), output)
