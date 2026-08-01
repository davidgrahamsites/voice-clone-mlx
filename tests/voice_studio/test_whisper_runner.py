"""Test the local MLX Whisper runtime seam without loading MLX or audio."""

import json
import subprocess
from pathlib import Path

import pytest

from voiceclonemlx.alignment.whisper_plan import WhisperPlan
from voiceclonemlx.alignment.whisper_runner import WhisperRunnerError, run_whisper_plan


def plan() -> WhisperPlan:
    return WhisperPlan(
        argv=("/local/python", "-m", "mlx_whisper", "sample.wav"),
        audio_path="/local/sample.wav",
        output_dir="/local/output",
        model_path="/local/model",
        language="en",
    )


def test_runs_the_plan_unchanged_and_emits_the_existing_manifest():
    seen = []

    def execute(argv, **kwargs):
        seen.append((argv, kwargs))

    def read_output(_plan):
        return json.dumps(
            {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello."}]}
        )

    manifest = run_whisper_plan(
        plan(),
        master_audio="sessions/one/master.wav",
        transcriber_version="0.4.1",
        executor=execute,
        output_reader=read_output,
    )

    assert seen == [
        (
            plan().argv,
            {
                "check": True,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": 120.0,
            },
        )
    ]
    row = json.loads(manifest)
    assert row == {
        "end_s": 1.0,
        "language": "en",
        "master_audio": "sessions/one/master.wav",
        "schema_version": "1",
        "segment_id": "master-0000",
        "start_s": 0.0,
        "text": "Hello.",
        "transcriber": "mlx_whisper",
        "transcriber_version": "0.4.1",
    }


def test_wraps_an_output_read_failure_in_a_typed_error():
    def read_output(_plan):
        raise OSError("output vanished")

    with pytest.raises(WhisperRunnerError, match="output"):
        run_whisper_plan(
            plan(),
            master_audio="sessions/one/master.wav",
            transcriber_version="0.4.1",
            executor=lambda *_args, **_kwargs: None,
            output_reader=read_output,
        )


def test_command_failure_does_not_read_or_fallback_to_another_backend():
    def execute(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, plan().argv)

    with pytest.raises(WhisperRunnerError, match="command failed"):
        run_whisper_plan(
            plan(),
            master_audio="sessions/one/master.wav",
            transcriber_version="0.4.1",
            executor=execute,
            output_reader=lambda _plan: pytest.fail("must not read failed output"),
        )


def test_malformed_backend_output_is_a_typed_failure():
    with pytest.raises(WhisperRunnerError, match="manifest"):
        run_whisper_plan(
            plan(),
            master_audio="sessions/one/master.wav",
            transcriber_version="0.4.1",
            executor=lambda *_args, **_kwargs: None,
            output_reader=lambda _plan: "not JSON",
        )


def test_default_reader_uses_the_local_mlx_json_filename(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    local_plan = WhisperPlan(
        argv=("/local/python", "-m", "mlx_whisper", "take.wav"),
        audio_path=str(tmp_path / "take.wav"),
        output_dir=str(output),
        model_path=str(tmp_path / "model"),
    )
    (output / "take.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "Hi"}]}),
        encoding="utf-8",
    )

    manifest = run_whisper_plan(
        local_plan,
        master_audio="sessions/one/master.wav",
        transcriber_version="0.4.1",
        executor=lambda *_args, **_kwargs: None,
    )

    assert json.loads(manifest)["text"] == "Hi"


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), True, "120"])
def test_refuses_an_unbounded_or_unusable_timeout(timeout_s):
    with pytest.raises(WhisperRunnerError, match="timeout_s"):
        run_whisper_plan(
            plan(),
            master_audio="sessions/one/master.wav",
            transcriber_version="0.4.1",
            executor=lambda *_args, **_kwargs: pytest.fail("must not execute"),
            output_reader=lambda _plan: pytest.fail("must not read"),
            timeout_s=timeout_s,
        )


def test_runner_never_imports_the_backend_package():
    source = Path(__file__).parents[2] / "src/voiceclonemlx/alignment/whisper_runner.py"

    assert "import mlx_whisper" not in source.read_text(encoding="utf-8")
