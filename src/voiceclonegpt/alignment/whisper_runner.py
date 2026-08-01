"""Run one local Whisper plan and return the established JSONL manifest."""

import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from .transcription_manifest import build_transcription_rows, rows_to_jsonl
from .whisper_json import WhisperJsonError, parse_whisper_json
from .whisper_plan import WhisperPlan

DEFAULT_TIMEOUT_S = 120.0


class WhisperRunnerError(RuntimeError):
    """A local Whisper plan could not produce a usable transcription."""


def _read_output(plan: WhisperPlan) -> str:
    """Read the JSON filename used by the local MLX Whisper command line tool."""
    path = Path(plan.output_dir) / f"{Path(plan.audio_path).stem}.json"
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise WhisperRunnerError(f"Whisper JSON output is unavailable: {path}") from exc


def _checked_timeout(timeout_s: Any) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise WhisperRunnerError(f"timeout_s must be a positive number: {timeout_s!r}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise WhisperRunnerError(f"timeout_s must be a positive number: {timeout_s!r}")
    return float(timeout_s)


def run_whisper_plan(
    plan: WhisperPlan,
    *,
    master_audio: str,
    transcriber_version: str,
    executor: Callable[..., Any] = subprocess.run,
    output_reader: Callable[[WhisperPlan], str] = _read_output,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Execute exactly one validated local plan and serialize its transcript.

    The plan supplies the local backend command. The runner imports no MLX
    package and offers no fallback or retry; test callers inject both process
    execution and output reading instead of loading a model or touching audio.
    """
    if not isinstance(plan, WhisperPlan):
        raise WhisperRunnerError(f"plan must be a WhisperPlan: {type(plan).__name__}")
    timeout = _checked_timeout(timeout_s)

    try:
        executor(
            plan.argv,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WhisperRunnerError("local Whisper command failed") from exc

    try:
        transcript = parse_whisper_json(output_reader(plan))
        rows = build_transcription_rows(
            transcript,
            master_audio=master_audio,
            transcriber="mlx_whisper",
            transcriber_version=transcriber_version,
            language=plan.language,
        )
    except (OSError, UnicodeError, WhisperJsonError, ValueError) as exc:
        raise WhisperRunnerError("Whisper output could not form a manifest") from exc

    return rows_to_jsonl(rows)
