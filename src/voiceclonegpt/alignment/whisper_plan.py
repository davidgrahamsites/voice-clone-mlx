"""Describe the command that would transcribe a session with MLX Whisper.

One job: build an argv. It starts no process, imports no `mlx`, writes no
file, and touches no network. A separate runner — which does not exist yet —
would execute the plan. Keeping the description separate means the argument
rules can be reviewed and tested without a model, a GPU, or an audio file of
any real size.

The model path is **required and must already exist locally**. That is the
whole point: `mlx_whisper` downloads a model when it is given a hub id or
nothing at all, and a transcription seam that silently pulls gigabytes from
the network is not local-first.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

#: The interpreter the plan will be run with.
#:
#: `sys.executable`, not a bare `"python"`: a bare name resolves through PATH
#: and can easily land on a system interpreter that has no `mlx_whisper`
#: installed, which fails at run time with a confusing error. Using the
#: running interpreter targets the environment that was actually set up.
PYTHON_ARGV0 = sys.executable

MODULE_NAME = "mlx_whisper"

OUTPUT_FORMAT = "json"

#: Anything containing this is a locator for somewhere else.
REMOTE_MARKERS = ("://",)


class WhisperPlanError(ValueError):
    """The transcription command could not be described."""


@dataclass(frozen=True)
class WhisperPlan:
    """An inert description of one transcription command."""

    argv: Tuple[str, ...]
    audio_path: str
    output_dir: str
    model_path: str
    language: Optional[str] = None


def _local_path(value, field: str) -> Path:
    """Coerce and resolve a local path, or refuse.

    Raises:
        WhisperPlanError: The value is not a path, names a remote locator, or
            does not exist.
    """
    if isinstance(value, str) and any(m in value for m in REMOTE_MARKERS):
        raise WhisperPlanError(
            f"{field} must be a local path, not a remote locator: {value!r}"
        )

    if isinstance(value, bool):
        raise WhisperPlanError(f"{field} must be a path: {value!r}")

    try:
        candidate = Path(value)
    except TypeError as exc:
        raise WhisperPlanError(f"{field} must be a path: {value!r}") from exc

    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WhisperPlanError(f"{field} does not exist: {value}") from exc


def _checked_language(language) -> Optional[str]:
    """Validate the optional language code."""
    if language is None:
        return None

    if isinstance(language, bool) or not isinstance(language, str):
        raise WhisperPlanError(f"language must be a string: {language!r}")

    if not language.strip():
        raise WhisperPlanError("language must not be blank")

    return language


def build_whisper_plan(
    audio_path, output_dir, model_path, *, language=None
) -> WhisperPlan:
    """Describe the MLX Whisper command for one audio file.

    Args:
        audio_path: Existing local audio **file**.
        output_dir: Existing local directory for the transcript.
        model_path: Existing local model directory or weights file. Required,
            so `mlx_whisper` never resolves a model from the network.
        language: Optional language code; omitted from argv when None.

    Returns:
        A frozen `WhisperPlan`. Nothing is executed and nothing is written.

    Raises:
        WhisperPlanError: A path is missing, remote, of the wrong kind, or not
            a path at all; or the language is blank or not a string.
    """
    audio = _local_path(audio_path, "audio_path")
    if not audio.is_file():
        raise WhisperPlanError(f"audio_path is not a file: {audio_path}")

    output = _local_path(output_dir, "output_dir")
    if not output.is_dir():
        raise WhisperPlanError(f"output_dir is not a directory: {output_dir}")

    model = _local_path(model_path, "model_path")

    checked_language = _checked_language(language)

    argv = [
        PYTHON_ARGV0,
        "-m",
        MODULE_NAME,
        "--model",
        str(model),
        "--output-dir",
        str(output),
        "--output-format",
        OUTPUT_FORMAT,
    ]

    if checked_language is not None:
        argv += ["--language", checked_language]

    argv.append(str(audio))

    return WhisperPlan(
        argv=tuple(argv),
        audio_path=str(audio),
        output_dir=str(output),
        model_path=str(model),
        language=checked_language,
    )

