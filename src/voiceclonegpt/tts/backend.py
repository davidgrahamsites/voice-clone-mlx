"""What every TTS backend must be, and a stub that is one.

The seam is deliberately one method wide: given text and a reference clip,
write one audio file and say what was written. Loading models, choosing a
backend, batching, retrying, and telling anyone about the result all belong to
callers — this module holds no state between calls.

Its one in-package import is `synthesis.null_runtime`, whose silence the stub
reuses rather than re-implements; a test pins that to being the only one. No
third-party dependency, so the stub costs nothing to ship.
"""

import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol, runtime_checkable

from voiceclonegpt.synthesis.null_runtime import NullRuntime


class SynthesisError(Exception):
    """Synthesis was refused, or failed to produce usable audio."""


@dataclass(frozen=True)
class SynthesisResult:
    """What one `synthesize` call produced.

    Attributes:
        out_path: The audio file that was written.
        duration_s: Its playing time in seconds, measured from the audio
            actually written — not estimated after the fact.
        backend_name: Which backend produced it, for provenance in logs and
            manifests.
    """

    out_path: Path
    duration_s: float
    backend_name: str


@runtime_checkable
class TTSBackend(Protocol):
    """Turn one piece of text into one audio file."""

    def synthesize(self, text: str, ref_audio_path, out_path) -> SynthesisResult:
        """Write speech for `text` to `out_path`.

        Args:
            text: The line to speak. Non-empty.
            ref_audio_path: Reference clip for the voice being reproduced.
            out_path: Where to write the audio. Its directory must exist.

        Returns:
            A `SynthesisResult` describing the file written.

        Raises:
            SynthesisError: The request was refused or generation failed.
        """


class FakeBackend:
    """Writes a valid WAV of silence, so the seam runs before a model exists.

    It reproduces nobody's voice: every sample is zero and `IS_VOICE_MODEL` is
    False. Never flip that flag, and never present its output as a clone —
    silence must stay visibly a placeholder rather than a failed attempt.

    The silence itself comes from `synthesis.null_runtime.NullRuntime`, which
    already owns that job. Sample rate, speaking pace, and the duration clamp
    are entirely its business: this class does not read them, mirror them, or
    re-expose them, because a pointer at another module's constant is still a
    coupling — one that breaks the moment that module reorganizes its
    settings. Only the produced audio is depended on.

    This backend adds only what the file-level seam needs: writing the bytes
    to a path, and measuring what was written.
    """

    #: This backend does not reproduce anyone's voice. Never flip this to True.
    IS_VOICE_MODEL = False

    BACKEND_NAME = "fake"

    def __init__(self, stub=None):
        """Args:
        stub: Anything with `synthesize(handle, text) -> WAV bytes`. Defaults
            to `NullRuntime`. Only that one method is required — no attribute
            of the stub is read.
        """
        self._stub = stub or NullRuntime()

    def synthesize(self, text: str, ref_audio_path, out_path) -> SynthesisResult:
        """Write silence of a plausible length. The reference clip is not read.

        Raises:
            SynthesisError: If the text is empty, the output directory does not
                exist, or the file could not be written.
        """
        require_speakable_text(text)
        out_path = require_writable_target(out_path)

        audio = self._stub.synthesize(None, text)
        duration_s = measure_wav_seconds(audio)

        write_audio(audio, out_path)

        return SynthesisResult(
            out_path=out_path,
            duration_s=duration_s,
            backend_name=self.BACKEND_NAME,
        )


def require_speakable_text(text) -> str:
    """Return `text` if there is something to say, else refuse.

    Shared by every backend so an empty line is rejected before any model is
    asked to speak it.

    Raises:
        SynthesisError: If `text` is not a non-blank string.
    """
    if not isinstance(text, str) or not text.strip():
        raise SynthesisError(f"cannot synthesize empty text: {text!r}")
    return text


def require_writable_target(out_path) -> Path:
    """Return `out_path` as a Path whose directory already exists.

    The directory is never created here: inventing one hides a caller passing
    the wrong session folder, and this module owns no layout decisions.

    Raises:
        SynthesisError: If the path is unusable or its directory is missing.
    """
    try:
        out_path = Path(out_path)
    except TypeError as exc:
        raise SynthesisError(f"out_path is not a path: {out_path!r}") from exc

    if not out_path.parent.is_dir():
        raise SynthesisError(f"output directory does not exist: {out_path.parent}")

    return out_path


def write_audio(audio: bytes, out_path: Path) -> None:
    """Write finished audio bytes to `out_path`.

    Raises:
        SynthesisError: If the write fails.
    """
    try:
        out_path.write_bytes(audio)
    except OSError as exc:
        raise SynthesisError(f"could not write {out_path}: {exc}") from exc


def measure_wav_seconds(source) -> float:
    """Return the playing time of a WAV, read from the audio itself.

    Measuring beats estimating: a reported duration taken from the audio
    cannot disagree with it. Doubling as a validity check, backends call this
    on the bytes **before** writing them, so unusable audio is refused rather
    than left on disk.

    Args:
        source: Raw WAV bytes, or a path to a WAV file.

    Raises:
        SynthesisError: If it is not a readable WAV.
    """
    stream = BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)

    try:
        with wave.open(stream, "rb") as handle:
            frame_rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, wave.Error, EOFError) as exc:
        raise SynthesisError(f"not a readable WAV: {exc}") from exc

    if not frame_rate:
        raise SynthesisError("WAV declares no frame rate")

    return frames / frame_rate
