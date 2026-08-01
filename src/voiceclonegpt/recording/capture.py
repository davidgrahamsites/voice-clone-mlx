"""Persist one locally captured PCM take as a lossless WAV and session record.

The capture callable is the only device boundary.  It is invoked only by a
caller that has obtained an explicit recording action; this module neither
opens a microphone nor requests operating-system permission.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional
import wave


class CaptureError(ValueError):
    """A capture cannot be accepted as a local recording take."""


@dataclass(frozen=True)
class CapturedPcm:
    """Interleaved signed PCM frames supplied by a local device adapter."""

    frames: bytes
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


@dataclass(frozen=True)
class CapturedSession:
    """Paths for one append-only recording take and its JSON session record."""

    wav_path: Path
    manifest_path: Path


def _require_id(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureError(f"{field} must be a non-empty file-safe string")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise CaptureError(f"{field} must be a file name, not a path: {value!r}")
    return value


def _validate_pcm(audio: CapturedPcm) -> None:
    if not isinstance(audio, CapturedPcm):
        raise CaptureError("capture must return CapturedPcm")
    if not isinstance(audio.frames, bytes) or not audio.frames:
        raise CaptureError("capture produced no audio frames")
    if not isinstance(audio.sample_rate_hz, int) or audio.sample_rate_hz <= 0:
        raise CaptureError("sample_rate_hz must be a positive integer")
    if not isinstance(audio.channels, int) or audio.channels <= 0:
        raise CaptureError("channels must be a positive integer")
    if audio.sample_width_bytes not in (1, 2, 3, 4):
        raise CaptureError("sample_width_bytes must be 1, 2, 3, or 4")
    frame_size = audio.channels * audio.sample_width_bytes
    if len(audio.frames) % frame_size:
        raise CaptureError("audio frames do not align to the supplied format")


def capture_session(
    output_dir,
    *,
    session_id: str,
    voice_id: str,
    capture,
    take_id: str,
    prompt_manifest_id: Optional[str] = None,
    device_name: Optional[str] = None,
) -> CapturedSession:
    """Call one local capture port and persist its lossless master WAV.

    ``prompt_manifest_id`` is optional: a take with it is scripted, while a
    take without it is free speech.  Existing outputs are never replaced.
    """
    session_id = _require_id(session_id, "session_id")
    voice_id = _require_id(voice_id, "voice_id")
    take_id = _require_id(take_id, "take_id")
    if prompt_manifest_id is not None:
        prompt_manifest_id = _require_id(prompt_manifest_id, "prompt_manifest_id")
    if device_name is not None:
        if not isinstance(device_name, str) or not device_name.strip():
            raise CaptureError("device_name must be a non-empty string or None")
    if not callable(capture):
        raise CaptureError("capture must be callable")

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise CaptureError(f"output_dir must be an existing directory: {output_dir}")
    wav_path = output_dir / f"{take_id}.wav"
    manifest_path = output_dir / f"{take_id}.json"
    if wav_path.exists() or manifest_path.exists():
        raise CaptureError(f"take {take_id!r} already exists; recordings are append-only")

    try:
        audio = capture()
    except Exception as exc:
        raise CaptureError(f"capture failed: {exc}") from exc
    _validate_pcm(audio)

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(audio.sample_width_bytes)
        wav.setframerate(audio.sample_rate_hz)
        wav.writeframes(audio.frames)

    frames = len(audio.frames) // (audio.channels * audio.sample_width_bytes)
    manifest = {
        "schema_version": "1.0.0",
        "session_id": session_id,
        "voice_id": voice_id,
        "prompt_manifest_id": prompt_manifest_id,
        "device_name": device_name,
        "status": "captured",
        "take_id": take_id,
        "master_audio": wav_path.name,
        "masters": [{
            "path": wav_path.name,
            "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            "sample_rate_hz": audio.sample_rate_hz,
            "channels": audio.channels,
            "bit_depth": audio.sample_width_bytes * 8,
            "duration_s": frames / audio.sample_rate_hz,
        }],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CapturedSession(wav_path=wav_path, manifest_path=manifest_path)
