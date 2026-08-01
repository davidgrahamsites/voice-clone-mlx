"""Encode Reader audio into the two supported user-facing formats.

One job: keep output-format policy and local encoding behind a small seam. WAV
bytes are already the runtime contract and pass through unchanged. MP3 output
uses an installed local encoder (LAME first, then ffmpeg, then macOS
``afconvert``); it never downloads a codec or contacts a service.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

SUPPORTED_FORMATS = ("wav", "mp3")


class AudioExportError(Exception):
    """Audio could not be encoded into the requested format."""


def output_format_for_name(output_name: str) -> str:
    """Return the supported format named by an output filename."""
    if not isinstance(output_name, str):
        raise AudioExportError("output name must end in .wav or .mp3")

    suffix = Path(output_name).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise AudioExportError(
            f"output format must be .wav or .mp3, got {output_name!r}"
        )
    return suffix


def encode_audio(
    wav_bytes: bytes,
    output_format: str,
    *,
    encoder: Optional[Callable[[bytes, str], bytes]] = None,
) -> bytes:
    """Return audio in ``output_format`` from the runtime's WAV bytes.

    ``encoder`` is injectable for tests and alternate local encoders. The
    default MP3 path uses one local process and a temporary directory.
    """
    if not isinstance(wav_bytes, bytes) or not wav_bytes:
        raise AudioExportError("audio encoder requires non-empty WAV bytes")

    normalized = str(output_format).lower() if isinstance(output_format, str) else ""
    if normalized not in SUPPORTED_FORMATS:
        raise AudioExportError(f"unsupported output format: {output_format!r}")

    if normalized == "wav":
        return wav_bytes

    try:
        encoded = encoder(wav_bytes, normalized) if encoder else _encode_mp3(wav_bytes)
    except AudioExportError:
        raise
    except Exception as exc:
        raise AudioExportError(f"MP3 encoding failed: {exc}") from exc

    if not isinstance(encoded, bytes) or not encoded:
        raise AudioExportError("MP3 encoder returned no audio bytes")
    return encoded


def _encode_mp3(wav_bytes: bytes) -> bytes:
    """Use one installed local encoder, with deterministic fallback order."""
    executables = (
        ("lame", _run_lame),
        ("ffmpeg", _run_ffmpeg),
        ("afconvert", _run_afconvert),
    )
    missing = []
    for name, runner in executables:
        executable = shutil.which(name)
        if executable is None:
            missing.append(name)
            continue
        try:
            return runner(executable, wav_bytes)
        except AudioExportError:
            raise
        except OSError as exc:
            raise AudioExportError(f"{name} could not start: {exc}") from exc

    joined = ", ".join(missing)
    raise AudioExportError(
        "MP3 export needs one installed local encoder (tried "
        f"{joined or 'lame, ffmpeg, afconvert'})"
    )


def _run_lame(executable: str, wav_bytes: bytes) -> bytes:
    """Encode a temporary WAV with LAME."""
    return _run_file_encoder(
        executable,
        wav_bytes,
        ("--silent", "--noreplaygain"),
        output_suffix=".mp3",
    )


def _run_ffmpeg(executable: str, wav_bytes: bytes) -> bytes:
    """Encode a temporary WAV with ffmpeg's local MP3 encoder."""
    return _run_file_encoder(
        executable,
        wav_bytes,
        ("-hide_banner", "-loglevel", "error", "-codec:a", "libmp3lame"),
        output_suffix=".mp3",
        ffmpeg=True,
    )


def _run_afconvert(executable: str, wav_bytes: bytes) -> bytes:
    """Encode a temporary WAV with the macOS system converter."""
    return _run_file_encoder(
        executable,
        wav_bytes,
        ("-f", "MPG3", "-d", ".mp3"),
        output_suffix=".mp3",
    )


def _run_file_encoder(
    executable: str,
    wav_bytes: bytes,
    options: tuple[str, ...],
    *,
    output_suffix: str,
    ffmpeg: bool = False,
) -> bytes:
    """Run one local file encoder inside a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="voiceclonegpt-mp3-") as directory:
        root = Path(directory)
        source = root / "input.wav"
        target = root / f"output{output_suffix}"
        source.write_bytes(wav_bytes)

        if ffmpeg:
            command = [
                executable,
                *options,
                "-i",
                str(source),
                "-f",
                "mp3",
                str(target),
            ]
        else:
            command = [executable, *options, str(source), str(target)]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "encoder failed").strip()
            raise AudioExportError(f"{Path(executable).name}: {detail}")
        try:
            encoded = target.read_bytes()
        except OSError as exc:
            raise AudioExportError(
                f"{Path(executable).name} did not create an MP3 file"
            ) from exc
        if not encoded:
            raise AudioExportError(f"{Path(executable).name} created an empty MP3")
        return encoded
