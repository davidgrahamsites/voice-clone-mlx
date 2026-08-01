"""Voice Reader logic: read a manifest, pair it with audio, play it back.

No UI toolkit is imported here so the whole thing is testable headlessly.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from voiceclonemlx.shared.integration_seam import emit_app_event

from . import APP_NAME

AUDIO_SUFFIXES = (".wav", ".m4a", ".mp3", ".aiff")

REQUIRED_FIELDS = ("id", "text", "style")

#: Anything that could make an id mean a path rather than a name.
_UNSAFE_IN_ID = ("/", "\\", "\x00")


@dataclass
class Take:
    """One manifest utterance and the audio file recorded for it, if any."""

    id: str
    text: str
    style: str
    audio_path: Optional[Path] = None

    @property
    def has_audio(self) -> bool:
        return self.audio_path is not None

    def label(self) -> str:
        marker = "▶" if self.has_audio else "—"
        return f"{marker}  {self.id}  [{self.style}]  {self.text}"


def load_manifest(manifest_path, audio_dir=None) -> List[Take]:
    """Read a JSONL script manifest and match each entry to an audio file.

    Audio is looked up as ``<audio_dir>/<utterance id><suffix>``; when
    ``audio_dir`` is omitted the manifest's own directory is used.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        ValueError: if a line is not valid JSON.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    audio_dir = Path(audio_dir) if audio_dir else manifest_path.parent

    takes = []
    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path}:{line_no} is not valid JSON: {exc}")

        _validate_record(record, manifest_path, line_no)

        takes.append(
            Take(
                id=record["id"],
                text=record["text"],
                style=record["style"],
                audio_path=find_audio(audio_dir, record["id"]),
            )
        )

    emit_app_event(
        APP_NAME,
        "manifest_loaded",
        {
            "manifest": str(manifest_path),
            "takes": len(takes),
            "with_audio": sum(1 for t in takes if t.has_audio),
        },
    )
    return takes


def _validate_record(record, manifest_path: Path, line_no: int) -> None:
    """Check one manifest row. A manifest is untrusted input.

    Raises:
        ValueError: If the row is not an object, a required field is missing or
            not a string, or the id is unsafe to use as a filename.
    """
    where = f"{manifest_path}:{line_no}"

    if not isinstance(record, dict):
        raise ValueError(f"{where} is not a JSON object")

    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"{where} is missing required field {field!r}")
        if not isinstance(record[field], str):
            raise ValueError(f"{where} field {field!r} must be a string")

    if not is_safe_utterance_id(record["id"]):
        raise ValueError(
            f"{where} field 'id' is not a usable file name: {record['id']!r}"
        )


def is_safe_utterance_id(utterance_id) -> bool:
    """Return True if an id can be used as a plain file name.

    Rejects anything empty or blank, containing a path separator or NUL, that
    is an absolute path, or that is a relative-path element such as `.`/`..`.
    """
    if not isinstance(utterance_id, str) or not utterance_id.strip():
        return False

    if any(token in utterance_id for token in _UNSAFE_IN_ID):
        return False

    if utterance_id in (".", ".."):
        return False

    return not Path(utterance_id).is_absolute()


def find_audio(audio_dir, utterance_id) -> Optional[Path]:
    """Return the audio file recorded for an utterance ID, or None.

    The id must be a safe file name, and the file it names must resolve to a
    real file inside ``audio_dir`` — a symlink pointing outside is refused.

    Returns the **resolved** path: the exact path that was checked for
    containment and confirmed to be a file, so a caller can never act on a
    path that would re-resolve somewhere else.
    """
    if not is_safe_utterance_id(utterance_id):
        return None

    audio_dir = Path(audio_dir)
    try:
        resolved_dir = audio_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    for suffix in AUDIO_SUFFIXES:
        candidate = audio_dir / f"{utterance_id}{suffix}"
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue

        try:
            resolved.relative_to(resolved_dir)
        except ValueError:
            continue

        if resolved.is_file():
            return resolved

    return None


class Player:
    """Plays one take at a time by shelling out to macOS ``afplay``."""

    def __init__(self, command: str = "afplay") -> None:
        self.command = command
        self._process: Optional[subprocess.Popen] = None

    def play(self, take: Take) -> str:
        """Start playback. Returns a status message for the UI."""
        if not take.has_audio:
            return f"No audio recorded for {take.id}."
        self.stop()
        try:
            self._process = subprocess.Popen([self.command, str(take.audio_path)])
        except OSError as exc:
            return f"Could not play {take.id}: {exc}"
        emit_app_event(APP_NAME, "playback_started", {"id": take.id})
        return f"Playing {take.id}."

    def stop(self) -> None:
        """Stop any current playback. Safe to call when nothing is playing."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None

    @property
    def is_playing(self) -> bool:
        return self._process is not None and self._process.poll() is None

