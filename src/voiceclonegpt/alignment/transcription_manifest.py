"""Record an already-parsed transcript as a durable JSONL manifest.

One job: turn a `whisper_json.Transcript` into rows that can be written,
reviewed, and read back. It runs no transcriber, opens no file, starts no
process, imports no backend, and reads no clock — whoever transcribed hands the
parsed result in, and this module only records it.

**Provider-neutral.** The transcriber's name and version are caller-supplied
strings, recorded verbatim and never detected. Nothing here knows what MLX,
Whisper, or Qwen are; a different transcriber produces the same manifest shape.

**Segment rules are not restated.** That segments are ordered and
non-overlapping is `whisper_json`'s rule, so `parse_transcription_jsonl`
re-establishes it by rebuilding a payload and running `parse_whisper_json` over
it. Duplicating the comparison here would create a second home for a rule that
could then drift.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from voiceclonegpt.alignment.transcription_row_schema import (
    ID_DIGITS,
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    ROW_KEYS,
    TRANSCRIPTION_SCHEMA_VERSION,
    TranscriptionRowError,
    checked_keys,
    checked_label,
    checked_optional_label,
    checked_relative_path,
    checked_schema_version,
    checked_segment_id,
    checked_text,
    checked_time,
    derive_segment_id,
)
from voiceclonegpt.alignment.whisper_json import (
    Transcript,
    WhisperJsonError,
    parse_whisper_json,
)

__all__ = [
    "ID_DIGITS",
    "OPTIONAL_KEYS",
    "REQUIRED_KEYS",
    "ROW_KEYS",
    "TRANSCRIPTION_SCHEMA_VERSION",
    "TranscriptionRow",
    "TranscriptionRowError",
    "build_transcription_rows",
    "derive_segment_id",
    "parse_transcription_jsonl",
    "rows_to_jsonl",
]


@dataclass(frozen=True)
class TranscriptionRow:
    """One transcribed segment, as it is recorded.

    Every field is validated in `__post_init__`, so direct construction and
    parsing face exactly the checks the builder does. A rule only the builder
    enforced would be no rule at all — a hand-edited manifest would bypass it.
    """

    schema_version: str
    master_audio: str
    segment_id: str
    start_s: float
    end_s: float
    text: str
    transcriber: str
    transcriber_version: str
    language: Optional[str] = None

    def __post_init__(self) -> None:
        # `object.__setattr__` because the dataclass is frozen: normalization
        # (int seconds becoming floats) has to be written past the freeze.
        set_field = object.__setattr__

        set_field(self, "schema_version", checked_schema_version(self.schema_version))
        set_field(
            self, "master_audio", checked_relative_path(self.master_audio, "master_audio")
        )
        set_field(self, "segment_id", checked_segment_id(self.segment_id))
        set_field(self, "start_s", checked_time(self.start_s, "start_s"))
        set_field(self, "end_s", checked_time(self.end_s, "end_s"))
        set_field(self, "text", checked_text(self.text))
        set_field(self, "transcriber", checked_label(self.transcriber, "transcriber"))
        set_field(
            self,
            "transcriber_version",
            checked_label(self.transcriber_version, "transcriber_version"),
        )
        set_field(self, "language", checked_optional_label(self.language, "language"))

        if self.start_s >= self.end_s:
            raise TranscriptionRowError(
                f"start_s must precede end_s: {self.start_s!r} >= {self.end_s!r}"
            )


def build_transcription_rows(
    transcript,
    *,
    master_audio: str,
    transcriber: str,
    transcriber_version: str,
    language: Optional[str] = None,
) -> Tuple[TranscriptionRow, ...]:
    """Record a parsed transcript as manifest rows.

    Args:
        transcript: A `whisper_json.Transcript`, already validated. Required as
            that type rather than a raw list: accepting loose mappings would
            make this a second place where segment rules are decided.
        master_audio: Relative path of the recording the transcript came from.
        transcriber: What produced it, e.g. `"mlx_whisper"`. Recorded verbatim.
        transcriber_version: Its version, recorded verbatim.
        language: Optional language code, caller-supplied and never detected.

    Returns:
        One frozen row per segment, in transcript order, with positional
        segment ids.

    Raises:
        TranscriptionRowError: `transcript` is not a `Transcript`, or any field
            is unusable.
    """
    if not isinstance(transcript, Transcript):
        raise TranscriptionRowError(
            f"transcript must be a whisper_json.Transcript: "
            f"{type(transcript).__name__}"
        )

    checked_relative_path(master_audio, "master_audio")

    return tuple(
        TranscriptionRow(
            schema_version=TRANSCRIPTION_SCHEMA_VERSION,
            master_audio=master_audio,
            segment_id=derive_segment_id(master_audio, index),
            start_s=segment.start,
            end_s=segment.end,
            text=segment.text,
            transcriber=transcriber,
            transcriber_version=transcriber_version,
            language=language,
        )
        for index, segment in enumerate(transcript.segments)
    )


def _to_mapping(row: TranscriptionRow) -> dict:
    """One row as a plain mapping, omitting an unset language."""
    data = {
        "schema_version": row.schema_version,
        "master_audio": row.master_audio,
        "segment_id": row.segment_id,
        "start_s": row.start_s,
        "end_s": row.end_s,
        "text": row.text,
        "transcriber": row.transcriber,
        "transcriber_version": row.transcriber_version,
    }

    # Omitted rather than written as null, so absence is a fact about the run
    # rather than a value someone has to interpret.
    if row.language is not None:
        data["language"] = row.language

    return data


def rows_to_jsonl(rows: Sequence[TranscriptionRow]) -> str:
    """Serialize rows as JSONL: one JSON object per line.

    Keys are sorted and `ensure_ascii` is off, so a manifest diffs cleanly and
    stays readable in the language it was spoken in. Text is written exactly as
    stored; a newline inside it is escaped by the encoder and so cannot split a
    line.
    """
    lines = []
    for index, row in enumerate(rows):
        if not isinstance(row, TranscriptionRow):
            raise TranscriptionRowError(
                f"row {index} is not a TranscriptionRow: {type(row).__name__}"
            )
        lines.append(
            json.dumps(_to_mapping(row), sort_keys=True, ensure_ascii=False)
        )

    return "".join(line + "\n" for line in lines)


def _row_from_mapping(data: Any, line_no: int) -> TranscriptionRow:
    """Build one row from a parsed manifest line, or raise."""
    if not isinstance(data, dict):
        raise TranscriptionRowError(
            f"line {line_no} is not a JSON object: {type(data).__name__}"
        )

    checked_keys(data.keys(), line_no)

    try:
        return TranscriptionRow(
            schema_version=data["schema_version"],
            master_audio=data["master_audio"],
            segment_id=data["segment_id"],
            start_s=data["start_s"],
            end_s=data["end_s"],
            text=data["text"],
            transcriber=data["transcriber"],
            transcriber_version=data["transcriber_version"],
            language=data.get("language"),
        )
    except TranscriptionRowError as exc:
        raise TranscriptionRowError(f"line {line_no}: {exc}") from exc


def _check_positional_ids(rows: Sequence[TranscriptionRow]) -> None:
    """Every id must be the one its position derives.

    Ids are positional, so a parser can recompute them. This catches a manifest
    whose rows were renamed, reordered, or dropped — none of which a per-row
    check could see.
    """
    for index, row in enumerate(rows):
        expected = derive_segment_id(row.master_audio, index)
        if row.segment_id != expected:
            raise TranscriptionRowError(
                f"line {index + 1}: segment_id must be {expected!r} at this "
                f"position: {row.segment_id!r}"
            )


def _check_segment_rules(rows: Sequence[TranscriptionRow]) -> None:
    """Re-establish ordering and non-overlap through their one home.

    The rows are handed back to `parse_whisper_json` as a payload. If it
    reorders them, the manifest was not written in time order; if it refuses
    them, they overlap. Either way the judgement is `whisper_json`'s, made by
    running it rather than by restating what it does.
    """
    payload = {
        "segments": [
            {"start": row.start_s, "end": row.end_s, "text": row.text}
            for row in rows
        ]
    }

    try:
        transcript = parse_whisper_json(payload)
    except WhisperJsonError as exc:
        raise TranscriptionRowError(f"manifest segments are unusable: {exc}") from exc

    written = [(row.start_s, row.end_s) for row in rows]
    ordered = [(segment.start, segment.end) for segment in transcript.segments]

    if written != ordered:
        raise TranscriptionRowError(
            "manifest rows are not in time order; a written artifact is not "
            "silently reordered"
        )


def parse_transcription_jsonl(text: Any) -> Tuple[TranscriptionRow, ...]:
    """Parse a JSONL manifest back into validated rows.

    Args:
        text: The manifest. Blank lines are ignored, so a trailing newline or a
            hand-added gap is not an error.

    Returns:
        The rows, in file order. An empty manifest yields an empty tuple.

    Raises:
        TranscriptionRowError: The text is not a string, a line is not a JSON
            object, a line has the wrong key set, any field is unusable, an id
            is not the one its position derives, or the rows are out of order
            or overlapping.
    """
    if not isinstance(text, str):
        raise TranscriptionRowError(
            f"manifest must be text: {type(text).__name__}"
        )

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            data = json.loads(line)
        except ValueError as exc:
            raise TranscriptionRowError(f"line {line_no} is not JSON: {exc}") from exc

        rows.append(_row_from_mapping(data, line_no))

    _check_positional_ids(rows)
    _check_segment_rules(rows)

    return tuple(rows)
