"""Parse MLX Whisper's JSON output into a validated transcript.

One job: take whatever a runner captured — a mapping, JSON text, or raw bytes
— and turn it into sorted, validated segments. It reads no file, starts no
process, imports no `mlx`, and touches no network. Whoever ran Whisper hands
the payload in; this module only judges it.

**Text is preserved exactly.** Whisper's spacing is evidence about what was
said and how it was segmented, and downstream alignment compares it against
expected script text. Trimming or collapsing it here would quietly change that
comparison, so nothing is normalized — a segment is only rejected when its text
is blank.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Tuple


class WhisperJsonError(ValueError):
    """The payload is not a usable Whisper transcript."""


@dataclass(frozen=True)
class TranscriptSegment:
    """One transcribed span, with its text exactly as produced."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """A whole transcript: sorted, non-overlapping segments."""

    segments: Tuple[TranscriptSegment, ...] = ()


def _decoded(payload: Any) -> Any:
    """Return a parsed object from a mapping, JSON text, or bytes."""
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WhisperJsonError(
                f"payload is not valid UTF-8 JSON: {exc}"
            ) from exc

    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except ValueError as exc:
            raise WhisperJsonError(f"payload is not valid JSON: {exc}") from exc

    return payload


def _valid_time(value) -> bool:
    """True for a real, finite, non-negative number.

    `bool` is excluded on purpose: it is an `int` subclass, so `True` would
    otherwise pass as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _segment(raw: Any, index: int) -> TranscriptSegment:
    """Validate one raw segment, or raise."""
    # Any Mapping, not just `dict`: a caller who guards a payload with
    # MappingProxyType or a UserDict is being careful, not wrong.
    if not isinstance(raw, Mapping):
        raise WhisperJsonError(f"segment {index} is not a JSON object: {raw!r}")

    start = raw.get("start")
    end = raw.get("end")
    text = raw.get("text")

    if not _valid_time(start) or not _valid_time(end):
        raise WhisperJsonError(
            f"segment {index} has unusable times: start={start!r} end={end!r}"
        )

    if float(start) >= float(end):
        raise WhisperJsonError(
            f"segment {index} does not advance in time: "
            f"start={start!r} end={end!r}"
        )

    if not isinstance(text, str) or not text.strip():
        raise WhisperJsonError(f"segment {index} has no text: {text!r}")

    # `text` is stored unchanged: emptiness is judged on a stripped copy, but
    # the value kept is exactly what Whisper produced.
    return TranscriptSegment(start=float(start), end=float(end), text=text)


def parse_whisper_json(payload: Any) -> Transcript:
    """Parse and validate a Whisper JSON transcript.

    Args:
        payload: A mapping, JSON text, or UTF-8 bytes containing a `segments`
            list. Never mutated or reordered in place.

    Returns:
        A frozen `Transcript` with segments sorted by time. An empty
        `segments` list is valid and yields an empty transcript.

    Raises:
        WhisperJsonError: Malformed JSON or bytes, a payload that is not an
            object, a missing or non-list `segments`, a segment with unusable
            times or blank text, or two segments that overlap.
    """
    decoded = _decoded(payload)

    if not isinstance(decoded, Mapping):
        raise WhisperJsonError(
            f"payload is not a JSON object: {type(decoded).__name__}"
        )

    if "segments" not in decoded:
        raise WhisperJsonError("payload has no 'segments'")

    raw_segments = decoded["segments"]
    if isinstance(raw_segments, bool) or not isinstance(raw_segments, list):
        raise WhisperJsonError(
            f"'segments' must be a list: {type(raw_segments).__name__}"
        )

    if not raw_segments:
        return Transcript()

    segments = [_segment(raw, index) for index, raw in enumerate(raw_segments)]

    # `sorted` builds a new list: the caller's payload is left untouched.
    ordered = sorted(segments, key=lambda s: (s.start, s.end))

    for position, (earlier, later) in enumerate(zip(ordered, ordered[1:])):
        # Touching (`earlier.end == later.start`) is adjacency, not overlap.
        # Identical segments land here too: they overlap themselves.
        if later.start < earlier.end:
            raise WhisperJsonError(
                f"segments overlap at position {position + 1}: "
                f"[{earlier.start}, {earlier.end}] and "
                f"[{later.start}, {later.end}]"
            )

    return Transcript(segments=tuple(ordered))

