"""Turn a timestamped transcript into style windows for later clipping.

One job: given transcript segments with times, find the spoken style markers
and report the time span each one governs. It decides *when* each style runs —
not what the audio contains, and not what happens to it afterwards.

Pure data in, windows out: no audio is opened, no model runs, nothing is read
from disk or the network. Standard library only, plus `marker_parser` for the
wording of a marker — which has exactly one home and is not re-implemented
here.

A window runs from the **end** of its marker to the **start** of the next one:
the marker announces the block and is never part of it. Gaps inside a window
are silence and are fine; overlapping segments are not, because a transcript
where two segments cover the same instant cannot be clipped unambiguously.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from voiceclonegpt.alignment.marker_parser import parse_style_marker

#: No marker was spoken anywhere in the transcript.
REASON_NO_MARKERS = "no_markers"


class MarkerWindowError(ValueError):
    """The transcript cannot be split into windows."""


@dataclass(frozen=True)
class StyleWindow:
    """One style block: the span its marker governs."""

    style: str
    start: float
    end: float
    marker_text: str


@dataclass(frozen=True)
class MarkerWindowResult:
    """Windows found in one transcript.

    `reason` is set only when there are no windows, so a caller never has to
    guess why an empty result came back.
    """

    windows: Tuple[StyleWindow, ...] = ()
    reason: Optional[str] = None


@dataclass(frozen=True)
class _Segment:
    """A validated transcript segment."""

    start: float
    end: float
    text: str


def _read(segment: Any, key: str):
    """Read a field from a mapping or an object, or return None."""
    if hasattr(segment, "get"):
        try:
            return segment.get(key)
        except Exception:
            return None
    return getattr(segment, key, None)


def _valid_time(value) -> bool:
    """True for a real, finite, non-negative number.

    `bool` is excluded on purpose: it is an `int` subclass, so `True` would
    otherwise pass as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _validated(segment: Any, index: int) -> _Segment:
    """Return a validated segment, or raise."""
    start = _read(segment, "start")
    end = _read(segment, "end")
    text = _read(segment, "text")

    if not _valid_time(start) or not _valid_time(end):
        raise MarkerWindowError(
            f"segment {index} has unusable times: start={start!r} end={end!r}"
        )

    if float(start) >= float(end):
        raise MarkerWindowError(
            f"segment {index} does not advance in time: "
            f"start={start!r} end={end!r}"
        )

    if not isinstance(text, str):
        raise MarkerWindowError(f"segment {index} has no text: {text!r}")

    return _Segment(start=float(start), end=float(end), text=text)


def _validate_recording_end(recording_end) -> Optional[float]:
    """Check the shape of an explicit recording end, or return None.

    Runs before anything else, including the empty-transcript shortcut: an
    unusable `recording_end` is a caller mistake whether or not there are
    segments, and silently returning `no_markers` would hide it.
    """
    if recording_end is None:
        return None

    if not _valid_time(recording_end):
        raise MarkerWindowError(
            f"recording_end must be a finite non-negative number: "
            f"{recording_end!r}"
        )

    return float(recording_end)


def _checked_against_transcript(
    recording_end: Optional[float], last_end: float
) -> Optional[float]:
    """Check an already-shape-valid recording end against the transcript."""
    if recording_end is None:
        return None

    if recording_end < last_end:
        raise MarkerWindowError(
            f"recording_end {recording_end!r} is before the transcript ends "
            f"({last_end})"
        )

    return recording_end


def split_style_windows(
    segments: Iterable[Any], recording_end=None
) -> MarkerWindowResult:
    """Split a transcript into one window per spoken style marker.

    Args:
        segments: Mappings or objects with `start`, `end`, and `text`. Never
            mutated or reordered in place.
        recording_end: Optional end of the recording. When given, the final
            window runs to it instead of to the last segment's end.

    Returns:
        A `MarkerWindowResult`. When no marker was spoken, `windows` is empty
        and `reason` is `REASON_NO_MARKERS`.

    Raises:
        MarkerWindowError: A segment is malformed, two segments overlap,
            `recording_end` is unusable or too early, or a marker has no
            content after it.
    """
    # Validated first: an unusable recording_end is a caller mistake whether
    # or not the transcript has segments, and the empty shortcut below would
    # otherwise swallow it.
    supplied_end = _validate_recording_end(recording_end)

    original = list(segments)

    if not original:
        return MarkerWindowResult(reason=REASON_NO_MARKERS)

    # `sorted` builds a new list: the caller's sequence is left untouched.
    ordered = sorted(
        (_validated(segment, index) for index, segment in enumerate(original)),
        key=lambda s: (s.start, s.end),
    )

    for earlier, later in zip(ordered, ordered[1:]):
        # Touching (`earlier.end == later.start`) is adjacency, not overlap.
        if later.start < earlier.end:
            raise MarkerWindowError(
                f"transcript segments overlap: [{earlier.start}, {earlier.end}] "
                f"and [{later.start}, {later.end}]"
            )

    last_end = ordered[-1].end
    explicit_end = _checked_against_transcript(supplied_end, last_end)
    final_end = last_end if explicit_end is None else explicit_end

    markers = []
    for segment in ordered:
        parsed = parse_style_marker(segment.text)
        if parsed is not None:
            markers.append((segment, parsed))

    if not markers:
        return MarkerWindowResult(reason=REASON_NO_MARKERS)

    windows = []
    for position, (segment, parsed) in enumerate(markers):
        is_last = position == len(markers) - 1
        end = final_end if is_last else markers[position + 1][0].start

        if end <= segment.end:
            raise MarkerWindowError(
                f"marker {parsed.style!r} at {segment.start}-{segment.end} has "
                f"no content after it (next boundary is {end})"
            )

        windows.append(
            StyleWindow(
                style=parsed.style,
                start=segment.end,
                end=end,
                marker_text=parsed.matched_text,
            )
        )

    return MarkerWindowResult(windows=tuple(windows))

