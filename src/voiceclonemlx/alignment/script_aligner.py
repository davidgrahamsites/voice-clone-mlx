"""Pair transcript segments with the utterances a reader was asked to say.

One job: for one style window, decide which observed segment corresponds to
which expected utterance, and record every disagreement as a reason on a
`pending` row. It matches by **order only** — the nth expected utterance pairs
with the nth segment inside the window.

Why order and not text similarity: a fuzzy match that guesses wrong attaches
real audio to the wrong words, and the resulting clip trains the model on a
mislabelled sentence. An unmatched row is reviewed by a human; a confidently
wrong match is not. The same reasoning that keeps `netural` the only accepted
marker typo applies here.

**Nothing here accepts a row.** Every row is `pending`. The lifecycle contract
calls an uncertain row `needs_review`; in `AlignmentManifest` v1 that is
*`pending` with `mismatch_reasons` populated* — there is no fourth state, and
this module cannot create one.

Pure: no audio, no model, no filesystem, no clock, no network. It builds rows
through `alignment_rows.build_alignment_row` and consumes `StyleWindow` and
`Transcript` as they are, duplicating neither.
"""

import re
from typing import Any, Iterable, Sequence, Tuple

from voiceclonemlx.alignment.alignment_rows import AlignmentRow, build_alignment_row
from voiceclonemlx.alignment.marker_windows import StyleWindow
from voiceclonemlx.alignment.whisper_json import Transcript

#: An expected utterance had no segment to pair with.
MISMATCH_EXPECTED_MISSING = "expected_missing"

#: More segments fell in the window than there were utterances to match.
MISMATCH_OBSERVED_SURPLUS = "observed_surplus"

#: The matched segment's words differ from the expected line.
MISMATCH_TEXT = "text_mismatch"

#: The matched segment crosses a window boundary instead of sitting inside it.
MISMATCH_OUTSIDE_WINDOW = "outside_window"

#: The complete vocabulary. Fixed and small on purpose: free-text reasons make
#: two manifests incomparable, and a reviewer cannot filter on them.
MISMATCH_REASONS = (
    MISMATCH_EXPECTED_MISSING,
    MISMATCH_OBSERVED_SURPLUS,
    MISMATCH_OUTSIDE_WINDOW,
    MISMATCH_TEXT,
)


class ScriptAlignerError(ValueError):
    """The inputs cannot be aligned."""


def _read(item: Any, key: str):
    """Read a field from a mapping or an object, or return None."""
    if hasattr(item, "get"):
        try:
            return item.get(key)
        except Exception:
            return None
    return getattr(item, key, None)


def _expected_pair(item: Any, index: int) -> Tuple[str, str]:
    """Return `(utterance_id, text)` from one expected utterance."""
    utterance_id = _read(item, "utterance_id")
    text = _read(item, "text")

    if utterance_id is None:
        raise ScriptAlignerError(
            f"expected utterance {index} has no utterance_id"
        )
    if isinstance(utterance_id, bool) or not isinstance(utterance_id, str) \
            or not utterance_id.strip():
        raise ScriptAlignerError(
            f"expected utterance {index} has an unusable utterance_id: "
            f"{utterance_id!r}"
        )

    if text is None:
        raise ScriptAlignerError(f"expected utterance {index} has no text")
    if isinstance(text, bool) or not isinstance(text, str) or not text.strip():
        raise ScriptAlignerError(
            f"expected utterance {index} has unusable text: {text!r}"
        )

    return utterance_id, text


def _comparable(text: str) -> str:
    """Normalize text for comparison only.

    Case and spacing differences are transcription noise, not disagreement
    about what was said. The row stores the original text unchanged; this value
    is never written anywhere.
    """
    return re.sub(r"\s+", " ", text).strip().casefold()


def align_window(
    window: StyleWindow,
    transcript: Transcript,
    expected_utterances: Iterable[Any],
    *,
    master_audio: str,
) -> Tuple[AlignmentRow, ...]:
    """Align one style window's transcript segments to its expected lines.

    Args:
        window: The `StyleWindow` whose span and style govern these rows.
        transcript: The whole `Transcript`; segments outside the window are
            ignored.
        expected_utterances: Mappings or objects with `utterance_id` and
            `text`, in the order they were to be read. Never mutated.
        master_audio: Relative reference to the session master, recorded on
            every row.

    Returns:
        One `pending` `AlignmentRow` per expected utterance, in order.

    Raises:
        ScriptAlignerError: Inputs of the wrong type, an unusable expected
            utterance, or segments in the window with no utterance to record
            them against.
    """
    if not isinstance(window, StyleWindow):
        raise ScriptAlignerError(f"window must be a StyleWindow: {window!r}")

    if not isinstance(transcript, Transcript):
        raise ScriptAlignerError(
            f"transcript must be a Transcript: {transcript!r}"
        )

    if isinstance(expected_utterances, (str, bytes)) or not isinstance(
        expected_utterances, (Sequence, Iterable)
    ):
        raise ScriptAlignerError(
            f"expected_utterances must be a sequence: {expected_utterances!r}"
        )

    try:
        items = list(expected_utterances)
    except TypeError as exc:
        raise ScriptAlignerError(
            f"expected_utterances must be iterable: {expected_utterances!r}"
        ) from exc

    pairs = [_expected_pair(item, index) for index, item in enumerate(items)]

    # Candidates are segments that touch the window at all. A segment that
    # merely straddles a boundary is still evidence — it is matched, and the
    # containment failure is recorded on its row rather than silently dropping
    # audio a reader actually produced.
    candidates = [
        segment
        for segment in transcript.segments
        if segment.end > window.start and segment.start < window.end
    ]

    if not pairs:
        if candidates:
            raise ScriptAlignerError(
                f"{len(candidates)} segment(s) fall in the {window.style!r} "
                f"window but no expected utterances were given; there is "
                f"nowhere to record them"
            )
        return ()

    rows = []
    for index, (utterance_id, expected_text) in enumerate(pairs):
        reasons = set()

        if index < len(candidates):
            segment = candidates[index]
            observed_text = segment.text
            start_s, end_s = segment.start, segment.end

            if segment.start < window.start or segment.end > window.end:
                reasons.add(MISMATCH_OUTSIDE_WINDOW)

            if _comparable(expected_text) != _comparable(observed_text):
                reasons.add(MISMATCH_TEXT)
        else:
            # Nothing was heard for this line. The window is the only span
            # known to be relevant, so the row points at it.
            observed_text = ""
            start_s, end_s = window.start, window.end
            reasons.add(MISMATCH_EXPECTED_MISSING)

        # A row is per *expected* utterance, so a surplus segment has no row of
        # its own; recording it on the final row keeps the evidence.
        is_last = index == len(pairs) - 1
        if is_last and len(candidates) > len(pairs):
            reasons.add(MISMATCH_OBSERVED_SURPLUS)

        rows.append(
            build_alignment_row(
                utterance_id=utterance_id,
                expected_text=expected_text,
                observed_text=observed_text,
                style=window.style,
                start_s=start_s,
                end_s=end_s,
                segment_ids=(),
                master_audio=master_audio,
                mismatch_reasons=tuple(sorted(reasons)),
            )
        )

    return tuple(rows)

