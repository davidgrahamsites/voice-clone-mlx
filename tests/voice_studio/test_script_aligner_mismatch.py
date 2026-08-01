"""Test how the script aligner records disagreement.

Count mismatches, window-boundary problems, and the fixed reason vocabulary —
every case where the transcript and the script do not line up. All of it lands
as `mismatch_reasons` on a row that stays `pending`; nothing here is ever an
acceptance.

Matching, text handling, validation, and purity live in
`test_script_aligner.py`.
"""

import pytest

from voiceclonegpt.alignment.marker_windows import StyleWindow
from voiceclonegpt.alignment.script_aligner import (
    MISMATCH_EXPECTED_MISSING,
    MISMATCH_OBSERVED_SURPLUS,
    MISMATCH_OUTSIDE_WINDOW,
    MISMATCH_REASONS,
    ScriptAlignerError,
    align_window,
)
from voiceclonegpt.alignment.whisper_json import Transcript, TranscriptSegment


MASTER = "01_recording/output/session-1.wav"


def window(start=10.0, end=60.0, style="neutral"):
    return StyleWindow(
        style=style,
        start=start,
        end=end,
        marker_text=f"This is the {style} reading.",
    )


def transcript(*spans):
    return Transcript(
        segments=tuple(
            TranscriptSegment(start=s, end=e, text=t) for s, e, t in spans
        )
    )


def expected(*pairs):
    return [{"utterance_id": uid, "text": text} for uid, text in pairs]


#: Distinguishes "argument omitted" from "argument explicitly None/empty",
#: which the validation tests rely on.
_UNSET = object()


def align(win=_UNSET, tr=_UNSET, exp=_UNSET, **kwargs):
    return align_window(
        window() if win is _UNSET else win,
        transcript((11.0, 13.0, "First line.")) if tr is _UNSET else tr,
        expected(("NEUTRAL-1", "First line.")) if exp is _UNSET else exp,
        master_audio=MASTER,
        **kwargs,
    )


class TestCountMismatch:
    """Too few or too many segments is data, not an exception."""

    def test_missing_segment_yields_a_row_with_empty_observed_text(self):
        rows = align(
            tr=transcript((11.0, 13.0, "a")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        assert len(rows) == 2
        assert rows[1].observed_text == ""

    def test_missing_segment_is_flagged(self):
        rows = align(
            tr=transcript((11.0, 13.0, "a")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        assert MISMATCH_EXPECTED_MISSING in rows[1].mismatch_reasons

    def test_missing_segment_row_spans_the_window(self):
        """Nothing was heard, so the only known span is the window itself."""
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((11.0, 13.0, "a")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        assert (rows[1].start_s, rows[1].end_s) == (10.0, 60.0)

    def test_no_segments_at_all_still_yields_every_expected_row(self):
        rows = align(
            tr=Transcript(), exp=expected(("N-1", "a"), ("N-2", "b"))
        )

        assert len(rows) == 2
        assert all(
            MISMATCH_EXPECTED_MISSING in r.mismatch_reasons for r in rows
        )

    def test_surplus_segments_are_flagged_on_the_last_row(self):
        """A row is per *expected* utterance, so a surplus segment has no row
        of its own; it is recorded rather than dropped."""
        rows = align(
            tr=transcript((11.0, 13.0, "a"), (14.0, 16.0, "b"), (17.0, 19.0, "c")),
            exp=expected(("N-1", "a")),
        )

        assert len(rows) == 1
        assert MISMATCH_OBSERVED_SURPLUS in rows[0].mismatch_reasons

    def test_surplus_with_no_expected_utterances_raises(self):
        """There is nowhere to record it, and silence would lose evidence."""
        with pytest.raises(ScriptAlignerError, match="expected"):
            align(tr=transcript((11.0, 13.0, "a")), exp=[])

    def test_no_expected_and_no_segments_is_empty(self):
        assert align(tr=Transcript(), exp=[]) == ()


class TestWindowBounds:
    """Only what falls in the window is considered."""

    def test_segments_before_the_window_are_ignored(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((1.0, 3.0, "before"), (11.0, 13.0, "a")),
            exp=expected(("N-1", "a")),
        )

        assert rows[0].observed_text == "a"

    def test_segments_after_the_window_are_ignored(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((11.0, 13.0, "a"), (70.0, 72.0, "after")),
            exp=expected(("N-1", "a")),
        )

        assert len(rows) == 1
        assert MISMATCH_OBSERVED_SURPLUS not in rows[0].mismatch_reasons

    def test_a_segment_straddling_the_start_is_flagged(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((9.0, 13.0, "a")),
            exp=expected(("N-1", "a")),
        )

        assert MISMATCH_OUTSIDE_WINDOW in rows[0].mismatch_reasons

    def test_a_segment_straddling_the_end_is_flagged(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((58.0, 65.0, "a")),
            exp=expected(("N-1", "a")),
        )

        assert MISMATCH_OUTSIDE_WINDOW in rows[0].mismatch_reasons

    def test_a_contained_segment_is_not_flagged(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((10.0, 60.0, "a")),
            exp=expected(("N-1", "a")),
        )

        assert MISMATCH_OUTSIDE_WINDOW not in rows[0].mismatch_reasons


class TestMismatchVocabulary:
    """The reason set is small, fixed, and documented."""

    def test_vocabulary_is_exactly_four_codes(self):
        assert set(MISMATCH_REASONS) == {
            "expected_missing",
            "observed_surplus",
            "text_mismatch",
            "outside_window",
        }

    def test_every_emitted_reason_is_in_the_vocabulary(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((9.0, 13.0, "different"), (14.0, 16.0, "surplus")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        for row in rows:
            for reason in row.mismatch_reasons:
                assert reason in MISMATCH_REASONS

    def test_reasons_are_unique_and_ordered(self):
        rows = align(
            win=window(10.0, 60.0),
            tr=transcript((9.0, 13.0, "different"), (14.0, 16.0, "x")),
            exp=expected(("N-1", "a")),
        )
        reasons = list(rows[0].mismatch_reasons)

        assert reasons == sorted(set(reasons))

