"""Test order-only alignment of transcript segments to expected utterances.

Matching, text handling, input validation, the result contract, and purity.
How disagreement is recorded — count mismatches, window boundaries, and the
reason vocabulary — lives in `test_script_aligner_mismatch.py`.

Pure: no audio, no model, no filesystem, no clock. Inputs are the existing
contracts — a `StyleWindow`, a `Transcript`, and expected utterances — and the
output is `AlignmentRow`s, all of them `pending`.
"""

import dataclasses
from pathlib import Path

import pytest

from voiceclonemlx.alignment import script_aligner
from voiceclonemlx.alignment.alignment_rows import REVIEW_PENDING
from voiceclonemlx.alignment.marker_windows import StyleWindow
from voiceclonemlx.alignment.script_aligner import (
    MISMATCH_TEXT,
    ScriptAlignerError,
    align_window,
)
from voiceclonemlx.alignment.whisper_json import Transcript, TranscriptSegment

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


class TestExactMatch:
    """The ordinary case: one segment per expected utterance."""

    def test_one_row_per_expected_utterance(self):
        rows = align(
            tr=transcript((11.0, 13.0, "First."), (14.0, 16.0, "Second.")),
            exp=expected(("N-1", "First."), ("N-2", "Second.")),
        )

        assert len(rows) == 2

    def test_rows_carry_the_expected_ids_in_order(self):
        rows = align(
            tr=transcript((11.0, 13.0, "a"), (14.0, 16.0, "b")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        assert [r.utterance_id for r in rows] == ["N-1", "N-2"]

    def test_row_times_come_from_the_matched_segment(self):
        rows = align(tr=transcript((11.5, 13.25, "a")), exp=expected(("N-1", "a")))

        assert rows[0].start_s == 11.5
        assert rows[0].end_s == 13.25

    def test_style_comes_from_the_window(self):
        rows = align(win=window(style="warm"))

        assert rows[0].style == "warm"

    def test_master_audio_is_recorded(self):
        assert align()[0].master_audio == MASTER

    def test_segment_ids_are_not_invented(self):
        """`whisper_json` has no segment ids, so none are fabricated."""
        assert align()[0].segment_ids == ()

    def test_matching_text_has_no_mismatch_reasons(self):
        assert align()[0].mismatch_reasons == ()

    def test_every_row_is_pending(self):
        rows = align(
            tr=transcript((11.0, 13.0, "a"), (14.0, 16.0, "zzz")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        assert all(r.review_state == REVIEW_PENDING for r in rows)

    def test_no_row_is_ever_attributed(self):
        rows = align()

        assert all(r.reviewed_by is None and r.reviewed_at is None for r in rows)


class TestTextPreservation:
    """Stored text is verbatim; only the comparison is normalized."""

    def test_expected_text_is_stored_exactly(self):
        rows = align(exp=expected(("N-1", "  Spaced  text.  ")))

        assert rows[0].expected_text == "  Spaced  text.  "

    def test_observed_text_is_stored_exactly(self):
        rows = align(
            tr=transcript((11.0, 13.0, "  loose  spacing ")),
            exp=expected(("N-1", "loose spacing")),
        )

        assert rows[0].observed_text == "  loose  spacing "

    def test_case_and_spacing_differences_are_not_a_mismatch(self):
        rows = align(
            tr=transcript((11.0, 13.0, "  FIRST   line. ")),
            exp=expected(("N-1", "First line.")),
        )

        assert MISMATCH_TEXT not in rows[0].mismatch_reasons

    def test_genuinely_different_text_is_a_mismatch(self):
        rows = align(
            tr=transcript((11.0, 13.0, "Something else entirely.")),
            exp=expected(("N-1", "First line.")),
        )

        assert MISMATCH_TEXT in rows[0].mismatch_reasons

    def test_a_text_mismatch_row_is_still_pending(self):
        """`needs_review` is pending with reasons, never a fourth state."""
        rows = align(
            tr=transcript((11.0, 13.0, "Different.")),
            exp=expected(("N-1", "First line.")),
        )

        assert rows[0].review_state == REVIEW_PENDING
        assert rows[0].mismatch_reasons


class TestInputValidation:
    """Bad inputs raise; they never produce a plausible row."""

    @pytest.mark.parametrize("value", [None, 42, "window", []])
    def test_bad_window_rejects(self, value):
        with pytest.raises(ScriptAlignerError, match="window"):
            align(win=value)

    @pytest.mark.parametrize("value", [None, 42, "transcript"])
    def test_bad_transcript_rejects(self, value):
        with pytest.raises(ScriptAlignerError, match="transcript"):
            align(tr=value)

    @pytest.mark.parametrize("value", [None, 42, "utterances"])
    def test_bad_expected_utterances_reject(self, value):
        with pytest.raises(ScriptAlignerError, match="expected"):
            align(exp=value)

    @pytest.mark.parametrize("field", ["utterance_id", "text"])
    def test_missing_utterance_field_rejects(self, field):
        item = {"utterance_id": "N-1", "text": "a"}
        del item[field]

        with pytest.raises(ScriptAlignerError, match=field):
            align(exp=[item])

    @pytest.mark.parametrize("value", ["", "   ", None, 7, True])
    def test_bad_utterance_id_rejects(self, value):
        with pytest.raises(ScriptAlignerError, match="utterance_id"):
            align(exp=[{"utterance_id": value, "text": "a"}])

    def test_utterance_objects_are_accepted(self):
        class Utterance:
            utterance_id = "N-1"
            text = "First line."

        rows = align(exp=[Utterance()])

        assert rows[0].utterance_id == "N-1"

    @pytest.mark.parametrize("value", ["", "   ", None, 7])
    def test_bad_master_audio_rejects(self, value):
        with pytest.raises(Exception):
            align_window(window(), transcript((11.0, 13.0, "a")),
                         expected(("N-1", "a")), master_audio=value)

    def test_input_sequence_is_not_mutated(self):
        items = expected(("N-1", "a"), ("N-2", "b"))
        before = [dict(i) for i in items]

        align(tr=transcript((11.0, 13.0, "a"), (14.0, 16.0, "b")), exp=items)

        assert items == before


class TestResultContract:
    """Rows are the existing immutable contract, not a new one."""

    def test_returns_a_tuple(self):
        assert isinstance(align(), tuple)

    def test_rows_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            align()[0].review_state = "accepted"

    def test_rows_serialize_through_the_existing_manifest_writer(self):
        from voiceclonemlx.alignment.alignment_rows import (
            parse_alignment_jsonl,
            rows_to_jsonl,
        )

        rows = align(
            tr=transcript((11.0, 13.0, "a"), (14.0, 16.0, "b")),
            exp=expected(("N-1", "a"), ("N-2", "b")),
        )

        assert parse_alignment_jsonl(rows_to_jsonl(rows)) == rows


class TestSeamIsPure:
    """No audio, model, clock, filesystem, or network."""

    def test_imports_nothing_dangerous(self):
        source = Path(script_aligner.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )
        imports = imports.replace("voiceclonemlx.", "")

        for banned in ("os", "pathlib", "datetime", "time", "random", "wave",
                       "numpy", "mlx", "urllib", "socket", "subprocess",
                       "tkinter"):
            assert banned not in imports

    def test_reuses_the_existing_contracts(self):
        """Row building and window/transcript types are not re-implemented."""
        source = Path(script_aligner.__file__).read_text(encoding="utf-8")

        assert "build_alignment_row" in source
        assert "class AlignmentRow" not in source
        assert "class StyleWindow" not in source
        assert "class TranscriptSegment" not in source

    def test_never_accepts_a_row(self):
        source = Path(script_aligner.__file__).read_text(encoding="utf-8")

        assert "accept_row" not in source
        assert "REVIEW_ACCEPTED" not in source

    def test_imports_no_app_or_ui(self):
        source = Path(script_aligner.__file__).read_text(encoding="utf-8")

        assert "studio_app" not in source
        assert "reader_app" not in source
