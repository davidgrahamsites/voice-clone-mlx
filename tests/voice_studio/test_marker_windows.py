"""Test splitting a timestamped transcript into style windows.

Pure data in, windows out: no audio is opened, no model runs, nothing is read
from disk. Segments are the shape Whisper-like transcribers emit.
"""

import dataclasses
from pathlib import Path

import pytest

from voiceclonemlx.alignment import marker_windows
from voiceclonemlx.alignment.marker_windows import (
    REASON_NO_MARKERS,
    MarkerWindowError,
    StyleWindow,
    split_style_windows,
)


def seg(start, end, text):
    """A transcript segment as a mapping."""
    return {"start": start, "end": end, "text": text}


class SegmentObject:
    """A transcript segment as an object."""

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def marker(start, end, style="neutral"):
    return seg(start, end, f"This is the {style} reading.")


class TestWindows:
    """A marker opens a window that runs until the next marker."""

    def test_two_markers_produce_two_windows(self):
        result = split_style_windows(
            [
                marker(0.0, 2.0, "neutral"),
                seg(2.5, 10.0, "The neutral passage."),
                marker(11.0, 13.0, "warm"),
                seg(13.5, 20.0, "The warm passage."),
            ]
        )

        assert [w.style for w in result.windows] == ["neutral", "warm"]

    def test_window_starts_at_the_marker_end(self):
        result = split_style_windows(
            [marker(0.0, 2.0), seg(2.5, 9.0, "Words.")]
        )

        assert result.windows[0].start == 2.0

    def test_window_ends_at_the_next_marker_start(self):
        result = split_style_windows(
            [
                marker(0.0, 2.0, "neutral"),
                seg(3.0, 9.0, "Words."),
                marker(10.0, 12.0, "warm"),
                seg(13.0, 20.0, "More."),
            ]
        )

        assert result.windows[0].end == 10.0

    def test_last_window_ends_at_the_last_segment(self):
        result = split_style_windows(
            [marker(0.0, 2.0), seg(3.0, 9.5, "Words.")]
        )

        assert result.windows[-1].end == 9.5

    def test_recording_end_overrides_the_last_segment_end(self):
        result = split_style_windows(
            [marker(0.0, 2.0), seg(3.0, 9.0, "Words.")], recording_end=30.0
        )

        assert result.windows[-1].end == 30.0

    def test_gaps_between_segments_are_allowed(self):
        result = split_style_windows(
            [marker(0.0, 2.0), seg(50.0, 60.0, "After a long silence.")]
        )

        assert result.windows[0].end == 60.0

    def test_marker_text_is_recorded(self):
        result = split_style_windows(
            [marker(0.0, 2.0, "somber"), seg(3.0, 9.0, "Words.")]
        )

        assert result.windows[0].marker_text == "This is the somber reading."

    def test_segment_objects_are_accepted(self):
        result = split_style_windows(
            [
                SegmentObject(0.0, 2.0, "This is the warm reading."),
                SegmentObject(3.0, 9.0, "Words."),
            ]
        )

        assert result.windows[0].style == "warm"

    def test_typo_marker_normalizes_to_neutral(self):
        result = split_style_windows(
            [seg(0.0, 2.0, "This is the netural reading."), seg(3.0, 9.0, "W.")]
        )

        assert result.windows[0].style == "neutral"
        assert "netural" in result.windows[0].marker_text

    def test_three_consecutive_blocks(self):
        result = split_style_windows(
            [
                marker(0.0, 1.0, "neutral"),
                seg(1.5, 5.0, "a"),
                marker(6.0, 7.0, "warm"),
                seg(7.5, 11.0, "b"),
                marker(12.0, 13.0, "energetic"),
                seg(13.5, 18.0, "c"),
            ]
        )

        assert [(w.style, w.start, w.end) for w in result.windows] == [
            ("neutral", 1.0, 6.0),
            ("warm", 7.0, 12.0),
            ("energetic", 13.0, 18.0),
        ]

    def test_repeated_style_gets_its_own_window(self):
        result = split_style_windows(
            [
                marker(0.0, 1.0, "neutral"),
                seg(2.0, 5.0, "a"),
                marker(6.0, 7.0, "neutral"),
                seg(8.0, 11.0, "b"),
            ]
        )

        assert len(result.windows) == 2
        assert {w.style for w in result.windows} == {"neutral"}


class TestMarkerRemoval:
    """A marker announces a block; it is never part of one."""

    def test_marker_span_is_excluded_from_its_window(self):
        result = split_style_windows(
            [marker(0.0, 2.0), seg(3.0, 9.0, "Words.")]
        )

        assert result.windows[0].start >= 2.0

    def test_marker_span_is_excluded_from_the_previous_window(self):
        result = split_style_windows(
            [
                marker(0.0, 1.0, "neutral"),
                seg(2.0, 5.0, "a"),
                marker(6.0, 8.0, "warm"),
                seg(9.0, 12.0, "b"),
            ]
        )

        assert result.windows[0].end == 6.0

    def test_prose_that_merely_mentions_a_style_is_not_a_marker(self):
        """Only exact markers split blocks; prose must not."""
        result = split_style_windows(
            [
                marker(0.0, 1.0, "neutral"),
                seg(2.0, 5.0, "Okay, this is the warm reading, I think."),
                seg(6.0, 9.0, "More words."),
            ]
        )

        assert len(result.windows) == 1
        assert result.windows[0].end == 9.0


class TestNoMarkers:
    """No markers is a stated outcome, not a crash and not a guess."""

    def test_transcript_without_markers_reports_no_markers(self):
        result = split_style_windows(
            [seg(0.0, 5.0, "Just talking."), seg(6.0, 9.0, "Still talking.")]
        )

        assert result.windows == ()
        assert result.reason == REASON_NO_MARKERS

    def test_empty_transcript_reports_no_markers(self):
        result = split_style_windows([])

        assert result.windows == ()
        assert result.reason == REASON_NO_MARKERS

    def test_a_result_with_windows_has_no_reason(self):
        result = split_style_windows(
            [marker(0.0, 1.0), seg(2.0, 5.0, "a")]
        )

        assert result.reason is None


class TestOrdering:
    """Input order is irrelevant and the caller's list is never touched."""

    def test_unsorted_input_produces_sorted_windows(self):
        result = split_style_windows(
            [
                seg(13.5, 18.0, "c"),
                marker(0.0, 1.0, "neutral"),
                marker(12.0, 13.0, "energetic"),
                seg(1.5, 5.0, "a"),
            ]
        )

        assert [w.style for w in result.windows] == ["neutral", "energetic"]
        assert result.windows[0].start < result.windows[1].start

    def test_input_list_is_not_mutated(self):
        segments = [seg(9.0, 10.0, "b"), marker(0.0, 1.0)]
        before = [dict(s) for s in segments]

        split_style_windows(segments)

        assert segments == before

    def test_accepts_any_iterable(self):
        result = split_style_windows(
            iter([marker(0.0, 1.0), seg(2.0, 5.0, "a")])
        )

        assert len(result.windows) == 1


class TestRejection:
    """Unusable input raises, rather than producing a plausible window."""

    def test_overlapping_segments_reject(self):
        with pytest.raises(MarkerWindowError, match="overlap"):
            split_style_windows(
                [marker(0.0, 2.0), seg(1.0, 5.0, "overlapping prose")]
            )

    def test_touching_segments_are_allowed(self):
        result = split_style_windows([marker(0.0, 2.0), seg(2.0, 5.0, "a")])

        assert result.windows[0].end == 5.0

    @pytest.mark.parametrize(
        "bad", [(-1.0, 5.0), (5.0, 1.0), (1.0, 1.0)]
    )
    def test_bad_times_reject(self, bad):
        with pytest.raises(MarkerWindowError, match="segment"):
            split_style_windows([seg(bad[0], bad[1], "a")])

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), "1.0", None, [], True]
    )
    def test_non_finite_or_non_numeric_times_reject(self, value):
        with pytest.raises(MarkerWindowError, match="segment"):
            split_style_windows([seg(0.0, value, "a")])

    @pytest.mark.parametrize("text", [None, 42, [], {}])
    def test_non_string_text_rejects(self, text):
        with pytest.raises(MarkerWindowError, match="segment"):
            split_style_windows([seg(0.0, 1.0, text)])

    @pytest.mark.parametrize("field", ["start", "end", "text"])
    def test_missing_field_rejects(self, field):
        segment = seg(0.0, 1.0, "a")
        del segment[field]

        with pytest.raises(MarkerWindowError, match="segment"):
            split_style_windows([segment])

    def test_marker_with_no_content_after_it_rejects(self):
        """A trailing marker announces a block that was never recorded."""
        with pytest.raises(MarkerWindowError, match="no content"):
            split_style_windows([seg(0.0, 5.0, "a"), marker(6.0, 8.0)])

    def test_recording_end_before_the_last_marker_rejects(self):
        with pytest.raises(MarkerWindowError, match="recording_end"):
            split_style_windows(
                [marker(0.0, 2.0), seg(3.0, 9.0, "a")], recording_end=1.0
            )

    @pytest.mark.parametrize(
        "value", [-1.0, float("nan"), float("inf"), "30", [], True]
    )
    def test_bad_recording_end_rejects(self, value):
        with pytest.raises(MarkerWindowError, match="recording_end"):
            split_style_windows(
                [marker(0.0, 2.0), seg(3.0, 9.0, "a")], recording_end=value
            )

    @pytest.mark.parametrize(
        "value", [-1.0, -0.5, float("nan"), float("inf"), float("-inf"),
                  "30", None.__class__, [], {}, True, False]
    )
    def test_bad_recording_end_rejects_even_with_no_segments(self, value):
        """An empty transcript must not excuse an unusable recording_end."""
        with pytest.raises(MarkerWindowError, match="recording_end"):
            split_style_windows([], recording_end=value)

    @pytest.mark.parametrize(
        "value", [-1.0, float("nan"), float("inf"), "30", [], True]
    )
    def test_bad_recording_end_rejects_when_no_markers_are_present(self, value):
        with pytest.raises(MarkerWindowError, match="recording_end"):
            split_style_windows([seg(0.0, 5.0, "just talking")], recording_end=value)

    def test_valid_recording_end_with_no_segments_still_reports_no_markers(self):
        result = split_style_windows([], recording_end=30.0)

        assert result.reason == REASON_NO_MARKERS

    def test_recording_end_of_none_is_allowed(self):
        assert split_style_windows(
            [marker(0.0, 2.0), seg(3.0, 9.0, "a")], recording_end=None
        ).windows


class TestResultContract:
    """Windows are a small immutable value."""

    def test_result_is_frozen(self):
        result = split_style_windows([marker(0.0, 1.0), seg(2.0, 5.0, "a")])

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.reason = "changed"

    def test_windows_are_frozen(self):
        result = split_style_windows([marker(0.0, 1.0), seg(2.0, 5.0, "a")])

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.windows[0].start = 99.0

    def test_windows_is_a_tuple(self):
        result = split_style_windows([marker(0.0, 1.0), seg(2.0, 5.0, "a")])

        assert isinstance(result.windows, tuple)

    def test_times_are_floats(self):
        result = split_style_windows([marker(0, 1), seg(2, 5, "a")])

        assert isinstance(result.windows[0].start, float)
        assert isinstance(result.windows[0].end, float)

    def test_style_window_is_constructible_directly(self):
        window = StyleWindow(style="warm", start=0.0, end=1.0, marker_text="m")

        assert window.style == "warm"


class TestSeamIsPure:
    """No audio, model, network, or IO."""

    def test_imports_nothing_dangerous(self):
        source = Path(marker_windows.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )
        imports = imports.replace("voiceclonemlx.", "")

        for banned in ("os", "pathlib", "wave", "numpy", "soundfile", "mlx",
                       "urllib", "requests", "socket", "subprocess", "tkinter"):
            assert banned not in imports

    def test_reuses_the_existing_marker_parser(self):
        """Marker wording has one home; this module must not re-implement it."""
        source = Path(marker_windows.__file__).read_text(encoding="utf-8")

        assert "parse_style_marker" in source
        assert "This is the" not in source.split('"""')[-1]

    def test_imports_no_app_or_ui(self):
        source = Path(marker_windows.__file__).read_text(encoding="utf-8")

        assert "studio_app" not in source
        assert "reader_app" not in source
