"""Test the single-speaker acceptance gate for diarization output.

Pure data in, verdict out: no audio is decoded, no model runs, nothing is read
from disk. Segments are plain objects or mappings.
"""

import dataclasses
from pathlib import Path

import pytest

from voiceclonegpt.studio_app import audio_acceptance
from voiceclonegpt.studio_app.audio_acceptance import (
    REASON_EMPTY,
    REASON_MALFORMED,
    REASON_NON_TARGET_SPEAKER,
    REASON_OVERLAP,
    validate_single_speaker_segments,
)

TARGET = "SPEAKER_00"


def seg(start, end, speaker=TARGET):
    """A segment as a mapping — the shape a diarizer usually emits."""
    return {"start": start, "end": end, "speaker": speaker}


class SegmentObject:
    """A segment as an object, to prove attribute access works too."""

    def __init__(self, start, end, speaker=TARGET):
        self.start = start
        self.end = end
        self.speaker = speaker


def check(segments, target=TARGET):
    return validate_single_speaker_segments(segments, target_speaker=target)


class TestAcceptance:
    """Target-only, non-overlapping speech is accepted."""

    def test_single_segment_is_accepted(self):
        result = check([seg(0.0, 1.5)])

        assert result.accepted
        assert result.reasons == ()

    def test_gaps_between_segments_are_allowed(self):
        """Silence between utterances is normal, not a defect."""
        result = check([seg(0.0, 1.0), seg(5.0, 6.0), seg(30.0, 31.5)])

        assert result.accepted

    def test_touching_boundaries_are_allowed(self):
        """`end == next start` is adjacency, not overlap."""
        result = check([seg(0.0, 1.0), seg(1.0, 2.0), seg(2.0, 3.0)])

        assert result.accepted

    def test_integer_times_are_accepted(self):
        assert check([seg(0, 1), seg(2, 3)]).accepted

    def test_segment_objects_are_accepted(self):
        result = check([SegmentObject(0.0, 1.0), SegmentObject(2.0, 3.0)])

        assert result.accepted

    def test_zero_start_is_allowed(self):
        assert check([seg(0.0, 0.5)]).accepted


class TestNormalization:
    """Accepted output is sorted, normalized, and never the caller's objects."""

    def test_segments_are_returned_sorted(self):
        result = check([seg(5.0, 6.0), seg(0.0, 1.0), seg(2.0, 3.0)])

        assert [s.start for s in result.segments] == [0.0, 2.0, 5.0]

    def test_input_list_is_not_mutated(self):
        segments = [seg(5.0, 6.0), seg(0.0, 1.0)]
        before = [dict(s) for s in segments]

        check(segments)

        assert segments == before

    def test_input_order_is_preserved_in_the_caller_list(self):
        segments = [seg(9.0, 10.0), seg(1.0, 2.0)]

        check(segments)

        assert segments[0]["start"] == 9.0

    def test_normalized_segments_are_immutable(self):
        result = check([seg(0.0, 1.0)])

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.segments[0].start = 99.0

    def test_segments_is_a_tuple(self):
        assert isinstance(check([seg(0.0, 1.0)]).segments, tuple)

    def test_times_are_normalized_to_float(self):
        result = check([seg(0, 1)])

        assert isinstance(result.segments[0].start, float)
        assert isinstance(result.segments[0].end, float)

    def test_accepts_any_iterable(self):
        assert check(iter([seg(0.0, 1.0), seg(2.0, 3.0)])).accepted


class TestOverlapRejection:
    """One overlap condemns the whole clip, not just the offending pair."""

    def test_overlapping_target_segments_reject(self):
        result = check([seg(0.0, 2.0), seg(1.0, 3.0)])

        assert not result.accepted
        assert REASON_OVERLAP in result.reasons

    def test_overlap_rejects_even_when_supplied_out_of_order(self):
        result = check([seg(1.0, 3.0), seg(0.0, 2.0)])

        assert not result.accepted
        assert REASON_OVERLAP in result.reasons

    def test_fully_contained_segment_is_an_overlap(self):
        result = check([seg(0.0, 10.0), seg(2.0, 3.0)])

        assert not result.accepted
        assert REASON_OVERLAP in result.reasons

    def test_identical_segments_are_an_overlap(self):
        result = check([seg(0.0, 1.0), seg(0.0, 1.0)])

        assert not result.accepted
        assert REASON_OVERLAP in result.reasons

    def test_one_overlap_rejects_the_whole_clip(self):
        result = check(
            [seg(0.0, 1.0), seg(2.0, 3.0), seg(2.5, 4.0), seg(9.0, 10.0)]
        )

        assert not result.accepted
        assert result.segments == ()

    def test_overlap_between_different_speakers_rejects(self):
        result = check([seg(0.0, 2.0), seg(1.0, 3.0, "SPEAKER_01")])

        assert not result.accepted
        assert REASON_OVERLAP in result.reasons
        assert REASON_NON_TARGET_SPEAKER in result.reasons


class TestSpeakerRejection:
    """Any voice that is not the target condemns the clip."""

    def test_a_second_speaker_rejects(self):
        result = check([seg(0.0, 1.0), seg(2.0, 3.0, "SPEAKER_01")])

        assert not result.accepted
        assert REASON_NON_TARGET_SPEAKER in result.reasons

    def test_a_non_overlapping_second_speaker_still_rejects(self):
        """Source separation cannot rescue a mixed clip; it is out."""
        result = check([seg(0.0, 1.0), seg(50.0, 51.0, "GUEST")])

        assert not result.accepted
        assert result.segments == ()

    def test_speaker_match_is_exact(self):
        result = check([seg(0.0, 1.0, "speaker_00")])

        assert not result.accepted
        assert REASON_NON_TARGET_SPEAKER in result.reasons

    def test_whitespace_padded_speaker_does_not_match(self):
        result = check([seg(0.0, 1.0, " SPEAKER_00 ")])

        assert not result.accepted


class TestMalformedRejection:
    """Unusable numbers or missing fields reject before anything else."""

    def test_empty_input_rejects(self):
        result = check([])

        assert not result.accepted
        assert result.reasons == (REASON_EMPTY,)

    def test_end_before_start_rejects(self):
        result = check([seg(3.0, 1.0)])

        assert REASON_MALFORMED in result.reasons

    def test_zero_length_segment_rejects(self):
        result = check([seg(1.0, 1.0)])

        assert REASON_MALFORMED in result.reasons

    def test_negative_start_rejects(self):
        result = check([seg(-1.0, 1.0)])

        assert REASON_MALFORMED in result.reasons

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")]
    )
    def test_non_finite_times_reject(self, value):
        assert REASON_MALFORMED in check([seg(0.0, value)]).reasons

    @pytest.mark.parametrize("value", ["1.0", None, [], {}, True])
    def test_non_numeric_times_reject(self, value):
        assert REASON_MALFORMED in check([seg(value, 5.0)]).reasons

    @pytest.mark.parametrize("speaker", ["", "   ", None, 7])
    def test_bad_speaker_rejects(self, speaker):
        assert REASON_MALFORMED in check([seg(0.0, 1.0, speaker)]).reasons

    @pytest.mark.parametrize("field", ["start", "end", "speaker"])
    def test_missing_field_rejects(self, field):
        segment = seg(0.0, 1.0)
        del segment[field]

        assert REASON_MALFORMED in check([segment]).reasons

    def test_malformed_is_the_only_reason_reported(self):
        """Times that cannot be trusted make overlap unknowable, so the
        verdict says so rather than guessing at further reasons."""
        result = check([seg("bad", 1.0), seg(0.0, 2.0, "GUEST")])

        assert result.reasons == (REASON_MALFORMED,)

    def test_a_single_malformed_segment_rejects_the_clip(self):
        result = check([seg(0.0, 1.0), seg(2.0, 3.0), seg(9.0, 4.0)])

        assert not result.accepted
        assert result.segments == ()


class TestResultContract:
    """The verdict is immutable and carries no segments when rejected."""

    def test_result_is_frozen(self):
        result = check([seg(0.0, 1.0)])

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.accepted = False

    def test_reasons_is_a_tuple(self):
        assert isinstance(check([]).reasons, tuple)

    def test_rejected_result_carries_no_segments(self):
        assert check([seg(0.0, 1.0, "GUEST")]).segments == ()

    def test_accepted_result_has_no_reasons(self):
        assert check([seg(0.0, 1.0)]).reasons == ()

    def test_reasons_are_sorted_and_unique(self):
        result = check(
            [seg(0.0, 2.0, "GUEST"), seg(1.0, 3.0, "GUEST"), seg(1.5, 4.0)]
        )

        assert list(result.reasons) == sorted(set(result.reasons))

    def test_reason_codes_are_plain_strings(self):
        for code in (REASON_EMPTY, REASON_MALFORMED, REASON_NON_TARGET_SPEAKER,
                     REASON_OVERLAP):
            assert isinstance(code, str)


class TestTargetSpeakerArgument:
    """The target is the question, not the data being judged."""

    @pytest.mark.parametrize("target", ["", "   ", None, 7])
    def test_invalid_target_raises(self, target):
        with pytest.raises(ValueError, match="target_speaker"):
            check([seg(0.0, 1.0)], target=target)


class TestSeamIsPure:
    """No decoding, no model, no network, no filesystem."""

    def test_imports_nothing_dangerous(self):
        source = Path(audio_acceptance.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("wave", "audioop", "numpy", "soundfile", "librosa",
                       "mlx", "urllib", "requests", "socket", "subprocess",
                       "pathlib", "os"):
            assert banned not in imports

    def test_imports_nothing_from_the_app(self):
        source = Path(audio_acceptance.__file__).read_text(encoding="utf-8")

        assert "reader_app" not in source
        assert "tkinter" not in source
        assert "from .ui" not in source

