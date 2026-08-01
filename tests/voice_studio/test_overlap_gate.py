import math

import pytest

from voiceclonegpt.alignment.overlap_gate import (
    ACCEPT_STATUS,
    REJECT_STATUS,
    ClipDecision,
    OverlapGateError,
    SpeakerTurn,
    decide_clip,
)


def test_accepts_clip_containing_only_the_enrolled_speaker() -> None:
    decision = decide_clip(
        clip_start_s=0.0,
        clip_end_s=5.0,
        target_speaker="owner",
        turns=[SpeakerTurn(0.2, 4.8, "owner")],
    )

    assert decision == ClipDecision.accept("target_only")


def test_rejects_entire_clip_when_speakers_overlap() -> None:
    decision = decide_clip(
        clip_start_s=0.0,
        clip_end_s=5.0,
        target_speaker="owner",
        turns=[
            SpeakerTurn(0.2, 4.8, "owner"),
            SpeakerTurn(2.0, 3.0, "guest", overlap=True),
        ],
    )

    assert decision == ClipDecision.reject("overlap_detected")


def test_rejects_entire_clip_when_another_speaker_is_present() -> None:
    decision = decide_clip(
        clip_start_s=0.0,
        clip_end_s=8.0,
        target_speaker="owner",
        turns=[
            SpeakerTurn(0.2, 3.0, "owner"),
            SpeakerTurn(3.2, 7.8, "guest"),
        ],
    )

    assert decision == ClipDecision.reject("other_speaker_detected")


def test_ignores_turns_outside_candidate_clip() -> None:
    decision = decide_clip(
        clip_start_s=10.0,
        clip_end_s=15.0,
        target_speaker="owner",
        turns=[
            SpeakerTurn(0.0, 5.0, "guest"),
            SpeakerTurn(10.2, 14.8, "owner"),
        ],
    )

    assert decision == ClipDecision.accept("target_only")


# --- malformed input -------------------------------------------------------
#
# The gate is a safety decision, so a malformed span must raise rather than
# quietly miss an intersection. A NaN end, for instance, compares False against
# everything, which would make an overlapping guest turn look like it was
# outside the clip. Valid inputs keep the behavior above, unchanged.


def call(**overrides):
    kwargs = {
        "clip_start_s": 0.0,
        "clip_end_s": 5.0,
        "target_speaker": "owner",
        "turns": [SpeakerTurn(0.2, 4.8, "owner")],
    }
    kwargs.update(overrides)
    return decide_clip(**kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_clip_bounds(bad) -> None:
    with pytest.raises(OverlapGateError):
        call(clip_end_s=bad)
    with pytest.raises(OverlapGateError):
        call(clip_start_s=bad)


def test_rejects_negative_clip_bounds() -> None:
    with pytest.raises(OverlapGateError):
        call(clip_start_s=-1.0)


def test_rejects_a_clip_that_does_not_advance() -> None:
    with pytest.raises(OverlapGateError):
        call(clip_start_s=5.0, clip_end_s=5.0)
    with pytest.raises(OverlapGateError):
        call(clip_start_s=6.0, clip_end_s=5.0)


@pytest.mark.parametrize("bad", [True, "0.0", None, [], object()])
def test_rejects_non_numeric_clip_bounds(bad) -> None:
    with pytest.raises(OverlapGateError):
        call(clip_start_s=bad)


def test_accepts_integer_clip_bounds() -> None:
    assert call(clip_start_s=0, clip_end_s=5) == ClipDecision.accept("target_only")


@pytest.mark.parametrize("bad", ["", "   ", None, 7, True])
def test_rejects_a_missing_or_blank_target_speaker(bad) -> None:
    with pytest.raises(OverlapGateError):
        call(target_speaker=bad)


@pytest.mark.parametrize("bad", [None, "owner", SpeakerTurn(0.0, 1.0, "owner")])
def test_rejects_turns_that_are_not_a_sequence(bad) -> None:
    with pytest.raises(OverlapGateError):
        call(turns=bad)


def test_rejects_a_turn_that_is_not_a_speaker_turn() -> None:
    with pytest.raises(OverlapGateError):
        call(turns=[{"start_s": 0.0, "end_s": 1.0, "speaker_id": "owner"}])


def test_accepts_a_tuple_of_turns() -> None:
    assert call(turns=(SpeakerTurn(0.2, 4.8, "owner"),)) == ClipDecision.accept(
        "target_only"
    )


def test_accepts_no_turns_as_target_not_detected() -> None:
    assert call(turns=[]) == ClipDecision.reject("target_not_detected")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, "0.0", None, True])
def test_speaker_turn_rejects_malformed_times(bad) -> None:
    with pytest.raises(OverlapGateError):
        SpeakerTurn(bad, 5.0, "owner")


def test_speaker_turn_rejects_a_span_that_does_not_advance() -> None:
    with pytest.raises(OverlapGateError):
        SpeakerTurn(3.0, 3.0, "owner")


@pytest.mark.parametrize("bad", ["", "  ", None, 7])
def test_speaker_turn_rejects_a_blank_speaker_id(bad) -> None:
    with pytest.raises(OverlapGateError):
        SpeakerTurn(0.0, 1.0, bad)


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_speaker_turn_rejects_a_non_boolean_overlap_flag(bad) -> None:
    with pytest.raises(OverlapGateError):
        SpeakerTurn(0.0, 1.0, "owner", overlap=bad)


def test_speaker_turn_accepts_integer_times() -> None:
    turn = SpeakerTurn(0, 5, "owner")

    assert (turn.start_s, turn.end_s) == (0, 5)


def test_the_input_turn_list_is_not_mutated() -> None:
    turns = [SpeakerTurn(3.2, 4.8, "owner"), SpeakerTurn(0.2, 3.0, "owner")]
    before = list(turns)

    call(turns=turns)

    assert turns == before


def test_overlap_gate_error_is_a_value_error() -> None:
    assert issubclass(OverlapGateError, ValueError)


# --- decision vocabulary ---------------------------------------------------
#
# `accepted` exists so a caller never spells the status string itself. A
# second copy of "accept" elsewhere could drift from this one.


def test_accepted_is_true_only_for_an_acceptance() -> None:
    assert ClipDecision.accept("target_only").accepted is True
    assert ClipDecision.reject("overlap_detected").accepted is False


def test_the_status_constants_are_what_the_decisions_carry() -> None:
    assert ClipDecision.accept("r").status == ACCEPT_STATUS
    assert ClipDecision.reject("r").status == REJECT_STATUS
