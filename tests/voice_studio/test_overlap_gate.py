from voiceclonegpt.alignment.overlap_gate import (
    ClipDecision,
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
