"""Test the Free Speech Mode candidate planner.

The planner is the one place that turns diarized free-speech candidates into
transcription plans. It owns no policy of its own: acceptance comes from
`overlap_gate.decide_clip` and plan construction from
`whisper_plan.build_whisper_plan`. What it does own is the bookkeeping —
every candidate stays in the result, in order, and a rejected one carries
`whisper_plan=None` rather than disappearing.

Safety invariants (no salvage, inertness, import purity) live in
`test_free_speech_plan_safety.py`.
"""

import pytest

from voiceclonemlx.alignment.free_speech_plan import (
    MAX_CANDIDATES,
    FreeSpeechCandidate,
    FreeSpeechPlanError,
    PlannedCandidate,
    plan_free_speech,
)
from voiceclonemlx.alignment.overlap_gate import ClipDecision, SpeakerTurn


@pytest.fixture
def local_assets(tmp_path):
    """An existing audio file, model directory, and output directory."""
    audio = tmp_path / "session.wav"
    audio.write_bytes(b"RIFF")
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    return {
        "audio_path": audio,
        "model_path": model,
        "output_dir": output,
    }


def candidate(start=0.0, end=5.0, turns=None, audio_path=None):
    return FreeSpeechCandidate(
        clip_start_s=start,
        clip_end_s=end,
        audio_path=str(audio_path) if audio_path is not None else "audio.wav",
        turns=tuple(turns if turns is not None else [SpeakerTurn(0.2, 4.8, "owner")]),
    )


def plan(candidates, local_assets, **kwargs):
    kwargs.setdefault("target_speaker", "owner")
    return plan_free_speech(
        candidates,
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
        **kwargs,
    )


# --- happy path ------------------------------------------------------------


def test_owner_only_candidate_gets_a_whisper_plan(local_assets):
    audio = local_assets["audio_path"]
    results = plan([candidate(audio_path=audio)], local_assets)

    assert len(results) == 1
    planned = results[0]
    assert planned.decision == ClipDecision.accept("target_only")
    assert planned.whisper_plan is not None
    assert planned.whisper_plan.audio_path == str(audio.resolve())
    assert planned.whisper_plan.model_path == str(
        local_assets["model_path"].resolve()
    )
    assert planned.whisper_plan.output_dir == str(
        local_assets["output_dir"].resolve()
    )


def test_language_is_passed_through_to_the_plan(local_assets):
    results = plan(
        [candidate(audio_path=local_assets["audio_path"])],
        local_assets,
        language="en",
    )

    assert results[0].whisper_plan.language == "en"


def test_language_defaults_to_unset(local_assets):
    results = plan([candidate(audio_path=local_assets["audio_path"])], local_assets)

    assert results[0].whisper_plan.language is None


def test_candidate_span_is_carried_through_unchanged(local_assets):
    results = plan(
        [candidate(start=1.5, end=9.25, audio_path=local_assets["audio_path"])],
        local_assets,
    )

    assert (results[0].clip_start_s, results[0].clip_end_s) == (1.5, 9.25)


def test_empty_candidate_list_plans_nothing(local_assets):
    assert plan([], local_assets) == ()


def test_result_is_a_tuple_of_planned_candidates(local_assets):
    results = plan([candidate(audio_path=local_assets["audio_path"])], local_assets)

    assert isinstance(results, tuple)
    assert all(isinstance(row, PlannedCandidate) for row in results)


# --- rejection retention ---------------------------------------------------


def test_overlapping_candidate_is_kept_with_no_plan(local_assets):
    results = plan(
        [
            candidate(
                audio_path=local_assets["audio_path"],
                turns=[
                    SpeakerTurn(0.2, 4.8, "owner"),
                    SpeakerTurn(2.0, 3.0, "guest", overlap=True),
                ],
            )
        ],
        local_assets,
    )

    assert len(results) == 1
    assert results[0].whisper_plan is None
    assert results[0].decision == ClipDecision.reject("overlap_detected")


def test_other_speaker_candidate_is_kept_with_no_plan(local_assets):
    results = plan(
        [
            candidate(
                audio_path=local_assets["audio_path"],
                turns=[
                    SpeakerTurn(0.2, 3.0, "owner"),
                    SpeakerTurn(3.2, 4.8, "guest"),
                ],
            )
        ],
        local_assets,
    )

    assert results[0].whisper_plan is None
    assert results[0].decision == ClipDecision.reject("other_speaker_detected")


def test_absent_target_candidate_is_kept_with_no_plan(local_assets):
    results = plan(
        [
            candidate(
                audio_path=local_assets["audio_path"],
                turns=[SpeakerTurn(0.2, 4.8, "guest")],
            )
        ],
        local_assets,
    )

    assert results[0].whisper_plan is None
    assert results[0].decision == ClipDecision.reject("target_not_detected")


def test_rejected_candidates_keep_their_input_order(local_assets):
    audio = local_assets["audio_path"]
    accepted = candidate(start=0.0, end=5.0, audio_path=audio)
    rejected = candidate(
        start=5.0,
        end=10.0,
        audio_path=audio,
        turns=[SpeakerTurn(5.2, 9.8, "guest")],
    )

    results = plan([rejected, accepted, rejected], local_assets)

    assert [row.clip_start_s for row in results] == [5.0, 0.0, 5.0]
    assert [row.whisper_plan is None for row in results] == [True, False, True]


def test_nothing_is_dropped_for_a_mixed_batch(local_assets):
    audio = local_assets["audio_path"]
    candidates = [
        candidate(start=float(i), end=float(i) + 1.0, audio_path=audio)
        if i % 2 == 0
        else candidate(
            start=float(i),
            end=float(i) + 1.0,
            audio_path=audio,
            turns=[SpeakerTurn(float(i) + 0.1, float(i) + 0.9, "guest")],
        )
        for i in range(10)
    ]

    results = plan(candidates, local_assets)

    assert len(results) == len(candidates)


# --- candidate cap ---------------------------------------------------------


def test_the_cap_is_one_hundred_and_forty_four():
    assert MAX_CANDIDATES == 144


def test_exactly_the_cap_is_planned(local_assets):
    audio = local_assets["audio_path"]
    candidates = [
        candidate(start=float(i), end=float(i) + 0.5, audio_path=audio)
        for i in range(MAX_CANDIDATES)
    ]

    assert len(plan(candidates, local_assets)) == MAX_CANDIDATES


def test_one_over_the_cap_is_refused(local_assets):
    audio = local_assets["audio_path"]
    candidates = [
        candidate(start=float(i), end=float(i) + 0.5, audio_path=audio)
        for i in range(MAX_CANDIDATES + 1)
    ]

    with pytest.raises(FreeSpeechPlanError) as excinfo:
        plan(candidates, local_assets)

    assert "144" in str(excinfo.value)


# --- immutability ----------------------------------------------------------


def test_planned_candidate_is_frozen(local_assets):
    results = plan([candidate(audio_path=local_assets["audio_path"])], local_assets)

    with pytest.raises(Exception):
        results[0].whisper_plan = None


def test_candidate_is_frozen():
    with pytest.raises(Exception):
        candidate().clip_start_s = 99.0


def test_input_sequence_is_not_mutated(local_assets):
    audio = local_assets["audio_path"]
    candidates = [
        candidate(start=1.0, end=2.0, audio_path=audio),
        candidate(start=3.0, end=4.0, audio_path=audio),
    ]
    before = list(candidates)

    plan(candidates, local_assets)

    assert candidates == before


def test_candidate_turns_are_not_mutated(local_assets):
    turns = [SpeakerTurn(0.2, 4.8, "owner"), SpeakerTurn(4.9, 5.0, "owner")]
    subject = candidate(audio_path=local_assets["audio_path"], turns=turns)

    plan([subject], local_assets)

    assert subject.turns == tuple(turns)


# --- input validation ------------------------------------------------------


def test_non_candidate_entries_are_refused(local_assets):
    with pytest.raises(FreeSpeechPlanError):
        plan([{"clip_start_s": 0.0}], local_assets)


def test_missing_model_path_is_refused(local_assets, tmp_path):
    with pytest.raises(Exception):
        plan_free_speech(
            [candidate(audio_path=local_assets["audio_path"])],
            target_speaker="owner",
            model_path=tmp_path / "nope",
            output_dir=local_assets["output_dir"],
        )


def test_blank_target_speaker_is_refused(local_assets):
    with pytest.raises(FreeSpeechPlanError):
        plan(
            [candidate(audio_path=local_assets["audio_path"])],
            local_assets,
            target_speaker="  ",
        )
