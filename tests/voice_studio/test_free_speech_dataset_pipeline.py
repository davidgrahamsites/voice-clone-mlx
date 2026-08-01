"""Public integration tests for the free-speech dataset composition seam."""

import pytest

from voiceclonegpt.alignment.free_speech_plan import (
    FreeSpeechCandidate,
    PlannedCandidate,
)
from voiceclonegpt.alignment.overlap_gate import ClipDecision, SpeakerTurn
from voiceclonegpt.alignment.transcription_manifest import (
    TRANSCRIPTION_SCHEMA_VERSION,
    TranscriptionRow,
    rows_to_jsonl,
)
from voiceclonegpt.alignment.whisper_plan import WhisperPlan
from voiceclonegpt.dataset.clip_probe import ClipMeasurement
from voiceclonegpt.dataset.dataset_rows import parse_dataset_json
from voiceclonegpt.dataset.split_plan import plan_splits
from voiceclonegpt.studio_app.free_speech_dataset_pipeline import (
    DatasetPipelineError,
    HumanReview,
    RuntimeCandidate,
    SpeakerRuntimeEvidence,
    finalize_free_speech_dataset,
    prepare_free_speech_dataset,
)


SHA_A = "a" * 64


def session_manifest():
    return {
        "schema_version": "1.0.0",
        "session_id": "session-001",
        "status": "captured",
        "master_audio": "session.wav",
        "masters": [{"path": "session.wav", "sha256": SHA_A}],
    }


def owner_candidate():
    candidate = FreeSpeechCandidate(
        clip_start_s=3.0,
        clip_end_s=7.0,
        audio_path="/local/clips/candidate-001.wav",
        turns=(SpeakerTurn(3.0, 7.0, "owner"),),
    )
    return RuntimeCandidate(
        candidate_id="candidate-001",
        style="neutral",
        clip_path="clips/candidate-001.wav",
        candidate=candidate,
    )


def speaker_evidence(*candidates):
    return SpeakerRuntimeEvidence(
        session_id="session-001",
        master_audio="session.wav",
        master_sha256=SHA_A,
        runtime_id="speaker-runtime",
        runtime_version="1.0.0",
        target_speaker="owner",
        candidates=tuple(candidates),
    )


def accepted_plan(candidate):
    return PlannedCandidate(
        clip_start_s=candidate.clip_start_s,
        clip_end_s=candidate.clip_end_s,
        audio_path=candidate.audio_path,
        decision=ClipDecision.accept("target_only"),
        whisper_plan=WhisperPlan(
            argv=("python", "-m", "mlx_whisper"),
            audio_path=candidate.audio_path,
            output_dir="/local/out",
            model_path="/local/model",
        ),
    )


def transcript_jsonl(clip_path="clips/candidate-001.wav"):
    return rows_to_jsonl(
        (
            TranscriptionRow(
                schema_version=TRANSCRIPTION_SCHEMA_VERSION,
                master_audio=clip_path,
                segment_id="candidate-001-0000",
                start_s=0.0,
                end_s=4.0,
                text="A clean owner-only sentence.",
                transcriber="mlx_whisper",
                transcriber_version="0.4.0",
            ),
        )
    )


def test_prepare_transcribes_an_owner_only_candidate_for_human_review():
    runtime_candidate = owner_candidate()

    bundle = prepare_free_speech_dataset(
        session_manifest(),
        speaker_evidence(runtime_candidate),
        model_path="/local/model",
        output_dir="/local/out",
        transcriber_version="0.4.0",
        planner=lambda candidates, **_: tuple(accepted_plan(item) for item in candidates),
        runner=lambda plan, **_: transcript_jsonl(),
    )

    assert bundle.session_id == "session-001"
    assert bundle.master_audio == "session.wav"
    assert len(bundle.review_candidates) == 1
    assert bundle.review_candidates[0].candidate_id == "candidate-001"
    assert bundle.review_candidates[0].observed_text == "A clean owner-only sentence."
    assert bundle.rejected_candidates == ()


def test_finalize_admits_only_an_attributed_human_acceptance():
    runtime_candidate = owner_candidate()
    bundle = prepare_free_speech_dataset(
        session_manifest(),
        speaker_evidence(runtime_candidate),
        model_path="/local/model",
        output_dir="/local/out",
        transcriber_version="0.4.0",
        planner=lambda candidates, **_: tuple(accepted_plan(item) for item in candidates),
        runner=lambda plan, **_: transcript_jsonl(),
    )
    measurement = ClipMeasurement(
        clip_sha256="b" * 64,
        audio_properties={
            "sample_rate_hz": 24_000,
            "channels": 1,
            "bit_depth": 16,
            "duration_s": 4.0,
        },
        byte_size=192_044,
    )

    composition = finalize_free_speech_dataset(
        bundle,
        reviews={
            "candidate-001": HumanReview(
                accepted=True,
                actor="owner@example.test",
                at="2026-08-01T13:00:00Z",
            )
        },
        measurements={"candidate-001": measurement},
        split_plan=plan_splits(("session-001",)),
    )

    rows = parse_dataset_json(composition.dataset_json)
    assert len(rows) == 1
    assert rows[0].utterance_id == "candidate-001"
    assert rows[0].text == "A clean owner-only sentence."
    assert rows[0].accepted_by == "owner@example.test"
    assert rows[0].clip_decision_reason == "target_only"
    assert composition.rejected_candidates == ()


@pytest.mark.parametrize(
    ("decision", "turns"),
    (
        (
            ClipDecision.reject("overlap_detected"),
            (
                SpeakerTurn(3.0, 7.0, "owner"),
                SpeakerTurn(4.0, 5.0, "guest", overlap=True),
            ),
        ),
        (
            ClipDecision.reject("other_speaker_detected"),
            (SpeakerTurn(3.0, 5.0, "owner"), SpeakerTurn(5.0, 7.0, "guest")),
        ),
    ),
)
def test_speaker_rejection_is_whole_and_never_reaches_runner_or_row_builder(
    decision, turns
):
    source = owner_candidate()
    source = RuntimeCandidate(
        candidate_id=source.candidate_id,
        style=source.style,
        clip_path=source.clip_path,
        candidate=FreeSpeechCandidate(
            clip_start_s=3.0,
            clip_end_s=7.0,
            audio_path=source.candidate.audio_path,
            turns=turns,
        ),
    )

    def rejected_planner(candidates, **_):
        candidate = candidates[0]
        return (
            PlannedCandidate(
                clip_start_s=candidate.clip_start_s,
                clip_end_s=candidate.clip_end_s,
                audio_path=candidate.audio_path,
                decision=decision,
                whisper_plan=None,
            ),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected candidates must stop before this seam")

    bundle = prepare_free_speech_dataset(
        session_manifest(),
        speaker_evidence(source),
        model_path="/local/model",
        output_dir="/local/out",
        transcriber_version="0.4.0",
        planner=rejected_planner,
        runner=forbidden,
    )
    composition = finalize_free_speech_dataset(
        bundle,
        reviews={},
        measurements={},
        split_plan=plan_splits(("session-001",)),
        row_builder=forbidden,
    )

    assert parse_dataset_json(composition.dataset_json) == ()
    assert composition.rejected_candidates[0].reason == decision.reason
    assert (composition.rejected_candidates[0].clip_start_s,
            composition.rejected_candidates[0].clip_end_s) == (3.0, 7.0)


def test_mismatched_speaker_provenance_stops_before_planning():
    evidence = speaker_evidence(owner_candidate())
    evidence = SpeakerRuntimeEvidence(
        session_id="different-session",
        master_audio=evidence.master_audio,
        master_sha256=evidence.master_sha256,
        runtime_id=evidence.runtime_id,
        runtime_version=evidence.runtime_version,
        target_speaker=evidence.target_speaker,
        candidates=evidence.candidates,
    )

    with pytest.raises(DatasetPipelineError, match="provenance"):
        prepare_free_speech_dataset(
            session_manifest(),
            evidence,
            model_path="/local/model",
            output_dir="/local/out",
            transcriber_version="0.4.0",
            planner=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("planner must not run")
            ),
            runner=lambda *args, **kwargs: "",
        )


def test_nonportable_candidate_clip_path_stops_before_planning():
    source = owner_candidate()
    source = RuntimeCandidate(
        candidate_id=source.candidate_id,
        style=source.style,
        clip_path="/absolute/candidate.wav",
        candidate=source.candidate,
    )

    with pytest.raises(DatasetPipelineError, match="relative"):
        prepare_free_speech_dataset(
            session_manifest(),
            speaker_evidence(source),
            model_path="/local/model",
            output_dir="/local/out",
            transcriber_version="0.4.0",
            planner=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("planner must not run")
            ),
            runner=lambda *args, **kwargs: "",
        )


def test_finalize_refuses_missing_human_review():
    runtime_candidate = owner_candidate()
    bundle = prepare_free_speech_dataset(
        session_manifest(),
        speaker_evidence(runtime_candidate),
        model_path="/local/model",
        output_dir="/local/out",
        transcriber_version="0.4.0",
        planner=lambda candidates, **_: tuple(accepted_plan(item) for item in candidates),
        runner=lambda plan, **_: transcript_jsonl(),
    )

    with pytest.raises(DatasetPipelineError, match="human review"):
        finalize_free_speech_dataset(
            bundle,
            reviews={},
            measurements={},
            split_plan=plan_splits(("session-001",)),
        )


def test_human_rejection_is_retained_and_never_reaches_row_builder():
    runtime_candidate = owner_candidate()
    bundle = prepare_free_speech_dataset(
        session_manifest(),
        speaker_evidence(runtime_candidate),
        model_path="/local/model",
        output_dir="/local/out",
        transcriber_version="0.4.0",
        planner=lambda candidates, **_: tuple(accepted_plan(item) for item in candidates),
        runner=lambda plan, **_: transcript_jsonl(),
    )

    composition = finalize_free_speech_dataset(
        bundle,
        reviews={
            "candidate-001": HumanReview(
                accepted=False,
                actor="reviewer@example.test",
                at="2026-08-01T13:05:00Z",
                reason="transcript_not_usable",
            )
        },
        measurements={},
        split_plan=plan_splits(("session-001",)),
        row_builder=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("rejected review must not reach row builder")
        ),
    )

    assert parse_dataset_json(composition.dataset_json) == ()
    assert len(composition.rejected_candidates) == 1
    assert composition.rejected_candidates[0].stage == "human_review"
    assert composition.rejected_candidates[0].reason == "transcript_not_usable"


def test_unplanned_session_stops_before_row_builder():
    runtime_candidate = owner_candidate()
    bundle = prepare_free_speech_dataset(
        session_manifest(),
        speaker_evidence(runtime_candidate),
        model_path="/local/model",
        output_dir="/local/out",
        transcriber_version="0.4.0",
        planner=lambda candidates, **_: tuple(accepted_plan(item) for item in candidates),
        runner=lambda plan, **_: transcript_jsonl(),
    )
    measurement = ClipMeasurement(
        clip_sha256="b" * 64,
        audio_properties={
            "sample_rate_hz": 24_000,
            "channels": 1,
            "bit_depth": 16,
            "duration_s": 4.0,
        },
        byte_size=192_044,
    )

    with pytest.raises(ValueError, match="unknown session"):
        finalize_free_speech_dataset(
            bundle,
            reviews={
                "candidate-001": HumanReview(
                    accepted=True,
                    actor="owner@example.test",
                    at="2026-08-01T13:00:00Z",
                )
            },
            measurements={"candidate-001": measurement},
            split_plan=plan_splits(("different-session",)),
            row_builder=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("row builder must not run")
            ),
        )
