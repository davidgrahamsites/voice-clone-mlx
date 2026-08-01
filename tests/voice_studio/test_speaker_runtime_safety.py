"""Test what the local speaker-evidence runtime refuses.

Accepted evidence and the runtime contract live in `test_speaker_runtime.py`.
Every backend is a plain fake: no test opens audio, loads a model, or reaches
 a network.
"""

import pytest

from voiceclonemlx.alignment.overlap_gate import SpeakerTurn
from voiceclonemlx.alignment.speaker_runtime import (
    MAX_DURATION_S,
    MAX_ENROLLMENT_FILES,
    MAX_TIMEOUT_S,
    MAX_TURNS,
    RuntimeProvenance,
    SpeakerRuntimeError,
    run_speaker_runtime,
)

from speaker_runtime_test_support import provenance


def test_provenance_refuses_blank_fields():
    with pytest.raises(SpeakerRuntimeError, match="diarizer"):
        RuntimeProvenance(
            diarizer=" ",
            diarizer_version="local-r1",
            verifier="speechbrain_ecapa",
            verifier_version="local-r2",
            enrollment_id="owner-enrollment-v1",
        )


def test_runtime_refuses_unvalidated_provenance_before_backends(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    def must_not_run(*_args, **_kwargs):
        pytest.fail("bad provenance must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match="provenance"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance={"diarizer": "fake"},
            diarizer=must_not_run,
            verifier=must_not_run,
        )


def test_oversized_duration_is_refused_before_a_backend_runs(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    def must_not_run(*_args, **_kwargs):
        pytest.fail("bounded input must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match="duration_s"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=MAX_DURATION_S + 0.1,
            provenance=provenance(),
            diarizer=must_not_run,
            verifier=must_not_run,
        )


@pytest.mark.parametrize(
    "timeout_s", [0, -1, float("inf"), True, "120", MAX_TIMEOUT_S + 1]
)
def test_unusable_or_unbounded_timeout_is_refused(tmp_path, timeout_s):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    def must_not_run(*_args, **_kwargs):
        pytest.fail("bad timeout must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match="timeout_s"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=must_not_run,
            verifier=must_not_run,
            timeout_s=timeout_s,
        )


@pytest.mark.parametrize("field", ["audio", "enrollment"])
def test_remote_paths_are_refused_before_a_backend_runs(tmp_path, field):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()
    audio_path = "https://example.invalid/session.wav" if field == "audio" else audio
    enrollment_paths = (
        ("https://example.invalid/owner.wav",) if field == "enrollment" else (enrollment,)
    )

    def must_not_run(*_args, **_kwargs):
        pytest.fail("remote paths must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match=field):
        run_speaker_runtime(
            audio_path,
            enrollment_paths,
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=must_not_run,
            verifier=must_not_run,
        )


def test_oversized_turn_result_is_refused_before_verification(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()
    turns = tuple(
        SpeakerTurn(float(index), float(index) + 0.5, "speaker_0")
        for index in range(MAX_TURNS + 1)
    )

    with pytest.raises(SpeakerRuntimeError, match="turns"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=MAX_DURATION_S,
            provenance=provenance(),
            diarizer=lambda _audio, *, timeout_s: turns,
            verifier=lambda *_args, **_kwargs: pytest.fail(
                "oversized diarization must not be verified"
            ),
        )


@pytest.mark.parametrize("threshold", [-1.1, 1.1, float("nan"), True, "0.75"])
def test_invalid_owner_threshold_is_refused_before_backends(tmp_path, threshold):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    def must_not_run(*_args, **_kwargs):
        pytest.fail("bad threshold must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match="owner_threshold"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=threshold,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=must_not_run,
            verifier=must_not_run,
        )


@pytest.mark.parametrize("owner_id", ["", " ", None, True])
def test_invalid_owner_id_is_refused_before_backends(tmp_path, owner_id):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    def must_not_run(*_args, **_kwargs):
        pytest.fail("bad owner id must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match="owner_id"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id=owner_id,
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=must_not_run,
            verifier=must_not_run,
        )


@pytest.mark.parametrize("enrollment_paths", [(), "owner.wav", None])
def test_missing_enrollment_is_refused_before_backends(tmp_path, enrollment_paths):
    audio = tmp_path / "session.wav"
    audio.touch()

    def must_not_run(*_args, **_kwargs):
        pytest.fail("missing enrollment must be refused before backend execution")

    with pytest.raises(SpeakerRuntimeError, match="enrollment_paths"):
        run_speaker_runtime(
            audio,
            enrollment_paths,
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=must_not_run,
            verifier=must_not_run,
        )


def test_enrollment_count_is_bounded_before_paths_or_backends_are_touched(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment_paths = tuple(
        tmp_path / f"missing-{index}.wav"
        for index in range(MAX_ENROLLMENT_FILES + 1)
    )

    with pytest.raises(SpeakerRuntimeError, match="enrollment_paths"):
        run_speaker_runtime(
            audio,
            enrollment_paths,
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=lambda *_args, **_kwargs: pytest.fail("must not diarize"),
            verifier=lambda *_args, **_kwargs: pytest.fail("must not verify"),
        )


@pytest.mark.parametrize(
    "scores",
    [{}, {"speaker_0": float("nan")}, {"speaker_0": 1.1}, {"speaker_0": True}, []],
)
def test_malformed_verifier_scores_fail_closed(tmp_path, scores):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    with pytest.raises(SpeakerRuntimeError, match="verifier"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=lambda _audio, *, timeout_s: (
                SpeakerTurn(0.0, 1.0, "speaker_0"),
            ),
            verifier=lambda _audio, _enrollment, _turns, *, timeout_s: scores,
        )


def test_turn_outside_declared_audio_duration_fails_before_verification(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    with pytest.raises(SpeakerRuntimeError, match="duration"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=lambda _audio, *, timeout_s: (
                SpeakerTurn(4.0, 5.1, "speaker_0"),
            ),
            verifier=lambda *_args, **_kwargs: pytest.fail(
                "out-of-bounds turns must not be verified"
            ),
        )


def test_backend_failure_is_not_retried_or_fallen_back(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()
    calls = []

    def fail_once(*_args, **_kwargs):
        calls.append("diarize")
        raise RuntimeError("local backend failed")

    with pytest.raises(RuntimeError, match="local backend failed"):
        run_speaker_runtime(
            audio,
            (enrollment,),
            owner_id="owner",
            owner_threshold=0.75,
            duration_s=5.0,
            provenance=provenance(),
            diarizer=fail_once,
            verifier=lambda *_args, **_kwargs: pytest.fail("must not verify"),
        )

    assert calls == ["diarize"]
