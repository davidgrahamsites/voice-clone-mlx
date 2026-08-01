"""Test accepted speaker-runtime evidence through its public seam.

Refusals and fail-closed behavior live in `test_speaker_runtime_safety.py`.
Every backend is a plain fake: no test opens audio, loads a model, or reaches
a network.
"""

import ast
from pathlib import Path

import pytest

from voiceclonemlx.alignment.free_speech_plan import FreeSpeechCandidate
from voiceclonemlx.alignment.overlap_gate import ClipDecision, SpeakerTurn, decide_clip
from voiceclonemlx.alignment.speaker_runtime import SpeakerScore, run_speaker_runtime

from speaker_runtime_test_support import provenance


def test_owner_only_evidence_is_accepted_by_the_existing_gate(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()
    runtime_provenance = provenance()

    evidence = run_speaker_runtime(
        audio,
        (enrollment,),
        owner_id="owner",
        owner_threshold=0.75,
        duration_s=5.0,
        provenance=runtime_provenance,
        diarizer=lambda _audio, *, timeout_s: (
            SpeakerTurn(0.2, 4.8, "speaker_0"),
        ),
        verifier=lambda _audio, _enrollment, _turns, *, timeout_s: {
            "speaker_0": 0.75
        },
    )
    assert evidence.audio_path == str(audio.resolve())
    assert evidence.enrollment_paths == (str(enrollment.resolve()),)
    assert evidence.provenance == runtime_provenance
    assert evidence.owner_id == "owner"
    assert evidence.owner_threshold == 0.75
    assert evidence.duration_s == 5.0
    assert evidence.scores == (SpeakerScore("speaker_0", 0.75),)
    assert evidence.turns == (SpeakerTurn(0.2, 4.8, "owner"),)
    assert FreeSpeechCandidate(
        clip_start_s=0.0,
        clip_end_s=5.0,
        audio_path=evidence.audio_path,
        turns=evidence.turns,
    ).turns is evidence.turns
    assert decide_clip(
        clip_start_s=0.0,
        clip_end_s=5.0,
        target_speaker="owner",
        turns=list(evidence.turns),
    ) == ClipDecision.accept("target_only")


def test_below_threshold_voice_is_not_labelled_as_the_owner(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    evidence = run_speaker_runtime(
        audio,
        (enrollment,),
        owner_id="owner",
        owner_threshold=0.75,
        duration_s=5.0,
        provenance=provenance(),
        diarizer=lambda _audio, *, timeout_s: (
            SpeakerTurn(0.2, 4.8, "speaker_1"),
        ),
        verifier=lambda _audio, _enrollment, _turns, *, timeout_s: {
            "speaker_1": 0.74
        },
    )

    assert evidence.turns == (SpeakerTurn(0.2, 4.8, "non_owner:speaker_1"),)
    assert decide_clip(
        clip_start_s=0.0,
        clip_end_s=5.0,
        target_speaker="owner",
        turns=list(evidence.turns),
    ) == ClipDecision.reject("target_not_detected")


def test_explicit_and_inferred_overlap_are_both_rejected(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()

    evidence = run_speaker_runtime(
        audio,
        (enrollment,),
        owner_id="owner",
        owner_threshold=0.75,
        duration_s=8.0,
        provenance=provenance(),
        diarizer=lambda _audio, *, timeout_s: (
            SpeakerTurn(5.0, 7.0, "speaker_0", overlap=True),
            SpeakerTurn(0.0, 3.0, "speaker_0"),
            SpeakerTurn(2.0, 4.0, "speaker_1"),
        ),
        verifier=lambda _audio, _enrollment, _turns, *, timeout_s: {
            "speaker_0": 0.9,
            "speaker_1": 0.2,
        },
    )

    assert evidence.turns == (
        SpeakerTurn(0.0, 3.0, "owner", overlap=True),
        SpeakerTurn(2.0, 4.0, "non_owner:speaker_1", overlap=True),
        SpeakerTurn(5.0, 7.0, "owner", overlap=True),
    )
    assert decide_clip(
        clip_start_s=0.0,
        clip_end_s=8.0,
        target_speaker="owner",
        turns=list(evidence.turns),
    ) == ClipDecision.reject("overlap_detected")


def test_each_backend_runs_once_with_the_same_bounded_timeout(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()
    calls = []

    def diarize(audio_path, *, timeout_s):
        calls.append(("diarize", audio_path, timeout_s))
        return (SpeakerTurn(0.0, 1.0, "speaker_0"),)

    def verify(audio_path, enrollment_paths, turns, *, timeout_s):
        calls.append(("verify", audio_path, enrollment_paths, turns, timeout_s))
        return {"speaker_0": 0.8}

    run_speaker_runtime(
        audio,
        (enrollment,),
        owner_id="owner",
        owner_threshold=0.75,
        duration_s=5.0,
        provenance=provenance(),
        diarizer=diarize,
        verifier=verify,
        timeout_s=30.0,
    )

    assert calls == [
        ("diarize", str(audio.resolve()), 30.0),
        (
            "verify",
            str(audio.resolve()),
            (str(enrollment.resolve()),),
            (SpeakerTurn(0.0, 1.0, "speaker_0"),),
            30.0,
        ),
    ]


def test_evidence_is_frozen(tmp_path):
    audio = tmp_path / "session.wav"
    audio.touch()
    enrollment = tmp_path / "owner.wav"
    enrollment.touch()
    evidence = run_speaker_runtime(
        audio,
        (enrollment,),
        owner_id="owner",
        owner_threshold=0.75,
        duration_s=5.0,
        provenance=provenance(),
        diarizer=lambda _audio, *, timeout_s: (),
        verifier=lambda _audio, _enrollment, _turns, *, timeout_s: {},
    )

    with pytest.raises(Exception):
        evidence.turns = ()


def test_module_imports_no_backend_process_network_or_audio_package():
    source_path = Path(__file__).parents[2] / "src/voiceclonemlx/alignment/speaker_runtime.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not roots.intersection(
        {
            "mlx",
            "mlx_audio",
            "torch",
            "speechbrain",
            "pyannote",
            "subprocess",
            "socket",
            "requests",
            "urllib",
            "soundfile",
            "librosa",
        }
    )
