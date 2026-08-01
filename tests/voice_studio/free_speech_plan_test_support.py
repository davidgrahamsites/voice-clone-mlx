"""Shared Free Speech planner safety test fixtures."""

import pytest

from voiceclonemlx.alignment.free_speech_plan import FreeSpeechCandidate
from voiceclonemlx.alignment.overlap_gate import SpeakerTurn

@pytest.fixture
def local_assets(tmp_path):
    audio = tmp_path / "session.wav"
    audio.write_bytes(b"RIFF")
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    return {"audio_path": audio, "model_path": model, "output_dir": output}


def candidate(start=0.0, end=5.0, turns=None, audio_path="audio.wav"):
    return FreeSpeechCandidate(
        clip_start_s=start,
        clip_end_s=end,
        audio_path=str(audio_path),
        turns=tuple(turns if turns is not None else [SpeakerTurn(0.2, 4.8, "owner")]),
    )
