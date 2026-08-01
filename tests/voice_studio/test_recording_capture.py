"""Tests for the detachable local recording-capture seam.

All audio comes from a fake capture function.  The suite must never open a
microphone, request macOS permission, or import a platform audio library.
"""

import json
import hashlib
import wave

import pytest

from voiceclonegpt.recording.capture import (
    CaptureError,
    CapturedPcm,
    capture_session,
)


def pcm_capture() -> CapturedPcm:
    return CapturedPcm(
        frames=b"\x00\x00\x01\x00",
        sample_rate_hz=24_000,
        channels=1,
        sample_width_bytes=2,
    )


def test_capture_session_writes_lossless_wav_and_session_record(tmp_path):
    result = capture_session(
        tmp_path,
        session_id="session-1",
        voice_id="alex",
        capture=pcm_capture,
        prompt_manifest_id="prompt-1",
        take_id="neutral-1",
        device_name="Built-in Microphone",
    )

    assert result.wav_path == tmp_path / "neutral-1.wav"
    assert result.manifest_path == tmp_path / "neutral-1.json"
    with wave.open(str(result.wav_path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
        assert wav.readframes(2) == b"\x00\x00\x01\x00"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == "session-1"
    assert manifest["voice_id"] == "alex"
    assert manifest["prompt_manifest_id"] == "prompt-1"
    assert manifest["master_audio"] == "neutral-1.wav"
    assert manifest["take_id"] == "neutral-1"
    assert manifest["device_name"] == "Built-in Microphone"
    assert manifest["masters"][0]["sha256"] == hashlib.sha256(
        result.wav_path.read_bytes()
    ).hexdigest()


def test_capture_session_supports_free_speech_without_a_prompt(tmp_path):
    result = capture_session(
        tmp_path,
        session_id="free-1",
        voice_id="alex",
        capture=pcm_capture,
        take_id="free-1",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["prompt_manifest_id"] is None


def test_capture_failure_writes_no_audio_or_manifest(tmp_path):
    def denied():
        raise PermissionError("microphone permission was denied")

    with pytest.raises(CaptureError, match="microphone permission was denied"):
        capture_session(
            tmp_path, session_id="session-1", voice_id="alex", capture=denied,
            take_id="neutral-1",
        )

    assert list(tmp_path.iterdir()) == []


def test_empty_capture_is_refused_without_writing(tmp_path):
    def empty():
        return CapturedPcm(b"", 24_000, 1, 2)

    with pytest.raises(CaptureError, match="no audio frames"):
        capture_session(
            tmp_path, session_id="session-1", voice_id="alex", capture=empty,
            take_id="neutral-1",
        )

    assert list(tmp_path.iterdir()) == []


def test_existing_take_is_not_replaced_or_recaptured(tmp_path):
    wav_path = tmp_path / "neutral-1.wav"
    wav_path.write_bytes(b"existing recording")
    called = []

    with pytest.raises(CaptureError, match="append-only"):
        capture_session(
            tmp_path, session_id="session-1", voice_id="alex",
            capture=lambda: called.append(True) or pcm_capture(), take_id="neutral-1",
        )

    assert wav_path.read_bytes() == b"existing recording"
    assert called == []


@pytest.mark.parametrize("take_id", ["", "../escape", "nested/take", "."])
def test_unsafe_take_id_is_refused_before_capture(tmp_path, take_id):
    called = []

    with pytest.raises(CaptureError, match="take_id"):
        capture_session(
            tmp_path, session_id="session-1", voice_id="alex",
            capture=lambda: called.append(True) or pcm_capture(), take_id=take_id,
        )

    assert called == []
