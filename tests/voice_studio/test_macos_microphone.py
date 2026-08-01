"""Tests for the optional macOS microphone provider.

The native boundary is always replaced with a fake.  This suite must never
open an input device or ask macOS for microphone permission.
"""

import pytest

from voiceclonegpt.recording.capture import CapturedPcm
from voiceclonegpt.recording import macos_microphone
from voiceclonegpt.recording.macos_microphone import (
    MacOSMicrophoneCapture,
    MicrophoneCaptureError,
)


def test_capture_is_inert_until_called():
    calls = []

    class FakeNativeCapture:
        def capture(self, **settings):
            calls.append(settings)
            return b"\x00\x00\x01\x00"

    capture = MacOSMicrophoneCapture(native_capture=FakeNativeCapture())

    assert calls == []
    assert capture() == CapturedPcm(b"\x00\x00\x01\x00", 24_000, 1, 2)
    assert calls == [
        {
            "duration_seconds": 5.0,
            "sample_rate_hz": 24_000,
            "channels": 1,
            "sample_width_bytes": 2,
        }
    ]


def test_default_construction_does_not_load_the_native_framework(monkeypatch):
    loads = []
    monkeypatch.setattr(
        macos_microphone.ctypes,
        "CDLL",
        lambda path: loads.append(path),
    )

    MacOSMicrophoneCapture()

    assert loads == []


@pytest.mark.parametrize("duration", [0, -1, 60.1, float("inf"), "five"])
def test_capture_duration_must_be_positive_and_bounded(duration):
    with pytest.raises(ValueError, match="duration_seconds"):
        MacOSMicrophoneCapture(duration_seconds=duration)


def test_default_capture_refuses_non_macos_without_loading_a_device(monkeypatch):
    capture = MacOSMicrophoneCapture()
    monkeypatch.setattr(macos_microphone.sys, "platform", "linux")

    with pytest.raises(MicrophoneCaptureError, match="macOS"):
        capture()


@pytest.mark.parametrize("sample_rate", [0, -1, 3_000, 193_000, 24_000.5, True])
def test_sample_rate_must_be_a_supported_whole_number(sample_rate):
    with pytest.raises(ValueError, match="sample_rate_hz"):
        MacOSMicrophoneCapture(sample_rate_hz=sample_rate)
