"""Optional standard-library macOS microphone capture provider.

Importing and constructing this module are inert.  The native input boundary
is reached only when :class:`MacOSMicrophoneCapture` is explicitly called.
"""

import ctypes
import ctypes.util
import math
import sys
import threading

from voiceclonegpt.recording.capture import CapturedPcm


class MicrophoneCaptureError(RuntimeError):
    """The local macOS input device could not produce a take."""


_AUDIO_QUEUE_PERMISSION_DENIED = -66676
_LINEAR_PCM = int.from_bytes(b"lpcm", "big")
_SIGNED_INTEGER = 1 << 2
_PACKED = 1 << 3


class _AudioStreamBasicDescription(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_double),
        ("format_id", ctypes.c_uint32),
        ("format_flags", ctypes.c_uint32),
        ("bytes_per_packet", ctypes.c_uint32),
        ("frames_per_packet", ctypes.c_uint32),
        ("bytes_per_frame", ctypes.c_uint32),
        ("channels_per_frame", ctypes.c_uint32),
        ("bits_per_channel", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _AudioQueueBuffer(ctypes.Structure):
    _fields_ = [
        ("audio_data_bytes_capacity", ctypes.c_uint32),
        ("audio_data", ctypes.c_void_p),
        ("audio_data_byte_size", ctypes.c_uint32),
        ("user_data", ctypes.c_void_p),
        ("packet_description_capacity", ctypes.c_uint32),
        ("packet_descriptions", ctypes.c_void_p),
        ("packet_description_count", ctypes.c_uint32),
    ]


_AudioQueueBufferPointer = ctypes.POINTER(_AudioQueueBuffer)
_InputCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_void_p,
    _AudioQueueBufferPointer,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_void_p,
)


class MacOSMicrophoneCapture:
    """Capture one bounded mono, signed 16-bit pulse-code modulation take."""

    def __init__(
        self,
        *,
        duration_seconds: float = 5.0,
        sample_rate_hz: int = 24_000,
        native_capture=None,
    ) -> None:
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not math.isfinite(duration_seconds)
            or not 0 < duration_seconds <= 60
        ):
            raise ValueError("duration_seconds must be between 0 and 60")
        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(sample_rate_hz, int)
            or not 8_000 <= sample_rate_hz <= 192_000
        ):
            raise ValueError("sample_rate_hz must be an integer from 8000 to 192000")
        self.duration_seconds = duration_seconds
        self.sample_rate_hz = sample_rate_hz
        self._native_capture = native_capture

    def __call__(self) -> CapturedPcm:
        native_capture = self._native_capture
        if native_capture is None:
            if sys.platform != "darwin":
                raise MicrophoneCaptureError("microphone capture requires macOS")
            native_capture = _AudioQueueCapture()
        frames = native_capture.capture(
            duration_seconds=self.duration_seconds,
            sample_rate_hz=self.sample_rate_hz,
            channels=1,
            sample_width_bytes=2,
        )
        return CapturedPcm(frames, self.sample_rate_hz, 1, 2)


class _AudioQueueCapture:
    """Native Audio Queue Services boundary, loaded only for an explicit take."""

    _BUFFER_COUNT = 3

    def __init__(self) -> None:
        library = ctypes.util.find_library("AudioToolbox")
        if not library:
            raise MicrophoneCaptureError("macOS AudioToolbox is unavailable")
        try:
            self._api = ctypes.CDLL(library)
        except OSError as exc:
            raise MicrophoneCaptureError(f"cannot load macOS AudioToolbox: {exc}") from exc
        self._declare_api()

    def _declare_api(self) -> None:
        self._api.AudioQueueNewInput.argtypes = [
            ctypes.POINTER(_AudioStreamBasicDescription),
            _InputCallback,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._api.AudioQueueNewInput.restype = ctypes.c_int32
        self._api.AudioQueueAllocateBuffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_AudioQueueBufferPointer),
        ]
        self._api.AudioQueueAllocateBuffer.restype = ctypes.c_int32
        self._api.AudioQueueEnqueueBuffer.argtypes = [
            ctypes.c_void_p,
            _AudioQueueBufferPointer,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._api.AudioQueueEnqueueBuffer.restype = ctypes.c_int32
        self._api.AudioQueueStart.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._api.AudioQueueStart.restype = ctypes.c_int32
        self._api.AudioQueueStop.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self._api.AudioQueueStop.restype = ctypes.c_int32
        self._api.AudioQueueDispose.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self._api.AudioQueueDispose.restype = ctypes.c_int32

    @staticmethod
    def _check_status(status: int, operation: str) -> None:
        if status == 0:
            return
        if status == _AUDIO_QUEUE_PERMISSION_DENIED:
            raise MicrophoneCaptureError(
                "microphone permission was denied; allow Voice Studio in "
                "System Settings > Privacy & Security > Microphone"
            )
        raise MicrophoneCaptureError(f"{operation} failed with macOS status {status}")

    def capture(
        self,
        *,
        duration_seconds: float,
        sample_rate_hz: int,
        channels: int,
        sample_width_bytes: int,
    ) -> bytes:
        bytes_per_frame = channels * sample_width_bytes
        target_bytes = int(duration_seconds * sample_rate_hz) * bytes_per_frame
        audio_format = _AudioStreamBasicDescription(
            sample_rate_hz,
            _LINEAR_PCM,
            _SIGNED_INTEGER | _PACKED,
            bytes_per_frame,
            1,
            bytes_per_frame,
            channels,
            sample_width_bytes * 8,
            0,
        )
        queue = ctypes.c_void_p()
        frames = bytearray()
        finished = threading.Event()
        callback_errors = []

        @_InputCallback
        def receive(_user_data, input_queue, buffer, _time, _packets, _descs):
            try:
                remaining = target_bytes - len(frames)
                byte_count = min(buffer.contents.audio_data_byte_size, remaining)
                if byte_count:
                    frames.extend(ctypes.string_at(buffer.contents.audio_data, byte_count))
                if len(frames) >= target_bytes:
                    finished.set()
                else:
                    status = self._api.AudioQueueEnqueueBuffer(
                        input_queue, buffer, 0, None
                    )
                    self._check_status(status, "re-enqueueing microphone buffer")
            except Exception as exc:
                callback_errors.append(exc)
                finished.set()

        status = self._api.AudioQueueNewInput(
            ctypes.byref(audio_format), receive, None, None, None, 0, ctypes.byref(queue)
        )
        self._check_status(status, "opening microphone input")

        started = False
        try:
            buffer_bytes = max(bytes_per_frame, sample_rate_hz * bytes_per_frame // 10)
            for _ in range(self._BUFFER_COUNT):
                buffer = _AudioQueueBufferPointer()
                self._check_status(
                    self._api.AudioQueueAllocateBuffer(
                        queue, buffer_bytes, ctypes.byref(buffer)
                    ),
                    "allocating microphone buffer",
                )
                self._check_status(
                    self._api.AudioQueueEnqueueBuffer(queue, buffer, 0, None),
                    "enqueueing microphone buffer",
                )
            self._check_status(
                self._api.AudioQueueStart(queue, None), "starting microphone input"
            )
            started = True
            if not finished.wait(duration_seconds + 5.0):
                raise MicrophoneCaptureError("microphone capture timed out")
            if callback_errors:
                raise MicrophoneCaptureError(
                    f"microphone callback failed: {callback_errors[0]}"
                ) from callback_errors[0]
        finally:
            if started:
                self._api.AudioQueueStop(queue, True)
            self._api.AudioQueueDispose(queue, True)

        if not frames:
            raise MicrophoneCaptureError("microphone produced no audio frames")
        return bytes(frames)
