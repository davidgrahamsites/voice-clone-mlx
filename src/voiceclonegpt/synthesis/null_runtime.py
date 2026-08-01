"""Deterministic local runtime used for contract and packaging smoke tests."""

from io import BytesIO
from pathlib import Path
import wave


class NullRuntime:
    """Emit bounded silent WAV bytes without loading weights or using a network."""

    sample_rate = 22_050

    def load(self, artifact_path: Path, manifest: dict) -> dict:
        """Return an opaque handle after confirming the artifact is a file."""
        if not artifact_path.is_file():
            raise FileNotFoundError(f"runtime artifact not found: {artifact_path}")
        return {"artifact_path": artifact_path, "manifest": manifest}

    def synthesize(self, model: dict, text: str) -> bytes:
        """Return a valid, deterministic, non-empty mono WAV for ``text``."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        duration = min(3.0, max(0.25, len(text.strip()) / 80.0))
        frames = int(self.sample_rate * duration)
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"\x00\x00" * frames)
        return output.getvalue()
