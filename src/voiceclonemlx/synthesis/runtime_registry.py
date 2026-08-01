"""Explicit runtime registration for Voice Reader composition roots."""

from typing import Dict

from voiceclonemlx.shared.roundtrip import RuntimeAdapter


class RuntimeRegistrationError(ValueError):
    """A runtime id cannot be registered or selected."""


class RuntimeRegistry:
    """Map exact runtime ids to adapters without implicit fallback."""

    def __init__(self) -> None:
        self._runtimes: Dict[str, RuntimeAdapter] = {}

    def register(self, runtime_id: str, runtime: RuntimeAdapter) -> None:
        if not isinstance(runtime_id, str) or not runtime_id.strip():
            raise RuntimeRegistrationError("runtime id must be a non-empty string")
        if runtime_id in self._runtimes:
            raise RuntimeRegistrationError(f"runtime {runtime_id!r} is already registered")
        if not callable(getattr(runtime, "load", None)) or not callable(
            getattr(runtime, "synthesize", None)
        ):
            raise RuntimeRegistrationError("runtime must provide load and synthesize")
        self._runtimes[runtime_id] = runtime

    def get(self, runtime_id: str) -> RuntimeAdapter:
        try:
            return self._runtimes[runtime_id]
        except KeyError as exc:
            raise RuntimeRegistrationError(f"unknown runtime {runtime_id!r}") from exc

    def ids(self) -> tuple[str, ...]:
        """Return registered ids in deterministic insertion order."""
        return tuple(self._runtimes)
