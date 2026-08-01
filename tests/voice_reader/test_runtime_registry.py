from pathlib import Path

import pytest

from voiceclonemlx.shared.roundtrip import run_round_trip
from voiceclonemlx.synthesis.null_runtime import NullRuntime
from voiceclonemlx.synthesis.runtime_registry import (
    RuntimeRegistry,
    RuntimeRegistrationError,
)

from test_model_bundle import _write_bundle


def test_registry_returns_registered_runtime_by_exact_id():
    registry = RuntimeRegistry()
    runtime = NullRuntime()

    registry.register("null", runtime)

    assert registry.get("null") is runtime


def test_registry_rejects_duplicate_or_unknown_ids():
    registry = RuntimeRegistry()
    registry.register("null", NullRuntime())

    with pytest.raises(RuntimeRegistrationError, match="already registered"):
        registry.register("null", NullRuntime())
    with pytest.raises(RuntimeRegistrationError, match="unknown runtime"):
        registry.get("mlx")


def test_null_runtime_produces_a_valid_nonempty_wav(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    audio = run_round_trip(
        bundle,
        runtime_id="mlx",
        text="A bounded local smoke test.",
        runtime=NullRuntime(),
    )

    assert audio[:4] == b"RIFF"
    assert audio[8:12] == b"WAVE"
    assert len(audio) > 44
