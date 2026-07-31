from pathlib import Path

import pytest

from voiceclonegpt.shared.roundtrip import RoundTripError, run_round_trip

from test_model_bundle import _write_bundle


class FakeRuntime:
    def __init__(self):
        self.loaded_path: Path | None = None

    def load(self, artifact_path: Path, manifest: dict):
        self.loaded_path = artifact_path
        return "loaded-model"

    def synthesize(self, model, text: str) -> bytes:
        return f"{model}:{text}".encode()


class FailingRuntime(FakeRuntime):
    def load(self, artifact_path: Path, manifest: dict):
        raise RuntimeError("loader unavailable")


class EmptyRuntime(FakeRuntime):
    def synthesize(self, model, text: str) -> bytes:
        return b""


def test_reader_round_trip_loads_selected_mlx_artifact_and_synthesizes(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    runtime = FakeRuntime()

    audio = run_round_trip(bundle, runtime_id="mlx", text="Hello from my voice", runtime=runtime)

    assert runtime.loaded_path == bundle / "runtimes" / "mlx" / "model.bin"
    assert audio == b"loaded-model:Hello from my voice"


def test_reader_round_trip_supports_two_explicit_voice_bundles(tmp_path: Path):
    alex = _write_bundle(tmp_path / "alex", bundle_id="alex@0.1.0")
    sam = _write_bundle(tmp_path / "sam", bundle_id="sam@0.1.0")

    assert run_round_trip(alex, runtime_id="mlx", text="Alex", runtime=FakeRuntime())
    assert run_round_trip(sam, runtime_id="mlx", text="Sam", runtime=FakeRuntime())


def test_reader_round_trip_rejects_unknown_runtime(tmp_path: Path):
    bundle = _write_bundle(tmp_path)

    with pytest.raises(RoundTripError, match="cannot select runtime"):
        run_round_trip(bundle, runtime_id="pytorch", text="Hello", runtime=FakeRuntime())


def test_reader_round_trip_wraps_adapter_failure(tmp_path: Path):
    bundle = _write_bundle(tmp_path)

    with pytest.raises(RoundTripError, match="failed"):
        run_round_trip(bundle, runtime_id="mlx", text="Hello", runtime=FailingRuntime())


def test_reader_round_trip_rejects_empty_audio(tmp_path: Path):
    bundle = _write_bundle(tmp_path)

    with pytest.raises(RoundTripError, match="audio bytes"):
        run_round_trip(bundle, runtime_id="mlx", text="Hello", runtime=EmptyRuntime())
