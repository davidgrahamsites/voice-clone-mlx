"""Reader synthesis selection through verified bundles and fake runtimes."""

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from voiceclonemlx.reader_app.synthesis_controller import (
    ReaderSynthesisController,
    ReaderSynthesisControllerError,
    VoiceChoice,
)
from voiceclonemlx.reader_app.synthesis_session import ReaderSynthesisSession


class FakeRegistry:
    def __init__(self, runtimes=None):
        self.runtimes = runtimes or {}
        self.requests = []

    def get(self, runtime_id):
        self.requests.append(runtime_id)
        if runtime_id not in self.runtimes:
            raise ValueError(f"unknown runtime {runtime_id!r}")
        return self.runtimes[runtime_id]

    def ids(self):
        return tuple(self.runtimes)


def bundle_with(*runtime_ids):
    return SimpleNamespace(
        bundle_id="alex@0.7.0",
        voice_id="alex",
        model_version="0.7.0",
        manifest={
            "runtime_variants": [
                {"id": runtime_id, "artifact": f"runtimes/{runtime_id}/config.json"}
                for runtime_id in runtime_ids
            ]
        },
    )


def test_tampered_bundle_fails_before_readiness_or_runtime_lookup(tmp_path):
    calls = []

    def rejected_reader(path):
        calls.append(("verify", Path(path)))
        raise ValueError("runtime artifact checksum mismatch")

    registry = FakeRegistry({"fake_voice": object()})
    controller = ReaderSynthesisController(
        registry=registry,
        verified_runtime_ids=("fake_voice",),
        bundle_reader=rejected_reader,
        readiness=lambda *args, **kwargs: calls.append(("readiness", args)),
    )

    with pytest.raises(ReaderSynthesisControllerError, match="checksum mismatch"):
        controller.select(tmp_path / "tampered", "fake_voice")

    assert calls == [("verify", tmp_path / "tampered")]
    assert registry.requests == []


def test_inspect_bundle_exposes_only_verified_manifest_identity(tmp_path):
    bundle_dir = tmp_path / "alex"
    controller = ReaderSynthesisController(
        registry=FakeRegistry(),
        bundle_reader=lambda path: bundle_with("fake_voice", "other_voice"),
    )

    assert controller.inspect_bundle(bundle_dir) == VoiceChoice(
        bundle_dir=bundle_dir,
        bundle_id="alex@0.7.0",
        voice_id="alex",
        model_version="0.7.0",
        runtime_ids=("fake_voice", "other_voice"),
    )


def test_runtime_must_be_declared_by_the_verified_bundle(tmp_path):
    readiness_calls = []
    registry = FakeRegistry({"other": object()})
    controller = ReaderSynthesisController(
        registry=registry,
        verified_runtime_ids=("other",),
        bundle_reader=lambda path: bundle_with("fake_voice"),
        readiness=lambda *args, **kwargs: readiness_calls.append(args),
    )

    with pytest.raises(ReaderSynthesisControllerError, match="not declared"):
        controller.select(tmp_path / "alex", "other")

    assert readiness_calls == []
    assert registry.requests == []


@pytest.mark.parametrize(
    ("runtime_id", "verified_ids", "message"),
    [
        ("null", ("null",), "silent placeholder"),
        ("fake_voice", (), "not verified against a real model"),
    ],
)
def test_silent_or_unverified_runtime_is_never_selectable(
    tmp_path, runtime_id, verified_ids, message
):
    registry = FakeRegistry({runtime_id: object()})
    controller = ReaderSynthesisController(
        registry=registry,
        verified_runtime_ids=verified_ids,
        bundle_reader=lambda path: bundle_with(runtime_id),
        readiness=lambda *args, **kwargs: pytest.fail("readiness must not run"),
    )

    with pytest.raises(ReaderSynthesisControllerError, match=message):
        controller.select(tmp_path / "alex", runtime_id)

    assert registry.requests == []


def test_unregistered_runtime_is_rejected_before_readiness(tmp_path):
    controller = ReaderSynthesisController(
        registry=FakeRegistry(),
        verified_runtime_ids=("fake_voice",),
        bundle_reader=lambda path: bundle_with("fake_voice"),
        readiness=lambda *args, **kwargs: pytest.fail("readiness must not run"),
    )

    with pytest.raises(ReaderSynthesisControllerError, match="not registered"):
        controller.select(tmp_path / "alex", "fake_voice")


def test_unready_report_keeps_synthesis_disabled(tmp_path):
    runtime = object()
    registry = FakeRegistry({"fake_voice": runtime})
    report = SimpleNamespace(ready=False, blockers=("dependency_missing",))
    readiness_calls = []

    def readiness(bundle_dir, *, runtime_id, registry):
        readiness_calls.append((bundle_dir, runtime_id, registry))
        return report

    controller = ReaderSynthesisController(
        registry=registry,
        verified_runtime_ids=("fake_voice",),
        bundle_reader=lambda path: bundle_with("fake_voice"),
        readiness=readiness,
    )

    assert controller.select(tmp_path / "alex", "fake_voice") is report
    assert controller.ready is False
    assert readiness_calls == [(tmp_path / "alex", "fake_voice", registry)]


def non_silent_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\x01\x00\x02\x00")
    return output.getvalue()


def test_ready_selection_synthesizes_exact_text_bundle_and_runtime(tmp_path):
    bundle_dir = tmp_path / "alex"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runtime = object()
    round_trips = []

    def round_trip(bundle, *, runtime_id, text, runtime):
        round_trips.append((bundle, runtime_id, text, runtime))
        return non_silent_wav()

    def session_factory(**kwargs):
        return ReaderSynthesisSession(**kwargs, round_trip=round_trip)

    controller = ReaderSynthesisController(
        registry=FakeRegistry({"fake_voice": runtime}),
        verified_runtime_ids=("fake_voice",),
        bundle_reader=lambda path: bundle_with("fake_voice"),
        readiness=lambda *args, **kwargs: SimpleNamespace(ready=True, blockers=()),
        session_factory=session_factory,
    )

    controller.select(bundle_dir, "fake_voice")
    controller.set_text("Speak this exact line.")
    result = controller.synthesize(output_dir, "chapter.wav")

    assert result == output_dir / "chapter.wav"
    assert round_trips == [
        (bundle_dir, "fake_voice", "Speak this exact line.", runtime)
    ]
    with wave.open(str(result), "rb") as stream:
        assert stream.readframes(stream.getnframes()) != b"\x00" * 4


def test_ingest_text_reads_utf8_exactly(tmp_path):
    text_path = tmp_path / "chapter.txt"
    text_path.write_text("Café\nSecond line.", encoding="utf-8")
    controller = ReaderSynthesisController(registry=FakeRegistry())

    assert controller.ingest_text(text_path) == "Café\nSecond line."
    assert controller.text == "Café\nSecond line."


@pytest.mark.parametrize("contents", ["", "   \n"])
def test_blank_entered_or_ingested_text_is_rejected(tmp_path, contents):
    controller = ReaderSynthesisController(registry=FakeRegistry())
    text_path = tmp_path / "blank.txt"
    text_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ReaderSynthesisControllerError, match="empty"):
        controller.set_text(contents)
    with pytest.raises(ReaderSynthesisControllerError, match="empty"):
        controller.ingest_text(text_path)


def test_synthesize_stays_disabled_after_an_unready_selection(tmp_path):
    controller = ReaderSynthesisController(
        registry=FakeRegistry({"fake_voice": object()}),
        verified_runtime_ids=("fake_voice",),
        bundle_reader=lambda path: bundle_with("fake_voice"),
        readiness=lambda *args, **kwargs: SimpleNamespace(
            ready=False, blockers=("model_absent",)
        ),
    )
    controller.select(tmp_path / "alex", "fake_voice")
    controller.set_text("This must not run.")

    with pytest.raises(ReaderSynthesisControllerError, match="disabled"):
        controller.synthesize(tmp_path, "blocked.wav")

    assert not (tmp_path / "blocked.wav").exists()


def test_inspecting_another_bundle_clears_the_previous_ready_selection(tmp_path):
    controller = ReaderSynthesisController(
        registry=FakeRegistry({"fake_voice": object()}),
        verified_runtime_ids=("fake_voice",),
        bundle_reader=lambda path: bundle_with("fake_voice"),
        readiness=lambda *args, **kwargs: SimpleNamespace(ready=True, blockers=()),
    )
    controller.select(tmp_path / "first", "fake_voice")
    assert controller.ready is True

    controller.inspect_bundle(tmp_path / "second")

    assert controller.ready is False
