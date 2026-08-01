"""Test what the Free Speech Mode planner refuses to do.

Three guarantees, each of which would be expensive to discover in production:

1. **No salvage.** A rejected candidate is rejected whole. There is no
   parameter, no code path, and no result field that trims a clip down to a
   clean prefix or suffix — the audio the gate refused is never partly kept.
2. **Policy reuse.** Acceptance is `overlap_gate.decide_clip` and plan
   construction is `whisper_plan.build_whisper_plan`. A second copy of either
   rule would drift from the reviewed one.
3. **Inertness.** Planning describes work; it never performs it. No process,
   no network, no model import.
"""

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path

import pytest

from voiceclonegpt.alignment import free_speech_plan
from voiceclonegpt.alignment.free_speech_plan import (
    MAX_CANDIDATES,
    FreeSpeechCandidate,
    FreeSpeechPlanError,
    PlannedCandidate,
    plan_free_speech,
)
from voiceclonegpt.alignment.overlap_gate import ClipDecision, SpeakerTurn
from voiceclonegpt.alignment.whisper_plan import WhisperPlanError

SOURCE_PATH = Path(free_speech_plan.__file__)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


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


# --- no salvage ------------------------------------------------------------


SALVAGE_WORDS = ("salvage", "trim", "shrink", "clamp", "truncate", "shave")


def test_no_salvage_parameter_exists_on_the_planner():
    names = set(inspect.signature(plan_free_speech).parameters)

    assert not [n for n in names if any(w in n.lower() for w in SALVAGE_WORDS)]


def test_no_salvage_field_exists_on_the_result():
    names = set(PlannedCandidate.__dataclass_fields__)

    assert not [n for n in names if any(w in n.lower() for w in SALVAGE_WORDS)]


def test_no_salvage_named_identifier_exists_in_the_module():
    identifiers = {
        node.id for node in ast.walk(TREE) if isinstance(node, ast.Name)
    } | {
        node.arg for node in ast.walk(TREE) if isinstance(node, ast.arg)
    }

    assert not [
        name
        for name in identifiers
        if any(word in name.lower() for word in SALVAGE_WORDS)
    ]


def test_a_partly_clean_candidate_is_rejected_whole(local_assets):
    """The first three seconds are owner-only. None of them survive."""
    results = plan_free_speech(
        [
            candidate(
                start=0.0,
                end=8.0,
                audio_path=local_assets["audio_path"],
                turns=[
                    SpeakerTurn(0.0, 3.0, "owner"),
                    SpeakerTurn(3.0, 8.0, "guest"),
                ],
            )
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert results[0].whisper_plan is None
    assert (results[0].clip_start_s, results[0].clip_end_s) == (0.0, 8.0)


def test_an_accepted_candidate_is_planned_at_its_full_span(local_assets):
    """Acceptance never narrows the span either."""
    results = plan_free_speech(
        [
            candidate(
                start=2.0,
                end=12.0,
                audio_path=local_assets["audio_path"],
                turns=[SpeakerTurn(2.5, 11.5, "owner")],
            )
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert (results[0].clip_start_s, results[0].clip_end_s) == (2.0, 12.0)


# --- policy reuse ----------------------------------------------------------


def test_acceptance_defers_to_decide_clip(monkeypatch, local_assets):
    seen = []

    def fake_decide_clip(**kwargs):
        seen.append(kwargs)
        return ClipDecision.reject("stubbed")

    monkeypatch.setattr(free_speech_plan, "decide_clip", fake_decide_clip)

    results = plan_free_speech(
        [
            candidate(
                start=1.0,
                end=6.0,
                audio_path=local_assets["audio_path"],
            )
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert len(seen) == 1
    assert seen[0]["clip_start_s"] == 1.0
    assert seen[0]["clip_end_s"] == 6.0
    assert seen[0]["target_speaker"] == "owner"
    assert results[0].decision == ClipDecision.reject("stubbed")
    assert results[0].whisper_plan is None


def test_a_stubbed_acceptance_reaches_plan_construction(monkeypatch, local_assets):
    """Nothing else in the planner may veto what the gate accepted."""
    monkeypatch.setattr(
        free_speech_plan,
        "decide_clip",
        lambda **_: ClipDecision.accept("stubbed"),
    )

    results = plan_free_speech(
        [
            candidate(
                audio_path=local_assets["audio_path"],
                turns=[SpeakerTurn(0.2, 4.8, "guest")],
            )
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert results[0].whisper_plan is not None


def test_plan_construction_defers_to_build_whisper_plan(monkeypatch, local_assets):
    seen = []

    def fake_build(audio_path, output_dir, model_path, *, language=None):
        seen.append((audio_path, output_dir, model_path, language))
        return "stub-plan"

    monkeypatch.setattr(free_speech_plan, "build_whisper_plan", fake_build)

    results = plan_free_speech(
        [candidate(audio_path=local_assets["audio_path"])],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
        language="en",
    )

    assert len(seen) == 1
    assert seen[0][3] == "en"
    assert results[0].whisper_plan == "stub-plan"


def test_build_whisper_plan_is_not_called_for_a_rejection(monkeypatch, local_assets):
    def explode(*_args, **_kwargs):
        raise AssertionError("a rejected candidate must not be planned")

    monkeypatch.setattr(free_speech_plan, "build_whisper_plan", explode)

    results = plan_free_speech(
        [
            candidate(
                audio_path=local_assets["audio_path"],
                turns=[SpeakerTurn(0.2, 4.8, "guest")],
            )
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert results[0].whisper_plan is None


def test_the_module_does_not_restate_gate_reasons():
    """Reason strings are `overlap_gate`'s vocabulary, not a second copy."""
    literals = {
        node.value
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    docstrings = {
        ast.get_docstring(node) or ""
        for node in ast.walk(TREE)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
    }
    code_literals = literals - docstrings

    for reason in (
        "overlap_detected",
        "target_not_detected",
        "other_speaker_detected",
        "target_only",
        "accept",
        "reject",
    ):
        assert reason not in code_literals


def test_the_cap_is_enforced_before_any_gate_call(monkeypatch, local_assets):
    monkeypatch.setattr(
        free_speech_plan,
        "decide_clip",
        lambda **_: pytest.fail("the cap must be checked first"),
    )

    with pytest.raises(FreeSpeechPlanError):
        plan_free_speech(
            [candidate(audio_path=local_assets["audio_path"])] * (MAX_CANDIDATES + 1),
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


# --- inertness -------------------------------------------------------------


FORBIDDEN_IMPORTS = {
    "subprocess",
    "os.system",
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "mlx",
    "mlx_whisper",
    "whisper",
    "torch",
    "numpy",
    "soundfile",
    "librosa",
    "asyncio",
    "multiprocessing",
    "shutil",
}


def imported_names():
    names = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_no_process_network_or_model_imports():
    for name in imported_names():
        root = name.split(".")[0]
        assert root not in FORBIDDEN_IMPORTS, name


def test_no_forbidden_call_names_appear():
    called = {
        node.func.id
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    for forbidden in (
        "run",
        "call",
        "check_output",
        "Popen",
        "system",
        "spawn",
        "urlopen",
        "eval",
        "exec",
        "compile",
        "open",
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
    ):
        assert forbidden not in called


def test_planning_writes_no_file(local_assets):
    output = local_assets["output_dir"]
    before = sorted(p.name for p in output.iterdir())

    plan_free_speech(
        [candidate(audio_path=local_assets["audio_path"])],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=output,
    )

    assert sorted(p.name for p in output.iterdir()) == before


def test_the_planner_returns_an_argv_it_does_not_run(local_assets):
    results = plan_free_speech(
        [candidate(audio_path=local_assets["audio_path"])],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    argv = results[0].whisper_plan.argv
    assert isinstance(argv, tuple)
    assert "-m" in argv


# --- path validation is not skippable by being rejected --------------------
#
# A candidate the gate refuses still had its path recorded by whatever
# produced it. A remote path there is a bug in the caller, and discovering it
# only for the candidates that happened to be accepted would mean the error
# surfaces or hides depending on who was talking.


def test_a_remote_path_on_an_accepted_candidate_is_refused(local_assets):
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [candidate(audio_path="http://example.com/session.wav")],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


@pytest.mark.parametrize(
    "remote",
    ["http://example.com/session.wav", "//server/share/session.wav"],
)
def test_a_remote_path_on_a_rejected_candidate_is_refused(local_assets, remote):
    """The candidate below would be rejected. The path still has to be real."""
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [
                candidate(
                    audio_path=remote,
                    turns=[SpeakerTurn(0.2, 4.8, "guest")],
                )
            ],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_a_missing_path_on_a_rejected_candidate_is_refused(local_assets, tmp_path):
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [
                candidate(
                    audio_path=tmp_path / "absent.wav",
                    turns=[SpeakerTurn(0.2, 4.8, "guest")],
                )
            ],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_paths_are_validated_before_any_gate_call(monkeypatch, local_assets):
    monkeypatch.setattr(
        free_speech_plan,
        "decide_clip",
        lambda **_: pytest.fail("paths must be validated first"),
    )

    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [candidate(audio_path="http://example.com/session.wav")],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_a_later_candidates_bad_path_refuses_the_whole_batch(local_assets):
    """Validation is a pass over the batch, not a per-row surprise."""
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [
                candidate(audio_path=local_assets["audio_path"]),
                candidate(audio_path="http://example.com/session.wav"),
            ],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_path_validation_defers_to_resolve_local_audio(monkeypatch, local_assets):
    seen = []

    def fake(value):
        seen.append(value)
        return Path(local_assets["audio_path"]).resolve()

    monkeypatch.setattr(free_speech_plan, "resolve_local_audio", fake)

    plan_free_speech(
        [
            candidate(audio_path="anything-at-all.wav"),
            candidate(
                audio_path="also-anything.wav",
                turns=[SpeakerTurn(0.2, 4.8, "guest")],
            ),
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert seen == ["anything-at-all.wav", "also-anything.wav"]


# --- the cap is enforced before the batch is materialized ------------------


class OversizedSequence(Sequence):
    """Reports a length over the cap and refuses to be read.

    A planner that measured `len(tuple(candidates))` would have to consume the
    sequence first. For a generator-backed or lazily-loaded diarizer result
    that is exactly the work the cap exists to avoid.
    """

    def __len__(self):
        return MAX_CANDIDATES + 1

    def __getitem__(self, index):
        raise AssertionError("an oversized batch must not be read")

    def __iter__(self):
        raise AssertionError("an oversized batch must not be iterated")


def test_an_oversized_sequence_is_never_iterated(local_assets):
    with pytest.raises(FreeSpeechPlanError):
        plan_free_speech(
            OversizedSequence(),
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )
