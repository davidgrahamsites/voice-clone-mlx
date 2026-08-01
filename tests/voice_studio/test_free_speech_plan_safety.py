"""Free Speech planner no-salvage and policy-reuse tests."""

import ast
import inspect
from pathlib import Path

import pytest

from free_speech_plan_test_support import candidate, local_assets
from voiceclonegpt.alignment import free_speech_plan
from voiceclonegpt.alignment.free_speech_plan import (
    MAX_CANDIDATES,
    FreeSpeechPlanError,
    PlannedCandidate,
    plan_free_speech,
)
from voiceclonegpt.alignment.overlap_gate import ClipDecision, SpeakerTurn

SOURCE = Path(free_speech_plan.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


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
