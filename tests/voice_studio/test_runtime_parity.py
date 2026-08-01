"""Public contract tests for automated parity followed by human approval."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from voiceclonemlx.training.runtime_parity import (
    AutomatedParityResult,
    CandidatePromptEvidence,
    InvalidParityRequest,
    InvalidParityResult,
    ListeningApproval,
    ParityApprovalError,
    ParityPrompt,
    ParityThresholds,
    RuntimeParityRequest,
    approve,
    evaluate,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def request() -> RuntimeParityRequest:
    return RuntimeParityRequest(
        source_release_sha256=SHA_A,
        candidate_sha256=SHA_B,
        parity_set_sha256=SHA_C,
        prompts=(
            ParityPrompt(
                prompt_id="neutral-001",
                prompt_sha256=SHA_D,
                source_audio_sha256=SHA_E,
            ),
        ),
        thresholds=ParityThresholds(
            max_intelligibility_error_rate=0.10,
            max_alignment_error_ms=80.0,
            max_duration_ratio_delta=0.15,
            max_realtime_factor=1.0,
        ),
    )


class PassingEvaluator:
    def evaluate(self, parity_request):
        return AutomatedParityResult(
            source_release_sha256=SHA_A,
            candidate_sha256=SHA_B,
            parity_set_sha256=SHA_C,
            prompts=(
                CandidatePromptEvidence(
                    prompt_id="neutral-001",
                    prompt_sha256=SHA_D,
                    source_audio_sha256=SHA_E,
                    candidate_audio_sha256=SHA_F,
                    intelligibility_error_rate=0.05,
                    alignment_error_ms=40.0,
                    duration_ratio_delta=0.08,
                    realtime_factor=0.5,
                ),
            ),
        )


def test_passing_automated_metrics_still_wait_for_human_listening():
    report = evaluate(request(), PassingEvaluator())

    assert report.decision == "pending_listening"
    assert report.all_metrics_passed is True
    assert report.report_sha256 == "8fa79d3a99a30d3023a4240584d051b369b62d766ae6cf7c493e95a24c3fe76b"


@pytest.mark.parametrize(
    "invalid_request",
    (
        replace(request(), source_release_sha256="bad"),
        replace(request(), candidate_sha256="bad"),
        replace(request(), parity_set_sha256="bad"),
        replace(request(), prompts=()),
        replace(
            request(),
            prompts=(replace(request().prompts[0], prompt_sha256="bad"),),
        ),
        replace(
            request(),
            prompts=(replace(request().prompts[0], source_audio_sha256="bad"),),
        ),
        replace(
            request(),
            thresholds=replace(
                request().thresholds,
                max_alignment_error_ms=-1.0,
            ),
        ),
    ),
)
def test_invalid_frozen_evidence_never_reaches_evaluator(invalid_request):
    class RecordingEvaluator(PassingEvaluator):
        calls = 0

        def evaluate(self, parity_request):
            self.calls += 1
            return super().evaluate(parity_request)

    evaluator = RecordingEvaluator()

    with pytest.raises(InvalidParityRequest):
        evaluate(invalid_request, evaluator)

    assert evaluator.calls == 0


@pytest.mark.parametrize(
    "prompts",
    (
        (replace(request().prompts[0], prompt_id="  "),),
        (request().prompts[0], request().prompts[0]),
    ),
)
def test_frozen_prompt_ids_must_be_named_and_unique(prompts):
    evaluator = PassingEvaluator()

    with pytest.raises(InvalidParityRequest, match="prompt id"):
        evaluate(replace(request(), prompts=prompts), evaluator)


@pytest.mark.parametrize(
    "result_change",
    (
        {"source_release_sha256": SHA_B},
        {"candidate_sha256": SHA_A},
        {"parity_set_sha256": SHA_A},
        {"prompts": ()},
        {
            "prompts": (
                replace(
                    PassingEvaluator().evaluate(request()).prompts[0],
                    prompt_sha256=SHA_A,
                ),
            )
        },
        {
            "prompts": (
                replace(
                    PassingEvaluator().evaluate(request()).prompts[0],
                    source_audio_sha256=SHA_A,
                ),
            )
        },
        {
            "prompts": (
                replace(
                    PassingEvaluator().evaluate(request()).prompts[0],
                    candidate_audio_sha256="bad",
                ),
            )
        },
    ),
)
def test_candidate_evidence_must_match_every_frozen_binding(result_change):
    class MismatchedEvaluator(PassingEvaluator):
        def evaluate(self, parity_request):
            return replace(super().evaluate(parity_request), **result_change)

    with pytest.raises(InvalidParityResult):
        evaluate(request(), MismatchedEvaluator())


@pytest.mark.parametrize(
    "metric, value",
    (
        ("intelligibility_error_rate", float("nan")),
        ("alignment_error_ms", -1.0),
        ("duration_ratio_delta", float("inf")),
        ("realtime_factor", -1.0),
    ),
)
def test_non_finite_or_negative_metrics_are_not_evidence(metric, value):
    class InvalidMetricEvaluator(PassingEvaluator):
        def evaluate(self, parity_request):
            result = super().evaluate(parity_request)
            prompt = replace(result.prompts[0], **{metric: value})
            return replace(result, prompts=(prompt,))

    with pytest.raises(InvalidParityResult):
        evaluate(request(), InvalidMetricEvaluator())


@pytest.mark.parametrize(
    "metric, value",
    (
        ("intelligibility_error_rate", 0.11),
        ("alignment_error_ms", 81.0),
        ("duration_ratio_delta", 0.16),
        ("realtime_factor", 1.01),
    ),
)
def test_each_mandatory_metric_can_block_acceptance(metric, value):
    class FailingMetricEvaluator(PassingEvaluator):
        def evaluate(self, parity_request):
            result = super().evaluate(parity_request)
            prompt = replace(result.prompts[0], **{metric: value})
            return replace(result, prompts=(prompt,))

    report = evaluate(request(), FailingMetricEvaluator())

    assert report.all_metrics_passed is False
    assert report.decision == "pending_listening"


def approval_for(report) -> ListeningApproval:
    return ListeningApproval(
        listener_name="Alex Reviewer",
        approved_at="2026-08-01T07:30:00-07:00",
        listened_prompt_ids=("neutral-001",),
        source_release_sha256=SHA_A,
        candidate_sha256=SHA_B,
        parity_set_sha256=SHA_C,
        report_sha256=report.report_sha256,
    )


def test_complete_named_listening_approval_accepts_the_exact_report():
    report = evaluate(request(), PassingEvaluator())

    accepted = approve(report, approval_for(report))

    assert accepted.decision == "accepted"
    assert accepted.report is report
    assert accepted.approval.listener_name == "Alex Reviewer"
    assert accepted.approval.listened_prompt_ids == ("neutral-001",)


@pytest.mark.parametrize(
    "approval_change",
    (
        {"listener_name": "  "},
        {"approved_at": ""},
        {"approved_at": "2026-08-01T07:30:00"},
        {"listened_prompt_ids": ()},
        {"listened_prompt_ids": ("other",)},
        {"source_release_sha256": SHA_B},
        {"candidate_sha256": SHA_A},
        {"parity_set_sha256": SHA_A},
        {"report_sha256": SHA_A},
    ),
)
def test_approval_requires_complete_named_exact_listening(approval_change):
    report = evaluate(request(), PassingEvaluator())
    approval = replace(approval_for(report), **approval_change)

    with pytest.raises(ParityApprovalError):
        approve(report, approval)


def test_failed_metric_report_cannot_be_human_accepted():
    class FailingEvaluator(PassingEvaluator):
        def evaluate(self, parity_request):
            result = super().evaluate(parity_request)
            prompt = replace(result.prompts[0], alignment_error_ms=81.0)
            return replace(result, prompts=(prompt,))

    report = evaluate(request(), FailingEvaluator())

    with pytest.raises(ParityApprovalError, match="metric"):
        approve(report, approval_for(report))


def test_tampered_report_cannot_be_accepted_with_its_old_checksum():
    report = evaluate(request(), PassingEvaluator())
    tampered_prompt = replace(report.prompts[0], candidate_audio_sha256=SHA_A)
    tampered = replace(report, prompts=(tampered_prompt,))

    with pytest.raises(ParityApprovalError, match="checksum"):
        approve(tampered, approval_for(report))


def test_evaluator_failure_is_not_retried():
    class FailingEvaluator:
        calls = 0

        def evaluate(self, parity_request):
            self.calls += 1
            raise RuntimeError("evaluation stopped")

    evaluator = FailingEvaluator()

    with pytest.raises(RuntimeError, match="evaluation stopped"):
        evaluate(request(), evaluator)

    assert evaluator.calls == 1


def test_module_cannot_register_or_publish_a_runtime():
    import voiceclonemlx.training.runtime_parity as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imports = {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert not any("runtime_registry" in name for name in imports)
    assert not any("publisher" in name for name in imports)
    assert "register(" not in source
    assert "publish(" not in source
