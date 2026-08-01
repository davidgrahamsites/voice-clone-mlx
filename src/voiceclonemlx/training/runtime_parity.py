"""Compare a runtime candidate with frozen source evidence before approval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from typing import Protocol


_LOWER_HEX = frozenset("0123456789abcdef")


class InvalidParityRequest(ValueError):
    """Frozen source, candidate, prompt, or threshold evidence is invalid."""


class InvalidParityResult(ValueError):
    """Automated parity evidence is incomplete or bound to other artifacts."""


class ParityApprovalError(ValueError):
    """A listening approval cannot accept this exact parity report."""


@dataclass(frozen=True)
class ParityPrompt:
    prompt_id: str
    prompt_sha256: str
    source_audio_sha256: str


@dataclass(frozen=True)
class ParityThresholds:
    max_intelligibility_error_rate: float
    max_alignment_error_ms: float
    max_duration_ratio_delta: float
    max_realtime_factor: float


@dataclass(frozen=True)
class RuntimeParityRequest:
    source_release_sha256: str
    candidate_sha256: str
    parity_set_sha256: str
    prompts: tuple[ParityPrompt, ...]
    thresholds: ParityThresholds


@dataclass(frozen=True)
class CandidatePromptEvidence:
    prompt_id: str
    prompt_sha256: str
    source_audio_sha256: str
    candidate_audio_sha256: str
    intelligibility_error_rate: float
    alignment_error_ms: float
    duration_ratio_delta: float
    realtime_factor: float


@dataclass(frozen=True)
class AutomatedParityResult:
    source_release_sha256: str
    candidate_sha256: str
    parity_set_sha256: str
    prompts: tuple[CandidatePromptEvidence, ...]


@dataclass(frozen=True)
class RuntimeEvaluationReport:
    source_release_sha256: str
    candidate_sha256: str
    parity_set_sha256: str
    prompts: tuple[CandidatePromptEvidence, ...]
    thresholds: ParityThresholds
    all_metrics_passed: bool
    decision: str
    report_sha256: str


@dataclass(frozen=True)
class ListeningApproval:
    listener_name: str
    approved_at: str
    listened_prompt_ids: tuple[str, ...]
    source_release_sha256: str
    candidate_sha256: str
    parity_set_sha256: str
    report_sha256: str


@dataclass(frozen=True)
class AcceptedRuntimeEvaluationReport:
    report: RuntimeEvaluationReport
    approval: ListeningApproval
    decision: str


class RuntimeParityEvaluator(Protocol):
    def evaluate(self, request: RuntimeParityRequest) -> AutomatedParityResult: ...


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _validate_request(request: RuntimeParityRequest) -> None:
    checksums = (
        request.source_release_sha256,
        request.candidate_sha256,
        request.parity_set_sha256,
    )
    if not all(_is_sha256(value) for value in checksums):
        raise InvalidParityRequest("Source, candidate, and parity checksums must be SHA-256.")
    if not request.prompts:
        raise InvalidParityRequest("At least one frozen parity prompt is required.")
    prompt_ids = tuple(prompt.prompt_id for prompt in request.prompts)
    if any(
        not isinstance(prompt_id, str) or not prompt_id.strip()
        for prompt_id in prompt_ids
    ) or len(set(prompt_ids)) != len(prompt_ids):
        raise InvalidParityRequest("Every frozen prompt id must be named and unique.")
    for prompt in request.prompts:
        if not _is_sha256(prompt.prompt_sha256) or not _is_sha256(
            prompt.source_audio_sha256
        ):
            raise InvalidParityRequest("Every prompt and source audio checksum must be SHA-256.")
    values = asdict(request.thresholds).values()
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise InvalidParityRequest("Every metric threshold must be finite and non-negative.")


def _validate_result(
    request: RuntimeParityRequest,
    result: AutomatedParityResult,
) -> None:
    if not isinstance(result, AutomatedParityResult):
        raise InvalidParityResult("The evaluator must return AutomatedParityResult.")
    for field in (
        "source_release_sha256",
        "candidate_sha256",
        "parity_set_sha256",
    ):
        if getattr(result, field) != getattr(request, field):
            raise InvalidParityResult(f"The result {field} does not match the request.")
    if len(result.prompts) != len(request.prompts):
        raise InvalidParityResult("The result must contain every frozen prompt exactly once.")
    for frozen, candidate in zip(request.prompts, result.prompts):
        if (
            candidate.prompt_id != frozen.prompt_id
            or candidate.prompt_sha256 != frozen.prompt_sha256
            or candidate.source_audio_sha256 != frozen.source_audio_sha256
        ):
            raise InvalidParityResult("Candidate prompt evidence does not match frozen evidence.")
        if not _is_sha256(candidate.candidate_audio_sha256):
            raise InvalidParityResult("Candidate audio checksum must be SHA-256.")
        metrics = (
            candidate.intelligibility_error_rate,
            candidate.alignment_error_ms,
            candidate.duration_ratio_delta,
            candidate.realtime_factor,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in metrics
        ):
            raise InvalidParityResult("Every metric must be finite and non-negative.")


def _report_checksum(
    request: RuntimeParityRequest,
    result: AutomatedParityResult,
) -> str:
    payload = {
        "source_release_sha256": request.source_release_sha256,
        "candidate_sha256": request.candidate_sha256,
        "parity_set_sha256": request.parity_set_sha256,
        "prompts": [asdict(prompt) for prompt in result.prompts],
        "thresholds": asdict(request.thresholds),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checksum_report(report: RuntimeEvaluationReport) -> str:
    request = RuntimeParityRequest(
        source_release_sha256=report.source_release_sha256,
        candidate_sha256=report.candidate_sha256,
        parity_set_sha256=report.parity_set_sha256,
        prompts=tuple(
            ParityPrompt(
                prompt_id=prompt.prompt_id,
                prompt_sha256=prompt.prompt_sha256,
                source_audio_sha256=prompt.source_audio_sha256,
            )
            for prompt in report.prompts
        ),
        thresholds=report.thresholds,
    )
    result = AutomatedParityResult(
        source_release_sha256=report.source_release_sha256,
        candidate_sha256=report.candidate_sha256,
        parity_set_sha256=report.parity_set_sha256,
        prompts=report.prompts,
    )
    return _report_checksum(request, result)


def _metrics_pass(
    prompts: tuple[CandidatePromptEvidence, ...],
    thresholds: ParityThresholds,
) -> bool:
    return all(
        prompt.intelligibility_error_rate
        <= thresholds.max_intelligibility_error_rate
        and prompt.alignment_error_ms <= thresholds.max_alignment_error_ms
        and prompt.duration_ratio_delta <= thresholds.max_duration_ratio_delta
        and prompt.realtime_factor <= thresholds.max_realtime_factor
        for prompt in prompts
    )


def evaluate(
    request: RuntimeParityRequest,
    evaluator: RuntimeParityEvaluator,
) -> RuntimeEvaluationReport:
    """Evaluate frozen evidence; automated success still awaits listening."""
    _validate_request(request)
    result = evaluator.evaluate(request)
    _validate_result(request, result)
    thresholds = request.thresholds
    passed = _metrics_pass(result.prompts, thresholds)
    return RuntimeEvaluationReport(
        source_release_sha256=request.source_release_sha256,
        candidate_sha256=request.candidate_sha256,
        parity_set_sha256=request.parity_set_sha256,
        prompts=result.prompts,
        thresholds=thresholds,
        all_metrics_passed=passed,
        decision="pending_listening",
        report_sha256=_report_checksum(request, result),
    )


def approve(
    report: RuntimeEvaluationReport,
    approval: ListeningApproval,
) -> AcceptedRuntimeEvaluationReport:
    """Accept an automated report only through explicit human listening."""
    if not isinstance(report, RuntimeEvaluationReport) or not isinstance(
        approval, ListeningApproval
    ):
        raise ParityApprovalError("A parity report and listening approval are required.")
    if report.decision != "pending_listening":
        raise ParityApprovalError("Only a pending listening report can be accepted.")
    if report.report_sha256 != _checksum_report(report):
        raise ParityApprovalError("The parity report checksum does not match its evidence.")
    if not report.all_metrics_passed or not _metrics_pass(
        report.prompts, report.thresholds
    ):
        raise ParityApprovalError("Every mandatory metric must pass before approval.")
    if not isinstance(approval.listener_name, str) or not approval.listener_name.strip():
        raise ParityApprovalError("Listening approval requires a named listener.")
    try:
        approved_at = datetime.fromisoformat(approval.approved_at)
    except (TypeError, ValueError) as exc:
        raise ParityApprovalError("Listening approval requires an ISO-8601 timestamp.") from exc
    if approved_at.tzinfo is None:
        raise ParityApprovalError("Listening approval timestamp must include a time zone.")
    expected_prompts = tuple(prompt.prompt_id for prompt in report.prompts)
    if approval.listened_prompt_ids != expected_prompts:
        raise ParityApprovalError("Every parity prompt must be listened to exactly once.")
    for field in (
        "source_release_sha256",
        "candidate_sha256",
        "parity_set_sha256",
        "report_sha256",
    ):
        if getattr(approval, field) != getattr(report, field):
            raise ParityApprovalError(f"Approval {field} does not match the report.")
    return AcceptedRuntimeEvaluationReport(
        report=report,
        approval=approval,
        decision="accepted",
    )
