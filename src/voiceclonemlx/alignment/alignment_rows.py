"""The AlignmentManifest v1 row: the artifact a reviewer signs off.

One job: define, validate, and serialize the rows that pair an expected script
utterance with what was actually transcribed. It matches nothing and reads
nothing — producing the pairings is the aligner's job, and cutting audio is the
preprocessor's.

**Every row is born `pending`, and every decision is attributed.**
`build_alignment_row` has no `review_state` parameter, so the builder cannot
produce a decided row; `accept_row` and `reject_row` are the transitions, and
both require an actor and a timestamp. The same invariant is enforced on
`AlignmentRow` itself, so constructing one directly cannot manufacture an
unattributed approval either.

What this module can enforce is **attribution** — that a decision names who
made it and when. It cannot verify that the actor is a person. The lifecycle
contract's rule (*only a human can change `pending` to `accepted`*) is a
process rule about who is permitted to sign; the mechanism here makes an
unsigned decision impossible and leaves the signer's identity to the process
around it.

Deterministic and pure: no clock, no filesystem, no audio, no model, no
network. Timestamps arrive as arguments so the same inputs always produce the
same manifest bytes.

Field names follow `AlignmentManifest` v1 in
`docs/architecture/voice-model-lifecycle.md`.
"""

import json
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

from voiceclonemlx.alignment.alignment_row_schema import (
    ALIGNMENT_SCHEMA_VERSION,
    REVIEW_ACCEPTED,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    REVIEW_STATES,
    AlignmentRowError,
    _DECIDED_STATES,
    _FIELDS,
    _confidence,
    _relative_reference,
    _string_tuple,
    _style,
    _text,
    _time,
)

#: Re-exported so callers import the whole contract from one module.
__all__ = [
    "ALIGNMENT_SCHEMA_VERSION",
    "REVIEW_PENDING",
    "REVIEW_ACCEPTED",
    "REVIEW_REJECTED",
    "REVIEW_STATES",
    "AlignmentRow",
    "AlignmentRowError",
    "build_alignment_row",
    "accept_row",
    "reject_row",
    "rows_to_jsonl",
    "parse_alignment_jsonl",
]


@dataclass(frozen=True)
class AlignmentRow:
    """One expected utterance and what was heard in its place.

    The review invariant is enforced here, not only in `build_alignment_row`:
    a decided row must name its actor and time, and a pending row must not.
    Constructing this type directly is allowed — it just cannot be used to
    manufacture an *unattributed* decision.
    """

    schema_version: str
    utterance_id: str
    expected_text: str
    observed_text: str
    style: str
    start_s: float
    end_s: float
    segment_ids: Tuple[str, ...]
    master_audio: str
    confidence: Optional[float]
    mismatch_reasons: Tuple[str, ...]
    review_state: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.review_state not in REVIEW_STATES:
            raise AlignmentRowError(
                f"review_state must be one of {', '.join(REVIEW_STATES)}: "
                f"{self.review_state!r}"
            )

        if self.review_state in _DECIDED_STATES:
            _text(self.reviewed_by, "reviewed_by")
            _text(self.reviewed_at, "reviewed_at")
        elif self.reviewed_by is not None or self.reviewed_at is not None:
            raise AlignmentRowError(
                "a pending row must not name a reviewer: "
                f"reviewed_by={self.reviewed_by!r} "
                f"reviewed_at={self.reviewed_at!r}"
            )


def build_alignment_row(
    *,
    utterance_id: str,
    expected_text: str,
    observed_text: str,
    style: str,
    start_s: float,
    end_s: float,
    segment_ids: Sequence[str] = (),
    master_audio: str,
    confidence=None,
    mismatch_reasons: Sequence[str] = (),
) -> AlignmentRow:
    """Build one `pending` alignment row.

    There is deliberately **no** `review_state` parameter: a row cannot be born
    accepted, so an approval can only come from `accept_row`.

    `expected_text` and `observed_text` are stored exactly as given —
    a reviewer compares them, and normalizing either would change what is being
    compared. `observed_text` may be empty: silence transcribed as nothing is a
    real outcome worth reviewing.

    Raises:
        AlignmentRowError: Any field is missing, of the wrong type, out of
            range, or names a non-canonical style or non-relative path.
    """
    start = _time(start_s, "start_s")
    end = _time(end_s, "end_s")
    if start >= end:
        raise AlignmentRowError(
            f"end_s must be after start_s: start_s={start_s!r} end_s={end_s!r}"
        )

    return AlignmentRow(
        schema_version=ALIGNMENT_SCHEMA_VERSION,
        utterance_id=_text(utterance_id, "utterance_id"),
        expected_text=_text(expected_text, "expected_text"),
        observed_text=_text(observed_text, "observed_text", allow_empty=True),
        style=_style(style),
        start_s=start,
        end_s=end,
        segment_ids=_string_tuple(segment_ids, "segment_ids"),
        master_audio=_relative_reference(master_audio),
        confidence=_confidence(confidence),
        mismatch_reasons=_string_tuple(mismatch_reasons, "mismatch_reasons"),
        review_state=REVIEW_PENDING,
    )


def _decided(row: AlignmentRow, state: str, actor, at) -> AlignmentRow:
    """Move a pending row to a decided state, or refuse."""
    if row.review_state != REVIEW_PENDING:
        raise AlignmentRowError(
            f"only a pending row can be reviewed; this row is "
            f"{row.review_state!r}"
        )

    return replace(
        row,
        review_state=state,
        reviewed_by=_text(actor, "actor"),
        reviewed_at=_text(at, "at"),
    )


def accept_row(row: AlignmentRow, *, actor: str, at: str) -> AlignmentRow:
    """Accept a pending row, attributed to an actor.

    Args:
        row: A pending row.
        actor: Who approved it. Required — an unattributed approval is not one.
            This module records the name; it cannot verify who it belongs to.
        at: When, as a caller-supplied timestamp string. Passed in rather than
            read from a clock so manifests stay reproducible.

    Raises:
        AlignmentRowError: The row is not pending, or actor/at are blank.
    """
    return _decided(row, REVIEW_ACCEPTED, actor, at)


def reject_row(
    row: AlignmentRow, *, actor: str, at: str, reason: str
) -> AlignmentRow:
    """Reject a pending row, recording why.

    Raises:
        AlignmentRowError: The row is not pending, or actor/at/reason are blank.
    """
    checked_reason = _text(reason, "reason")
    decided = _decided(row, REVIEW_REJECTED, actor, at)
    return replace(
        decided, mismatch_reasons=decided.mismatch_reasons + (checked_reason,)
    )


def _to_mapping(row: AlignmentRow) -> dict:
    """Return the JSON-ready mapping for one row."""
    return {
        "schema_version": row.schema_version,
        "utterance_id": row.utterance_id,
        "expected_text": row.expected_text,
        "observed_text": row.observed_text,
        "style": row.style,
        "start_s": row.start_s,
        "end_s": row.end_s,
        "segment_ids": list(row.segment_ids),
        "master_audio": row.master_audio,
        "confidence": row.confidence,
        "mismatch_reasons": list(row.mismatch_reasons),
        "review_state": row.review_state,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at,
    }


def rows_to_jsonl(rows) -> str:
    """Serialize rows as JSONL: one JSON object per line.

    Keys are sorted and non-ASCII is left unescaped, so the same rows always
    produce the same bytes and the file stays readable by a human.
    """
    lines = [
        json.dumps(_to_mapping(row), sort_keys=True, ensure_ascii=False)
        for row in rows
    ]
    return "".join(line + "\n" for line in lines)


def _row_from_mapping(data: Any, line_no: int) -> AlignmentRow:
    """Rebuild and re-validate one row from a parsed line."""
    where = f"line {line_no}"

    if not isinstance(data, dict):
        raise AlignmentRowError(f"{where} is not a JSON object")

    # Exact key set. A missing field means the writer produced something this
    # schema does not describe; an extra one means the writer knew something
    # this reader would silently discard. Neither is safe to guess through.
    present = set(data)
    expected = set(_FIELDS)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise AlignmentRowError(
            f"{where} is missing required field(s): {', '.join(missing)}"
        )
    if unexpected:
        raise AlignmentRowError(
            f"{where} has unexpected field(s): {', '.join(unexpected)}"
        )

    if data.get("schema_version") != ALIGNMENT_SCHEMA_VERSION:
        raise AlignmentRowError(
            f"{where} has unsupported schema_version: "
            f"{data.get('schema_version')!r}"
        )

    state = data.get("review_state")
    if state not in REVIEW_STATES:
        raise AlignmentRowError(f"{where} has unknown review_state: {state!r}")

    # Rebuild through the normal constructor so a hand-edited file faces the
    # same validation as a freshly built row.
    try:
        row = build_alignment_row(
            utterance_id=data.get("utterance_id"),
            expected_text=data.get("expected_text"),
            observed_text=data.get("observed_text"),
            style=data.get("style"),
            start_s=data.get("start_s"),
            end_s=data.get("end_s"),
            segment_ids=data.get("segment_ids", ()),
            master_audio=data.get("master_audio"),
            confidence=data.get("confidence"),
            mismatch_reasons=data.get("mismatch_reasons", ()),
        )
    except AlignmentRowError as exc:
        raise AlignmentRowError(f"{where}: {exc}") from exc

    reviewed_by = data.get("reviewed_by")
    reviewed_at = data.get("reviewed_at")

    if state in _DECIDED_STATES:
        # A decided row must name who decided and when. Whether that actor is
        # a person is a process question this module cannot answer.
        if not isinstance(reviewed_by, str) or not reviewed_by.strip():
            raise AlignmentRowError(
                f"{where} is {state!r} but has no reviewed_by"
            )
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise AlignmentRowError(
                f"{where} is {state!r} but has no reviewed_at"
            )
    elif reviewed_by is not None or reviewed_at is not None:
        raise AlignmentRowError(
            f"{where} is pending but claims a reviewer"
        )

    return replace(
        row,
        review_state=state,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
    )


def parse_alignment_jsonl(text) -> Tuple[AlignmentRow, ...]:
    """Parse a JSONL manifest back into validated rows.

    Every row is re-validated against the exact v1 key set, and a decided row
    must name its reviewer — so a hand-edited manifest cannot pass an
    unattributed or inconsistent decision through the gate.

    Raises:
        AlignmentRowError: Malformed JSON, an unknown schema version or review
            state, an unattributed approval, or any invalid field. Errors name
            the line number.
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AlignmentRowError(f"manifest is not valid UTF-8: {exc}") from exc

    if not isinstance(text, str):
        raise AlignmentRowError(
            f"manifest must be text: {type(text).__name__}"
        )

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            data = json.loads(line)
        except ValueError as exc:
            raise AlignmentRowError(f"line {line_no} is not valid JSON: {exc}") from exc

        rows.append(_row_from_mapping(data, line_no))

    return tuple(rows)

