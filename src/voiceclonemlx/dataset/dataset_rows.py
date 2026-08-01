"""The DatasetManifest v1 row: what actually reaches training.

One job: turn an **accepted** alignment row plus a produced clip into a dataset
row, and serialize the collection. It decodes no audio, computes no checksum,
reads no file, and consults no clock.

Two gates it inherits rather than re-implements:

* **Acceptance.** Only an alignment row whose `review_state` is `accepted` may
  be admitted, and its acceptance actor and time are carried across — never
  re-invented here. A `pending` row raises. This is the mirror of the
  pending-by-construction rule in `alignment_rows`: nothing unreviewed reaches
  the dataset.
* **Clip admission.** Whether a clip is single-speaker and overlap-free is
  decided by `alignment/overlap_gate.py`, which is canonical. This module does
  not duplicate that rule, and a test asserts it names neither `SpeakerTurn`
  nor `decide_clip`.

Session-exclusive splits are enforced when rows are written or parsed, not per
row: a single row cannot know what the others do. Leakage between train and
test invalidates every evaluation number that follows, so it fails loudly.

Pure and deterministic: standard library only, no clock, no randomness.
Field names follow `DatasetManifest` v1 in
`docs/architecture/voice-model-lifecycle.md`.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from voiceclonemlx.alignment.alignment_rows import REVIEW_ACCEPTED, AlignmentRow
from voiceclonemlx.dataset.dataset_row_schema import (
    ACCEPT_STATUS,
    AUDIO_FIELDS,
    AUDIO_INTEGER_FIELDS,
    DATASET_SCHEMA_VERSION,
    SPLITS,
    DatasetRowError,
    _DOCUMENT_FIELDS,
    _ROW_FIELDS,
    _accept_status,
    _audio_properties,
    _schema_version,
    _relative_path,
    _sha256,
    _split,
    _style,
    _text,
)

try:  # pragma: no cover - exercised by whichever checkout has the module
    from voiceclonemlx.alignment.overlap_gate import ClipDecision
except ImportError:  # pragma: no cover
    #: `alignment/overlap_gate.py` is on `main` but absent from some
    #: worktrees. Where it resolves the decision is type-checked; where it does
    #: not, the decision is still checked structurally and by status, so an
    #: unaccepted clip can never be admitted either way.
    ClipDecision = None

#: Re-exported so callers import the whole contract from one module.
__all__ = [
    "DATASET_SCHEMA_VERSION",
    "SPLITS",
    "AUDIO_FIELDS",
    "AUDIO_INTEGER_FIELDS",
    "ACCEPT_STATUS",
    "DatasetRow",
    "DatasetRowError",
    "build_dataset_row",
    "check_session_exclusive_splits",
    "rows_to_json",
    "parse_dataset_json",
]


@dataclass(frozen=True)
class DatasetRow:
    """One accepted utterance and the clip produced for it."""

    schema_version: str
    utterance_id: str
    text: str
    style: str
    clip_path: str
    clip_sha256: str
    audio_properties: Mapping
    session_id: str
    split: str
    alignment_master_audio: str
    accepted_by: str
    accepted_at: str
    clip_decision_status: str
    clip_decision_reason: str

    def __post_init__(self) -> None:
        """Enforce every admission invariant, whichever path built the row.

        `build_dataset_row` returning only signed, accepted rows is half a
        guarantee if `DatasetRow(...)` can be constructed with an empty
        `accepted_by` or a rejected clip decision. The same checks run here,
        so direct construction and parsing cannot bypass them.
        """
        object.__setattr__(
            self, "schema_version", _schema_version(self.schema_version)
        )
        object.__setattr__(self, "utterance_id", _text(self.utterance_id, "utterance_id"))
        object.__setattr__(self, "text", _text(self.text, "text"))
        object.__setattr__(self, "style", _style(self.style))
        object.__setattr__(
            self, "clip_path", _relative_path(self.clip_path, "clip_path")
        )
        object.__setattr__(self, "clip_sha256", _sha256(self.clip_sha256))
        object.__setattr__(
            self, "audio_properties", _audio_properties(self.audio_properties)
        )
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "split", _split(self.split))
        object.__setattr__(
            self,
            "alignment_master_audio",
            _relative_path(self.alignment_master_audio, "alignment_master_audio"),
        )
        object.__setattr__(self, "accepted_by", _text(self.accepted_by, "accepted_by"))
        object.__setattr__(self, "accepted_at", _text(self.accepted_at, "accepted_at"))
        object.__setattr__(
            self,
            "clip_decision_status",
            _accept_status(self.clip_decision_status),
        )
        object.__setattr__(
            self,
            "clip_decision_reason",
            _text(self.clip_decision_reason, "clip_decision_reason"),
        )


def _decision_evidence(clip_decision) -> tuple:
    """Return `(status, reason)` from the gate's decision, or refuse.

    Where `alignment.overlap_gate` resolves, the decision must be a real
    `ClipDecision`. Where it does not, the shape is still required — so an
    object that merely looks like one cannot smuggle a rejected clip through,
    because the status is checked regardless.
    """
    if ClipDecision is not None and not isinstance(clip_decision, ClipDecision):
        raise DatasetRowError(
            f"clip_decision must be an overlap_gate.ClipDecision: "
            f"{clip_decision!r}"
        )

    status = getattr(clip_decision, "status", None)
    reason = getattr(clip_decision, "reason", None)

    if status is None and reason is None:
        raise DatasetRowError(
            f"clip_decision must carry a status and reason: {clip_decision!r}"
        )

    return _accept_status(status), _text(reason, "clip_decision_reason")


def build_dataset_row(
    *,
    accepted_row: AlignmentRow,
    clip_path: str,
    clip_sha256: str,
    audio_properties: Mapping,
    session_id: str,
    split: str,
    clip_decision,
) -> DatasetRow:
    """Admit one accepted alignment row into the dataset.

    There is deliberately no `accepted_by`/`accepted_at` parameter: the
    attribution comes from the alignment row that was signed, so a dataset row
    cannot claim an approval that never happened.

    Args:
        accepted_row: An `AlignmentRow` whose `review_state` is `accepted`.
        clip_path: Relative path to the produced clip.
        clip_sha256: Lowercase 64-hex digest, computed by the caller.
        audio_properties: Exactly `sample_rate_hz`, `channels`, `bit_depth`,
            `duration_s`.
        session_id: Recording session the clip came from.
        split: One of `SPLITS`.
        clip_decision: The `ClipDecision` from
            `alignment/overlap_gate.py`. Required, and only `accept` admits.
            Its status and reason are recorded as evidence; the rule that
            produced them is not re-implemented here.

    Raises:
        DatasetRowError: The row is not accepted, the clip decision is missing,
            not an accept, or unreasoned, or any field is unusable.
    """
    if not isinstance(accepted_row, AlignmentRow):
        raise DatasetRowError(
            f"accepted_row must be an AlignmentRow: {accepted_row!r}"
        )

    if accepted_row.review_state != REVIEW_ACCEPTED:
        raise DatasetRowError(
            f"only an accepted alignment row may enter the dataset; "
            f"{accepted_row.utterance_id!r} is {accepted_row.review_state!r}"
        )

    status, reason = _decision_evidence(clip_decision)

    return DatasetRow(
        schema_version=DATASET_SCHEMA_VERSION,
        utterance_id=_text(accepted_row.utterance_id, "utterance_id"),
        text=accepted_row.expected_text,
        style=accepted_row.style,
        clip_path=_relative_path(clip_path, "clip_path"),
        clip_sha256=_sha256(clip_sha256),
        audio_properties=_audio_properties(audio_properties),
        session_id=_text(session_id, "session_id"),
        split=_split(split),
        alignment_master_audio=accepted_row.master_audio,
        accepted_by=_text(accepted_row.reviewed_by, "accepted_by"),
        accepted_at=_text(accepted_row.reviewed_at, "accepted_at"),
        clip_decision_status=status,
        clip_decision_reason=reason,
    )


def check_session_exclusive_splits(rows: Iterable[DatasetRow]) -> None:
    """Refuse a collection where one session appears in two splits.

    Session leakage between train and test invalidates every evaluation number
    that follows, and it is invisible once the manifest is written — so it is
    caught here rather than trusted.

    Raises:
        DatasetRowError: A session id appears under more than one split.
    """
    seen = {}
    for row in rows:
        first = seen.setdefault(row.session_id, row.split)
        if first != row.split:
            raise DatasetRowError(
                f"session {row.session_id!r} appears in more than one split: "
                f"{first!r} and {row.split!r}"
            )


def _to_mapping(row: DatasetRow) -> dict:
    """Return the JSON-ready mapping for one row."""
    return {
        "schema_version": row.schema_version,
        "utterance_id": row.utterance_id,
        "text": row.text,
        "style": row.style,
        "clip_path": row.clip_path,
        "clip_sha256": row.clip_sha256,
        "audio_properties": dict(row.audio_properties),
        "session_id": row.session_id,
        "split": row.split,
        "alignment_master_audio": row.alignment_master_audio,
        "accepted_by": row.accepted_by,
        "accepted_at": row.accepted_at,
        "clip_decision_status": row.clip_decision_status,
        "clip_decision_reason": row.clip_decision_reason,
    }


def rows_to_json(rows: Iterable[DatasetRow]) -> str:
    """Serialize rows as one JSON document.

    `DatasetManifest` v1 is a single document, not JSONL. Keys are sorted and
    non-ASCII is left unescaped, so the same rows always produce the same
    bytes and the file stays readable.

    Raises:
        DatasetRowError: A session appears in more than one split.
    """
    materialized = list(rows)
    check_session_exclusive_splits(materialized)

    document = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "rows": [_to_mapping(row) for row in materialized],
    }
    return json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2)


def _row_from_mapping(data: Any, index: int) -> DatasetRow:
    """Rebuild and re-validate one row from a parsed document."""
    where = f"row {index}"

    if not isinstance(data, Mapping):
        raise DatasetRowError(f"{where} is not a JSON object")

    present = set(data)
    expected = set(_ROW_FIELDS)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise DatasetRowError(
            f"{where} is missing required field(s): {', '.join(missing)}"
        )
    if unexpected:
        raise DatasetRowError(
            f"{where} has unexpected field(s): {', '.join(unexpected)}"
        )

    if data["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetRowError(
            f"{where} has unsupported schema_version: "
            f"{data['schema_version']!r}"
        )

    return DatasetRow(
        schema_version=DATASET_SCHEMA_VERSION,
        utterance_id=_text(data["utterance_id"], "utterance_id"),
        text=_text(data["text"], "text"),
        style=_style(data["style"]),
        clip_path=_relative_path(data["clip_path"], "clip_path"),
        clip_sha256=_sha256(data["clip_sha256"]),
        audio_properties=_audio_properties(data["audio_properties"]),
        session_id=_text(data["session_id"], "session_id"),
        split=_split(data["split"]),
        alignment_master_audio=_relative_path(
            data["alignment_master_audio"], "alignment_master_audio"
        ),
        accepted_by=_text(data["accepted_by"], "accepted_by"),
        accepted_at=_text(data["accepted_at"], "accepted_at"),
        clip_decision_status=data["clip_decision_status"],
        clip_decision_reason=data["clip_decision_reason"],
    )


def parse_dataset_json(text) -> Tuple[DatasetRow, ...]:
    """Parse a dataset manifest back into validated rows.

    Every row is re-validated and the session-exclusivity rule is re-checked,
    so a hand-edited manifest cannot introduce leakage or an unattributed row.

    Raises:
        DatasetRowError: Malformed JSON, an unknown schema version, an
            unexpected or missing field, an invalid value, or split leakage.
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatasetRowError(f"manifest is not valid UTF-8: {exc}") from exc

    if not isinstance(text, str):
        raise DatasetRowError(f"manifest must be text: {type(text).__name__}")

    try:
        document = json.loads(text)
    except ValueError as exc:
        raise DatasetRowError(f"manifest is not valid JSON: {exc}") from exc

    if not isinstance(document, Mapping):
        raise DatasetRowError("manifest is not a JSON object")

    present = set(document)
    expected = set(_DOCUMENT_FIELDS)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise DatasetRowError(
            f"manifest is missing: {', '.join(missing)}"
        )
    if unexpected:
        raise DatasetRowError(
            f"manifest has unexpected key(s): {', '.join(unexpected)}"
        )

    if document["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetRowError(
            f"unsupported schema_version: {document['schema_version']!r}"
        )

    raw_rows = document["rows"]
    if isinstance(raw_rows, bool) or not isinstance(raw_rows, list):
        raise DatasetRowError(
            f"'rows' must be a list: {type(raw_rows).__name__}"
        )

    rows = tuple(
        _row_from_mapping(raw, index) for index, raw in enumerate(raw_rows)
    )
    check_session_exclusive_splits(rows)
    return rows
