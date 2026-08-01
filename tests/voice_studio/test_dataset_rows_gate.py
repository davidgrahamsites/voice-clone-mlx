"""Test the clip-decision gate and the bypass hardening around it.

Two things live here. First, that `alignment/overlap_gate.py`'s decision is
required and only `accept` admits — with its status and reason recorded as
evidence. Second, that none of the admission invariants can be sidestepped:
not by constructing `DatasetRow` directly, and not by hand-editing a manifest.

Admission of accepted alignment rows, splits, and session exclusivity live in
`test_dataset_rows_admission.py`; field shape and serialization in
`test_dataset_rows.py`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from voiceclonemlx.alignment.alignment_rows import accept_row, build_alignment_row
from voiceclonemlx.dataset import dataset_rows
from voiceclonemlx.dataset.dataset_rows import (
    ACCEPT_STATUS,
    DATASET_SCHEMA_VERSION,
    DatasetRowError,
    build_dataset_row,
    parse_dataset_json,
    rows_to_json,
)


try:
    from voiceclonemlx.alignment.overlap_gate import ClipDecision
except ImportError:  # pragma: no cover - child worktrees lack the module
    # `alignment/overlap_gate.py` is on `main` but absent from some worktrees.
    # This mirrors its shape exactly so the same assertions run either way;
    # where the real module resolves, it is used and the production
    # `isinstance` check applies.
    @dataclass(frozen=True)
    class ClipDecision:  # type: ignore[no-redef]
        status: str
        reason: str

        @classmethod
        def accept(cls, reason: str) -> "ClipDecision":
            return cls("accept", reason)

        @classmethod
        def reject(cls, reason: str) -> "ClipDecision":
            return cls("reject", reason)


def accept_decision(reason="target_only"):
    """The canonical accept decision used by most tests."""
    return ClipDecision.accept(reason)


def reject_decision(reason="overlap_detected"):
    return ClipDecision.reject(reason)

ACTOR = "alex"
AT = "2026-07-31T10:00:00Z"
SHA = "a" * 64

AUDIO = {
    "sample_rate_hz": 24000,
    "channels": 1,
    "bit_depth": 16,
    "duration_s": 2.75,
}


def alignment_row(**overrides):
    base = dict(
        utterance_id="NEUTRAL-a1b2c3d4",
        expected_text="It was good to hear from you.",
        observed_text="It was good to hear from you.",
        style="neutral",
        start_s=12.5,
        end_s=15.25,
        segment_ids=(),
        master_audio="01_recording/output/session-1.wav",
    )
    base.update(overrides)
    return build_alignment_row(**base)


def accepted(**overrides):
    return accept_row(alignment_row(**overrides), actor=ACTOR, at=AT)


def build(**overrides):
    kwargs = dict(
        accepted_row=accepted(),
        clip_path="04_dataset/output/audio/neutral/NEUTRAL-a1b2c3d4.wav",
        clip_sha256=SHA,
        audio_properties=dict(AUDIO),
        session_id="session-1",
        split="train",
        clip_decision=accept_decision(),
    )
    kwargs.update(overrides)
    return build_dataset_row(**kwargs)


class TestClipDecision:
    """The gate's verdict is required, and only `accept` admits."""

    def test_accept_decision_admits(self):
        assert build(clip_decision=accept_decision()).clip_decision_status == "accept"

    def test_decision_reason_is_recorded(self):
        row = build(clip_decision=accept_decision("target_only"))

        assert row.clip_decision_reason == "target_only"

    def test_reject_decision_is_refused(self):
        """A clip the gate refused cannot be talked into the dataset."""
        with pytest.raises(DatasetRowError, match="clip_decision"):
            build(clip_decision=reject_decision())

    @pytest.mark.parametrize(
        "status", ["", "accepted", "ACCEPT", "pending", "reject", None, 7, True]
    )
    def test_only_the_accept_status_admits(self, status):
        with pytest.raises(DatasetRowError, match="clip_decision"):
            build(clip_decision=ClipDecision(status, "why"))

    @pytest.mark.parametrize("reason", ["", "   ", None, 7, True])
    def test_a_decision_without_a_reason_is_refused(self, reason):
        """Evidence with no stated reason is not evidence."""
        with pytest.raises(DatasetRowError, match="reason"):
            build(clip_decision=ClipDecision("accept", reason))

    def test_the_decision_is_required(self):
        with pytest.raises(TypeError):
            build_dataset_row(
                accepted_row=accepted(),
                clip_path="a.wav",
                clip_sha256=SHA,
                audio_properties={
                    "sample_rate_hz": 24000,
                    "channels": 1,
                    "bit_depth": 16,
                    "duration_s": 1.0,
                },
                session_id="s1",
                split="train",
            )

    @pytest.mark.parametrize("value", [None, 42, "accept", {}, []])
    def test_a_non_decision_is_refused(self, value):
        with pytest.raises(DatasetRowError, match="clip_decision"):
            build(clip_decision=value)

    def test_accept_status_constant(self):
        assert ACCEPT_STATUS == "accept"

    def test_the_gate_is_not_reimplemented(self):
        """Evidence is recorded; the rule stays in `overlap_gate`."""
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")

        assert "class " + "SpeakerTurn" not in source
        assert "def " + "decide_clip" not in source


class TestCanonicalGateType:
    """The canonical `ClipDecision` is enforced wherever it can be imported."""

    def test_the_module_attempts_the_canonical_import(self):
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")

        assert "overlap_gate" in source
        assert "ClipDecision" in source

    def test_the_type_check_is_active_when_the_module_resolves(self):
        """A look-alike is refused wherever the canonical type resolves.

        In a checkout without `overlap_gate` the check is structural instead,
        so this skips there.
        """
        if dataset_rows.ClipDecision is None:
            pytest.skip("alignment.overlap_gate is not present in this checkout")

        class ForeignDecision:
            """Right shape, wrong type — must not be accepted."""

            status = "accept"
            reason = "target_only"

        with pytest.raises(DatasetRowError, match="clip_decision"):
            build(clip_decision=ForeignDecision())


class TestDirectConstructionCannotBypassAdmission:
    """The dataclass enforces the invariants, not just the builder."""

    def _fields(self, **overrides):
        base = dict(
            schema_version=DATASET_SCHEMA_VERSION,
            utterance_id="X",
            text="a",
            style="neutral",
            clip_path="a.wav",
            clip_sha256=SHA,
            audio_properties={
                "sample_rate_hz": 24000,
                "channels": 1,
                "bit_depth": 16,
                "duration_s": 1.0,
            },
            session_id="s1",
            split="train",
            alignment_master_audio="m.wav",
            accepted_by=ACTOR,
            accepted_at=AT,
            clip_decision_status=ACCEPT_STATUS,
            clip_decision_reason="target_only",
        )
        base.update(overrides)
        return base

    def test_a_well_formed_row_is_constructible(self):
        assert dataset_rows.DatasetRow(**self._fields()).split == "train"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_unsigned_admission_is_refused(self, value):
        with pytest.raises(DatasetRowError, match="accepted_by"):
            dataset_rows.DatasetRow(**self._fields(accepted_by=value))

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_undated_admission_is_refused(self, value):
        with pytest.raises(DatasetRowError, match="accepted_at"):
            dataset_rows.DatasetRow(**self._fields(accepted_at=value))

    @pytest.mark.parametrize("status", ["reject", "", None, "ACCEPT"])
    def test_a_non_accept_decision_is_refused(self, status):
        with pytest.raises(DatasetRowError, match="clip_decision"):
            dataset_rows.DatasetRow(**self._fields(clip_decision_status=status))

    def test_a_decision_without_a_reason_is_refused(self):
        with pytest.raises(DatasetRowError, match="reason"):
            dataset_rows.DatasetRow(**self._fields(clip_decision_reason=""))

    @pytest.mark.parametrize("value", ["sarcastic", "NEUTRAL", "netural", ""])
    def test_a_non_canonical_style_is_refused(self, value):
        with pytest.raises(DatasetRowError, match="style"):
            dataset_rows.DatasetRow(**self._fields(style=value))

    @pytest.mark.parametrize("value", ["holdout", "", None])
    def test_an_unknown_split_is_refused(self, value):
        with pytest.raises(DatasetRowError, match="split"):
            dataset_rows.DatasetRow(**self._fields(split=value))

    def test_a_bad_checksum_is_refused(self):
        with pytest.raises(DatasetRowError, match="clip_sha256"):
            dataset_rows.DatasetRow(**self._fields(clip_sha256="A" * 64))

    def test_an_absolute_clip_path_is_refused(self):
        with pytest.raises(DatasetRowError, match="clip_path"):
            dataset_rows.DatasetRow(**self._fields(clip_path="/abs/a.wav"))

    @pytest.mark.parametrize(
        "value", ["9.9.9", "2.0.0", "1.0", "", None, 7, True]
    )
    def test_a_foreign_schema_version_is_refused(self, value):
        """A row cannot claim a schema this code does not implement.

        Every other field was validated in `__post_init__` while
        `schema_version` was taken on trust, so direct construction could mint
        a row labelled with a version whose rules were never applied.
        """
        with pytest.raises(DatasetRowError, match="schema_version"):
            dataset_rows.DatasetRow(**self._fields(schema_version=value))

    def test_the_current_schema_version_is_accepted(self):
        row = dataset_rows.DatasetRow(
            **self._fields(schema_version=DATASET_SCHEMA_VERSION)
        )

        assert row.schema_version == DATASET_SCHEMA_VERSION

    def test_audio_properties_are_frozen_by_construction(self):
        row = dataset_rows.DatasetRow(**self._fields())

        with pytest.raises(TypeError):
            row.audio_properties["channels"] = 2


class TestParsedRowsCannotBypassAdmission:
    """A hand-edited manifest faces the same invariants."""

    def _document(self, **row_overrides):
        document = json.loads(rows_to_json([build()]))
        document["rows"][0].update(row_overrides)
        return json.dumps(document)

    def test_decision_fields_round_trip(self):
        restored = parse_dataset_json(rows_to_json([build()]))[0]

        assert restored.clip_decision_status == ACCEPT_STATUS
        assert restored.clip_decision_reason == "target_only"

    def test_a_rejected_decision_in_the_file_is_refused(self):
        with pytest.raises(DatasetRowError, match="clip_decision"):
            parse_dataset_json(self._document(clip_decision_status="reject"))

    def test_a_blank_decision_reason_in_the_file_is_refused(self):
        with pytest.raises(DatasetRowError, match="reason"):
            parse_dataset_json(self._document(clip_decision_reason=""))

    def test_a_non_canonical_style_in_the_file_is_refused(self):
        with pytest.raises(DatasetRowError, match="style"):
            parse_dataset_json(self._document(style="sarcastic"))

    @pytest.mark.parametrize(
        "field", ["clip_decision_status", "clip_decision_reason"]
    )
    def test_missing_decision_field_is_refused(self, field):
        document = json.loads(rows_to_json([build()]))
        del document["rows"][0][field]

        with pytest.raises(DatasetRowError, match="missing|field"):
            parse_dataset_json(json.dumps(document))
