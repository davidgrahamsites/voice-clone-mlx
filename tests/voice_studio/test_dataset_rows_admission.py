"""Test what may enter the dataset, and what the parser refuses.

Admission — only an *accepted* alignment row, with its attribution carried
across — the split vocabulary, the session-exclusivity rule that keeps train
and test apart, and full re-validation of a hand-edited manifest.

The clip-decision gate and the bypass hardening live in
`test_dataset_rows_gate.py`; field shape and serialization in
`test_dataset_rows.py`.
"""

import json
from dataclasses import dataclass

import pytest

from voiceclonemlx.alignment.alignment_rows import (
    accept_row,
    build_alignment_row,
    reject_row,
)
from voiceclonemlx.dataset.dataset_rows import (
    DATASET_SCHEMA_VERSION,
    SPLITS,
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


class TestAdmission:
    """Only an accepted alignment row may enter the dataset."""

    def test_accepted_row_is_admitted(self):
        assert build().utterance_id == "NEUTRAL-a1b2c3d4"

    def test_pending_row_is_refused(self):
        """The mirror of pending-by-construction: nothing unreviewed gets in."""
        with pytest.raises(DatasetRowError, match="accepted"):
            build(accepted_row=alignment_row())

    def test_rejected_row_is_refused(self):
        rejected = reject_row(alignment_row(), actor=ACTOR, at=AT, reason="noise")

        with pytest.raises(DatasetRowError, match="accepted"):
            build(accepted_row=rejected)

    @pytest.mark.parametrize("value", [None, 42, "row", {}])
    def test_non_alignment_row_is_refused(self, value):
        with pytest.raises(DatasetRowError, match="accepted_row"):
            build(accepted_row=value)

    def test_acceptance_attribution_is_carried_over(self):
        row = build()

        assert row.accepted_by == ACTOR
        assert row.accepted_at == AT

    def test_attribution_cannot_be_supplied_separately(self):
        """It comes from the alignment row or not at all."""
        with pytest.raises(TypeError):
            build(accepted_by="someone-else")


class TestSplit:
    """Splits are a closed set."""

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_every_split_is_accepted(self, split):
        assert build(split=split).split == split

    def test_split_vocabulary(self):
        assert set(SPLITS) == {"train", "validation", "test"}

    @pytest.mark.parametrize(
        "value", ["", "TRAIN", "val", "holdout", None, 7, True]
    )
    def test_unknown_split_rejects(self, value):
        with pytest.raises(DatasetRowError, match="split"):
            build(split=value)


class TestSessionExclusiveSplits:
    """A session may not appear in two splits."""

    def test_one_session_in_one_split_is_fine(self):
        rows = [
            build(session_id="s1", split="train"),
            build(session_id="s1", split="train"),
        ]

        assert json.loads(rows_to_json(rows))["rows"]

    def test_two_sessions_in_different_splits_are_fine(self):
        rows = [
            build(session_id="s1", split="train"),
            build(session_id="s2", split="test"),
        ]

        assert len(json.loads(rows_to_json(rows))["rows"]) == 2

    def test_a_session_in_two_splits_rejects(self):
        """Leakage between train and test invalidates every later number."""
        rows = [
            build(session_id="s1", split="train"),
            build(session_id="s1", split="test"),
        ]

        with pytest.raises(DatasetRowError, match="session"):
            rows_to_json(rows)

    def test_the_error_names_the_session_and_splits(self):
        rows = [
            build(session_id="s7", split="train"),
            build(session_id="s7", split="validation"),
        ]

        with pytest.raises(DatasetRowError) as excinfo:
            rows_to_json(rows)

        message = str(excinfo.value)
        assert "s7" in message
        assert "train" in message and "validation" in message

    def test_parse_also_enforces_exclusivity(self):
        rows = [
            build(session_id="s1", split="train"),
            build(session_id="s1", split="test"),
        ]
        document = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "rows": [json.loads(rows_to_json([r]))["rows"][0] for r in rows],
        }

        with pytest.raises(DatasetRowError, match="session"):
            parse_dataset_json(json.dumps(document))


class TestParseRejection:
    """A hand-edited manifest is re-validated in full."""

    def _document(self, **row_overrides):
        document = json.loads(rows_to_json([build()]))
        document["rows"][0].update(row_overrides)
        return json.dumps(document)

    @pytest.mark.parametrize("raw", ["", "   ", "not json", "{"])
    def test_malformed_json_rejects(self, raw):
        with pytest.raises(DatasetRowError, match="JSON"):
            parse_dataset_json(raw)

    @pytest.mark.parametrize("raw", ["[]", "null", '"text"', "42"])
    def test_document_that_is_not_an_object_rejects(self, raw):
        with pytest.raises(DatasetRowError, match="object"):
            parse_dataset_json(raw)

    def test_unknown_schema_version_rejects(self):
        with pytest.raises(DatasetRowError, match="schema_version"):
            parse_dataset_json(
                json.dumps({"schema_version": "2.0.0", "rows": []})
            )

    def test_missing_rows_rejects(self):
        with pytest.raises(DatasetRowError, match="rows"):
            parse_dataset_json(
                json.dumps({"schema_version": DATASET_SCHEMA_VERSION})
            )

    def test_unexpected_document_key_rejects(self):
        with pytest.raises(DatasetRowError, match="unexpected"):
            parse_dataset_json(
                json.dumps(
                    {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "rows": [],
                        "notes": "extra",
                    }
                )
            )

    def test_unexpected_row_key_rejects(self):
        with pytest.raises(DatasetRowError, match="unexpected"):
            parse_dataset_json(self._document(speaker="SPEAKER_00"))

    @pytest.mark.parametrize(
        "field",
        ["schema_version", "utterance_id", "text", "style", "clip_path",
         "clip_sha256", "audio_properties", "session_id", "split",
         "alignment_master_audio", "accepted_by", "accepted_at",
         "clip_decision_status", "clip_decision_reason"],
    )
    def test_missing_row_field_rejects(self, field):
        document = json.loads(rows_to_json([build()]))
        del document["rows"][0][field]

        with pytest.raises(DatasetRowError, match="missing|field"):
            parse_dataset_json(json.dumps(document))

    def test_invalid_row_field_rejects(self):
        with pytest.raises(DatasetRowError, match="split"):
            parse_dataset_json(self._document(split="holdout"))

    def test_unattributed_row_rejects(self):
        """A dataset row without an acceptance signature is not admissible."""
        with pytest.raises(DatasetRowError, match="accepted_by"):
            parse_dataset_json(self._document(accepted_by=""))
