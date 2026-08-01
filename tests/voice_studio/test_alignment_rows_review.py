"""Test the review rules and parse integrity of AlignmentManifest v1 rows.

The safety-critical half: a row is born `pending`, every decided row must name
an actor and a time, and a hand-edited manifest cannot pass an unattributed or
inconsistent decision — or an unexpected field — through the parser.

The code cannot tell whether an actor is a person — it enforces *attribution*,
not humanity. Who may appear in `reviewed_by` is a process rule, not a
guarantee this module can make.

Field validation, JSONL serialization, and purity live in
`test_alignment_rows.py`.
"""

import json

import pytest

from voiceclonegpt.alignment.alignment_rows import (
    AlignmentRow,
    REVIEW_ACCEPTED,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    AlignmentRowError,
    accept_row,
    build_alignment_row,
    parse_alignment_jsonl,
    reject_row,
    rows_to_jsonl,
)


VALID = dict(
    utterance_id="NEUTRAL-a1b2c3d4",
    expected_text="It was good to hear from you.",
    observed_text="It was good to hear from you.",
    style="neutral",
    start_s=12.5,
    end_s=15.25,
    segment_ids=("seg-3", "seg-4"),
    master_audio="01_recording/output/session-1.wav",
)

ACTOR = "alex"
AT = "2026-07-31T10:00:00Z"


def row(**overrides):
    return build_alignment_row(**{**VALID, **overrides})


class TestRowIsBornPending:
    """A row starts pending; every decision is attributed."""

    def test_new_row_is_pending(self):
        assert row().review_state == REVIEW_PENDING

    def test_no_argument_can_create_an_accepted_row(self):
        """`build_alignment_row` must not expose review_state at all."""
        with pytest.raises(TypeError):
            build_alignment_row(**VALID, review_state=REVIEW_ACCEPTED)

    def test_new_row_has_no_reviewer(self):
        created = row()

        assert created.reviewed_by is None
        assert created.reviewed_at is None

    def test_accept_marks_the_actor_and_time(self):
        accepted = accept_row(row(), actor=ACTOR, at=AT)

        assert accepted.review_state == REVIEW_ACCEPTED
        assert accepted.reviewed_by == ACTOR
        assert accepted.reviewed_at == AT

    def test_accept_returns_a_new_row(self):
        original = row()

        accept_row(original, actor=ACTOR, at=AT)

        assert original.review_state == REVIEW_PENDING

    def test_reject_records_a_reason(self):
        rejected = reject_row(row(), actor=ACTOR, at=AT, reason="mis-heard")

        assert rejected.review_state == REVIEW_REJECTED
        assert "mis-heard" in rejected.mismatch_reasons

    @pytest.mark.parametrize("actor", ["", "   ", None, 7, True])
    def test_accept_requires_a_named_actor(self, actor):
        with pytest.raises(AlignmentRowError, match="actor"):
            accept_row(row(), actor=actor, at=AT)

    @pytest.mark.parametrize("at", ["", "   ", None, 7, True])
    def test_accept_requires_a_timestamp(self, at):
        with pytest.raises(AlignmentRowError, match="at"):
            accept_row(row(), actor=ACTOR, at=at)

    @pytest.mark.parametrize("reason", ["", "   ", None, 7])
    def test_reject_requires_a_reason(self, reason):
        with pytest.raises(AlignmentRowError, match="reason"):
            reject_row(row(), actor=ACTOR, at=AT, reason=reason)

    def test_an_accepted_row_cannot_be_accepted_again(self):
        accepted = accept_row(row(), actor=ACTOR, at=AT)

        with pytest.raises(AlignmentRowError, match="pending"):
            accept_row(accepted, actor=ACTOR, at=AT)

    def test_a_rejected_row_cannot_be_accepted(self):
        rejected = reject_row(row(), actor=ACTOR, at=AT, reason="noise")

        with pytest.raises(AlignmentRowError, match="pending"):
            accept_row(rejected, actor=ACTOR, at=AT)

    def test_an_accepted_row_cannot_be_rejected(self):
        accepted = accept_row(row(), actor=ACTOR, at=AT)

        with pytest.raises(AlignmentRowError, match="pending"):
            reject_row(accepted, actor=ACTOR, at=AT, reason="changed my mind")


class TestParseRejection:
    """Parsing refuses unattributed and inconsistent decisions.

    Editing a manifest by hand is allowed; what is refused is a decision with
    no actor or time, or a pending row that claims one.
    """

    def _line(self, **overrides):
        data = json.loads(rows_to_jsonl([row()]).strip())
        data.update(overrides)
        return json.dumps(data) + "\n"

    def test_unknown_review_state_rejects(self):
        with pytest.raises(AlignmentRowError, match="review_state"):
            parse_alignment_jsonl(self._line(review_state="approved"))

    def test_missing_review_state_rejects(self):
        data = json.loads(rows_to_jsonl([row()]).strip())
        del data["review_state"]

        with pytest.raises(AlignmentRowError, match="review_state"):
            parse_alignment_jsonl(json.dumps(data))

    def test_accepted_without_a_reviewer_rejects(self):
        """An unattributed approval is refused on the way back in."""
        with pytest.raises(AlignmentRowError, match="reviewed_by"):
            parse_alignment_jsonl(self._line(review_state=REVIEW_ACCEPTED))

    def test_accepted_without_a_timestamp_rejects(self):
        with pytest.raises(AlignmentRowError, match="reviewed_at"):
            parse_alignment_jsonl(
                self._line(review_state=REVIEW_ACCEPTED, reviewed_by=ACTOR)
            )

    def test_rejected_without_a_reviewer_rejects(self):
        with pytest.raises(AlignmentRowError, match="reviewed_by"):
            parse_alignment_jsonl(self._line(review_state=REVIEW_REJECTED))

    def test_pending_with_a_reviewer_rejects(self):
        """A pending row that claims a reviewer is inconsistent evidence."""
        with pytest.raises(AlignmentRowError, match="pending"):
            parse_alignment_jsonl(self._line(reviewed_by=ACTOR, reviewed_at=AT))

    def test_unknown_schema_version_rejects(self):
        with pytest.raises(AlignmentRowError, match="schema_version"):
            parse_alignment_jsonl(self._line(schema_version="2.0.0"))

    def test_missing_schema_version_rejects(self):
        data = json.loads(rows_to_jsonl([row()]).strip())
        del data["schema_version"]

        with pytest.raises(AlignmentRowError, match="schema_version"):
            parse_alignment_jsonl(json.dumps(data))

    def test_malformed_json_line_names_its_line_number(self):
        text = rows_to_jsonl([row()]) + "not json\n"

        with pytest.raises(AlignmentRowError, match="2"):
            parse_alignment_jsonl(text)

    def test_line_that_is_not_an_object_rejects(self):
        with pytest.raises(AlignmentRowError):
            parse_alignment_jsonl("[1,2,3]\n")

    def test_invalid_field_in_a_line_rejects(self):
        with pytest.raises(AlignmentRowError, match="style"):
            parse_alignment_jsonl(self._line(style="sarcastic"))

    @pytest.mark.parametrize("value", [None, 42, [], {}])
    def test_non_string_payload_rejects(self, value):
        with pytest.raises(AlignmentRowError):
            parse_alignment_jsonl(value)


class TestDirectConstructionCannotForgeAnApproval:
    """The dataclass enforces the invariant, not just the builder.

    `build_alignment_row` returning `pending` is only half a guarantee if
    `AlignmentRow(...)` can be handed `review_state="accepted"` directly.
    """

    def _fields(self, **overrides):
        base = dict(
            schema_version="1.0.0",
            utterance_id="X",
            expected_text="a",
            observed_text="a",
            style="neutral",
            start_s=0.0,
            end_s=1.0,
            segment_ids=(),
            master_audio="a.wav",
            confidence=None,
            mismatch_reasons=(),
            review_state=REVIEW_PENDING,
        )
        base.update(overrides)
        return base

    def test_pending_row_is_constructible(self):
        assert AlignmentRow(**self._fields()).review_state == REVIEW_PENDING

    @pytest.mark.parametrize("state", [REVIEW_ACCEPTED, REVIEW_REJECTED])
    def test_decided_row_without_an_actor_is_refused(self, state):
        with pytest.raises(AlignmentRowError, match="reviewed_by"):
            AlignmentRow(**self._fields(review_state=state))

    @pytest.mark.parametrize("state", [REVIEW_ACCEPTED, REVIEW_REJECTED])
    def test_decided_row_without_a_time_is_refused(self, state):
        with pytest.raises(AlignmentRowError, match="reviewed_at"):
            AlignmentRow(**self._fields(review_state=state, reviewed_by=ACTOR))

    @pytest.mark.parametrize("state", [REVIEW_ACCEPTED, REVIEW_REJECTED])
    def test_attributed_decided_row_is_allowed(self, state):
        row = AlignmentRow(
            **self._fields(review_state=state, reviewed_by=ACTOR, reviewed_at=AT)
        )

        assert row.review_state == state

    def test_pending_row_claiming_a_reviewer_is_refused(self):
        with pytest.raises(AlignmentRowError, match="pending"):
            AlignmentRow(**self._fields(reviewed_by=ACTOR, reviewed_at=AT))

    @pytest.mark.parametrize("state", ["approved", "", None, 7])
    def test_unknown_review_state_is_refused(self, state):
        with pytest.raises(AlignmentRowError, match="review_state"):
            AlignmentRow(**self._fields(review_state=state))

    @pytest.mark.parametrize("actor", ["", "   "])
    def test_blank_actor_is_refused(self, actor):
        with pytest.raises(AlignmentRowError, match="reviewed_by"):
            AlignmentRow(
                **self._fields(
                    review_state=REVIEW_ACCEPTED, reviewed_by=actor, reviewed_at=AT
                )
            )


class TestExactKeySet:
    """A manifest line must carry the v1 keys — no more, no fewer."""

    def _line(self, **overrides):
        data = json.loads(rows_to_jsonl([row()]).strip())
        data.update(overrides)
        return json.dumps(data) + "\n"

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version", "utterance_id", "expected_text", "observed_text",
            "style", "start_s", "end_s", "segment_ids", "master_audio",
            "confidence", "mismatch_reasons", "review_state", "reviewed_by",
            "reviewed_at",
        ],
    )
    def test_missing_field_rejects(self, field):
        data = json.loads(rows_to_jsonl([row()]).strip())
        del data[field]

        with pytest.raises(AlignmentRowError, match="missing|field"):
            parse_alignment_jsonl(json.dumps(data))

    def test_unknown_field_rejects(self):
        """An extra key means the writer knew something this reader does not."""
        with pytest.raises(AlignmentRowError, match="unexpected|field"):
            parse_alignment_jsonl(self._line(speaker="SPEAKER_00"))

    def test_several_unknown_fields_are_all_named(self):
        with pytest.raises(AlignmentRowError) as excinfo:
            parse_alignment_jsonl(self._line(alpha=1, beta=2))

        message = str(excinfo.value)
        assert "alpha" in message and "beta" in message

    def test_error_names_the_line_number(self):
        text = rows_to_jsonl([row()]) + self._line(extra=1)

        with pytest.raises(AlignmentRowError, match="2"):
            parse_alignment_jsonl(text)

    def test_a_row_we_wrote_round_trips(self):
        """The writer and reader must agree on the exact key set."""
        assert parse_alignment_jsonl(rows_to_jsonl([row()]))[0] == row()

