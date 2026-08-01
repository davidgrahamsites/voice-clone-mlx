"""Test the AlignmentManifest v1 row shape.

Contract fields, validation, text preservation, JSONL serialization, the
immutable result, and purity. The review rules — pending by default, the
attributed transitions, and rejection of unattributed or inconsistent
decisions on parse — live in `test_alignment_rows_review.py`.

Pure artifact: no audio, no model, no filesystem, no clock.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from voiceclonegpt.alignment import alignment_row_schema, alignment_rows
from voiceclonegpt.alignment.alignment_rows import (
    ALIGNMENT_SCHEMA_VERSION,
    REVIEW_ACCEPTED,
    REVIEW_PENDING,
    AlignmentRow,
    AlignmentRowError,
    accept_row,
    build_alignment_row,
    parse_alignment_jsonl,
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


class TestSchema:
    """Fields match AlignmentManifest v1 in the lifecycle contract."""

    def test_schema_version_is_recorded_per_row(self):
        assert row().schema_version == ALIGNMENT_SCHEMA_VERSION

    def test_schema_version_is_semver_one(self):
        assert ALIGNMENT_SCHEMA_VERSION.startswith("1.")

    def test_contract_fields_are_present(self):
        serialized = json.loads(rows_to_jsonl([row()]).strip())

        for field in (
            "schema_version", "utterance_id", "segment_ids", "master_audio",
            "start_s", "end_s", "expected_text", "observed_text", "style",
            "confidence", "mismatch_reasons", "review_state",
            "reviewed_by", "reviewed_at",
        ):
            assert field in serialized

    def test_confidence_defaults_to_none(self):
        assert row().confidence is None

    def test_confidence_is_recorded(self):
        assert row(confidence=0.87).confidence == 0.87

    def test_mismatch_reasons_default_to_empty(self):
        assert row().mismatch_reasons == ()

    def test_mismatch_reasons_are_recorded(self):
        created = row(mismatch_reasons=("text_differs", "low_confidence"))

        assert created.mismatch_reasons == ("text_differs", "low_confidence")

    def test_segment_ids_may_be_empty(self):
        """Nothing was transcribed for this utterance; that is reviewable."""
        assert row(segment_ids=()).segment_ids == ()


class TestTextPreservation:
    """Expected and observed text are evidence; they are not normalized."""

    @pytest.mark.parametrize(
        "text", ["  leading", "trailing  ", "double  space", "Café 🎤", "a\nb"]
    )
    def test_expected_text_is_exact(self, text):
        assert row(expected_text=text).expected_text == text

    @pytest.mark.parametrize(
        "text", ["  leading", "trailing  ", "double  space", "Café 🎤"]
    )
    def test_observed_text_is_exact(self, text):
        assert row(observed_text=text).observed_text == text

    def test_text_survives_a_jsonl_round_trip(self):
        created = row(expected_text="  spaced  ", observed_text="Café 🎤")

        restored = parse_alignment_jsonl(rows_to_jsonl([created]))[0]

        assert restored.expected_text == "  spaced  "
        assert restored.observed_text == "Café 🎤"

    def test_observed_text_may_be_empty(self):
        """Silence transcribed as nothing is a real, reviewable outcome."""
        assert row(observed_text="").observed_text == ""


class TestValidation:
    """A row that cannot be reviewed is refused at construction."""

    @pytest.mark.parametrize("value", ["", "   ", None, 7, True])
    def test_bad_utterance_id_rejects(self, value):
        with pytest.raises(AlignmentRowError, match="utterance_id"):
            row(utterance_id=value)

    @pytest.mark.parametrize("value", ["", "   ", None, 7, True])
    def test_bad_expected_text_rejects(self, value):
        with pytest.raises(AlignmentRowError, match="expected_text"):
            row(expected_text=value)

    @pytest.mark.parametrize("value", [None, 7, True, []])
    def test_non_string_observed_text_rejects(self, value):
        with pytest.raises(AlignmentRowError, match="observed_text"):
            row(observed_text=value)

    @pytest.mark.parametrize("value", ["", "sarcastic", "NEUTRAL", None, 7])
    def test_unknown_style_rejects(self, value):
        with pytest.raises(AlignmentRowError, match="style"):
            row(style=value)

    def test_netural_typo_is_not_silently_accepted(self):
        """Normalization belongs upstream; a manifest records canonical only."""
        with pytest.raises(AlignmentRowError, match="style"):
            row(style="netural")

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), -1.0, "1.0", None, True]
    )
    def test_bad_times_reject(self, value):
        with pytest.raises(AlignmentRowError, match="start_s|end_s"):
            row(start_s=value)

    def test_end_before_start_rejects(self):
        with pytest.raises(AlignmentRowError, match="end_s"):
            row(start_s=5.0, end_s=1.0)

    def test_zero_length_span_rejects(self):
        with pytest.raises(AlignmentRowError, match="end_s"):
            row(start_s=1.0, end_s=1.0)

    @pytest.mark.parametrize("value", [None, 7, "seg-1", True])
    def test_non_sequence_segment_ids_reject(self, value):
        with pytest.raises(AlignmentRowError, match="segment_ids"):
            row(segment_ids=value)

    @pytest.mark.parametrize("value", [("", ), ("  ",), (7,), (None,)])
    def test_bad_segment_id_entries_reject(self, value):
        with pytest.raises(AlignmentRowError, match="segment_ids"):
            row(segment_ids=value)

    @pytest.mark.parametrize("value", ["", "   ", None, 7])
    def test_bad_master_audio_rejects(self, value):
        with pytest.raises(AlignmentRowError, match="master_audio"):
            row(master_audio=value)

    @pytest.mark.parametrize(
        "value",
        ["/abs/session.wav", "../escape.wav", "a/../../b.wav", "C:\\x.wav"],
    )
    def test_master_audio_must_be_a_relative_path(self, value):
        """The contract forbids absolute paths inside a manifest."""
        with pytest.raises(AlignmentRowError, match="master_audio"):
            row(master_audio=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), "0.5", True])
    def test_bad_confidence_rejects(self, value):
        with pytest.raises(AlignmentRowError, match="confidence"):
            row(confidence=value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_confidence_bounds_are_inclusive(self, value):
        assert row(confidence=value).confidence == value

    @pytest.mark.parametrize("value", [None, "text", 7, (7,), ("",)])
    def test_bad_mismatch_reasons_reject(self, value):
        with pytest.raises(AlignmentRowError, match="mismatch_reasons"):
            row(mismatch_reasons=value)


class TestJsonl:
    """The manifest is one JSON object per line."""

    def test_one_line_per_row(self):
        text = rows_to_jsonl([row(), row(utterance_id="WARM-1")])

        assert len(text.strip().split("\n")) == 2

    def test_empty_rows_produce_empty_text(self):
        assert rows_to_jsonl([]) == ""

    def test_round_trip_preserves_every_field(self):
        created = row(confidence=0.9, mismatch_reasons=("text_differs",))

        restored = parse_alignment_jsonl(rows_to_jsonl([created]))[0]

        assert restored == created

    def test_round_trip_preserves_review_state(self):
        accepted = accept_row(row(), actor=ACTOR, at=AT)

        restored = parse_alignment_jsonl(rows_to_jsonl([accepted]))[0]

        assert restored.review_state == REVIEW_ACCEPTED
        assert restored.reviewed_by == ACTOR

    def test_serialization_is_deterministic(self):
        assert rows_to_jsonl([row()]) == rows_to_jsonl([row()])

    def test_keys_are_sorted(self):
        line = rows_to_jsonl([row()]).strip()
        keys = list(json.loads(line).keys())

        assert keys == sorted(keys)

    def test_non_ascii_is_not_escaped(self):
        text = rows_to_jsonl([row(observed_text="Café 🎤")])

        assert "Café 🎤" in text

    def test_blank_lines_are_ignored_on_parse(self):
        text = rows_to_jsonl([row()]) + "\n\n"

        assert len(parse_alignment_jsonl(text)) == 1

    def test_parse_empty_text_gives_no_rows(self):
        assert parse_alignment_jsonl("") == ()

    def test_parse_returns_a_tuple(self):
        assert isinstance(parse_alignment_jsonl(rows_to_jsonl([row()])), tuple)


class TestResultContract:
    """Rows are immutable values."""

    def test_row_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            row().review_state = REVIEW_ACCEPTED

    def test_equal_rows_compare_equal(self):
        assert row() == row()

    def test_segment_ids_and_reasons_are_tuples(self):
        created = row(segment_ids=["a", "b"], mismatch_reasons=["x"])

        assert isinstance(created.segment_ids, tuple)
        assert isinstance(created.mismatch_reasons, tuple)

    def test_times_are_floats(self):
        created = row(start_s=1, end_s=2)

        assert isinstance(created.start_s, float)
        assert isinstance(created.end_s, float)

    def test_pending_row_is_constructible_directly(self):
        """Direct construction is allowed, but the invariant still applies —
        see `test_alignment_rows_review.py` for the decided-state cases."""
        assert AlignmentRow(
            schema_version=ALIGNMENT_SCHEMA_VERSION,
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
        ).utterance_id == "X"


class TestSeamIsPure:
    """No clock, no filesystem, no audio, no model, no network.

    Both modules are checked: splitting the contract in two must not create a
    corner where a dependency can hide.
    """

    def _sources(self):
        return {
            "alignment_rows": Path(alignment_rows.__file__).read_text(
                encoding="utf-8"
            ),
            "alignment_row_schema": Path(
                alignment_row_schema.__file__
            ).read_text(encoding="utf-8"),
        }

    def test_imports_nothing_dangerous(self):
        for name, source in self._sources().items():
            imports = " ".join(
                line for line in source.splitlines()
                if line.startswith(("import ", "from "))
            )

            for banned in ("os", "pathlib", "datetime", "time", "random",
                           "wave", "numpy", "mlx", "urllib", "socket",
                           "subprocess", "tkinter"):
                assert banned not in imports, f"{name} imports {banned}"

    def test_no_hidden_clock(self):
        """Timestamps are supplied by the caller so rows are reproducible."""
        for name, source in self._sources().items():
            for banned in ("now(", "utcnow", "time()", "uuid"):
                assert banned not in source, f"{name} contains {banned}"

    def test_styles_have_one_home(self):
        """The canonical list is imported, never restated."""
        schema = self._sources()["alignment_row_schema"]

        assert "CANONICAL_STYLES" in schema
        assert '"somber"' not in schema

    def test_the_row_module_does_not_restate_the_style_list(self):
        assert "CANONICAL_STYLES" not in self._sources()["alignment_rows"]

    def test_imports_no_app_or_ui(self):
        for name, source in self._sources().items():
            assert "studio_app" not in source, name
            assert "reader_app" not in source, name

    def test_public_names_are_importable_from_the_row_module(self):
        """The split must not move the public import surface."""
        for name in (
            "ALIGNMENT_SCHEMA_VERSION", "REVIEW_PENDING", "REVIEW_ACCEPTED",
            "REVIEW_REJECTED", "REVIEW_STATES", "AlignmentRow",
            "AlignmentRowError", "build_alignment_row", "accept_row",
            "reject_row", "rows_to_jsonl", "parse_alignment_jsonl",
        ):
            assert hasattr(alignment_rows, name), name

    def test_the_error_type_is_shared_not_duplicated(self):
        assert (
            alignment_rows.AlignmentRowError
            is alignment_row_schema.AlignmentRowError
        )

