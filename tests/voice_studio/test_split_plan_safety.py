"""Test what split planning refuses.

Session id and ratio validation, and purity. Allocation and determinism live
in `test_split_plan.py`.

A plan decides what a model never sees. Every refusal here exists because the
alternative is a silent assignment that invalidates the evaluation numbers
downstream.
"""

from pathlib import Path

import pytest

from voiceclonemlx.dataset import split_plan
from voiceclonemlx.dataset.split_plan import (
    SPLIT_RATIOS,
    SplitPlan,
    SplitPlanError,
    plan_splits,
)


class TestSessionIdValidation:
    """An id is a non-blank string, and each one appears once."""

    @pytest.mark.parametrize("value", ["", "   ", "\n", "\t"])
    def test_a_blank_id_is_refused(self, value):
        with pytest.raises(SplitPlanError, match="session"):
            plan_splits(["s1", value])

    @pytest.mark.parametrize("value", [None, 42, 3.5, True, [], {}, Path("s1")])
    def test_a_non_string_id_is_refused(self, value):
        with pytest.raises(SplitPlanError, match="session"):
            plan_splits(["s1", value])

    def test_a_duplicate_id_is_refused(self):
        """Collapsing silently would change the corpus size behind the caller."""
        with pytest.raises(SplitPlanError, match="duplicate"):
            plan_splits(["s1", "s2", "s1"])

    def test_the_duplicate_is_named(self):
        with pytest.raises(SplitPlanError, match="s2"):
            plan_splits(["s1", "s2", "s2"])

    def test_ids_differing_only_by_whitespace_are_distinct(self):
        """Ids are recorded values; nothing is stripped, so these are two."""
        plan = plan_splits(["s1", " s1"])

        assert len(plan.assignments) == 2

    def test_a_non_iterable_is_refused(self):
        with pytest.raises(SplitPlanError):
            plan_splits(42)

    def test_a_bare_string_is_refused(self):
        """A string is iterable; treating it as ids would plan one per letter."""
        with pytest.raises(SplitPlanError):
            plan_splits("s1")


class TestRatioValidation:
    """Ratios must name exactly the splits and sum to one."""

    @pytest.mark.parametrize(
        "ratios",
        [
            {"train": 0.8, "validation": 0.1},
            {"train": 1.0},
            {},
            {"train": 0.8, "validation": 0.1, "test": 0.05, "holdout": 0.05},
            {"train": 0.8, "validation": 0.1, "tset": 0.1},
        ],
    )
    def test_a_wrong_split_vocabulary_is_refused(self, ratios):
        with pytest.raises(SplitPlanError, match="split|ratio"):
            plan_splits(["s1"], ratios=ratios)

    @pytest.mark.parametrize(
        "ratios",
        [
            {"train": 0.8, "validation": 0.1, "test": 0.2},
            {"train": 0.5, "validation": 0.1, "test": 0.1},
            {"train": 0.9, "validation": 0.9, "test": 0.9},
        ],
    )
    def test_ratios_that_do_not_sum_to_one_are_refused(self, ratios):
        with pytest.raises(SplitPlanError, match="sum"):
            plan_splits(["s1"], ratios=ratios)

    @pytest.mark.parametrize("bad", [0.0, -0.1, -1.0])
    def test_a_non_positive_ratio_is_refused(self, bad):
        ratios = {"train": 0.8, "validation": 0.2 - bad, "test": bad}

        with pytest.raises(SplitPlanError):
            plan_splits(["s1"], ratios=ratios)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_ratio_is_refused(self, bad):
        ratios = {"train": 0.8, "validation": 0.1, "test": bad}

        with pytest.raises(SplitPlanError):
            plan_splits(["s1"], ratios=ratios)

    @pytest.mark.parametrize("bad", [None, "0.1", [], {}])
    def test_a_non_numeric_ratio_is_refused(self, bad):
        ratios = {"train": 0.8, "validation": 0.1, "test": bad}

        with pytest.raises(SplitPlanError, match="number|ratio"):
            plan_splits(["s1"], ratios=ratios)

    def test_bool_is_not_a_ratio(self):
        """`bool` is an `int` subclass, so `True` would otherwise pass as 1."""
        ratios = {"train": True, "validation": 0.1, "test": 0.1}

        with pytest.raises(SplitPlanError):
            plan_splits(["s1"], ratios=ratios)

    def test_a_non_mapping_ratios_is_refused(self):
        with pytest.raises(SplitPlanError):
            plan_splits(["s1"], ratios=[0.8, 0.1, 0.1])

    def test_tiny_float_drift_is_tolerated(self):
        """0.1 + 0.2 style drift must not reject an honest set of ratios."""
        ratios = {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}

        assert plan_splits(["s1"], ratios=ratios) is not None


class TestDirectConstruction:
    """`SplitPlan` is public, so every builder rule applies to it too.

    An earlier version validated only inside `plan_splits`. A hand-built plan
    could then claim ten test sessions it did not have, name a split that does
    not exist, or carry counts contradicting its own assignments — and be
    accepted, because the only checks lived in the function nobody had to call.
    """

    def valid(self, **overrides):
        fields = {
            "assignments": {"s1": "train", "s2": "validation"},
            "counts": {"train": 1, "validation": 1, "test": 0},
            "ratios": dict(SPLIT_RATIOS),
        }
        fields.update(overrides)
        return fields

    def test_a_valid_plan_still_constructs(self):
        assert SplitPlan(**self.valid()).counts["train"] == 1

    @pytest.mark.parametrize(
        "counts",
        [
            {"train": 1, "validation": 1},
            {"train": 1, "validation": 1, "test": 0, "holdout": 0},
            {},
        ],
    )
    def test_counts_must_name_exactly_the_splits(self, counts):
        with pytest.raises(SplitPlanError, match="counts"):
            SplitPlan(**self.valid(counts=counts))

    @pytest.mark.parametrize("bad", [-1, -10])
    def test_a_negative_count_is_refused(self, bad):
        with pytest.raises(SplitPlanError, match="negative"):
            SplitPlan(**self.valid(counts={"train": 1, "validation": 1, "test": bad}))

    @pytest.mark.parametrize("bad", [1.0, "1", None, [], True])
    def test_a_non_integer_count_is_refused(self, bad):
        """`True` included: `bool` is an `int` subclass."""
        with pytest.raises(SplitPlanError, match="integer"):
            SplitPlan(**self.valid(counts={"train": bad, "validation": 1, "test": 0}))

    def test_counts_must_match_the_assignments(self):
        with pytest.raises(SplitPlanError, match="assigned"):
            SplitPlan(**self.valid(counts={"train": 9, "validation": 1, "test": 0}))

    def test_a_plan_cannot_claim_sessions_it_does_not_assign(self):
        with pytest.raises(SplitPlanError, match="assigned"):
            SplitPlan(**self.valid(counts={"train": 1, "validation": 1, "test": 4}))

    @pytest.mark.parametrize("split", ["holdout", "TRAIN", "", None, 7])
    def test_an_unknown_assigned_split_is_refused(self, split):
        with pytest.raises(SplitPlanError, match="unknown split"):
            SplitPlan(
                **self.valid(
                    assignments={"s1": "train", "s2": split},
                    counts={"train": 1, "validation": 0, "test": 0},
                )
            )

    @pytest.mark.parametrize("session_id", ["", "   ", None, 42])
    def test_a_blank_or_non_string_session_id_is_refused(self, session_id):
        with pytest.raises(SplitPlanError, match="session id"):
            SplitPlan(
                **self.valid(
                    assignments={session_id: "train"},
                    counts={"train": 1, "validation": 0, "test": 0},
                )
            )

    @pytest.mark.parametrize("value", [None, 42, [("s1", "train")]])
    def test_non_mapping_assignments_are_refused(self, value):
        with pytest.raises(SplitPlanError, match="assignments"):
            SplitPlan(**self.valid(assignments=value))

    @pytest.mark.parametrize(
        "ratios",
        [
            {"train": 0.8, "validation": 0.1, "test": 0.2},
            {"train": 0.8, "validation": 0.1},
            {"train": 0.8, "validation": 0.1, "test": 0.0},
            {"train": 0.8, "validation": 0.1, "test": float("nan")},
            {"train": True, "validation": 0.1, "test": 0.1},
        ],
    )
    def test_malformed_ratios_are_refused_on_construction(self, ratios):
        with pytest.raises(SplitPlanError):
            SplitPlan(**self.valid(ratios=ratios))

    def test_an_empty_plan_is_valid(self):
        plan = SplitPlan(
            assignments={}, counts={"train": 0, "validation": 0, "test": 0}
        )

        assert dict(plan.assignments) == {}

    @pytest.mark.parametrize("field", ["assignments", "counts", "ratios"])
    def test_mappings_are_frozen_after_direct_construction(self, field):
        plan = SplitPlan(**self.valid())

        with pytest.raises(TypeError):
            getattr(plan, field)["train"] = "nonsense"

    def test_mutating_the_source_dict_cannot_reach_a_plan(self):
        assignments = {"s1": "train", "s2": "validation"}
        plan = SplitPlan(**self.valid(assignments=assignments))
        assignments["s3"] = "test"

        assert len(plan.assignments) == 2


class TestSeamIsPure:
    """No hashing, no randomness, no clock, no file, no model, no network."""

    def _imports(self):
        source = Path(split_plan.__file__).read_text(encoding="utf-8")
        return " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        ).replace("voiceclonemlx.", "")

    @pytest.mark.parametrize(
        "banned",
        ["hashlib", "random", "secrets", "socket", "urllib", "requests",
         "subprocess", "mlx", "qwen", "torch", "tkinter"],
    )
    def test_no_dangerous_import(self, banned):
        assert banned not in self._imports()

    def test_no_clock_import(self):
        """Needles built at runtime so this cannot match its own source."""
        for name in ("time", "date" + "time"):
            assert f"import {name}" not in self._imports()

    def test_no_file_is_opened(self):
        source = Path(split_plan.__file__).read_text(encoding="utf-8")

        for needle in ("open(", "read_text", "write_text", "Path("):
            assert needle not in source

    def test_assignment_is_not_hashed(self):
        """A hashed plan is stable but unverifiable by hand."""
        source = Path(split_plan.__file__).read_text(encoding="utf-8")

        for needle in ("hash(", "md5", "sha1", "sha256"):
            assert needle not in source

    def test_the_split_vocabulary_has_one_home(self):
        source = Path(split_plan.__file__).read_text(encoding="utf-8")

        assert "SPLITS" in source
        assert 'SPLITS = (' not in source
