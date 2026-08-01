"""Test deterministic split assignment.

Allocation, determinism, and the plan contract. Refusals and purity live in
`test_split_plan_safety.py`.

Nothing here hashes, randomizes, reads a clock, opens a file, or touches a
model: a plan is arithmetic over sorted session ids.
"""

import dataclasses

import pytest

from voiceclonegpt.dataset.dataset_rows import SPLITS
from voiceclonegpt.dataset.split_plan import (
    SPLIT_RATIOS,
    SplitPlan,
    SplitPlanError,
    plan_splits,
    split_for,
)


def sessions(n, prefix="s"):
    """`n` session ids, zero-padded so sorted order is obvious."""
    return [f"{prefix}{i:03d}" for i in range(n)]


class TestAllocation:
    """Counts follow the ratios by largest remainder."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (10, {"train": 8, "validation": 1, "test": 1}),
            (20, {"train": 16, "validation": 2, "test": 2}),
            (30, {"train": 24, "validation": 3, "test": 3}),
            (100, {"train": 80, "validation": 10, "test": 10}),
        ],
    )
    def test_clean_ratios_allocate_exactly(self, n, expected):
        assert dict(plan_splits(sessions(n)).counts) == expected

    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, {"train": 1, "validation": 0, "test": 0}),
            (2, {"train": 2, "validation": 0, "test": 0}),
            (3, {"train": 3, "validation": 0, "test": 0}),
            (4, {"train": 3, "validation": 1, "test": 0}),
            (5, {"train": 4, "validation": 1, "test": 0}),
            (7, {"train": 5, "validation": 1, "test": 1}),
        ],
    )
    def test_small_corpora_favour_train(self, n, expected):
        """Leftover seats go to the largest remainder, ties by SPLITS order.

        A tiny corpus lands entirely in `train`. That is the honest outcome:
        one session cannot be both trained on and evaluated against, and
        inventing a one-clip test split would produce a meaningless number.
        """
        assert dict(plan_splits(sessions(n)).counts) == expected

    @pytest.mark.parametrize("n", [1, 2, 5, 9, 17, 50, 101])
    def test_every_session_is_assigned_exactly_once(self, n):
        plan = plan_splits(sessions(n))

        assert len(plan.assignments) == n
        assert sum(plan.counts.values()) == n

    @pytest.mark.parametrize("n", [1, 3, 8, 23, 64])
    def test_counts_match_the_assignments(self, n):
        plan = plan_splits(sessions(n))

        for split in SPLITS:
            actual = sum(1 for value in plan.assignments.values() if value == split)
            assert plan.counts[split] == actual

    def test_an_empty_corpus_yields_an_empty_plan(self):
        plan = plan_splits([])

        assert dict(plan.assignments) == {}
        assert dict(plan.counts) == {"train": 0, "validation": 0, "test": 0}

    def test_every_split_is_present_in_counts_even_at_zero(self):
        assert set(plan_splits(sessions(1)).counts) == set(SPLITS)


class TestDeterminism:
    """The same corpus always yields the same plan."""

    def test_two_runs_agree(self):
        first = plan_splits(sessions(17))
        second = plan_splits(sessions(17))

        assert dict(first.assignments) == dict(second.assignments)

    def test_input_order_does_not_matter(self):
        ids = sessions(17)
        forward = plan_splits(ids)
        backward = plan_splits(list(reversed(ids)))

        assert dict(forward.assignments) == dict(backward.assignments)

    def test_a_shuffled_input_agrees(self):
        ids = sessions(12)
        scrambled = ids[7:] + ids[:3] + ids[3:7]

        assert dict(plan_splits(scrambled).assignments) == dict(
            plan_splits(ids).assignments
        )

    def test_any_iterable_is_accepted(self):
        ids = sessions(9)

        assert dict(plan_splits(tuple(ids)).assignments) == dict(
            plan_splits(iter(ids)).assignments
        )

    def test_assignment_follows_sorted_order(self):
        """Sorted ids fill train, then validation, then test."""
        plan = plan_splits(sessions(10))

        assert plan.assignments["s000"] == "train"
        assert plan.assignments["s008"] == "validation"
        assert plan.assignments["s009"] == "test"

    def test_ids_are_sorted_by_bytes_not_by_length(self):
        plan = plan_splits(["s10", "s9", "s1"])

        assert list(plan.assignments) == ["s1", "s10", "s9"]


class TestCustomRatios:
    """Ratios are overridable per call."""

    def test_an_even_split_is_honoured(self):
        ratios = {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}

        counts = plan_splits(sessions(9), ratios=ratios).counts

        assert dict(counts) == {"train": 3, "validation": 3, "test": 3}

    def test_a_heavier_test_split_is_honoured(self):
        ratios = {"train": 0.5, "validation": 0.2, "test": 0.3}

        counts = plan_splits(sessions(10), ratios=ratios).counts

        assert dict(counts) == {"train": 5, "validation": 2, "test": 3}

    def test_the_default_is_eighty_ten_ten(self):
        assert SPLIT_RATIOS == {"train": 0.8, "validation": 0.1, "test": 0.1}

    def test_the_requested_ratios_are_recorded(self):
        ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

        assert dict(plan_splits(sessions(5), ratios=ratios).ratios) == ratios

    def test_the_default_ratios_are_recorded_when_not_given(self):
        assert dict(plan_splits(sessions(5)).ratios) == SPLIT_RATIOS

    def test_the_module_default_cannot_be_mutated_through_a_plan(self):
        plan = plan_splits(sessions(5))

        with pytest.raises(TypeError):
            plan.ratios["train"] = 0.9

        assert SPLIT_RATIOS["train"] == 0.8


class TestSplitFor:
    """Looking a session up is explicit, never a guess."""

    def test_returns_the_assigned_split(self):
        plan = plan_splits(sessions(10))

        assert split_for(plan, "s000") == "train"

    def test_agrees_with_the_assignments_mapping(self):
        plan = plan_splits(sessions(10))

        for session_id, split in plan.assignments.items():
            assert split_for(plan, session_id) == split

    def test_an_unknown_session_raises(self):
        plan = plan_splits(sessions(3))

        with pytest.raises(SplitPlanError, match="unknown"):
            split_for(plan, "s999")

    def test_an_unknown_session_is_not_silently_trained_on(self):
        """A default of `train` is exactly how leakage gets in."""
        plan = plan_splits(sessions(3))

        with pytest.raises(SplitPlanError):
            split_for(plan, "")


class TestPlanContract:
    """The plan is a small immutable value."""

    def test_plan_is_frozen(self):
        plan = plan_splits(sessions(5))

        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.counts = {}

    @pytest.mark.parametrize("field", ["assignments", "counts", "ratios"])
    def test_mappings_are_immutable(self, field):
        plan = plan_splits(sessions(5))

        with pytest.raises(TypeError):
            getattr(plan, field)["train"] = "nonsense"

    def test_mutating_the_caller_list_does_not_change_a_plan(self):
        ids = sessions(5)
        plan = plan_splits(ids)
        ids.append("s999")

        assert len(plan.assignments) == 5

    def test_mutating_the_caller_ratios_does_not_change_a_plan(self):
        ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
        plan = plan_splits(sessions(5), ratios=ratios)
        ratios["train"] = 0.99

        assert plan.ratios["train"] == 0.8

    def test_plan_is_constructible_directly(self):
        plan = SplitPlan(
            assignments={"s1": "train"},
            counts={"train": 1, "validation": 0, "test": 0},
            ratios=SPLIT_RATIOS,
        )

        assert plan.assignments["s1"] == "train"


class TestFeedsTheDatasetContract:
    """A plan-derived manifest satisfies session exclusivity by construction."""

    def test_no_session_appears_in_two_splits(self):
        plan = plan_splits(sessions(30))
        seen = {}

        for session_id, split in plan.assignments.items():
            seen.setdefault(session_id, split)
            assert seen[session_id] == split

    def test_every_assigned_split_is_in_the_dataset_vocabulary(self):
        plan = plan_splits(sessions(30))

        assert set(plan.assignments.values()) <= set(SPLITS)

    def test_the_split_vocabulary_is_borrowed_not_restated(self):
        from voiceclonegpt.dataset import split_plan

        assert set(split_plan.SPLIT_RATIOS) == set(SPLITS)
