"""Decide which sessions go to train, validation, and test.

One job: turn a set of session ids into a session-exclusive split assignment.
It reads nothing, writes nothing, hashes nothing, and randomizes nothing — a
plan is arithmetic over sorted ids, so the same corpus always yields the same
answer and a reviewer can check it by hand.

**Why not a hash.** Hashing session ids would also be stable, and it is the
usual trick. But nobody can verify a hashed assignment by reading the manifest;
they can only re-run the code. Sorted largest-remainder allocation is auditable
with a pencil, and a split decision is exactly the kind of thing that should be
checkable without trusting the tool that made it.

**Session exclusivity is structural here.** Each id is assigned once, so a
plan-derived manifest cannot trip `dataset_rows.check_session_exclusive_splits`.
That check remains the authority for manifests built any other way; this module
does not restate it.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from voiceclonemlx.dataset.dataset_rows import SPLITS

#: The default corpus shape. Overridable per call; recorded on every plan so a
#: manifest's provenance shows what was asked for, not only what came out.
SPLIT_RATIOS = MappingProxyType({"train": 0.8, "validation": 0.1, "test": 0.1})

#: How far a set of ratios may sum from 1.0. Three thirds do not sum to one in
#: binary floating point, and refusing an honest set of ratios over the last
#: bit would be a bug, not rigour.
RATIO_TOLERANCE = 1e-9

#: Nudge applied before flooring, so a product like `0.8 * 5` landing at
#: 3.9999999999999996 floors to 4 rather than 3. Smaller than any ratio
#: difference that could matter and larger than the drift it corrects.
FLOOR_EPSILON = 1e-9


class SplitPlanError(ValueError):
    """A corpus or a set of ratios cannot be planned."""


@dataclass(frozen=True)
class SplitPlan:
    """Which split each session belongs to.

    All three mappings are frozen: a plan is evidence about a decision, and
    evidence that can be edited after the fact is not evidence.
    """

    assignments: Mapping[str, str]
    counts: Mapping[str, int]
    ratios: Mapping[str, float] = SPLIT_RATIOS

    def __post_init__(self) -> None:
        """Validate, then freeze.

        Every rule `plan_splits` applies is applied here too. A rule only the
        builder enforced would be no rule at all: `SplitPlan` is public and
        constructible, so a hand-built plan claiming ten test sessions it does
        not have would sail past and reshape a dataset manifest.
        """
        # Copied *then* frozen, so a caller mutating the dict they passed
        # cannot reach inside an existing plan.
        set_field = object.__setattr__
        set_field(self, "ratios", MappingProxyType(_checked_ratios(self.ratios)))
        set_field(self, "counts", MappingProxyType(_checked_counts(self.counts)))
        set_field(
            self,
            "assignments",
            MappingProxyType(_checked_assignments(self.assignments)),
        )

        _check_counts_match(self.assignments, self.counts)


def _checked_ratios(ratios: Any) -> dict:
    """Validate the ratio mapping, or raise.

    The split vocabulary comes from `dataset_rows.SPLITS` rather than being
    restated, so adding a split there cannot leave this module silently
    disagreeing about what a split is.
    """
    if not isinstance(ratios, Mapping):
        raise SplitPlanError(
            f"ratios must be a mapping: {type(ratios).__name__}"
        )

    if set(ratios) != set(SPLITS):
        expected = ", ".join(SPLITS)
        raise SplitPlanError(
            f"ratios must name exactly the splits ({expected}): "
            f"{sorted(ratios)}"
        )

    checked = {}
    for split in SPLITS:
        value = ratios[split]

        # `bool` first: it is an `int` subclass, so `True` would pass as 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SplitPlanError(
                f"ratio for {split!r} must be a number: {value!r}"
            )

        if not math.isfinite(value) or value <= 0:
            raise SplitPlanError(
                f"ratio for {split!r} must be finite and positive: {value!r}"
            )

        checked[split] = float(value)

    total = math.fsum(checked.values())
    if abs(total - 1.0) > RATIO_TOLERANCE:
        raise SplitPlanError(f"ratios must sum to 1.0: {total!r}")

    return checked


def _checked_counts(counts: Any) -> dict:
    """Validate the per-split counts, or raise.

    Counts are seat totals, so they are non-negative integers naming exactly
    the splits. `bool` is excluded before the integer check: it is an `int`
    subclass, so `True` would otherwise be accepted as one session.
    """
    if not isinstance(counts, Mapping):
        raise SplitPlanError(f"counts must be a mapping: {type(counts).__name__}")

    if set(counts) != set(SPLITS):
        expected = ", ".join(SPLITS)
        raise SplitPlanError(
            f"counts must name exactly the splits ({expected}): {sorted(counts)}"
        )

    checked = {}
    for split in SPLITS:
        value = counts[split]

        if isinstance(value, bool) or not isinstance(value, int):
            raise SplitPlanError(
                f"count for {split!r} must be an integer: {value!r}"
            )

        if value < 0:
            raise SplitPlanError(
                f"count for {split!r} must not be negative: {value!r}"
            )

        checked[split] = value

    return checked


def _checked_assignments(assignments: Any) -> dict:
    """Validate the session-to-split mapping, or raise.

    Ids follow the same rule the builder applies — non-blank strings, nothing
    stripped. A mapping's keys are unique by construction, so duplicate ids
    cannot arise here; what must be checked is that every value is a real
    split, since an unknown one would put clips somewhere the dataset
    vocabulary does not describe.
    """
    if not isinstance(assignments, Mapping):
        raise SplitPlanError(
            f"assignments must be a mapping: {type(assignments).__name__}"
        )

    checked = {}
    for session_id, split in assignments.items():
        if not isinstance(session_id, str) or not session_id.strip():
            raise SplitPlanError(
                f"session id must be a non-empty string: {session_id!r}"
            )

        if split not in SPLITS:
            expected = ", ".join(SPLITS)
            raise SplitPlanError(
                f"session {session_id!r} is assigned to an unknown split "
                f"{split!r}; expected one of {expected}"
            )

        checked[session_id] = split

    return checked


def _check_counts_match(assignments: Mapping, counts: Mapping) -> None:
    """Refuse a plan whose counts disagree with its assignments.

    The two carry the same fact twice, so they can disagree. A plan claiming
    more test sessions than it assigns would misstate the corpus to anyone
    reading `counts` alone — which is what a reviewer reads.
    """
    for split in SPLITS:
        actual = sum(1 for value in assignments.values() if value == split)
        if counts[split] != actual:
            raise SplitPlanError(
                f"count for {split!r} is {counts[split]!r} but "
                f"{actual} session(s) are assigned to it"
            )


def _checked_sessions(session_ids: Any) -> list:
    """Validate and sort the session ids, or raise.

    Nothing is stripped: an id is a recorded value, and two ids differing only
    by whitespace are two ids. Duplicates are refused rather than collapsed,
    because collapsing would change the corpus size behind the caller's back
    and quietly reshape every count that follows.
    """
    if isinstance(session_ids, (str, bytes)) or isinstance(session_ids, Mapping):
        raise SplitPlanError(
            f"session_ids must be a collection of ids, not "
            f"{type(session_ids).__name__}"
        )

    try:
        ids = list(session_ids)
    except TypeError as exc:
        raise SplitPlanError(
            f"session_ids must be iterable: {type(session_ids).__name__}"
        ) from exc

    seen = set()
    for value in ids:
        if not isinstance(value, str) or not value.strip():
            raise SplitPlanError(
                f"session id must be a non-empty string: {value!r}"
            )
        if value in seen:
            raise SplitPlanError(f"duplicate session id: {value!r}")
        seen.add(value)

    # Plain `sorted`: byte order, no locale, so a plan does not depend on the
    # machine that made it.
    return sorted(ids)


def _allocate(total: int, ratios: Mapping[str, float]) -> dict:
    """Split `total` seats across the splits by largest remainder.

    Each split takes the whole part of its share; the seats left over go one
    each to the largest fractional remainders, ties broken by `SPLITS` order so
    the result never depends on dict iteration.

    A small corpus therefore lands entirely in `train`. That is the honest
    outcome: one session cannot be both trained on and evaluated against, and
    minting a one-clip test split would produce a number nobody should quote.
    """
    exact = {split: total * ratios[split] for split in SPLITS}
    counts = {split: int(math.floor(exact[split] + FLOOR_EPSILON)) for split in SPLITS}

    remaining = total - sum(counts.values())

    # Sort by remainder descending, then by position in SPLITS for ties.
    order = sorted(
        SPLITS,
        key=lambda split: (-(exact[split] - counts[split]), SPLITS.index(split)),
    )

    for split in order[:remaining]:
        counts[split] += 1

    return counts


def plan_splits(session_ids, *, ratios=SPLIT_RATIOS) -> SplitPlan:
    """Assign every session to exactly one split.

    Args:
        session_ids: The corpus. Any iterable of non-blank, unique strings;
            order is irrelevant because they are sorted.
        ratios: Share per split, naming exactly `SPLITS` and summing to 1.0.
            Defaults to 80/10/10.

    Returns:
        A frozen `SplitPlan`. Assignments follow sorted id order, filling
        `train`, then `validation`, then `test`.

    Raises:
        SplitPlanError: An id is blank, non-string, or duplicated; the corpus
            is not a collection; or the ratios are malformed.
    """
    checked_ratios = _checked_ratios(ratios)
    ordered = _checked_sessions(session_ids)

    counts = _allocate(len(ordered), checked_ratios)

    assignments = {}
    position = 0
    for split in SPLITS:
        for session_id in ordered[position : position + counts[split]]:
            assignments[session_id] = split
        position += counts[split]

    return SplitPlan(
        assignments=assignments, counts=counts, ratios=checked_ratios
    )


def split_for(plan: SplitPlan, session_id: str) -> str:
    """The split a session was assigned to.

    Raises:
        SplitPlanError: The session is not in the plan. There is deliberately
            no default — silently answering `train` for an unplanned session is
            precisely how a test clip ends up in the training set.
    """
    try:
        return plan.assignments[session_id]
    except (KeyError, TypeError) as exc:
        raise SplitPlanError(
            f"unknown session {session_id!r}; it was not part of this plan"
        ) from exc
