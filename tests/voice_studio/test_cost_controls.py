from datetime import datetime, timedelta, timezone

import pytest

from voiceclonegpt.training.cost_manifest import (
    CostManifest,
    InvalidCostManifest,
    validate_manifest,
)
from voiceclonegpt.training.cost_preflight import run_preflight


NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def make_manifest(**overrides: object) -> CostManifest:
    fields: dict[str, object] = {
        "provider": "runpod",
        "gpu_type": "A100-80GB",
        "price_source": "provider price page checked 2026-08-01",
        "quoted_price_usd_per_hour": 2.0,
        "max_runtime_hours": 4.0,
        "estimated_max_cost_usd": 8.0,
        "hard_cost_cap_usd": 12.0,
        "deadline": NOW + timedelta(hours=6),
        "checkpoint_uri": "s3://voiceclonegpt/checkpoints/run-1",
        "shutdown_verified": True,
        "user_approved": True,
    }
    fields.update(overrides)
    return CostManifest(**fields)  # type: ignore[arg-type]


def test_complete_manifest_validates() -> None:
    validate_manifest(make_manifest())


@pytest.mark.parametrize("field", ["provider", "gpu_type", "price_source", "checkpoint_uri"])
@pytest.mark.parametrize("value", [None, 123, 1.5, True, ["s3://x"], object()])
def test_rejects_non_text_required_fields(field: str, value: object) -> None:
    with pytest.raises(InvalidCostManifest, match="must be text"):
        validate_manifest(make_manifest(**{field: value}))


@pytest.mark.parametrize("field", [
    "quoted_price_usd_per_hour", "max_runtime_hours",
    "estimated_max_cost_usd", "hard_cost_cap_usd",
])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf"), True, "2.0", None])
def test_rejects_invalid_amounts(field: str, value: object) -> None:
    with pytest.raises(InvalidCostManifest):
        validate_manifest(make_manifest(**{field: value}))


@pytest.mark.parametrize("field", ["shutdown_verified", "user_approved"])
@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_rejects_non_boolean_approval_fields(field: str, value: object) -> None:
    with pytest.raises(InvalidCostManifest, match="true or false"):
        validate_manifest(make_manifest(**{field: value}))


def test_rejects_estimate_below_rate_times_runtime() -> None:
    with pytest.raises(InvalidCostManifest, match="at least"):
        validate_manifest(make_manifest(estimated_max_cost_usd=7.5))


def test_allows_one_cent_rounding_difference() -> None:
    validate_manifest(make_manifest(
        quoted_price_usd_per_hour=0.335,
        max_runtime_hours=3.0,
        estimated_max_cost_usd=1.00,
    ))


def test_rejects_non_utc_deadline() -> None:
    with pytest.raises(InvalidCostManifest, match="Coordinated Universal Time"):
        validate_manifest(make_manifest(deadline=datetime(2026, 8, 1, 18, 0)))


def test_allows_run_within_cap_and_deadline() -> None:
    decision = run_preflight(make_manifest(), observed_price_usd_per_hour=2.0, now=NOW)
    assert decision.approved is True
    assert decision.reasons == ()
    assert decision.projected_cost_usd == 8.0


def test_blocks_unapproved_or_unverified_run() -> None:
    decision = run_preflight(
        make_manifest(user_approved=False, shutdown_verified=False),
        observed_price_usd_per_hour=2.0,
        now=NOW,
    )
    assert decision.approved is False
    assert len(decision.reasons) == 2


def test_blocks_expired_run() -> None:
    decision = run_preflight(
        make_manifest(deadline=NOW - timedelta(minutes=1)),
        observed_price_usd_per_hour=2.0,
        now=NOW,
    )
    assert decision.approved is False
    assert any("deadline" in reason for reason in decision.reasons)


def test_blocks_live_price_over_cap() -> None:
    decision = run_preflight(make_manifest(), observed_price_usd_per_hour=5.0, now=NOW)
    assert decision.approved is False
    assert decision.projected_cost_usd == 20.0
    assert any("cap" in reason for reason in decision.reasons)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "2.0"])
def test_rejects_invalid_observed_price(value: object) -> None:
    with pytest.raises(ValueError):
        run_preflight(make_manifest(), observed_price_usd_per_hour=value, now=NOW)  # type: ignore[arg-type]


def test_rejects_naive_current_time() -> None:
    with pytest.raises(InvalidCostManifest, match="Coordinated Universal Time"):
        run_preflight(make_manifest(), observed_price_usd_per_hour=2.0, now=datetime(2026, 8, 1, 12))
