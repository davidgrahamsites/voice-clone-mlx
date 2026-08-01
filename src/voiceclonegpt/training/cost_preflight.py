"""Decide whether one planned paid training run is allowed to start."""

from dataclasses import dataclass
from datetime import datetime

from voiceclonegpt.training.cost_manifest import (
    CostManifest,
    InvalidCostManifest,
    check_amount,
    check_utc,
    validate_manifest,
)


@dataclass(frozen=True)
class PreflightDecision:
    """The go/no-go answer, with every blocking reason in plain words."""

    approved: bool
    reasons: tuple[str, ...]
    projected_cost_usd: float


def run_preflight(
    manifest: CostManifest,
    *,
    observed_price_usd_per_hour: float,
    now: datetime,
) -> PreflightDecision:
    """Return the decision for `manifest` at the price seen right now."""
    observed_price = check_amount(observed_price_usd_per_hour, "price seen right now")
    if observed_price <= 0:
        raise ValueError(
            "The price seen right now must be greater than zero, but it is "
            f"{observed_price}."
        )
    check_utc(now, "current time")

    try:
        validate_manifest(manifest)
    except InvalidCostManifest as error:
        return PreflightDecision(False, (str(error),), 0.0)

    projected_cost = observed_price * manifest.max_runtime_hours
    reasons: list[str] = []
    if not manifest.user_approved:
        reasons.append("The user has not approved this run.")
    if not manifest.shutdown_verified:
        reasons.append("Nobody has verified that the rented machine will shut down when the run ends.")
    if now >= manifest.deadline:
        reasons.append(f"The run's deadline ({manifest.deadline.isoformat()}) has already passed.")
    if projected_cost > manifest.hard_cost_cap_usd:
        reasons.append(
            f"At the price seen now (${observed_price_usd_per_hour:.2f} per hour) the run would cost up to "
            f"${projected_cost:.2f}, which is over the hard cost cap of ${manifest.hard_cost_cap_usd:.2f}."
        )
    return PreflightDecision(not reasons, tuple(reasons), projected_cost)
