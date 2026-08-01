"""Describe and check the written cost plan for one paid remote training run."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

ROUNDING_TOLERANCE_USD = 0.01


class InvalidCostManifest(ValueError):
    """Raised when a cost plan is incomplete or impossible to honor."""


@dataclass(frozen=True)
class CostManifest:
    """The full, written-down plan for spending money on one training run."""

    provider: str
    gpu_type: str
    price_source: str
    quoted_price_usd_per_hour: float
    max_runtime_hours: float
    estimated_max_cost_usd: float
    hard_cost_cap_usd: float
    deadline: datetime
    checkpoint_uri: str
    shutdown_verified: bool
    user_approved: bool


_REQUIRED_TEXT = (
    ("provider", "provider name"),
    ("gpu_type", "GPU type"),
    ("price_source", "price source"),
    ("checkpoint_uri", "checkpoint location"),
)

_REQUIRED_POSITIVE = (
    ("quoted_price_usd_per_hour", "quoted price per hour"),
    ("max_runtime_hours", "maximum run time in hours"),
    ("estimated_max_cost_usd", "estimated maximum cost"),
    ("hard_cost_cap_usd", "hard cost cap"),
)

_REQUIRED_YES_NO = (
    ("shutdown_verified", "shutdown-verified answer"),
    ("user_approved", "user-approved answer"),
)


def check_amount(value: object, label: str) -> float:
    """Return `value` as a plain number, or raise if it cannot be spent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCostManifest(f"The {label} must be a number, but it is {value!r}.")
    if not math.isfinite(value):
        raise InvalidCostManifest(f"The {label} must be a real number, but it is {value}.")
    return float(value)


def check_utc(moment: object, label: str) -> datetime:
    """Return `moment` unchanged, or raise unless it is an explicit UTC time."""
    if not isinstance(moment, datetime):
        raise InvalidCostManifest(f"The {label} must be a date and time.")
    if moment.utcoffset() is None:
        raise InvalidCostManifest(
            f"The {label} must say which time zone it is in, and that time zone "
            "must be Coordinated Universal Time (UTC)."
        )
    if moment.utcoffset() != timedelta(0):
        raise InvalidCostManifest(
            f"The {label} must be given in Coordinated Universal Time (UTC), "
            f"but it is offset by {moment.utcoffset()}."
        )
    return moment


def validate_manifest(manifest: CostManifest) -> None:
    """Raise `InvalidCostManifest` if the plan is unusable."""
    for field, label in _REQUIRED_TEXT:
        value = getattr(manifest, field)
        if not isinstance(value, str):
            raise InvalidCostManifest(f"The {label} must be text, but it is {value!r}.")
        if not value.strip():
            raise InvalidCostManifest(f"The {label} is missing.")

    for field, label in _REQUIRED_YES_NO:
        value = getattr(manifest, field)
        if not isinstance(value, bool):
            raise InvalidCostManifest(
                f"The {label} must be exactly true or false, but it is {value!r}."
            )

    amounts = {}
    for field, label in _REQUIRED_POSITIVE:
        value = check_amount(getattr(manifest, field), label)
        if value <= 0:
            raise InvalidCostManifest(
                f"The {label} must be greater than zero, but it is {value}."
            )
        amounts[field] = value

    check_utc(manifest.deadline, "deadline")
    rate_times_time = amounts["quoted_price_usd_per_hour"] * amounts["max_runtime_hours"]
    if amounts["estimated_max_cost_usd"] < rate_times_time - ROUNDING_TOLERANCE_USD:
        raise InvalidCostManifest(
            "The estimated maximum cost "
            f"(${amounts['estimated_max_cost_usd']:.2f}) must be at least the "
            "quoted price per hour times the maximum run time "
            f"(${rate_times_time:.2f}), give or take ${ROUNDING_TOLERANCE_USD:.2f} for rounding."
        )
    if manifest.estimated_max_cost_usd > manifest.hard_cost_cap_usd:
        raise InvalidCostManifest(
            "The estimated maximum cost "
            f"(${manifest.estimated_max_cost_usd:.2f}) cannot be more than the "
            f"hard cost cap (${manifest.hard_cost_cap_usd:.2f})."
        )
