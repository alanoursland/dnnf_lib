"""Runtime checks for ModeNexus-produced numerical results."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Optional


class ModeNexusInvariantError(RuntimeError):
    """Raised when an internal result violates a documented contract."""


def _scale_tolerance(*values: float, tolerance: float = 1e-9) -> float:
    finite = [abs(value) for value in values if math.isfinite(value)]
    return tolerance * max([1.0, *finite])


def check_probability(
    value: float,
    context: str,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Validate and lightly clamp a ModeNexus-produced probability."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ModeNexusInvariantError(
            f"{context} must be a finite probability; got {value!r}"
        ) from None
    if (
        not math.isfinite(numeric)
        or numeric < -tolerance
        or numeric > 1.0 + tolerance
    ):
        raise ModeNexusInvariantError(
            f"{context} must be in [0, 1]; got {numeric!r}"
        )
    return min(1.0, max(0.0, numeric))


def check_distribution(
    values: Mapping[object, float] | Iterable[float],
    context: str,
    *,
    tolerance: float = 1e-6,
) -> None:
    """Validate a ModeNexus-produced probability distribution."""
    entries = values.values() if isinstance(values, Mapping) else values
    numeric = [
        check_probability(value, f"{context} entry")
        for value in entries
    ]
    if not numeric:
        raise ModeNexusInvariantError(f"{context} must not be empty")
    total = sum(numeric)
    if not math.isclose(
        total,
        1.0,
        rel_tol=tolerance,
        abs_tol=min(tolerance, 1e-9),
    ):
        raise ModeNexusInvariantError(
            f"{context} must sum to 1 within {tolerance:g}; got {total!r}"
        )


def check_interval(
    value: float,
    lower: float,
    upper: float,
    context: str,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Validate a finite result against a closed numerical interval."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ModeNexusInvariantError(
            f"{context} must be finite; got {value!r}"
        ) from None
    if not math.isfinite(numeric):
        raise ModeNexusInvariantError(
            f"{context} must be finite; got {numeric!r}"
        )
    slack = _scale_tolerance(
        numeric, lower, upper, tolerance=tolerance
    )
    if numeric < lower - slack or numeric > upper + slack:
        raise ModeNexusInvariantError(
            f"{context} must be in [{lower!r}, {upper!r}]; got {numeric!r}"
        )
    return min(upper, max(lower, numeric))


def check_bounds(
    lower: float,
    upper: float,
    context: str,
    *,
    value: Optional[float] = None,
    tolerance: float = 1e-9,
    allow_infinite: bool = False,
) -> None:
    """Validate an ordered bound pair and optional bracketed value."""
    try:
        low = float(lower)
        high = float(upper)
    except (TypeError, ValueError):
        raise ModeNexusInvariantError(
            f"{context} bounds must be numeric; got {lower!r}, {upper!r}"
        ) from None
    if math.isnan(low) or math.isnan(high) or (
        not allow_infinite and (
            not math.isfinite(low) or not math.isfinite(high)
        )
    ):
        raise ModeNexusInvariantError(
            f"{context} bounds must be "
            f"{'non-NaN' if allow_infinite else 'finite'}; "
            f"got {low!r}, {high!r}"
        )
    slack = _scale_tolerance(low, high, tolerance=tolerance)
    if low > high + slack:
        raise ModeNexusInvariantError(
            f"{context} lower bound {low!r} exceeds upper bound {high!r}"
        )
    if value is not None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ModeNexusInvariantError(
                f"{context} value must be numeric; got {value!r}"
            ) from None
        value_slack = _scale_tolerance(
            low, numeric, high, tolerance=tolerance
        )
        if (
            math.isnan(numeric)
            or numeric < low - value_slack
            or numeric > high + value_slack
        ):
            raise ModeNexusInvariantError(
                f"{context} bounds [{low!r}, {high!r}] do not bracket "
                f"{numeric!r}"
            )


def check_nondecreasing(
    previous: float,
    current: float,
    context: str,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Validate a non-decreasing numerical trace step."""
    if not math.isfinite(previous) or not math.isfinite(current):
        raise ModeNexusInvariantError(
            f"{context} values must be finite; got {previous!r}, {current!r}"
        )
    slack = _scale_tolerance(
        previous, current, tolerance=tolerance
    )
    if current < previous - slack:
        raise ModeNexusInvariantError(
            f"{context} decreased from {previous!r} to {current!r}"
        )
