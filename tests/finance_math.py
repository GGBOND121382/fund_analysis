"""Tiny mathematical reference checks for algorithm integration.

This is NOT an optimizer, NAV downloader, trading engine, or verified investment strategy.
No external dependencies. All risk measures are illustrative and require valid inputs.
"""

from __future__ import annotations

from math import isfinite
from typing import Sequence


def max_drawdown(unit_nav: Sequence[float]) -> float:
    """Positive worst peak-to-trough drawdown of strictly positive unit NAV series."""
    if not unit_nav:
        raise ValueError("empty unit nav")
    peak = 0.0
    largest = 0.0
    for x in unit_nav:
        if not isfinite(x) or x <= 0:
            raise ValueError("nonpositive/nonfinite unit nav")
        peak = max(peak, x)
        largest = max(largest, 1.0 - x / peak)
    return largest


def drawdown_path(unit_nav: Sequence[float]) -> list[float]:
    if not unit_nav:
        raise ValueError("empty unit nav")
    peak = 0.0
    result = []
    for x in unit_nav:
        if not isfinite(x) or x <= 0:
            raise ValueError("invalid unit nav")
        peak = max(peak, x)
        result.append(1.0 - x / peak)
    return result


def empirical_cdar(drawdowns: Sequence[float], alpha: float = 0.95) -> float:
    """Exact discrete empirical CVaR of drawdown observations (Chekhlov CDaR proxy).

    Uses min_z z + sum(max(0, d-z)) / ((1-alpha)*n). Not scenario MaxDD quantile.
    """
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    if not drawdowns:
        raise ValueError("empty observations")
    if any(not isfinite(d) or not 0 <= d <= 1 for d in drawdowns):
        raise ValueError("invalid drawdown")
    n = len(drawdowns)
    candidates = set(drawdowns) | {0.0}
    return min(z + sum(max(d-z, 0.0) for d in drawdowns) / ((1-alpha)*n)
               for z in candidates)


def external_cash_unitization(pre_flow_value: float, existing_units: float,
                              external_flow: float) -> tuple[float, float]:
    """Apply external deposit/withdrawal AFTER valuation and BEFORE next market move.

    Returns new_units, unchanged_unit_nav. Flow positive for deposit.
    """
    if not (isfinite(pre_flow_value) and isfinite(existing_units)
            and isfinite(external_flow)):
        raise ValueError("nonfinite value")
    if pre_flow_value <= 0 or existing_units <= 0:
        raise ValueError("initial NAV must be positive")
    nav = pre_flow_value / existing_units
    next_units = existing_units + external_flow / nav
    if next_units <= 0:
        raise ValueError("withdrawal exhausted all investment units")
    return next_units, nav


def single_fund_stress_impact(fund_value: float, account_value: float,
                              fund_loss_fraction: float) -> float:
    """Contribution to *account loss* if only one fund drops and others unchanged."""
    if not (account_value > 0 and 0 <= fund_value <= account_value
            and 0 <= fund_loss_fraction <= 1):
        raise ValueError("invalid stress inputs")
    return fund_value / account_value * fund_loss_fraction


def check_new_cash_plan(amounts_by_fund: dict[str, float],
                        uninvested_cash: float, new_cash: float,
                        forbidden_new_buy: set[str] | frozenset[str],
                        caps: dict[str, float] | None = None,
                        tolerance: float = 1e-7) -> None:
    """Assert feasibility invariants; does not choose optimal amounts."""
    if uninvested_cash < 0 or new_cash < 0:
        raise ValueError("negative cash")
    caps = caps or {}
    for fund, amount in amounts_by_fund.items():
        if amount < 0 or not isfinite(amount):
            raise ValueError("negative or invalid subscription")
        if fund in forbidden_new_buy and amount > tolerance:
            raise ValueError(f"forbidden new buy: {fund}")
        if fund in caps and amount > caps[fund] + tolerance:
            raise ValueError(f"cap exceeded: {fund}")
    if abs(sum(amounts_by_fund.values()) + uninvested_cash - new_cash) > tolerance:
        raise ValueError("cash conservation failed")


def signal_observable(published_at_epoch: float, decision_at_epoch: float) -> bool:
    """A disclosure/NAV cannot affect a decision before its publication timestamp."""
    return published_at_epoch <= decision_at_epoch
