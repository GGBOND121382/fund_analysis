"""Risk-budget opportunity scoring with pathwise buy-and-hold cash/fund wealth.

Model-derived scores require verified inputs and are EXPERIMENTAL until
independent walk-forward acceptance. Legacy percentile/Pareto never participates.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


BUDGETS = (0.01, 0.02, 0.04, 0.08)


@dataclass(frozen=True)
class OpportunityInput:
    forecast_annual: float
    cash_annual: float
    fee_rate_roundtrip: float
    economic_shock_loss: float
    decision_months: int = 6
    risk_quantile: float = 0.95
    grid_points: int = 101


def stationary_bootstrap(values, *, steps, scenarios=256, block_length=10, seed=42):
    """Circular stationary-block resampling of backward-observed log returns.

    No future data: caller must truncate the input NAV at the formation date.
    This is a NAV-only empirical model, never a source of economic-risk proof.
    """
    v = np.asarray(values, dtype=float)
    if (v.ndim != 1 or len(v) < 65 or np.any(~np.isfinite(v)) or
            np.any(v <= 0) or steps < 2 or scenarios < 2 or block_length < 1):
        raise ValueError("NAV history or bootstrap parameters invalid")
    log_returns = np.diff(np.log(v))
    n = len(log_returns)
    rng = np.random.default_rng(seed)
    idx = rng.integers(n, size=scenarios)
    returns = np.empty((scenarios, steps), dtype=float)
    restart = 1.0 / block_length
    for t in range(steps):
        if t:
            choose_new = rng.random(scenarios) < restart
            idx = np.where(choose_new, rng.integers(n, size=scenarios), (idx+1) % n)
        returns[:, t] = log_returns[idx]
    return np.concatenate([np.ones((scenarios,1)),
                           np.exp(np.cumsum(returns, axis=1))],axis=1)


def cash_paths(rate_annual, *, steps, scenarios):
    rate = float(rate_annual)
    if not np.isfinite(rate) or rate <= -1:
        raise ValueError("cash forecast annual return invalid")
    v = (1+rate)**(np.arange(steps+1)/252)
    return np.broadcast_to(v,(scenarios,steps+1))


def max_drawdown(wealth):
    v = np.asarray(wealth,dtype=float)
    if v.ndim != 2 or v.shape[1] < 2 or np.any(~np.isfinite(v)) or np.any(v<=0):
        raise ValueError("gross wealth scenarios must be strictly positive")
    return np.max(1-v/np.maximum.accumulate(v,axis=1),axis=1)


def buy_hold_wealth(fund_gross, cash_gross, w, roundtrip_fee=0.0):
    """Pathwise, fixed initial mix without imaginary daily rebalancing."""
    f,c = np.asarray(fund_gross,dtype=float),np.asarray(cash_gross,dtype=float)
    if f.ndim != 2 or c.shape!=f.shape or f.shape[1] < 2:
        raise ValueError("fund/cash scenario shapes differ")
    if np.max(np.abs(f[:,0]-1))>1e-9 or np.max(np.abs(c[:,0]-1))>1e-9:
        raise ValueError("scenario paths must start at one")
    if not 0 <= w <= 1 or not 0 <= roundtrip_fee < 1:
        raise ValueError("weight or transaction cost invalid")
    wealth = (1-w)*c+w*(1-roundtrip_fee)*f
    # Include the initial fee as a drawdown from the pre-trade wealth of 1.
    return np.concatenate([np.ones((len(f),1)),wealth],axis=1)


def risk_upper(fund_gross,cash_gross,w,fee,economic_shock_loss,quantile=0.95):
    """Conservative max of empirical path quantile and specified loss shock.

    This is a *scenario risk estimate*, not a guaranteed future drawdown cap.
    """
    if not 0<=economic_shock_loss<1:
        raise ValueError("economic shock loss must be in [0,1)")
    if not 0<quantile<1:
        raise ValueError("risk quantile must be between 0 and 1")
    base = buy_hold_wealth(fund_gross,cash_gross,w,fee)
    empirical = float(np.quantile(max_drawdown(base),quantile,method="higher"))
    # A conservative instantaneous economic loss in the fund at the first step.
    # Calculate the resulting MIXTURE drawdown from a simulated path; do not
    # assert MDD(cash+fund) = w * MDD(fund) for general paths.
    shock_fund = np.ones((1,fund_gross.shape[1]),dtype=float)
    shock_fund[:,1:] = 1-economic_shock_loss
    shadow = buy_hold_wealth(shock_fund,cash_gross[:1],w,fee)
    stress = float(max_drawdown(shadow)[0])
    return max(empirical,stress)


def optimize_opportunity(fund_gross,cash_gross,setting: OpportunityInput,
                         budgets=BUDGETS):
    """Grid-certified optimum under an explicit budget and scenario family.

    Returns model-conditional valuations. Does NOT claim OOS verified forecasts.
    """
    f,c = np.asarray(fund_gross,dtype=float),np.asarray(cash_gross,dtype=float)
    if f.shape != c.shape or f.ndim != 2 or f.shape[1]<2:
        raise ValueError("invalid pathwise scenarios")
    numeric = [setting.forecast_annual, setting.cash_annual,
               setting.fee_rate_roundtrip,setting.economic_shock_loss]
    if not np.isfinite(numeric).all() or setting.decision_months < 1:
        raise ValueError("missing predictive, cost or economic-risk evidence")
    if not 0<=setting.fee_rate_roundtrip<1 or not 0<=setting.economic_shock_loss<1:
        raise ValueError("invalid fee or risk evidence")
    if setting.grid_points<2:
        raise ValueError("grid_points must be at least two")
    r = []
    gross_excess = setting.forecast_annual-setting.cash_annual
    annual_fee = setting.fee_rate_roundtrip*12/setting.decision_months
    # Costs not subtracted twice: optimization is on the annualized increment,
    # while cost is also included in risk path at entry.
    for budget in budgets:
        if not 0<budget<1:
            raise ValueError("budget out of range")
        best = dict(budget=float(budget),opportunity_score_annual=0.0,
                    weight=0.0,risk_upper=0.0,status="NO_POSITIVE_EXCESS")
        for w in np.linspace(0.,1.,setting.grid_points):
            risk = risk_upper(f,c,float(w),setting.fee_rate_roundtrip,
                              setting.economic_shock_loss,setting.risk_quantile)
            if risk>budget+1e-12:
                continue
            score=float(w)*(gross_excess-annual_fee)
            if score>best["opportunity_score_annual"]+1e-12:
                best.update(opportunity_score_annual=score,weight=float(w),
                            risk_upper=risk,status="MODEL_SCORE_UNVALIDATED")
        r.append(best)
    return r
