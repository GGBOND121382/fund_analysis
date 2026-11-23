"""Point-in-time, lookback-specific fund research and joint return/risk scenarios.

Descriptive NAV leaderboards are not investment recommendations or predictions.
Forecast scenarios share a single distribution for expected return AND drawdown.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from fundkit.industry_rank import HORIZONS
from fundkit.opportunity_rank import stationary_bootstrap


def windowed_values(nav, asof, lookback):
    """Exactly the chosen historical window, with boundary and gap checks."""
    if lookback not in HORIZONS:
        raise ValueError("unsupported historical horizon")
    t = pd.Timestamp(asof)
    start = t - pd.DateOffset(months=HORIZONS[lookback][0])
    days = pd.DatetimeIndex(nav.dates)
    a = int(days.searchsorted(start, side="right") - 1)
    b = int(days.searchsorted(t, side="right") - 1)
    if a < 0 or b <= a or (start-days[a]).days > 10 or (t-days[b]).days > 7:
        raise ValueError("historical boundary missing")
    v = np.asarray(nav.value[a:b+1],dtype=float)
    if len(v) < HORIZONS[lookback][1] or not np.isfinite(v).all() or (v<=0).any():
        raise ValueError("insufficient historical NAV observations")
    if np.max(np.diff(days[a:b+1]).astype("timedelta64[D]").astype(int)) > 14:
        raise ValueError("historical NAV has gaps")
    return v


def nav_mean_uncertainty(values, lag=10):
    """Newey-West long-run variance SE of annualized mean log return.

    Estimation uncertainty, not a calibrated forward prediction interval.
    """
    x = np.diff(np.log(np.asarray(values, dtype=float)))
    if len(x)<64 or not np.isfinite(x).all():
        raise ValueError("insufficient NAV for uncertainty estimate")
    x=x-x.mean()
    n=len(x)
    m=min(lag,n-1)
    gamma0=float(np.dot(x,x)/n)
    long_var=gamma0
    for k in range(1,m+1):
        gamma=float(np.dot(x[k:],x[:-k])/n)
        long_var+=2*(1-k/(m+1))*gamma
    return 252*np.sqrt(max(0.,long_var)/n)


def joint_forecast_paths(values, forecast_annual, *, steps, scenarios=256,
                         block_length=10, seed=42, forecast_se_annual=None):
    """Joint paths: mean-shift bootstrap innovations to forecast log CAGR.

    Each scenario draws ONE forecast-mean error for its entire horizon.
    Both expected-return calculation and risk use these SAME paths.
    Does not correct unobservable appraisal smoothing or credit event risk.
    """
    mu=float(forecast_annual)
    if not np.isfinite(mu) or mu<=-1 or steps<2 or scenarios<2:
        raise ValueError("bad forecast or scenario shape")
    observed=np.asarray(values,dtype=float)
    raw=stationary_bootstrap(observed,steps=steps,scenarios=scenarios,
                             block_length=block_length,seed=seed)
    draws=np.diff(np.log(raw),axis=1)
    history=np.diff(np.log(observed))
    # Center at the formation-period mean, not at the realized simulated mean.
    residual=draws-history.mean()
    uncertainty=(nav_mean_uncertainty(observed) if forecast_se_annual is None
                 else float(forecast_se_annual))
    if not np.isfinite(uncertainty) or uncertainty<0:
        raise ValueError("forecast uncertainty invalid")
    rng=np.random.default_rng(int(seed)+1009)
    mean_error=rng.normal(0.,uncertainty,size=(scenarios,1))/252
    per_day=np.log1p(mu)/252 + mean_error
    log_path=np.cumsum(residual+per_day,axis=1)
    if np.any(log_path>680) or np.any(log_path< -680):
        raise ValueError("scenario growth overflow")
    return np.concatenate([np.ones((scenarios,1)),
                           np.exp(log_path)],axis=1)


def descriptive_research(frame, asof, horizon):
    """All-category historical return discovery; no hidden risk weights.

    `historical_return_rank` is an explicitly backward-looking *return only*
    diagnostic. An absence of verified valuation NEVER becomes a zero-risk score.
    """
    if horizon not in HORIZONS:
        raise ValueError("unsupported horizon")
    d=frame.copy()
    d["historical_return_cagr"]=np.nan
    d["historical_return_rank"]=np.nan
    d["category_return_rank"]=np.nan
    d["descriptive_research_status"]="HISTORY_UNAVAILABLE"
    d["risk_comparability"]="NOT_ASSESSED"
    eligible=d.feature_status.eq("HISTORICAL_FEATURE_AVAILABLE")
    if not {"start","end","period_return","max_drawdown"}.issubset(d.columns):
        d.loc[eligible,"descriptive_research_status"]="HISTORICAL_DATES_MISSING"
        return d
    if not eligible.any():
        return d
    x=d.loc[eligible]
    years=(pd.to_datetime(x["end"])-pd.to_datetime(x["start"])).dt.days/365.2425
    ret=pd.to_numeric(x.period_return,errors="coerce")
    mdd=pd.to_numeric(x.max_drawdown,errors="coerce")
    valid=ret.gt(-1)&years.gt(0)&years.notna()&mdd.between(0,1)
    idx=x.index[valid]
    d.loc[idx,"historical_return_cagr"]=np.expm1(
        np.log1p(ret.loc[idx])/years.loc[idx])
    d.loc[idx,"descriptive_research_status"]="NAV_HISTORY_ONLY"
    if len(idx):
        d.loc[idx,"historical_return_rank"]=d.loc[idx,"historical_return_cagr"].rank(
            method="min",ascending=False)
        d.loc[idx,"category_return_rank"]=d.loc[idx].groupby(
            "group")["historical_return_cagr"].rank(method="min",ascending=False)
        verified=d.loc[idx,"economic_risk_verified"].fillna(False).eq(True)
        flagged=d.loc[idx,"zero_or_tiny_nav_mdd"].fillna(False).eq(True)
        d.loc[idx,"risk_comparability"]=np.select(
            [verified.to_numpy(),flagged.to_numpy()],
            ["REPRODUCIBLE_ECONOMIC_RISK","NAV_SMOOTHING_REVIEW"],
            default="NAV_RISK_NOT_ECONOMICALLY_VERIFIED")
    return d
