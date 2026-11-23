"""Risk-adjusted fund discovery from existing NAVs, NO external predictions required.

Primary score: conservative (category-EB lower-quartile) buy-and-hold CAGR
at the largest initial fund weight consistent with the *historical* 4%-MDD
budget and the 95th-percentile six-month bootstrapped MDD scenarios.
This is conditional historical research, not an executable forward forecast.

Cash in this layer is a ZERO-RETURN accounting numeraire (no assumed cash
yield, transaction fee or risk-free rate). No actual trading claims follow.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from fundkit.industry_rank import HORIZONS
from fundkit.discovery_scoring_v3 import windowed_values

BUDGETS = (0.01, 0.02, 0.04, 0.08)
ONE_SIDED_QUARTILE_Z = 0.6744897501960817
# Transparent exposure labels only for strong index/theme matches; they do
# not exclude any candidate from the full ranking.
THEMES = ("银行", "证券", "半导体", "芯片", "医药", "光伏", "煤炭",
          "房地产", "白酒", "有色金属", "军工", "传媒", "人工智能",
          "沪深300", "中证500", "上证50", "创业板", "科创50")


def _monthly(values, dates):
    s = pd.Series(values, index=pd.DatetimeIndex(dates))
    a = s.resample("ME").last().dropna()
    vals = a.to_numpy(dtype=float)
    if len(vals) < 5 or np.any(vals <= 0):
        raise ValueError("at least 5 complete monthly points required")
    lr = np.diff(np.log(vals))
    if len(lr) < 4 or not np.isfinite(lr).all():
        raise ValueError("not enough monthly returns")
    return lr


def _date_window(nav, asof, horizon):
    # Same window contract as v3, but expose dated points for monthly data.
    values = windowed_values(nav, asof, horizon)
    t = pd.Timestamp(asof)
    start = t - pd.DateOffset(months=HORIZONS[horizon][0])
    dates = pd.DatetimeIndex(nav.dates)
    a = int(dates.searchsorted(start, side="right") - 1)
    b = int(dates.searchsorted(t, side="right") - 1)
    return values, dates[a:b+1]


def _seed(code, h, salt=42):
    return int(hashlib.sha256(f"{code}:{h}:{salt}".encode()).hexdigest()[:8], 16)


def maximum_weight_for_budget(fund_gross, budget):
    """EXACT admissible initial buy-and-hold fund weight for 0%-cash NAV.

    v_t = fund_gross at t, p_t = running maximum v_0..v_t.
    wealth_t = 1-w + w*v_t, wealth_peak = 1-w + w*p_t.
    MDD<=B iff for every t:
        w * [p_t-v_t - B*(p_t-1)] <= B.
    Thus wmax=min(1, B / max_t positive bracket) in closed form.
    Vectorized for 1 or N scenario paths; no weighted-MDD approximation.
    """
    v = np.asarray(fund_gross, dtype=float)
    if v.ndim not in (1, 2) or v.shape[-1] < 2:
        raise ValueError("fund path must be 1D/2D with >=2 dates")
    if not np.isfinite(v).all() or (v <= 0).any() or not 0 < budget < 1:
        raise ValueError("invalid fund gross path or risk budget")
    if np.max(np.abs(v[..., 0] - 1.)) > 1e-9:
        raise ValueError("fund path must start at one")
    p = np.maximum.accumulate(v, axis=-1)
    k = np.max(p - v - budget*(p-1), axis=-1)
    with np.errstate(divide="ignore",invalid="ignore"):
        w = np.where(k>0, np.minimum(1., budget/np.maximum(k, 1e-16)), 1.)
    return float(w) if np.ndim(w)==0 else w


def monthly_stationary_paths(log_returns, *, scenarios=256, months=6, seed=42,
                             block_length=2):
    """A monthly stationary bootstrap, deterministic per fund and window.

    Monthly scenarios are only a sensitivity check; daily historical NAV
    MDD independently remains a binding constraint. No synthetic alpha.
    """
    x = np.asarray(log_returns,dtype=float)
    if x.ndim!=1 or len(x)<4 or scenarios<16 or months<1 or block_length<1:
        raise ValueError("invalid bootstrap arguments")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite monthly returns")
    rng = np.random.default_rng(int(seed))
    n = len(x)
    ids = rng.integers(0,n,size=scenarios)
    sampled = np.empty((scenarios,months),dtype=float)
    restart = 1./block_length
    for i in range(months):
        if i:
            ids = np.where(rng.random(scenarios)<restart,
                           rng.integers(0,n,size=scenarios), (ids+1)%n)
        sampled[:,i]=x[ids]
    # Overflow gives no usable scenario, not artificially high values.
    cum=np.cumsum(sampled,axis=1)
    if np.max(np.abs(cum))>600:
        raise ValueError("monthly scenario log return overflow")
    return np.concatenate([np.ones((scenarios,1)),np.exp(cum)],axis=1)


def theme_bucket(name, group):
    name=str(name)
    group=str(group)
    if not (("指数" in group) or "ETF" in name.upper() or "LOF" in name.upper()):
        return ""
    for term in THEMES:
        if term in name:
            return term
    return ""


def _posteriors(rows):
    """Within-category normal-normal EB of annual log CAGR; no external model.

    Center is peer median, between-fund variance is debiased by each
    fund's historical sampling variance. For small groups use overall prior,
    with no false zero-variance certainty. Full predictive validation pending.
    """
    if rows.empty:
        return rows
    d=rows.copy()
    global_prior=float(d.observed_log_cagr.median())
    d["eb_log_cagr"]=np.nan
    d["eb_se"]=np.nan
    d["peer_count"]=0
    for group, g in d.groupby("group",dropna=False):
        n=len(g)
        prior=float(g.observed_log_cagr.median()) if n>=10 else global_prior
        tau2=(max(0., float(g.observed_log_cagr.var(ddof=1)
                           - np.mean(g.annual_se.to_numpy()**2))) if n>=10 else 0.)
        se2=g.annual_se.to_numpy(dtype=float)**2
        # If true between-fund alpha cannot be resolved reliably, use peer
        # mean with sample uncertainty; DO NOT set posterior uncertainty to 0.
        shrink=tau2/(tau2+se2+1e-18)
        post=prior+shrink*(g.observed_log_cagr.to_numpy()-prior)
        post_se=np.sqrt(np.maximum(1e-12,
                         tau2*se2/(tau2+se2+1e-18)))
        # Estimating the peer mean also costs uncertainty, especially at n<10.
        prior_se=np.sqrt(np.mean(se2)/max(1,n))
        post_se=np.sqrt(post_se**2 + prior_se**2)
        if n<10:
            post_se=np.maximum(post_se,g.annual_se.to_numpy())
        d.loc[g.index,"eb_log_cagr"]=post
        d.loc[g.index,"eb_se"]=post_se
        d.loc[g.index,"peer_count"]=n
    return d


def _rank_diversified(rows, max_results=10):
    """Greedy distinct exposure representatives, never a fabricated factor fit.

    An explicit index/theme bucket identifies strongly related products.
    The full unrestricted score and rank remain available in research CSV.
    """
    chosen=[]
    seen_products=set()
    seen_themes=set()
    for r in rows.itertuples(index=False):
        key=str(r.product_key) if pd.notna(r.product_key) else str(r.product)
        if key in seen_products:
            continue
        theme=str(r.exposure_bucket)
        if theme and theme in seen_themes:
            continue
        chosen.append(r.code)
        seen_products.add(key)
        if theme:
            seen_themes.add(theme)
        if len(chosen)>=max_results:
            break
    return rows.loc[rows.code.isin(chosen)].copy().sort_values(
        ["risk_adjusted_score","code"],ascending=[False,True])


def rank_historical_risk_adjusted(frame, paths, asof, horizon, *,
                                  budgets=BUDGETS, scenarios=256,
                                  future_months=6, seed=42, top=10):
    """Compute four research lists from the user's existing dated NAV cache.

    Output:
      research: all previous rows, explicit missing/suspicious reasons
      diversified_top: <=top, sorted by EB lower-quartile risk-budget CAGR
      raw_top: <=top, without index exposure deduplication
      audit: count and labels; not a realized backtest.
    """
    if horizon not in HORIZONS or paths is None:
        raise ValueError("full dated NAV paths are required for risk-adjusted ranking")
    d=frame.copy()
    required={"code","name","group","period_return","max_drawdown",
              "feature_status"}
    if not required.issubset(d.columns):
        raise ValueError(f"missing fund features: {required-set(d.columns)}")
    d["risk_adjusted_status"]="NO_COMPLETE_HISTORY"
    d["risk_adjusted_score"]=np.nan
    d["risk_adjusted_rank"]=np.nan
    d["risk_weight_4pct"]=np.nan
    d["risk_budget_4pct"]=.04
    d["risk_standalone_mdd"]=pd.to_numeric(d.max_drawdown,errors="coerce")
    d["risk_bootstrap_mdd95"]=np.nan
    d["observed_log_cagr"]=np.nan
    d["annual_se"]=np.nan
    d["eb_log_cagr"]=np.nan
    d["eb_se"]=np.nan
    d["lower_quartile_log_cagr"]=np.nan
    d["peer_count"]=0
    d["exposure_bucket"]=[theme_bucket(r.name,r.group) for r in d.itertuples()]
    for b in budgets:
        d[f"risk_weight_{int(b*100)}pct"]=np.nan
        d[f"risk_score_{int(b*100)}pct"]=np.nan
    inputs=[]
    histories={}
    for idx, row in d.loc[d.feature_status.eq("HISTORICAL_FEATURE_AVAILABLE")].iterrows():
        code=str(row.code)
        nav=paths.get(code)
        if nav is None:
            d.at[idx,"risk_adjusted_status"]="NAV_PATH_MISSING"
            continue
        if bool(row.get("zero_or_tiny_nav_mdd", False)):
            d.at[idx,"risk_adjusted_status"]="NAV_SMOOTHING_REVIEW"
            continue
        # Any contractual holding restrictions require verification before
        # being compared as a liquid fund. Research row is retained.
        if (bool(row.get("is_long_periodic", False)) or
                bool(row.get("is_periodic_unverified", False)) or
                bool(row.get("contract_review_hint", False))):
            d.at[idx,"risk_adjusted_status"]="CONTRACT_REVIEW_REQUIRED"
            continue
        try:
            v, dates = _date_window(nav,asof,horizon)
            monthly = _monthly(v,dates)
            period_years=(dates[-1]-dates[0]).days/365.2425
            if period_years<=0:
                raise ValueError("invalid time span")
            obs=np.log(v[-1]/v[0])/period_years
            # Annual log-return sampling uncertainty, not future risk truth.
            annual_se=float(12*np.std(monthly,ddof=1)/np.sqrt(len(monthly)))
            if not np.isfinite(annual_se):
                raise ValueError("invalid annual return uncertainty")
        except (ValueError,OverflowError,TypeError,IndexError):
            d.at[idx,"risk_adjusted_status"]="RISK_HISTORY_INVALID"
            continue
        histories[idx]=(v/v[0], monthly, period_years)
        inputs.append(dict(idx=idx,code=code,group=row.group,
                           observed_log_cagr=float(obs),annual_se=annual_se))
    if not inputs:
        return d,d.iloc[:0].copy(),d.iloc[:0].copy(),{"ranked":0,"reason":"NO_ELIGIBLE_NAVS"}
    params=_posteriors(pd.DataFrame(inputs).set_index("idx"))
    for idx, r in params.iterrows():
        v,monthly,years=histories[idx]
        try:
            sc=monthly_stationary_paths(monthly,scenarios=scenarios,
                 months=future_months,seed=_seed(r.code,horizon,seed))
            fund_sample=sc
            hist_mdd=float(np.max(1-v/np.maximum.accumulate(v)))
            mc_mdd=np.max(1-fund_sample/np.maximum.accumulate(fund_sample,axis=1),axis=1)
            q95=float(np.quantile(mc_mdd,.95,method="higher"))
            lower=float(r.eb_log_cagr-ONE_SIDED_QUARTILE_Z*r.eb_se)
            d.at[idx,"risk_bootstrap_mdd95"]=q95
            d.at[idx,"observed_log_cagr"]=r.observed_log_cagr
            d.at[idx,"annual_se"]=r.annual_se
            d.at[idx,"eb_log_cagr"]=r.eb_log_cagr
            d.at[idx,"eb_se"]=r.eb_se
            d.at[idx,"peer_count"]=r.peer_count
            d.at[idx,"lower_quartile_log_cagr"]=lower
            d.at[idx,"risk_standalone_mdd"]=hist_mdd
            for B in budgets:
                cap_hist=maximum_weight_for_budget(v,float(B))
                caps_scenario=maximum_weight_for_budget(fund_sample,float(B))
                # 95th percentile of scenario MDD corresponds to the
                # conservative 5th percentile of per-scenario caps.
                cap_sim=float(np.quantile(caps_scenario,.05,method="lower"))
                weight=min(cap_hist,cap_sim)
                # Gross ZERO-CASH-equivalent return, not a predicted trade.
                hypothetical_total=np.expm1(np.clip(lower*years,-500,500))
                score=float(np.expm1(np.log1p(weight*hypothetical_total)/years))
                d.at[idx,f"risk_weight_{int(B*100)}pct"]=weight
                d.at[idx,f"risk_score_{int(B*100)}pct"]=score
            d.at[idx,"risk_adjusted_score"]=d.at[idx,"risk_score_4pct"]
            d.at[idx,"risk_weight_4pct"]=d.at[idx,"risk_weight_4pct"]
            d.at[idx,"risk_adjusted_status"]="NAV_RISK_ADJUSTED_RESEARCH_ONLY"
        except (ValueError,OverflowError,IndexError,FloatingPointError):
            d.at[idx,"risk_adjusted_status"]="RISK_SIMULATION_FAILED"
    positive=d.loc[(d.risk_adjusted_status=="NAV_RISK_ADJUSTED_RESEARCH_ONLY")
                   &d.risk_adjusted_score.gt(0)].copy()
    positive=positive.sort_values(["risk_adjusted_score","code"],
                                   ascending=[False,True])
    d.loc[positive.index,"risk_adjusted_rank"]=np.arange(1,len(positive)+1)
    raw=positive.head(top)
    div=_rank_diversified(positive,max_results=top)
    audit={"ranked":int(len(positive)),
           "total":int(len(d)),
           "not_ranked":int(len(d)-len(positive)),
           "statuses":d.risk_adjusted_status.value_counts().to_dict(),
           "budget_reference":.04,
           "cash_proxy_return":0.,
           "risk_estimation":"observed_daily_MDD + monthly_stationary_bootstrap_q95",
           "score_model":"within-category EB lower quartile logCAGR, 4pct budget",
           "ranking_certified":False,
           "oos_validated":False}
    return d,div,raw,audit
