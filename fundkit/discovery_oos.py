"""Point-in-time provenance and nonoverlapping OOS comparison primitives.

This module does not accept an as-of-now fund list as a historical universe.
It audits selections before allowing any economic performance claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PIT = ("catalog_known_at", "forecast_known_at", "valuation_known_at",
                "liquidity_known_at", "price_known_at", "fees_known_at")
REQUIRED_SELECTION = ("formation_date","evaluation_end","code","method","horizon",
                      "weight","fee_rate",*REQUIRED_PIT)


def validate_pit_selections(selections, require_execution=True):
    """Flags missing PIT evidence and future data. No lookahead repair by backfilling."""
    d=selections.copy()
    missing=set(REQUIRED_SELECTION)-set(d.columns)
    if missing:
        raise ValueError(f"missing point-in-time selection fields {sorted(missing)}")
    d["formation_date"]=pd.to_datetime(d.formation_date,errors="coerce")
    d["evaluation_end"]=pd.to_datetime(d.evaluation_end,errors="coerce")
    if d[["formation_date","evaluation_end"]].isna().any().any():
        raise ValueError("bad OOS signal/evaluation dates")
    if not (d.evaluation_end>d.formation_date).all():
        raise ValueError("evaluation must end strictly after formation")
    if d.duplicated(["formation_date","method","horizon","code"]).any():
        raise ValueError("duplicate selection for formation/method/horizon/code")
    failures=[]
    for row in d.itertuples(index=False):
        reasons=[]
        for c in REQUIRED_PIT:
            value=getattr(row,c)
            when=pd.to_datetime(value,errors="coerce")
            if pd.isna(when):
                reasons.append("MISSING_"+c.upper())
            elif when>row.formation_date:
                reasons.append("LOOKAHEAD_"+c.upper())
        w,fee=float(row.weight),float(row.fee_rate)
        if not (np.isfinite(w) and 0<=w<=1 and np.isfinite(fee) and 0<=fee<1):
            reasons.append("INVALID_WEIGHT_OR_FEE")
        failures.append("|".join(reasons) if reasons else "PIT_OK")
    d["pit_status"]=failures
    # All six PIT fields are required even for diagnostic OOS. Real execution
    # additionally needs actual historical fills, which these records cannot prove.
    d["execution_status"]=np.where(d.pit_status.eq("PIT_OK"),
         "EXECUTION_NOT_VERIFIED" if require_execution else "PIT_RESEARCH_ONLY",
         "EVIDENCE_INSUFFICIENT")
    return d


def realized_buy_hold(share_nav, cash_nav, weight, fee=0.0):
    """Equal-date NAV paths, static cash/fund mix, including entry/exit expense."""
    a=np.asarray(share_nav,dtype=float); c=np.asarray(cash_nav,dtype=float)
    if a.ndim!=1 or c.shape!=a.shape or len(a)<3 or np.any(~np.isfinite(a)) or np.any(~np.isfinite(c)):
        raise ValueError("future NAV paths insufficient")
    if np.any(a<=0) or np.any(c<=0):
        raise ValueError("invalid NAV")
    if not(0<=weight<=1 and 0<=fee<1):
        raise ValueError("invalid weight/fee")
    f=a/a[0]; z=c/c[0]
    wealth=(1-weight)*z+weight*(1-fee)*f
    # Entry fee is a loss versus pre-trade unit wealth; include it in DD.
    wealth=np.concatenate([[1.],wealth])
    peak=np.maximum.accumulate(wealth)
    return dict(net_return=float(wealth[-1]-1),
                max_drawdown=float(np.max(1-wealth/peak)),
                cash_excess=float(wealth[-1]-z[-1]))


def nonoverlapping_pairs(evaluations,primary,baseline,min_folds=5,seed=42):
    """Same horizon/date paired comparison; reject overlapping outcome windows."""
    required={"formation_date","evaluation_end","horizon","method","cash_excess","max_drawdown"}
    if not required.issubset(evaluations.columns):
        raise ValueError(f"paired evaluations missing {sorted(required-set(evaluations.columns))}")
    g=evaluations.loc[evaluations.method.isin([primary,baseline])].copy()
    if "pit_status" in g:
        g=g.loc[g.pit_status.eq("PIT_OK")]
    keys=["formation_date","evaluation_end","horizon","method"]
    summary=g.groupby(keys).agg(cash_excess=("cash_excess","mean"),
                                max_drawdown=("max_drawdown","mean")).reset_index()
    a=summary.loc[summary.method.eq(primary)].drop(columns="method")
    b=summary.loc[summary.method.eq(baseline)].drop(columns="method")
    pair=a.merge(b,on=["formation_date","evaluation_end","horizon"],
                 suffixes=("_primary","_baseline"),validate="one_to_one")
    results=[]
    rng=np.random.default_rng(seed)
    for horizon,part in pair.groupby("horizon"):
        part=part.sort_values("formation_date")
        # Non-overlap of realised holding periods prevents pseudo-replication.
        prev_end=pd.Timestamp.min
        accepted=[]
        for row in part.itertuples(index=False):
            if row.formation_date >= prev_end:
                accepted.append(row)
                prev_end=row.evaluation_end
        n=len(accepted)
        if n<min_folds:
            results.append(dict(horizon=horizon,folds=n,decision="EVIDENCE_INSUFFICIENT",
                 return_delta=np.nan,return_ci_lower=np.nan,return_ci_upper=np.nan,
                 mdd_delta=np.nan))
            continue
        d=np.asarray([r.cash_excess_primary-r.cash_excess_baseline for r in accepted])
        risk=np.asarray([r.max_drawdown_primary-r.max_drawdown_baseline for r in accepted])
        means=np.mean(rng.choice(d,(2000,n),replace=True),axis=1)
        low,high=np.quantile(means,[0.025,0.975])
        risk_means=np.mean(rng.choice(risk,(2000,n),replace=True),axis=1)
        risk_high=float(np.quantile(risk_means,.975))
        # Require an upper confidence bound for risk non-inferiority, not
        # merely the realized mean sign. Still NAV-only, not execution OOS.
        decision=("RESEARCH_EVIDENCE_POSITIVE" if low>0 and risk_high<=0
                  else "NOT_PROVEN_BETTER")
        results.append(dict(horizon=horizon,folds=n,decision=decision,
            return_delta=float(np.mean(d)),return_ci_lower=float(low),
            return_ci_upper=float(high),mdd_delta=float(np.mean(risk)),
            mdd_ci_upper=risk_high))
    return pd.DataFrame(results,columns=["horizon","folds","decision","return_delta",
                     "return_ci_lower","return_ci_upper","mdd_delta","mdd_ci_upper"])
