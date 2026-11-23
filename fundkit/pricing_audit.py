"""Valuation comparability and *reproducible* economic stress-loss audit.

An arbitrary positive economic_mdd_upper with a URL is NOT risk evidence.
No identified pricing regime or independently supplied model inputs => UNKNOWN.
A duration shock is a documented scenario proxy, never a guaranteed future MDD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fundkit.fund_master import VALID_PRICING

STRESS_MODEL = "BOND_DURATION_STRESS_v1"
NUMERIC = ("modified_duration","rate_shock_bp","spread_duration",
           "spread_shock_bp","convexity","default_loss")
NEEDED = {"code","known_at","source_url","pricing_method","risk_model",
          "risk_input_source_url","risk_input_known_at",*NUMERIC}
OUTPUT_COLS=("pricing_method","economic_mdd_upper","economic_risk_verified",
             "valuation_warning","zero_or_tiny_nav_mdd","risk_quality",
             "risk_evidence_status","source_url","known_at","risk_model")


def duration_stress_loss(modified_duration,rate_shock_bp,spread_duration,
                         spread_shock_bp,convexity,default_loss):
    """Positive yield/spread changes; positive extra credit loss.

    Bond price loss is duration*Δy - .5*convexity*Δy²
    + spread_duration*Δspread + default_loss. No negative-loss credit.
    """
    vals=np.asarray([modified_duration,rate_shock_bp,spread_duration,
                     spread_shock_bp,convexity,default_loss],dtype=float)
    if (not np.isfinite(vals).all() or (vals<0).any()
            or vals[0]>100 or vals[2]>100 or vals[4]>10000
            or vals[5]>=1 or vals[1]>10000 or vals[3]>10000):
        raise ValueError("unphysical duration / shock factor inputs")
    dy=vals[1]/10000.
    ds=vals[3]/10000.
    return float(np.clip(vals[0]*dy - .5*vals[4]*dy*dy
                         + vals[2]*ds + vals[5],0,1))


def audit_valuation(frame: pd.DataFrame, evidence: pd.DataFrame):
    """Classify and compute a factor stress proxy only from model inputs.

    `economic_risk_verified` retains old output compatibility but means
    reproducible input provenance, NOT proof of future risk calibration.
    """
    d=frame.copy()
    if "code" not in d:
        raise ValueError("fund table must have code")
    e=evidence.copy()
    if len(e):
        missing={"code","pricing_method","source_url","known_at"}-set(e.columns)
        if missing:
            raise ValueError(f"valuation evidence missing {sorted(missing)}")
        if e.duplicated("code").any():
            # Highest known_at is selected; same-timestamp contradictions fail.
            e=e.sort_values("known_at")
            most=e.groupby("code",sort=False).tail(1)
            same=e.merge(most[["code","known_at"]],on=["code","known_at"])
            if same.duplicated(["code","known_at"]).any():
                raise ValueError("conflicting valuation evidence for same known_at")
            e=most
        e=e.copy()
        for col in set(NEEDED).union({"economic_mdd_upper","shadow_price_source_url"}):
            if col not in e:
                e[col]=np.nan
        used=["code","pricing_method","source_url","known_at","risk_model",
              "risk_input_source_url","risk_input_known_at","shadow_price_source_url","economic_mdd_upper",
              *NUMERIC]
        # Code-level evidence must be point-in-time filtered before this call.
        d=d.merge(e[used],on="code",how="left",validate="many_to_one",
                  suffixes=("","_evidence"))
    else:
        d["pricing_method"],d["economic_mdd_upper"]="unknown",np.nan
        d["source_url"],d["known_at"],d["risk_model"]="","",""
        for col in (*NUMERIC,"risk_input_source_url","risk_input_known_at",
                    "shadow_price_source_url"):
            d[col]=np.nan
    d["pricing_method"]=d.pricing_method.fillna("unknown").replace("","unknown")
    if not d.pricing_method.isin(VALID_PRICING).all():
        raise ValueError("unsupported pricing method")
    d["risk_model"]=d.risk_model.fillna("").astype(str)
    d["risk_evidence_status"]="SOURCE_ONLY_NO_REPRODUCIBLE_RISK"
    d["economic_risk_verified"]=False
    d["economic_mdd_upper"]=np.nan
    source_when=pd.to_datetime(d.get("known_at"),utc=True,errors="coerce")
    exposure_when=pd.to_datetime(d["risk_input_known_at"],utc=True,errors="coerce")
    candidate=(d.risk_model.eq(STRESS_MODEL) &
               d.pricing_method.ne("unknown") &
               d.source_url.fillna("").astype(str).str.strip().ne("") &
               d.risk_input_source_url.fillna("").astype(str).str.strip().ne("") &
               exposure_when.notna() & source_when.notna() &
               exposure_when.le(source_when))
    # Amortized NAV is acceptable only if independent bond risk exposures
    # can be stress-valued; the reported NAV itself supplies no risk guarantee.
    for idx in d.index[candidate]:
        r=d.loc[idx]
        try:
            inputs=[float(r[x]) for x in NUMERIC]
            risk=duration_stress_loss(*inputs)
        except (ValueError,TypeError):
            continue
        if risk<=0 or risk>=1:
            continue
        d.at[idx,"economic_mdd_upper"]=risk
        d.at[idx,"economic_risk_verified"]=True
        d.at[idx,"risk_evidence_status"]="REPRODUCIBLE_STRESS_PROXY_NOT_VALIDATED"
    mdd=pd.to_numeric(d.get("max_drawdown",pd.Series(np.nan,index=d.index)),errors="coerce")
    ret=pd.to_numeric(d.get("period_return",pd.Series(np.nan,index=d.index)),errors="coerce")
    d["zero_or_tiny_nav_mdd"]=mdd.le(.0002)&ret.gt(0)
    d["valuation_warning"]=np.select(
        [d.pricing_method.eq("amortized_cost"),d.pricing_method.eq("unknown"),
         d.zero_or_tiny_nav_mdd],
        ["AMORTIZED_COST_NAV_NOT_MARKET_RISK","PRICING_METHOD_UNKNOWN",
         "NEAR_ZERO_NAV_DRAWDOWN_REVIEW"],default="NONE")
    d["risk_quality"]=np.where(d.economic_risk_verified,
        "REPRODUCIBLE_STRESS_NOT_OOS_VALIDATED","RISK_UNVERIFIED")
    return d


def valid_economic_risk(valuation_row):
    bound=float(valuation_row["economic_mdd_upper"])
    return bool(valuation_row["economic_risk_verified"]) and np.isfinite(bound) and 0<bound<1
