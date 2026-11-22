"""NAV diagnostics are not equivalent to an economic-risk verification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fundkit.fund_master import VALID_PRICING


def audit_valuation(frame: pd.DataFrame, evidence: pd.DataFrame):
    """No verified valuation model => no economically verified upper risk bound."""
    d = frame.copy()
    if "code" not in d:
        raise ValueError("fund table must have code")
    e = evidence.copy()
    required = {"code","pricing_method","economic_mdd_upper","source_url","known_at"}
    if len(e) and not required.issubset(e.columns):
        raise ValueError(f"valuation evidence missing: {sorted(required-set(e.columns))}")
    if len(e):
        e = e.sort_values("known_at").drop_duplicates("code", keep="last")
        e = e[["code","pricing_method","economic_mdd_upper","source_url","known_at"]]
        d = d.merge(e, on="code",how="left",validate="many_to_one",suffixes=("","_evidence"))
    else:
        d["pricing_method"],d["economic_mdd_upper"] = "unknown",np.nan
        d["source_url"],d["known_at"] = "",""
    d["pricing_method"] = d.pricing_method.fillna("unknown").replace("", "unknown")
    if not d.pricing_method.isin(VALID_PRICING).all():
        raise ValueError("unsupported valuation method")
    d["economic_mdd_upper"] = pd.to_numeric(d.economic_mdd_upper, errors="coerce")
    # User-supplied economic risk needs a verifiable source, finite positive range,
    # and compatible pricing method. Unknown and amortized-cost NAV cannot certify risk.
    risk_evidence = (d.source_url.fillna("").astype(str).str.len().gt(0)
                     & d.economic_mdd_upper.gt(0) & d.economic_mdd_upper.lt(1)
                     & d.pricing_method.ne("unknown"))
    d["economic_risk_verified"] = risk_evidence
    mdd = pd.to_numeric(d.get("max_drawdown",pd.Series(np.nan,index=d.index)),errors="coerce")
    ret = pd.to_numeric(d.get("period_return",pd.Series(np.nan,index=d.index)),errors="coerce")
    d["zero_or_tiny_nav_mdd"] = mdd.le(0.0002) & ret.gt(0)
    d["valuation_warning"] = np.select(
        [d.pricing_method.eq("amortized_cost"),d.pricing_method.eq("unknown"),
         d.zero_or_tiny_nav_mdd],
        ["AMORTIZED_COST_NAV_NOT_MARKET_RISK","PRICING_METHOD_UNKNOWN",
         "NEAR_ZERO_NAV_DRAWDOWN_REVIEW"],
        default="NONE")
    d["risk_quality"] = np.where(d.economic_risk_verified,
                                 "ECONOMIC_RISK_EVIDENCE", "RISK_UNVERIFIED")
    # Zero is never a certified economic risk upper bound for an investment fund.
    # This verifies presence and format of evidence, not source authenticity or OOS.
    # Never translate observed NAV MDD=0 into an economic risk upper bound of 0.
    return d


def valid_economic_risk(valuation_row):
    bound = float(valuation_row["economic_mdd_upper"])
    return bool(valuation_row["economic_risk_verified"]) and np.isfinite(bound)
