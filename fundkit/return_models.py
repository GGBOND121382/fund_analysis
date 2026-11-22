"""Evidence-locked forecasts. Unvalidated historical fit is never a live forecast."""
from __future__ import annotations
import numpy as np
import pandas as pd


def empirical_bayes_alpha(alpha_hat, se_alpha, prior, between_std):
    """Simple normal-normal posterior; parameters must be estimated before formation time."""
    a, s, tau = float(alpha_hat),float(se_alpha),float(between_std)
    if not np.isfinite([a,s,tau,prior]).all() or s < 0 or tau < 0:
        raise ValueError("alpha or uncertainty invalid")
    den = tau*tau+s*s
    return float(prior) if den == 0 else (tau*tau*a+s*s*float(prior))/den


def resolve_forecasts(codes, evidence, asof, decision_months):
    """Requires known-at, training cutoff and exact decision horizon."""
    columns=["code","forecast_mu_annual","forecast_source","forecast_known_at",
             "forecast_model","forecast_status"]
    out=[]
    asof=pd.Timestamp(asof)
    f=evidence.copy()
    required={"code","known_at","trained_through","source_url","model_id",
              "forecast_mu_annual","decision_months"}
    if len(f) and not required.issubset(f.columns):
        raise ValueError(f"forecast evidence missing: {sorted(required-set(f.columns))}")
    if len(f):
        f=f.copy()
        f["known_at"]=pd.to_datetime(f.known_at,errors="coerce")
        f["trained_through"]=pd.to_datetime(f.trained_through,errors="coerce")
        if f[["known_at","trained_through"]].isna().any().any():
            raise ValueError("forecast known_at or trained_through invalid")
        f["decision_months"]=pd.to_numeric(f.decision_months,errors="coerce")
        f["forecast_mu_annual"]=pd.to_numeric(f.forecast_mu_annual,errors="coerce")
        f=f[(f.known_at<=asof)&(f.trained_through<=f.known_at)
            &(f.decision_months.eq(decision_months))
            &f.source_url.astype(str).str.strip().ne("")
            &np.isfinite(f.forecast_mu_annual)]
        f=f.sort_values("known_at").drop_duplicates("code", keep="last").set_index("code")
    for code in codes:
        code=str(code)
        if len(f) and code in f.index:
            r=f.loc[code]
            out.append(dict(code=code,forecast_mu_annual=float(r.forecast_mu_annual),
                            forecast_source=str(r.source_url),forecast_known_at=str(r.known_at),
                            forecast_model=str(r.model_id),forecast_status="EVIDENCE_SUPPLIED_UNVALIDATED"))
        else:
            out.append(dict(code=code,forecast_mu_annual=np.nan,forecast_source="",
                            forecast_known_at="",forecast_model="",
                            forecast_status="FORECAST_MISSING"))
    return pd.DataFrame(out,columns=columns)
