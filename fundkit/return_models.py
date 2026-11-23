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


def resolve_forecasts(codes, evidence, asof, decision_months, lookback_horizon=None):
    """Requires known-at, training cutoff and exact decision horizon."""
    columns=["code","forecast_mu_annual","forecast_se_annual","forecast_source",
             "forecast_known_at","forecast_model","forecast_status"]
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
        if lookback_horizon is not None:
            # Never reuse a six-month trained forecast as a five-year model.
            # External forecast files require explicit historical window.
            f=(f.loc[f.lookback_horizon.eq(lookback_horizon)] if
               "lookback_horizon" in f.columns else f.iloc[:0])
        f=f[(f.known_at<=asof)&(f.trained_through<=f.known_at)
            &(f.decision_months.eq(decision_months))
            &f.source_url.astype(str).str.strip().ne("")
            &np.isfinite(f.forecast_mu_annual)]
        if "forecast_se_annual" not in f.columns:
            f["forecast_se_annual"]=np.nan
        f["forecast_se_annual"]=pd.to_numeric(f.forecast_se_annual,errors="coerce")
        f=f.sort_values("known_at").drop_duplicates("code", keep="last").set_index("code")
    for code in codes:
        code=str(code)
        if len(f) and code in f.index:
            r=f.loc[code]
            out.append(dict(code=code,forecast_mu_annual=float(r.forecast_mu_annual),
                            forecast_se_annual=(float(r.forecast_se_annual)
                               if np.isfinite(r.forecast_se_annual) and r.forecast_se_annual>=0
                               else np.nan),
                            forecast_source=str(r.source_url),forecast_known_at=str(r.known_at),
                            forecast_model=str(r.model_id),forecast_status="EVIDENCE_SUPPLIED_UNVALIDATED"))
        else:
            out.append(dict(code=code,forecast_mu_annual=np.nan,
                            forecast_se_annual=np.nan,forecast_source="",
                            forecast_known_at="",forecast_model="",
                            forecast_status="FORECAST_MISSING"))
    return pd.DataFrame(out,columns=columns)

def fit_category_forecasts(fund_rows, nav_paths, benchmark_history, asof, lookback_months,
                           decision_months=6, min_group=5):
    """Research-only PIT EB forecast; benchmark and fund data must precede t.

    M0: rolling historical annualized category-benchmark log return.
    M1: M0 plus empirical-Bayes-shrunk monthly active return. Both are
    explicitly unvalidated until independent nested rolling OOS.
    """
    required={"group","date","nav","source_url","known_at"}
    if not required.issubset(benchmark_history.columns):
        raise ValueError(f"benchmark history missing {sorted(required-set(benchmark_history.columns))}")
    t=pd.Timestamp(asof)
    cutoff=pd.offsets.BMonthEnd().rollback(t)
    b=benchmark_history.copy()
    b["date"]=pd.to_datetime(b.date,errors="coerce")
    b["known_at"]=pd.to_datetime(b.known_at,errors="coerce")
    b["nav"]=pd.to_numeric(b.nav,errors="coerce")
    if b[["date","known_at"]].isna().any().any() or not np.isfinite(b.nav).all():
        raise ValueError("bad benchmark NAV history or time")
    if (b.nav<=0).any() or b.source_url.astype(str).str.strip().eq("").any():
        raise ValueError("benchmark NAV requires positive values and source URLs")
    # Never train on prices or benchmark releases after the formation date.
    b=b.loc[b.date.le(t)&b.known_at.le(t)].copy()
    if b.duplicated(["group","date"]).any():
        raise ValueError("duplicate benchmark observations")
    monthly_b={}
    for group,g in b.groupby("group"):
        s=g.set_index("date").nav.sort_index().resample("BME").last().dropna()
        monthly_b[group]=s.loc[s.index<=cutoff]
    details=[]
    for row in fund_rows.itertuples(index=False):
        if row.code not in nav_paths or row.group not in monthly_b:
            continue
        nav=nav_paths[row.code]
        series=pd.Series(nav.value,index=nav.dates)
        series=series.loc[series.index<=t].resample("BME").last().dropna()
        bench=monthly_b[row.group]
        # Rolling history as of t, never borrow longer history from a fund.
        begin=t-pd.DateOffset(months=int(lookback_months))
        z=pd.concat([series.rename("fund"),bench.rename("benchmark")],axis=1).dropna()
        z=z.loc[(z.index<=cutoff)&(z.index>=begin)]
        if len(z)<5:
            continue
        if (z.index.to_series().diff().dropna().dt.days>40).any():
            continue
        f=np.diff(np.log(z.fund.to_numpy(dtype=float)))
        base=np.diff(np.log(z.benchmark.to_numpy(dtype=float)))
        if len(f)<4:
            continue
        excess=f-base
        a=float(12*np.mean(excess))
        se=float(12*np.std(excess,ddof=1)/np.sqrt(len(excess)))
        bench_log=float(12*np.mean(base))
        base_se=float(12*np.std(base,ddof=1)/np.sqrt(len(base)))
        details.append(dict(code=row.code,group=row.group,alpha_hat=a,se_alpha=se,
                            bench_log=bench_log,benchmark_se=base_se,
                            month_pairs=len(excess),
                            trained_through=str(z.index[-1].date()),
                            source_url=b.loc[b.group.eq(row.group),"source_url"].iloc[-1]))
    if not details:
        return pd.DataFrame(columns=["code","known_at","trained_through","source_url",
                    "model_id","forecast_mu_annual","decision_months",
                    "prior_alpha","between_std","se_alpha","month_pairs",
                    "lookback_horizon","forecast_se_annual"])
    d=pd.DataFrame(details)
    result=[]
    for group,g in d.groupby("group"):
        # Estimate cross-fund alpha dispersion from observed variance minus
        # the average sampling variance (empirical Bayes, lower-bounded at 0).
        if len(g)>=min_group:
            prior=float(g.alpha_hat.median())
            tau2=max(0.,float(g.alpha_hat.var(ddof=1)-np.mean(g.se_alpha**2)))
        else:
            prior,tau2=0.,0.
        for row in g.itertuples(index=False):
            alpha=empirical_bayes_alpha(row.alpha_hat,row.se_alpha,prior,np.sqrt(tau2))
            predicted=float(np.expm1(row.bench_log+alpha))
            result.append(dict(code=row.code,known_at=str(t.date()),trained_through=row.trained_through,
                    source_url=row.source_url,model_id="EB_CLASS_BENCHMARK_UNVALIDATED_v1",
                    forecast_mu_annual=predicted,decision_months=int(decision_months),
                    lookback_horizon=({6:"6m",12:"1y",36:"3y",60:"5y"}.get(int(lookback_months),
                                     f"{int(lookback_months)}m")),
                    forecast_se_annual=float(row.se_alpha+row.benchmark_se),
                    prior_alpha=prior,between_std=float(np.sqrt(tau2)),
                    se_alpha=row.se_alpha,month_pairs=row.month_pairs))
    return pd.DataFrame(result)
