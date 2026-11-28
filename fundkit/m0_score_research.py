"""Fixed M0-style percentile score experiments; no fitted return forecast.

All configurations are specified before the new replay. Historical monthly
stress proxy is deliberately not called the original event-shock metric.
"""
import numpy as np
import pandas as pd

# Each tuple: return, drawdown, stress, fourth axis; all fixed ex ante.
WEIGHTS={
    "BASE":(.40,.35,.15,.10),
    "BALANCED":(.35,.35,.15,.15),
    "RETURN":(.50,.25,.15,.10),
    "DRAWDOWN":(.30,.45,.15,.10),
    "STRESS":(.30,.30,.30,.10),
    "DOWNSIDE":(.30,.30,.15,.25),
    "NEAR_RETURN":(.45,.30,.15,.10),
    "NEAR_DRAWDOWN":(.35,.40,.15,.10),
}
FOURTH={
    "DAILY":"daily",
    "SHARPE":"sharpe",
    "SORTINO":"sortino",
    "RECOVERY":"recovery",
    "CONSISTENCY":"consistency",
}
# Monthly zero-hurdle ratios: same threshold for every window, no
# risk-free data imputation. 1bp denominator floor avoids singularity.
MONTHLY_FLOOR=.0001
CALMAR_FLOOR=.0001
METRICS=("R","D","stress_proxy","monthly","daily","sharpe","sortino",
         "calmar","recovery","consistency")


def monthly_metrics(monthly, cutoff):
    m=monthly.copy()
    if m.duplicated(["code","month"]).any():
        raise ValueError("duplicate code-month")
    m["month"]=pd.PeriodIndex(m.month,freq="M")
    if len(m) and m.month.max()>=pd.Timestamp(cutoff).to_period("M"):
        raise ValueError("partial/future monthly observations")
    m=m.sort_values(["code","month"])
    m["r"]=pd.to_numeric(m.monthly_return,errors="raise")
    if not np.isfinite(m.r).all() or (m.r<=-1).any():
        raise ValueError("invalid monthly return")
    m["x"]=np.log1p(m.r)
    m["neg2"]=np.minimum(m.r,0)**2
    m["positive"]=(m.r>0).astype(float)
    g=m.groupby("code",sort=False)
    d=g.agg(n_months=("r","size"),mean=("r","mean"),sd=("r","std"),
            downside2=("neg2","mean"),consistency=("positive","mean"))
    # Require contiguous complete months. No missing-month forward fill.
    ends=g.month.agg(["min","max"])
    span=(ends["max"].astype("int64")-ends["min"].astype("int64")+1)
    d["complete"]=span.eq(d.n_months)&d.n_months.ge(5)
    d["sharpe"]=np.sqrt(12)*d["mean"]/d.sd.clip(lower=MONTHLY_FLOOR)
    d["sortino"]=np.sqrt(12)*d["mean"]/np.sqrt(d.downside2).clip(lower=MONTHLY_FLOOR)
    d["sharpe_floor"]=d.sd.lt(MONTHLY_FLOOR)
    d["sortino_floor"]=np.sqrt(d.downside2).lt(MONTHLY_FLOOR)
    # Worst three consecutive complete months, not event-specific stress.
    roll=g.x.rolling(3,min_periods=3).sum().reset_index(level=0,drop=True)
    m["rolling3"]=roll
    d["stress_proxy"]=np.expm1(m.groupby("code").rolling3.min())
    # Fraction of observed month ends below their own prior peak,
    # including initial wealth=1. Lower is better; no future recovery.
    m["logwealth"]=g.x.cumsum()
    peak=m.groupby("code").logwealth.cummax().clip(lower=0)
    m["underwater"]=(m.logwealth<peak-1e-10).astype(float)
    d["recovery"]=m.groupby("code").underwater.mean()
    return d.drop(columns=["mean","sd","downside2"]).reset_index()


def percentile(x, higher=True):
    n=x.notna().sum()
    if n<2:
        return x.where(x.isna(),50.)
    p=100*(x.rank(method="average")-1)/(n-1)
    return p if higher else 100-p


def product_representatives(d):
    """Choose one share ex ante, not separately per model or future outcome."""
    if d.code.duplicated().any():
        raise ValueError("duplicate fund code")
    if d.product_key.isna().any() or d.product_key.eq("").any():
        raise ValueError("missing product identity")
    z=d.copy()
    # Prefer an A share, then stable code order. Fee-optimal share unknown.
    z["_preferred"]=z.name.str.contains(r"(?<=[\u4e00-\u9fff)）])\s*A(?:类)?\s*$",regex=True,na=False)
    return z.sort_values(["_preferred","code"],ascending=[False,True]).drop_duplicates("product_key").drop(columns="_preferred").copy()


def build_scores(d, actual_shock=False):
    z=d.copy()
    z["R"]=z.annualized_return
    z["monthly"]=z.L
    z["daily"]=z.L_daily
    z["calmar"]=z.R/z.D.clip(lower=CALMAR_FLOOR)
    z["calmar_floor"]=z.D.lt(CALMAR_FLOOR)
    required=list(METRICS)
    if actual_shock:
        required+=["Aret","Add"]
    if not np.isfinite(z[required].to_numpy(dtype=float)).all():
        raise ValueError("score inputs must be finite on common cohort")
    low={"D","monthly","daily","recovery"}
    for c in METRICS:
        z["P_"+c]=percentile(z[c],c not in low)
    pR,pD,pL=z.P_R,z.P_D,z.P_monthly
    stress=(percentile(z.Aret)+percentile(z.Add,False))/2 if actual_shock else z.P_stress_proxy
    z["M0_3"]= (.40*pR+.35*pD+.10*pL)/.85
    for model,(wr,wd,wa,wl) in WEIGHTS.items():
        z[model]=wr*pR+wd*pD+wa*stress+wl*pL
    for model,metric in FOURTH.items():
        z[model]=.40*pR+.35*pD+.15*stress+.10*z["P_"+metric]
    # Replace the return axis, rather than double-count a ratio as return.
    z["CALMAR"]=.40*z.P_calmar+.35*pD+.15*stress+.10*pL
    return z


MODELS=["M0_3",*WEIGHTS,*FOURTH,"CALMAR"]


def future_matrix(post, codes, market_dates, suspensions):
    """Strict common entry/exit calendar, no future-dependent reselection.

    Returns gross wealth per fund; missing outcome stays missing. Mark only
    independently verified interior suspension dates, never entry/exit.
    """
    days=pd.DatetimeIndex(market_dates).strftime("%Y-%m-%d")
    if len(days)<60:
        raise ValueError("insufficient evaluation calendar")
    p=post.loc[post.code.isin(codes)&post.date.isin(days),["code","date","nav"]]
    if p.duplicated(["code","date"]).any():
        raise ValueError("duplicate evaluation quote")
    w=p.pivot(index="code",columns="date",values="nav").reindex(index=list(codes),columns=days)
    for c,dates in suspensions.items():
        if c not in w.index: continue
        for day in sorted(dates):
            if day in days[1:-1] and pd.isna(w.loc[c,day]):
                i=days.get_loc(day)
                w.loc[c,day]=w.loc[c,days[i-1]]
    a=w.to_numpy(dtype=float)
    valid=np.isfinite(a).all(axis=1)&(a>0).all(axis=1)
    gross=np.full_like(a,np.nan)
    gross[valid]=a[valid]/a[valid,:1]
    return pd.DataFrame(gross,index=w.index,columns=w.columns)


def portfolio_stats(gross, codes, fee=.001):
    x=gross.reindex(codes).to_numpy(dtype=float)
    if not len(codes) or len(set(codes))!=len(codes):
        return {"status":"INVALID_SELECTION"}
    ok=np.isfinite(x).all(axis=1)
    if not ok.all():
        return {"status":"REJECTED_MISSING_OUTCOME","missing_codes":[c for c,k in zip(codes,ok) if not k]}
    w=x.mean(axis=0)*(1-fee/2)
    w[-1]*=(1-fee/2)
    w=np.r_[1.,w]
    dd=1-w/np.maximum.accumulate(w)
    dr=w[1:]/w[:-1]-1
    sd=dr.std(ddof=1)
    return {"status":"OK","net_return":float(w[-1]-1),"mdd":float(dd.max()),
            "zero_hurdle_sharpe":float(np.sqrt(252)*dr.mean()/sd) if sd>0 else None}
