"""Automatic point-in-time NAV-ONLY walk-forward fund discovery research.

Rebuilds each formation from a dated catalog snapshot, trains from history
at or before t, selects funds, and only then evaluates t->t+H.

No historical catalog => HARD FAIL, not today's surviving funds.
No official contract/valuation/execution evidence => never certified OOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from fundkit.industry_rank import HORIZONS,assemble
from fundkit.discovery_scoring_v3 import (descriptive_research,windowed_values,
    joint_forecast_paths)
from fundkit.pricing_audit import audit_valuation
from fundkit.return_models import fit_category_forecasts
from fundkit.opportunity_rank import cash_paths,optimize_opportunity,OpportunityInput
from fundkit.discovery_oos import nonoverlapping_pairs, realized_buy_hold
from fundkit.fund_master import read_timestamped_evidence
from scripts.run_industry_fund_ranking import load_catalog, load_paths


METHODS=("HIST_RETURN_ONLY","M0_CATEGORY_BASELINE","M1_EB_FORECAST",
         "M1_NAV_RISK_PROXY")


def read_cash_history(path):
    d=pd.read_csv(path)
    required={"date","nav","known_at","source_url"}
    if not required.issubset(d.columns):
        raise ValueError(f"cash history missing {sorted(required-set(d.columns))}")
    d=d.copy()
    for col in ("date","known_at"):
        d[col]=pd.to_datetime(d[col],utc=True,errors="coerce").dt.tz_convert(None)
    d["nav"]=pd.to_numeric(d.nav,errors="coerce")
    if (d[["date","known_at"]].isna().any().any()
            or d.duplicated("date").any() or
            d.nav.isna().any() or d.nav.le(0).any() or
            d.source_url.fillna("").astype(str).str.strip().eq("").any()):
        raise ValueError("invalid dated cash total-return observations")
    return d.sort_values("date")


def cash_formation_rate(cash, t, lookback_months=6):
    available=cash.loc[(cash.date<=t)&(cash.known_at<=t)]
    if len(available)<25:
        raise ValueError("insufficient point-in-time investable cash history")
    end=available.iloc[-1]
    begin=t-pd.DateOffset(months=lookback_months)
    prior=available.loc[available.date<=begin]
    if prior.empty or (t-end["date"]).days>7 or (begin-prior.iloc[-1]["date"]).days>10:
        raise ValueError("cash benchmark formation endpoints unavailable")
    left=prior.iloc[-1]
    years=(end["date"]-left["date"]).days/365.2425
    if years<=0:
        raise ValueError("invalid cash rate window")
    return float((end.nav/left.nav)**(1/years)-1)


def date_catalog(folder,t):
    p=Path(folder)/f"{t.date()}.csv"
    if not p.is_file():
        raise ValueError(f"missing as-of historical catalog {p}")
    raw=pd.read_csv(p,dtype={"code":str},keep_default_na=False)
    required={"code","name","type","known_at","source_url"}
    if not required.issubset(raw.columns):
        raise ValueError(f"historical catalog missing {sorted(required-set(raw.columns))}")
    known=pd.to_datetime(raw.known_at,utc=True,errors="coerce").dt.tz_convert(None)
    if (known.isna().any() or known.gt(t).any()
            or raw.source_url.str.strip().eq("").any()
            or raw.code.duplicated().any()):
        raise ValueError("historical catalog has lookahead, conflicts or absent evidence")
    return load_catalog(p,periodic_overrides={})


def future_cash_nav(cash,dates,formation,evaluation):
    # Future cash NAV is used strictly AFTER models and fund selections freeze.
    z=cash.set_index("date").nav.sort_index()
    xs=z.reindex(pd.DatetimeIndex(dates),method="ffill")
    if xs.isna().any() or evaluation-cash.loc[cash.date<=evaluation,"date"].max()>pd.Timedelta(days=10):
        raise ValueError("missing realized cash benchmark")
    return xs.to_numpy(dtype=float)


def fold_signals(catalog, paths, benchmark, cash, valuation, t, h, H,
                 *, top=10,scenarios=64,grid_points=31,seed=42):
    frame,_=assemble(catalog,paths,t,h,[])
    if frame.empty:
        return pd.DataFrame()
    frame["feature_status"]="HISTORICAL_FEATURE_AVAILABLE"
    assessed=audit_valuation(frame,valuation)
    d=descriptive_research(assessed,t,h)
    models=fit_category_forecasts(frame,paths,benchmark,t,HORIZONS[h][0],H)
    if models.empty:
        return pd.DataFrame()
    joined=d.merge(models[["code","forecast_mu_annual","forecast_se_annual",
                            "bench_log"]],on="code",how="inner",
                   validate="one_to_one")
    joined=joined.loc[joined.historical_return_cagr.notna()].copy()
    if joined.empty:
        return pd.DataFrame()
    cash_rate=cash_formation_rate(cash,t)
    joined["HIST_RETURN_ONLY"]=joined.historical_return_cagr
    joined["M0_CATEGORY_BASELINE"]=np.expm1(joined.bench_log)
    joined["M1_EB_FORECAST"]=joined.forecast_mu_annual
    joined["M1_NAV_RISK_PROXY"]=np.nan
    joined["proxy_weight"]=np.nan
    for row in joined.itertuples():
        # A NAV risk proxy is NOT economic risk. Exclude appraisal smoothing,
        # unknown-pricing and amortized-cost NAV from this *particular* method.
        if row.pricing_method!="market_value" or bool(row.zero_or_tiny_nav_mdd):
            continue
        try:
            vals=windowed_values(paths[row.code],t,h)
            key=int(hashlib.sha256(f"{row.code}:{h}".encode()).hexdigest()[:8],16)
            f=joint_forecast_paths(vals,row.forecast_mu_annual,
                 steps=21*H,scenarios=scenarios,seed=(seed+key)%2**32,
                 forecast_se_annual=row.forecast_se_annual)
            c=cash_paths(cash_rate,steps=21*H,scenarios=scenarios)
            setting=OpportunityInput(row.forecast_mu_annual,cash_rate,
                 0.,0.,decision_months=H,grid_points=grid_points)
            score=optimize_opportunity(f,c,setting)
            best=next(x for x in score if x["budget"]==.04)
            joined.at[row.Index,"M1_NAV_RISK_PROXY"]=best["opportunity_score_annual"]
            joined.at[row.Index,"proxy_weight"]=best["weight"]
        except (ValueError,OverflowError,FloatingPointError):
            continue
    # For fair paired selection, compare M0, M1 and history on the SAME cohort.
    rows=[]
    for method in METHODS:
        subset=joined.loc[joined[method].notna()].copy()
        if method != "M1_NAV_RISK_PROXY":
            subset=subset.loc[subset["M1_NAV_RISK_PROXY"].notna()]
        else:
            subset=subset.loc[subset[method].gt(0)]
        subset=subset.sort_values([method,"code"],ascending=[False,True]).head(top)
        for code,r in subset.set_index("code").iterrows():
            rows.append(dict(formation_date=t,code=code,horizon=h,
                 method=method,selection_value=float(r[method]),
                 weight=(float(r.proxy_weight) if method=="M1_NAV_RISK_PROXY"
                         else 1.),historical_cagr=r.historical_return_cagr,
                 valuation_status=r.risk_quality,
                 decision_months=H,model_status="NAV_RESEARCH_NOT_EXECUTABLE"))
    return pd.DataFrame(rows)


def realized_signals(signals,paths,cash,evaluation):
    rows=[]
    for r in signals.itertuples(index=False):
        out=r._asdict()
        out["evaluation_end"]=evaluation
        path=paths.get(r.code)
        if path is None:
            continue
        a,b=path.right(r.formation_date),path.right(evaluation)
        if a<0 or b<a+3 or (evaluation-path.dates[b]).days>10:
            continue
        date=path.dates[a:b+1]
        try:
            z=future_cash_nav(cash,date,r.formation_date,evaluation)
            score=realized_buy_hold(path.value[a:b+1],z,r.weight,0.)
        except ValueError:
            continue
        out.update(score,pit_status="PIT_OK",evaluation_status="NAV_ONLY_NO_FEES")
        rows.append(out)
    return pd.DataFrame(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-catalogs",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--benchmark-history",type=Path,required=True)
    p.add_argument("--cash-nav",type=Path,required=True)
    p.add_argument("--valuation-evidence",type=Path)
    p.add_argument("--dates",type=Path,required=True,
                   help="CSV with formation_date: one explicit historical snapshot per fold")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--lookback",choices=list(HORIZONS),default="1y")
    p.add_argument("--decision-months",type=int,choices=[3,6,12],default=6)
    p.add_argument("--validation-end",type=str,required=True)
    p.add_argument("--test-start",type=str,required=True)
    p.add_argument("--top",type=int,default=10)
    p.add_argument("--min-test-folds",type=int,default=5)
    p.add_argument("--scenarios",type=int,default=64)
    p.add_argument("--grid-points",type=int,default=31)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(argv)
    if a.top<1 or a.scenarios<32 or a.grid_points<2:
        p.error("invalid search parameters")
    val_end,test_start=pd.Timestamp(a.validation_end),pd.Timestamp(a.test_start)
    if test_start <= val_end+pd.DateOffset(months=a.decision_months):
        p.error("test split must be purged by more than the prediction horizon")
    periods=pd.read_csv(a.dates)
    if "formation_date" not in periods:
        p.error("formation dates CSV needs formation_date")
    dates=pd.to_datetime(periods.formation_date,errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        p.error("formation dates missing or duplicate")
    benchmark=pd.read_csv(a.benchmark_history)
    cash=read_cash_history(a.cash_nav)
    folds=[]
    for t in sorted(dates):
        evaluate=t+pd.DateOffset(months=a.decision_months)
        catalog=date_catalog(a.historical_catalogs,t)
        # At t, features/forecasts are constrained <=t even though NAV history
        # is loaded through evaluation date for the separate label stage.
        paths,_=load_paths(catalog,a.data_dir,evaluate)
        evidence=read_timestamped_evidence(a.valuation_evidence,t,
                         ("pricing_method",))
        sig=fold_signals(catalog,paths,benchmark,cash,evidence,t,
              a.lookback,a.decision_months,top=a.top,
              scenarios=a.scenarios,grid_points=a.grid_points,seed=a.seed)
        if sig.empty:
            continue
        realized=realized_signals(sig,paths,cash,evaluate)
        if len(realized):
            realized["split"]=np.where(realized.evaluation_end.le(val_end),
                "validation",np.where(realized.formation_date.ge(test_start),
                "test","purged"))
            folds.append(realized)
    a.output.mkdir(parents=True,exist_ok=True)
    result=pd.concat(folds,ignore_index=True) if folds else pd.DataFrame()
    result.to_csv(a.output/"historical_folds.csv",index=False,encoding="utf-8-sig")
    # Freeze method choice BEFORE looking at test labels; no re-tuning on test.
    validation=result.loc[result.split.eq("validation")] if len(result) else pd.DataFrame()
    candidates=("M1_EB_FORECAST","M0_CATEGORY_BASELINE","HIST_RETURN_ONLY")
    chosen=None
    if len(validation):
        metrics=validation.groupby(["formation_date","method"])["cash_excess"].mean().reset_index()
        piv=metrics.pivot(index="formation_date",columns="method",values="cash_excess")
        common=piv.dropna(subset=list(candidates)) if set(candidates).issubset(piv.columns) else pd.DataFrame()
        if len(common)>=a.min_test_folds:
            chosen=max(candidates,key=lambda m:(common[m].mean(),-candidates.index(m)))
    test=result.loc[result.split.eq("test")] if len(result) else pd.DataFrame()
    comparisons=[]
    if chosen is not None and len(test):
        for baseline in candidates:
            if baseline==chosen:
                continue
            delta=nonoverlapping_pairs(test,chosen,baseline,
                                        min_folds=a.min_test_folds,seed=a.seed)
            comparisons.extend(delta.assign(baseline=baseline).to_dict("records"))
    pd.DataFrame(comparisons).to_csv(a.output/"test_paired_deltas.csv",
                                  index=False,encoding="utf-8-sig")
    status=dict(method_preselected=chosen or "",valuation_source_supplied=bool(a.valuation_evidence),
        fold_signal_records=len(result),validation_records=len(validation),
        test_records=len(test),oos_certified=False,
        research_status=("NAV_RESEARCH_TEST_ONLY" if chosen and comparisons
                         else "EVIDENCE_INSUFFICIENT"),
        asof_catalogs_required=True,execution_verified=False,
        economic_risk_validated=False,
        note="A rolling NAV-only exploratory test, not official valuation or investable OOS.")
    (a.output/"audit.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False),flush=True)
    return status


if __name__=="__main__":
    main()
