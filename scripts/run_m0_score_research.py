#!/usr/bin/env python3
"""Predefined M0 score ablation on the 2026-11-27 debug export.

Historical stress uses a common definition (worst rolling 3-month return),
not the current event-specific shock metric. All historical folds are
development evidence, not an untouched test set. No look-ahead selection.
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_debug_data_20261127 import DATA, ASOF, HORIZONS, read, verify, clean, month_features, LOCK_RE
from fundkit.oos_calendar import complete_calendar, VERIFIED_SUSPENSIONS
from run_oos_model_comparison import FORMATION_DATES
from fundkit.m0_score_research import (WEIGHTS, MODELS, METRICS, monthly_metrics,
    product_representatives, build_scores, future_matrix, portfolio_stats)

OUT=Path(__file__).resolve().parents[1]/"artifacts/m0_score_research_20261128"
DEFENSIVE=["中长纯债","短债/中短债","利率债","信用债","指数型-固收"]
TOPS=(10,20,50)


def prepare(f,monthly,cutoff,current=False):
    d,a=clean(f,month_features(monthly))
    m=monthly_metrics(monthly,cutoff)
    d=d.merge(m,on="code",validate="one_to_one")
    valid=d.complete & ~d.name.str.contains(LOCK_RE,na=False)
    if current:
        valid &= d.sub_status.isin(["开放申购","限大额"]) & d.red_status.eq("开放赎回")
        valid &= np.isfinite(d[["Aret","Add"]]).all(axis=1)
    d=d.loc[valid].copy()
    a["eligible_shares_after_lock_and_status"]=len(d)
    d=product_representatives(d)
    a["eligible_products"]=len(d)
    a["ratio_floor_counts"]={c:int(d[c].sum()) for c in ["sharpe_floor","sortino_floor"]}
    return d,a


def universes(d):
    return {"all":d,"defensive_bond":d.loc[d.group.isin(DEFENSIVE)&
        ~d.name.str.contains(r"转债|可转换",na=False)]}


def ranked(z,m,k):
    return z.sort_values([m,"code"],ascending=[False,True]).head(k)


def rank_corr(z):
    cols=["P_"+x for x in METRICS]
    return z[cols].corr(method="spearman")


def overlaps(z):
    b={k:set(ranked(z,"BASE",k).code) for k in TOPS}
    return [dict(model=m,k=k,common=len(b[k]&set(ranked(z,m,k).code))) for m in MODELS for k in TOPS]


def outcome_analysis(z,gross):
    # Outcome-only availability filter used exclusively for cross-sectional
    # diagnostics, never for selection. Report missing coverage explicitly.
    x=gross.reindex(z.code).to_numpy()
    good=np.isfinite(x).all(axis=1)
    q=z.loc[good].copy()
    v=x[good]
    if not len(q):
        return [],[],dict(eligible=len(z),outcome_valid=0)
    q["future_return"]=v[:,-1]-1
    q["future_mdd"]=np.max(1-v/np.maximum.accumulate(v,axis=1),axis=1)
    ic=[]
    for m in [*["P_"+c for c in METRICS],*MODELS]:
        ranks=q[[m,"future_return","future_mdd"]].corr(method="spearman")
        ic.append(dict(metric=m,ic_return=float(ranks.loc[m,"future_return"]),
                       ic_low_drawdown=float(-ranks.loc[m,"future_mdd"])))
    quint=[]
    for m in MODELS:
        p=q.sort_values([m,"code"],ascending=[False,True])
        for i,ids in enumerate(np.array_split(np.arange(len(p)),5),1):
            if not len(ids):continue
            g=p.iloc[ids]
            quint.append(dict(model=m,quintile=i,n=len(g),
                mean_future_return=float(g.future_return.mean()),mean_future_mdd=float(g.future_mdd.mean())))
    return ic,quint,dict(eligible=len(z),outcome_valid=len(q))


def current():
    audits=[]; tops=[]; sensitivity=[]
    for h in HORIZONS:
        f=read(DATA/f"features/features_{h}.csv.gz")
        monthly=read(DATA/f"features/monthly_{h}.csv.gz")
        d,a=prepare(f.loc[f.status.eq("SCORED")],monthly,ASOF,True)
        print("CURRENT_COVERAGE",h,json.dumps(a),flush=True)
        for u,cohort in universes(d).items():
            for shock in (False,True):
                label="actual_shock" if shock else "rolling3_proxy"
                z=build_scores(cohort,actual_shock=shock)
                z.to_csv(OUT/f"current_{h}_{u}_{label}_full.csv.gz",index=False)
                rank_corr(z).to_csv(OUT/f"current_{h}_{u}_{label}_metric_corr.csv")
                sensitivity.extend(dict(horizon=h,universe=u,stress=label,**x) for x in overlaps(z))
                for m in MODELS:
                    cols=["code","name","group","product_key","annualized_return","D","stress_proxy","L","L_daily","sharpe","sortino","calmar","recovery","consistency",m]
                    v=ranked(z,m,50)[cols].copy()
                    v.insert(0,"rank",np.arange(1,len(v)+1))
                    v=v.rename(columns={m:"score"})
                    v.to_csv(OUT/f"current_{h}_{u}_{label}_{m}_top50.csv",index=False)
                    # One combined table simplifies downstream rank comparisons.
                    tops.append(v.assign(horizon=h,universe=u,stress=label,model=m))
                    if shock and u=="all" and m in ("BASE","BALANCED","DRAWDOWN","DAILY","SHARPE","SORTINO"):
                        print("CURRENT_TOP",h,m,v.head(10)[["code","name","annualized_return","D","score"]].round(6).to_json(orient="records",force_ascii=False),flush=True)
        audits.append(dict(horizon=h,**a))
    pd.concat(tops,ignore_index=True).to_csv(OUT/"current_top50.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/"current_sensitivity.csv",index=False)
    return audits


def oos():
    calendar=complete_calendar(read(DATA/"oos/market_calendar.csv").trade_date,ASOF)
    rows=[]; ics=[]; quint=[]; sensitivity=[]; coverage=[]
    for day in FORMATION_DATES:
        t=pd.Timestamp(day);end=t+pd.DateOffset(months=6)
        f=read(DATA/f"oos/fold_{day}_pre_features.csv.gz")
        monthly=read(DATA/f"oos/fold_{day}_pre_monthly.csv.gz")
        d,a=prepare(f,monthly,t)
        cohorts={u:build_scores(g) for u,g in universes(d).items()}
        # Freeze every selection before reading post-formation data.
        chosen={}
        for u,z in cohorts.items():
            for m in MODELS:
                for k in TOPS:
                    chosen[u,m,k]=ranked(z,m,k).code.tolist()
            rank_corr(z).to_csv(OUT/f"fold_{day}_{u}_metric_corr.csv")
            sensitivity.extend(dict(formation=day,universe=u,**x) for x in overlaps(z))
        post=read(DATA/f"oos/fold_{day}_post_daily.csv.gz")
        market=calendar[(calendar>t)&(calendar<=end)]
        gross=future_matrix(post,d.code.tolist(),market,VERIFIED_SUSPENSIONS)
        for u,z in cohorts.items():
            i,q,c=outcome_analysis(z,gross)
            coverage.append(dict(formation=day,universe=u,**c))
            ics.extend(dict(formation=day,universe=u,**x) for x in i)
            quint.extend(dict(formation=day,universe=u,**x) for x in q)
        for (u,m,k),codes in chosen.items():
            perf=portfolio_stats(gross,codes)
            rows.append(dict(formation=day,universe=u,model=m,k=k,codes="|".join(codes),**perf))
        print("FOLD_COVERAGE",day,json.dumps([x for x in coverage if x["formation"]==day]),flush=True)
    for name,data in [("oos",rows),("ic",ics),("quintiles",quint),("oos_sensitivity",sensitivity),("outcome_coverage",coverage)]:
        pd.DataFrame(data).to_csv(OUT/f"{name}.csv",index=False)
    return pd.DataFrame(rows),pd.DataFrame(ics),pd.DataFrame(quint)


def summarize(rows,ic,quint):
    result=[]
    for (u,k),g in rows.groupby(["universe","k"]):
        # Primary: dates jointly valid across all predefined models.
        valid=g.groupby("formation").status.apply(lambda x:len(x)==len(MODELS) and x.eq("OK").all())
        common=valid[valid].index.tolist()
        # Pairwise vs BASE additionally retained, never compare unmatched periods.
        base=g.loc[g.model.eq("BASE")].set_index("formation")
        for m,v in g.groupby("model"):
            a=v.loc[v.formation.isin(common)].sort_values("formation")
            pair=v.set_index("formation")
            pair=pair.loc[pair.status.eq("OK")&base.status.eq("OK")]
            delta=pair.net_return-base.loc[pair.index,"net_return"]
            z=dict(universe=u,k=int(k),model=m,common_folds=len(a),dates=common,
                   valid_folds=int(v.status.eq("OK").sum()),pair_folds=len(pair),
                   pair_mean_delta=float(delta.mean()) if len(delta) else None,
                   pair_wins=int((delta>1e-10).sum()))
            if len(a):
                z.update(fold_equivalent_cagr=float(np.prod(1+a.net_return)**(2/len(a))-1),
                    mean_fold_mdd=float(a.mdd.mean()),worst_fold_mdd=float(a.mdd.max()))
            result.append(z)
            print("SUMMARY",json.dumps(z),flush=True)
    pd.DataFrame(result).to_csv(OUT/"summary.csv",index=False)
    for (u,m),g in ic.groupby(["universe","metric"]):
        print("IC",json.dumps(dict(universe=u,metric=m,mean_return_ic=g.ic_return.mean(),
             mean_low_dd_ic=g.ic_low_drawdown.mean(),folds=int(g.ic_return.notna().sum()))),flush=True)
    for (u,m),g in quint.groupby(["universe","model"]):
        q=g.pivot(index="formation",columns="quintile",values="mean_future_return")
        print("QUINTILE",json.dumps(dict(universe=u,model=m,means=q.mean().tolist(),
            top_minus_bottom=float((q[1]-q[5]).mean()))),flush=True)
    # Development-only rolling choice: rank configs by worst drawdown first
    # under a 4% observed fold cap, then CAGR. Only earlier complete outcomes.
    # It is not an independent holdout; no choice if <3 prior common folds.
    walk=[]
    for (u,k),g in rows.groupby(["universe","k"]):
        dates=sorted(g.formation.unique())
        for i,day in enumerate(dates):
            # Holding periods end on/after next formation. Require strict
            # completion before selection, excluding same-day endpoint NAV.
            earlier=[x for x in dates[:i] if pd.Timestamp(x)+pd.DateOffset(months=6)<pd.Timestamp(day)]
            train=g.loc[g.formation.isin(earlier)]
            good=train.groupby("formation").status.apply(lambda s:len(s)==len(MODELS) and s.eq("OK").all())
            train=train.loc[train.formation.isin(good[good].index)]
            if train.formation.nunique()<3:continue
            stats=train.groupby("model").agg(worst=("mdd","max"),growth=("net_return",lambda s:float(np.prod(1+s))))
            feasible=stats.loc[stats.worst<=.04]
            chosen=(feasible.sort_values(["growth","worst"],ascending=[False,True]).index[0]
                    if len(feasible) else "BASE")
            test=g.loc[(g.formation==day)&(g.model==chosen)].iloc[0]
            baseline=g.loc[(g.formation==day)&(g.model=="BASE")].iloc[0]
            walk.append(dict(universe=u,k=int(k),formation=day,model=chosen,
                training_folds=int(train.formation.nunique()),status=test.status,
                net_return=test.net_return if test.status=="OK" else None,
                mdd=test.mdd if test.status=="OK" else None,
                baseline_return=baseline.net_return if baseline.status=="OK" else None))
    pd.DataFrame(walk).to_csv(OUT/"walk_forward_choice.csv",index=False)
    for x in walk: print("WALK_FORWARD",json.dumps(x),flush=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=verify()
    a=current()
    rows,ic,q=oos()
    summary=summarize(rows,ic,q)
    (OUT/"audit.json").write_text(json.dumps(dict(source=source,weights=WEIGHTS,models=MODELS,
        current=a,summary=summary,limitations=[
        "past seven folds repeatedly examined; development evidence, no untouched holdout",
        "only one-year historical formation features; current 5y/3y/6m not independently validated",
        "historical rolling 3-month stress proxy is not current event-specific shock",
        "monthly zero-hurdle Sharpe/Sortino with 1bp denominator floor, not risk-free ratios",
        "one product one vote, A share preferred before ranking; current status only, historical eligibility unavailable",
        "individual outcomes complete-case only for IC/quintiles; selected portfolios rejected if missing, no outcome-based replacement",
        "uniform 10bp roundtrip, no actual historical fee or survivorship correction",
        "fold drawdown not full continuous portfolio risk; no guaranteed 4% cap",
        "walk-forward strict availability, illustrative and sparse"
    ]),ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":
    main()
