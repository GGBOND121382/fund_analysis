#!/usr/bin/env python3
"""Strict historical D95 reconstruction from ALREADY elapsed post-fold slices.

For formation k, only slices k-2 and k-1 are permitted. Verify reconstructed
R, D, daily downside and annualization against the original pre-feature row.
M0/M1/M2/M6/M7 are ranked on the same reconstructable cohort. The first two
folds are unavailable, and the last fold is provisional (calendar incomplete).
No data from the evaluated post slice may enter the score.
"""
import json
import sys
from pathlib import Path
from collections import Counter
from functools import lru_cache
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_debug_data_20261127 import DATA, read, verify, clean, month_features, board, LOCK_RE, score as full_score
from rank_models_m123 import pct, eb_posteriors, BOOT_STEPS
from fundkit.industry_rank import NavPath, nav_quality_issue
from fundkit.opportunity_rank import stationary_bootstrap, max_drawdown
from run_oos_model_comparison import FORMATION_DATES, future_perf, ROUNDTRIP_FEE
from fundkit.oos_calendar import complete_calendar, VERIFIED_SUSPENSIONS, CALENDAR_SOURCE, SUSPENSION_SOURCE

OUT = Path(__file__).resolve().parents[1]/"artifacts/bootstrap_audit_20261128"
MODELS = ["M0","M1","M2","M6","M7"]
DEFENSIVE = ["中长纯债","短债/中短债","利率债","信用债","指数型-固收"]
SCENARIOS = 500
BLOCK = 10
SEED = 42


@lru_cache(maxsize=512)
def bootstrap_indices(n, scenarios=SCENARIOS, steps=BOOT_STEPS, block=BLOCK, seed=SEED):
    # Exactly the same random draw order as stationary_bootstrap.
    rng=np.random.default_rng(seed)
    idx=rng.integers(n,size=scenarios)
    a=np.empty((scenarios,steps),dtype=np.int32)
    for t in range(steps):
        if t:
            choose=rng.random(scenarios)<1/block
            idx=np.where(choose,rng.integers(n,size=scenarios),(idx+1)%n)
        a[:,t]=idx
    return a


def fast_d95(v, scenarios=SCENARIOS, block=BLOCK, seed=SEED):
    x=np.diff(np.log(np.asarray(v,dtype=float)))
    if len(x)<64 or not np.isfinite(x).all():
        raise ValueError("invalid bootstrap history")
    w=np.cumsum(x[bootstrap_indices(len(x),scenarios,BOOT_STEPS,block,seed)],axis=1)
    peak=np.maximum(0,np.maximum.accumulate(w,axis=1))
    dd=-np.expm1(np.min(w-peak,axis=1))
    return float(np.quantile(dd,.95))


def reconstruct(d, history, t):
    begin=t-pd.DateOffset(months=12)
    if len(history) and pd.Timestamp(history.date.max())>t:
        raise ValueError("future history provided to reconstruction")
    counts=Counter()
    rows=[]
    by=d.set_index("code")
    hist=history.loc[history.code.isin(by.index)].sort_values(["code","date"])
    for code,g in hist.groupby("code",sort=False):
        r=by.loc[code]
        # The same historical observation in adjacent slices must agree.
        dates=g.date.to_numpy()
        vals=g.nav.to_numpy(dtype=float)
        dup=dates[1:]==dates[:-1]
        if dup.any() and not np.allclose(vals[1:][dup],vals[:-1][dup],rtol=1e-10,atol=1e-12):
            counts["overlap_conflict"]+=1; continue
        if dup.any():
            g=g.drop_duplicates("date")
        n=NavPath(pd.DatetimeIndex(g.date),g.nav.to_numpy(dtype=float))
        i,j=n.right(begin),n.right(t)
        if i<0 or j-i+1<200 or n.day(begin)-n.days[i]>7 or n.day(t)-n.days[j]>7:
            counts["window_coverage"]+=1; continue
        if np.any(n.gaps[i:j]>14):
            counts["gap"]+=1; continue
        v=n.value[i:j+1]
        if nav_quality_issue(v):
            counts["nav_quality"]+=1; continue
        ret,dd=n.stats(i,j)
        years=(n.days[j]-n.days[i])/365.2425
        daily=np.diff(v)/v[:-1]
        down=np.sqrt(252*np.mean(np.minimum(daily,0)**2))
        actual=np.array([ret,dd,(1+ret)**(1/years)-1,down])
        expected=np.array([r.period_return,r.D,r.annualized_return,r.L_daily],dtype=float)
        if not np.allclose(actual,expected,rtol=1e-7,atol=1e-9):
            counts["pre_feature_mismatch"]+=1; continue
        rows.append(dict(code=code,D95=fast_d95(v),history_start=str(n.dates[i].date()),
                         history_end=str(n.dates[j].date()),history_obs=len(v)))
    counts["missing_history"]=len(by)-hist.code.nunique()
    counts["verified"]=len(rows)
    return pd.DataFrame(rows,columns=["code","D95","history_start","history_end","history_obs"]),dict(counts)


def scores(d):
    z=eb_posteriors(d)
    R=pct(z.annualized_return); E=pct(z.mu_lower)
    D=100-pct(z.D); B=100-pct(z.D95)
    L=100-pct(z.L); Y=100-pct(z.L_daily)
    for model,r,dd,l in [("M0",R,D,L),("M1",E,D,L),("M2",E,B,L),("M6",R,B,L),("M7",R,D,Y)]:
        z[model]=(.40*r+.35*dd+.10*l)/.85
    return z


def paths_for(post,codes):
    out={}
    for c,g in post.loc[post.code.isin(codes)].groupby("code"):
        g=g.sort_values("date")
        if g.date.duplicated().any() or not np.isfinite(g.nav).all() or not g.nav.gt(0).all():
            continue
        out[c]=NavPath(pd.DatetimeIndex(g.date),g.nav.to_numpy(dtype=float))
    return out


def attribution(top,paths,t,end,model,label,calendar):
    rows=[]
    for r in top.itertuples(index=False):
        n=paths.get(r.code)
        if n is None:
            rows.append(dict(formation=str(t.date()),model=model,universe=label,code=r.code,name=r.name,issue="MISSING_NAV"))
            continue
        i,j=n.right(t)+1,n.right(end)
        v=n.value[i:j+1]
        if len(v)<2: continue
        daily=v[1:]/v[:-1]-1
        k=int(np.argmax(np.abs(daily)))
        missing=calendar[(calendar>t)&(calendar<=end)&~calendar.isin(n.dates[i:j+1])]
        rows.append(dict(formation=str(t.date()),model=model,universe=label,code=r.code,name=r.name,group=r.group,
            pre_return=r.annualized_return,pre_D=r.D,pre_D95=getattr(r,"D95",None),
            post_return=float(v[-1]/v[0]-1),post_D=float(np.max(1-v/np.maximum.accumulate(v))),
            largest_daily_return=float(daily[k]),largest_move_date=str(n.dates[i+k+1].date()),
            missing_calendar_count=len(missing),missing_dates="|".join(str(x.date()) for x in missing),
            post_first=str(n.dates[i].date()),post_last=str(n.dates[j].date())))
    return rows


def calendar_review(post,d,t,calendar):
    # No inference of an official trading calendar: consensus is a diagnostic only.
    domestic=d.loc[d.group.isin(DEFENSIVE),"code"]
    p=post.loc[post.code.isin(domestic)]
    counts=p.groupby("date").code.nunique().sort_index()
    base=counts.loc[counts.index.isin(calendar.strftime("%Y-%m-%d"))]
    median=float(base.median()) if len(base) else 0
    tail=counts.loc[counts.index>str(calendar.max().date())]
    return dict(official_last=str(calendar.max().date()),reference_median=median,
                tail_observations={str(k):int(v) for k,v in tail.items()},
                proxy_candidates=[str(k) for k,v in tail.items() if median>0 and v>=.8*median])


def summary(rows):
    z=pd.DataFrame(rows)
    out=[]
    for u,g in z.groupby("universe"):
        for include in (False,True):
            a=g if include else g.loc[g.calendar_complete]
            # All models must have a valid outcome on a common fold.
            good=a.groupby("formation").net_return.apply(lambda x:len(x)==len(MODELS) and x.notna().all())
            a=a.loc[a.formation.isin(good[good].index)]
            base=a.loc[a.model.eq("M0")].set_index("formation")
            for m,v in a.groupby("model"):
                v=v.set_index("formation").sort_index()
                delta=v.net_return-base.loc[v.index,"net_return"]
                out.append(dict(universe=u,include_provisional=include,model=m,folds=len(v),
                    dates=v.index.tolist(),fold_equivalent_cagr=float(np.prod(1+v.net_return)**(2/len(v))-1),
                    mean_return=float(v.net_return.mean()),worst_fold_dd=float(v.mdd.max()),
                    mean_fold_dd=float(v.mdd.mean()),wins=int((delta>1e-10).sum()),mean_delta=float(delta.mean())))
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=verify()
    # Independently verify optimized bootstrap against the reference.
    v=np.exp(np.cumsum(np.r_[0,np.random.default_rng(19).normal(.0001,.007,243)]))
    ref=float(np.quantile(max_drawdown(stationary_bootstrap(v,steps=BOOT_STEPS,scenarios=SCENARIOS,block_length=BLOCK,seed=SEED)),.95))
    assert abs(ref-fast_d95(v))<1e-12
    original_calendar=pd.DatetimeIndex(pd.to_datetime(read(DATA/"oos/market_calendar.csv").trade_date))
    calendar=complete_calendar(original_calendar,"2026-11-20")
    days=calendar.to_numpy(dtype="datetime64[D]").astype("int64")
    rows=[]; attr=[]; full_attr=[]; coverage=[]; cal=[]
    previous=[]
    for k,day in enumerate(FORMATION_DATES):
        t=pd.Timestamp(day); end=t+pd.DateOffset(months=6)
        f=read(DATA/f"oos/fold_{day}_pre_features.csv.gz")
        monthly=read(DATA/f"oos/fold_{day}_pre_monthly.csv.gz")
        d,_=clean(f,month_features(monthly))
        d=d.loc[~d.name.str.contains(LOCK_RE,na=False)].copy()
        if k>=2:
            hist=pd.concat(previous[-2:],ignore_index=True)
            # A previous fold may end exactly on t; never accept later observations.
            hist=hist.loc[hist.date.le(day)]
            features,counts=reconstruct(d,hist,t)
            z=d.merge(features,on="code",validate="one_to_one")
            del hist
            cov=dict(formation=day,eligible_before=len(d),**counts)
            coverage.append(cov); print("COVERAGE",json.dumps(cov),flush=True)
            universes={"all":z,"defensive_bond":z.loc[z.group.isin(DEFENSIVE)&~z.name.str.contains(r"转债|可转换",na=False)]}
            selections={}
            for label,universe in universes.items():
                ranked=scores(universe)
                for model in MODELS:
                    top=board(ranked,model)
                    selections[label,model]=top
                    top[["code","name","group","product_key","annualized_return","D","D95","L_daily",model]].to_csv(OUT/f"{day}_{label}_{model}.csv",index=False)
            # Reproduce the previous FULL-cohort extreme selections separately.
            # They are not used to fit D95 or select the restricted cohort.
            full_selections={}
            if day in ("2025-05-20","2026-05-20"):
                ranked_full=full_score(d,monthly,False,False)
                for model in ("M0","M1","M3","M4","M5","M7"):
                    full_selections[model]=board(ranked_full,model)
            # Future is loaded only after scores and selections are finalized.
            post=read(DATA/f"oos/fold_{day}_post_daily.csv.gz")
            paths=paths_for(post,{c for top in selections.values() for c in top.code})
            complete=bool(calendar.max()>=end-pd.Timedelta(days=3))
            for (label,model),top in selections.items():
                perf=future_perf(paths,top.code.tolist(),t,6,names=dict(zip(top.code,top.name)),market_days=days,suspensions=VERIFIED_SUSPENSIONS)
                row=dict(formation=day,universe=label,model=model,eligible=len(universes[label]),calendar_complete=complete,codes=top.code.tolist(),**perf)
                rows.append(row); print("BOOT_OOS",json.dumps(row,ensure_ascii=False),flush=True)
                attr.extend(attribution(top,paths,t,end,model,label,calendar))
            if full_selections:
                full_paths=paths_for(post,{c for top in full_selections.values() for c in top.code})
                for model,top in full_selections.items():
                    full_attr.extend(attribution(top,full_paths,t,end,model,"previous_full_cohort",calendar))
            if day=="2026-05-20":
                review=calendar_review(post,d,t,original_calendar); cal.append(review)
                print("CALENDAR",json.dumps(review),flush=True)
        else:
            post=read(DATA/f"oos/fold_{day}_post_daily.csv.gz")
        previous.append(post)
        previous=previous[-2:]
    pd.DataFrame(rows).to_csv(OUT/"oos.csv",index=False)
    pd.DataFrame(attr).to_csv(OUT/"attribution.csv",index=False)
    pd.DataFrame(full_attr).to_csv(OUT/"previous_full_cohort_attribution.csv",index=False)
    s=summary(rows)
    for x in s: print("BOOT_SUMMARY",json.dumps(x),flush=True)
    # Show extreme constituents and any missing-date causes, not fabricated replacements.
    for x in attr:
        if x["formation"]=="2026-05-20" and x["universe"]=="all" or x.get("missing_calendar_count",0):
            print("ATTRIBUTION",json.dumps(x,ensure_ascii=False),flush=True)
    for x in full_attr:
        if x["formation"]=="2026-05-20" or x.get("missing_calendar_count",0):
            print("FULL_ATTRIBUTION",json.dumps(x,ensure_ascii=False),flush=True)
    result=dict(source=source,coverage=coverage,summary=s,calendar=cal,calendar_source=CALENDAR_SOURCE,suspension_source=SUSPENSION_SOURCE,
        limits=["first two formation dates unavailable","cohort restricted to funds present in both prior export slices and reconciled to pre-features",
        "November 10-20 scheduled trading days added from official exchange calendar; only known suspension allowed for interior marking","historical daily returns reconstructed from prior elapsed slices, never current post",
        "fees uniform 10bp; survivorship and historical eligibility unavailable; no claim of guaranteed risk cap",
        "circular stationary bootstrap, 500 scenarios, 126 steps, block 10; no parameter search"])
    (OUT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
