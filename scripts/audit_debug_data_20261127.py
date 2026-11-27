#!/usr/bin/env python3
"""Recompute M0-M7 from exported full-market features; independent OOS ablation.

No raw full-market daily history required for this diagnostic. Current D95
is reused, not re-estimated. Historical D95 is absent from the package;
therefore do not claim M2 or M6 OOS results. No fitting on future NAV data.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from rank_models_m123 import eb_posteriors, pct
from fundkit.industry_rank import nav_quality_issue
from run_oos_model_comparison import future_perf, FORMATION_DATES
from fundkit.industry_rank import NavPath
from fundkit.model_ablation import add_return_ablation, identity_keys, cny_comparable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "debug_data/20261127"
OUT = ROOT / "artifacts/debug_ablation_20261128"
LOCK_RE = r"滚动|持有|定开|定期开放|定期债券|封闭"
CURRENT_MODELS = ["M0","M1","M2","M3","M4","M5","M6","M7"]
OOS_MODELS = ["M0","M1","M3","M4","M5","M7"]
HORIZONS = {"5y": 60, "3y": 36, "1y": 12, "6m": 6}
ASOF = pd.Timestamp("2026-11-20")


def read(p):
    return pd.read_csv(p, dtype={"code": str}, low_memory=False)


def verify():
    manifest = json.loads((DATA/"MANIFEST.json").read_text())
    for f in manifest["files"]:
        p = DATA/f["path"]
        assert p.is_file() and p.stat().st_size == f["bytes"], p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"], p
    return manifest["git_sha"]


def month_features(m):
    m = m.copy()
    assert not m.duplicated(["code","month"]).any()
    m = m.sort_values(["code","month"])
    m["x"] = np.log1p(m.monthly_return)
    assert np.isfinite(m.x).all()
    g = m.groupby("code").x
    d = g.agg(["mean","std","count"]).rename(columns={"count":"complete_months"})
    d["eb_observed_log_mean"] = 12*d["mean"]
    d["annual_se_recomputed"] = 12*d["std"]/np.sqrt(d.complete_months)
    d["zero_last5_months"] = g.apply(lambda x: len(x)>=5 and (x.iloc[-5:].abs() <= 1e-10).all())
    return d.drop(columns=["mean","std"]).reset_index()


def sample_audit():
    sample = pd.concat([read(p) for p in sorted((DATA/"sample").glob("nav_shard_*.csv"))],ignore_index=True)
    meta = read(DATA/"sample/sample_funds.csv").set_index("code")
    rows = []
    for c,g in sample.groupby("code"):
        if meta.loc[c,"pricing_basis"] == "income_per_10000":
            continue  # daily per-10000 income is NOT a wealth NAV
        g = g.sort_values("date")
        for h,n in HORIZONS.items():
            v = g.loc[g.date.between(str((ASOF-pd.DateOffset(months=n)).date()),str(ASOF.date())),"nav"].to_numpy()
            if not len(v):
                continue
            rows.append(dict(code=c,horizon=h,observations=len(v),
                             changes=int(np.count_nonzero(np.abs(np.diff(v))>1e-10)),
                             issue=nav_quality_issue(v),first=float(v[0]),last=float(v[-1])))
    a = pd.DataFrame(rows)
    a.to_csv(OUT/"sample_quality.csv",index=False)
    return a


def score(d, monthly, shock, bootstrap):
    d = add_return_ablation(eb_posteriors(d.copy()), monthly, eb_posteriors)
    sr=pct(d.annualized_return); eb=pct(d.mu_lower)
    sd=100-pct(d.D); sl=100-pct(d.L); dl=100-pct(d.L_daily)
    factor=1 if shock else 1/.85
    sa=.15*(.5*pct(d.Aret)+.5*(100-pct(d.Add))) if shock else 0
    d["M0"]=factor*(.4*sr+.35*sd+.1*sl)+sa
    d["M1"]=factor*(.4*eb+.35*sd+.1*sl)+sa
    d["M3"]=factor*(.4*eb+.35*sd+.1*dl)+sa
    d["M4"]=factor*(.4*pct(d.return_M4)+.35*sd+.1*sl)+sa
    d["M5"]=factor*(.4*pct(d.return_M5)+.35*sd+.1*sl)+sa
    d["M7"]=factor*(.4*sr+.35*sd+.1*dl)+sa
    if bootstrap:
        dd=100-pct(d.D95)
        d["M2"]=factor*(.4*eb+.35*dd+.1*sl)+sa
        d["M6"]=factor*(.4*sr+.35*dd+.1*sl)+sa
    return d


def clean(d, m, stale_codes=()):
    d=d.merge(m,on="code",how="left",validate="one_to_one")
    d["annual_se"]=d.annual_se_recomputed
    d["annualized_return"]=d.get("annualized_return",d.get("R"))
    # A nearly zero complete-month suffix is a review proxy only, not proof
    # of flat daily NAV. Exclude from this conservative diagnostic cohort.
    flat=(d.D.abs()<=1e-10)&(d.period_return.abs()<=1e-10)
    stale=d.zero_last5_months.eq(True)|d.code.isin(stale_codes)
    required=["annualized_return","D","L","L_daily","annual_se","eb_observed_log_mean"]
    if "D95" in d:
        required+=["D95","Aret","Add"]
    good=np.isfinite(d[required]).all(axis=1)&d.complete_months.ge(4)
    report=dict(input_rows=len(d),flat=int(flat.sum()),zero_month_suffix=int(stale.sum()),
                incomplete=int((~good).sum()))
    d=d.loc[good & ~flat & ~stale].copy()
    d["original_product_key"]=d.product_key
    d["product_key"]=identity_keys(d)
    cny=cny_comparable(d)
    report["excluded_non_cny_or_currency_review"]=int((~cny).sum())
    d=d.loc[cny].copy()
    report["eligible"]=len(d)
    report["unique_products"]=int(d.product_key.nunique())
    return d,report


def board(d,model):
    return d.sort_values([model,"code"],ascending=[False,True]).drop_duplicates("product_key").head(10)


def current(sample):
    results={}
    for h in HORIZONS:
        f=read(DATA/f"features/features_{h}.csv.gz")
        monthly=read(DATA/f"features/monthly_{h}.csv.gz")
        m=month_features(monthly)
        stale=sample.loc[sample.horizon.eq(h)&sample.issue.ne(""),"code"]
        d,a=clean(f.loc[f.status.eq("SCORED")].copy(),m,stale)
        d=score(d,monthly,True,True)
        locked=d.name.str.contains(LOCK_RE,na=False)
        free=(~locked & d.sub_status.isin(["开放申购","限大额"]) & d.red_status.eq("开放赎回"))
        a["eb_status"]=d.eb_status.value_counts().to_dict()
        a["factor_status"]=d.factor_status.value_counts().to_dict()
        a["factor_EB_tau_positive"]=int(d.factor_status.eq("FACTOR_EB").sum())
        d[["code","name","group","product_key","original_product_key","eb_status","eb_weight","mu_lower","return_M4","return_M5","factor_status","factor_eb_weight","factor_peer_count","factor_months","factor_mean","factor_se","residual_mean","residual_se",*CURRENT_MODELS]].to_csv(OUT/f"{h}_model_diagnostics.csv.gz",index=False)
        a["trade_status_snapshot_count"]=int(free.sum())
        a["boards"]={}
        for model in CURRENT_MODELS:
            cols=["code","name","annualized_return","D","mu_lower","return_M4","return_M5",model]
            a["boards"][model]={}
            for label,subset in [("research",d),("snapshot_free",d.loc[free])]:
                top=board(subset,model)[cols]
                top.to_csv(OUT/f"{h}_{model}_{label}.csv",index=False)
                a["boards"][model][label]=top.round(6).to_dict("records")
        results[h]=a
    return results


def oos():
    rows=[]
    calendar=pd.to_datetime(read(DATA/"oos/market_calendar.csv").trade_date)
    days=calendar.to_numpy(dtype="datetime64[D]").astype("int64")
    for day in FORMATION_DATES:
        t=pd.Timestamp(day)
        f=read(DATA/f"oos/fold_{day}_pre_features.csv.gz")
        monthly=read(DATA/f"oos/fold_{day}_pre_monthly.csv.gz")
        assert pd.PeriodIndex(monthly.month,freq="M").max() < t.to_period("M"), day
        m=month_features(monthly)
        d,a=clean(f,m)
        d=d.loc[~d.name.str.contains(LOCK_RE,na=False)].copy()
        # Use defensive-bond universe separately from all-market. Current
        # product names are proxies, not point-in-time contract evidence.
        universes={"all":d,"defensive_bond":d.loc[d.group.isin(["中长纯债","短债/中短债","利率债","信用债","指数型-固收"]) &
                  ~d.name.str.contains(r"转债|可转换",na=False)]}
        selections={}
        for label,universe in universes.items():
            ranked=score(universe,monthly,False,False)
            for model in OOS_MODELS:
                top=board(ranked,model)
                selections[label,model]=top
                top[["code","name","group","product_key","annualized_return","D","factor_status",model]].to_csv(OUT/f"fold_{day}_{label}_{model}.csv",index=False)
        codes=set(c for top in selections.values() for c in top.code)
        post=read(DATA/f"oos/fold_{day}_post_daily.csv.gz")
        paths={}
        for c,g in post.loc[post.code.isin(codes)].groupby("code"):
            g=g.sort_values("date")
            paths[c]=NavPath(pd.DatetimeIndex(g.date),g.nav.to_numpy())
        for (label,model),top in selections.items():
            perf=future_perf(paths,top.code.tolist(),t,6,names=dict(zip(top.code,top.name)),market_days=days)
            rows.append(dict(formation=day,universe=label,model=model,eligible=len(universes[label]),
                             factor_eb_selected=int(top.factor_status.eq("FACTOR_EB").sum()),
                             calendar_complete=bool(calendar.max()>=t+pd.DateOffset(months=6)-pd.Timedelta(days=3)),
                             codes=top.code.tolist(),**perf))
    pd.DataFrame(rows).to_csv(OUT/"oos.csv",index=False)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=verify()
    sample=sample_audit()
    a={"source_sha":src,"limitations":["current D95 reused from old exported feature table",
         "zero-month suffix review proxy for nonsampled funds","current status snapshot 2026-11-17, no PIT trading or fees",
         "historical D95 absent, M2/M6 OOS not computed","calendar ends 2026-11-09, final fold incomplete calendar",
         "surviving catalog; conditional EB not a calibrated expected return forecast",
         "CNY comparable share cohort; no FX imputation; legacy currency metadata may be over-conservative",
         "name-based lock exclusion applied before OOS ranking; actual PIT terms unknown",
         "M5 internal category factor, fixed beta=1; no external factor forecast; conservative sum-of-SE penalty"],
         "current":current(sample),"oos":oos()}
    (OUT/"audit.json").write_text(json.dumps(a,ensure_ascii=False,indent=2,default=str))
    for h,v in a["current"].items():
        print("AUDIT",h,{k:x for k,x in v.items() if k!="boards"},flush=True)
        for m in CURRENT_MODELS:
            print("BOARD",h,m,json.dumps(v["boards"][m]["snapshot_free"],ensure_ascii=False),flush=True)
    for row in a["oos"]:
        print("OOS",json.dumps(row,ensure_ascii=False,default=str),flush=True)


if __name__=="__main__":
    main()
