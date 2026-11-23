#!/usr/bin/env python3
"""Independent verification and quality analysis for the M0 comprehensive boards."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fundkit.industry_rank import NavPath, window_dates
from scripts.rank_all_funds_comprehensive import SPECIAL

W = dict(annualized_return=0.40,max_drawdown=0.35,shock_resilience=0.15,monthly_downside=0.10)
OLD_DIAG = ROOT/"reports"/"all_funds"/"fund_discovery_v3_diagnostic_20261120"
OLD_INDUSTRY = ROOT/"reports"/"all_funds"/"industry_rank_20261120"
V4_BUDGET = ROOT/"reports"/"all_funds"/"fund_discovery_v3_cash_only_20261120"
HMONTHS = {"5y":60,"3y":36,"1y":12,"6m":6}
SHOCK = ("2026-10-29","2026-11-13")


def load_navpath(data_dir,code):
    p = data_dir/"total_return"/f"{code}.csv"
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    return NavPath(pd.DatetimeIndex(pd.to_datetime(d["date"])),d["nav"].to_numpy(dtype=float))


def independent_metrics(nav,end,h):
    """Arithmetic rewritten from scratch; shared lib supplies dates only."""
    begin,period,cm = window_dates(end,HMONTHS[h])
    i,j = nav.right(begin),nav.right(end)
    if i < 0 or j < 0 or j <= i:
        return None
    v = nav.value[i:j+1]
    R = float((v[-1]/v[0])**(365.2425/(nav.days[j]-nav.days[i]))-1.)
    peak = np.maximum.accumulate(v)
    D = float(np.max(1.-v/peak))
    monthly = nav.monthly(end,period,cm)
    if monthly is None:
        return None
    L = float(np.sqrt(np.mean(np.minimum(monthly,0.)**2)))
    a,b = nav.left(pd.Timestamp(SHOCK[0])),nav.right(pd.Timestamp(SHOCK[1]))
    if a > b or nav.dates[a] != pd.Timestamp(SHOCK[0]) or nav.dates[b] != pd.Timestamp(SHOCK[1]):
        return None
    s = nav.value[a:b+1]
    return dict(R=R,D=D,L=L,Aret=float(s[-1]/s[0]-1.),Add=float(np.max(1.-s/np.maximum.accumulate(s))))


def percentile(frame):
    n = len(frame)
    one = {c:(100.*(frame[c].rank(method="average")-1.)/(n-1.) if n > 1 else 50.)
           for c in ("R","D","L","Aret","Add")}
    return (one["R"],100.-one["D"],100.-one["L"],0.5*one["Aret"]+0.5*(100.-one["Add"]))


def main():
    out = ROOT/"reports"/"all_funds"/"comprehensive_20261120"
    data = ROOT/"data"/"all_funds"
    end = pd.Timestamp("2026-11-20")
    verify,quality = {},{}
    rng = np.random.default_rng(42)
    for h in ("5y","3y","1y","6m"):
        df = pd.read_csv(out/f"all_scored_{h}.csv",dtype={"code":str},low_memory=False)
        scored = df[df.status.eq("SCORED")].copy()
        pool = sorted(set(rng.choice(scored.code.to_numpy(),size=20,replace=False)) | set(SPECIAL))
        SR,SD,SL,SA = percentile(scored)
        full_score = 0.40*SR+0.35*SD+0.15*SA+0.10*SL
        checks,maxdiff = 0,0.
        for code in pool:
            nav = load_navpath(data,code)
            row = df[df.code.eq(code)]
            if nav is None or not len(row):
                continue
            row = row.iloc[0]
            m = independent_metrics(nav,end,h)
            if m is None or row.status != "SCORED":
                continue
            for k in ("R","D","L","Aret","Add"):
                maxdiff = max(maxdiff,abs(m[k]-row[k]))
                assert abs(m[k]-row[k]) < 1e-9,(h,code,k,m[k],row[k])
            got = float(full_score.loc[scored.code.eq(code)].iloc[0])
            assert abs(got-row.SCORE_M0) < 1e-9,(h,code,got,row.SCORE_M0)
            checks += 1
        verify[h] = {"recomputed_codes":checks,"max_abs_metric_diff":maxdiff}
        top = scored.nsmallest(50,"raw_rank")
        sens = {}
        for key in W:
            w = {x:W[x]+(0.05 if x == key else 0.) for x in W}
            w = {x:w[x]/sum(w.values()) for x in W}
            s = (w["annualized_return"]*SR+w["max_drawdown"]*SD
                 +w["shock_resilience"]*SA+w["monthly_downside"]*SL)
            tmp = scored[["code"]].assign(_s=s.to_numpy())
            vtop = set(tmp.sort_values(["_s","code"],ascending=[False,True]).head(10).code)
            sens[f"{key}+5pp"] = {"top10_overlap":len(set(top.head(10).code) & vtop),
                                  "spearman":round(float(pd.Series(s).corr(scored.SCORE_M0,method="spearman")),4)}
        overlap = {}
        diag = OLD_DIAG/f"top10_historical_return_DIAGNOSTIC_{h}.csv"
        if diag.is_file():
            overlap["vs_pure_return_top10"] = len(set(pd.read_csv(diag,dtype={"code":str}).code) & set(top.head(10).code))
        ind = OLD_INDUSTRY/f"top10_{h}.csv"
        if ind.is_file():
            overlap["vs_industry_research_top10"] = len(set(pd.read_csv(ind,dtype={"code":str}).code) & set(top.head(10).code))
        v4 = V4_BUDGET/f"top10_experimental_{h}.csv"
        overlap["v4_budget_board_rows"] = int(len(pd.read_csv(v4))) if v4.is_file() else None
        high = scored[scored.D > .30].nsmallest(5,"raw_rank")
        quality[h] = {"scored_shares":len(scored),"scored_fund_level":int(scored.product_key.nunique()),
                      "top10_R_median":float(top.head(10).R.median()),"top10_D_median":float(top.head(10).D.median()),
                      "top10_L_median":float(top.head(10).L.median()),"top10_Aret_median":float(top.head(10).Aret.median()),
                      "top20_D_max":float(top.head(20).D.max()),"top50_D_max":float(top.D.max()),
                      "high_mdd_cases":[{"code":r.code,"name":r.name,"R":round(r.R,4),"D":round(r.D,4),
                                         "rank":int(r.raw_rank)} for r in high.itertuples()],
                      "zero_mdd_leaked_into_scored":int((scored.D <= .0002).sum()),
                      "overlap":overlap,"weight_sensitivity":sens}
    (out/"verification.json").write_text(json.dumps({"recompute":verify,"quality":quality},
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"recompute":verify},ensure_ascii=False))
    for h,q in quality.items():
        print(h,json.dumps({k:q[k] for k in ("scored_shares","scored_fund_level","top10_R_median","top10_D_median","zero_mdd_leaked_into_scored","overlap")},ensure_ascii=False))
        print("  sens:",json.dumps(q["weight_sensitivity"]))


if __name__ == "__main__":
    main()
