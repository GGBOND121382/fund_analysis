"""Read-only, nonoverlapping PIT fund-discovery research OOS diagnostics.

Only explicitly supplied historical *formation-time* selections are accepted.
No historical selections/archives => EVIDENCE_INSUFFICIENT, not fake OOS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from fundkit.discovery_oos import (validate_pit_selections,realized_buy_hold,
                                    nonoverlapping_pairs)


def read_nav(path):
    d=pd.read_csv(path)
    if not {"date","nav"}.issubset(d.columns):
        raise ValueError(f"{path}: expected date,nav")
    d.date=pd.to_datetime(d.date,errors="coerce")
    d.nav=pd.to_numeric(d.nav,errors="coerce")
    if d.date.isna().any() or not d.date.is_monotonic_increasing or d.date.duplicated().any():
        raise ValueError(f"{path}: invalid historical dates")
    if not np.isfinite(d.nav).all() or (d.nav<=0).any():
        raise ValueError(f"{path}: invalid gross NAV")
    return d.set_index("date")["nav"]


def one_forward(fund,cash,formation,evaluation,weight,fee):
    a=fund.loc[(fund.index>formation)&(fund.index<=evaluation)]
    b=cash.loc[(cash.index>formation)&(cash.index<=evaluation)]
    common=a.index.intersection(b.index)
    if len(common)<3 or (common[0]-formation).days>10 or (evaluation-common[-1]).days>10:
        raise ValueError("incomplete observed forward holding window")
    result=realized_buy_hold(a.loc[common],b.loc[common],weight,fee)
    result.update(realized_start=str(common[0].date()),realized_end=str(common[-1].date()),
                  observed_days=len(common))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections",type=Path,required=True,
                   help="PIT selections containing historical evidence timestamps")
    p.add_argument("--fund-nav-dir",type=Path,required=True,
                   help="historical price paths, each code.csv with date,nav")
    p.add_argument("--cash-nav",type=Path,required=True,
                   help="historically observed investable cash alternative date,nav")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--primary",default="opportunity_experimental")
    p.add_argument("--baselines",default="legacy_pareto,return_only,drawdown_only")
    p.add_argument("--min-test-folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.min_test_folds<2:
        p.error("at least two nonoverlapping test folds required")
    if not a.selections.is_file() or not a.cash_nav.is_file():
        outcome={"status":"EVIDENCE_INSUFFICIENT","reason":"historical selections or observed cash NAV missing",
                 "primary":a.primary,"oos_certified":False}
        (a.output/"oos_status.json").write_text(json.dumps(outcome,indent=2)+"\n")
        return outcome
    selections=pd.read_csv(a.selections,dtype={"code":str})
    checked=validate_pit_selections(selections)
    cash=read_nav(a.cash_nav)
    cache={}
    rows=[]
    for r in checked.itertuples(index=False):
        record=r._asdict()
        if r.pit_status != "PIT_OK":
            record.update(oos_status="BLOCKED_PIT",cash_excess=np.nan,max_drawdown=np.nan,
                          net_return=np.nan)
            rows.append(record);continue
        path=a.fund_nav_dir/f"{r.code}.csv"
        if not path.is_file():
            record.update(oos_status="NAV_MISSING",cash_excess=np.nan,max_drawdown=np.nan,
                          net_return=np.nan)
            rows.append(record);continue
        try:
            if r.code not in cache:
                cache[r.code]=read_nav(path)
            out=one_forward(cache[r.code],cash,r.formation_date,r.evaluation_end,
                            float(r.weight),float(r.fee_rate))
            record.update(out,oos_status="PIT_RESEARCH_NAV_ONLY")
        except (ValueError,KeyError):
            record.update(oos_status="INCOMPLETE_FORWARD_WINDOW",
                          cash_excess=np.nan,max_drawdown=np.nan,net_return=np.nan)
        rows.append(record)
    report=pd.DataFrame(rows)
    report.to_csv(a.output/"oos_results.csv",index=False,encoding="utf-8-sig")
    valid=report.loc[report.oos_status.eq("PIT_RESEARCH_NAV_ONLY")]
    comparisons=[]
    for baseline in filter(None,(s.strip() for s in a.baselines.split(","))):
        delta=nonoverlapping_pairs(valid,a.primary,baseline,a.min_test_folds,a.seed)
        if delta.empty:
            comparisons.append(dict(baseline=baseline,decision="EVIDENCE_INSUFFICIENT",folds=0))
        else:
            comparisons.extend([dict(baseline=baseline,**row) for row in
                                delta.to_dict("records")])
    pd.DataFrame(comparisons).to_csv(a.output/"oos_paired_deltas.csv",
                                    index=False,encoding="utf-8-sig")
    # These data provide NAV-only research evidence. Historical fund universe,
    # contracts, historical fills and a locked nested train/val selection are
    # not authenticated here. Never certify the production investment score.
    outcome=dict(status="PIT_RESEARCH_DIAGNOSTIC_ONLY" if len(valid) else "EVIDENCE_INSUFFICIENT",
                 oos_certified=False,primary=a.primary,
                 rows_total=len(report),rows_pit_valid=int(checked.pit_status.eq("PIT_OK").sum()),
                 forward_evaluated=len(valid),missing_pit=int(checked.pit_status.ne("PIT_OK").sum()),
                 historical_universe_verified=False,nested_selection_verified=False,
                 execution_verified=False,
                 note="not a prospective executable OOS or evidence of optimized parameter generalization")
    (a.output/"oos_status.json").write_text(json.dumps(outcome,indent=2,ensure_ascii=False)+"\n",
                                            encoding="utf-8")
    print(json.dumps(outcome,ensure_ascii=False))
    return outcome


if __name__=="__main__":
    main()
