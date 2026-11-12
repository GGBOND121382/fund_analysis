#!/usr/bin/env python3
"""Find defensively performing bond funds since the 2026-10-30 selloff.

Uses the existing full-market NAV cache and dividend/split adjustments.
Requires identical EXACT comparison dates; never ranks stale NAVs as winners.
This is retrospective ranking, not a purchase recommendation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from fundkit import metrics, storage, fetcher


def screen_bond_resilience(*, baseline="2026-10-29",end="2026-11-11",
                           min_days=6,need_tradeable=False):
    baseline=pd.Timestamp(baseline);end=pd.Timestamp(end)
    if end<=baseline:
        raise ValueError("end must be later than the pre-shock baseline")
    funds=storage.load_fund_list()
    bond=funds[funds["type"].astype(str).str.contains("债券型",na=False)]
    dividends=storage.load_dividends()
    splits=storage.load_splits()
    tradable=None
    try:
        status=storage.load_trade_status()
        tradable=fetcher.tradable_codes(status)
    except FileNotFoundError:
        if need_tradeable:
            raise ValueError("tradeable status missing: run python main.py list")
    rows=[];audit={"no_cache":0,"missing_baseline":0,"missing_end":0,
                   "short_window":0,"invalid_nav":0}
    for _,f in bond.iterrows():
        code=str(f["code"]).zfill(6)
        if need_tradeable and code not in tradable:
            continue
        if not storage.has_nav(code):
            audit["no_cache"]+=1;continue
        try:
            history=storage.load_nav(code)
            # Adjust *before* date slicing to retain correct ex-dividend factors.
            hist=metrics.adjust_nav(history,dividends.get(code,()),splits.get(code,()))
            hist=hist.sort_values("date")
            dates=set(hist["date"].tolist())
            if baseline not in dates:
                audit["missing_baseline"]+=1;continue
            if end not in dates:
                audit["missing_end"]+=1;continue
            window=metrics.slice_by_date(hist,baseline,end)
            if len(window)<min_days:
                audit["short_window"]+=1;continue
            m=metrics.compute_metrics(window,min_days=min_days)
            if m is None:
                audit["short_window"]+=1;continue
            prior=hist[hist["date"]<=end].tail(253)
            full=metrics.compute_metrics(prior,min_days=190)
            # Single-day 2026-10-30 shock includes that day's total-return NAV.
            shock_day=baseline+pd.offsets.BDay(1)
            shock=hist[hist["date"]==shock_day]
            base=float(hist.loc[hist["date"]==baseline,"nav"].iloc[0])
            first_loss=(float(shock["nav"].iloc[0])/base-1.) if len(shock) else None
            rows.append({
                "code":code,"name":str(f["name"]),"fund_type":str(f["type"]),
                "return_since_shock":m["total_return"],
                "max_drawdown_since_shock":m["max_drawdown"],
                "shock_day_return":first_loss,
                "one_year_total_return":full["total_return"] if full else None,
                "one_year_mdd":full["max_drawdown"] if full else None,
                "one_year_observations":full["days"] if full else 0,
                "days_since_shock":m["days"],
                "tradable":None if tradable is None else code in tradable,
                "baseline_nav_date":str(baseline.date()),
                "end_nav_date":str(end.date()),
            })
        except (KeyError,ValueError,IndexError,TypeError,ZeroDivisionError):
            audit["invalid_nav"]+=1
    result=pd.DataFrame(rows)
    if not result.empty:
        result=result.sort_values(["return_since_shock","max_drawdown_since_shock","code"],
                                  ascending=[False,True,True]).reset_index(drop=True)
        result.insert(0,"rank",result.index+1)
    return result,{
        "baseline_date":str(baseline.date()),"end_date":str(end.date()),
        "screened_bond_funds":len(bond),"compared_funds":len(result),
        "rejection_reasons":audit,
        "dividend_data_available":bool(dividends),
        "same_date_comparison":True,
        "limitations":[
            "Very short post-shock window: total return/MDD only, no annualized forecasting",
            "NAV data and dividends are cached locally; update before screening",
            "Ranking does not establish duration, credit-quality or future resilience",
            "Tradability and fee/lock restrictions must be audited before purchase",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description="Bond fund resilience ranking, fixed NAV dates")
    p.add_argument("--baseline",default="2026-10-29")
    p.add_argument("--end",default="2026-11-11")
    p.add_argument("--min-days",type=int,default=6)
    p.add_argument("--top",type=int,default=50)
    p.add_argument("--tradeable-only",action="store_true")
    p.add_argument("--output",default="reports/bond_resilience_20261030.csv")
    a=p.parse_args(argv)
    data,audit=screen_bond_resilience(
        baseline=a.baseline,end=a.end,min_days=a.min_days,
        need_tradeable=a.tradeable_only)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(out,index=False,encoding="utf-8-sig")
    out.with_suffix(".audit.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Compared",audit["compared_funds"],"of",audit["screened_bond_funds"],
          "bond funds on EXACT NAV dates")
    print(data.head(a.top).to_string(index=False) if not data.empty else "NO_SAME_DATE_DATA")
    print("CSV:",out,"Audit:",out.with_suffix(".audit.json"))


if __name__=="__main__":
    main()
