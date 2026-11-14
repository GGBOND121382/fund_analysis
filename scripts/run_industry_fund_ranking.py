#!/usr/bin/env python3
"""One-command NAV refresh, fund discovery rankings, and forward validation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fundkit.defensive_screen import Shock, adjusted_asof, market_category, money_index, share_identity
from fundkit.industry_rank import (HORIZONS, RANK_COLS, NavPath, assemble, compare_forward,
                                   effective_group, fold_dates, ranking, stress_metrics)
from scripts import fetch_all_fund_rank_data as fetch

WATCH = ("025368","019851")
STRESS_COLS = ["code","name","group","event","label","baseline","stress_end","source",
               "stress_return","stress_drawdown","peak_date","trough_date","recovery_date",
               "recovery_calendar_days","recovery_status","last_observed"]


def read(path):
    return pd.read_csv(path,dtype={"code":str})


def optional(path):
    return read(path) if path.is_file() else pd.DataFrame()


def load_catalog(path, codes=None):
    d = read(path).drop_duplicates("code")
    if codes is not None:
        missing = set(codes)-set(d.code)
        if missing:
            raise ValueError(f"code absent from catalog: {sorted(missing)}")
        d = d[d.code.isin(codes)]
    if d.empty or not d.code.str.fullmatch(r"\d{6}").fillna(False).all():
        raise ValueError("empty catalog or invalid six-digit code")
    rows = []
    for r in d.itertuples(index=False):
        name = str(r.name)
        group,family,currency,basis = market_category(name,r.type)
        group,periodic = effective_group(name,group,r.type)
        product,share = share_identity(name)
        rows.append(dict(code=r.code,name=name,fund_type=r.type,group=group,product=product,
                         share=share,scenario_family=family,currency=currency,
                         pricing_basis=basis,is_long_periodic=periodic))
    return pd.DataFrame(rows)


def load_paths(catalog, data, end):
    div = optional(data/"dividends.csv")
    dividends = {c:list(zip(s.ex_date,s["div"])) for c,s in div.groupby("code")} if len(div) else {}
    sp = data/"splits.json"
    splits = json.loads(sp.read_text(encoding="utf-8-sig")) if sp.is_file() else {}
    paths,audit = {},[]
    for row in catalog.itertuples(index=False):
        p = data/fetch.BUCKET[row.pricing_basis]/f"{row.code}.csv"
        adjusted = row.pricing_basis != "fund_nav" or p.is_file()
        if not adjusted:
            p = data/"nav"/p.name
        status = dict(code=row.code,group=row.group,pricing_basis=row.pricing_basis,
                      source_bucket=p.parent.name,valid=False)
        if not p.is_file():
            audit.append(dict(status,issue="missing_cache"))
            continue
        try:
            raw = read(p)
            money = row.pricing_basis=="income_per_10000"
            nav = (money_index(raw,end) if money else adjusted_asof(
                raw,end,() if adjusted else dividends.get(row.code,()),
                () if adjusted else splits.get(row.code,())))
            if nav.empty:
                raise ValueError("no observations as of cutoff")
            paths[row.code] = NavPath(pd.DatetimeIndex(nav.index),nav.to_numpy(dtype=float),money)
            status.update(valid=True,nav_start=nav.index[0],nav_end=nav.index[-1],
                          exact_end=nav.index[-1]==end,observations=len(nav),issue="")
        except (ValueError,TypeError,KeyError,OSError,OverflowError) as e:
            status["issue"] = str(e)
        audit.append(status)
    return paths,pd.DataFrame(audit)


def read_events(*files):
    out = {}
    for path in files:
        for d in json.loads(path.read_text(encoding="utf-8-sig")):
            if d["id"] in out:
                raise ValueError(f"duplicate scenario {d['id']}")
            out[d["id"]] = Shock(d["id"],d["label"],pd.Timestamp(d["baseline"]),
                                 pd.Timestamp(d["end"]),d.get("source",""),tuple(d.get("families",())))
    return sorted(out.values(),key=lambda e:e.end)


def load_historical_catalogs(folder,end,folds,months):
    if folder is None:
        return None
    # Exact snapshots prevent silently selecting a later surviving universe
    # or borrowing a future category when the dated archive is incomplete.
    return {t:load_catalog(folder/f"{t.date()}.csv") for t in fold_dates(end,folds,months)}


def save(df,path):
    df.drop(columns=["_monthly"],errors="ignore").to_csv(path,index=False,encoding="utf-8-sig")


def current_rank(catalog, paths, events, end, out, top, min_peers):
    coverage,stress,by_status,watch = [],[],[],[]
    legacy_cache = {}
    for h in HORIZONS:
        f, missing = assemble(catalog,paths,end,h,events,stress_cache=legacy_cache)
        result,matched = [],[]
        for group,g in f.groupby("group",sort=True):
            z = ranking(g,min_peers)
            result.append(z)
            cohort = g[g.stress_count.eq(2)]
            if len(cohort)>=min_peers:
                matched.append(ranking(cohort,min_peers))
            coverage.append(dict(horizon=h,group=group,eligible=len(g),matched=len(cohort),
                                 ranked=int(z.research_rank.notna().sum())))
        full = pd.concat(result,ignore_index=True) if result else ranking(f,min_peers)
        same = pd.concat(matched,ignore_index=True) if matched else full.iloc[0:0]
        for label,table in (("ranking",full),("matched",same)):
            sorted_ = table.sort_values(["group","research_rank","code"],na_position="last")
            save(sorted_,out/f"{label}_{h}.csv")
        # Require agreement rather than a new weighted average of the ranks.
        consensus = same.research_rank.le(top) & same.legacy_rank.le(top)
        consensus &= same.industry_proxy_rank.isna() | same.industry_proxy_rank.le(top)
        save(same.loc[consensus].sort_values(["group","research_rank","code"]),out/f"consensus_{h}.csv")
        shortlist = full.loc[full[list(RANK_COLS)].le(top).any(axis=1) | full.pareto.eq(True)]
        save(shortlist.sort_values(["group","research_rank","code"]),out/f"shortlist_{h}.csv")
        rejected = pd.DataFrame(missing,columns=["code","group","reason"])
        save(rejected,out/f"unranked_{h}.csv")
        for code in WATCH:
            hit = full.loc[full.code.eq(code)]
            why = rejected.loc[rejected.code.eq(code),"reason"]
            if len(hit):
                row = hit.iloc[0].drop(labels="_monthly").to_dict()
                row["status"] = "已排名" if pd.notna(row["research_rank"]) else "同类样本不足"
            else:
                row = dict(code=code,status=why.iloc[0] if len(why) else "清单中无代码")
            watch.append(dict(horizon=h,**row))
        by_status.append(dict(horizon=h,eligible=len(full),ranked=int(full.research_rank.notna().sum()),
                              matched_ranked=len(same),rejected=dict(Counter(rejected.reason))))
        print(f"{h}: eligible={len(full)} matched={len(same)}",flush=True)

    for row in catalog.itertuples(index=False):
        if row.code not in paths:
            continue
        nav = paths[row.code]
        for event in events:
            if event.end > end or (event.families and row.scenario_family not in event.families):
                continue
            # Report missing windows instead of encoding missing as zero loss.
            r = stress_metrics(nav,event,end) or {"recovery_status":"冲击净值覆盖不足"}
            stress.append(dict(code=row.code,name=row.name,group=row.group,event=event.id,label=event.label,
                               baseline=event.baseline,stress_end=event.end,source=event.source,**r))
    save(pd.DataFrame(stress).reindex(columns=STRESS_COLS),out/"stress_events.csv")
    save(pd.DataFrame(coverage,columns=["horizon","group","eligible","matched","ranked"]),out/"coverage_by_group.csv")
    save(pd.DataFrame(watch),out/"watchlist.csv")
    return by_status


def rollup(summary):
    keys = ["horizon","group","cohort","method"]
    if summary.empty:
        return pd.DataFrame(columns=[*keys,"folds_total","folds_paired_complete","selected_total","observed_total","median_forward_return","median_forward_mdd"])
    rows = []
    for group,g in summary.groupby(keys,sort=True):
        q = g[g.paired_complete]
        rows.append(dict(zip(keys,group),folds_total=len(g),folds_paired_complete=len(q),
                         selected_total=int(g.selected.sum()),observed_total=int(g.observed.sum()),
                         median_forward_return=q.median_fwd_return.median(),
                         median_forward_mdd=q.median_fwd_mdd.median()))
    return pd.DataFrame(rows)


def paired_deltas(summary):
    keys = ["horizon","group","cohort","signal","evaluation_end"]
    cols = [*keys,"method","baseline","median_return_delta","median_mdd_delta"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    s = summary[summary.paired_complete]
    rows = []
    for base in ("legacy","industry_proxy","return_only","drawdown_only","preservation"):
        a = s[s.method.eq("research")].merge(s[s.method.eq(base)],on=keys,suffixes=("_new","_old"))
        rows.append(a.assign(method="research",baseline=base,
                             median_return_delta=a.median_fwd_return_new-a.median_fwd_return_old,
                             median_mdd_delta=a.median_fwd_mdd_new-a.median_fwd_mdd_old)[cols])
    return pd.concat(rows,ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asof",required=True)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data"/"all_funds")
    p.add_argument("--output",type=Path,default=ROOT/"reports"/"industry_rank")
    p.add_argument("--fetch",action="store_true",help="resume and extend public data cache before ranking")
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--codes",help="comma-separated explicitly scoped debug subset")
    p.add_argument("--shocks",type=Path,default=ROOT/"config"/"all_fund_shocks.json")
    p.add_argument("--extra-shocks",type=Path,default=ROOT/"config"/"industry_rank_extra_shocks.json")
    p.add_argument("--oos-folds",type=int,default=12)
    p.add_argument("--oos-months",type=int,default=3)
    p.add_argument("--pit-catalog-dir",type=Path,help="exact YYYY-MM-DD.csv catalog for every formation date")
    p.add_argument("--top",type=int,default=10)
    p.add_argument("--min-peers",type=int,default=5)
    a = p.parse_args(argv)
    if a.oos_folds<0 or a.oos_months<1 or a.top<1 or a.min_peers<2:
        p.error("invalid fold count, holding months, top, or peer count")
    if a.pit_catalog_dir and a.codes:
        p.error("--pit-catalog-dir and --codes cannot be combined")
    end = pd.Timestamp(a.asof).normalize()
    historical = load_historical_catalogs(a.pit_catalog_dir,end,a.oos_folds,a.oos_months)
    if a.fetch:
        years = 5+math.ceil((a.oos_folds*a.oos_months+1)/12)
        args = ["--asof",a.asof,"--data-dir",str(a.data_dir),"--history-years",str(years),"--workers",str(a.workers)]
        fetch.main(args+(["--codes",a.codes] if a.codes else []))
    events = read_events(a.shocks,a.extra_shocks)
    catalog = load_catalog(a.data_dir/"fund_list.csv",a.codes.split(",") if a.codes else None)
    path_catalog = (pd.concat([catalog,*historical.values()],ignore_index=True)
                    .drop_duplicates("code") if historical else catalog)
    paths,audit = load_paths(path_catalog,a.data_dir,end)
    a.output.mkdir(parents=True,exist_ok=True)
    save(audit,a.output/"nav_audit.csv")
    if not paths:
        p.error("no valid NAV cache; see nav_audit.csv and run with --fetch or point at populated --data-dir")
    start = time.monotonic()
    coverage = current_rank(catalog,paths,events,end,a.output,a.top,a.min_peers)
    # Without an archive, never label the supplied snapshot an unbiased OOS test.
    replay = catalog[catalog.code.isin(paths)]
    if historical is not None:
        historical = {t:d[d.code.isin(paths)] for t,d in historical.items()}
    selected,summary = compare_forward(replay,paths,events,end,a.oos_folds,a.oos_months,a.top,a.min_peers,historical)
    if selected.empty:
        selected = selected.reindex(columns=["signal","evaluation_end","horizon","group","cohort","method","code","name","observed","fwd_return","fwd_mdd","entry","exit"])
    if summary.empty:
        summary = summary.reindex(columns=["signal","evaluation_end","horizon","group","cohort","method","candidate_count","selected","observed","coverage","median_fwd_return","median_fwd_mdd","worst_selected_mdd","paired_complete"])
    for name,table in (("oos_selections",selected),("oos_folds",summary),("oos_summary",rollup(summary)),("oos_paired_deltas",paired_deltas(summary))):
        save(table,a.output/f"{name}.csv")
    cov = {"asof":a.asof,"catalog_share_codes":len(catalog),"valid_nav_paths":len(paths),
           "nav_audit_share_codes":len(audit),
           "valid_current_catalog_paths":int(catalog.code.isin(paths).sum()),
           "exact_asof_paths":int(audit.get("exact_end",pd.Series(dtype=bool)).eq(True).sum()),
           "explicit_subset":bool(a.codes),"horizons":coverage,"oos_requested_folds":a.oos_folds,
           "oos_holding_months":a.oos_months,"oos_fold_rows":len(summary),
           "oos_paired_complete_rows":int(summary.paired_complete.eq(True).sum()),
           "oos_mode":("dated catalog replay; NAV availability and execution assumptions remain"
                       if historical is not None else
                       "historical replay on supplied snapshot; survivor/current-classification bias possible"),
           "elapsed_seconds":round(time.monotonic()-start,2)}
    (a.output/"audit.json").write_text(json.dumps(cov,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(cov,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
