#!/usr/bin/env python3
"""One-command NAV refresh, fund discovery rankings, and forward validation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fundkit.defensive_screen import Shock, adjusted_asof, market_category, money_index, share_identity
from fundkit.industry_rank import (HORIZONS, RANK_COLS, NavPath, assemble, compare_forward,
                                   add_main_ranking, add_global_ranking, effective_group, periodic_classification, fold_dates, main_stress_features,
                                   ranking, stress_metrics)
from scripts import fetch_all_fund_rank_data as fetch

WATCH = ("025368","019851","021559","004400","710301")
LP_REASON = "长封闭定开（≥24月）默认排除排名"
STRESS_COLS = ["code","name","group","event","label","baseline","stress_end","source",
               "stress_return","stress_drawdown","peak_date","trough_date","recovery_date",
               "recovery_calendar_days","recovery_status","last_observed"]


def read(path):
    return pd.read_csv(path,dtype={"code":str})


def optional(path):
    return read(path) if path.is_file() else pd.DataFrame()


def load_periodic_overrides(path=ROOT/"config"/"fund_periodic_overrides.csv"):
    if not path.is_file():
        return {}
    d = pd.read_csv(path,dtype={"code":str},keep_default_na=False)
    required = {"code","cycle_months","source_url"}
    if not required.issubset(d.columns):
        raise ValueError(f"periodic override missing columns: {sorted(required-set(d.columns))}")
    if d.code.duplicated().any() or not d.code.str.fullmatch(r"\d{6}").all():
        raise ValueError("periodic override codes must be unique six-digit strings")
    if (pd.to_numeric(d.cycle_months,errors="coerce").isna()
            | (pd.to_numeric(d.cycle_months,errors="coerce") <= 0)).any():
        raise ValueError("periodic override cycle_months must be positive")
    if d.source_url.str.strip().eq("").any():
        raise ValueError("periodic override requires a source_url for every code")
    return {r.code:{"cycle_months":int(r.cycle_months),"source_url":r.source_url,
                      "source_date":getattr(r,"source_date",""),
                      "note":getattr(r,"note","")}
            for r in d.itertuples(index=False)}


def load_catalog(path, codes=None, periodic_overrides=None):
    periodic_overrides = load_periodic_overrides() if periodic_overrides is None else periodic_overrides
    d = pd.read_csv(path, dtype={"code": str, "product_id": str},
                    keep_default_na=False).drop_duplicates("code")
    if codes is not None:
        missing = set(codes)-set(d.code)
        if missing:
            raise ValueError(f"code absent from catalog: {sorted(missing)}")
    if d.empty or not d.code.str.fullmatch(r"\d{6}").fillna(False).all():
        raise ValueError("empty catalog or invalid six-digit code")
    rows = []
    for r in d.itertuples(index=False):
        name = str(r.name)
        group,family,currency,basis = market_category(name,r.type)
        group,is_long,is_unverified,periodic_status,periodic_months,periodic_source = periodic_classification(
            name,group,r.type,r.code,periodic_overrides)
        product,share = share_identity(name)
        reason = ("长封闭定开（≥24月）默认排除排名" if is_long else
                  "定开运作周期未核实，暂不参与排名" if is_unverified else "")
        rows.append(dict(code=r.code,name=name,fund_type=r.type,group=group,product=product,
                         product_id=getattr(r,"product_id","").strip(),
                         share=share,scenario_family=family,currency=currency,
                         pricing_basis=basis,is_long_periodic=is_long,
                         is_periodic_unverified=is_unverified,periodic_status=periodic_status,
                         periodic_months=periodic_months,periodic_source=periodic_source,
                         periodic_exclusion_reason=reason))
    catalog = pd.DataFrame(rows)
    # Optional product IDs must come from the input catalog. Use a name to
    # inherit an ID only when that name points to a single ID in this cohort.
    name_cols = ["group","currency","pricing_basis","product"]
    known = catalog.loc[catalog.product_id.ne(""), name_cols+["product_id"]].drop_duplicates()
    unique = known.loc[~known.duplicated(name_cols,keep=False)].rename(columns={"product_id":"_name_id"})
    catalog = catalog.merge(unique,on=name_cols,how="left",validate="many_to_one")
    identity = catalog.product_id.where(catalog.product_id.ne(""),catalog.pop("_name_id"))
    catalog["product_key"] = ("id:"+identity).fillna("name:"+catalog["product"])
    return catalog if codes is None else catalog.loc[catalog.code.isin(codes)]


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


def legacy_top(full, top, stress_cache):
    """Export the existing legacy rank and the actual two windows it used."""
    result = full.loc[full.legacy_rank.le(top)].sort_values(["group","legacy_rank","code"]).copy()
    records = []
    for row in result.itertuples(index=False):
        first, second = row.stress_ids.split(",")
        dd1, ret1 = stress_cache[row.code, first]
        dd2, ret2 = stress_cache[row.code, second]
        records.append((first, ret1, dd1, second, ret2, dd2))
    columns = [f"shock_{i}_{metric}" for i in (1,2) for metric in ("id","return","mdd")]
    return result.join(pd.DataFrame(records,index=result.index,columns=columns))


def stress_table(catalog, paths, events, end):
    stress = []
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
    return pd.DataFrame(stress).reindex(columns=STRESS_COLS)


def export_rankings(full, out, h, top):
    full = full.sort_values(["group", "main_rank", "code"], na_position="last")
    save(full,out/f"ranking_{h}.csv")
    for label, key in (("", "main"), ("research_", "research"), ("industry_proxy_", "industry_proxy")):
        col = f"{key}_rank"
        subset = full.loc[full[col].le(top)].sort_values(["group",col,"code"])
        save(subset,out/f"top10_{label}{h}.csv")
    # A stress-only view, independent of period return, full-period drawdown or legacy score.
    st = full.loc[full.legacy_rank.notna()].sort_values(
        ["group","worst_stress_drawdown","worst_stress_return","code"],
        ascending=[True,True,False,True]).copy()
    st["stress_rank"] = st.groupby("group").cumcount() + 1
    save(st.loc[st.stress_rank.le(top)],out/f"top10_stress_{h}.csv")
    no_main = full.loc[full.main_rank.isna(),
                       ["code","group","main_status","main_stress_count","main_stress_required",
                        "main_peer_count","main_missing_stress_ids"]]
    save(no_main,out/f"unranked_main_{h}.csv")

    global_ranked = full.loc[full.global_rank.notna()].sort_values(["global_rank","code"])
    save(global_ranked,out/f"ranking_global_{h}.csv")
    save(global_ranked.head(top),out/f"top10_global_{h}.csv")
    no_global = full.loc[full.global_rank.isna(),
                         ["code","name","fund_type","group","global_status",
                          "global_stress_ids","global_stress_count","global_stress_required",
                          "global_peer_count","global_missing_stress_ids"]]
    save(no_global,out/f"unranked_global_{h}.csv")
    return full


def coverage_row(h, full):
    return dict(horizon=h, eligible=len(full), ranked=int(full.main_rank.notna().sum()),
                research_ranked=int(full.research_rank.notna().sum()),
                legacy_ranked=int(full.legacy_rank.notna().sum()),
                global_ranked=int(full.global_rank.notna().sum()),
                global_top10=int(full.global_rank.le(10).sum()),
                global_peer_count=int(full.global_peer_count.max()) if len(full) else 0,
                global_status=dict(Counter(full.global_status)),
                main_status=dict(Counter(full.main_status)))


def main_group_coverage(h, full, top):
    for group, g in full.groupby("group", sort=True):
        yield dict(horizon=h,group=group,eligible=len(g),matched=int(g.stress_count.eq(2).sum()),
                   ranked=int(g.main_rank.notna().sum()),research_ranked=int(g.research_rank.notna().sum()),
                   legacy_ranked=int(g.legacy_rank.notna().sum()),
                   main_top=int(g.main_rank.le(top).sum()),research_top=int(g.research_rank.le(top).sum()),
                   global_ranked=int(g.global_rank.notna().sum()),
                   global_top10=int(g.global_rank.le(top).sum()),
                   main_research_overlap=int((g.main_rank.le(top) & g.research_rank.le(top)).sum()))


def current_rank(catalog, paths, events, main_events, end, out, top, min_peers, excluded=None):
    coverage,by_status,watch = [],[],[]
    excluded = excluded if excluded is not None else catalog.iloc[0:0]
    stress = stress_table(catalog,paths,events,end)
    main_stress = main_stress_features(stress,main_events,end)
    legacy_cache = {}
    for h in HORIZONS:
        f, missing = assemble(catalog,paths,end,h,events,stress_cache=legacy_cache)
        result,matched = [],[]
        for _,g in f.groupby("group",sort=True):
            result.append(ranking(g,min_peers))
            cohort = g[g.stress_count.eq(2)]
            if len(cohort)>=min_peers:
                matched.append(ranking(cohort,min_peers))
        full = pd.concat(result,ignore_index=True) if result else ranking(f,min_peers)
        same = pd.concat(matched,ignore_index=True) if matched else full.iloc[0:0]
        full = add_main_ranking(full,main_stress,main_events,end,min_peers)
        full = add_global_ranking(full,main_stress,main_events,end,min_peers)
        coverage.extend(main_group_coverage(h,full,top))
        full = export_rankings(full,out,h,top)
        save(same.sort_values(["group","research_rank","code"]),out/f"matched_{h}.csv")
        save(legacy_top(full,top,legacy_cache),out/f"top10_legacy_{h}.csv")
        # Research/legacy/industry agreement on their original shared cohort.
        consensus = same.research_rank.le(top) & same.legacy_rank.le(top)
        consensus &= same.industry_proxy_rank.isna() | same.industry_proxy_rank.le(top)
        save(same.loc[consensus].sort_values(["group","research_rank","code"]),out/f"consensus_{h}.csv")
        shortlist = full.loc[full[["main_rank", "global_rank", *RANK_COLS]].le(top).any(axis=1)
                             | full.main_pareto.eq(True) | full.global_pareto.eq(True)
                             | full.pareto.eq(True)]
        save(shortlist,out/f"shortlist_{h}.csv")
        rejected = pd.DataFrame(missing,columns=["code","group","reason"])
        if len(excluded):
            rejected = pd.concat([rejected,pd.DataFrame(
                {"code":excluded.code,"group":excluded.group,
                 "reason":excluded.periodic_exclusion_reason})],
                ignore_index=True)
        save(rejected,out/f"unranked_{h}.csv")
        for code in WATCH:
            hit = full.loc[full.code.eq(code)]
            why = rejected.loc[rejected.code.eq(code),"reason"]
            if len(hit):
                row = hit.iloc[0].drop(labels="_monthly").to_dict()
                row["status"] = row["main_status"]
            else:
                row = dict(code=code,status=why.iloc[0] if len(why) else "清单中无代码")
            watch.append(dict(horizon=h,**row))
        by_status.append(dict(coverage_row(h,full),matched_ranked=len(same),
                              rejected=dict(Counter(rejected.reason))))
        print(f"{h}: eligible={len(full)} main_ranked={full.main_rank.notna().sum()} matched={len(same)}",flush=True)

    save(stress,out/"stress_events.csv")
    save(pd.DataFrame(coverage),out/"coverage_by_group.csv")
    save(pd.DataFrame(watch),out/"watchlist.csv")
    return by_status


def method_audit(events, end, top, min_peers):
    return dict(main_method="historical three-axis Pareto → balance → weakest → code; fixed horizon",
                main_top=top, main_min_peers=min_peers,
                main_events=[dict(id=e.id,baseline=e.baseline.date().isoformat(),end=e.end.date().isoformat(),
                                  families=list(e.families)) for e in events if e.end <= end],
                global_method="same three-axis Pareto → balance → weakest; percentiles computed once across all asset classes",
                global_stress_policy="only completed family-neutral market-wide events enter the cross-asset shock axis; family-specific events remain in category diagnostics",
                global_stress_events=[dict(id=e.id,baseline=e.baseline.date().isoformat(),end=e.end.date().isoformat())
                                      for e in events if e.end <= end and not e.families],
                global_min_common_shocks=2)


def replay_snapshot(source, out, end, events, top, min_peers):
    """Recompute current main ranks from an immutable, same-date feature/stress export."""
    if out.resolve() == source.resolve():
        raise ValueError("replay output must differ from source directory")
    files = ["audit.json","stress_events.csv",
             *(f"{prefix}_{h}.csv" for h in HORIZONS for prefix in ("ranking","unranked"))]
    manifest = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
    src = json.loads((source/"audit.json").read_text(encoding="utf-8-sig"))
    if pd.Timestamp(src["asof"]) != end:
        raise ValueError("snapshot cutoff differs from --asof")
    stress = pd.read_csv(source/"stress_events.csv",dtype={"code":str},float_precision="round_trip")
    metrics = main_stress_features(stress,events,end)
    periodic_audits = []
    stress_index = stress.set_index(["code","event"],verify_integrity=True)
    all_rows = {}
    for h in HORIZONS:
        frame = pd.read_csv(source/f"ranking_{h}.csv",dtype={"code":str,"product_id":str},
                            float_precision="round_trip")
        rejected = read(source/f"unranked_{h}.csv")
        overrides = load_periodic_overrides()
        periodic_rows, keep = [], []
        for row in frame.itertuples(index=False):
            _,is_long,is_unverified,status,months,evidence = periodic_classification(
                row.name,row.group,row.fund_type,row.code,overrides)
            if is_long or is_unverified:
                periodic_rows.append(dict(code=row.code,name=row.name,fund_type=row.fund_type,
                    group=row.group,periodic_status=status,periodic_months=months,
                    periodic_source=evidence,
                    audit_status=("已核验长周期定开，排除排名" if is_long else "定开运作周期未核实，暂不参与排名")))
            else:
                keep.append(row.Index if hasattr(row,"Index") else row.code)
        if periodic_rows:
            excluded_codes = {x["code"] for x in periodic_rows}
            periodic_mask = frame.code.isin(excluded_codes)
            frame = frame.loc[~periodic_mask].copy()
            rejected = pd.concat([rejected,pd.DataFrame(
                {"code":[x["code"] for x in periodic_rows],
                 "group":[x["group"] for x in periodic_rows],
                 "reason":[x["audit_status"] for x in periodic_rows]})],ignore_index=True)
        # Product-level overrides also cover non-representative shares already
        # present in the snapshot's duplicate-share rejection table.
        override_codes = set(overrides)
        for idx,row in rejected.loc[rejected.code.isin(override_codes)].iterrows():
            override = overrides[row.code]
            reason = "已核验长周期定开，排除排名"
            rejected.loc[idx,"reason"] = reason
            if row.code not in {x["code"] for x in periodic_rows}:
                periodic_rows.append(dict(code=row.code,name="",fund_type="",group=row.group,
                    periodic_status="合同核验：{}个月".format(override["cycle_months"]),
                    periodic_months=override["cycle_months"],periodic_source=override["source_url"],
                    audit_status=reason+"（同基金其他份额）"))
                periodic_audits.append(dict(horizon=h,**periodic_rows[-1]))
        if (not pd.to_datetime(frame.end).eq(end).all()
                or not np.isfinite(frame[["return_for_rank","max_drawdown"]]).all().all()
                or frame.duplicated(["group","product_key","currency","pricing_basis"]).any()
                or frame.code.duplicated().any() or rejected.code.duplicated().any()
                or set(frame.code).intersection(rejected.code)
                or len(frame)+len(rejected) != src["catalog_share_codes"]):
            raise ValueError(f"inconsistent cutoff, product identity, or catalog coverage: {h}")
        if periodic_rows:
            periodic_audits.extend([dict(horizon=h,**x) for x in periodic_rows])
        for family,g in frame.groupby("scenario_family"):
            ids = [e.id for e in events if e.end <= end and (not e.families or family in e.families)]
            i = pd.MultiIndex.from_product([g.code, ids], names=["code","event"])
            if not i.isin(metrics.index).all():
                raise ValueError(f"snapshot missing expected event rows: {h}, {family}")
        ranked = add_main_ranking(frame,metrics,events,end,min_peers)
        ranked = add_global_ranking(ranked,metrics,events,end,min_peers)
        all_rows[h] = (ranked,rejected)
    out.mkdir(parents=True,exist_ok=True)
    coverage,horizons,watch = [],[],[]
    for h,(full,rejected) in all_rows.items():
        coverage.extend(main_group_coverage(h,full,top))
        full = export_rankings(full,out,h,top)
        horizons.append(dict(coverage_row(h,full),
                              global_ranked=int(full.global_rank.notna().sum()),
                              global_top=int(full.global_rank.le(top).sum()),
                              global_unranked=int(full.global_rank.isna().sum()),
                              rejected=dict(Counter(rejected.reason))))
        save(rejected,out/f"unranked_{h}.csv")
        result = full.loc[full.legacy_rank.le(top)]
        cache = {(row.code, event): (r.stress_drawdown, r.stress_return)
                 for row in result.itertuples(index=False) for event in row.stress_ids.split(",")
                 for r in (stress_index.loc[(row.code,event)],)}
        save(legacy_top(full,top,cache),out/f"top10_legacy_{h}.csv")
        for code in WATCH:
            hit = full.loc[full.code.eq(code)]
            why = rejected.loc[rejected.code.eq(code),"reason"]
            row = (hit.iloc[0].to_dict() if len(hit) else dict(code=code))
            row["status"] = row.get("main_status", why.iloc[0] if len(why) else "清单中无代码")
            watch.append(dict(horizon=h,**row))
    save(pd.DataFrame(coverage),out/"coverage_by_group.csv")
    save(pd.DataFrame(watch),out/"watchlist.csv")
    periodic_audit = pd.DataFrame(periodic_audits)
    save(periodic_audit,out/"periodic_exclusion_audit.csv")
    cov = {key:src[key] for key in ("asof","catalog_share_codes","valid_nav_paths",
                                   "exact_asof_paths","explicit_subset","include_long_periodic")}
    cov["long_periodic_excluded"] = int(periodic_audit.loc[
        periodic_audit.audit_status.str.startswith("已核验长周期定开"),"code"].nunique()) if len(periodic_audit) else 0
    cov["periodic_unverified_excluded"] = int(periodic_audit.loc[
        periodic_audit.audit_status.eq("定开运作周期未核实，暂不参与排名"),"code"].nunique()) if len(periodic_audit) else 0
    cov["periodic_overrides"] = load_periodic_overrides()
    cov.update(method_audit(events,end,top,min_peers),horizons=horizons,
               input_mode="published feature/stress snapshot; no NAV refresh or new historical replay",
               source_snapshot=source.name, source_sha256=manifest, oos_includes_main=False)
    (out/"audit.json").write_text(json.dumps(cov,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(cov,ensure_ascii=False),flush=True)


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


def long_periodic_audit(catalog, paths, events, end, out):
    """Audit confirmed long-cycle and unverified periodic bond funds outside rankings."""
    rows = []
    for h in HORIZONS:
        f, missing = assemble(catalog,paths,end,h,events)
        if len(f):
            f = f.copy()
            f["horizon"] = h
            f["audit_status"] = np.where(f.is_long_periodic,
                "已核验长周期定开，排除排名","定开运作周期未核实，暂不参与排名")
            rows.append(f)
        for code,group,reason in missing:
            meta = catalog.loc[catalog.code.eq(code)].iloc[0].to_dict()
            rows.append(pd.DataFrame([dict(meta,horizon=h,code=code,group=group,
                                           audit_status=reason)]))
    table = pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(
        columns=["horizon","code","group","audit_status"])
    save(table,out/"periodic_exclusion_audit.csv")
    save(table.loc[table.get("is_long_periodic",pd.Series(False,index=table.index)).eq(True)],
         out/"long_periodic_audit.csv")
    return table


def recency_audit(catalog, paths, end, out, days=5):
    """Per-code presence of the most recent business days up to the cutoff."""
    expected = pd.bdate_range(end=end,periods=days)
    rows = []
    for row in catalog.itertuples(index=False):
        nav = paths.get(row.code)
        if nav is None:
            rows.append(dict(code=row.code,group=row.group,nav_end=None,exact_end=False,
                             missing_recent_days=",".join(str(d.date()) for d in expected)))
            continue
        have = set(nav.dates.normalize())
        missing = [str(d.date()) for d in expected if d not in have]
        rows.append(dict(code=row.code,group=row.group,
                         nav_end=str(nav.dates[-1].date()),exact_end=bool(nav.dates[-1]==end),
                         missing_recent_days=",".join(missing)))
    table = pd.DataFrame(rows)
    save(table,out/"nav_recency_audit.csv")
    summary = {"recent_business_days":[str(d.date()) for d in expected]}
    for d in expected:
        summary[f"exact_{d.date()}"] = int(table.missing_recent_days.fillna("").str.contains(
            str(d.date())).eq(False).sum()) if len(table) else 0
    summary["codes_missing_any_recent_day"] = int(table.missing_recent_days.fillna("").ne("").sum())
    return table,summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asof",required=True)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data"/"all_funds")
    p.add_argument("--output",type=Path,default=ROOT/"reports"/"industry_rank")
    p.add_argument("--replay-snapshot",type=Path,help="rank published features and events at the same cutoff; no NAV or OOS recompute")
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
    p.add_argument("--include-long-periodic",action="store_true",
                   help="rank ≥24-month closed-interval bond funds; default excludes them from eligibility")
    a = p.parse_args(argv)
    if a.oos_folds<0 or a.oos_months<1 or a.top<1 or a.min_peers<2:
        p.error("invalid fold count, holding months, top, or peer count")
    if a.pit_catalog_dir and a.codes:
        p.error("--pit-catalog-dir and --codes cannot be combined")
    end = pd.Timestamp(a.asof).normalize()
    main_events = read_events(a.shocks)
    if a.replay_snapshot:
        if a.fetch or a.codes or a.pit_catalog_dir or a.include_long_periodic:
            p.error("snapshot replay uses the original universe; cannot combine with fetch, codes, PIT or long-periodic overrides")
        return replay_snapshot(a.replay_snapshot,a.output,end,main_events,a.top,a.min_peers)
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
    excluded = catalog.iloc[0:0]
    rankable = catalog
    if not a.include_long_periodic:
        excluded = catalog[catalog.is_long_periodic | catalog.is_periodic_unverified]
        rankable = catalog[~(catalog.is_long_periodic | catalog.is_periodic_unverified)]
    coverage = current_rank(rankable,paths,events,main_events,end,a.output,a.top,a.min_peers,excluded)
    if len(excluded):
        long_periodic_audit(excluded,paths,events,end,a.output)
    _,recency = recency_audit(catalog,paths,end,a.output)
    # Without an archive, never label the supplied snapshot an unbiased OOS test.
    replay = rankable[rankable.code.isin(paths)]
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
           "long_periodic_excluded":int(excluded.is_long_periodic.sum()),
           "periodic_unverified_excluded":int(excluded.is_periodic_unverified.sum()),
           "periodic_overrides":load_periodic_overrides(),
           "include_long_periodic":bool(a.include_long_periodic),
           "nav_recency":recency,
           "nav_audit_share_codes":len(audit),
           "valid_current_catalog_paths":int(catalog.code.isin(paths).sum()),
           "exact_asof_paths":int(audit.get("exact_end",pd.Series(dtype=bool)).eq(True).sum()),
           "explicit_subset":bool(a.codes),"horizons":coverage,"oos_requested_folds":a.oos_folds,
           "oos_holding_months":a.oos_months,"oos_fold_rows":len(summary),
           "oos_paired_complete_rows":int(summary.paired_complete.eq(True).sum()),
           "oos_mode":("dated catalog replay; NAV availability and execution assumptions remain"
                       if historical is not None else
                       "historical replay on supplied snapshot; survivor/current-classification bias possible"),
           "elapsed_seconds":round(time.monotonic()-start,2),
           **method_audit(main_events,end,a.top,a.min_peers),
           "oos_includes_main":False}
    (a.output/"audit.json").write_text(json.dumps(cov,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(cov,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
