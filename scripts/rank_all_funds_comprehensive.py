#!/usr/bin/env python3
"""M0 comprehensive full-market fund ranking (spec: docs/FUND_RANKING_KIMICODE_FINAL_SPEC_20261124.md).

Reuses catalog loading, NAV paths and window features from the industry
ranking pipeline. Scores every catalog share per horizon; nothing is
silently dropped - every exclusion keeps an explicit status/reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundkit.industry_rank import HORIZONS, LOW_VOL, feature
from scripts.run_industry_fund_ranking import load_catalog, load_paths

CONFIG = json.loads((ROOT/"config"/"comprehensive_ranking_m0_v1.json").read_text(encoding="utf-8"))
SPECIAL = ("025368","007589","008693","007680","008338")
CHANNEL_STATUS = ROOT/"reports"/"all_funds"/"purchase_channel_status.csv"
W = CONFIG["weights"]
assert (W["annualized_return"],W["max_drawdown"],W["shock_resilience"],W["monthly_downside"]) == (0.4,0.35,0.15,0.1)
SMOOTH_MDD = CONFIG["scoring"]["risk_smoothing_review_mdd_lte"]
THEMES = ("银行","证券","保险","医药","医疗","红利","黄金","军工","半导体","芯片","消费","白酒","新能源","地产","煤炭","有色")


def shock_metrics(nav, start, end, gap_max=14):
    """Spec (4): exact endpoints plus valid observations inside the window."""
    i, j = nav.left(start), nav.right(end)
    if i > j or nav.dates[i] != start or nav.dates[j] != end:
        return None
    if np.any(nav.gaps[i:j] > gap_max):
        return None
    return nav.stats(i, j)  # (period_return, max_drawdown) within the event


def horizon_rows(catalog, paths, end, h, shock_start, shock_end):
    """One full-catalog record per code; metrics NaN unless fully computable."""
    rows = []
    for row in catalog.itertuples(index=False):
        rec = dict(code=row.code,name=row.name,group=row.group,product_key=row.product_key,
                   share=row.share,fund_type=row.fund_type,currency=row.currency,
                   pricing_basis=row.pricing_basis,is_long_periodic=row.is_long_periodic,
                   is_periodic_unverified=row.is_periodic_unverified,
                   periodic_source=row.periodic_source)
        nav = paths.get(row.code)
        if nav is None:
            rows.append(dict(rec,status="缺有效净值缓存",risk_status="",reason="nav cache missing"))
            continue
        f, why = feature(nav,end,h,bool(LOW_VOL.search(row.group)))
        if f is None:
            rows.append(dict(rec,status=why,risk_status="",reason=why))
            continue
        years = (f["end"]-f["start"]).days/365.2425
        R = (1.+f["period_return"])**(1./years)-1.
        monthly = f["_monthly"]
        L = float(np.sqrt(np.mean(np.minimum(monthly,0.)**2)))
        rec.update(obs=f["obs"],window_start=str(f["start"].date()),window_end=str(f["end"].date()),
                   months_used=len(monthly),R=R,D=f["max_drawdown"],L=L,
                   period_return=f["period_return"])
        shock = shock_metrics(nav,pd.Timestamp(shock_start),pd.Timestamp(shock_end))
        if shock is None:
            rows.append(dict(rec,status="事件期缺数据",risk_status="",reason="shock endpoints or coverage missing"))
            continue
        rec["Aret"],rec["Add"] = shock
        risk = ("SMOOTHING_REVIEW" if f["period_return"] > 0 and f["max_drawdown"] <= SMOOTH_MDD else "")
        rows.append(dict(rec,status="SCORED" if not risk else "SMOOTHING_REVIEW",
                         risk_status=risk,reason="positive return with MDD<=0.02%: valuation review" if risk else ""))
    return pd.DataFrame(rows)


def score(df):
    """Spec (5)(6): percentiles on one fixed candidate set, then frozen weights."""
    u = df[df.status.eq("SCORED")].copy()
    n = len(u)
    out = df.copy()
    for c in ("SR","SD","SL","SA","SCORE_M0"):
        out[c] = np.nan
    if n == 0:
        out["raw_rank"] = np.nan
        return out,u
    if n == 1:
        for c in ("SR","SD","SL","SA"):
            u[c] = 50.
    else:
        p = {c: 100.*(u[c].rank(method="average")-1.)/(n-1.) for c in ("R","D","L","Aret","Add")}
        u["SR"] = p["R"]; u["SD"] = 100.-p["D"]; u["SL"] = 100.-p["L"]
        u["SA"] = 0.5*p["Aret"]+0.5*(100.-p["Add"])
    u["SCORE_M0"] = 0.40*u.SR+0.35*u.SD+0.15*u.SA+0.10*u.SL
    u = u.sort_values(["SCORE_M0","code"],ascending=[False,True])
    u["raw_rank"] = np.arange(1,n+1)
    cols = ["code","SR","SD","SL","SA","SCORE_M0","raw_rank"]
    out = out.drop(columns=[c for c in cols[1:] if c in out],errors="ignore").merge(
        u[cols],on="code",how="left",validate="one_to_one")
    peers = out[out.status.eq("SCORED")].copy()
    if "group" in peers.columns:
        peers["peer_rank"] = peers.groupby("group")["SCORE_M0"].rank(ascending=False,method="min")
        out = out.drop(columns=["peer_rank"],errors="ignore").merge(
            peers[["code","peer_rank"]],on="code",how="left",validate="one_to_one")
    else:
        out["peer_rank"] = np.nan
    return out,u


def attach_trade_status(df):
    """PIT discipline: contract-verified blocks, single-channel observations, else UNKNOWN."""
    status = pd.Series("UNKNOWN",index=df.index)
    blocked = df.periodic_source.astype(str).str.len().gt(0) & df.is_long_periodic.fillna(False)
    status[blocked] = "CONFIRMED_BLOCKED"
    if CHANNEL_STATUS.is_file():
        ch = pd.read_csv(CHANNEL_STATUS,dtype={"code":str})
        known_at = str(ch.channel_status_checked_at.dropna().max()) if "channel_status_checked_at" in ch else ""
        plat = set(ch.code)
        status[df.code.isin(plat) & status.eq("UNKNOWN")] = "PLATFORM_SPECIFIC"
    else:
        known_at = ""
    df["trade_status"] = status
    return df,dict(channel_snapshot=CHANNEL_STATUS.name if CHANNEL_STATUS.is_file() else None,
                   channel_known_at=known_at)


def boards(u, top):
    """Display boards: research top (product-deduped) and theme-diversified top."""
    pool = u[~u.trade_status.eq("CONFIRMED_BLOCKED")].sort_values(["SCORE_M0","code"],ascending=[False,True])
    research = pool.drop_duplicates("product_key").head(top).copy()
    research["board"] = "RESEARCH_UNKNOWN_TRADE_ALLOWED"
    seen, kept, themes = set(), [], {}
    for r in pool.itertuples(index=False):
        if r.product_key in seen:
            continue
        theme = next((t for t in THEMES if t in r.name),None)
        if theme and themes.get(theme,0) >= 2:
            continue
        seen.add(r.product_key); themes[theme] = themes.get(theme,0)+1 if theme else themes.get(theme,0)
        kept.append(dict(r._asdict(),theme=theme or ""))
        if len(kept) >= top:
            break
    diversified = pd.DataFrame(kept)
    return research,diversified


def audit(df, u, h, n_codes, research, diversified, trade_meta, elapsed):
    counts = df.status.value_counts().to_dict()
    return {"horizon":h,"catalog_codes":n_codes,"audit_total":int(df.code.nunique()),
            "conservation_ok":bool(df.code.nunique()==n_codes),"status_counts":counts,
            "scored_candidates":len(u),"smoothing_review":int(df.risk_status.eq("SMOOTHING_REVIEW").sum()),
            "trade_status_counts":df.trade_status.value_counts().to_dict(),**trade_meta,
            "research_board_rows":len(research),"diversified_board_rows":len(diversified),
            "verified_tradeable_board_rows":0,
            "verified_tradeable_reason":"no point-in-time contract+fee evidence at asof; board intentionally empty",
            "elapsed_seconds":round(elapsed,2)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asof",required=True)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data"/"all_funds")
    p.add_argument("--shock-start",default=CONFIG["shock"]["baseline"])
    p.add_argument("--shock-end",default=CONFIG["shock"]["end"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--top",type=int,default=CONFIG["ranking"]["top"])
    a = p.parse_args(argv)
    end = pd.Timestamp(a.asof).normalize()
    t0 = time.monotonic()
    catalog = load_catalog(a.data_dir/"fund_list.csv")
    paths,_ = load_paths(catalog,a.data_dir,end)
    a.output.mkdir(parents=True,exist_ok=True)
    meta = {"model_id":CONFIG["model_id"],"asof":a.asof,"weights":W,
            "shock":{"start":a.shock_start,"end":a.shock_end},
            "git_sha":__import__("subprocess").check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "catalog_codes":int(catalog.code.nunique()),"horizons":{}}
    keep = [c for c in ("code","name","group","product_key","share","fund_type","currency","pricing_basis",
            "obs","window_start","window_end","months_used","R","D","L","Aret","Add","period_return",
            "SR","SD","SL","SA","SCORE_M0","raw_rank","peer_rank","status","risk_status","reason",
            "trade_status","is_long_periodic","is_periodic_unverified","periodic_source")]
    for h in HORIZONS:
        th = time.monotonic()
        df = horizon_rows(catalog,paths,end,h,a.shock_start,a.shock_end)
        df,u = score(df)
        df,trade_meta = attach_trade_status(df)
        u = u.merge(df[["code","trade_status"]],on="code",how="left",validate="one_to_one")
        research,diversified = boards(u,a.top)
        df.reindex(columns=keep).to_csv(a.output/f"all_scored_{h}.csv",index=False,encoding="utf-8-sig")
        research.to_csv(a.output/f"top10_research_{h}.csv",index=False,encoding="utf-8-sig")
        diversified.to_csv(a.output/f"top10_diversified_{h}.csv",index=False,encoding="utf-8-sig")
        df.head(0).to_csv(a.output/f"top10_verified_tradeable_{h}.csv",index=False,encoding="utf-8-sig")
        meta["horizons"][h] = audit(df,u,h,int(catalog.code.nunique()),research,diversified,trade_meta,time.monotonic()-th)
        print(h,"scored",meta["horizons"][h]["scored_candidates"],meta["horizons"][h]["status_counts"],flush=True)
    meta["elapsed_seconds"] = round(time.monotonic()-t0,2)
    (a.output/"audit.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"elapsed":meta["elapsed_seconds"],
                      "scored":{h:v["scored_candidates"] for h,v in meta["horizons"].items()}}),flush=True)


if __name__ == "__main__":
    main()
