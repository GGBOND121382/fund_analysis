#!/usr/bin/env python3
"""P0-T3 净值平滑定量审计。

对象：1y/6m research 榜前 100（product 去重）+ 两期限全部 SMOOTHING_REVIEW + 专项基金。
指标（窗口内，与主榜同一窗口口径）：
- mdd             窗口最大回撤
- daily_neg_ratio 日度负收益比例
- monthly_neg_ratio 月度负收益比例
- L_daily         日度下行半标准差 sqrt(252/n * sum(min(r,0)^2))
- L_monthly       月度下行半标准差 sqrt(12/m * sum(min(rm,0)^2))
- zero_change_ratio 净值零变化日比例
- max_nonneg_streak 最长连续非负收益天数
- pricing_basis / periodic_status / periodic_months（catalog）
- fund_size（专项基金，同花顺基金基本信息接口）

判定原则：不凭单一零回撤下结论——结合零变化比例、连续非负天数、下行半标准差
与定价口径（摊余成本/市值法）综合标记 audit_flag：
- "摊余成本法"           : pricing_basis 显示摊余/万份收益口径
- "高度平滑待人工核验"   : mdd<=0.02% 且 (zero_change_ratio>=30% 或 max_nonneg_streak>=60)
- "低回撤但波动结构正常" : 其余低回撤对象
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_industry_fund_ranking import load_catalog, load_paths  # noqa: E402
from fundkit.industry_rank import HORIZONS, window_dates  # noqa: E402

WATCHLIST = ["020952", "007225", "007589", "008693", "007680", "008338", "025368"]


def smooth_metrics(nav, begin, end):
    i, j = nav.right(begin), nav.right(end)
    if i < 0 or j < 0 or j <= i:
        return None
    v = nav.value[i:j + 1]
    r = v[1:] / v[:-1] - 1
    if len(r) < 5:
        return None
    mdd = float(max(0.0, np.max(1.0 - v / np.maximum.accumulate(v))))
    n = len(r)
    neg = r[r < 0]
    streak = cur = 0
    for x in r:
        cur = cur + 1 if x >= 0 else 0
        streak = max(streak, cur)
    return {
        "obs": j - i + 1,
        "mdd": mdd,
        "daily_neg_ratio": float(np.mean(r < 0)),
        "L_daily": float(np.sqrt(252.0 / n * np.sum(np.minimum(r, 0.0) ** 2))),
        "zero_change_ratio": float(np.mean(r == 0)),
        "max_nonneg_streak": int(streak),
    }


def monthly_metrics(nav, end, horizon):
    n, _ = HORIZONS[horizon]
    begin, period, complete = window_dates(end, n)
    monthly = nav.monthly(end, period, complete)
    if monthly is None or len(monthly) < 2:
        return begin, {"monthly_neg_ratio": np.nan, "L_monthly": np.nan}
    m = len(monthly)
    return begin, {
        "monthly_neg_ratio": float(np.mean(monthly < 0)),
        "L_monthly": float(np.sqrt(12.0 / m * np.sum(np.minimum(monthly, 0.0) ** 2))),
    }


def audit_flag(row):
    if row["pricing_basis"] != "fund_nav":
        return "摊余成本法/非市值口径"
    if pd.notna(row["mdd"]) and row["mdd"] <= 0.0002:
        if row["zero_change_ratio"] >= 0.30 or row["max_nonneg_streak"] >= 60:
            return "高度平滑待人工核验"
        return "低回撤但波动结构正常"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-11-20")
    ap.add_argument("--input", default="reports/all_funds/comprehensive_20261120")
    ap.add_argument("--data-dir", default="data/all_funds")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    end = pd.Timestamp(args.asof)
    inp = Path(args.input)

    catalog = load_catalog(Path(args.data_dir) / "fund_list.csv")
    paths, _ = load_paths(catalog, Path(args.data_dir), end)
    cat_idx = catalog.set_index("code")

    targets = {}
    for p in ("1y", "6m"):
        df = pd.read_csv(inp / f"all_scored_{p}.csv", low_memory=False, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        scored = df[df["status"] == "SCORED"].sort_values("SCORE_M0", ascending=False)
        top100 = scored.drop_duplicates("product_key").head(args.top)
        smooth = df[df["status"] == "SMOOTHING_REVIEW"]
        watch = df[df["code"].isin(WATCHLIST)]
        sel = pd.concat([top100, smooth, watch]).drop_duplicates("code")
        targets[p] = sel
        print(p, "top100:", len(top100), "smoothing:", len(smooth), "union:", len(sel))

    summary = {}
    for p in ("1y", "6m"):
        n, _ = HORIZONS[p]
        begin, _, _ = window_dates(end, n)
        rows = []
        for _, r in targets[p].iterrows():
            code = r["code"]
            rec = {"code": code, "name": r["name"], "group": r["group"],
                   "board_status": r["status"], "SCORE_M0": r["SCORE_M0"]}
            nav = paths.get(code)
            if nav is None:
                rec["error"] = "nav cache missing"
                rows.append(rec)
                continue
            sm = smooth_metrics(nav, begin, end)
            _, mm = monthly_metrics(nav, end, p)
            if sm is None:
                rec["error"] = "insufficient observations"
                rows.append(rec)
                continue
            rec.update(sm)
            rec.update(mm)
            cat = cat_idx.loc[code] if code in cat_idx.index else None
            rec["pricing_basis"] = cat["pricing_basis"] if cat is not None else ""
            rec["periodic_status"] = cat["periodic_status"] if cat is not None else ""
            rec["periodic_months"] = cat["periodic_months"] if cat is not None else ""
            rows.append(rec)
        out_df = pd.DataFrame(rows)
        out_df["audit_flag"] = out_df.apply(audit_flag, axis=1)
        out_df["is_watchlist"] = out_df["code"].isin(WATCHLIST)
        out_df.to_csv(inp / f"smoothing_audit_{p}.csv", index=False)
        flagged = out_df["audit_flag"].value_counts().to_dict()
        summary[p] = {"audited": int(len(out_df)), "flags": flagged,
                      "top100_flagged": int((out_df.head(args.top)["audit_flag"] != "").sum())}
        print(p, summary[p])

    with open(inp / "smoothing_audit_summary.json", "w") as f:
        json.dump({"asof": args.asof, "watchlist": WATCHLIST,
                   "principle": "不凭单一零回撤下结论，结合零变化比例/连续非负天数/定价口径综合判断",
                   "periods": summary}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
