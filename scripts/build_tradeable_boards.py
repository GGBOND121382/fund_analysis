#!/usr/bin/env python3
"""从 M0 全量评分表 + 渠道申购状态（PIT 证据 known_at<=asof）生成可交易展示榜。

过滤规则（证据：reports/all_funds/purchase_channel_status.csv，known_at=2026-11-17）：
- sub_status == 开放申购 -> 可交易
- sub_status == 限大额 且 daily_limit >= min_limit -> 可交易（标注限额）
- 其余（暂停申购/封闭期/认购期/未知） -> 过滤，原因写入 excluded CSV
- 场内交易（ETF）-> 不进场外榜，原因标注 exchange_traded

P0-T1 去重顺序（2026-11-24 修复）：先按份额交易状态过滤，再在可交易份额中
按 product_key 去重，每个产品保留"评分最高且可交易"的份额。原顺序
（先去重再过滤）会在最高分份额暂停申购时把整个产品误删。被跳过的最高分
份额以 best_share_code / share_switched=True 审计。研究全榜（all_scored）
保留全部份额，不受影响。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MIN_LIMIT_DEFAULT = 500.0
UNTRADEABLE = {"暂停申购", "封闭期", "认购期"}


def load_channel_status(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip("﻿") for c in df.columns]
    df["code"] = df["code"].str.zfill(6)
    df["daily_limit_num"] = pd.to_numeric(df["daily_limit"], errors="coerce")
    return df.set_index("code")


def classify(row, ch, min_limit):
    if row["status"] != "SCORED":
        return None, None
    code = str(row["code"]).zfill(6)
    if code not in ch.index:
        return False, "无渠道申购状态记录"
    c = ch.loc[code]
    sub = c["sub_status"] if isinstance(c, pd.Series) else c.iloc[0]["sub_status"]
    limit = c["daily_limit_num"] if isinstance(c, pd.Series) else c.iloc[0]["daily_limit_num"]
    if sub == "开放申购":
        return True, "开放申购"
    if sub == "限大额":
        if pd.notna(limit) and limit >= min_limit:
            return True, f"限大额({limit:.0f}元/日)"
        return False, f"限额过低({limit:.0f}元/日)" if pd.notna(limit) else "限额未知"
    if sub in UNTRADEABLE:
        return False, str(sub)
    if sub == "场内交易":
        return False, "场内交易(需证券账户)"
    return False, f"状态待核实({sub})"


def process_period(df, ch, min_limit, top):
    """份额级评分 -> 交易状态过滤 -> 可交易份额内 product 去重。"""
    df = df.copy()
    df["code"] = df["code"].str.zfill(6)
    scored = df[df["status"] == "SCORED"].copy()
    # P0-T2: 收益双口径——R 为年化（annualized_return），period_return 为窗口累计
    if "annualized_return" not in scored.columns and "R" in scored.columns:
        scored["annualized_return"] = scored["R"]
    verdicts = scored.apply(lambda r: classify(r, ch, min_limit), axis=1)
    scored["tradeable"] = [v[0] for v in verdicts]
    scored["trade_note"] = [v[1] for v in verdicts]
    scored = scored.sort_values("SCORE_M0", ascending=False)
    # 产品原始最高分份额（可能不可交易），用于审计 share 切换
    best = scored.drop_duplicates(subset=["product_key"], keep="first")[
        ["product_key", "code", "name", "tradeable", "trade_note"]].rename(
        columns={"code": "best_share_code", "name": "best_share_name",
                 "tradeable": "best_share_tradeable", "trade_note": "best_share_note"})
    # P0-T1: filter first, then dedup within tradeable shares
    trad_shares = scored[scored["tradeable"]]
    trad = trad_shares.drop_duplicates(subset=["product_key"], keep="first").merge(
        best, on="product_key", how="left")
    trad["share_switched"] = trad["code"] != trad["best_share_code"]
    trad = trad.reset_index(drop=True)
    trad.insert(0, "rank", range(1, len(trad) + 1))
    # 产品级排除：没有任何可交易份额的产品（代表行=原始最高分份额）
    tradable_products = set(trad_shares["product_key"])
    excl = scored.drop_duplicates(subset=["product_key"], keep="first")
    excl = excl[~excl["product_key"].isin(tradable_products)].reset_index(drop=True)
    stats = {
        "scored_fund_level": int(len(best)),
        "tradeable": int(len(trad)),
        "share_switched": int(trad["share_switched"].sum()),
        "excluded": int(len(excl)),
        "excluded_reasons": excl["trade_note"].value_counts().to_dict(),
        "top10": trad.head(top)[["rank", "code", "name", "SCORE_M0", "trade_note",
                                 "share_switched"]].to_dict("records"),
    }
    return trad, excl, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/all_funds/comprehensive_20261120")
    ap.add_argument("--channel-status", default="reports/all_funds/purchase_channel_status.csv")
    ap.add_argument("--min-limit", type=float, default=MIN_LIMIT_DEFAULT)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    inp = Path(args.input)
    ch = load_channel_status(Path(args.channel_status))
    summary = {}
    for period in ("5y", "3y", "1y", "6m"):
        df = pd.read_csv(inp / f"all_scored_{period}.csv", low_memory=False, dtype={"code": str})
        trad, excl, stats = process_period(df, ch, args.min_limit, args.top)
        trad.head(args.top).to_csv(inp / f"top10_channel_tradeable_{period}.csv", index=False)
        trad.to_csv(inp / f"top_candidates_channel_tradeable_{period}.csv", index=False)
        excl.to_csv(inp / f"excluded_untradeable_{period}.csv", index=False)
        summary[period] = stats
    with open(inp / "tradeable_filter_audit.json", "w") as f:
        json.dump({"min_daily_limit": args.min_limit,
                   "evidence": "purchase_channel_status.csv known_at=2026-11-17 (PIT <= asof)",
                   "dedup_order": "filter-tradeable-first then product_key dedup (P0-T1 fix)",
                   "periods": summary}, f, ensure_ascii=False, indent=1)
    for p, s in summary.items():
        print(p, "tradeable:", s["tradeable"], "excluded:", s["excluded"],
              "share_switched:", s["share_switched"], s["excluded_reasons"])


if __name__ == "__main__":
    main()
