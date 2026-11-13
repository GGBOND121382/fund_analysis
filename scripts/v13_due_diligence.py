#!/usr/bin/env python3
"""P2: 候选债基季报尽调分类 + 研究候选池落地。

输入: reports/v13_bond_shortlist.csv + data/disclosures/{code}.csv
输出: reports/v13_due_diligence.csv, config/v13_candidate_pool_research.json
分类: resilient_explainable / resilient_opaque / short_term_only
证据规则: 只用 published_at <= 决策日(2026-11-12) 的最新一期披露。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fundkit import storage  # noqa: E402

ASOF = pd.Timestamp("2026-11-12")
SHORTLIST = "reports/v13_bond_shortlist.csv"
OUT_CSV = "reports/v13_due_diligence.csv"
OUT_POOL = "config/v13_candidate_pool_research.json"


def latest_disclosure(code: str) -> dict | None:
    p = Path(f"data/disclosures/{code}.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "published_at" in df.columns:
        df = df[pd.to_datetime(df["published_at"], errors="coerce") <= ASOF]
    if df.empty:
        return None
    df = df.sort_values("report_asof")
    r = df.iloc[-1].to_dict()
    r["n_periods"] = len(df)
    return r


def classify(row: pd.Series, disc: dict | None) -> tuple[str, str]:
    """返回 (分类, 依据)。"""
    notes = []
    if disc is None:
        return "resilient_opaque", "无可用季报披露（未公开或采集失败）"
    rep = str(disc.get("report_asof", ""))
    pub = str(disc.get("published_at", ""))
    size = float(disc.get("fund_size_cny") or 0)
    eq = float(disc.get("equity_nav_pct") or 0)
    cb = float(disc.get("convertible_nav_pct_lower_bound") or 0)
    credit = float(disc.get("credit_bond_nav_pct_lower_bound") or 0)
    gov = float(disc.get("gov_bond_nav_pct_lower_bound") or 0)
    policy = float(disc.get("policy_bank_nav_pct_lower_bound") or 0)
    notes = str(disc.get("quality_flag", ""))
    notes_list = [f"报告期{rep}", f"披露日{pub}", f"规模{size/1e8:.2f}亿"]
    if "SINGLE_SOURCE" in notes or "LOWER_BOUND" in notes:
        notes_list.append("披露为单源/下限口径")
    if size < 2e8:
        notes_list.append("规模<2亿(清盘风险)")
    if rep < "2026-06-30":
        return "resilient_opaque", ";".join(notes_list + ["披露滞后超过一个季度"])
    if size < 2e8:
        return "resilient_opaque", ";".join(notes_list)
    # 风险可解释性：利率/政策金债占比高 → 久期风险为主，可解释；信用/转债高 → 信用不透明
    if eq > 5 or cb > 10:
        return "resilient_opaque", ";".join(notes_list + [f"股票{eq:.1f}%/转债下限{cb:.1f}%"])
    if credit > 50:
        return "resilient_opaque", ";".join(notes_list + [f"信用债下限{credit:.1f}%（发行人集中度未知）"])
    if gov + policy >= 30:
        return "resilient_explainable", ";".join(notes_list + [f"利率债+政策金债下限{gov+policy:.1f}%"])
    # 长期适配性：近一年收益<2% 的作为短期抗跌但长期收益不足
    if float(row.get("one_year_total_return") or 0) < 0.02:
        return "short_term_only", ";".join(notes_list + ["近一年收益<2%"])
    return "resilient_explainable", ";".join(notes_list + ["敞口以前五大债券下限为准"])


def unexplained_spike_days(code: str, threshold: float = 0.03) -> list[str]:
    """债基单日 |收益|>3% 且当日无分红除息 → 疑似巨额赎回/净值异常。"""
    from fundkit import metrics
    hist = storage.load_nav(code)
    divs = storage.load_dividends().get(code, ())
    ex_dates = {str(d) for d, _ in divs}
    hist = hist.sort_values("date")
    ret = hist["nav"].pct_change()
    hits = hist.loc[ret.abs() > threshold, "date"].astype(str)
    return [d for d in hits if d not in ex_dates]


def main() -> None:
    short = pd.read_csv(SHORTLIST, dtype={"code": str})
    rows = []
    for _, r in short.iterrows():
        disc = latest_disclosure(r["code"])
        cls, why = classify(r, disc)
        rows.append({
            "code": r["code"], "name": r["name"], "group": r["group"],
            "return_since_shock": r["return_since_shock"],
            "mdd_since_shock": r["max_drawdown_since_shock"],
            "one_year_total_return": r["one_year_total_return"],
            "one_year_mdd": r["one_year_mdd"],
            "dd_class": cls, "evidence": why,
            "has_disclosure": disc is not None,
        })
    out = pd.DataFrame(rows)
    # 净值异常（巨额赎回/错误单日跳变）一律降为 opaque，不进入可投资池
    spikes = {c: unexplained_spike_days(c) for c in out["code"]}
    for i, r in out.iterrows():
        sp = spikes.get(r["code"]) or []
        if sp and r["dd_class"] == "resilient_explainable":
            out.loc[i, "dd_class"] = "resilient_opaque"
            out.loc[i, "evidence"] = f"{r['evidence']};单日异常收益{sp[:3]}(疑似巨额赎回)"
    out["unexplained_spike_days"] = out["code"].map(lambda c: ";".join(spikes.get(c) or []))
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 研究候选池 = 13 持仓 + 尽调通过的 explainable 候选
    snap = json.load(open("config/account_snapshot.json"))
    held = {f["code"] for f in snap["funds"]}
    fl = storage.load_fund_list()
    name_of = dict(zip(fl["code"].astype(str), fl["name"]))
    pool = {"asof": str(ASOF.date()), "purpose": "v1.3 pool-B research only; not a trade list",
            "holdings": [{"code": f["code"], "name": f.get("name", name_of.get(f["code"])),
                          "category": f.get("category"), "role": "existing_holding"}
                         for f in snap["funds"]],
            "candidates": []}
    for _, r in out[(out["dd_class"] == "resilient_explainable") &
                    (~out["code"].isin(held))].iterrows():
        pool["candidates"].append({
            "code": r["code"], "name": r["name"], "category": "bond",
            "group": r["group"], "role": "new_candidate",
            "evidence": r["evidence"]})
    Path(OUT_POOL).write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out["dd_class"].value_counts().to_string())
    print("pool candidates:", len(pool["candidates"]))
    print("OUT:", OUT_CSV, OUT_POOL)


if __name__ == "__main__":
    main()
