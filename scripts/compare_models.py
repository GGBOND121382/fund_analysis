#!/usr/bin/env python3
"""P1 跨模型定量比较：M0/M1/M2/M3 四模型 × 四期限。

产出（reports/all_funds/models_20261120/）：
- model_comparison.json：TOP10/20/50 交集、Spearman 矩阵、指标统计、类别集中度、异常低回撤计数
- MODEL_COMPARISON.md：可读报告
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("M0", "M1", "M2", "M3")
PERIODS = ("5y", "3y", "1y", "6m")


def load_all(inp_m0: Path, inp: Path):
    boards = {}
    for p in PERIODS:
        m0 = pd.read_csv(inp_m0 / f"all_scored_{p}.csv", low_memory=False, dtype={"code": str})
        m0 = m0[m0["status"] == "SCORED"].copy()
        m0["code"] = m0["code"].str.zfill(6)
        m0 = m0.sort_values("SCORE_M0", ascending=False).reset_index(drop=True)
        per = {"M0": m0}
        for m in ("M1", "M2", "M3"):
            per[m] = pd.read_csv(inp / f"full_ranking_{m.lower()}_{p}.csv", dtype={"code": str})
        boards[p] = per
    return boards


def spearman(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str):
    m = a[["code", col_a]].merge(b[["code", col_b]], on="code", how="inner").dropna()
    if len(m) < 3:
        return None, len(m)
    return float(m[col_a].rank().corr(m[col_b].rank(), method="pearson")), len(m)


def board_stats(df, score_col, k=10):
    top = df.dropna(subset=[score_col]).head(k)
    out = {}
    for c in ("period_return", "D", "L"):
        if c in top.columns:
            out[c] = {"mean": float(top[c].mean()), "median": float(top[c].median()),
                      "p95": float(top[c].quantile(0.95))}
    for c in ("D95", "L_daily", "mu_lower"):
        if c in top.columns and top[c].notna().any():
            out[c] = {"mean": float(top[c].mean()), "median": float(top[c].median()),
                      "p95": float(top[c].quantile(0.95))}
    out["zero_mdd_count"] = int((top["D"] <= 0.0002).sum())
    return out


def category_conc(df, score_col, k=50):
    top = df.dropna(subset=[score_col]).head(k)
    vc = top["group"].value_counts(normalize=True)
    return {"top_group": str(vc.index[0]), "top_share": float(vc.iloc[0]),
            "n_groups": int(top["group"].nunique())}


def main():
    inp_m0 = Path("reports/all_funds/comprehensive_20261120")
    inp = Path("reports/all_funds/models_20261120")
    boards = load_all(inp_m0, inp)
    result = {}
    md = ["# M0/M1/M2/M3 跨模型定量比较（asof 2026-11-20，份额级）", ""]

    for p in PERIODS:
        per = boards[p]
        r = {"top_overlap": {}, "spearman": {}, "top10_stats": {}, "category": {}}
        for k in (10, 20, 50):
            sets = {m: set(per[m].dropna(subset=[f"SCORE_{m}" if m != "M0" else "SCORE_M0"]).head(k)["code"])
                    for m in MODELS}
            r["top_overlap"][str(k)] = {f"{a}x{b}": len(sets[a] & sets[b])
                                        for a in MODELS for b in MODELS if a < b}
        for a in MODELS:
            for b in MODELS:
                if a < b:
                    ca = "SCORE_M0" if a == "M0" else f"SCORE_{a}"
                    cb = "SCORE_M0" if b == "M0" else f"SCORE_{b}"
                    rho, n = spearman(per[a], per[b], ca, cb)
                    r["spearman"][f"{a}x{b}"] = {"rho": rho, "n": n}
        for m in MODELS:
            sc = "SCORE_M0" if m == "M0" else f"SCORE_{m}"
            r["top10_stats"][m] = board_stats(per[m], sc)
            r["category"][m] = category_conc(per[m], sc)
        result[p] = r

        md.append(f"## {p}")
        md.append("")
        md.append("TOPk 交集（只数）：")
        for k in (10, 20, 50):
            ov = r["top_overlap"][str(k)]
            md.append(f"- TOP{k}: " + ", ".join(f"{key}={v}" for key, v in ov.items()))
        md.append("")
        md.append("Spearman（全量份额对齐）：")
        for key, v in r["spearman"].items():
            md.append(f"- {key}: rho={v['rho']:.4f} (n={v['n']})")
        md.append("")
        md.append("| 模型 | TOP10累计收益中位 | TOP10回撤中位 | TOP10 D95中位 | TOP10月下行中位 | TOP10日下行中位 | 零回撤只数 | TOP50最大类别占比 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in MODELS:
            s = r["top10_stats"][m]
            c = r["category"][m]
            def g(col, key="median"):
                return f"{s[col][key]*100:.2f}%" if col in s else "—"
            md.append(f"| {m} | {g('period_return')} | {g('D')} | {g('D95')} | {g('L')} | {g('L_daily')} "
                      f"| {s['zero_mdd_count']} | {c['top_share']*100:.0f}%({c['top_group']}) |")
        md.append("")

    with open(inp / "model_comparison.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    (inp / "MODEL_COMPARISON.md").write_text("\n".join(md))
    print(json.dumps({p: {"overlap10": result[p]["top_overlap"]["10"],
                          "spearman": {k: round(v["rho"], 4) for k, v in result[p]["spearman"].items()}}
                      for p in PERIODS}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
