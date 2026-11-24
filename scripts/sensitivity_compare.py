#!/usr/bin/env python3
"""P1 敏感性比较：不同 bootstrap 区块长度/情景数/种子下 D95 与 M2 TOP10 的稳定性。

对照主实验（block=10, scenarios=2000, seed=42）：
- sensitivity_block5_s2000_seed42
- sensitivity_block20_s2000_seed42
- sensitivity_block10_s5000_seed42
- sensitivity_block10_s2000_seed7
产出 sensitivity_report.json / .md。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = ("5y", "3y", "1y", "6m")
BASE = Path("reports/all_funds/models_20261120")
CONFIGS = {
    "block5_s2000_seed42": "区块长度5",
    "block20_s2000_seed42": "区块长度20",
    "block10_s5000_seed42": "情景数5000",
    "block10_s2000_seed7": "随机种子7",
}


def main():
    result, md = {}, ["# M2 Bootstrap 敏感性报告（asof 2026-11-20）", ""]
    for p in PERIODS:
        base = pd.read_csv(BASE / f"full_ranking_m2_{p}.csv", dtype={"code": str})
        base_top10 = set(base.head(10)["code"])
        r = {}
        for cfg, label in CONFIGS.items():
            d = pd.read_csv(BASE / f"sensitivity_{cfg}" / f"full_ranking_m2_{p}.csv",
                            dtype={"code": str})
            m = base[["code", "D95", "SCORE_M2"]].merge(
                d[["code", "D95", "SCORE_M2"]], on="code", suffixes=("_base", "_alt")).dropna()
            top10_overlap = len(base_top10 & set(d.head(10)["code"]))
            r[cfg] = {
                "label": label,
                "d95_corr": float(m["D95_base"].corr(m["D95_alt"])),
                "d95_median_abs_diff": float((m["D95_base"] - m["D95_alt"]).abs().median()),
                "score_spearman": float(m["SCORE_M2_base"].rank().corr(m["SCORE_M2_alt"].rank())),
                "top10_overlap": top10_overlap,
                "n": int(len(m)),
            }
        result[p] = r
        md.append(f"## {p}")
        md.append("| 配置 | D95相关 | D95绝对差中位 | 总分Spearman | TOP10重合 | n |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for cfg, label in CONFIGS.items():
            v = r[cfg]
            md.append(f"| {label} | {v['d95_corr']:.4f} | {v['d95_median_abs_diff']*100:.3f}% "
                      f"| {v['score_spearman']:.4f} | {v['top10_overlap']}/10 | {v['n']} |")
        md.append("")
    with open(BASE / "sensitivity_report.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    (BASE / "sensitivity_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
