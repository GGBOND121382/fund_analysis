#!/usr/bin/env python3
"""两个极端 OOS 测试折（2025-11-20、2026-05-20）的逐基金归因审计。

对每个模型 TOP10 的每只基金：持有窗口真实净值收益、等权组合亏损贡献、
组合内回撤、净值数据完整性（观测数、最大间隔、起止对齐）。只读 fold CSV
与本地净值缓存，不重打分、不改结论。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_industry_fund_ranking import load_catalog, load_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OOS = ROOT / "reports" / "all_funds" / "models_20261120" / "oos"
FOLDS = ("2025-11-20", "2026-05-20")
MODELS = ("S_M0", "S_M1", "S_M2", "S_M3")
HOLD_MONTHS = 6
FEE = 0.001


def audit_fold(paths, t_str):
    t = pd.Timestamp(t_str)
    end = t + pd.DateOffset(months=HOLD_MONTHS)
    rows = []
    for model in MODELS:
        f = OOS / f"fold_{t_str}_{model}.csv"
        if not f.is_file():
            rows.append(dict(formation=t_str, model=model,
                             status="FOLD_CSV_MISSING"))
            continue
        top = pd.read_csv(f, dtype={"code": str})
        top["code"] = top["code"].str.zfill(6)
        for _, r in top.iterrows():
            nav = paths.get(r["code"])
            base = dict(formation=t_str, model=model, code=r["code"],
                        name=r["name"], score=float(r[model]))
            if nav is None:
                rows.append(dict(base, status="NAV_MISSING"))
                continue
            i, j = nav.right(t), nav.right(end)
            v = nav.value[i:j + 1]
            d = nav.dates[i:j + 1]
            gross = v / v[0]
            net = gross[-1] * (1 - FEE / 2) ** 2 - 1.0
            mdd = float(np.max(1 - gross / np.maximum.accumulate(gross)))
            gaps = np.diff(d.values.astype("datetime64[D]").astype("int64"))
            rows.append(dict(base, status="OK",
                             obs=len(v), max_gap_days=int(gaps.max()) if len(gaps) else 0,
                             first_nav_date=str(d[0].date()),
                             last_nav_date=str(d[-1].date()),
                             edge_gap_start_days=int((t - d[0]).days),
                             edge_gap_end_days=int((end - d[-1]).days),
                             fund_net_return=float(net),
                             port_contribution=float(net / 10.0),
                             fund_mdd=mdd))
    return rows


def main():
    out_dir = ROOT / "reports" / "all_funds" / "models_20261120" / "oos"
    codes = set()
    for t_str in FOLDS:
        for model in MODELS:
            f = OOS / f"fold_{t_str}_{model}.csv"
            if f.is_file():
                codes.update(pd.read_csv(f, dtype={"code": str})["code"]
                             .str.zfill(6))
    catalog = load_catalog(ROOT / "data" / "all_funds" / "fund_list.csv",
                           codes=sorted(codes))
    paths, audit = load_paths(catalog, ROOT / "data" / "all_funds",
                              pd.Timestamp("2026-11-20"))
    rows = []
    for t_str in FOLDS:
        rows.extend(audit_fold(paths, t_str))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "extreme_fold_attribution.csv", index=False)
    ok = df[df.status.eq("OK")]
    summary = (ok.groupby(["formation", "model"])
               .agg(n=("code", "count"),
                    port_net=("port_contribution", "sum"),
                    worst_fund=("fund_net_return", "min"),
                    best_fund=("fund_net_return", "max"),
                    max_mdd=("fund_mdd", "max"),
                    min_obs=("obs", "min"),
                    max_gap=("max_gap_days", "max"))
               .reset_index())
    summary.to_csv(out_dir / "extreme_fold_summary.csv", index=False)
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (out_dir / "extreme_fold_audit_meta.json").write_text(json.dumps(
        dict(git_sha=sha, folds=FOLDS, hold_months=HOLD_MONTHS, fee=FEE,
             nav_codes_loaded=len(paths),
             load_issues=audit.loc[~audit.valid].to_dict("records")),
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
