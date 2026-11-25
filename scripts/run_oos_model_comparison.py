#!/usr/bin/env python3
"""P2 历史样本外模型比较：M0-noSA / M1 / M2 / M3。

设计与已知局限（如实声明）：
- 历史形成日滚动：半年间隔 7 个独立折（持有期 6 个月不重叠，相邻折独立）；
  辅助报告季度形成日（重叠持有期，不作为独立统计证据）。
- 每个形成日只用截至当日的净值（NavPath 截断），不使用形成日后数据。
- 历史冲击事件无法构成统一事前规则 -> SA 项去除，M0 降级为三因子版
  （SR/SD/SL 按 0.40/0.35/0.10 归一化为 0.4706/0.4118/0.1176），记 M0'。
  本验证不声称覆盖当前 M0 完整四因子版本。
- 基金目录为 2026-11 当前存续目录：已清盘/转型基金不在池内 -> 幸存者偏差，
  所有模型同池，相对比较有效，绝对水平偏乐观。
- 交易费用：统一 roundtrip 0.1%（无法 PIT 核验各基金各时点费率，偏保守）。
- Bootstrap：500 情景（P1 敏感性已验证 2000/5000 下 D95 排名稳定；500 情景的
  蒙特卡洛噪声对 TOP10 组合层面影响有限，如实标注）。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_industry_fund_ranking import load_catalog, load_paths  # noqa: E402
from fundkit.industry_rank import NavPath, window_dates  # noqa: E402
from fundkit.opportunity_rank import stationary_bootstrap, max_drawdown  # noqa: E402
from rank_models_m123 import eb_posteriors, pct, BOOT_STEPS, EB_Z, annual_log_mean_se  # noqa: E402

W3 = {"return": 0.40 / 0.85, "drawdown": 0.35 / 0.85, "downside": 0.10 / 0.85}
SMOOTH_MDD = 0.0002
ROUNDTRIP_FEE = 0.001
FORMATION_DATES = ["2023-05-19", "2023-11-20", "2024-05-20", "2024-11-20",
                   "2025-05-20", "2025-11-20", "2026-05-20"]
HOLD_MONTHS = 6
BOOT_SCENARIOS = 500


def truncate(nav: NavPath, t: pd.Timestamp) -> NavPath | None:
    j = nav.right(t)
    if j < 65:
        return None
    return NavPath(nav.dates[: j + 1], nav.value[: j + 1], nav.money)


def fold_features(catalog, paths, t, low_vol_re):
    """截至 t 的 1y 窗特征（三因子 + EB/bootstrap/L_daily 原料）。"""
    n = 12
    begin, period, complete = window_dates(t, n)
    rows = []
    for row in catalog.itertuples(index=False):
        nav = truncate(paths.get(row.code), t) if paths.get(row.code) else None
        if nav is None:
            continue
        i, j = nav.right(begin), nav.right(t)
        if i < 0 or j < 0 or j - i + 1 < 200 or nav.day(t) - nav.days[j] > 7 \
                or nav.day(begin) - nav.days[i] > 7:
            continue
        if np.any(nav.gaps[i:j] > 14):
            continue
        if low_vol_re.search(row.group) and np.any(nav.jumps[i:j] > 0.03):
            continue
        v = nav.value[i:j + 1]
        ret, mdd = nav.stats(i, j)
        monthly = nav.monthly(t, period, complete)
        if monthly is None or len(monthly) < 4:
            continue
        # 与主榜一致的平滑闸门（历史形成日同样适用）
        if ret > 0 and mdd <= SMOOTH_MDD:
            continue
        years = (nav.days[j] - nav.days[i]) / 365.2425
        dr = v[1:] / v[:-1] - 1
        rows.append({
            "code": row.code, "name": row.name, "group": row.group,
            "product_key": row.product_key,
            "period_return": ret, "annualized_return": (1 + ret) ** (1 / years) - 1,
            "D": mdd, "L": float(np.sqrt(np.mean(np.minimum(monthly, 0.0) ** 2))),
            "observed_log_cagr": float(np.log1p(ret) / years),
            "annual_se": annual_log_mean_se(monthly),
            "L_daily": float(np.sqrt(252.0 / len(dr) * np.sum(np.minimum(dr, 0.0) ** 2))),
            "nav_obj": nav, "nav_i": i, "nav_j": j})
    return pd.DataFrame(rows)


def score_fold(df, seed):
    d = eb_posteriors(df.dropna(subset=["annual_se", "observed_log_cagr"]).copy())
    d["mu_lower"] = np.exp(d["eb_log_cagr"] - EB_Z * d["eb_se"]) - 1.0
    d95 = {}
    for r in d.itertuples(index=False):
        v = r.nav_obj.value[r.nav_i: r.nav_j + 1]
        try:
            w = stationary_bootstrap(v, steps=BOOT_STEPS, scenarios=BOOT_SCENARIOS,
                                     block_length=10, seed=seed)
            d95[r.code] = float(np.quantile(max_drawdown(w), 0.95))
        except ValueError:
            pass
    d["D95"] = d["code"].map(d95)
    d["SR"] = pct(d["annualized_return"])
    d["SD"] = 100.0 - pct(d["D"])
    d["SL"] = 100.0 - pct(d["L"])
    d["SR_EB"] = pct(d["mu_lower"])
    d["SD95"] = 100.0 - pct(d["D95"])
    d["SL_daily"] = 100.0 - pct(d["L_daily"])
    d["S_M0"] = W3["return"] * d["SR"] + W3["drawdown"] * d["SD"] + W3["downside"] * d["SL"]
    d["S_M1"] = W3["return"] * d["SR_EB"] + W3["drawdown"] * d["SD"] + W3["downside"] * d["SL"]
    d["S_M2"] = W3["return"] * d["SR_EB"] + W3["drawdown"] * d["SD95"] + W3["downside"] * d["SL"]
    d["S_M3"] = W3["return"] * d["SR_EB"] + W3["drawdown"] * d["SD"] + W3["downside"] * d["SL_daily"]
    return d


def future_perf(paths, codes, t, hold_months):
    end = t + pd.DateOffset(months=hold_months)
    rets, series = [], []
    for c in codes:
        nav = paths.get(c)
        if nav is None:
            continue
        i, j = nav.right(t), nav.right(end)
        if i < 0 or j < 0 or j <= i or nav.days[j] - nav.day(end) > 10:
            continue
        v = nav.value[i:j + 1]
        rets.append(v[-1] / v[0] - 1)
        series.append(v / v[0])
    if not series:
        return None
    min_len = min(len(s) for s in series)
    port = np.mean([s[:min_len] for s in series], axis=0) * (1 - ROUNDTRIP_FEE)
    mdd = float(np.max(1 - port / np.maximum.accumulate(port)))
    dr = port[1:] / port[:-1] - 1
    downside = dr[dr < 0]
    ann = port[-1] ** (365.2425 / ((end - t).days)) - 1
    sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 0 else np.nan
    sortino = float(np.mean(dr) / np.std(downside) * np.sqrt(252)) if len(downside) > 1 else np.nan
    return {"net_return": float(port[-1] - 1), "annualized": float(ann), "mdd": mdd,
            "sharpe": sharpe, "sortino": sortino, "n_funds": len(series)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-11-20")
    ap.add_argument("--data-dir", default="data/all_funds")
    ap.add_argument("--output", default="reports/all_funds/models_20261120/oos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import re
    low_vol_re = re.compile("债|固收|绝对收益|货币|FOF-稳健")
    catalog = load_catalog(Path(args.data_dir) / "fund_list.csv")
    paths, _ = load_paths(catalog, Path(args.data_dir), asof)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    folds = []
    for t_str in FORMATION_DATES:
        t = pd.Timestamp(t_str)
        if t + pd.DateOffset(months=HOLD_MONTHS) > asof:
            continue
        df = fold_features(catalog, paths, t, low_vol_re)
        if df.empty:
            print(t_str, "empty fold")
            continue
        d = score_fold(df, args.seed)
        row = {"formation": t_str, "candidates": int(len(d))}
        for model in ("S_M0", "S_M1", "S_M2", "S_M3"):
            top = d.dropna(subset=[model]).sort_values(model, ascending=False)
            top = top.drop_duplicates("product_key").head(10)
            perf = future_perf(paths, top["code"].tolist(), t, HOLD_MONTHS)
            row[model] = perf
            top[["code", "name", model]].to_csv(out / f"fold_{t_str}_{model}.csv", index=False)
        folds.append(row)
        print(t_str, "candidates:", row["candidates"],
              {m[2:]: round(f["net_return"] * 100, 2) for m, f in row.items()
               if m.startswith("S_") and f}, flush=True)

    # 汇总：配对差 + 小样本 t 置信区间
    summary = {}
    for model in ("S_M1", "S_M2", "S_M3"):
        dr, dm = [], []
        for f in folds:
            if f.get(model) and f.get("S_M0"):
                dr.append(f[model]["net_return"] - f["S_M0"]["net_return"])
                dm.append(f[model]["mdd"] - f["S_M0"]["mdd"])
        n = len(dr)
        if n >= 2:
            from scipy import stats as sst
            tr = sst.t.interval(0.95, n - 1, loc=np.mean(dr), scale=sst.sem(dr))
            tm = sst.t.interval(0.95, n - 1, loc=np.mean(dm), scale=sst.sem(dm))
        else:
            tr = tm = (np.nan, np.nan)
        summary[model] = {"folds": n, "mean_delta_R": float(np.mean(dr)) if n else None,
                          "ci95_delta_R": [float(x) for x in tr],
                          "mean_delta_MDD": float(np.mean(dm)) if n else None,
                          "ci95_delta_MDD": [float(x) for x in tm],
                          "win_rate_R": float(np.mean([x > 0 for x in dr])) if n else None}
    result = {"asof": args.asof, "git_sha": sha, "formation_dates": FORMATION_DATES,
              "hold_months": HOLD_MONTHS, "roundtrip_fee": ROUNDTRIP_FEE,
              "boot_scenarios": BOOT_SCENARIOS, "seed": args.seed,
              "limitations": ["M0 degraded to 3-factor (no SA) - shock window not PIT-definable",
                              "current catalog only: survivorship bias on absolute levels",
                              "fees approximated at 10bp roundtrip, not PIT verified"],
              "folds": folds, "summary_vs_M0": summary}
    with open(out / "oos_result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
