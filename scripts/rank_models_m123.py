#!/usr/bin/env python3
"""P1: M1/M2/M3 模型实验（M0 固定对照，不改 M0 任何输出）。

候选集合与 M0 完全相同：all_scored_{horizon}.csv 中 status==SCORED 的份额。
窗口、asof、冲击窗与 M0 一致。四期限分别拟合，不使用形成日后数据。

- M1（原始收益 EB 研究版，非基准调整——本地无类别基准 PIT 净值历史）：
  月度对数收益 -> observed_log_cagr/annual_se -> 类内正态-正态 EB
  （peer median 先验，tau2 去偏，复用 fundkit.discovery_risk_rank._posteriors 思路）
  mu_lower = exp(eb_log_cagr - 0.67448975*eb_se) - 1
  S_M1 = 0.40*P(mu_lower) + 0.35*SD + 0.15*SA + 0.10*SL
- M2：窗口内日收益 stationary_bootstrap（复用 fundkit.opportunity_rank），
  126 交易日持有期，2000 情景，block=10（主），seed=42
  D95 = quantile(MDD_b, 0.95)；S_M2 = 0.40*SR_EB + 0.35*P_rev(D95) + 0.15*SA + 0.10*SL
- M3：L_daily = sqrt(252/n * sum(min(r,0)^2))
  S_M3 = 0.40*SR_EB + 0.35*SD + 0.15*SA + 0.10*P_rev(L_daily)

分位口径与 M0 一致：候选集内升序平均并列，N=1 给 50，缺失显式 NaN 不补零。
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
from fundkit.industry_rank import HORIZONS, window_dates  # noqa: E402
from fundkit.opportunity_rank import stationary_bootstrap, max_drawdown  # noqa: E402

W = {"return": 0.40, "drawdown": 0.35, "shock": 0.15, "downside": 0.10}
EB_Z = 0.67448975
BOOT_STEPS = 126  # 6 个月约 126 个交易日
MIN_GROUP = 10


def annual_log_mean_se(monthly):
    """SE of 12 * mean(monthly log returns), assuming independent months.

    Month-to-month dependence is not modeled here; this is an estimation
    uncertainty measure, not a future return volatility forecast.
    """
    x = np.asarray(monthly, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(x <= -1):
        raise ValueError("at least two finite monthly returns > -100% required")
    return float(12.0 * np.std(np.log1p(x), ddof=1) / np.sqrt(len(x)))


def pct(s):
    """M0 口径分位：升序平均并列，N=1 -> 50。"""
    n = s.notna().sum()
    if n == 0:
        return s * np.nan
    if n == 1:
        return s.where(s.isna(), 50.0)
    return 100.0 * (s.rank(method="average") - 1.0) / (n - 1.0)


def eb_posteriors(df):
    """类内 EB（研究版，无基准）。返回 eb_log_cagr, eb_se, peer_count。"""
    d = df.copy()
    global_prior = float(d["observed_log_cagr"].median())
    d["eb_log_cagr"] = np.nan
    d["eb_se"] = np.nan
    d["peer_count"] = 0
    for group, g in d.groupby("group", dropna=False):
        n = len(g)
        prior = float(g["observed_log_cagr"].median()) if n >= MIN_GROUP else global_prior
        tau2 = (max(0.0, float(g["observed_log_cagr"].var(ddof=1)
                             - np.mean(g["annual_se"].to_numpy() ** 2))) if n >= MIN_GROUP else 0.0)
        se2 = g["annual_se"].to_numpy(dtype=float) ** 2
        shrink = tau2 / (tau2 + se2 + 1e-18)
        post = prior + shrink * (g["observed_log_cagr"].to_numpy() - prior)
        post_se = np.sqrt(np.maximum(1e-12, tau2 * se2 / (tau2 + se2 + 1e-18)))
        prior_se = np.sqrt(np.mean(se2) / max(1, n))
        post_se = np.sqrt(post_se ** 2 + prior_se ** 2)
        if n < MIN_GROUP:
            post_se = np.maximum(post_se, g["annual_se"].to_numpy())
        d.loc[g.index, "eb_log_cagr"] = post
        d.loc[g.index, "eb_se"] = post_se
        d.loc[g.index, "peer_count"] = n
    d["mu_lower"] = np.exp(d["eb_log_cagr"] - EB_Z * d["eb_se"]) - 1.0
    return d


def process_period(period, asof, inp, paths, boot_scenarios=2000, boot_block=10, seed=42):
    df = pd.read_csv(inp / f"all_scored_{period}.csv", low_memory=False, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    cand = df[df["status"] == "SCORED"].copy().reset_index(drop=True)
    n_months, _ = HORIZONS[period]
    begin, period_m, complete = window_dates(asof, n_months)

    # --- 逐只计算 M1/M2/M3 所需序列指标 ---
    monthly_stats, daily_stats, d95 = {}, {}, {}
    for _, r in cand.iterrows():
        nav = paths.get(r["code"])
        if nav is None:
            continue
        monthly = nav.monthly(asof, period_m, complete)
        if monthly is not None and len(monthly) >= 4:
            monthly_stats[r["code"]] = annual_log_mean_se(monthly)
        i, j = nav.right(begin), nav.right(asof)
        if 0 <= i < j:
            v = nav.value[i:j + 1]
            dr = v[1:] / v[:-1] - 1
            if len(dr) >= 5:
                daily_stats[r["code"]] = float(
                    np.sqrt(252.0 / len(dr) * np.sum(np.minimum(dr, 0.0) ** 2)))
            try:
                wealth = stationary_bootstrap(v, steps=BOOT_STEPS,
                                              scenarios=boot_scenarios,
                                              block_length=boot_block, seed=seed)
                d95[r["code"]] = float(np.quantile(max_drawdown(wealth), 0.95))
            except ValueError:
                pass

    cand["annual_se"] = cand["code"].map(monthly_stats)
    cand["observed_log_cagr"] = np.log1p(cand["period_return"]) / (
        (pd.to_datetime(cand["window_end"]) - pd.to_datetime(cand["window_start"])).dt.days / 365.2425)
    ok = cand.dropna(subset=["annual_se", "observed_log_cagr"]).copy()
    ok = eb_posteriors(ok)
    cand = cand.merge(ok[["code", "eb_log_cagr", "eb_se", "peer_count", "mu_lower"]],
                      on="code", how="left")
    cand["L_daily"] = cand["code"].map(daily_stats)
    cand["D95"] = cand["code"].map(d95)

    # --- 打分（各模型在自己的可覆盖集内取分位） ---
    cand["SR_EB"] = pct(cand["mu_lower"])
    cand["SD95"] = 100.0 - pct(cand["D95"])
    cand["SL_daily"] = 100.0 - pct(cand["L_daily"])
    cand["SCORE_M1"] = W["return"] * cand["SR_EB"] + W["drawdown"] * cand["SD"] \
        + W["shock"] * cand["SA"] + W["downside"] * cand["SL"]
    cand["SCORE_M2"] = W["return"] * cand["SR_EB"] + W["drawdown"] * cand["SD95"] \
        + W["shock"] * cand["SA"] + W["downside"] * cand["SL"]
    cand["SCORE_M3"] = W["return"] * cand["SR_EB"] + W["drawdown"] * cand["SD"] \
        + W["shock"] * cand["SA"] + W["downside"] * cand["SL_daily"]
    coverage = {
        "candidates_m0": int(len(cand)),
        "m1_scored": int(cand["SCORE_M1"].notna().sum()),
        "m2_scored": int(cand["SCORE_M2"].notna().sum()),
        "m3_scored": int(cand["SCORE_M3"].notna().sum()),
        "eb_missing": int(cand["mu_lower"].isna().sum()),
        "d95_missing": int(cand["D95"].isna().sum()),
        "l_daily_missing": int(cand["L_daily"].isna().sum()),
    }
    return cand, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-11-20")
    ap.add_argument("--input", default="reports/all_funds/comprehensive_20261120")
    ap.add_argument("--data-dir", default="data/all_funds")
    ap.add_argument("--output", default="reports/all_funds/models_20261120")
    ap.add_argument("--scenarios", type=int, default=2000)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof)
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(Path(args.data_dir) / "fund_list.csv")
    paths, _ = load_paths(catalog, Path(args.data_dir), asof)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {}
    for period in ("5y", "3y", "1y", "6m"):
        cand, coverage = process_period(period, asof, inp, paths,
                                        args.scenarios, args.block, args.seed)
        keep = ["code", "name", "group", "fund_type", "pricing_basis", "periodic_status",
                "period_return", "R", "D", "L", "Aret", "Add", "SCORE_M0",
                "annual_se", "observed_log_cagr", "eb_log_cagr", "eb_se", "peer_count",
                "mu_lower", "L_daily", "D95",
                "SR", "SD", "SL", "SA", "SR_EB", "SD95", "SL_daily",
                "SCORE_M1", "SCORE_M2", "SCORE_M3", "risk_status", "trade_status"]
        keep = [c for c in keep if c in cand.columns]
        cand_out = cand[keep].rename(columns={"R": "annualized_return"})
        for model in ("M1", "M2", "M3"):
            col = f"SCORE_{model}"
            ranked = cand_out.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
            ranked.insert(0, f"rank_{model}", range(1, len(ranked) + 1))
            for k in (10, 20, 50):
                ranked.head(k).to_csv(out / f"top{k}_{model.lower()}_{period}.csv", index=False)
            ranked.to_csv(out / f"full_ranking_{model.lower()}_{period}.csv", index=False)
            audit = {"model": model, "horizon": period, "asof": args.asof, "git_sha": sha,
                     "weights": W, "coverage": coverage,
                     "m1_note": "原始收益EB研究版：类内peer median先验，非基准调整（本地无类别基准PIT净值）",
                     "bootstrap": {"scenarios": args.scenarios, "block": args.block,
                                   "seed": args.seed, "steps": BOOT_STEPS}}
            with open(out / f"model_audit_{model.lower()}_{period}.json", "w") as f:
                json.dump(audit, f, ensure_ascii=False, indent=1)
        summary[period] = coverage
        print(period, coverage, flush=True)
    with open(out / "run_summary.json", "w") as f:
        json.dump({"asof": args.asof, "git_sha": sha, "weights": W,
                   "bootstrap": {"scenarios": args.scenarios, "block": args.block,
                                 "seed": args.seed, "steps": BOOT_STEPS},
                   "periods": summary}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
