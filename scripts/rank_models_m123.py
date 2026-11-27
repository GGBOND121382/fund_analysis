#!/usr/bin/env python3
"""P1: M1/M2/M3 模型实验（M0 固定对照，不改 M0 任何输出）。

候选从 M0 SCORED 出发，重查净值质量，所有模型在共同有效候选集计算百分位。
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
from fundkit.industry_rank import HORIZONS, window_dates, nav_quality_issue  # noqa: E402
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
    """Research-only conditional EB; no pooling when dispersion is unidentified.

    Use one deterministic representative per product to fit the prior.
    Use the same complete-month sample for the observation and its SE where
    available. The SE is conditional on the fitted prior; no independence
    assumption across peer funds is used to invent a prior standard error.
    """
    d = df.copy()
    obs_col = "eb_observed_log_mean" if "eb_observed_log_mean" in d else "observed_log_cagr"
    d["eb_log_cagr"] = np.nan
    d["eb_se"] = np.nan
    d["peer_count"] = 0
    d["eb_tau2"] = np.nan
    d["eb_weight"] = np.nan
    d["eb_status"] = "INVALID_INPUT"
    valid = np.isfinite(d[obs_col]) & np.isfinite(d["annual_se"]) & d["annual_se"].ge(0)
    for group, g in d.loc[valid].groupby("group", dropna=False):
        peers = g.sort_values("code") if "code" in g else g
        if "product_key" in peers:
            peers = peers.copy()
            keys = peers.product_key.fillna("").astype(str)
            peers["_peer_key"] = keys.where(keys.str.strip().ne(""), "share:" + peers.code.astype(str))
            peers = peers.drop_duplicates("_peer_key")
        n = len(peers)
        x = g[obs_col].to_numpy(dtype=float)
        se = g["annual_se"].to_numpy(dtype=float)
        # Never replace a thin peer group with a cross-asset global median.
        tau2 = max(0.0, float(peers[obs_col].var(ddof=1) -
                               np.mean(peers.annual_se.to_numpy() ** 2))) if n >= MIN_GROUP else 0.0
        if n < MIN_GROUP or tau2 <= 1e-18:
            # Zero dispersion may genuinely occur. It does not establish that
            # a category-constant forecast is useful for ranking. Explicit
            # no-pooling fallback, not an artificial positive variance floor.
            post, post_se = x, se
            weight = np.ones(len(g))
            status = "NO_POOL_SMALL_GROUP" if n < MIN_GROUP else "NO_POOL_ZERO_DISPERSION"
        else:
            prior = float(peers[obs_col].median())
            se2 = se ** 2
            weight = tau2 / (tau2 + se2)
            post = prior + weight * (x - prior)
            post_se = np.sqrt(weight * se2)
            status = "EB_CONDITIONAL"
        d.loc[g.index, "eb_log_cagr"] = post
        d.loc[g.index, "eb_se"] = post_se
        d.loc[g.index, "peer_count"] = n
        d.loc[g.index, "eb_tau2"] = tau2
        d.loc[g.index, "eb_weight"] = weight
        d.loc[g.index, "eb_status"] = status
    d["mu_lower"] = np.expm1(d["eb_log_cagr"] - EB_Z * d["eb_se"])
    return d


def process_period(period, asof, inp, paths, boot_scenarios=2000, boot_block=10, seed=42):
    df = pd.read_csv(inp / f"all_scored_{period}.csv", low_memory=False, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    cand = df[df["status"] == "SCORED"].copy().reset_index(drop=True)
    n_months, _ = HORIZONS[period]
    begin, period_m, complete = window_dates(asof, n_months)

    # --- 逐只计算 M1/M2/M3 所需序列指标 ---
    monthly_stats, monthly_means, daily_stats, d95, quality = {}, {}, {}, {}, {}
    for _, r in cand.iterrows():
        nav = paths.get(r["code"])
        code = r["code"]
        if nav is None:
            quality[code] = "MISSING_NAV"
            continue
        i, j = nav.right(begin), nav.right(asof)
        if not 0 <= i < j:
            quality[code] = "INVALID_WINDOW"
            continue
        v = nav.value[i:j + 1]
        issue = nav_quality_issue(v)
        if issue:
            quality[code] = issue
            continue
        monthly = nav.monthly(asof, period_m, complete)
        if monthly is None or len(monthly) < 4:
            quality[code] = "INSUFFICIENT_MONTHS"
            continue
        monthly_stats[code] = annual_log_mean_se(monthly)
        monthly_means[code] = float(12 * np.mean(np.log1p(monthly)))
        if 0 <= i < j:
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

    cand["model_quality_status"] = cand["code"].map(quality).fillna("OK")
    cand["annual_se"] = cand["code"].map(monthly_stats)
    cand["eb_observed_log_mean"] = cand["code"].map(monthly_means)
    cand["observed_log_cagr"] = np.log1p(cand["period_return"]) / (
        (pd.to_datetime(cand["window_end"]) - pd.to_datetime(cand["window_start"])).dt.days / 365.2425)
    ok = cand.dropna(subset=["annual_se", "observed_log_cagr"]).copy()
    ok = eb_posteriors(ok)
    cand = cand.merge(ok[["code", "eb_log_cagr", "eb_se", "peer_count", "mu_lower", "eb_tau2", "eb_weight", "eb_status"]],
                      on="code", how="left")
    cand["L_daily"] = cand["code"].map(daily_stats)
    cand["D95"] = cand["code"].map(d95)

    # Common complete candidate universe for every model, including M0.
    eligible = cand.model_quality_status.eq("OK") & np.isfinite(cand[["mu_lower", "D95", "L_daily", "R", "D", "L", "Aret", "Add"]]).all(axis=1)
    cand["SCORE_M0_original"] = cand["SCORE_M0"]
    for col, raw, reverse in (("SR", "R", False), ("SD", "D", True), ("SL", "L", True),
                               ("SR_EB", "mu_lower", False), ("SD95", "D95", True), ("SL_daily", "L_daily", True)):
        p = pct(cand[raw].where(eligible))
        cand[col] = 100.0 - p if reverse else p
    cand["SA"] = 0.5*pct(cand["Aret"].where(eligible)) + 0.5*(100-pct(cand["Add"].where(eligible)))
    cand["SCORE_M0"] = W["return"]*cand["SR"] + W["drawdown"]*cand["SD"] + W["shock"]*cand["SA"] + W["downside"]*cand["SL"]
    cand["SCORE_M1"] = W["return"] * cand["SR_EB"] + W["drawdown"] * cand["SD"] \
        + W["shock"] * cand["SA"] + W["downside"] * cand["SL"]
    cand["SCORE_M2"] = W["return"] * cand["SR_EB"] + W["drawdown"] * cand["SD95"] \
        + W["shock"] * cand["SA"] + W["downside"] * cand["SL"]
    cand["SCORE_M3"] = W["return"] * cand["SR_EB"] + W["drawdown"] * cand["SD"] \
        + W["shock"] * cand["SA"] + W["downside"] * cand["SL_daily"]
    coverage = {
        "candidates_m0": int(len(cand)),
        "eligible_common": int(eligible.sum()),
        "quality_status_counts": cand.model_quality_status.value_counts().to_dict(),
        "eb_status_counts": cand.eb_status.fillna("NOT_FITTED").value_counts().to_dict(),
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
        keep = ["code", "name", "group", "product_key", "fund_type", "pricing_basis", "periodic_status",
                "period_return", "R", "D", "L", "Aret", "Add", "SCORE_M0",
                "annual_se", "observed_log_cagr", "eb_observed_log_mean", "eb_log_cagr", "eb_se", "peer_count",
                "eb_tau2", "eb_weight", "eb_status", "model_quality_status", "SCORE_M0_original",
                "mu_lower", "L_daily", "D95",
                "SR", "SD", "SL", "SA", "SR_EB", "SD95", "SL_daily",
                "SCORE_M1", "SCORE_M2", "SCORE_M3", "risk_status", "trade_status"]
        keep = [c for c in keep if c in cand.columns]
        cand_out = cand[keep].rename(columns={"R": "annualized_return"})
        cand_out.to_csv(out / f"model_candidate_audit_{period}.csv", index=False)
        for model in ("M1", "M2", "M3"):
            col = f"SCORE_{model}"
            ranked = cand_out.dropna(subset=[col]).sort_values([col, "code"], ascending=[False, True]).reset_index(drop=True)
            ranked.insert(0, f"rank_{model}", range(1, len(ranked) + 1))
            for k in (10, 20, 50):
                ranked.drop_duplicates("product_key").head(k).to_csv(out / f"top{k}_{model.lower()}_{period}.csv", index=False)
            ranked.to_csv(out / f"full_ranking_{model.lower()}_{period}.csv", index=False)
            audit = {"model": model, "horizon": period, "asof": args.asof, "git_sha": sha,
                     "weights": W, "coverage": coverage,
                     "m1_note": "原始收益条件EB研究版：同产品一票、零离散/小样本不收缩、同月样本；非基准调整，非未来收益预测",
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
