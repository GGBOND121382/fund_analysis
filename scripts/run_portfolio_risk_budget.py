#!/usr/bin/env python3
"""P3 组合层 4% 风险预算多资产仓位研究（与基金品质排名分离）。

资产池：M0 主榜（3y）中 THS 可交易的债券/宽基指数/黄金基金 + 现金（国债 1Y）。
相关性处理：对 pooled 日收益矩阵做"向量版" stationary bootstrap（同一时间
下标重采样全部资产），保留横截面相关性；不把单基金 4% 预算相加当组合风控。
目标检验：P95 情景 MDD<=4% 且历史窗口混合 MDD<=4% 约束下，预期年化
（M1 mu_lower 保守口径）能否达到 5%。输出有效前沿与 3 折 walk-forward。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_industry_fund_ranking import load_catalog, load_paths  # noqa: E402
from fundkit.discovery_risk_rank import maximum_weight_for_budget  # noqa: E402
from fundkit.opportunity_rank import (OpportunityInput, buy_hold_wealth,  # noqa: E402
                                      cash_paths, max_drawdown,
                                      optimize_opportunity,
                                      stationary_bootstrap)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "all_funds"
RANK = ROOT / "reports" / "all_funds" / "models_20261120" / "full_ranking_m1_3y.csv"
THS = ROOT / "reports" / "all_funds" / "comprehensive_20261120" / "ths_tradeable_status_3y.csv"
FACTORS = ROOT / "data" / "factors" / "daily.csv"
EB_Z = 0.67448975
STEPS = 126  # 6 个月约 126 个交易日
FEE = 0.001  # roundtrip，按 P2 统一口径
SHOCK = {"债券": 0.03, "指数": 0.15, "黄金": 0.10}  # 研究假设，非评级结论


def joint_stationary_bootstrap(ret_matrix, *, steps, scenarios, block_length, seed):
    """向量版 stationary bootstrap：同一时间下标采样所有资产，保留相关性。

    与 fundkit.opportunity_rank.stationary_bootstrap 同一算法族，仅扩展为
    多资产同步采样；输入为 T x F 的日对数收益矩阵（已按公共交易日对齐）。
    """
    x = np.asarray(ret_matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 65 or x.shape[1] < 1:
        raise ValueError("invalid joint return matrix")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite returns")
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(n, size=scenarios)
    out = np.empty((scenarios, steps, x.shape[1]), dtype=float)
    restart = 1.0 / block_length
    for t in range(steps):
        if t:
            idx = np.where(rng.random(scenarios) < restart,
                           rng.integers(n, size=scenarios), (idx + 1) % n)
        out[:, t, :] = x[idx]
    paths = np.exp(np.cumsum(out, axis=1))
    return np.concatenate([np.ones((scenarios, 1, x.shape[1])), paths], axis=1)


def build_pool(asof, n_bond=6, n_equity=3, n_gold=2):
    rank = pd.read_csv(RANK, dtype={"code": str})
    ths = pd.read_csv(THS, dtype={"code": str, "buy": str})
    gold_st = Path(ROOT / "reports" / "portfolio_risk_budget_20261120"
                   / "ths_gold_status.csv")
    if gold_st.is_file():
        ths = pd.concat([ths, pd.read_csv(gold_st, dtype={"code": str, "buy": str})],
                        ignore_index=True).drop_duplicates("code")
    ok = set(ths.loc[ths["buy"].eq("1")
                     & ths["sgstat"].isin(["开放", "限制大额"]), "code"])
    rank = rank.loc[rank.code.isin(ok)].copy()
    BOND_GROUPS = ("中长纯债", "短债/中短债", "一级债基", "二级债基",
                   "利率债", "信用债", "指数型-固收", "偏债混合")
    bond = rank.loc[rank.group.isin(BOND_GROUPS)]
    bond = bond.loc[~bond.name.str.contains("可转债|转债", na=False)]
    bond = bond.nlargest(n_bond, "SCORE_M0")
    eq = rank.loc[rank.group.str.contains("指数型-股票", na=False)]
    eq = eq.loc[eq.name.str.contains("沪深300|中证500|中证A500|中证800", na=False)]
    eq = eq.nlargest(n_equity, "SCORE_M0")
    gold = rank.loc[rank.name.str.contains("黄金", na=False)].nlargest(n_gold, "SCORE_M0")
    pool = pd.concat([bond, eq, gold])
    pool["asset_class"] = (["债券"] * len(bond) + ["指数"] * len(eq)
                           + ["黄金"] * len(gold))
    return pool[["code", "name", "group", "asset_class", "SCORE_M0",
                 "mu_lower", "D95", "D", "observed_log_cagr", "annual_se"]]


def common_window(paths, codes, start, end):
    """公共交易日对齐；任何基金缺口的日期整体剔除（严禁补零/前值填充）。"""
    series = {}
    for c in codes:
        nav = paths[c]
        m = (nav.dates >= start) & (nav.dates <= end)
        if m.sum() < 100:
            raise ValueError(f"{c}: insufficient obs in window")
        series[c] = pd.Series(nav.value[m], index=nav.dates[m])
    df = pd.DataFrame(series).dropna()
    if len(df) < 250:
        raise ValueError(f"common window too short: {len(df)} days")
    gross = df / df.iloc[0]
    logret = np.diff(np.log(df.to_numpy()), axis=0)
    return df.index, gross, logret


def per_fund_mu(df):
    """trailing 窗口 obs_log_cagr 与 annual_se（月收益口径，同 M1 定义）。"""
    out = {}
    for c in df.columns:
        s = df[c].resample("ME").last().dropna()
        lr = np.diff(np.log(s.to_numpy()))
        years = (df[c].index[-1] - df[c].index[0]).days / 365.2425
        obs = np.log(df[c].iloc[-1] / df[c].iloc[0]) / years
        se = float(12 * np.std(lr, ddof=1) / np.sqrt(len(lr)))
        out[c] = (obs, se, float(np.expm1(obs - EB_Z * se)))
    return out


def optimize_weights(paths, cash, mu_ann, hist_gross, budget, *, samples=4000,
                     seed=42, cash_rate=0.0):
    """Dirichlet 权重搜索：max sum(w*mu)-fee s.t. P95MDD<=B 且历史混合MDD<=B。

    paths: S x T x F 基金情景路径；cash: S x T 现金路径。
    历史约束使用含现金的真实混合财富（现金按无风险利率单调累积）。
    """
    s_, t_, f = paths.shape
    rng = np.random.default_rng(seed)
    w_list = [np.eye(f + 1)]  # 角点（含纯现金）
    w_list.append(rng.dirichlet(np.ones(f + 1), size=samples))
    w_list.append(rng.dirichlet(np.full(f + 1, 3.0), size=samples))  # 偏分散
    W = np.concatenate(w_list).reshape(-1, f + 1)
    hist_M = hist_gross.to_numpy()  # D x F
    days = (hist_gross.index - hist_gross.index[0]).days.to_numpy(dtype=float)
    hist_cash = (1 + cash_rate) ** (days / 365.2425)
    entry_fee = exit_fee = FEE / 2.0
    hist_wealth = (hist_M @ (W[:, 1:] * (1 - entry_fee)).T
                   + np.outer(hist_cash, W[:, 0]))
    hist_wealth[-1] -= exit_fee * (hist_M[-1] @ W[:, 1:].T)
    hist_wealth = np.vstack([np.ones(len(W)), hist_wealth])
    hist_mdd = np.max(1 - hist_wealth / np.maximum.accumulate(hist_wealth, axis=0),
                      axis=0)
    best = None
    for lo in range(0, len(W), 50):
        wk = W[lo:lo + 50]  # K x (F+1)
        fund_gross = np.einsum("kf,stf->kst", wk[:, 1:], paths)
        wealth = wk[:, :1, None] * cash[None] + (1 - entry_fee) * fund_gross
        wealth[:, :, -1] -= exit_fee * fund_gross[:, :, -1]
        wealth = np.concatenate([np.ones((len(wk), s_, 1)), wealth], axis=2)
        mdd = np.max(1 - wealth / np.maximum.accumulate(wealth, axis=2), axis=2)
        p95 = np.quantile(mdd, 0.95, axis=1)
        years = (t_ - 1) / 252.0
        if np.any(mu_ann <= -1):
            raise ValueError("annual forecast must exceed -100%")
        growth = (1 + mu_ann) ** years
        horizon_wealth = wk[:, 0] * growth[0] + (wk[:, 1:] @ growth[1:]) * (1 - FEE)
        ret = horizon_wealth ** (1.0 / years) - 1.0
        feas = (p95 <= budget + 1e-12) & (hist_mdd[lo:lo + 50] <= budget + 1e-12)
        for j in np.where(feas)[0]:
            if best is None or ret[j] > best["expected_annual"]:
                best = dict(expected_annual=float(ret[j]), p95_mdd=float(p95[j]),
                            hist_mdd=float(hist_mdd[lo + j]),
                            weights=wk[j].tolist())
    return best


def walk_forward(pool, paths, cash_rate, formation_dates, budget=0.04, seed=42):
    """Block historical claims when the input pool is selected using 2026-11 data.

    PIT historical universe and channel eligibility are absent. The legacy
    backtest used today's selected winners in past folds (lookahead bias).
    """
    return pd.DataFrame([
        dict(formation=str(pd.Timestamp(fd).date()),
             status="PIT_SELECTION_DATA_BLOCKED",
             reason="Historical selection needs dated fund universe and trade eligibility")
        for fd in formation_dates
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-11-20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", type=int, default=2000)
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof)
    outdir = ROOT / "reports" / "portfolio_risk_budget_20261120"
    outdir.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    fac = pd.read_csv(FACTORS, parse_dates=["date"])
    fac = fac.loc[fac.date <= asof].dropna(subset=["risk_free_annual"])
    cash_rate = float(fac.risk_free_annual.iloc[-1])
    print(f"cash rate ({fac.date.iloc[-1].date()}): {cash_rate:.4%}")

    pool = build_pool(asof)
    print(f"pool: {len(pool)} funds")
    catalog = load_catalog(DATA / "fund_list.csv", codes=list(pool.code))
    paths, audit = load_paths(catalog, DATA, asof)
    missing = audit.loc[~audit.valid]
    if len(missing):
        print("MISSING NAV:", missing.code.tolist())
        pool = pool.loc[pool.code.isin(paths)].reset_index(drop=True)
    pool.to_csv(outdir / "pool.csv", index=False)

    start = asof - pd.DateOffset(years=3)
    dates, gross, logret = common_window(paths, list(pool.code), start, asof)
    print(f"common window: {dates[0].date()}..{dates[-1].date()}, {len(dates)} days")
    corr = pd.DataFrame(logret, columns=pool.code).corr()
    corr.to_csv(outdir / "correlation.csv")

    # 1) 单基金 4% 预算上限（仅参考，不可相加）
    caps = []
    for c in pool.code:
        v = gross[c].to_numpy()
        caps.append(dict(code=c, standalone_cap_4pct=maximum_weight_for_budget(v, 0.04),
                         hist_mdd=float(np.max(1 - v / np.maximum.accumulate(v)))))
    caps = pd.DataFrame(caps)
    caps.to_csv(outdir / "standalone_caps.csv", index=False)

    # 2) 每类代表基金的 optimize_opportunity（单资产口径）
    opp_rows = []
    for cls, g in pool.groupby("asset_class"):
        rep = g.iloc[0]
        nav = paths[rep.code]
        m = (nav.dates >= start) & (nav.dates <= asof)
        v = nav.value[m]
        sc = stationary_bootstrap(v, steps=STEPS, scenarios=args.scenarios,
                                  block_length=10, seed=args.seed)
        cash = cash_paths(cash_rate, steps=STEPS, scenarios=args.scenarios)
        setting = OpportunityInput(forecast_annual=float(rep.mu_lower),
                                   cash_annual=cash_rate, fee_rate_roundtrip=FEE,
                                   economic_shock_loss=SHOCK[cls], decision_months=6)
        for r in optimize_opportunity(sc, cash, setting):
            opp_rows.append(dict(asset_class=cls, rep_code=rep.code,
                                 rep_name=rep["name"], **r))
    pd.DataFrame(opp_rows).to_csv(outdir / "opportunity_rows.csv", index=False)

    # 3) 联合情景组合优化（保留相关性）
    sc_joint = joint_stationary_bootstrap(logret, steps=STEPS,
                                          scenarios=args.scenarios,
                                          block_length=10, seed=args.seed)
    cash_joint = cash_paths(cash_rate, steps=STEPS, scenarios=args.scenarios)
    mu_ann = np.array([cash_rate] + pool.mu_lower.tolist())
    frontier = []
    for budget in (0.01, 0.02, 0.04, 0.08):
        best = optimize_weights(sc_joint, cash_joint, mu_ann, gross, budget,
                                samples=4000, seed=args.seed, cash_rate=cash_rate)
        row = dict(budget=budget)
        if best is None:
            row.update(status="NO_FEASIBLE_WEIGHT")
        else:
            row.update(status="OK", expected_annual=best["expected_annual"],
                       p95_mdd=best["p95_mdd"], hist_mdd=best["hist_mdd"],
                       cash_w=best["weights"][0],
                       **{f"w_{c}": round(x, 4)
                          for c, x in zip(pool.code, best["weights"][1:]) if x > 0.005})
        frontier.append(row)
        print("frontier", row.get("budget"), row["status"],
              row.get("expected_annual"))
    frontier = pd.DataFrame(frontier)
    frontier.to_csv(outdir / "frontier.csv", index=False)

    # 4) 朴素等权对照（相关性影响演示）
    eq_w = np.full(len(pool), 1.0 / len(pool))
    wealth_eq = (1 - FEE / 2) * np.einsum("f,stf->st", eq_w, sc_joint)
    wealth_eq[:, -1] -= (FEE / 2) * np.einsum("f,sf->s", eq_w, sc_joint[:, -1, :])
    wealth_eq = np.concatenate([np.ones((args.scenarios, 1)), wealth_eq], axis=1)
    eq_p95 = float(np.quantile(max_drawdown(wealth_eq), 0.95))
    hist_eq = (1 - FEE / 2) * (gross.to_numpy() @ eq_w)
    hist_eq[-1] -= (FEE / 2) * (gross.to_numpy()[-1] @ eq_w)
    hist_eq = np.r_[1.0, hist_eq]
    eq_hist_mdd = float(np.max(1 - hist_eq / np.maximum.accumulate(hist_eq)))
    naive = dict(equal_weight_p95_mdd=eq_p95, equal_weight_hist_mdd=eq_hist_mdd,
                 note="等权组合情景P95回撤：相关性使组合风险不可由单基金预算相加推断")

    # 5) walk-forward（3 折，说明性）
    wf = walk_forward(pool, paths, cash_rate,
                      ["2025-05-20", "2025-11-20", "2026-05-20"], seed=args.seed)
    wf.to_csv(outdir / "walkforward.csv", index=False)
    print(wf.to_string())

    run = dict(asof=str(asof.date()), git_sha=git_sha, seed=args.seed,
               scenarios=args.scenarios, cash_rate=cash_rate,
               pool_size=len(pool), common_days=len(dates),
               naive_equal_weight=naive,
               target="conditional forecast annual return>=5% & scenario MDD<=4%",
               forecast_feasible_4pct=bool((frontier.budget.eq(0.04)
                                   & frontier.status.eq("OK")
                                   & frontier.expected_annual.ge(0.05)).any()),
               oos_status="PIT_SELECTION_DATA_BLOCKED")
    (outdir / "run_summary.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
