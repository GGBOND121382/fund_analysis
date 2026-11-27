#!/usr/bin/env python3
"""P2 历史样本外模型比较：M0-noSA / M1 / M2 / M3。

设计与已知局限（如实声明）：
- 历史形成日滚动：半年间隔 7 个不重叠折（持有期 6 个月，不保证统计独立）；
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
from fundkit.industry_rank import NavPath, window_dates, nav_quality_issue  # noqa: E402
from fundkit.opportunity_rank import stationary_bootstrap, max_drawdown  # noqa: E402
from rank_models_m123 import eb_posteriors, pct, BOOT_STEPS, EB_Z, annual_log_mean_se  # noqa: E402

W3 = {"return": 0.40 / 0.85, "drawdown": 0.35 / 0.85, "downside": 0.10 / 0.85}
SMOOTH_MDD = 0.0002
ROUNDTRIP_FEE = 0.001
FORMATION_DATES = ["2023-05-19", "2023-11-20", "2024-05-20", "2024-11-20",
                   "2025-05-20", "2025-11-20", "2026-05-20"]
HOLD_MONTHS = 6
BOOT_SCENARIOS = 500
ROOT = Path(__file__).resolve().parents[1]


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
        if nav_quality_issue(v):
            continue
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
            "eb_observed_log_mean": float(12 * np.mean(np.log1p(monthly))),
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
    # All models use identical finite-feature cohorts, including the control.
    d = d.loc[np.isfinite(d[["mu_lower", "D95", "L_daily", "D", "L", "annualized_return"]]).all(axis=1)].copy()
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
    d["S_RISK_D95"] = W3["return"] * d["SR"] + W3["drawdown"] * d["SD95"] + W3["downside"] * d["SL"]
    d["S_RISK_DAILY"] = W3["return"] * d["SR"] + W3["drawdown"] * d["SD"] + W3["downside"] * d["SL_daily"]
    return d


def future_perf(paths, codes, t, hold_months, names=None, market_days=None):
    """PIT buy-and-hold equal-weight TOP10, with calendar-aligned NAV paths.

    Funds must all cover the same formation/exit window. No positional
    averaging, silent fund removal, implicit price forward-fill or truncation.
    Fees are a 10bp *roundtrip*, split equally between entry and exit.
    Gaps are judged against the market calendar (union of all funds' NAV
    dates): a market-wide holiday is NOT a data hole; a fund missing a day
    on which the market traded IS. On rejection returns
    {"status": "REJECTED", "reason": ..., "fund": ...} so a fold failure is
    never silent.
    """
    def reject(reason, fund=None):
        name = (names or {}).get(fund) if fund else None
        return {"status": "REJECTED", "reason": reason,
                "fund": fund, "fund_name": name}

    end = t + pd.DateOffset(months=hold_months)
    if len(codes) != 10 or len(set(codes)) != 10:
        return reject("TOP10 list short or duplicated", None)
    series = {}
    for code in codes:
        nav = paths.get(code)
        if nav is None:
            return reject("nav_path_missing", code)
        # Formation-day NAV is only known after the subscription cutoff.
        i, j = nav.right(t) + 1, nav.right(end)
        if i >= len(nav.days) or j <= i:
            return reject("no_nav_spanning_formation_exit", code)
        if nav.days[i] - nav.day(t) > 10 or nav.day(end) - nav.days[j] > 10:
            return reject("formation_or_exit_nav_gap_gt_10d", code)
        if market_days is not None:
            have = set(nav.days[i:j + 1].tolist())
            missing = [int(d) for d in market_days if nav.day(t) < int(d) <= nav.day(end) and int(d) not in have]
            if missing:
                d0 = pd.Timestamp(np.datetime64(int(missing[0]), "D"))
                return reject(f"missing_market_day:{d0.date()}", code)
        elif np.any(np.diff(nav.days[i:j + 1]) > 10):
            return reject("interior_nav_gap_gt_10d", code)
        series[code] = pd.Series(nav.value[i:j + 1], index=nav.dates[i:j + 1])
    # Common observation calendar is essential: equal array positions need not
    # correspond to the same date for different funds.
    aligned = pd.concat(series, axis=1, join="inner").sort_index()
    if len(aligned) < 60:
        return reject("aligned_calendar_lt_60_days")
    if (aligned.index[0] - pd.Timestamp(t)).days > 10 or \
            (pd.Timestamp(end) - aligned.index[-1]).days > 10:
        return reject("aligned_window_edge_gt_10d")
    if market_days is None and \
            np.any(np.diff(aligned.index.values.astype("datetime64[D]").astype("int64")) > 10):
        return reject("aligned_interior_gap_gt_10d")
    gross = aligned.div(aligned.iloc[0]).to_numpy(dtype=float)
    entry_fee = ROUNDTRIP_FEE / 2.0
    exit_fee = ROUNDTRIP_FEE - entry_fee
    # Initial capital is 1. Starting investment is reduced by entry fee;
    # exiting the fund incurs the remaining half on the terminal observation.
    port = np.mean(gross, axis=1) * (1.0 - entry_fee)
    port[-1] *= 1.0 - exit_fee
    wealth = np.r_[1.0, port]
    mdd = float(np.max(1.0 - wealth / np.maximum.accumulate(wealth)))
    dr = wealth[1:] / wealth[:-1] - 1.0
    # Zero hurdle for both ratios; report it explicitly, not a risk-free Sharpe.
    daily_downside = float(np.sqrt(np.mean(np.minimum(dr, 0.0) ** 2)))
    sd = float(np.std(dr, ddof=1)) if len(dr) > 1 else np.nan
    ann = float(wealth[-1] ** (365.2425 / ((aligned.index[-1] - aligned.index[0]).days)) - 1.0)
    sharpe = float(np.mean(dr) / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    sortino = (float(np.mean(dr) / daily_downside * np.sqrt(252.0))
               if daily_downside > 0 else np.nan)
    return {"net_return": float(wealth[-1] - 1.0), "annualized": ann,
            "mdd": mdd, "sharpe": sharpe, "sortino": sortino,
            "n_funds": len(codes), "start_date": str(aligned.index[0].date()),
            "end_date": str(aligned.index[-1].date()),
            "risk_ratio_hurdle": "zero_daily_return"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-11-20")
    ap.add_argument("--data-dir", default="data/all_funds")
    ap.add_argument("--output", default="reports/all_funds/models_20261120/oos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--universe", choices=["all", "defensive_bond"], default="all")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof)
    out = Path(args.output)
    if args.universe != "all":
        out = out / args.universe
    out.mkdir(parents=True, exist_ok=True)

    import re
    low_vol_re = re.compile("债|固收|绝对收益|货币|FOF-稳健")
    catalog = load_catalog(Path(args.data_dir) / "fund_list.csv")
    if args.universe == "defensive_bond":
        catalog = catalog.loc[catalog.group.isin(["中长纯债", "短债/中短债", "利率债", "信用债", "指数型-固收"])
                              & ~catalog.name.str.contains(r"滚动|持有|定开|封闭|可转债|转债", na=False)].copy()
    paths, _ = load_paths(catalog, Path(args.data_dir), asof)
    fac = pd.read_csv(ROOT / "data" / "factors" / "daily.csv",
                      parse_dates=["date"])
    trading_days = pd.DatetimeIndex(
        fac.loc[fac.csi300_price_index.notna(), "date"])
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    folds = []
    for t_str in FORMATION_DATES:
        t = pd.Timestamp(t_str)
        if t + pd.DateOffset(months=HOLD_MONTHS) > asof:
            continue
        # 市场日历：沪深交易所真实交易日（factors 中沪深300有行情的日期）。
        # 不用全体基金净值日期并集——QDII/货币等周末披露会污染并集，
        # 导致"周六缺失"误判为数据洞；也不用纯日历日阈值——节假日误判。
        end = t + pd.DateOffset(months=HOLD_MONTHS)
        mdmask = (trading_days >= t) & (trading_days <= end)
        market_days = np.array(sorted(
            trading_days[mdmask].values.astype("datetime64[D]").astype("int64")),
            dtype=np.int64)
        df = fold_features(catalog, paths, t, low_vol_re)
        if df.empty:
            print(t_str, "empty fold")
            continue
        d = score_fold(df, args.seed)
        row = {"formation": t_str, "candidates": int(len(d))}
        for model in ("S_M0", "S_M1", "S_M2", "S_M3", "S_RISK_D95", "S_RISK_DAILY"):
            top = d.dropna(subset=[model]).sort_values([model, "code"], ascending=[False, True])
            top = top.drop_duplicates("product_key").head(10)
            perf = future_perf(paths, top["code"].tolist(), t, HOLD_MONTHS,
                               names=dict(zip(top["code"], top["name"])),
                               market_days=market_days)
            row[model] = perf
            top[["code", "name", model]].to_csv(out / f"fold_{t_str}_{model}.csv", index=False)
        folds.append(row)
        print(t_str, "candidates:", row["candidates"],
              {m[2:]: (round(f["net_return"] * 100, 2) if "net_return" in f
                       else f"REJECTED:{f['reason']}")
               for m, f in row.items()
               if m.startswith("S_") and f}, flush=True)

    # 汇总：配对差 + 小样本 t 置信区间
    summary = {}
    for model in ("S_M1", "S_M2", "S_M3", "S_RISK_D95", "S_RISK_DAILY"):
        dr, dm = [], []
        for f in folds:
            fm, f0 = f.get(model) or {}, f.get("S_M0") or {}
            if "net_return" in fm and "net_return" in f0:
                dr.append(fm["net_return"] - f0["net_return"])
                dm.append(fm["mdd"] - f0["mdd"])
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
    result = {"universe": args.universe, "execution": "first NAV strictly after formation", "asof": args.asof, "git_sha": sha, "formation_dates": FORMATION_DATES,
              "hold_months": HOLD_MONTHS, "roundtrip_fee": ROUNDTRIP_FEE,
              "boot_scenarios": BOOT_SCENARIOS, "seed": args.seed,
              "limitations": ["M0 degraded to 3-factor (no SA) - shock window not PIT-definable",
                              "current catalog only: survivorship bias on absolute levels",
                              "fees approximated at 10bp roundtrip, not PIT verified",
                              "OOS return requires all 10 funds to have aligned full-horizon NAV"],
              "folds": folds, "summary_vs_M0": summary}
    with open(out / "oos_result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
