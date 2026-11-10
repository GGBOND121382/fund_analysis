"""第5步：滚动样本外反证（04_BACKTEST_VALIDATION.md / 06 第5步实现）。

结构（03 F 伪代码）：
1. 时间序列滚动折：train（调参记录）→ validation（风险优先选参）→ test（冻结后只跑一次）
2. 小维度预声明网格（禁止笛卡尔大爆炸；全部试验落盘 reports/oos_trials.csv）
3. 每个窗口起点用**明确标注的反事实组合**（快照权重×标准化本金，
   COUNTERFACTUAL_START_PORTFOLIO——不是用户真实历史）
4. B4/B5 的风格信号按窗口起点 as-of 重算（不得用今天的 style_report 回填历史）
5. 压力测试：块自助法组合 MDD 分布（P(MDD>5%) 及置信区间）+ 2022年11月债灾回放
6. 可行性：OOS 同时满足 CAGR≥4% 且 MDD≤5% 才可行，否则 NO_FEASIBLE_STRATEGY +
   风险—收益帕累托前沿（04/06 禁止为凑 4% 调高收益假设）
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from fundkit import policies, simulator, style
from fundkit.adapters import CST, LocalDataAdapter, load_account_snapshot
from tests import finance_math as fm

CAPITAL = 195123.58  # 标准化本金（快照总额），反事实起点
BUDGET_PER_DAY = 1005.0
MDD_BUDGET = 0.05
CAGR_TARGET = 0.04

FOLDS = [
    {"train": ("2023-06-01", "2025-06-01"), "val": ("2025-06-01", "2025-12-01"),
     "test": ("2025-12-01", "2026-06-01")},
    {"train": ("2024-01-01", "2026-01-01"), "val": ("2026-01-01", "2026-06-01"),
     "test": ("2026-06-01", "2026-11-09")},
]

# 预声明小网格（search_space.yaml 子集；每次试验记录）
GRID = [
    ("B0", {}),
    ("B1", {}),
    *[("B2", {"cash": cf}) for cf in (0.0, 0.05, 0.10)],
    *[("B3", {"weight_trigger": t, "destination_fraction": d})
      for t in (0.02, 0.05) for d in (0.25, 0.5)],
    *[("B4", {"pause_cdar95": c}) for c in (0.01, 0.02, 0.03)],
    *[("B5", {"internal_drawdown_margin": m, "fund_pause_drawdown": f})
      for m in (0.01, 0.02) for f in (0.02, 0.03)],
]


def _params_for(family: str, overrides: dict) -> dict:
    params = policies.load_params()
    if family == "B2":
        params["target_weights"] = {**params["target_weights"],
                                    "cash": overrides.get("cash", 0.05)}
    elif family == "B3":
        params["B3"] = {**params["B3"], **overrides}
    elif family == "B4":
        params["B4"] = {**params["B4"], **overrides}
    elif family == "B5":
        params["B5"] = {**params["B5"], **overrides}
    return params


def _style_signals_asof(adapter: LocalDataAdapter, codes, until) -> dict:
    """按窗口起点 as-of 重算风格信号（防未来信息，03 F / 04）。"""
    out = {}
    for c in codes:
        try:
            r = style.analyze_fund(adapter, c, until)
            out[c] = {"combined_shift_evidence": r["combined_shift_evidence"],
                      "risk_views": r["risk_views"],
                      "current_style_state": r["current_style_state"]}
        except Exception as e:  # noqa: BLE001
            out[c] = {"combined_shift_evidence": {"state": "STYLE_UNCERTAIN",
                                                  "reason": f"分析失败: {str(e)[:60]}"},
                      "risk_views": {}}
    return out


def _initial_lots(adapter: LocalDataAdapter, snap: dict, start: str,
                  capital: float = CAPITAL):
    """反事实起点组合：快照权重×标准化本金；成立日晚于起点的基金持有现金等待。

    COUNTERFACTUAL_START_PORTFOLIO：非用户真实历史持仓。
    """
    total_snap = sum(f["value_cny"] for f in snap["funds"])
    lots = {}
    start_ts = pd.Timestamp(start)
    for f in snap["funds"]:
        code = f["code"]
        value = capital * f["value_cny"] / total_snap
        try:
            df = adapter.nav([code], start_ts.tz_localize(CST) if start_ts.tzinfo is None
                             else start_ts)
            df = df[df["nav_date"] <= start_ts]
            if df.empty:
                continue  # 基金尚无数据：该份额留现金（由调用方处理）
            nav0 = float(df.iloc[-1]["adj_nav_total_return"])
        except Exception:  # noqa: BLE001
            continue
        lots[code] = [simulator.Lot(code, value / nav0, start_ts, simulated=True)]
    return lots


def _run_window(family, overrides, window, adapter, snap, sim_config_base,
                style_signals=None):
    params = _params_for(family, overrides)
    params["daily_limits"] = sim_config_base["daily_limits"]
    params["equity_index_codes"] = sim_config_base["equity_index_codes"]
    if style_signals is not None:
        params["_style_signals"] = style_signals
    lots = _initial_lots(adapter, snap, window[0])
    # 起始无数据基金的份额按现金处理
    total_snap = sum(f["value_cny"] for f in snap["funds"])
    held = set(lots)
    cash0 = sum(CAPITAL * f["value_cny"] / total_snap
                for f in snap["funds"] if f["code"] not in held)
    config = simulator.SimConfig(
        budget_per_day=BUDGET_PER_DAY,
        allow_new_equity_index=(family == "B0"),
        equity_index_codes=frozenset(sim_config_base["equity_index_codes"]),
        daily_limits=sim_config_base["daily_limits"])
    sim = simulator.Simulator([f["code"] for f in snap["funds"]], config, adapter)
    policy = policies.build_policy(family, params)
    result = sim.run(window[0], window[1], initial_lots=lots,
                     initial_cash=cash0, policy=policy)
    return result["metrics"]


def _select_risk_first(trials: pd.DataFrame) -> pd.Series:
    """风险优先选择：先 MDD≤5%，再 CAGR 最大；都不满足时选 MDD 最小（如实记录）。"""
    feasible = trials[trials["max_drawdown"] <= MDD_BUDGET]
    pool = feasible if len(feasible) else trials
    key = "cagr" if len(feasible) else "max_drawdown"
    return pool.sort_values(key, ascending=len(feasible) == 0).iloc[0]


# -- 压力测试 -----------------------------------------------------------------

def stress_bootstrap(adapter: LocalDataAdapter, snap: dict, horizon=252,
                     block=20, n_paths=2000, lookback_days=756, seed=42) -> dict:
    """块自助法：当前组合权重 × 近三年各基金真实日收益，估计 horizon 内 MDD>5% 频率。

    组合为当前快照权重（成立不足三年的基金按可得样本）。数据驱动，不伪造冲击。
    """
    rng = np.random.default_rng(seed)
    total = sum(f["value_cny"] for f in snap["funds"])
    rets = {}
    for f in snap["funds"]:
        df = adapter.nav([f["code"]], pd.Timestamp.now().tz_localize(CST))
        if df.empty:
            continue
        r = df.sort_values("nav_date")["adj_nav_total_return"].pct_change().dropna()
        rets[f["code"]] = (f["value_cny"] / total, r.tail(lookback_days).to_numpy())
    if not rets:
        return {"error": "no data"}
    min_len = min(len(r) for _, r in rets.values())
    if min_len < block * 5:
        return {"error": "insufficient common history"}
    R = np.column_stack([r[-min_len:] for _, r in rets.values()])
    w = np.array([wt for wt, _ in rets.values()])
    port = R @ w  # 组合日收益序列
    exceed, mdds = 0, []
    for _ in range(n_paths):
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, len(port) - block, size=n_blocks)
        path = np.concatenate([port[s: s + block] for s in starts])[:horizon]
        nav = np.cumprod(1 + path)
        m = fm.max_drawdown(nav.tolist())
        mdds.append(m)
        exceed += m > MDD_BUDGET
    p = exceed / n_paths
    ci = 1.96 * np.sqrt(p * (1 - p) / n_paths)
    return {"horizon_days": horizon, "block_days": block, "n_paths": n_paths,
            "p_mdd_exceed_5pct": p, "ci95": [max(0, p - ci), min(1, p + ci)],
            "mdd_median": float(np.median(mdds)), "mdd_p95": float(np.quantile(mdds, 0.95))}


def stress_2022_bond_crash(adapter: LocalDataAdapter, snap: dict,
                           window=("2022-10-31", "2022-12-15")) -> dict:
    """2022年11月债灾真实回放：当前权重在当时的组合收益（AS_TRADED 冲击）。"""
    total = sum(f["value_cny"] for f in snap["funds"])
    losses = {}
    for f in snap["funds"]:
        code = f["code"]
        try:
            df = adapter.nav([code], pd.Timestamp(window[1]).tz_localize(CST))
            w_df = df[(df["nav_date"] >= window[0]) & (df["nav_date"] <= window[1])]
            if len(w_df) < 2:
                losses[code] = None
                continue
            nav = w_df.sort_values("nav_date")["adj_nav_total_return"]
            losses[code] = float(nav.iloc[-1] / nav.iloc[0] - 1)
        except Exception:  # noqa: BLE001
            losses[code] = None
    port = sum(f["value_cny"] / total * (losses.get(f["code"]) or 0.0)
               for f in snap["funds"])
    return {"window": window, "portfolio_return": port,
            "per_fund": {c: r for c, r in losses.items()},
            "note": "当前权重×2022-11债灾真实区间收益；成立晚于窗口的基金按0计并列出"}


# -- 主流程 -------------------------------------------------------------------

def run_validation(out_dir="reports") -> dict:
    adapter = LocalDataAdapter()
    snap = load_account_snapshot()
    codes = [f["code"] for f in snap["funds"]]
    equity = sorted(f["code"] for f in snap["funds"] if f["category"] == "equity_index")
    from fundkit import storage
    limits = {}
    status = storage.load_trade_status().set_index("code")["daily_limit"]
    for c in codes:
        if c in status.index and pd.notna(status[c]) and 0 < status[c] < 1e9:
            limits[c] = float(status[c])
    sim_base = {"daily_limits": limits, "equity_index_codes": equity}
    bond_funds = [f["code"] for f in snap["funds"] if f["category"] == "bond"]

    trials_rows, selected = [], {}
    for fold_i, fold in enumerate(FOLDS):
        # B4/B5 信号按各窗口起点 as-of 重算（每个窗口独立，防未来信息）
        signals_train = _style_signals_asof(adapter, bond_funds, fold["train"][0])
        signals_val = _style_signals_asof(adapter, bond_funds, fold["val"][0])
        signals_test = _style_signals_asof(adapter, bond_funds, fold["test"][0])
        for family, overrides in GRID:
            tag = f"{family}|{json.dumps(overrides, sort_keys=True)}"
            m_train = _run_window(family, overrides, fold["train"], adapter, snap,
                                  sim_base, signals_train if family in ("B4", "B5") else None)
            m_val = _run_window(family, overrides, fold["val"], adapter, snap,
                                sim_base, signals_val if family in ("B4", "B5") else None)
            trials_rows.append({
                "fold": fold_i, "family": family, "params": tag,
                "phase": "train", **{k: v for k, v in m_train.items() if v is not None}})
            trials_rows.append({
                "fold": fold_i, "family": family, "params": tag,
                "phase": "val", **{k: v for k, v in m_val.items() if v is not None}})
        # 按族在 validation 上风险优先选参，冻结后跑 test
        df = pd.DataFrame(trials_rows)
        for family in {g[0] for g in GRID}:
            cand = df[(df.fold == fold_i) & (df.family == family) & (df.phase == "val")]
            pick = _select_risk_first(cand)
            overrides = json.loads(pick["params"].split("|", 1)[1])
            m_test = _run_window(family, overrides, fold["test"], adapter, snap,
                                 sim_base, signals_test if family in ("B4", "B5") else None)
            trials_rows.append({"fold": fold_i, "family": family,
                                "params": pick["params"], "phase": "test", "frozen": True,
                                **{k: v for k, v in m_test.items() if v is not None}})
            selected.setdefault(family, []).append(
                {"fold": fold_i, "params": overrides, "val_mdd": pick["max_drawdown"],
                 "val_cagr": pick.get("cagr"), "test_metrics": m_test})

    trials = pd.DataFrame(trials_rows)
    os.makedirs(out_dir, exist_ok=True)
    trials.to_csv(os.path.join(out_dir, "oos_trials.csv"), index=False)

    # 压力测试（当前快照组合）
    boot = stress_bootstrap(adapter, snap)
    crash = stress_2022_bond_crash(adapter, snap)

    # 可行性判定（全部 OOS test 折）
    test_rows = trials[trials.phase == "test"]
    feasibility = {}
    for family, grp in test_rows.groupby("family"):
        oos_ok = ((grp["cagr"] >= CAGR_TARGET) & (grp["max_drawdown"] <= MDD_BUDGET)).all()
        feasibility[family] = {
            "all_folds_cagr_ge_4pct_and_mdd_le_5pct": bool(oos_ok),
            "folds": grp[["fold", "cagr", "max_drawdown", "sharpe", "xirr"]].to_dict("records"),
        }
    any_feasible = any(v["all_folds_cagr_ge_4pct_and_mdd_le_5pct"]
                       for v in feasibility.values())
    # 帕累托前沿（test 折均值）
    front = (test_rows.groupby("family")[["cagr", "max_drawdown"]].mean()
             .reset_index().sort_values("max_drawdown"))
    pareto = []
    best = -np.inf
    for _, r in front.iterrows():
        if r["cagr"] > best:
            pareto.append({"family": r["family"], "cagr": r["cagr"],
                           "max_drawdown": r["max_drawdown"]})
            best = r["cagr"]

    summary = {
        "asof": pd.Timestamp.now().isoformat(),
        "folds": FOLDS, "grid_size": len(GRID),
        "counterfactual_start": "COUNTERFACTUAL_START_PORTFOLIO（快照权重×标准化本金，非真实历史）",
        "selected": selected,
        "feasibility": feasibility,
        "decision": ("FEASIBLE_CANDIDATES_EXIST" if any_feasible
                     else "NO_FEASIBLE_STRATEGY"),
        "pareto_front_test_mean": pareto,
        "stress": {"bootstrap": boot, "bond_crash_2022": crash},
        "trials_file": "reports/oos_trials.csv",
        "status": "RESEARCH_ONLY",
    }
    with open(os.path.join(out_dir, "oos_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return summary
