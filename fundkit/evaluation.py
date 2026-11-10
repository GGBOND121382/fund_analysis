"""第5步：滚动样本外反证（04_BACKTEST_VALIDATION.md / 06 第5步实现）。

结构（03 F 伪代码）：
1. 时间序列滚动折：train（调参记录）→ validation（风险优先选参）→ test（冻结后只跑一次）
2. 小维度预声明网格（禁止笛卡尔大爆炸；全部试验落盘 reports/oos_trials.csv）
3. 每个窗口起点用**明确标注的反事实组合**（快照权重×标准化本金，
   COUNTERFACTUAL_START_PORTFOLIO——不是用户真实历史）
4. B4/B5 的风格信号按窗口起点 as-of 重算（不得用今天的 style_report 回填历史）
5. 压力测试：块自助法组合 MDD 分布（P(MDD>4%) 及置信区间）+ 2022年11月债灾回放
6. 可行性：OOS 同时满足 CAGR>5% 且 MDD≤4% 才可行，否则 NO_FEASIBLE_STRATEGY +
   风险—收益帕累托前沿（04/06 禁止为凑 4% 调高收益假设）
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from fundkit import policies, simulator, style
from fundkit.adapters import CST, LocalDataAdapter, load_account_snapshot
from fundkit import finance_math as fm

CAPITAL = 195123.58  # 标准化本金（快照总额），反事实起点
BUDGET_PER_DAY = 1005.0
MDD_BUDGET = 0.04
CAGR_TARGET = 0.05

FOLDS = [
    {"train": ("2023-06-01", "2025-06-01"), "val": ("2025-06-01", "2025-12-01"),
     "test": ("2025-12-01", "2026-06-01")},
    {"train": ("2024-01-01", "2026-01-01"), "val": ("2026-01-01", "2026-06-01"),
     "test": ("2026-06-01", "2026-11-09")},
    # 12个月测试窗（验收要求）；train/val 与 fold0 相同，test 拉长到数据末端
    {"train": ("2023-06-01", "2025-06-01"), "val": ("2025-06-01", "2025-12-01"),
     "test": ("2025-12-01", "2026-11-09")},
    # 24个月测试窗（跨 710301 风格切换 2025-08-29，结果含切换前权益敞口，如实标注）
    {"train": ("2022-06-01", "2024-06-01"), "val": ("2024-06-01", "2024-11-01"),
     "test": ("2024-11-01", "2026-11-09")},
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


def _params_for(family: str, overrides: dict, base_target: dict | None = None) -> dict:
    params = policies.load_params()
    if base_target is not None:
        # 研究回测基准：维持快照配置。config 里的 cash:1.0 是决策端 fail-closed 占位，
        # 不代表任何研究结论，不能作为回测目标权重。
        params["target_weights"] = dict(base_target)
    if family == "B2":
        # 更改现金目标时，非现金权重必须按比例重新归一化，否则总权重≠1。
        target = dict(params["target_weights"])
        cash_weight = float(overrides.get("cash", 0.05))
        original_risky = sum(float(v) for k, v in target.items()
                             if k != "cash" and not k.startswith("_"))
        if not 0 <= cash_weight <= 1 or original_risky <= 0:
            raise ValueError("目标现金权重或风险权重无效")
        params["target_weights"] = {
            k: ((float(v) * (1 - cash_weight) / original_risky)
                if k not in ("cash",) and not k.startswith("_") else v)
            for k, v in target.items()}
        params["target_weights"]["cash"] = cash_weight
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
            nav0 = float(df.iloc[-1]["unit_nav"])
        except Exception:  # noqa: BLE001
            continue
        lots[code] = [simulator.Lot(code, value / nav0, start_ts, simulated=True)]
    return lots


def _run_window(family, overrides, window, adapter, snap, sim_config_base,
                style_signals=None, base_target=None):
    params = _params_for(family, overrides, base_target=base_target)
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
    m = dict(result["metrics"])
    # 统一报告口径：年化卖出换手率、交易费用、平均现金占比
    events, daily = result["events"], result["daily"]
    avg_val = float(daily["value"].mean()) if len(daily) else 0.0
    years = max((daily["date"].iloc[-1] - daily["date"].iloc[0]).days / 365, 1e-9)
    sells = fees = 0.0
    if not events.empty:
        red = events[events["event"] == "REDEEM"]
        if len(red):
            sells = float((red["shares"].astype(float) * red["nav"].astype(float)).sum())
            fees += sells * config.params["redeem_fee_rate"]
        buy = events[events["event"] == "BUY"]
        if len(buy):
            fees += float(buy["amount"].astype(float).sum()) * config.params["subscribe_fee_rate"]
    m["turnover_ann"] = sells / avg_val / years if avg_val > 0 else 0.0
    m["fees_total"] = fees
    m["cash_ratio_avg"] = float((daily["cash"] / daily["value"]).mean()) if len(daily) else 0.0
    return m


def _select_risk_first(trials: pd.DataFrame) -> pd.Series:
    """风险优先选择：先 MDD≤4%，再 CAGR 最大；都不满足时选 MDD 最小（如实记录）。"""
    feasible = trials[trials["max_drawdown"] <= MDD_BUDGET]
    pool = feasible if len(feasible) else trials
    key = "cagr" if len(feasible) else "max_drawdown"
    return pool.sort_values(key, ascending=len(feasible) == 0).iloc[0]


# -- 压力测试 -----------------------------------------------------------------

def stress_bootstrap(adapter: LocalDataAdapter, snap: dict, horizon=252,
                     block=20, n_paths=2000, lookback_days=756, seed=42) -> dict:
    """按共同日期对齐历史日收益并对组合收益进行整块自助；缺数据直接阻断。"""
    if horizon <= 0 or block <= 0 or n_paths <= 0:
        raise ValueError("horizon/block/n_paths 必须为正数")
    rng = np.random.default_rng(seed)
    total = sum(f["value_cny"] for f in snap["funds"])
    series = []
    weights = []
    missing = []
    for f in snap["funds"]:
        code = f["code"]
        df = adapter.nav([code], pd.Timestamp.now(tz=CST))
        if df.empty or len(df) < 2:
            missing.append(code)
            continue
        df = df.sort_values("nav_date").drop_duplicates("nav_date", keep="last")
        ser = (df.set_index("nav_date")["adj_nav_total_return"].astype(float)
               .pct_change(fill_method=None).rename(code)).dropna()
        if ser.empty or not np.isfinite(ser.to_numpy()).all():
            missing.append(code)
            continue
        series.append(ser)
        weights.append(float(f["value_cny"]) / total)
    if missing:
        return {"error": "DATA_BLOCKED_INCOMPLETE_ASSET_RETURNS", "missing_codes": missing}
    if not series:
        return {"error": "DATA_BLOCKED_NO_NAV"}
    aligned = pd.concat(series, axis=1, join="inner").dropna().tail(lookback_days)
    if len(aligned) < block * 5:
        return {"error": "DATA_BLOCKED_INSUFFICIENT_COMMON_DATES",
                "n_common": len(aligned)}
    R = aligned.to_numpy(float)
    port = R @ np.asarray(weights)
    mdds = []
    for _ in range(n_paths):
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, len(port) - block + 1, size=n_blocks)
        path = np.concatenate([port[s:s + block] for s in starts])[:horizon]
        nav = np.r_[1.0, np.cumprod(1 + path)]
        mdds.append(fm.max_drawdown(nav.tolist()))
    mdds = np.asarray(mdds)
    k = int(np.sum(mdds > MDD_BUDGET))
    rate = k / n_paths
    z = 1.95996398454
    denom = 1 + z * z / n_paths
    center = (rate + z * z / (2 * n_paths)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n_paths +
                       z * z / (4 * n_paths * n_paths)) / denom
    return {"horizon_days": horizon, "block_days": block, "n_paths": n_paths,
            "n_common_dates": len(aligned),
            "p_mdd_exceed_4pct": rate,
            "ci95": [max(0.0, center - half), min(1.0, center + half)],
            "mdd_median": float(np.median(mdds)),
            "mdd_p95": float(np.quantile(mdds, 0.95)),
            "note": "仅历史联合冲击的区块重采样估计，非未来风险保证"}


def stress_2022_bond_crash(adapter: LocalDataAdapter, snap: dict,
                           window=("2022-10-31", "2022-12-15")) -> dict:
    """当前权重买入持有回放：缺基金历史时明确拒绝计算，不填0收益。"""
    total = sum(f["value_cny"] for f in snap["funds"])
    ser = []
    weights = []
    missing = []
    per_fund = {}
    for f in snap["funds"]:
        code = f["code"]
        df = adapter.nav([code], pd.Timestamp(window[1]).tz_localize(CST) +
                         pd.Timedelta(days=4))
        df = df[(df["nav_date"] >= window[0]) & (df["nav_date"] <= window[1])]
        if len(df) < 2:
            missing.append(code)
            per_fund[code] = None
            continue
        s = df.sort_values("nav_date").set_index("nav_date")["adj_nav_total_return"]
        if not s.index.is_unique:
            s = s[~s.index.duplicated(keep="last")]
        per_fund[code] = float(s.iloc[-1] / s.iloc[0] - 1)
        ser.append(s.rename(code))
        weights.append(float(f["value_cny"]) / total)
    if missing:
        return {"window": window, "portfolio_return": None,
                "portfolio_max_drawdown": None, "per_fund": per_fund,
                "missing_codes": missing, "error": "DATA_BLOCKED_INCOMPLETE_2022_HISTORY",
                "note": "缺失基金不能按0收益填充"}
    common = pd.concat(ser, axis=1, join="inner").dropna()
    if len(common) < 2:
        return {"window": window, "portfolio_return": None,
                "portfolio_max_drawdown": None, "error": "DATA_BLOCKED_NO_COMMON_DATES"}
    rel = common.div(common.iloc[0])
    value = rel.to_numpy(float) @ np.asarray(weights)
    return {"window": window, "portfolio_return": float(value[-1] - 1),
            "portfolio_max_drawdown": fm.max_drawdown(value.tolist()),
            "per_fund": per_fund,
            "note": "反事实当前持仓买入持有回放，非历史实际投资组合；无风格迁移修正"}


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
    # 研究基准目标权重 = 快照配置（与反事实起点组合一致）
    total_snap0 = sum(float(f["value_cny"]) for f in snap["funds"])
    base_target = {f["code"]: float(f["value_cny"]) / total_snap0 for f in snap["funds"]}
    base_target["cash"] = 0.0

    # 必须先补齐本地原始基金净值。缺数据不得用旧 reports/ 回填伪结果。
    absent = [code for code in codes if adapter.nav([code], pd.Timestamp.now(tz=CST)).empty]
    if absent:
        return {"status": "DATA_BLOCKED", "decision": "NOT_VALIDATED",
                "missing_nav_codes": absent,
                "note": "缺少历史净值；未运行任何收益回测；不得复用旧版报告"}

    trials_rows, selected = [], {}
    for fold_i, fold in enumerate(FOLDS):
        # B4/B5 信号按各窗口起点 as-of 重算（每个窗口独立，防未来信息）
        signals_train = _style_signals_asof(adapter, bond_funds, fold["train"][0])
        signals_val = _style_signals_asof(adapter, bond_funds, fold["val"][0])
        signals_test = _style_signals_asof(adapter, bond_funds, fold["test"][0])
        for family, overrides in GRID:
            tag = f"{family}|{json.dumps(overrides, sort_keys=True)}"
            m_train = _run_window(family, overrides, fold["train"], adapter, snap,
                                  sim_base, signals_train if family in ("B4", "B5") else None,
                                  base_target)
            m_val = _run_window(family, overrides, fold["val"], adapter, snap,
                                sim_base, signals_val if family in ("B4", "B5") else None,
                                base_target)
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
                                 sim_base, signals_test if family in ("B4", "B5") else None,
                                 base_target)
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
        oos_ok = ((grp["cagr"] > CAGR_TARGET) & (grp["max_drawdown"] <= MDD_BUDGET)).all()
        feasibility[family] = {
            "all_folds_cagr_gt_5pct_and_mdd_le_4pct": bool(oos_ok),
            "folds": grp[["fold", "cagr", "max_drawdown", "sharpe", "xirr"]].to_dict("records"),
        }
    any_feasible = any(v["all_folds_cagr_gt_5pct_and_mdd_le_4pct"]
                       for v in feasibility.values())
    # 当前测试窗不足完整一年；半年收益年化放大不能作为4%长期目标的验收。
    oos_days = [len(pd.bdate_range(*fold["test"])) for fold in FOLDS]
    validated_horizon = all(n >= 252 for n in oos_days)
    # B4/B5 固定了起点 as-of 风格信号，并非每日重新采集。不可声称动态系统通过样本外验证。
    style_signal_refresh_verified = False
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
        "decision": ("INSUFFICIENT_OOS_HORIZON" if not validated_horizon
                     else "FEASIBLE_CANDIDATES_EXIST" if any_feasible
                     else "NO_FEASIBLE_STRATEGY"),
        "oos_test_days": oos_days,
        "style_signal_refresh_verified": style_signal_refresh_verified,
        "pareto_front_test_mean": pareto,
        "stress": {"bootstrap": boot, "bond_crash_2022": crash},
        "trials_file": "reports/oos_trials.csv",
        "status": "RESEARCH_ONLY",
        "validation_version": "UNVALIDATED_SHORT_FOLDS",
    }
    with open(os.path.join(out_dir, "oos_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return summary