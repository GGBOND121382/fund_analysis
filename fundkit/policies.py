"""策略族 B0-B5（03_POLICY_ALGORITHMS.md 实现）。

策略接口：policy(state) -> {"subscribe": {code: 金额}, "redeem": {code: 份额}}
state 只含决策时点已公开的信息（由 simulator 保证），含：
  date / cash / value / unit_nav / nav（已公开净值）/ holdings（逐基金份额、
  已公开估值、可赎回份额）/ unit_nav_history（已公开口径组合单位净值历史）

- B0 snapshot_dca:        原截图固定定投（含股指），研究对照
- B1 no_new_equity_dca:   停股指新增 205 元留现金，其余不变
- B2 cashflow_rebalance:  只用新增现金流向目标权重 w* 靠拢（不卖存量；内层 QP 的
                          赤字注水近似解，并输出实际可达与目标偏差）
- B3 threshold_rebalance: B2 + 周度偏离阈值检查；默认不真赎回（allow_redeem=false
                          时只记 REDUCE_REVIEW 事件）
- B4 style_factor_risk:   B2 + 风格/风险状态机（PAUSE/REENTER 带迟滞与冷却期），
                          信号来自 style 模块的披露断点与反事实压力
- B5 joint_mdd_cdar:      B4 + 组合回撤预算守卫（预算 4% − 内控缓冲）+
                          逐基金回撤暂停/再入场闭环

所有策略输出均为 RESEARCH_ONLY：不自动交易；赎回仅"审议"性质。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from functools import lru_cache

from fundkit.adapters import load_account_snapshot
from fundkit.finance_math import drawdown_path, empirical_cdar

PARAMS_FILE = os.path.join("config", "policy_params.json")
STYLE_REPORT_FILE = os.path.join("reports", "style_report.json")


@lru_cache(maxsize=1)
def _plans_cached() -> tuple:
    return tuple(load_account_snapshot()["funds"])


def load_params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _plans() -> list[dict]:
    return list(_plans_cached())


def _weights(state) -> tuple[dict, float]:
    """按截至决策时点已公开原始净值估值；缺任意非零仓位数据则拒绝下单。"""
    if state.get("value_quality") == "MISSING":
        return {}, 0.0
    total = float(state["cash"]) + float(state.get("pending_value", 0.0))
    vals = {}
    for code, h in state["holdings"].items():
        if h["shares"] > 0 and h["value_pub"] is None:
            return {}, 0.0
        v = float(h["value_pub"] or 0.0)
        vals[code] = v
        total += v
    if total <= 0:
        return {}, 0.0
    return {c: v / total for c, v in vals.items()}, total


def _water_filling(cash: float, weights: dict, total: float, target: dict,
                   limits: dict, allowed: list[str]) -> dict:
    """在已含现金的总资产 V 上，求非负有界最小二乘补足目标仓位。

    min Σ_i (v_i+a_i - w_i^* V)^2, 0≤a_i≤limit_i, Σa_i≤cash。
    解析水位 a_i=clip(deficit_i-λ, 0, cap_i)，λ由二分法寻找。
    不把 cash 加入 total 第二次；剩余现金保留。
    """
    if total <= 0 or cash <= 0:
        return {}
    allowed = sorted(set(allowed))
    deficit = np.array([max(0.0, target.get(c, 0.0) * total -
                            weights.get(c, 0.0) * total) for c in allowed], dtype=float)
    cap = np.array([max(0.0, float(limits.get(c, float("inf")))) for c in allowed])
    desired = np.minimum(deficit, cap)
    if desired.sum() <= cash:
        alloc = desired
    else:
        low, high = 0.0, float(np.max(deficit))
        for _ in range(80):
            mid = (low + high) / 2.0
            alloc = np.minimum(np.maximum(deficit - mid, 0.0), cap)
            if alloc.sum() > cash:
                low = mid
            else:
                high = mid
        alloc = np.minimum(np.maximum(deficit - high, 0.0), cap)
    return {c: float(x) for c, x in zip(allowed, alloc) if x > 1e-8}


# -- B0 / B1 ---------------------------------------------------------------

def b0_snapshot_dca(state) -> dict:
    return {"subscribe": {f["code"]: f["daily_plan_cny"]
                          for f in _plans() if f["daily_plan_cny"] > 0}}


def b1_no_new_equity_dca(state) -> dict:
    return {"subscribe": {f["code"]: f["daily_plan_cny"] for f in _plans()
                          if f["daily_plan_cny"] > 0 and f["category"] != "equity_index"}}


# -- B2 ---------------------------------------------------------------------

def make_b2(params: dict):
    target = {k: v for k, v in params["target_weights"].items()
              if k != "cash" and not k.startswith("_")}
    forbidden = set(params.get("disabled_new_buy_codes", ()))
    allowed = sorted(set(target) - forbidden)
    limits = params.get("daily_limits", {})  # 真实申购限额（fund_status 快照）

    def policy(state) -> dict:
        weights, total = _weights(state)
        if total <= 0:
            return {}
        cash_floor = params["target_weights"].get("cash", 0.0)
        # 保留现金下限后的可投放现金
        investable = max(0.0, state["cash"] - cash_floor * total)
        if investable <= 0:
            return {}
        buys = _water_filling(investable, weights, total, target, limits, allowed)
        return {"subscribe": buys}

    return policy


# -- B3 ---------------------------------------------------------------------

def make_b3(params: dict):
    base = make_b2(params)
    p3 = params["B3"]
    last_check = {"day": None}

    def policy(state) -> dict:
        orders = base(state)
        freq_days = {"weekly": 7, "monthly": 30, "quarterly": 90}[p3["check_frequency"]]
        day = state["date"]
        if last_check["day"] is not None and (day - last_check["day"]).days < freq_days:
            return orders
        last_check["day"] = day
        weights, total = _weights(state)
        target = {k: v for k, v in params["target_weights"].items()
                  if k != "cash" and not k.startswith("_")}
        for c, tw in target.items():
            w = weights.get(c, 0.0)
            if w - tw > p3["weight_trigger"]:
                gap_value = (w - tw) * total * p3["destination_fraction"]
                h = state["holdings"].get(c)
                if h and h["redeemable_shares"] > 1e-6 and h["value_pub"]:
                    shares = min(h["redeemable_shares"],
                                 gap_value / (h["value_pub"] / h["shares"]))
                    if shares > 1e-6:
                        if p3["allow_redeem"]:
                            orders.setdefault("redeem", {})[c] = shares
                        else:
                            orders.setdefault("redeem_review", {})[c] = round(shares, 2)
        return orders

    return policy


# -- B4 ---------------------------------------------------------------------

def _load_style_signals() -> dict:
    if not os.path.exists(STYLE_REPORT_FILE):
        return {}
    with open(STYLE_REPORT_FILE, encoding="utf-8") as f:
        return json.load(f)


def make_b4(params: dict, make_base=None):
    base = (make_base or make_b2)(params)
    p4 = params["B4"]
    # 优先使用调用方注入的 as-of 信号（回测防未来信息），否则读当前 style 报告
    # 历史回测禁止读取当前 reports/style_report.json，否则会泄露未来信息。
    signals = params.get("_style_signals") or {}
    # 外层联合优化可注入按披露时点预计算的风格时间线：
    # {"YYYY-MM-DD": {fund_code: style_signal}}。每天只读取不晚于当日的最近一次信号。
    signals_by_date = params.get("_style_signals_by_date") or {}
    signal_dates = sorted(signals_by_date)
    z_state: dict[str, dict] = {}  # code -> {"state": ..., "paused_since": ..., "stable_days": int}

    def signal_asof(code: str, day: pd.Timestamp):
        if not signal_dates:
            return signals.get(code)
        key = pd.Timestamp(day).strftime("%Y-%m-%d")
        eligible = [d for d in signal_dates if d <= key]
        if not eligible:
            return None
        return signals_by_date[eligible[-1]].get(code)

    def fund_paused(code: str, day: pd.Timestamp) -> bool:
        sig = signal_asof(code, day)
        if not sig:
            return False
        z = z_state.setdefault(code, {"state": "NORMAL", "paused_since": None,
                                      "stable": 0})
        shift = sig.get("combined_shift_evidence", {}).get("state")
        views = sig.get("risk_views", {})
        cdar = (views.get("CURRENT_STYLE_REWEIGHTED") or {}).get("cdar95")
        risk_high = (shift in p4["pause_shift_states"]
                     and cdar is not None and cdar > p4["pause_cdar95"])
        risk_low = (shift not in p4["pause_shift_states"]
                    or (cdar is not None and cdar < p4["reenter_cdar95_low"]))
        if z["state"] == "NORMAL" and risk_high:
            z["state"], z["paused_since"], z["stable"] = "PAUSE", day, 0
        elif z["state"] == "PAUSE":
            z["stable"] = z["stable"] + 1 if risk_low else 0
            if z["stable"] >= p4["reenter_cooldown_days"]:
                z["state"] = "NORMAL"  # REENTER_WAIT 期满恢复（迟滞+冷却，03 C）
        return z["state"] == "PAUSE"

    def policy(state) -> dict:
        orders = base(state) or {}
        for c in list((orders.get("subscribe") or {})):
            if fund_paused(c, state["date"]):
                orders["subscribe"].pop(c)
                orders.setdefault("events", []).append(
                    {"code": c, "action": "PAUSE_NEW_BUY",
                     "reason": "style/risk signal (B4)"})
        return orders

    return policy


# -- B5 ---------------------------------------------------------------------

def make_b5(params: dict):
    """B5：组合回撤保护 + 基金级波动率止损 + 有成本基础的移动止盈。"""
    from fundkit.risk_engine import risk_contributions

    base = make_b4(params)
    p5 = params["B5"]
    guard = {"de_risk": False, "stable": 0, "last_exit_mdd": None}
    fund_state: dict[str, dict] = {}
    peaks: dict[str, float] = {}
    tp_peaks: dict[str, float] = {}

    def history_returns(state):
        frames = {}
        for code, rows in (state.get("fund_nav_history") or {}).items():
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            if not {"date", "total_return_nav"}.issubset(frame.columns):
                continue
            frame["date"] = pd.to_datetime(frame["date"])
            frame["total_return_nav"] = pd.to_numeric(frame["total_return_nav"], errors="coerce")
            frame = frame.dropna().drop_duplicates("date").sort_values("date")
            if len(frame) < 2 or (frame["total_return_nav"] <= 0).any():
                continue
            frames[code] = frame.set_index("date")["total_return_nav"].pct_change(fill_method=None)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1, join="inner").dropna().tail(756)

    def portfolio_risk_order(state):
        returns = history_returns(state)
        if len(returns) < int(p5.get("portfolio_risk_min_observations", 20)):
            return [], {}
        total = float(state.get("value", 0.0))
        if total <= 0:
            return [], {}
        weights = {
            code: float(h.get("value_pub") or 0.0) / total
            for code, h in state.get("holdings", {}).items()
            if float(h.get("value_pub") or 0.0) > 0 and code in returns.columns
        }
        if not weights:
            return [], {}
        try:
            result = risk_contributions(
                returns, weights,
                shrinkage=float(p5.get("risk_covariance_shrinkage", 0.25)),
                min_observations=int(p5.get("portfolio_risk_min_observations", 20)),
            )
        except ValueError:
            return [], {}
        contributions = {
            code: max(0.0, float(item["fraction"]))
            for code, item in result["contributions"].items()
        }
        order = sorted(contributions, key=contributions.get, reverse=True)
        return order, contributions

    def add_redeem(orders, code, shares, review=False):
        if shares <= 1e-8:
            return False
        key = "redeem_review" if review else "redeem"
        orders.setdefault(key, {})
        orders[key][code] = float(orders[key].get(code, 0.0)) + float(shares)
        return True

    def policy(state) -> dict:
        orders = base(state) or {}
        orders.setdefault("subscribe", {})
        orders.setdefault("events", [])
        if state.get("value_quality") == "MISSING":
            orders["subscribe"] = {}
            orders["events"].append({
                "code": "PORTFOLIO", "action": "RISK_DATA_BLOCKED",
                "reason": "存在非零持仓缺失可用估值；停止新增申购，禁止推断赎回量",
            })
            return orders

        hist = list(state.get("unit_nav_history") or [])
        mdd_now = None
        if len(hist) >= 2 and max(hist) > 0:
            mdd_now = max(0.0, float(1.0 - hist[-1] / max(hist)))
        lookback = int(p5.get("cdar_lookback", 126))
        win = hist[-lookback:]
        cdar_now = (empirical_cdar(drawdown_path(win), .95)
                    if len(win) >= int(p5.get("portfolio_risk_min_observations", 20)) else None)

        budget = float(p5.get("mdd_budget", 0.04))
        margin = float(p5.get("internal_drawdown_margin", 0.01))
        soft = budget - margin
        if not (0 < soft < budget < 1):
            raise ValueError("B5 组合风险缓冲配置非法")
        cdar_threshold = p5.get("cdar_guard_threshold")
        cdar_alert = (cdar_now is not None and cdar_threshold is not None
                      and cdar_now >= float(cdar_threshold))
        risk_trigger = (mdd_now is not None and mdd_now >= soft) or cdar_alert

        if not guard["de_risk"] and risk_trigger:
            guard.update(de_risk=True, stable=0, last_exit_mdd=None)
        elif guard["de_risk"]:
            mdd_recovered = mdd_now is not None and mdd_now < soft * float(
                p5.get("reenter_mdd_fraction", 0.6))
            cdar_recovered = (cdar_now is None or cdar_threshold is None or
                              cdar_now < float(cdar_threshold) * float(
                                  p5.get("reenter_cdar_fraction", 0.6)))
            guard["stable"] = guard["stable"] + 1 if (mdd_recovered and cdar_recovered) else 0
            if guard["stable"] >= int(p5.get("reenter_cooldown_days", 10)):
                guard.update(de_risk=False, stable=0, last_exit_mdd=None)

        if guard["de_risk"]:
            orders["subscribe"] = {}
            orders["events"].append({
                "code": "PORTFOLIO", "action": "DE_RISK_PAUSE_BUYS",
                "reason": (f"组合当前回撤={mdd_now if mdd_now is not None else float('nan'):.2%}; "
                           f"CDaR95={cdar_now if cdar_now is not None else float('nan'):.2%}; "
                           f"软预算={soft:.2%}"),
            })
            escalation = (guard["last_exit_mdd"] is None or mdd_now is None or
                          mdd_now >= guard["last_exit_mdd"] +
                          float(p5.get("portfolio_exit_escalation_step", 0.005)))
            if p5.get("allow_redeem", False) and escalation:
                order, contrib = portfolio_risk_order(state)
                coverage = float(p5.get("portfolio_risk_coverage", 0.80))
                selected, cumulative = [], 0.0
                for code in order:
                    if contrib.get(code, 0.0) <= 0:
                        continue
                    selected.append(code)
                    cumulative += contrib[code]
                    if cumulative >= coverage:
                        break
                placed = False
                frac = max(0.0, min(1.0, float(p5.get("portfolio_redeem_fraction", 0.5))))
                for code in selected:
                    h = state["holdings"].get(code, {})
                    shares = float(h.get("redeemable_shares") or 0.0) * frac
                    placed = add_redeem(orders, code, shares) or placed
                if placed:
                    guard["last_exit_mdd"] = mdd_now
                    orders["events"].append({
                        "code": "PORTFOLIO", "action": "PORTFOLIO_RISK_REDUCE",
                        "risk_contributions": contrib, "selected_codes": selected,
                        "redeem_fraction": frac,
                    })
                elif not order:
                    orders["events"].append({
                        "code": "PORTFOLIO", "action": "PORTFOLIO_REDEEM_BLOCKED_RISK_HISTORY",
                        "reason": "缺少足够共同日期收益，不能可靠排序组合风险贡献",
                    })
                else:
                    orders["events"].append({
                        "code": "PORTFOLIO", "action": "PORTFOLIO_REDEEM_BLOCKED_LOCKED",
                        "reason": "风险贡献较高的基金当前没有可赎回份额",
                    })
            elif not p5.get("allow_redeem", False):
                orders["events"].append({
                    "code": "PORTFOLIO", "action": "PORTFOLIO_REDEEM_REVIEW",
                    "reason": "当前配置仅暂停申购；真实赎回未获执行授权",
                })
            return orders

        histories = state.get("fund_nav_history") or {}
        categories = params.get("asset_categories", {})
        stop_k_by_category = p5.get("fund_stop_vol_multiplier_by_category", {})
        stop_cap = float(p5.get("fund_stop_drawdown_cap", 0.12))
        escalation_step = float(p5.get("fund_stop_escalation_step", 0.005))
        cooldown_days = int(p5.get("reenter_cooldown_days", 10))

        for code, pub in (state.get("nav") or {}).items():
            nav = float(pub.get("total_return_nav") or pub["nav"])
            if not np.isfinite(nav) or nav <= 0:
                continue
            rows = histories.get(code) or []
            h = state.get("holdings", {}).get(code, {})
            peak_lookback = int(p5.get("fund_peak_lookback_days", 252))
            position_start = h.get("position_start_date")
            relevant_rows = [r for r in rows
                             if position_start is not None and
                             pd.Timestamp(r["date"]) >= pd.Timestamp(position_start)]
            if relevant_rows:
                hist_values = pd.to_numeric(
                    pd.DataFrame(relevant_rows)["total_return_nav"], errors="coerce"
                ).dropna().to_numpy(dtype=float)
            else:
                hist_values = pd.to_numeric(
                    pd.DataFrame(rows)["total_return_nav"], errors="coerce"
                ).dropna().to_numpy(dtype=float) if rows else np.asarray([], dtype=float)
                hist_values = hist_values[-peak_lookback:] if len(hist_values) else hist_values
            peak = max(float(np.max(hist_values)) if len(hist_values) else nav, nav)
            peaks[code] = peak
            dd = max(0.0, 1.0 - nav / peak)
            z = fund_state.setdefault(code, {
                "paused": False, "stable": 0, "last_exit_dd": None,
                "last_exit_redeemable_after": 0.0,
                "take_profit_cooldown_until": None,
                "take_profit_peak_reset_date": None,
            })

            vol20 = None
            if len(rows) >= 21:
                values = pd.to_numeric(
                    pd.DataFrame(rows)["total_return_nav"], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                if len(values) >= 21 and np.all(values > 0):
                    rets = np.diff(values[-61:]) / values[-61:-1]
                    if len(rets) >= 20:
                        vol20 = float(np.std(rets[-20:], ddof=1) * np.sqrt(20.0))
            category = categories.get(code, "other")
            base_stop = float(p5.get("fund_pause_drawdown_by_code", {}).get(
                code, p5.get("fund_pause_drawdown", 0.02)))
            k_vol = float(stop_k_by_category.get(category, p5.get("fund_stop_vol_multiplier", 2.0)))
            stop_threshold = min(stop_cap, max(base_stop, k_vol * vol20 if vol20 is not None else base_stop))
            reenter_threshold = float(p5.get("fund_reenter_drawdown_by_code", {}).get(
                code, p5.get("fund_reenter_drawdown", base_stop * 0.5)))
            if not (0 <= reenter_threshold < stop_threshold < 1):
                raise ValueError(f"{code} 基金级止损/恢复阈值非法")

            if not z["paused"] and dd >= stop_threshold:
                z.update(paused=True, stable=0, last_exit_dd=None,
                         last_exit_redeemable_after=0.0)
            elif z["paused"]:
                z["stable"] = z["stable"] + 1 if dd < reenter_threshold else 0
                if z["stable"] >= cooldown_days:
                    z.update(paused=False, stable=0, last_exit_dd=None,
                             last_exit_redeemable_after=0.0)
                    peaks[code] = nav

            if z["paused"]:
                orders["subscribe"].pop(code, None)
                orders["events"].append({
                    "code": code, "action": "PAUSE_NEW_BUY",
                    "reason": f"总回报净值回撤={dd:.2%}，动态止损阈值={stop_threshold:.2%}",
                })
                shares = float(h.get("redeemable_shares") or 0.0)
                frac = float(p5.get("fund_redeem_fraction_by_code", {}).get(
                    code, p5.get("fund_redeem_fraction", 0.5)))
                frac = max(0.0, min(1.0, frac))
                escalation = (z["last_exit_dd"] is None or
                              dd >= z["last_exit_dd"] + escalation_step or
                              shares > z["last_exit_redeemable_after"] + 1e-8)
                if p5.get("allow_redeem", False) and shares > 1e-8 and escalation:
                    amount = shares * frac
                    if add_redeem(orders, code, amount):
                        z["last_exit_dd"] = dd
                        z["last_exit_redeemable_after"] = max(0.0, shares - amount)
                        orders["events"].append({
                            "code": code, "action": "FUND_STOP_LOSS_REDEEM",
                            "drawdown": dd, "threshold": stop_threshold,
                            "shares": amount,
                        })
                elif p5.get("allow_redeem", False) and shares <= 1e-8:
                    orders["events"].append({
                        "code": code, "action": "STOP_LOSS_BLOCKED_LOCKED",
                        "reason": "触发止损但当前没有可赎回批次；保留暂停状态并继续监测",
                    })
                elif not p5.get("allow_redeem", False) and shares > 1e-8:
                    add_redeem(orders, code, shares * frac, review=True)
                continue

            tp_cooldown = z.get("take_profit_cooldown_until")
            if tp_cooldown is not None and pd.Timestamp(state["date"]) < tp_cooldown:
                orders["subscribe"].pop(code, None)
                continue
            position_return = h.get("redeemable_position_return")
            redeemable_start = h.get("redeemable_start_date")
            tp_reset = z.get("take_profit_peak_reset_date")
            tp_start = (max(pd.Timestamp(redeemable_start), pd.Timestamp(tp_reset))
                        if redeemable_start is not None and tp_reset is not None
                        else pd.Timestamp(redeemable_start) if redeemable_start is not None
                        else pd.Timestamp(tp_reset) if tp_reset is not None else None)
            tp_rows = [r for r in rows if tp_start is not None and
                       pd.Timestamp(r["date"]) >= tp_start]
            if tp_reset is not None and not tp_rows:
                tp_peak = tp_peaks.get(code, nav)
            elif tp_rows:
                tp_values = pd.to_numeric(
                    pd.DataFrame(tp_rows)["total_return_nav"], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                tp_peak = max(float(np.max(tp_values)) if len(tp_values) else nav, nav)
            else:
                tp_peak = max(peaks.get(code, nav), nav)
            tp_peaks[code] = max(tp_peaks.get(code, tp_peak), tp_peak, nav)
            tp_dd = max(0.0, 1.0 - nav / tp_peaks[code])
            activation = float(p5.get("take_profit_activation_by_code", {}).get(
                code, p5.get("take_profit_activation_by_category", {}).get(category, 0.0)))
            trailing = float(p5.get("take_profit_trailing_drawdown_by_code", {}).get(
                code, p5.get("take_profit_trailing_drawdown_by_category", {}).get(category, 0.0)))
            tp_fraction = float(p5.get("take_profit_redeem_fraction_by_code", {}).get(
                code, p5.get("take_profit_redeem_fraction", 0.25)))
            if (activation > 0 and trailing > 0 and position_return is not None
                    and float(position_return) >= activation and tp_dd >= trailing):
                shares = float(h.get("redeemable_shares") or 0.0)
                amount = shares * max(0.0, min(1.0, tp_fraction))
                if amount > 1e-8:
                    if p5.get("allow_redeem", False):
                        add_redeem(orders, code, amount)
                        action = "TAKE_PROFIT_REDEEM"
                    else:
                        add_redeem(orders, code, amount, review=True)
                        action = "TAKE_PROFIT_REVIEW"
                    orders["subscribe"].pop(code, None)
                    z["take_profit_cooldown_until"] = (
                        pd.Timestamp(state["date"]) +
                        pd.Timedelta(days=int(p5.get("take_profit_cooldown_days", 10)))
                    )
                    tp_peaks[code] = nav
                    z["take_profit_peak_reset_date"] = pd.Timestamp(state["date"]).date().isoformat()
                    orders["events"].append({
                        "code": code, "action": action,
                        "position_return": float(position_return),
                        "trailing_drawdown": tp_dd, "shares": amount,
                        "fraction": max(0.0, min(1.0, tp_fraction)),
                    })
                else:
                    orders["events"].append({
                        "code": code, "action": "TAKE_PROFIT_BLOCKED_LOCKED",
                        "reason": "盈利达到触发条件，但没有可赎回份额",
                    })
        return orders

def build_policy(name: str, params: dict | None = None):
    params = params or load_params()
    table = {
        "B0": lambda: b0_snapshot_dca,
        "B1": lambda: b1_no_new_equity_dca,
        "B2": lambda: make_b2(params),
        "B3": lambda: make_b3(params),
        "B4": lambda: make_b4(params),
        "B5": lambda: make_b5(params),
    }
    if name not in table:
        raise ValueError(f"未知策略 {name}，可选: {', '.join(table)}")
    return table[name]()


# 兼容旧引用（cmd_sim 此前直接用 POLICIES dict 的场景已迁移到 build_policy）
POLICIES = {k: (lambda k: lambda: build_policy(k))(k) for k in ("B0", "B1", "B2", "B3", "B4", "B5")}