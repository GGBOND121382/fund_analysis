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
- B5 joint_mdd_cdar:      B4 + 组合回撤预算守卫（预算 5% − 内控缓冲）+
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
    """当前各基金已公开估值权重 + 组合已公开总估值。"""
    total = state["cash"]
    vals = {}
    for code, h in state["holdings"].items():
        v = h["value_pub"] if h["value_pub"] is not None else 0.0
        vals[code] = v
        total += v
    if total <= 0:
        return {}, 0.0
    return {c: v / total for c, v in vals.items()}, total


def _water_filling(cash: float, weights: dict, total: float, target: dict,
                   limits: dict, allowed: list[str]) -> dict:
    """赤字注水法分配现金：使配置向 w* 靠拢（内层 QP 的近似解，03 D2）。

    每轮把现金按目标赤字比例注入最低配基金，受日限额截断；
    迭代至现金耗尽或无赤字。返回 {code: 金额}。
    """
    buys: dict[str, float] = {}
    values = {c: w * total for c, w in weights.items()}
    cash_left = cash
    for _ in range(20):
        if cash_left <= 1e-6:
            break
        deficits = {}
        for c in allowed:
            tw = target.get(c, 0.0)
            cur = values.get(c, 0.0) + buys.get(c, 0.0)
            deficits[c] = max(0.0, tw * (total + cash) - cur)
        # 现金自身也是目标资产：先满足 cash_floor 以外的部分才投放
        active = {c: d for c, d in deficits.items() if d > 1e-6}
        if not active:
            break
        s = sum(active.values())
        progressed = False
        for c, d in active.items():
            amt = min(cash_left * d / s, d, cash_left)
            cap = limits.get(c)
            if cap is not None:
                already = buys.get(c, 0.0)
                amt = min(amt, max(0.0, cap - already))
            if amt > 1e-6:
                buys[c] = buys.get(c, 0.0) + amt
                cash_left -= amt
                progressed = True
        if not progressed:
            break
    return buys


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
    equity = set(params.get("equity_index_codes", ()))
    plans = {f["code"] for f in _plans() if f["daily_plan_cny"] > 0}
    allowed = sorted((set(target) | plans) - equity)
    limits = params.get("daily_limits", {})  # 真实申购限额（fund_status 快照）

    def policy(state) -> dict:
        weights, total = _weights(state)
        if total <= 0:
            return {}
        cash_floor = params["target_weights"].get("cash", 0.0)
        # 保留现金下限后的可投放现金
        investable = max(0.0, state["cash"] - cash_floor * (total + state["cash"]))
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
    signals = params.get("_style_signals") or _load_style_signals()
    z_state: dict[str, dict] = {}  # code -> {"state": ..., "paused_since": ..., "stable_days": int}

    def fund_paused(code: str, day: pd.Timestamp) -> bool:
        sig = signals.get(code)
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
    base = make_b4(params)
    p5 = params["B5"]
    guard = {"de_risk": False, "stable": 0}
    fund_state: dict[str, dict] = {}
    peaks: dict[str, float] = {}

    def policy(state) -> dict:
        orders = base(state) or {}
        hist = state["unit_nav_history"]
        mdd_now = 0.0
        if len(hist) >= 2:
            arr = np.asarray(hist)
            mdd_now = float(np.max(1 - arr / np.maximum.accumulate(arr)))
        soft = p5["mdd_budget"] - p5["internal_drawdown_margin"]
        hard_recover = soft * p5["reenter_mdd_fraction"]
        if not guard["de_risk"] and mdd_now > soft:
            guard["de_risk"], guard["stable"] = True, 0
        elif guard["de_risk"]:
            guard["stable"] = guard["stable"] + 1 if mdd_now < hard_recover else 0
            if guard["stable"] >= p5["reenter_cooldown_days"]:
                guard["de_risk"] = False
        if guard["de_risk"]:
            orders["subscribe"] = {}  # 回撤守卫：全部新增留现金
            orders.setdefault("events", []).append(
                {"code": "PORTFOLIO", "action": "DE_RISK_CASH",
                 "reason": f"组合MDD {mdd_now:.2%} 超过软预算 {soft:.2%}"})
            return orders

        # 逐基金回撤暂停/再入场（按已公开净值）
        for c, pub in (state["nav"] or {}).items():
            nav = pub["nav"]
            peaks[c] = max(peaks.get(c, nav), nav)
            dd = 1 - nav / peaks[c]
            z = fund_state.setdefault(c, {"paused": False, "stable": 0})
            if not z["paused"] and dd > p5["fund_pause_drawdown"]:
                z["paused"], z["stable"] = True, 0
            elif z["paused"]:
                z["stable"] = z["stable"] + 1 if dd < p5["fund_reenter_drawdown"] else 0
                if z["stable"] >= p5["reenter_cooldown_days"]:
                    z["paused"] = False
            if z["paused"]:
                (orders.get("subscribe") or {}).pop(c, None)
                orders.setdefault("events", []).append(
                    {"code": c, "action": "PAUSE_NEW_BUY",
                     "reason": f"基金回撤 {dd:.2%}（B5）"})
        return orders

    return policy


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
