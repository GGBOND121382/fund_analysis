"""第6步：研究建议输出（schemas/decision.schema.json 校验）。

产出：
- decisions/YYYY-MM-DD.json    结构化决策（jsonschema 校验，默认 RESEARCH_ONLY）
- reports/fund_decisions.csv   可比较的逐基金决策表
- reports/strategy_report.md   简洁策略报告
- reports/full_research_audit.md  完整审计（公式/数据来源/调参/样本外结果）

输出语义（05 §5 / 06 第6步）：只提建议，不自动交易；赎回仅"审议"性质；
数据不足的动作一律 DATA_BLOCKED，不用文字推理猜金额。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

import jsonschema
import pandas as pd

from fundkit import policies, simulator
from fundkit.adapters import (ACCOUNT_DRAWDOWN_UNVERIFIABLE, CST,
                              LocalDataAdapter, load_account_snapshot)

SCHEMA_FILE = os.path.join("schemas", "decision.schema.json")
MODEL_VERSION = "algorithm-handoff-v1.0+local-impl-0.6"

ACTION_BY_STATE = {
    "NORMAL": "BUY_PROPOSAL",
    "PAUSE": "PAUSE_NEW_BUY",
    "LOCKED": "LOCKED",
    "STYLE_UNCERTAIN": "MANUAL_REVIEW",
}


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_dict(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_reports(decision: dict, out_dir="reports") -> tuple[str, str]:
    """strategy_report.md（简洁）+ full_research_audit.md（完整审计）。"""
    oos = {}
    oos_path = os.path.join(out_dir, "oos_summary.json")
    if os.path.exists(oos_path):
        oos = json.load(open(oos_path, encoding="utf-8"))

    s = ["# 策略报告（RESEARCH_ONLY）", "",
         f"asof: {decision['asof']}｜状态: {decision['decision_status']}｜"
         f"证据: {decision['evidence_status']}", "",
         f"每日预算 {decision['cash_budget_cny']:.0f} 元，"
         f"建议持有现金 {decision['recommended_cash_hold_cny']:.0f} 元", "",
         "## 逐基金建议", "",
         "| 代码 | 当前权重 | 目标权重 | 建议申购(元/日) | 动作 | 依据 |",
         "|---|---|---|---|---|---|"]
    for d in decision["fund_decisions"]:
        s.append(f"| {d['fund_code']} | {d['current_weight']:.1%} | "
                 f"{(d['target_weight'] or 0):.0%} | {d['subscribe_cny']:.0f} | "
                 f"{d['action']} | {', '.join(d['rule_ids']) or '-'} |")
    s += ["", "## 关键风险", ""]
    boot = (oos.get("stress") or {}).get("bootstrap") or {}
    if boot.get("p_mdd_exceed_5pct") is not None:
        ci = boot["ci95"]
        s.append(f"- 块自助压力（{boot['n_paths']}×{boot['horizon_days']}日）: "
                 f"P(MDD>5%) = {boot['p_mdd_exceed_5pct']:.1%} "
                 f"[{ci[0]:.1%}, {ci[1]:.1%}]，MDD 中位 {boot['mdd_median']:.2%}")
    crash = (oos.get("stress") or {}).get("bond_crash_2022") or {}
    if crash.get("portfolio_return") is not None:
        s.append(f"- 2022-11 债灾回放（当前权重）: {crash['portfolio_return']:.2%}")
    s += ["", "**注意**: 本报告为研究建议，不自动交易；账户为模拟批次口径，"
          "真实历史回撤不可验证（ACCOUNT_DRAWDOWN_UNVERIFIABLE）。", ""]
    p1 = os.path.join(out_dir, "strategy_report.md")
    open(p1, "w", encoding="utf-8").write("\n".join(s))

    a = ["# 完整研究审计", "", f"asof: {decision['asof']}", "",
         "## 1. 数据来源与口径", "",
         "- 净值：本地缓存（danjuan 主源，B/SINGLE_SOURCE），published_at 规则推断 T+2 自然日",
         "- 复权：分红/拆分事件重建（dividends.csv / splits.json），260 只远古对账不符已排除",
         "- 披露：东财 FundMNAssetAllocation + F10 zqcc，published_at 按法定披露时限规则推断，重仓为 lower_bound",
         "- 因子：中债国债收益率曲线（1/3/5/10/30Y）、沪深300 价格指数（非全收益）、上海金；信用利差/汇率 MISSING 不伪造",
         "- 账户：模拟批次（SIMULATED），" + ACCOUNT_DRAWDOWN_UNVERIFIABLE, "",
         "## 2. 方法与公式", "",
         "- TWR 单位净值：外部现金流按当时单位净值折算份额（external_cash_unitization）",
         "- 风险指标：MDD / CDaR95（Chekhlov 离散经验 CVaR，tests/finance_math.py 审计实现）",
         "- 风格识别：EWMA 加权岭回归（lookback/halflife/ridge/smooth 为候选超参）+ 披露断点双证据合并",
         "- 策略族 B0-B5：03_POLICY_ALGORITHMS.md；内层现金流分配为赤字注水近似（非全局最优 QP）",
         "- 反事实压力：CURRENT_STYLE_REWEIGHTED = 当前 beta × 历史因子冲击 + 近期残差自助，非真实业绩", "",
         "## 3. 样本外验证", ""]
    if oos:
        a += [f"- 折: {json.dumps(oos.get('folds'), ensure_ascii=False)}",
              f"- 网格大小: {oos.get('grid_size')}（全部试验见 reports/oos_trials.csv）",
              f"- 可行性判定: **{oos.get('decision')}**", "",
              "### 各策略 test 折结果", "",
              "| 策略 | 全部折满足 CAGR≥4% 且 MDD≤5% | 明细 |", "|---|---|---|"]
        for fam, v in (oos.get("feasibility") or {}).items():
            det = "; ".join(f"fold{f['fold']} CAGR {f['cagr']:.2%} MDD {f['max_drawdown']:.2%}"
                            for f in v["folds"])
            a.append(f"| {fam} | {'是' if v['all_folds_cagr_ge_4pct_and_mdd_le_5pct'] else '否'} | {det} |")
        a += ["", "### 帕累托前沿（test 均值）", ""]
        for p in oos.get("pareto_front_test_mean", []):
            a.append(f"- {p['family']}: CAGR {p['cagr']:.2%} / MDD {p['max_drawdown']:.2%}")
        a += ["", "### 冻结参数（每 fold 所选）", ""]
        for fam, sels in (oos.get("selected") or {}).items():
            for s_ in sels:
                a.append(f"- {fam} fold{s_['fold']}: {json.dumps(s_['params'], ensure_ascii=False)}")
    else:
        a.append("（尚未运行 python main.py validate）")
    a += ["", "## 4. 失败与局限（如实记录）", "",
          "- 利率因子回归 beta≈0：日频债基收益以 carry 为主，R² 低；风格状态多依赖披露证据，"
          "STYLE_UNCERTAIN 从严",
          "- B2 候选现金下限 10% 导致短期全留现金（候选参数语义，第5步网格含 0/5%/10% 对比）",
          "- 反事实起点组合非用户真实历史；真实账户 MDD 不可验证",
          "- QDII 成交用最近披露净值近似；费率默认 0（未知，费用压力测试见 oos_trials）",
          "- 沪深300 为价格指数，权益 beta 略低估；信用利差缺失使信用冲击以披露 lower_bound 表达", "",
          "状态: RESEARCH_ONLY", ""]
    p2 = os.path.join(out_dir, "full_research_audit.md")
    open(p2, "w", encoding="utf-8").write("\n".join(a))
    return p1, p2


def build_decision(asof: str | None = None, oos_summary_path="reports/oos_summary.json",
                   style_report_path="reports/style_report.json") -> dict:
    adapter = LocalDataAdapter()
    snap = load_account_snapshot()
    params = policies.load_params()
    asof_ts = pd.Timestamp(asof).tz_localize(CST) if asof else pd.Timestamp.now(
        datetime.now().astimezone().tzinfo).tz_convert(CST)

    # 数据 hash（输入可复现性）
    nav_cache = {c: adapter.nav([c], asof_ts) for c in [f["code"] for f in snap["funds"]]}
    input_hash = _hash_dict({
        "nav_latest": {c: str(df["nav_date"].max().date())
                       for c, df in nav_cache.items() if not df.empty},
        "factors": _sha256(os.path.join("data", "factors", "daily.csv")),
        "disclosures": {c: _sha256(os.path.join("data", "disclosures", f"{c}.csv"))
                        for c in [f["code"] for f in snap["funds"]]},
    })

    # 冻结策略：B5 参数取样本外验证所选（不可得时回退候选默认并注明）
    frozen_note = "候选默认（未完成样本外选择）"
    if os.path.exists(oos_summary_path):
        oos = json.load(open(oos_summary_path, encoding="utf-8"))
        sel = (oos.get("selected") or {}).get("B5") or []
        if sel:
            params["B5"] = {**params["B5"], **sel[-1]["params"]}
            frozen_note = f"样本外验证所选（fold {sel[-1]['fold']}）"
        oos_decision = oos.get("decision")
        stress = oos.get("stress", {})
    else:
        oos_decision, stress = None, {}

    # 当前账户状态（模拟批次，已公开净值估值）
    lots_df = adapter.cashflows_and_lots()
    pub = pd.concat([df for df in nav_cache.values() if not df.empty])
    # 估值必须用单位净值（原始价）：份额按原始净值确认，乘复权净值会重复计分红
    latest_nav = {c: float(g.iloc[-1]["unit_nav"]) for c, g in pub.groupby("fund_code")}
    style_report = {}
    if os.path.exists(style_report_path):
        style_report = json.load(open(style_report_path, encoding="utf-8"))

    total = float(sum(r["shares_confirmed"] * latest_nav.get(r["fund_code"], 0)
                      for _, r in lots_df.iterrows()))
    weights = {r["fund_code"]: r["shares_confirmed"] * latest_nav.get(r["fund_code"], 0) / total
               for _, r in lots_df.iterrows()} if total > 0 else {}

    target = {k: v for k, v in params["target_weights"].items() if not k.startswith("_")}
    fund_decisions = []
    for f in snap["funds"]:
        code = f["code"]
        style_info = style_report.get(code, {})
        style_state = style_info.get("current_style_state")
        shift = (style_info.get("combined_shift_evidence") or {}).get("state")
        value = weights.get(code, 0) * total
        tw = target.get(code, 0.0)
        # 动作推导：股指默认禁新买；其余按原计划继续（提案性质，RESEARCH_ONLY）
        if f["category"] == "equity_index":
            action, rule_ids = "PAUSE_NEW_BUY", ["B1.no_new_equity_index"]
            sub = 0.0
        else:
            sub = f["daily_plan_cny"]
            action = "BUY_PROPOSAL" if sub > 0 else "HOLD"
            rule_ids = ["B0.continue_existing_plan"]
        if style_state in (None, "UNKNOWN") and f["category"] == "bond":
            rule_ids.append("style.UNKNOWN_uncertain_widen_stress")
        if code == "021559":
            rule_ids.append("rolling_lock_30d_per_lot")
        if shift in ("POSSIBLE_SHIFT", "CONFIRMED_SHIFT"):
            rule_ids.append(f"style_shift.{shift}")
        dq = "SINGLE_SOURCE"
        rationale = (
            f"当前权重 {weights.get(code, 0):.1%}（目标 {tw:.0%}）；"
            f"风格 {style_state or '未评估'}"
            + (f"，迁移证据 {shift}" if shift else "")
            + f"；{frozen_note}。RESEARCH_ONLY，不构成交易指令")
        fund_decisions.append({
            "fund_code": code,
            "current_value_cny": round(value, 2),
            "current_weight": round(weights.get(code, 0), 4),
            "target_weight": tw,
            "subscribe_cny": sub,
            "estimated_confirmed_units": (round(sub / latest_nav[code], 2)
                                        if sub > 0 and code in latest_nav else None),
            "redeem_review_cny": None,
            "redeemable_lots": None,   # 模拟批次下不假造可赎回明细
            "next_redeem_date": ("2026-12-05" if code == "021559" else None),
            "action": action,
            "rule_ids": rule_ids,
            "style_state": style_state,
            "style_confidence": None,  # 不输出伪精确概率
            "data_quality": dq,
            "available_asof": asof_ts.isoformat(),
            "source_refs": ["data/nav", "data/disclosures", "data/factors/daily.csv",
                            "reports/style_report.json", "reports/oos_summary.json"],
            "rationale": rationale,
        })

    decision = {
        "asof": asof_ts.isoformat(),
        "model_version": MODEL_VERSION,
        "config_hash": _sha256(os.path.join("config", "policy_params.json")),
        "input_data_hash": input_hash,
        "decision_status": ("RESEARCH_ONLY" if oos_decision != "NO_FEASIBLE_STRATEGY"
                            else "NO_FEASIBLE_STRATEGY"),
        "evidence_status": "PARTIAL",
        "cash_budget_cny": snap["new_cash_budget_cny_per_execution_day"],
        "recommended_cash_hold_cny": (
            snap["new_cash_budget_cny_per_execution_day"]
            - sum(d["subscribe_cny"] for d in fund_decisions)),
        "portfolio_current_mdd": None,  # ACCOUNT_DRAWDOWN_UNVERIFIABLE
        "portfolio_oos_cagr": None,
        "portfolio_oos_sharpe": None,
        "portfolio_oos_mdd": None,
        "portfolio_oos_cdar95": None,
        "scenario_mdd_exceed_5pct_rate": (
            (stress.get("bootstrap") or {}).get("p_mdd_exceed_5pct")),
        "fund_decisions": fund_decisions,
        "_notes": [
            ACCOUNT_DRAWDOWN_UNVERIFIABLE,
            "本输出为研究建议（RESEARCH_ONLY），不自动交易；赎回仅审议性质",
            f"参数状态：{frozen_note}",
        ],
    }

    schema = json.load(open(SCHEMA_FILE, encoding="utf-8"))
    jsonschema.validate({k: v for k, v in decision.items() if not k.startswith("_")},
                        schema)
    return decision


def write_decision(decision: dict, out_dir="decisions") -> str:
    os.makedirs(out_dir, exist_ok=True)
    day = decision["asof"][:10]
    path = os.path.join(out_dir, f"{day}.json")
    clean = {k: v for k, v in decision.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2, default=str)
    # CSV 对照表
    rows = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
             for k, v in d.items()} for d in decision["fund_decisions"]]
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join("reports", "fund_decisions.csv"),
                              index=False, encoding="utf-8-sig")
    return path
