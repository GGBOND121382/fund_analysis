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
MODEL_VERSION = "algorithm-joint-v0.9"

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
        with open(oos_path, encoding="utf-8") as fh:
            candidate = json.load(fh)
        # 禁止将旧模拟器产出的业绩纳入本次报告。
        if candidate.get("validation_version") == MODEL_VERSION:
            oos = candidate

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
                 f"{f'{d["target_weight"]:.0%}' if d['target_weight'] is not None else '未验证'} | {d['subscribe_cny']:.0f} | "
                 f"{d['action']} | {', '.join(d['rule_ids']) or '-'} |")
    s += ["", "## 关键风险", ""]
    boot = (oos.get("stress") or {}).get("bootstrap") or {}
    if boot.get("p_mdd_exceed_4pct") is not None:
        ci = boot["ci95"]
        s.append(f"- 块自助压力（{boot['n_paths']}×{boot['horizon_days']}日）: "
                 f"P(MDD>4%) = {boot['p_mdd_exceed_4pct']:.1%} "
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
              "| 策略 | 全部折满足 CAGR>5% 且 MDD≤4% | 明细 |", "|---|---|---|"]
        for fam, v in (oos.get("feasibility") or {}).items():
            det = "; ".join(f"fold{f['fold']} CAGR {f['cagr']:.2%} MDD {f['max_drawdown']:.2%}"
                            for f in v["folds"])
            a.append(f"| {fam} | {'是' if v['all_folds_cagr_gt_5pct_and_mdd_le_4pct'] else '否'} | {det} |")
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
    """严格 fail-closed：无真实净值或未复验回测时，禁止输出正数交易建议。

    截图只用于展示历史快照金额/比例；模拟批次不可推导真实可赎回日。
    已有旧 OOS 报告与修正后算法版本不兼容，必须重新验证并冻结。
    """
    adapter = LocalDataAdapter()
    snap = load_account_snapshot()
    params = policies.load_params()
    asof_ts = pd.Timestamp(asof).tz_localize(CST) if asof else pd.Timestamp.now(tz=CST)
    funds = snap["funds"]
    codes = [f["code"] for f in funds]
    nav_cache = {c: adapter.nav([c], asof_ts) for c in codes}
    latest_nav = {c: float(df.iloc[-1]["unit_nav"])
                  for c, df in nav_cache.items() if not df.empty}
    missing = [c for c in codes if c not in latest_nav]
    snapshot_total = sum(float(f["value_cny"]) for f in funds)
    if snapshot_total <= 0:
        raise ValueError("DATA_BLOCKED: 账户快照总额非正")

    # 旧版 oos_summary 并未验证修复后的算法，不能当作可执行策略结果。
    verified = False
    oos = {}
    if os.path.exists(oos_summary_path):
        with open(oos_summary_path, encoding="utf-8") as fh:
            oos = json.load(fh)
        verified = (oos.get("validation_version") == MODEL_VERSION and
                    oos.get("status") == "VALIDATED" and
                    oos.get("decision") == "FEASIBLE_CANDIDATES_EXIST")
    # 没有完整账户历史、锁定期批次与费用时，即使有验证也只输出研究候选。
    blocked = bool(missing)
    status = "DATA_BLOCKED" if blocked else "MANUAL_REVIEW"
    note = (f"净值缺失: {','.join(missing)}；" if blocked else "") + (
        "上次回测参数与本次修复代码不匹配；必须重新跑完整样本外验证。"
        if not verified else "仅研究候选；真实账户流水及费率未核验。")
    # 未经独立样本外验证的候选权重，禁止以“目标权重”身份出现在正式决策表。
    target = ({k: v for k, v in params["target_weights"].items()
               if not k.startswith("_")} if verified else {})
    style_report = {}
    if os.path.exists(style_report_path):
        with open(style_report_path, encoding="utf-8") as fh:
            style_report = json.load(fh)

    # 只在输入完整且优化器真实可被调用时，计算观察性 B2 配置建议。
    # 无论如何不得把原日定投计划默认为优化结果。
    suggested = {}
    if not blocked:
        from fundkit import storage
        limits = {}
        st = storage.load_trade_status()
        if len(st) and "code" in st and "daily_limit" in st:
            for _, row in st.iterrows():
                if pd.notna(row["daily_limit"]) and float(row["daily_limit"]) > 0:
                    limits[str(row["code"])] = float(row["daily_limit"])
        p2 = {**params, "daily_limits": limits,
              "equity_index_codes": codes and [f["code"] for f in funds
                                               if f["category"] == "equity_index"]}
        # 估值仅用快照金额作为观察性代理，不据此推导真实持仓批次。
        state = {"cash": float(snap["new_cash_budget_cny_per_execution_day"]),
                 "pending_value": 0.0,
                 "value_quality": "COMPLETE",
                 "holdings": {f["code"]: {"shares": f["value_cny"] / latest_nav[f["code"]],
                                           "value_pub": f["value_cny"],
                                           "redeemable_shares": 0.0}
                              for f in funds},
                 "nav": {c: {"nav": nav} for c, nav in latest_nav.items()}}
        suggested = (policies.make_b2(p2)(state).get("subscribe") or {})

    fund_decisions = []
    for f in funds:
        code = f["code"]
        wi = float(f["value_cny"]) / snapshot_total
        style_info = style_report.get(code, {})
        sf = style_info.get("current_style_state")
        is_eq = f["category"] == "equity_index"
        # 没有重跑验证前，只呈现算法的研究候选；实际交易金额必须置0。
        sub = 0.0
        action = "PAUSE_NEW_BUY" if is_eq else ("DATA_BLOCKED" if blocked else "HOLD")
        rule_ids = (["B1.no_new_equity_index"] if is_eq else
                    ["DATA_BLOCKED.no_nav"] if blocked else
                    ["VALIDATION_REQUIRED.not_trade_ready"])
        fund_decisions.append({
            "fund_code": code, "current_value_cny": round(float(f["value_cny"]), 2),
            "current_weight": round(wi, 6), "target_weight": target.get(code),
            "subscribe_cny": sub, "estimated_confirmed_units": None,
            "redeem_review_cny": None, "redeemable_lots": None,
            "next_redeem_date": None, "action": action,
            "rule_ids": rule_ids,
            "style_state": sf, "style_confidence": None,
            "data_quality": "MISSING" if code in missing else "PROXY",
            "available_asof": asof_ts.isoformat(),
            "source_refs": ["config/account_snapshot.json", "data/nav"],
            "rationale": (f"截图历史快照占比 {wi:.2%}；观察性B2分配"
                          f" {suggested.get(code, 0):.2f}元/日（非交易建议）。"
                          f"{note} 未知真实份额和可赎回日期，禁止自动交易。"),
        })

    budget = float(snap["new_cash_budget_cny_per_execution_day"])
    decision = {
        "asof": asof_ts.isoformat(), "model_version": MODEL_VERSION,
        "config_hash": _sha256(os.path.join("config", "policy_params.json")),
        "input_data_hash": _hash_dict({
            "nav_latest": {c: str(df["nav_date"].max().date())
                           for c, df in nav_cache.items() if not df.empty},
            "snapshot": _sha256("config/account_snapshot.json")}),
        "decision_status": status,
        "evidence_status": "INSUFFICIENT" if blocked else "PARTIAL",
        "cash_budget_cny": budget, "recommended_cash_hold_cny": budget,
        "portfolio_current_mdd": None, "portfolio_oos_cagr": None,
        "portfolio_oos_sharpe": None, "portfolio_oos_mdd": None,
        "portfolio_oos_cdar95": None, "scenario_mdd_exceed_4pct_rate": None,
        "fund_decisions": fund_decisions,
        "_notes": [ACCOUNT_DRAWDOWN_UNVERIFIABLE, note,
                   "任何观察性B2权重仅供数学调试，不是交易指令；无账户流水不推断赎回日。"],
    }
    with open(SCHEMA_FILE, encoding="utf-8") as fh:
        schema = json.load(fh)
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