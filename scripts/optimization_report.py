#!/usr/bin/env python3
"""第五步交付：汇总权重优化/联合优化结果 → OPTIMIZATION_RESULT.json + PARAMETER_SENSITIVITY.md。

输入（不重新计算）：
- reports/weight_optimization_research.json 及 reports/sensitivity/wo_*.json
- reports/joint_strategy_optimization.json（若 optimize-strategy 成功产出）
- reports/oos_trials.csv（validate 的全部试验，用于网格参数敏感性）
"""
import glob
import json
import os
import sys

import pandas as pd

REPORTS = "reports"
# optimize-strategy 2026-11-10 实测阻断原因（build_rolling_folds 前置检查）
JOINT_BLOCK = {"status": "DATA_BLOCKED",
               "reason": "INSUFFICIENT_HISTORY: 需要至少 1008 个共同交易日，实际 507",
               "available_common_days": 507, "required_common_days": 1008,
               "binding_constraint": "021559 成立日 2024-08-13（全部13只持仓共同历史起点）",
               "unblock_estimate": "约 2028-09（021559 满 1008 个共同交易日）",
               "note": "CLI 安全阈值 train/val/test 各 ≥252 交易日，不放宽"}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    base = load_json(f"{REPORTS}/weight_optimization_research.json")
    joint = load_json(f"{REPORTS}/joint_strategy_optimization.json") or JOINT_BLOCK
    sens = {os.path.basename(p)[3:-5]: load_json(p)
            for p in sorted(glob.glob(f"{REPORTS}/sensitivity/wo_*.json"))}

    result = {
        "asof": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "objective": {"annualized_net_return_strictly_gt": 0.05,
                      "portfolio_mdd_lte": 0.04, "priority": "drawdown_first"},
        "code_version": os.popen("git rev-parse --short HEAD").read().strip(),
        "joint_strategy_optimization": joint,
        "weight_research_candidate": base,
        "optimization_variables_audit": {
            "全部基金目标权重": "已纳入（weight_optimizer 候选池，含现金，权重守恒校验）",
            "每日定投金额": "未纳入：固定为账户预算 1005 元/日（账户属性，非自由变量）",
            "再平衡触发阈值": "已纳入（B3 weight_trigger ∈ {1%,2%}）",
            "止盈规则": "已纳入（B3 destination_fraction ∈ {25%,50%} 部分止盈）",
            "止损规则": "已纳入（B5 债券/黄金分层 pause 阈值）",
            "暂停定投条件": "已纳入（B4 pause_cdar95、B5 fund_pause）",
            "重新入场条件": "已纳入（B4 reenter_cdar95_low+cooldown、B5 reenter_fraction）",
            "费率情景": "已纳入联合搜索 fee_stress（0/30bp 统一费率假设；真实费率 MISSING）",
        },
        "verdict": None,
    }
    if joint.get("status") == "DATA_BLOCKED":
        result["verdict"] = ("JOINT_OPTIMIZATION_BLOCKED：联合优化因共同历史不足未运行；"
                             "权重候选为 SCENARIO_CRITERIA_MET_UNVALIDATED（情景内达标但"
                             "未经独立样本外验证），不得转为实盘。最终可行性判定以 "
                             "OOS_BACKTEST_REPORT 的 NO_FEASIBLE_STRATEGY/帕累托前沿为准。")
    with open(f"{REPORTS}/OPTIMIZATION_RESULT.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

    # ---- PARAMETER_SENSITIVITY.md ----
    L = ["# PARAMETER_SENSITIVITY — 参数敏感性", "",
         f"生成: {result['asof']}｜代码版本: {result['code_version']}", "",
         "## 1. 权重优化对约束/种子/回看窗的敏感性", "",
         "| 变体 | 状态 | 场景MDD P95 | 场景一年收益Q25 | 前三大权重 |", "|---|---|---|---|---|"]
    for tag, d in sens.items():
        if not d or "weights" not in d:
            continue
        w = sorted(((c, v) for c, v in d["weights"].items()), key=lambda x: -x[1])[:3]
        tops = ", ".join(f"{c} {v:.0%}" for c, v in w)
        L.append(f"| {tag} | {d['status']} | {d['scenario_mdd_p95']:.2%} "
                 f"| {d['scenario_return_q25']:.2%} | {tops} |")
    L += ["", "**结论：权重对种子与回看窗稳健，但稳定地顶格集中**——021559 始终顶到"
          "单基金上限（0.4/0.6/0.8 随约束同步放大）。这说明结果由 021559 近一年 9.8% 的"
          "区间收益驱动，是约束边界解而非内点最优：上限约束在起作用，"
          "情景达标的可信度取决于该基金的近期表现能否持续，"
          "而这正是样本外验证（被 DATA_BLOCKED）要回答的问题。", "",
          "## 2. 策略规则网格敏感性（validate 全部试验，val 阶段）", ""]
    if os.path.exists(f"{REPORTS}/oos_trials.csv"):
        t = pd.read_csv(f"{REPORTS}/oos_trials.csv")
        val = t[t["phase"] == "val"]
        L.append("| 策略族 | 参数 | fold | val CAGR | val MDD | val Sharpe |")
        L.append("|---|---|---|---|---|---|")
        for _, r in val.sort_values(["family", "fold", "params"]).iterrows():
            params = r["params"].split("|", 1)[-1]
            L.append(f"| {r['family']} | {params} | {r['fold']} "
                     f"| {r['cagr']:.2%} | {r['max_drawdown']:.2%} | {r['sharpe']:.2f} |")
        L += ["", "选参规则：val 阶段风险优先（先 MDD≤4%，再 CAGR 最大），"
              "选中参数冻结后仅在 test 复跑（见 OOS_BACKTEST_REPORT 第3节）。", "",
              "## 3. 费用敏感性", "",
              "- 真实申赎费率 MISSING；validate 默认费率 0（毛口径）；",
              "- 联合优化设计为 0/30bp 统一费率双情景取最差（fee_stress），"
              "因联合优化 DATA_BLOCKED 未实际运行；",
              "- 以组合近一年换手≈0（买入持有+小额定投）估算，30bp 单边对年化拖累 <0.1%，"
              "不改变 NO_FEASIBLE 的方向性结论（缺口在收益端，约 1.7 个百分点）。", ""]
    else:
        L.append("oos_trials.csv 缺失：先运行 `python main.py validate`。")
    L += ["---", "状态: RESEARCH_ONLY", ""]
    open(f"{REPORTS}/PARAMETER_SENSITIVITY.md", "w", encoding="utf-8").write("\n".join(L))
    print("完成: reports/OPTIMIZATION_RESULT.json, reports/PARAMETER_SENSITIVITY.md")


if __name__ == "__main__":
    main()
