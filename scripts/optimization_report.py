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
JOINT_BLOCK = {"status": "NOT_RUN",
               "reason": "尚无本次运行的 reports/joint_strategy_optimization.json；禁止复用旧版数据阻断结论"}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    base = load_json(f"{REPORTS}/weight_optimization_research.json") or {"status": "NOT_RUN"}
    joint = load_json(f"{REPORTS}/joint_strategy_optimization.json") or JOINT_BLOCK
    sens = {os.path.basename(p)[3:-5]: load_json(p)
            for p in sorted(glob.glob(f"{REPORTS}/sensitivity/wo_*.json"))}

    result = {
        "asof": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "objective": {"annualized_net_return_strictly_gt": 0.05,
                      "portfolio_mdd_lte": 0.04, "priority": "drawdown_and_return_constraints_then_sharpe"},
        "code_version": os.popen("git rev-parse --short HEAD").read().strip(),
        "joint_strategy_optimization": joint,
        "weight_research_candidate": base,
        "optimization_variables_audit": {
            "全部基金目标权重": "已纳入（债券、黄金、沪深300、标普500、纳斯达克指数基金均作为基金资产类别；不含直接个股）",
            "每日定投金额": "固定为账户预算（账户属性，不作为自由优化变量）",
            "再平衡规则": "已纳入（B3 权重偏离阈值与部分赎回比例）",
            "止盈规则": "已纳入（B3 部分再平衡 + B5 基金成本基础上的移动止盈）",
            "止损规则": "已纳入（B5 基金级动态暂停/分批赎回 + 组合回撤保护）",
            "Sharpe": "已纳入（收益/MDD约束达标候选按费后情景Sharpe Q25排序；无风险利率默认2%假设）",
            "暂停/重新入场": "已纳入（B4 风格CDaR迟滞 + B5 回撤恢复和冷却期）",
            "申赎费用": "首次反事实建仓、后续申购和赎回均计入统一费率压力情景；实际费率表仍缺失",
        },
        "verdict": None,
    }
    if joint.get("status") in ("DATA_BLOCKED", "PARTIAL_DATA_BLOCKED"):
        if base.get("status") == "DATA_BLOCKED":
            result["verdict"] = ("DATA_BLOCKED：权重优化与联合优化均未取得可用结果；"
                                 "不得引用 analysis/ 中旧版 v0.9 数字作为本次验证。")
        elif base.get("status") == "SCENARIO_CRITERIA_MET_UNVALIDATED":
            result["verdict"] = ("JOINT_OPTIMIZATION_BLOCKED：联合权重/退出规则优化未完成；"
                                 "单独权重候选仅通过历史情景筛选，未经独立时间样本外验证，"
                                 "不得转为实盘。")
        else:
            result["verdict"] = "JOINT_OPTIMIZATION_BLOCKED：联合优化数据不足或运行被阻断；不得转为实盘。"
    elif joint.get("status") == "RESEARCH_ONLY":
        result["verdict"] = ("已生成联合优化研究结果；仍须检查每折验证/测试约束、费用情景、"
                             "Sharpe、样本独立性及数据质量，RESEARCH_ONLY 不等于实盘通过。")
    else:
        result["verdict"] = "尚无可验收的联合优化结果；保持 RESEARCH_ONLY。"
    with open(f"{REPORTS}/OPTIMIZATION_RESULT.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

    # ---- PARAMETER_SENSITIVITY.md ----
    L = ["# PARAMETER_SENSITIVITY — 参数敏感性", "",
         f"生成: {result['asof']}｜代码版本: {result['code_version']}", "",
         "## 1. 权重优化对约束/种子/回看窗的敏感性", "",
         "| 变体 | 状态 | 场景MDD P95 | 场景收益Q25 | Sharpe Q25 | 前三大权重 |",
         "|---|---|---|---|---|---|"]
    for tag, d in sens.items():
        if not d or "weights" not in d:
            continue
        w = sorted(((c, v) for c, v in d["weights"].items()), key=lambda x: -x[1])[:3]
        tops = ", ".join(f"{c} {v:.0%}" for c, v in w)
        L.append(f"| {tag} | {d['status']} | {d['scenario_mdd_p95']:.2%} "
                 f"| {d['scenario_return_q25']:.2%} | {d.get('scenario_sharpe_q25', float('nan')):.2f} | {tops} |")
    if base.get("status") == "DATA_BLOCKED":
        L += ["", f"本次权重搜索被数据阻断：{base.get('reason', '原因未记录')}。"]
    elif base.get("weights"):
        top = sorted(((c, v) for c, v in base["weights"].items() if c != "cash"),
                     key=lambda x: -x[1])[:3]
        L += ["", "本次权重候选（仅情景筛选，不是实盘目标）："
              + "、".join(f"{c} {v:.1%}" for c, v in top)
              + f"；现金 {base['weights'].get('cash', 0):.1%}。"
              + f" Sharpe Q25={base.get('scenario_sharpe_q25', float('nan')):.2f}。"]
    else:
        L += ["", "本次权重搜索未生成可展示候选；查看 weight_optimization_research.json 的阻断原因。"]
    L += ["", "## 2. 策略规则网格敏感性（validate 全部试验，val 阶段）", ""]
    if os.path.exists(f"{REPORTS}/oos_trials.csv"):
        t = pd.read_csv(f"{REPORTS}/oos_trials.csv")
        val = t[t["phase"] == "val"]
        L.append("| 策略族 | 参数 | fold | val CAGR | val MDD | val Sharpe |")
        L.append("|---|---|---|---|---|---|")
        for _, r in val.sort_values(["family", "fold", "params"]).iterrows():
            params = r["params"].split("|", 1)[-1]
            sharpe = float(r["sharpe"]) if pd.notna(r.get("sharpe")) else float("nan")
            L.append(f"| {r['family']} | {params} | {r['fold']} "
                     f"| {r['cagr']:.2%} | {r['max_drawdown']:.2%} | {sharpe:.2f} |")
        L += ["", "选参规则：验证期先检查年化收益>5%且MDD≤4%；同时达标时按Sharpe优先，"
              "再比较CAGR/MDD。若收益目标未达标但MDD可行，则优先提高CAGR；若MDD也不达标，"
              "优先选择MDD较小的诊断候选。选中参数冻结后仅在test复跑。", "",
              "## 3. 费用敏感性", "",
              "- 真实逐基金申购/赎回费率仍缺失，30/60bp仅为统一单边压力假设；",
              "- v1.0.1 将首次反事实建仓费、后续申购费和赎回费统一计入模拟净值、Sharpe、XIRR和换手；",
              "- 不得把统一费率压力情景等同于真实账户净收益。", ""]
    else:
        L.append("oos_trials.csv 缺失：先运行 `python main.py validate`。")
    L += ["---", "状态: RESEARCH_ONLY", ""]
    open(f"{REPORTS}/PARAMETER_SENSITIVITY.md", "w", encoding="utf-8").write("\n".join(L))
    print("完成: reports/OPTIMIZATION_RESULT.json, reports/PARAMETER_SENSITIVITY.md")


if __name__ == "__main__":
    main()
