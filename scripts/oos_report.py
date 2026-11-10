#!/usr/bin/env python3
"""第四步交付：由 reports/oos_trials.csv + oos_summary.json 生成 OOS_BACKTEST_REPORT.md。

不重新回测，只整理 validate 的产出；全部数字可溯源到 oos_trials.csv 行。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPORTS = "reports"


def pct(x, digits=2):
    return "NA" if x is None or (isinstance(x, float) and x != x) else f"{x:.{digits}%}"


def main():
    with open(f"{REPORTS}/oos_summary.json", encoding="utf-8") as fh:
        summ = json.load(fh)
    trials = pd.read_csv(f"{REPORTS}/oos_trials.csv")

    L = ["# OOS_BACKTEST_REPORT — 滚动样本外验证", "",
         f"生成时间: {summ['asof']}｜算法版本: {summ.get('validation_version')}｜"
         f"试验总数: {len(trials)}（网格 {summ['grid_size']} 配置 × 折 × 阶段）", "",
         "口径：反事实起点组合（快照权重×标准化本金，非真实历史）；"
         "净收益口径=费后（费率 MISSING 默认 0，费用压力见联合优化 fee_stress）；"
         "MDD 在外部现金流单位化（TWR）净值上计算。", "",
         "## 1. 时间序列切分（train→val 选参→test 冻结复跑）", "",
         "| fold | train | val | test | test 长度 |", "|---|---|---|---|---|"]
    for i, f in enumerate(summ["folds"]):
        n = len(pd.bdate_range(*f["test"]))
        L.append(f"| {i} | {f['train'][0]}~{f['train'][1]} | {f['val'][0]}~{f['val'][1]} "
                 f"| {f['test'][0]}~{f['test'][1]} | {n} 交易日 |")
    L += ["", "fold3 的 24 个月测试窗跨 710301 风格切换（2025-08-29 确认纯债化），"
          "其 test 段含切换前权益/转债敞口，是该基金真实历史行为而非数据错误。", "",
          "## 2. 各策略族 test 折结果（冻结参数，全部试验见 oos_trials.csv）", "",
          "| 策略族 | fold | CAGR | XIRR | MDD | CDaR95 | Sharpe | Sortino | Calmar "
          "| 年化换手 | 费用(元) | 平均现金占比 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    test = trials[trials["phase"] == "test"]
    for _, r in test.sort_values(["family", "fold"]).iterrows():
        calmar = r["cagr"] / r["max_drawdown"] if r.get("max_drawdown", 0) > 0 else np.nan
        L.append(f"| {r['family']} | {r['fold']} | {pct(r['cagr'])} | {pct(r.get('xirr'))} "
                 f"| {pct(r['max_drawdown'])} | {pct(r.get('cdar95'))} "
                 f"| {r.get('sharpe', float('nan')):.2f} | {r.get('sortino', float('nan')):.2f} "
                 f"| {calmar:.2f} | {pct(r.get('turnover_ann', 0))} "
                 f"| {r.get('fees_total', 0):.0f} | {pct(r.get('cash_ratio_avg', 0))} |")

    L += ["", "## 3. 冻结参数记录（val 风险优先选择，test 不得参与调参）", ""]
    for fam, sels in (summ.get("selected") or {}).items():
        for s in sels:
            L.append(f"- {fam} fold{s['fold']}: `{json.dumps(s['params'], ensure_ascii=False)}`"
                     f"（val MDD {pct(s['val_mdd'])}）")

    L += ["", "## 4. 可行性判定（验收：CAGR>5% 且 MDD≤4%，全部 test 折同时满足）", "",
          f"**判定: {summ['decision']}**", "",
          "| 策略族 | 全部折满足 | 各折 CAGR/MDD |", "|---|---|---|"]
    for fam, v in (summ.get("feasibility") or {}).items():
        det = "; ".join(f"fold{f['fold']}: {pct(f['cagr'])}/{pct(f['max_drawdown'])}"
                        for f in v["folds"])
        L.append(f"| {fam} | {'是' if v['all_folds_cagr_gt_5pct_and_mdd_le_4pct'] else '否'} | {det} |")

    L += ["", "## 5. 收益—回撤有效前沿（test 折均值）", "",
          "| 策略族 | CAGR | MDD |", "|---|---|---|"]
    for pp in summ.get("pareto_front_test_mean", []):
        L.append(f"| {pp['family']} | {pct(pp['cagr'])} | {pct(pp['max_drawdown'])} |")

    boot = (summ.get("stress") or {}).get("bootstrap") or {}
    crash = (summ.get("stress") or {}).get("bond_crash_2022") or {}
    L += ["", "## 6. 压力测试", ""]
    if "p_mdd_exceed_4pct" in boot:
        ci = boot["ci95"]
        L.append(f"- 块自助（{boot['n_paths']} 路径 × {boot['horizon_days']} 日，"
                 f"块长 {boot['block_days']}，种子42，共同日期 {boot['n_common_dates']} 天）: "
                 f"P(MDD>4%) = {pct(boot['p_mdd_exceed_4pct'], 1)} "
                 f"[{pct(ci[0], 1)}, {pct(ci[1], 1)}]，MDD 中位 {pct(boot['mdd_median'])}，"
                 f"P95 {pct(boot['mdd_p95'])}")
    else:
        L.append(f"- 块自助: {boot.get('error', '未运行')}")
    if crash.get("portfolio_return") is not None:
        L.append(f"- 2022-11 债灾回放（当前权重）: 组合 {pct(crash['portfolio_return'])}，"
                 f"MDD {pct(crash.get('portfolio_max_drawdown'))}（反事实回放，无风格迁移修正）")
    else:
        L.append(f"- 2022-11 债灾回放: {crash.get('error', '未运行')}")
    L.append("- 注：利率/信用利差/权益/黄金/汇率的**联合参数化**压力测试受限于因子缺失"
             "（信用利差、汇率 MISSING），当前以历史真实窗口回放 + 块自助替代，"
             "债券信用敞口以季报披露 lower_bound 表达（见 PORTFOLIO_RISK_REPORT 第6节）。")

    if summ["decision"] == "NO_FEASIBLE_STRATEGY" or summ["decision"] == "INSUFFICIENT_OOS_HORIZON":
        L += ["", "## 7. 不可行原因分解", ""]
        testg = test.copy()
        for fam, grp in testg.groupby("family"):
            bad_ret = grp[grp["cagr"] <= 0.05]
            bad_mdd = grp[grp["max_drawdown"] > 0.04]
            reasons = []
            if len(bad_ret):
                reasons.append(f"收益不达 5%（{len(bad_ret)}/{len(grp)} 折，"
                               f"最低 {pct(bad_ret['cagr'].min())}）")
            if len(bad_mdd):
                reasons.append(f"回撤超 4%（{len(bad_mdd)}/{len(grp)} 折，"
                               f"最差 {pct(bad_mdd['max_drawdown'].max())}）")
            L.append(f"- {fam}: " + ("；".join(reasons) if reasons else "满足"))
        L += ["", "结论：不放宽 5%/4% 目标；以上前沿即当前数据与策略族下的客观边界。", ""]

    L += ["---", f"状态: {summ.get('status')}｜数据截止见 DATA_COVERAGE｜"
          f"复现: `python main.py validate`（确定性，无随机性；块自助种子42）", ""]
    open(f"{REPORTS}/OOS_BACKTEST_REPORT.md", "w", encoding="utf-8").write("\n".join(L))
    print("完成: reports/OOS_BACKTEST_REPORT.md")


if __name__ == "__main__":
    main()
