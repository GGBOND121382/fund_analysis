# PARAMETER_SENSITIVITY — 参数敏感性

生成: 2026-11-10T23:26:13.832418+08:00｜代码版本: bec32bd

## 1. 权重优化对约束/种子/回看窗的敏感性

| 变体 | 状态 | 场景MDD P95 | 场景一年收益Q25 | 前三大权重 |
|---|---|---|---|---|
| base | SCENARIO_CRITERIA_MET_UNVALIDATED | 1.07% | 6.09% | 021559 60%, 004400 26%, 014846 8% |
| lb504 | SCENARIO_CRITERIA_MET_UNVALIDATED | 1.07% | 6.09% | 021559 60%, 004400 26%, 014846 8% |
| mw0.4 | SCENARIO_CRITERIA_MET_UNVALIDATED | 0.77% | 5.26% | 004400 40%, 021559 40%, 014846 11% |
| mw0.8 | SCENARIO_CRITERIA_MET_UNVALIDATED | 1.51% | 6.70% | 021559 80%, 004400 9%, cash 6% |
| seed1 | SCENARIO_CRITERIA_MET_UNVALIDATED | 1.17% | 5.98% | 021559 60%, 004400 27%, 014846 5% |
| seed7 | SCENARIO_CRITERIA_MET_UNVALIDATED | 0.82% | 6.12% | 021559 58%, 004400 36%, 014846 3% |

**结论：权重对种子与回看窗稳健，但稳定地顶格集中**——021559 始终顶到单基金上限（0.4/0.6/0.8 随约束同步放大）。这说明结果由 021559 近一年 9.8% 的区间收益驱动，是约束边界解而非内点最优：上限约束在起作用，情景达标的可信度取决于该基金的近期表现能否持续，而这正是样本外验证（被 DATA_BLOCKED）要回答的问题。

## 2. 策略规则网格敏感性（validate 全部试验，val 阶段）

| 策略族 | 参数 | fold | val CAGR | val MDD | val Sharpe |
|---|---|---|---|---|---|
| B0 | {} | 0 | 6.10% | 0.39% | 3.46 |
| B0 | {} | 1 | 5.28% | 0.64% | 2.85 |
| B0 | {} | 2 | 6.10% | 0.39% | 3.46 |
| B0 | {} | 3 | 1.96% | 1.27% | 0.01 |
| B1 | {} | 0 | 5.00% | 0.34% | 3.21 |
| B1 | {} | 1 | 4.24% | 0.44% | 2.56 |
| B1 | {} | 2 | 5.00% | 0.34% | 3.21 |
| B1 | {} | 3 | 0.70% | 1.11% | -0.43 |
| B2 | {"cash": 0.05} | 0 | 3.90% | 0.28% | 3.24 |
| B2 | {"cash": 0.0} | 0 | 4.07% | 0.30% | 3.37 |
| B2 | {"cash": 0.1} | 0 | 3.77% | 0.27% | 3.16 |
| B2 | {"cash": 0.05} | 1 | 4.50% | 0.07% | 6.99 |
| B2 | {"cash": 0.0} | 1 | 4.73% | 0.08% | 7.28 |
| B2 | {"cash": 0.1} | 1 | 4.32% | 0.08% | 6.67 |
| B2 | {"cash": 0.05} | 2 | 3.90% | 0.28% | 3.24 |
| B2 | {"cash": 0.0} | 2 | 4.07% | 0.30% | 3.37 |
| B2 | {"cash": 0.1} | 2 | 3.77% | 0.27% | 3.16 |
| B2 | {"cash": 0.05} | 3 | 0.51% | 1.42% | -0.38 |
| B2 | {"cash": 0.0} | 3 | 0.52% | 1.50% | -0.36 |
| B2 | {"cash": 0.1} | 3 | 0.48% | 1.37% | -0.41 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 0 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 0 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 0 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 0 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 1 | 4.73% | 0.08% | 7.28 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 1 | 4.73% | 0.08% | 7.28 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 1 | 4.73% | 0.08% | 7.28 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 1 | 4.73% | 0.08% | 7.28 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 2 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 2 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 2 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 2 | 4.07% | 0.30% | 3.37 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 3 | 0.52% | 1.50% | -0.36 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 3 | 0.52% | 1.50% | -0.36 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 3 | 0.52% | 1.50% | -0.36 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 3 | 0.52% | 1.50% | -0.36 |
| B4 | {"pause_cdar95": 0.01} | 0 | 4.07% | 0.30% | 3.37 |
| B4 | {"pause_cdar95": 0.02} | 0 | 4.07% | 0.30% | 3.37 |
| B4 | {"pause_cdar95": 0.03} | 0 | 4.07% | 0.30% | 3.37 |
| B4 | {"pause_cdar95": 0.01} | 1 | 4.73% | 0.08% | 7.28 |
| B4 | {"pause_cdar95": 0.02} | 1 | 4.73% | 0.08% | 7.28 |
| B4 | {"pause_cdar95": 0.03} | 1 | 4.73% | 0.08% | 7.28 |
| B4 | {"pause_cdar95": 0.01} | 2 | 4.07% | 0.30% | 3.37 |
| B4 | {"pause_cdar95": 0.02} | 2 | 4.07% | 0.30% | 3.37 |
| B4 | {"pause_cdar95": 0.03} | 2 | 4.07% | 0.30% | 3.37 |
| B4 | {"pause_cdar95": 0.01} | 3 | 0.52% | 1.50% | -0.36 |
| B4 | {"pause_cdar95": 0.02} | 3 | 0.52% | 1.50% | -0.36 |
| B4 | {"pause_cdar95": 0.03} | 3 | 0.52% | 1.50% | -0.36 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 0 | 4.05% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 0 | 4.05% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 0 | 4.06% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 0 | 4.06% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 1 | 4.74% | 0.08% | 7.49 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 1 | 4.74% | 0.08% | 7.49 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 1 | 4.74% | 0.08% | 7.49 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 1 | 4.74% | 0.08% | 7.49 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 2 | 4.05% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 2 | 4.05% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 2 | 4.06% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 2 | 4.06% | 0.30% | 3.34 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 3 | 0.16% | 1.49% | -0.49 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 3 | 0.16% | 1.49% | -0.49 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 3 | 0.52% | 1.50% | -0.36 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 3 | 0.52% | 1.50% | -0.36 |

选参规则：val 阶段风险优先（先 MDD≤4%，再 CAGR 最大），选中参数冻结后仅在 test 复跑（见 OOS_BACKTEST_REPORT 第3节）。

## 3. 费用敏感性

- 真实申赎费率 MISSING；validate 默认费率 0（毛口径）；
- 联合优化设计为 0/30bp 统一费率双情景取最差（fee_stress），因联合优化 DATA_BLOCKED 未实际运行；
- 以组合近一年换手≈0（买入持有+小额定投）估算，30bp 单边对年化拖累 <0.1%，不改变 NO_FEASIBLE 的方向性结论（缺口在收益端，约 1.7 个百分点）。

---
状态: RESEARCH_ONLY
