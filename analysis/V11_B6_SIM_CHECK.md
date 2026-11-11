# B6 持仓比例配置与模拟验证记录（2026-11-11）

## 配置变更

`config/policy_params.json` 的 `target_weights` 由空仓占位 `{"cash": 1.0}` 改为
v1.0.1 权重优化的研究候选（`reports/weight_optimization_research.json`，asof 2026-11-11）：

| 代码 | 目标权重 | 当前权重 | 偏离 |
|---|---|---|---|
| 710301 | 38.24% | 55.89% | 超配 |
| 004400 | 24.93% | 41.64% | 超配 |
| 021559 | 20.92% | 0.72% | 低配最多 |
| 000218 | 5.19% | 0.58% | 低配 |
| 019441 | 4.80% | 0.02% | 低配（股指，禁新买） |
| 其余 QDII | ~5.2% | <1% | 低配（股指，禁新买） |
| cash | ≈0 | - | - |

状态声明：该权重为 RETURN_TARGET_UNMET 的研究候选（稳健收益 LCB 3.45% < 5% 目标），
未经独立样本外验证；`decide` 端 verified=False，不会输出为交易建议。

## 模拟结果（2026-11-05 ~ 2026-11-11，2 个披露日，冒烟口径）

命令：`python main.py sim --policy B2` / `--policy B6`

- **B2**：按目标权重注水分配，每日预算全部买入最低配的 021559
  （2026-11-05 买 1002.45 元，2026-11-06 买 1004.99 元）。组合 TWR -0.0113%。
- **B6**：`B6_CASH_PLAN approved=0.0, tactical_blocked=true`——全额递延。
  所有基金 `FUND_RISK_DATA_BLOCKED`：当前持仓阶段可用总回报净值少于 2 个观测，
  暂停新增申购且不推断止损赎回。fail-closed 行为符合设计。组合 TWR -0.0118%。
- B5 既有止损/止盈逻辑未被 B6 修改（B6 用逐批次 armed trailing 止盈替代组合级止盈）。

## 结论对照（v1.0.1 流水线，asof 2026-11-11）

| 模块 | 判定 |
|---|---|
| 权重优化（费后 30bp） | RETURN_TARGET_UNMET（稳健收益 LCB 3.45% < 5%） |
| 联合优化 | DATA_BLOCKED（共同历史 507 < 1008 交易日） |
| 样本外验证 | INSUFFICIENT_INDEPENDENT_OOS_WINDOWS（独立全年窗 1/2；主窗各策略 CAGR 1.6%~2.6%） |

状态: RESEARCH_ONLY。无自动交易。
