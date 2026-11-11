# PARAMETER_SENSITIVITY — 参数敏感性

生成: 2026-11-11T04:53:22.218829+08:00｜代码版本: 3563f76

## 1. 权重优化对约束/种子/回看窗的敏感性

| 变体 | 状态 | 场景MDD P95 | 场景收益Q25 | Sharpe Q25 | 前三大权重 |
|---|---|---|---|---|---|

本次权重候选（仅情景筛选，不是实盘目标）：710301 38.2%、004400 24.9%、021559 20.9%；现金 0.0%。 Sharpe Q25=1.39。

## 2. 策略规则网格敏感性（validate 全部试验，val 阶段）

| 策略族 | 参数 | fold | val CAGR | val MDD | val Sharpe |
|---|---|---|---|---|---|
| B0 | {} | 0 | 5.15% | 0.39% | 2.46 |
| B0 | {} | 1 | 4.19% | 0.66% | 1.77 |
| B0 | {} | 2 | 5.15% | 0.39% | 2.46 |
| B0 | {} | 3 | 0.96% | 1.41% | -0.28 |
| B1 | {} | 0 | 4.12% | 0.35% | 2.03 |
| B1 | {} | 1 | 3.22% | 0.46% | 1.26 |
| B1 | {} | 2 | 4.12% | 0.35% | 2.03 |
| B1 | {} | 3 | -0.23% | 1.24% | -0.75 |
| B2 | {"cash": 0.05} | 0 | 3.04% | 0.30% | 1.39 |
| B2 | {"cash": 0.0} | 0 | 3.18% | 0.32% | 1.53 |
| B2 | {"cash": 0.1} | 0 | 2.94% | 0.30% | 1.29 |
| B2 | {"cash": 0.05} | 1 | 3.53% | 0.30% | 2.59 |
| B2 | {"cash": 0.0} | 1 | 3.72% | 0.31% | 2.81 |
| B2 | {"cash": 0.1} | 1 | 3.39% | 0.30% | 2.39 |
| B2 | {"cash": 0.05} | 2 | 3.04% | 0.30% | 1.39 |
| B2 | {"cash": 0.0} | 2 | 3.18% | 0.32% | 1.53 |
| B2 | {"cash": 0.1} | 2 | 2.94% | 0.30% | 1.29 |
| B2 | {"cash": 0.05} | 3 | -0.41% | 1.58% | -0.62 |
| B2 | {"cash": 0.0} | 3 | -0.43% | 1.66% | -0.60 |
| B2 | {"cash": 0.1} | 3 | -0.40% | 1.50% | -0.66 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 0 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 0 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 0 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 0 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 1 | 3.72% | 0.31% | 2.81 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 1 | 3.72% | 0.31% | 2.81 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 1 | 3.72% | 0.31% | 2.81 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 1 | 3.72% | 0.31% | 2.81 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 2 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 2 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 2 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 2 | 3.18% | 0.32% | 1.53 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.02} | 3 | -0.43% | 1.66% | -0.60 |
| B3 | {"destination_fraction": 0.25, "weight_trigger": 0.05} | 3 | -0.43% | 1.66% | -0.60 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.02} | 3 | -0.43% | 1.66% | -0.60 |
| B3 | {"destination_fraction": 0.5, "weight_trigger": 0.05} | 3 | -0.43% | 1.66% | -0.60 |
| B4 | {"pause_cdar95": 0.01} | 0 | 3.18% | 0.32% | 1.53 |
| B4 | {"pause_cdar95": 0.02} | 0 | 3.18% | 0.32% | 1.53 |
| B4 | {"pause_cdar95": 0.03} | 0 | 3.18% | 0.32% | 1.53 |
| B4 | {"pause_cdar95": 0.01} | 1 | 3.72% | 0.31% | 2.81 |
| B4 | {"pause_cdar95": 0.02} | 1 | 3.72% | 0.31% | 2.81 |
| B4 | {"pause_cdar95": 0.03} | 1 | 3.72% | 0.31% | 2.81 |
| B4 | {"pause_cdar95": 0.01} | 2 | 3.18% | 0.32% | 1.53 |
| B4 | {"pause_cdar95": 0.02} | 2 | 3.18% | 0.32% | 1.53 |
| B4 | {"pause_cdar95": 0.03} | 2 | 3.18% | 0.32% | 1.53 |
| B4 | {"pause_cdar95": 0.01} | 3 | -0.43% | 1.66% | -0.60 |
| B4 | {"pause_cdar95": 0.02} | 3 | -0.43% | 1.66% | -0.60 |
| B4 | {"pause_cdar95": 0.03} | 3 | -0.43% | 1.66% | -0.60 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 0 | 3.18% | 0.32% | 1.53 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 0 | 3.18% | 0.32% | 1.53 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 0 | 3.18% | 0.32% | 1.53 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 0 | 3.18% | 0.32% | 1.53 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.02, "equity_index": 0.1, "gold": 0.06}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 0 | 3.16% | 0.32% | 1.50 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.03, "equity_index": 0.12, "gold": 0.08}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 0 | 3.16% | 0.32% | 1.50 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 1 | 3.75% | 0.31% | 2.95 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 1 | 3.75% | 0.31% | 2.95 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 1 | 3.75% | 0.31% | 2.95 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 1 | 3.75% | 0.31% | 2.95 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.02, "equity_index": 0.1, "gold": 0.06}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 1 | 3.74% | 0.31% | 2.99 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.03, "equity_index": 0.12, "gold": 0.08}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 1 | 3.75% | 0.31% | 2.99 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 2 | 3.18% | 0.32% | 1.53 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 2 | 3.18% | 0.32% | 1.53 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 2 | 3.18% | 0.32% | 1.53 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 2 | 3.18% | 0.32% | 1.53 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.02, "equity_index": 0.1, "gold": 0.06}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 2 | 3.16% | 0.32% | 1.50 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.03, "equity_index": 0.12, "gold": 0.08}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 2 | 3.16% | 0.32% | 1.50 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01} | 3 | -3.01% | 1.69% | -2.32 |
| B5 | {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.02} | 3 | -3.01% | 1.69% | -2.32 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01} | 3 | -1.93% | 1.76% | -1.28 |
| B5 | {"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.02} | 3 | -1.93% | 1.76% | -1.28 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.02, "equity_index": 0.1, "gold": 0.06}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 3 | -3.01% | 1.69% | -2.32 |
| B5 | {"take_profit_activation_by_category": {"bond": 0.03, "equity_index": 0.12, "gold": 0.08}, "take_profit_redeem_fraction": 0.25, "take_profit_trailing_drawdown_by_category": {"bond": 0.0075, "equity_index": 0.05, "gold": 0.03}} | 3 | -3.01% | 1.69% | -2.32 |

选参规则：验证期先检查年化收益>5%且MDD≤4%；同时达标时按Sharpe优先，再比较CAGR/MDD。若收益目标未达标但MDD可行，则优先提高CAGR；若MDD也不达标，优先选择MDD较小的诊断候选。选中参数冻结后仅在test复跑。

## 3. 费用敏感性

- 真实逐基金申购/赎回费率仍缺失，30/60bp仅为统一单边压力假设；
- v1.0.1 将首次反事实建仓费、后续申购费和赎回费统一计入模拟净值、Sharpe、XIRR和换手；
- 不得把统一费率压力情景等同于真实账户净收益。

---
状态: RESEARCH_ONLY
