# 完整研究审计

asof: 2026-11-09T22:25:30.248614+08:00

## 1. 数据来源与口径

- 净值：本地缓存（danjuan 主源，B/SINGLE_SOURCE），published_at 规则推断 T+2 自然日
- 复权：分红/拆分事件重建（dividends.csv / splits.json），260 只远古对账不符已排除
- 披露：东财 FundMNAssetAllocation + F10 zqcc，published_at 按法定披露时限规则推断，重仓为 lower_bound
- 因子：中债国债收益率曲线（1/3/5/10/30Y）、沪深300 价格指数（非全收益）、上海金；信用利差/汇率 MISSING 不伪造
- 账户：模拟批次（SIMULATED），ACCOUNT_DRAWDOWN_UNVERIFIABLE: 无逐笔确认流水，批次为快照市值反推的模拟批次；真实历史 MDD/TWR/可赎回份额不可验证

## 2. 方法与公式

- TWR 单位净值：外部现金流按当时单位净值折算份额（external_cash_unitization）
- 风险指标：MDD / CDaR95（Chekhlov 离散经验 CVaR，tests/finance_math.py 审计实现）
- 风格识别：EWMA 加权岭回归（lookback/halflife/ridge/smooth 为候选超参）+ 披露断点双证据合并
- 策略族 B0-B5：03_POLICY_ALGORITHMS.md；内层现金流分配为赤字注水近似（非全局最优 QP）
- 反事实压力：CURRENT_STYLE_REWEIGHTED = 当前 beta × 历史因子冲击 + 近期残差自助，非真实业绩

## 3. 样本外验证

- 折: [{"train": ["2023-06-01", "2025-06-01"], "val": ["2025-06-01", "2025-12-01"], "test": ["2025-12-01", "2026-06-01"]}, {"train": ["2024-01-01", "2026-01-01"], "val": ["2026-01-01", "2026-06-01"], "test": ["2026-06-01", "2026-11-09"]}]
- 网格大小: 16（全部试验见 reports/oos_trials.csv）
- 可行性判定: **NO_FEASIBLE_STRATEGY**

### 各策略 test 折结果

| 策略 | 全部折满足 CAGR≥4% 且 MDD≤5% | 明细 |
|---|---|---|
| B0 | 否 | fold0 CAGR 4.88% MDD 1.26%; fold1 CAGR 0.02% MDD 0.90% |
| B1 | 否 | fold0 CAGR 3.65% MDD 0.90%; fold1 CAGR 1.64% MDD 0.44% |
| B2 | 否 | fold0 CAGR 4.28% MDD 1.65%; fold1 CAGR 0.93% MDD 0.94% |
| B3 | 否 | fold0 CAGR 2.61% MDD 1.50%; fold1 CAGR 1.37% MDD 0.85% |
| B4 | 否 | fold0 CAGR 2.61% MDD 1.50%; fold1 CAGR 1.37% MDD 0.85% |
| B5 | 否 | fold0 CAGR 2.50% MDD 1.17%; fold1 CAGR 1.79% MDD 0.20% |

### 帕累托前沿（test 均值）

- B1: CAGR 2.64% / MDD 0.67%

### 冻结参数（每 fold 所选）

- B1 fold0: {}
- B1 fold1: {}
- B4 fold0: {"pause_cdar95": 0.01}
- B4 fold1: {"pause_cdar95": 0.01}
- B2 fold0: {"cash": 0.0}
- B2 fold1: {"cash": 0.0}
- B5 fold0: {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01}
- B5 fold1: {"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01}
- B0 fold0: {}
- B0 fold1: {}
- B3 fold0: {"destination_fraction": 0.25, "weight_trigger": 0.02}
- B3 fold1: {"destination_fraction": 0.25, "weight_trigger": 0.02}

## 4. 失败与局限（如实记录）

- 利率因子回归 beta≈0：日频债基收益以 carry 为主，R² 低；风格状态多依赖披露证据，STYLE_UNCERTAIN 从严
- B2 候选现金下限 10% 导致短期全留现金（候选参数语义，第5步网格含 0/5%/10% 对比）
- 反事实起点组合非用户真实历史；真实账户 MDD 不可验证
- QDII 成交用最近披露净值近似；费率默认 0（未知，费用压力测试见 oos_trials）
- 沪深300 为价格指数，权益 beta 略低估；信用利差缺失使信用冲击以披露 lower_bound 表达

状态: RESEARCH_ONLY
