# OOS_BACKTEST_REPORT — 滚动样本外验证

生成时间: 2026-11-11T04:53:21.357875｜算法版本: NON_OVERLAPPING_PRIMARY_OOS_GUARD_V1｜试验总数: 168（网格 18 配置 × 折 × 阶段）

口径：反事实起点组合（快照权重×标准化本金，非真实历史）；首次建仓申购费、后续申购费和赎回费均按统一压力假设计入，非真实基金费率；MDD/Sharpe 在含首次建仓成本的外部现金流单位化（TWR）净值上计算。

## 1. 时间序列切分（train→val 选参→test 冻结复跑）

| fold | train | val | test | test 长度 |
|---|---|---|---|---|
| 0 | 2023-06-01~2025-06-01 | 2025-06-01~2025-12-01 | 2025-12-01~2026-06-01 | 131 交易日 |
| 1 | 2024-01-01~2026-01-01 | 2026-01-01~2026-06-01 | 2026-06-01~2026-11-09 | 116 交易日 |
| 2 | 2023-06-01~2025-06-01 | 2025-06-01~2025-12-01 | 2025-12-01~2026-11-09 | 246 交易日 |
| 3 | 2022-06-01~2024-06-01 | 2024-06-01~2024-11-01 | 2024-11-01~2026-11-09 | 527 交易日 |

fold3 的 24 个月测试窗跨 710301 风格切换（2025-08-29 确认纯债化），其 test 段含切换前权益/转债敞口，是该基金真实历史行为而非数据错误。

## 2. 各策略族 test 折结果（冻结参数，全部试验见 oos_trials.csv）

| 策略族 | fold | 测试窗口角色 | CAGR | XIRR | MDD | CDaR95 | Sharpe | Sortino | Calmar | 年化换手 | 费用(元) | 平均现金占比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0 | short_diagnostic | 3.92% | 4.17% | 0.93% | 0.72% | 1.34 | 1.85 | 4.21 | 245.28% | 944 | 0.89% |
| B0 | 1 | short_diagnostic | -0.39% | -0.42% | 0.71% | 0.57% | -2.04 | -2.43 | -0.56 | 280.55% | 911 | 0.00% |
| B0 | 2 | primary_12m_single_window | 1.63% | 1.23% | 1.09% | 0.85% | -0.18 | -0.23 | 1.49 | 144.92% | 1266 | 0.92% |
| B0 | 3 | overlapping_24m_diagnostic | 5.43% | 4.21% | 2.69% | 1.71% | 1.24 | 1.75 | 2.02 | 73.03% | 2059 | 0.55% |
| B1 | 0 | short_diagnostic | 3.00% | 3.07% | 0.65% | 0.47% | 0.90 | 1.19 | 4.61 | 226.41% | 871 | 5.36% |
| B1 | 1 | short_diagnostic | 0.93% | 1.06% | 0.42% | 0.38% | -1.19 | -1.40 | 2.23 | 259.88% | 845 | 4.20% |
| B1 | 2 | primary_12m_single_window | 2.23% | 2.14% | 0.65% | 0.46% | 0.24 | 0.32 | 3.42 | 129.18% | 1128 | 7.72% |
| B1 | 3 | overlapping_24m_diagnostic | 5.00% | 4.08% | 1.96% | 1.37% | 1.39 | 2.00 | 2.55 | 62.97% | 1759 | 10.28% |
| B2 | 0 | short_diagnostic | 3.32% | 3.50% | 0.40% | 0.39% | 2.32 | 2.88 | 8.20 | 248.13% | 954 | 0.12% |
| B2 | 1 | short_diagnostic | 0.88% | 1.04% | 0.39% | 0.35% | -1.70 | -1.82 | 2.26 | 280.42% | 911 | 0.00% |
| B2 | 2 | primary_12m_single_window | 2.55% | 2.49% | 0.40% | 0.37% | 1.12 | 1.37 | 6.32 | 146.08% | 1277 | 0.06% |
| B2 | 3 | overlapping_24m_diagnostic | 4.22% | 3.54% | 0.99% | 0.49% | 2.04 | 3.54 | 4.27 | 67.40% | 1865 | 9.83% |
| B3 | 0 | short_diagnostic | 3.32% | 3.50% | 0.40% | 0.39% | 2.32 | 2.88 | 8.20 | 248.13% | 954 | 0.12% |
| B3 | 1 | short_diagnostic | 0.88% | 1.04% | 0.39% | 0.35% | -1.70 | -1.82 | 2.26 | 280.42% | 911 | 0.00% |
| B3 | 2 | primary_12m_single_window | 2.55% | 2.49% | 0.40% | 0.37% | 1.12 | 1.37 | 6.32 | 146.08% | 1277 | 0.06% |
| B3 | 3 | overlapping_24m_diagnostic | 4.63% | 3.91% | 1.03% | 0.52% | 2.28 | 3.94 | 4.49 | 74.99% | 2083 | 0.07% |
| B4 | 0 | short_diagnostic | 3.31% | 3.49% | 0.40% | 0.39% | 2.32 | 2.88 | 8.20 | 248.05% | 954 | 0.13% |
| B4 | 1 | short_diagnostic | 0.88% | 1.04% | 0.39% | 0.35% | -1.70 | -1.82 | 2.26 | 280.42% | 911 | 0.00% |
| B4 | 2 | primary_12m_single_window | 2.55% | 2.49% | 0.40% | 0.37% | 1.12 | 1.37 | 6.32 | 146.00% | 1276 | 0.09% |
| B4 | 3 | overlapping_24m_diagnostic | 4.63% | 3.91% | 1.03% | 0.52% | 2.28 | 3.94 | 4.49 | 74.99% | 2083 | 0.07% |
| B5 | 0 | short_diagnostic | 3.31% | 3.50% | 0.40% | 0.39% | 2.38 | 2.93 | 8.20 | 248.44% | 958 | 0.32% |
| B5 | 1 | short_diagnostic | 0.93% | 1.11% | 0.39% | 0.35% | -1.66 | -1.76 | 2.40 | 281.98% | 923 | 0.25% |
| B5 | 2 | primary_12m_single_window | 2.63% | 2.58% | 0.40% | 0.37% | 1.32 | 1.61 | 6.51 | 146.75% | 1293 | 0.49% |
| B5 | 3 | overlapping_24m_diagnostic | 4.58% | 3.88% | 1.03% | 0.52% | 2.25 | 3.89 | 4.44 | 76.29% | 2146 | 0.38% |

## 3. 冻结参数记录（val 风险优先选择，test 不得参与调参）

- B2 fold0: `{"cash": 0.0}`（val MDD 0.32%）
- B2 fold1: `{"cash": 0.0}`（val MDD 0.31%）
- B2 fold2: `{"cash": 0.0}`（val MDD 0.32%）
- B2 fold3: `{"cash": 0.1}`（val MDD 1.50%）
- B3 fold0: `{"destination_fraction": 0.25, "weight_trigger": 0.02}`（val MDD 0.32%）
- B3 fold1: `{"destination_fraction": 0.25, "weight_trigger": 0.02}`（val MDD 0.31%）
- B3 fold2: `{"destination_fraction": 0.25, "weight_trigger": 0.02}`（val MDD 0.32%）
- B3 fold3: `{"destination_fraction": 0.25, "weight_trigger": 0.02}`（val MDD 1.66%）
- B4 fold0: `{"pause_cdar95": 0.01}`（val MDD 0.32%）
- B4 fold1: `{"pause_cdar95": 0.01}`（val MDD 0.31%）
- B4 fold2: `{"pause_cdar95": 0.01}`（val MDD 0.32%）
- B4 fold3: `{"pause_cdar95": 0.01}`（val MDD 1.66%）
- B0 fold0: `{}`（val MDD 0.39%）
- B0 fold1: `{}`（val MDD 0.66%）
- B0 fold2: `{}`（val MDD 0.39%）
- B0 fold3: `{}`（val MDD 1.41%）
- B5 fold0: `{"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01}`（val MDD 0.32%）
- B5 fold1: `{"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01}`（val MDD 0.31%）
- B5 fold2: `{"fund_pause_drawdown": 0.02, "internal_drawdown_margin": 0.01}`（val MDD 0.32%）
- B5 fold3: `{"fund_pause_drawdown": 0.03, "internal_drawdown_margin": 0.01}`（val MDD 1.76%）
- B1 fold0: `{}`（val MDD 0.35%）
- B1 fold1: `{}`（val MDD 0.46%）
- B1 fold2: `{}`（val MDD 0.35%）
- B1 fold3: `{}`（val MDD 1.24%）

## 4. 可行性判定（主验收：独立完整年度OOS；单一主窗口仅作初步证据）

**判定: INSUFFICIENT_INDEPENDENT_OOS_WINDOWS**
独立完整年度主测试窗口数: 1；重叠的24个月窗口只作诊断，不参与主验收。

| 策略族 | 单一主窗口是否满足5%/4% | 主窗口/诊断窗口 CAGR、MDD、Sharpe |
|---|---|---|
| B0 | 否/证据不足 | fold2[test]: 1.63%/1.09%/Sh=-0.18; fold0[short_diagnostic]: 3.92%/0.93%/Sh=1.34; fold1[short_diagnostic]: -0.39%/0.71%/Sh=-2.04; fold3[overlapping_24m_diagnostic]: 5.43%/2.69%/Sh=1.24 |
| B1 | 否/证据不足 | fold2[test]: 2.23%/0.65%/Sh=0.24; fold0[short_diagnostic]: 3.00%/0.65%/Sh=0.90; fold1[short_diagnostic]: 0.93%/0.42%/Sh=-1.19; fold3[overlapping_24m_diagnostic]: 5.00%/1.96%/Sh=1.39 |
| B2 | 否/证据不足 | fold2[test]: 2.55%/0.40%/Sh=1.12; fold0[short_diagnostic]: 3.32%/0.40%/Sh=2.32; fold1[short_diagnostic]: 0.88%/0.39%/Sh=-1.70; fold3[overlapping_24m_diagnostic]: 4.22%/0.99%/Sh=2.04 |
| B3 | 否/证据不足 | fold2[test]: 2.55%/0.40%/Sh=1.12; fold0[short_diagnostic]: 3.32%/0.40%/Sh=2.32; fold1[short_diagnostic]: 0.88%/0.39%/Sh=-1.70; fold3[overlapping_24m_diagnostic]: 4.63%/1.03%/Sh=2.28 |
| B4 | 否/证据不足 | fold2[test]: 2.55%/0.40%/Sh=1.12; fold0[short_diagnostic]: 3.31%/0.40%/Sh=2.32; fold1[short_diagnostic]: 0.88%/0.39%/Sh=-1.70; fold3[overlapping_24m_diagnostic]: 4.63%/1.03%/Sh=2.28 |
| B5 | 否/证据不足 | fold2[test]: 2.63%/0.40%/Sh=1.32; fold0[short_diagnostic]: 3.31%/0.40%/Sh=2.38; fold1[short_diagnostic]: 0.93%/0.39%/Sh=-1.66; fold3[overlapping_24m_diagnostic]: 4.58%/1.03%/Sh=2.25 |

## 5. 主样本外有效前沿

| 策略族 | CAGR | MDD |
|---|---|---|
| - | NA | NA |

### 重叠窗口诊断前沿（不得视为独立OOS证据）

| 策略族 | CAGR | MDD |
|---|---|---|
| B2 | 2.80% | 0.59% |
| B4 | 2.94% | 0.61% |
| B3 | 2.94% | 0.61% |
| B1 | 2.98% | 1.01% |
| B0 | 2.98% | 1.44% |

## 6. 压力测试

- 块自助（2000 路径 × 252 日，块长 20，种子42，共同日期 507 天）: P(MDD>4%) = 0.0% [0.0%, 0.2%]，MDD 中位 1.02%，P95 1.38%
- 2022-11 债灾回放: DATA_BLOCKED_INCOMPLETE_2022_HISTORY
- 注：利率/信用利差/权益/黄金/汇率的**联合参数化**压力测试受限于因子缺失（信用利差、汇率 MISSING），当前以历史真实窗口回放 + 块自助替代，债券信用敞口以季报披露 lower_bound 表达（见 PORTFOLIO_RISK_REPORT 第6节）。

## 7. 不可行原因分解

- B0: 收益不达 5%（1/1 折，最低 1.63%）
- B1: 收益不达 5%（1/1 折，最低 2.23%）
- B2: 收益不达 5%（1/1 折，最低 2.55%）
- B3: 收益不达 5%（1/1 折，最低 2.55%）
- B4: 收益不达 5%（1/1 折，最低 2.55%）
- B5: 收益不达 5%（1/1 折，最低 2.63%）

结论：不放宽 5%/4% 目标；以上前沿即当前数据与策略族下的客观边界。

---
状态: RESEARCH_ONLY｜数据截止见 DATA_COVERAGE｜复现: `python main.py validate`（确定性；随机块自助种子42）
