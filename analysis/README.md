# 全账户基金组合风险分析与样本外验证（2026-11-10）

本目录是 `fund_analysis` v0.9 算法层一次完整分析的**脱敏导出**，供外部评审（GPT）使用。
所有绝对持仓金额已替换为组合权重百分比；指标、日期、判定、方法论均为原始计算结果，未做任何修饰。

- 代码版本：`a39d715`（main 分支）
- 数据截止：2026-11-06（净值）/ 2026-11-09（因子）
- 投资目标（固定，未放宽）：长期年化净收益 **> 5%** 且组合最大回撤 **≤ 4%**，风险优先于收益
- 账户：13 只场外基金（4 债券型 + 1 黄金 + 8 股票指数其中 7 只 QDII），每日新增定投预算（流量）
- 账户口径：仅有持仓快照金额，无真实逐笔流水（ACCOUNT_DRAWDOWN_UNVERIFIABLE）；
  回测起点为"快照权重×标准化本金"的反事实组合，非真实历史

## 核心结论

1. **NO_FEASIBLE（实质性）**：4 折滚动样本外验证（152 次试验，策略族 B0-B5，
   train→val 选参→test 冻结复跑），没有任何策略族在全部 test 折同时满足
   CAGR>5% 且 MDD≤4%。最佳为 B5（test 均值 3.57% / 0.42%）。
   风险端极其宽裕（块自助 P(MDD>4%)≈0%），**缺口全在收益端**：
   含 2024-2025 行情的长窗可接近 5%，纯 2026 年窗口全部跌至 2% 上下。
2. **联合优化 DATA_BLOCKED**：全账户共同历史仅 507 个共同交易日
   （最晚成立的基金 2024-08 才有净值），联合优化硬性需要 ≥1008 天
   （train/val/test 各 ≥252），未放宽阈值。
3. **权重优化候选不可实盘**：情景内达标（Q25 收益 6.09%、MDD P95 1.07%）
   但稳定地把单只基金顶到持仓上限（边界解），且未经独立样本外验证。

## 文件导读

| 文件 | 内容 |
|---|---|
| DATA_COVERAGE.md | 13 只基金数据清单：净值区间/缺失率/限额/披露/因子覆盖 |
| DATA_QUALITY_AUDIT.md | 七项质量检查（分红一致性、复权重复计入、异常收益归因、风格敞口、批次可重建性、QDII 滞后） |
| PORTFOLIO_RISK_REPORT.md | 当前权重组合风险：多窗口指标、相关性矩阵、滚动风险、历史情景回放、债券敞口与隐含久期 |
| OOS_BACKTEST_REPORT.md | 滚动样本外验证全结果：切分、冻结参数、统一指标、可行性判定、有效前沿、压力测试、不可行原因分解 |
| OPTIMIZATION_RESULT.json | 联合优化阻断原因 + 优化变量审计 + 权重候选 |
| PARAMETER_SENSITIVITY.md | 权重对约束/种子/回看窗的敏感性 + 规则网格 val 敏感性 + 费用敏感性 |
| oos_trials.csv | 全部 152 次试验原始记录（含换手率/费用/现金占比） |
| oos_summary.json / data_coverage.json / portfolio_risk.json / weight_optimization_research.json | 上述报告的机器可读源数据 |
| validate.log | 验证运行日志 |

## 数据与方法要点

- 净值：danjuan 主源本地缓存，分红/拆分事件后复权（逐日对账一致）；
  published_at 为 T+2 规则推断（C 级证据）
- 组合 MDD：买入持有单位净值（无外部现金流），另有 TWR 单位化实现于模拟器
- 债券风险：季报披露 lower_bound（前五债券分类）+ 日收益对 5Y 国债收益率回归（R²≈0，仅量级参考）
- 缺失项（不伪造）：申赎费率、信用利差、汇率因子、QDII 历史限购变化
- 已知局限：单源净值、反事实起点、费用以压力情景近似、021559 真实批次不可重建

## 复现

```bash
python3 -m unittest discover -s tests -p 'test_*.py'   # 65 条
python3 main.py validate                                # OOS 验证（确定性，块自助种子42）
python3 scripts/data_audit.py && python3 scripts/portfolio_risk.py
python3 scripts/oos_report.py && python3 scripts/optimization_report.py
python3 scripts/export_analysis.py                      # 本目录的脱敏导出
```
