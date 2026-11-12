# v1.2: 连续组合优化（研究实现；禁止实盘自动执行）

## 本次真正替换的部分

\`fundkit/weight_optimizer.py\` 的两个入口
\`optimize_target_weights()\`、\`generate_weight_candidates()\`
不再生成Dirichlet随机权重，而是调用
\`fundkit/convex_portfolio.py\` 中的
\`solve_continuous_weights()\`（SciPy HiGHS LP）。

**仅第一层战略目标权重优化已更换。** v1.1 的B6每日投入/止盈止损策略
仍为原实现，其参数未在本次连续优化中校准。没有真实NAV全量回测，
不声称5%收益或4%未来回撤目标已实现。

## 数学模型及严格适用边界

13只基金+现金，非负权重w、总和1；基金上下限、不可交易/冻结持仓在LP中硬约束。

对当前训练日之前已披露的总回报收益矩阵，使用现有
\`estimate_robust_returns\` 计算各基金 \`robust_lcb_annual\`；
这是一个估计模型，使用504日类别先验收缩系数及z=1不确定性惩罚，
**尚未独立校准**。

最大化以下**线性**的保守收益与换仓成本目标：

\`\`\`
annual_lower_returns @ risky_weights
+ annual_cash_return * cash_weight
- turnover_cost_rate * sum(abs(weight_i - current_weight_i))
\`\`\`

如需要风险收益有效前沿，在同一问题上改变最小风险敞口比例
（0、10%、20%、30%等，**只是生成对照候选的外生约束，并非最优投资比例**）。
严格去重后最多返回\`top_k\`个非现金风险可行候选。实际没有5个可行候选时
按真实数量返回，**不通过随机点或高风险点填充**。

对于某条 **期初配置之后买入持有、不再交易** 的固定历史情景，
每只基金的复利财富倍数为G[i,t]，组合财富

\`\`\`
V_t(w) = sum_i w_i * G[i,t] + w_cash * G_cash[t]
\`\`\`

按定义 \`MDD<=d\` 当且仅当每一对 \`0<=s<=t\` 有

\`\`\`
V_t(w) >= (1-d) * V_s(w)
\`\`\`

对w是**线性不等式**。算法使用最大违反约束分离，在历史训练路径上
强制约束MDD<=4%。独立情景采用联合20/60日区块Bootstrap，
通过LP中对有符号违约财富缺口的经验CVaR95<=0实现保守风险预算：

\`\`\`
Z_s(w) = max_{0<=u<=t} ((1-d)*V_{s,u}(w) - V_{s,t}(w))
min_eta eta + sum_s max(0, Z_s(w)-eta) / ((1-alpha)*n_scenarios) <= 0
\`\`\`

该约束是 \`P(MDD<=4%)>=95%\` 的**有限经验情景充分条件**，
比直接要求第95百分位MDD<=4%可能更保守。
所有违反约束通过切平面迭代加入。求解状态必须为\`OPTIMAL\`，
还要复验原始历史MDD、实际情景有符号CVaR和线性约束残差。

该数学凸性只适用于**固定初始权重下的买入持有**。
对于每日固定比例再平衡，旧\`_risk_metrics()\`会另外复核历史和情景MDD，
两类估计中最坏者超过4%，不得标记为风险可行。
对真实基金T日未知净值申赎、每日新增现金、在途/锁定份额、动态止盈止损，
仍必须另由\`fundkit/simulator.py\`逐日模拟，LP解不构成真实交易路径的全局最优。

## 运行及验证

\`\`\`powershell
git fetch origin
git switch --track origin/codex/fund-analysis-v1.2-convex-portfolio-20261112
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -q

# 只用本地已公开原始净值，不能编造历史
python scripts/run_v11_exploratory.py --asof 2026-11-11 --mode halfyear
python scripts/run_v11_exploratory.py --asof 2026-11-11 --mode quarterly
\`\`\`

两个回测仍是\`EXPLORATORY\`；它们可能用时较长。
默认最多比较5组独立连续LP风险可行投资权重，
以相同验证/测试分割对B2/B6做有限对照。
模型训练、验证、测试之间仍采用3交易日信息隔离。

结果：
- \`reports/joint_strategy_optimization.json\`：求解器/权重/验证/测试原始字段
- \`reports/V11_EXPLORATORY_REPORT.md\`：策略统计对照
- \`reports/v11_exploratory_summary.json\`：小型摘要

如果没有任何非现金风险可行解，报告
\`NO_NONCASH_RISK_FEASIBLE_CANDIDATE\`，**不再用全现金覆盖投资消融**。
全现金仍保留为独立的现金收益/风险诊断基准。
这也**不是**强制配置风险资产；正式权重优化若现金费后目标更优，
可以合法选择现金，但不把它当作止盈止损效果的证据。

## 未完成与边界

- 尚未运行依赖个人本地\`data/nav/\`的真实基金多月历史回测；
- 尚未将B6每日投入/减仓量改造成连续决策优化，也没有重新校准止盈止损阈值；
- 100%格林、等权债基、当前真实权重等外部基准尚未独立接入本轮运行入口；
- 年化收益置信不确定性与债券Carry/信用利差模型尚未样本外校准；
- 费率仍使用30/60bp统一敏感性假设，不能视为真实逐笔费率；
- 限定范围与滚动持有期可能阻止实盘达到目标权重；
- 即使有限情景LP证明数学最优，依然**不能保证未来最大回撤≤4%**；
- \`candidate_count\`旧参数兼容保留，但随机生成权重的语义已废弃；
  正式模型中必须审计并逐步清理该旧字段。

**建议按上面命令本地回测后提交原始JSON，而不是重新截图两日模拟。**
