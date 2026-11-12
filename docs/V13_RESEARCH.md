# v1.3 研究交接：近一年预测、稳健权重优化与债基抗跌筛选

> 状态：**RESEARCH_ONLY / NOT TRADE READY**。CI仅验证算法与模拟器兼容性。
> 当前公开仓库不包含个人 data/nav 缓存和真实账户批次，因此尚无v1.3真实
> 基金净值回测结果，也未证明净年化>5%、组合MDD<=4%。
> 相关源码和报告只适用于研究，不会修改账户目标仓位或自动交易。

## 与v1.2真正不同的地方

- \`fundkit/v13_model.py\`：收益预测只用截至决策日已公开的最近**252个共同交易日**。不再使用固定504日收缩强度；采用**先训练、后评价**的21交易日历史滚动验证，从0、0.25、0.5、0.75、1.0选择样本均值收缩权重。训练历史不能被它自己的未来信息污染。
- 126/63日均值作为**状态诊断**，其分歧增加预测误差协方差；不直接将短期业绩年化外推成alpha。协方差矩阵使用估计的对角收缩比例；误差协方差来自最近一年联合区块重采样，并进行PSD校验。
- \`fundkit/v13_portfolio.py\`：**CVXPY/CLARABEL稳健凸SOCP**替代v1.2线性收益目标，直接惩罚组合协方差和收益预测不确定性，用历史线性回撤切平面加有限情景有符号缺口CVaR管理风险。
- \`fundkit/weight_optimizer.py\`：风险预算2%、3%、4%，配合不同预先声明的风险/不确定性惩罚强度，获得**风险—收益差异化**研究候选，而不是改变完全不生效的最低风险投资比例。研究候选会严格去重、逐一复验恒定权重组合风险。
- **格林021559反事实**：\`grinn_021559_sensitivity\`比较格林恰好为0以及至少10%/20%/40%时，其他权重重新求解的模型评分、历史/情景回撤及情景收益。报告不可行原因，不会为了讨好预设格林必须持有。新增冻结历史测试基准：100%格林、格林40%＋其他债基、等权债基。
- \`fundkit/v13_execution.py\` + \`fundkit/v11_policy.py\`：新增**A7风险预算金额策略**，预先由真实缓存现金账本限定上限，再比较8个非零申购额度方案（实际限额与目标缺口决定每只基金金额），逐个接受或拒绝当时的情景风险门。数据不全返回\`RISK_BLOCKED\`原因而非默默0交易。是受约束**有限网格**、非全局MPC最优；旧A1—A6仍保持原样以供对照。
- \`fundkit/v11_signals.py\`：情景风险门只要求**当前持有和拟买入**基金的共同净值历史，不再因为账户没有持有的其他基金历史短而误判全局数据不足。
- \`scripts/bond_resilience.py\`：本地所有债基数据中，强制采用同一个10月29日震荡前基准净值、同一截止日期，分红拆分调整后按冲击后总收益率降序、回撤升序列出抗跌榜，并列出最近一年表现及缺失数据排除数量。很短的冲击期**绝不外推为年化收益预测**。

## 组合求解模型

在最近一年公开训练数据上估计预期收益 \`mu\`、收益协方差 \`Sigma\` 和
预测误差协方差 \`Omega\`。对每条历史情景，固定目标权重的初期买入持有
组合财富 \`V[t](w)=sum_i w_i*G[i,t]\` 为权重的仿射函数。求解：

\`\`\`
maximize    mu @ w - rho*sqrt(w.T @ Omega @ w)
             - 0.5*lambda*w.T @ Sigma @ w - TC(w,w_current)
subject to  sum(w)==1, lower_i<=w_i<=upper_i,
            cash>=cash_floor,
            V[t](w)>=(1-d)*V[s](w) for every historical 0<=s<=t,
            CVaR_95( max_{s<=t}[(1-d)V[s](w)-V[t](w)] )<=0
\`\`\`

只对固定买入持有的有限情景数学模型给出锥优化求解结果。
**不等于每日再平衡或含现金流真实组合的全局最优，也不保证未来4%回撤硬上限。**
协方差收缩、预测误差估计和rho/lambda离散候选都只能靠训练/验证期判断，
不能用测试期挑选最漂亮的结果。少于指定数目的不同可行组合，
严格按实际数返回，不作随机填充。

2022债券冲击等更旧历史**不参与当前收益均值估计**；如需用其作为
历史风险压力数据，在 \`optimize-weights\` 使用 \`--lookback 756\`
（尽管预测仍仅使用末252个交易日）。未来仍可接入有证据的因子情景
及债券真实Carry、久期和信用利差。

## 运行命令（在项目根目录）

\`\`\`powershell
git fetch origin
git switch --track origin/codex/fund-analysis-v1.3-one-year-robust-20261112
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -q

# 先更新本地全市场债基净值和申赎状态快照，需网络
python main.py list
python main.py fetch --types 债券型 --workers 8

# 2026年10月30日债市冲击：前日净值作统一基准
python -m scripts.bond_resilience --baseline 2026-10-29 --end 2026-11-11 --min-days 6 --top 50
# 生成 reports/bond_resilience_20261030.csv
# 及 reports/bond_resilience_20261030.audit.json

# 最近一年最新已公开数据求解目标权重
python main.py optimize-weights --asof 2026-11-12 --lookback 252 --paths 96
# 输出 reports/weight_optimization_research.json
# 检查 predictive_model,frontier_grid,grinn_021559_sensitivity

# 如需更长历史*风险压力*但不影响最近一年的alpha估计
python main.py optimize-weights --asof 2026-11-12 --lookback 756 --paths 96

# B2,A1-A6,A7用相同窗口/同一持仓权重进行验证+冻结测试
python scripts/run_v11_exploratory.py --asof 2026-11-11 --mode halfyear
python scripts/run_v11_exploratory.py --asof 2026-11-11 --mode quarterly
# 每折 raw JSON 增加 fixed_benchmarks、A7和grinn敏感性
\`\`\`

注意：两种回测窗口协议和报告脚本名称仍保留旧\`v11\`标识，
但是其组合优化入口已经调用v1.3研究方法，不得混淆代码版本。
用于训练的历史样本严格截止于验证开始之前，不得把2026年11月的
最新预测模型参数倒灌到2025年训练窗口。

## 验收与禁止事项

- 单基金40%上限只是治理假设；100%格林为**研究基准、不是实盘配置**。
- 若情景不允许加仓，A7可能仍然只有初始建仓交易，必须输出
  风控缺数据、真实超预算、资金不足等阻断原因及逐日事件。
- A7只完成单日**目标仓位缺口约束下的8档申购额度择优**；
  不等于连续金额多期MPC、不代表优化止盈止损参数。止盈和止损仍为B6原规则。
- 新权重研究只在最近一年统计收益上拟合，未使用最新可靠的基金债券
  持仓、信用利差、久期与逐笔申赎费用；真实资产风险模型尚需继续补齐。
- 同日期债基抗跌排名在极短窗口内只能作为候选发现，不能用短期排名
  推断下一年收益，也不能因为净值延迟而让落后基金“抗跌”。
- 必须用本地真实净值对v1.3跑半年/季度以上探索性测试，再比较
  CAGR、MDD、Sharpe、TWR/XIRR、申赎次数与费用。超过4%回撤或年化
  不大于5%，必须明确返回未达标；不得默认100%现金掩盖失败。
