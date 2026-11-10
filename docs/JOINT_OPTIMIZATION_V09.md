# v0.9：目标权重与定投/止盈/止损规则联合优化

**正式目标：长期年化净收益严格高于 5%，组合最大回撤风险预算不超过 4%。**
风险预算优先；无可行候选时返回不可行，不放宽 5%/4%。所有输出均为 `RESEARCH_ONLY`。

## 1. v0.9解决的缺口

v0.8 已经可以从训练收益情景中生成目标权重候选，但只输出单一权重结果，退出规则仍沿用独立阈值，二者没有联合比较。v0.9 新增 `fundkit/joint_optimizer.py`，把以下对象放进同一个外层搜索：

- 目标权重向量 `w`（由训练期收益数据生成候选）；
- B3 阈值再平衡参数；
- B4 风格风险暂停/恢复参数；
- B5 组合回撤、单基金回撤、CDaR、赎回比例、恢复阈值与冷却期。

训练期只生成权重候选；验证期比较 `weight × rule × fee scenario`；测试期只运行验证集选出的一个冻结候选。测试集不得参与选参。

## 2. 决策变量

### 2.1 资产配置变量

\[
\boldsymbol w=(w_1,\ldots,w_n,w_c),\quad
w_i\geq0,\quad w_c\geq0,\quad
\sum_iw_i+w_c=1.
\]

当前研究设定将股票指数基金的新增目标权重设为 0（用户已表示不再持有股票）。这是一个“退出股票后的目标组合”反事实，不代表系统已经替用户卖出存量股票；实际账户转换仍需真实批次、赎回和费用数据。

### 2.2 动态交易变量

\[
a_{i,t}\geq0
\]
表示第 t 个执行日向基金 i 的申购金额；

\[
v_{i,\ell,t}\geq0
\]
表示赎回基金 i 的第 \ell 笔可赎回份额。B2 水位法分配现金；B3/B5 在研究模拟中可发出真实赎回事件，但不自动交易。

### 2.3 外层策略参数

\[
\theta=\left(
\delta^{reb},\rho^{reb},
\delta^{style}_{pause},\delta^{style}_{reenter},\tau^{style}_{cool},
\delta^{port}_{soft},\delta^{fund}_{pause,i},
\delta^{fund}_{reenter,i},\delta^{CDaR},
\rho^{redeem},\tau^{cool}
\right).
\]

这里的阈值均为候选超参数，不能因为出现在代码网格中就解释为行业公认阈值。

## 3. 目标函数与约束

对给定目标权重和规则，在验证期与费用压力情景集合 \(\mathcal C\) 下计算：

\[
G_{val}(w,\theta)=\min_{c\in\mathcal C}\operatorname{CAGR}_{val}^{net}(w,\theta,c),
\]

\[
D_{val}(w,\theta)=\max_{c\in\mathcal C}\operatorname{MDD}_{val}(w,\theta,c).
\]

验证可行集：

\[
\mathcal F_{val}=\left\{(w,\theta):
G_{val}(w,\theta)>0.05,\;
D_{val}(w,\theta)\leq0.04
\right\}.
\]

若可行集非空，选择：

\[
(w^*,\theta^*)=
\arg\max_{(w,\theta)\in\mathcal F_{val}}
\left(G_{val},-D_{val},\operatorname{Sharpe}_{val}^{conservative},-Turnover\right)
\]

按字典序比较：先最大化最差费用情景下的净 CAGR，再降低回撤，再考虑夏普与换手。

若 \(\mathcal F_{val}=\emptyset\)，只输出验证期回撤最小的诊断候选，状态为 `NO_VALIDATION_FEASIBLE_CANDIDATE`，不生成推荐。

测试期只评估冻结的 \((w^*,\theta^*)\)：

\[
G_{test}>0.05,\quad D_{test}\leq0.04,
\]

且测试窗口至少252个有效披露日，才可能得到 `TEST_WINDOW_CRITERIA_MET_NOT_GENERALIZED`。即使达到，也只是单个测试窗口通过，不是未来保证；至少需要多个不重叠滚动测试折和不同市场状态重复验证。

## 4. 权重候选生成

`generate_weight_candidates()` 在训练期收益矩阵上生成：

1. 现金基准；
2. 单资产上限候选；
3. 等比例可交易资产候选；
4. 多个 Dirichlet 浓度下的随机可行权重；
5. 按训练期历史 MDD、联合移动块 Bootstrap MDD P95、收益 Q25 筛选的风险—收益多样化候选。

**重要修正：** 静态权重 MDD 超过4%的高收益候选不会一律提前剔除，因为 B3/B5 的退出规则可能改变实际现金流路径；候选池会保留最低风险端点、最高收益端点及静态可行点，再交由外层真实模拟器评价。

这属于有限候选搜索，不保证全局最优。候选池大小、Dirichlet 浓度、块长度和随机种子均需记录。

## 5. 退出与恢复规则网格

`build_rule_grid(asset_categories)` 预先生成小型候选集：

- B2：仅现金流再平衡基线；
- B3：权重偏离阈值 1%/2%，偏离部分处理比例25%/50%；
- B4：风格 CDaR 暂停阈值1%/2%，恢复阈值为暂停阈值的一半，冷却期5/10个执行日；
- B5：组合内部回撤缓冲0.5%/1%/1.5%；债券基金止损候选1%/1.5%/2%；黄金基金止损候选2%/3%/4%；CDaR阈值1.5%/2.5%；赎回比例0/25%/50%；恢复阈值为各自暂停阈值的一半；恢复比例与冷却期在预先声明的配对设计点中变化。

债券和黄金采用不同阈值搜索空间是风险建模上的候选分层，不代表这些数值已经被实证证明最优。真实止损阈值必须由样本外结果决定。

B4 风格信号现在支持按日期更新：联合优化 CLI 每21个工作日调用一次 as-of 风格分析，只允许读取该日已公开的季报、净值和因子。若验证期或测试期没有可计算的 `CURRENT_STYLE_REWEIGHTED.cdar95`，该折会从规则网格中剔除B4，避免把无信号的空操作误当作有效风格风控。实际 `published_at` 仍有本地 T+2 规则推断的局限。

B5 单基金回撤信号改用分红调整后的总回报净值 `total_return_nav`，而成交和资产估值仍使用原始单位净值 `unit_nav`，避免除息日误触发止损。B5 支持 `fund_pause_drawdown_by_code`、`fund_reenter_drawdown_by_code` 和 `fund_redeem_fraction_by_code`，因此不同基金可采用不同阈值。

## 6. 费用敏感性

CLI 默认以0bp和30bp两种统一费率压力情景重跑验证/测试，候选按最差费用情景选取。这只是统一费率敏感性，不是基金真实费率表；没有真实费率时不能把结果称为真实净收益。真实版本应替换为基金代码×持有期×申赎方向的实际费率函数。

## 7. 数据接口与执行命令

本模块不采集数据，直接复用本地 `LocalDataAdapter`/AKShare缓存。

```powershell
# 默认：训练504日、验证252日、测试252日、1个滚动折
python main.py optimize-strategy --asof 2026-11-10

# 两个不重叠测试折（至少约1260个共同交易日）
python main.py optimize-strategy --asof 2026-11-10 --folds 2 --step-days 252

# 若有经季报证据确认的统一风格起点，可限制训练收益窗口
python main.py optimize-strategy --asof 2026-11-10 --regime-start 2025-06-30

# 调整搜索预算/费用敏感性
python main.py optimize-strategy --weight-pool 1200 --weight-candidates 8 --weight-paths 128 --fee-stress-bps 0,10,30
```

默认要求训练504日、验证252日、测试252日；CLI拒绝任何少于252日的训练/验证/测试窗口。若当前风格阶段不足以提供训练长度，程序应返回 `DATA_BLOCKED` 或 `INSUFFICIENT_HISTORY`，不得跨越风格断点静默补足。当前上传包没有 `data/nav/`，所以本次只能运行数学/逻辑测试，不能产生真实基金权重。

## 8. 反事实组合与交易限制

- 为避免将2026年账户持仓倒灌进2024年训练期，训练权重搜索的初始权重代理设为全现金；股票指数基金目标权重为0。
- 验证/测试模拟从“候选目标权重组合”起步，初始资金按当前快照总资产标准化。这是策略路径对照，不是用户真实账户历史，也不计入从当前账户切换至目标组合的初始交易成本。
- 021559 初始模拟批次的申购申请日未知，按不可赎回处理；回测期内新申购批次才按模拟器的滚动持有规则管理。
- 当前申购限额快照无历史生效时间，不能当作历史准确限额；缺失时会标记质量问题。
- 费用敏感性仅为统一费率假设；真实赎回费按持有期差异化处理尚未完成。

## 9. 科学与行业依据及边界

1. Lyu et al. (2024), *Robust Online Portfolio Optimization with Cash Flows*, Omega：持续现金流与在线组合配置的研究基础。
2. Boyd et al. (2017), *Multi-Period Trading via Convex Optimization*：多期权重、交易成本与约束框架；本实现是有限候选外层搜索，不是该论文完整求解器。
3. Chekhlov, Uryasev & Zabarankin (2005), *Drawdown Measure in Portfolio Optimization*：CDaR 路径风险度量；不证明4%阈值最优。
4. Kaminski & Lo (2014), *When Do Stop-Loss Rules Stop Losses?*：止损的有效性依赖收益动态，不支持机械照搬固定阈值。
5. Vanguard (2024), *The Rebalancing Edge*：阈值再平衡的行业证据；其参数不应直接移植到本账户。

## 10. 尚未完成的工作

- 债券信用利差、银行二级资本债/永续债专项因子仍缺少可靠日频数据；B4不能完整识别信用风险突变。
- 统一费率压力需要替换为真实基金费率表及持有期规则。
- 当前账户逐笔申购/赎回确认流水缺失，真实账户 MDD、XIRR 和滚动基金可赎份额不可验证。
- 当前持仓转成目标组合的初始交易成本与股票存量退出没有纳入历史策略回测。
- 单个测试折达标不等于长期有效；需要至少多个不重叠、每折不少于252日的测试窗口。
- 本搜索为预声明有限网格，不保证全局最优；搜索预算增大时需报告所有试验，防止多重检验偏差。

**结论：** v0.9 实现了“目标权重候选 × B2/B3/B4/B5退出规则 × 费用压力情景”的外层联合搜索接口，并用训练—验证—测试隔离防止测试集选参。它可以在现有 AKShare 数据缓存完整后运行；当前无净值数据时必须 `DATA_BLOCKED`，不得输出基金实际最优金额。

### 历史输出归档

v0.8 的 `reports/fund_decisions.csv`、`reports/weight_optimization_research.json` 和 `decisions/2026-11-10.json` 已移动到 `LEGACY_UNVERIFIED_V08/` 子目录，避免旧版权重/决策被误认为v0.9结果。它们不参与新版本决策。