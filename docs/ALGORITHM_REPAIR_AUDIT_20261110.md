# 基金动态定投算法 v0.7 修正审计（2026-11-10，历史归档）

> **本文件为v0.7历史审计记录，后续目标已更新为年化净收益严格超过5%、最大回撤预算不超过4%。本文件中的4%收益/5%回撤旧目标、42项测试统计均不代表v0.9。当前算法说明以 `docs/JOINT_OPTIMIZATION_V09.md` 为准。**

## 范围、基线与证据

- **基线**：用户提供 `fund_analysis_review_20261110.zip`，保留其 AKShare/蛋卷/东财数据接口、基金快照、采集器和策略配置。
- **目标**：修复数学、资金核算、交易模拟、因子归因、风险压力测试与研究建议闭环；**不拉取网络数据、不宣称实盘验证**。
- **可复现命令**：在项目根目录运行 `python -m unittest discover -s tests -p 'test_*.py' -v`。
- **结果**：42项确定性/代码层单元测试通过；原包为25项中23项通过、2项报错。测试使用合成净值和手工可核对预期，不代表真实投资策略有效。
- **现有压缩包缺少 `data/nav/*.csv`**，因此不能验证13只基金真实历史净值、用户的真实最大回撤、风险模型的统计预测能力及投资目标。`python main.py validate` 当前返回 `DATA_BLOCKED / NOT_VALIDATED`，不会复用旧业绩。

## 实际修复的缺陷

| ID | 原始问题 | 修正方式 | 验证 |
|---|---|---|---|
| F01 | T日资金用前日单位净值入账，扭曲现金流调整后收益 | T日市场变动和除息记账后，按**入金前T日**组合单位净值发行份额；之后T日净值交易 | `test_price_doubles_before_new_money_joined` |
| F02 | 复权净值用于份额市值并重复记现金分红 | 持仓及决策用**原始净值**，总回报指数仅用于收益统计/因子 | `test_ex_dividend_published_valuation_uses_raw_nav` |
| F03 | 份额折算后只修改净值、未修改真实持有份额 | 份额按折算比例同步更新；原始净值估值 | `test_split_adjusts_shares_without_creating_loss` |
| F04 | 格林“每30天开放5天”的自造赎回规则 | 对订单日的第30k天按基金交易日顺延；**仅到期日允许赎回**；无真实申请日的模拟批次不得推断赎回 | `test_simulated_lot_cannot_invent_rolling_unlock` |
| F05 | B2将现金重复加进总资产和目标赤字 | 统一使用含现金总资产V；求有申购上限的水位二分解 | `test_target_deficits_do_not_count_cash_twice` |
| F06 | Sortino把负收益子样本标准差当下行偏差 | 使用所有观测的 `sqrt(mean(min(r-rf,0)^2))` | `test_sortino_is_downside_deviation_not_selected_subsample_sd` |
| F07 | B5用历史最大回撤当当前回撤，触发后难以恢复 | 使用当前高水位回撤；增加可选滚动CDaR指标与**真实模拟赎回**开关 | `test_mdd_guard_can_recover_after_drawdown`、`test_b5_redeem_guard_executes_and_stays_research_only` |
| F08 | B4加载现时报告用于历史决策 | 不再默认加载当前报告，必须注入 as-of 风格状态 | `test_b4_cannot_read_future_style_file_by_default` |
| F09 | 风格回归使用全样本标准化，反事实把原始基金收益再次叠加 | 每个滚动窗口独立标准化，系数还原物理量纲；匹配日期后的真实残差区块重采样、多路径压力统计 | `test_rolling_factor_scaler_is_causal` |
| F10 | 股票/可转债披露缺失值当零 | 缺失值不触发确认；重仓下限为零不代表全部转债已清仓 | `test_missing_disclosure_not_treated_as_zero` |
| F11 | 组合Bootstrap错位拼接基金收益，0次超限报告零置信区间 | 按日期内连接，对缺失基金 BLOCKED，采用Wilson区间 | `test_bootstrap_aligns_funds_by_calendar_and_wilson_zero_upper_bound` |
| F12 | 2022年未上市基金当作零收益、只报起止收益 | 缺数据 BLOCKED；可计算时对完整共同路径求MDD | `test_missing_historical_series_blocks_stress_instead_of_zero_return` |
| F13 | 决策模块无数据崩溃、将原计划当优化金额、虚构格林可赎回日期 | 快照真实金额展示；禁止无验证买入；输出 `DATA_BLOCKED`；不再编造赎回日 | `tests/test_decision.py` |
| F14 | B2在搜索不同现金权重时总权重不为1 | 每个现金候选按比例重新归一化非现金权重 | `_params_for` |
| F15 | `main.py validate/sim` 数据缺失时错误异常或误报绩效 | `DATA_BLOCKED` 并明确缺失输入 | 命令行验证 |

## 修正后的核心数学定义

### 1. 组合现金流调整后的单位净值（TWR）

设 `U_{t-1}` 为在T日外部入金之前的持有单位数，`V^-_t` 为T日估值且包含待收现金、分红应收等**市场价值**。

```
P^-_t = V^-_t / U_{t-1}
ΔU_t = C_t / P^-_t
U_t = U_{t-1} + ΔU_t
P_t = (V^-_t + C_t - 交易费) / U_t
```

无交易费用时，入金本身不能改变`P_t`；日收益反映原持仓市场变化。对应第1个市场日的初始单位化另作约定。

### 2. B2补缺型现金流再平衡

已包含现金和应收款的总资产为 `V`，有效现金为`C`，现金底线为`w*_cash V`：

```
C_invest = max(0, C - w*_cash × V)
d_i = max(0, w*_i × V - V_i)
min Σ_i(V_i + a_i - w*_i V)^2
s.t. 0 ≤ a_i ≤ daily_cap_i,  Σ_i a_i ≤ C_invest
```

KKT水位法：`a_i = clip(d_i − λ, 0, daily_cap_i)`，以二分法求`λ`。现金目标候选变更时，重新归一化其他目标权重。该近似/分配模型不能替代多期动态规划。

### 3. 可选赎回与再入场

B3支持以`allow_redeem=True`在**模拟器内**真正执行减仓（不再仅仅记“审议事件”）。B5同理，支持`portfolio_redeem_fraction`和`fund_redeem_fraction`，并跟踪`exit_requested`防止日复一日重复卖出。**配置默认仍为`allow_redeem=False`**。未知真实费率与持有批次的实盘执行不得据此发出自动指令。

### 4. 现时风格风险

模型估计 `r_t = α_t + F_t β_t + ε_t`，回归每期只使用当时及之前的信息。压力测试采用当前`β_t`、历史联合因子变化和**从近期同日期样本估算的真实残差**进行区块Bootstrap，给出多路径分位数，而非一条路径的伪精确结果。信用利差等关键因子缺失时，仍然属于`FACTOR_UNIDENTIFIED`风险，不应将95%情景分位数理解为可靠概率保证。

## 仍未完成、不得宣称完成的部分

1. 13只基金真实复权净值、分红/份额折算日期对账、明显异常净值核实。**包里没有`data/nav`**；现有旧报告均来自尚有错误的版本，必须重新计算。
2. 真实申购确认T+1、赎回到账及特殊渠道交易时点；当前模拟器仅提供**显式假设**，仍没有真实费率曲线、历史限额、真实交易日历和用户逐笔申购流水。滚动运作期模型需以基金合同/实际份额确认记录最终复核。
3. B4当前注入的风格信号在每一回测窗口起点固定，尚未构建每次新公告到来时的**事件驱动重算**。因此B4/B5历史实证只能是研究候选，不能当作已验证的风格动态交易策略。
4. 未建立2026年当前风格的充分信用利差、二级资本债利差和基金有效久期因子，低R²时不能可靠归因。
5. 原版两段样本外测试不足12个月；现在会标记`INSUFFICIENT_OOS_HORIZON`，不能凭半年年化结果宣称长期收益满足4%。
6. CDaR与最大回撤是路径统计/优化代理，不构成未来5%的硬性上限；完整多期优化、真实最优权重搜索尚未完成。
7. `policy_params.json`内的比例仍为**历史候选参数**，不是经本次验证的最优比例；新决策接口**只提供数据质量状态/研究性诊断，不给实盘金额**。

## 必需的下一轮验收门槛

- 将用户本地已有的净值缓存放到`data/nav/<基金代码>.csv`，或让现有AKShare/基金采集器重建（不要求本修复包联网）。
- 对基金净值跳变、分红/拆分及历史日期覆盖逐项审计。尤其先解释金信民兴此前报告的约79%单日异常变化。
- 配置真实费用、限购、格林逐笔申请/确认日期及赎回日期。
- 冻结模型版本后重新做至少1年持有期的多窗口滚动样本外验证；新增真正随公告时间更新的B4风格信号。
- 样本外比较B0固定定投、B2现金流再平衡、B3可执行再平衡、B4风格控制和B5组合减仓；报告CAGR/MDD/CDaR/Sharpe/Sortino/XIRR、交易费用、真实可赎回比例和参数稳定性。
- 若长期年化收益超过5%与最大回撤4%的要求无足够实证支持，必须返回`NOT_VALIDATED / NO_FEASIBLE`，不得给出“最优金额”。

## 常用命令

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
python main.py validate
python main.py sim --policy B2
python main.py decision
```

缺数据时，`validate`和`sim`应明确打印`DATA_BLOCKED`；决策将13只截图快照金额作为历史展示，所有建议申购为0，现金预算留存，直到真实净值与验证恢复。