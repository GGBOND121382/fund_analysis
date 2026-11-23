# 基金发现 v3｜算法实现、实证边界与 KimiCode 执行单

**定位**：2026-11-20 全市场基金发现实验的算法升级。取代“按名字判断可交易”“三轴分位+Pareto绝对优先”和“缺证据全部空榜”的混乱做法。**保留旧榜单作为基线，不宣称新的研究结果已经通过样本外检验。**

## A. 已实现的关键数学算法

### A1. 两条并行但不混淆的研究层

**历史描述性发现（不用外部交易/预测输入）**：对每一只具有完整历史净值的基金，在形成日 t 及观测窗口 h∈{6,12,36,60} 月内计算：

```
historic_CAGR = exp(log(NAV_t / NAV_(t-h)) / ((t-(t-h))/365.2425)) - 1
```

使用相同时间点及相同样本资格计算**全市场历史收益名次**、同类历史收益名次。**不是未来收益预测、不声称最优基金，不把 MDD=0 变成真实风险=0。** 当历史净值回撤极小且期间收益为正时触发 `NAV_SMOOTHING_REVIEW`；标记估值口径尚未核验。没有 `pricing_method` 不会自动标记为市值法。

**预测型投资价值发现（带风险预算）**：以形成日可知的 `forecast_mu_annual`、现金基准 `cash_annual`、往返费用 `fee_rate_roundtrip` 和经过可复现参数重估的风险情景 `risk_shock_loss`，求

```
max_{0 <= w <= 1} w * (forecast_mu_annual - cash_annual
                          - fee_rate_roundtrip * 12/H)
subject to Q95(MDD[wealth_path(w)]) and stress_MDD(w) <= B
B in {1%, 2%, 4%, 8%}
```

其中 `wealth_path(w)=(1-w)*cash_path + w*(1-fee)*fund_path`（不假装每日再平衡），回撤按完整路径计算；对 B 同一历史和相同随机情景下重复利用计算结果。**网格最优只指有限网格，不冒充连续全局最优。** H 是拟持有月数，不等于历史观测窗口 h。最终输出为“实验模型分数”，并非用户可以保证获得的收益。

### A2. 四期限独立窗口而不是重复同一基金全历史

代码：`fundkit/discovery_scoring_v3.py`

`windowed_values(nav,t,h)` 精确截取截止日以前的 h 窗口并校验历史端点和间断缺失；`joint_forecast_paths` 对此窗口的收益创新做 Stationary Bootstrap，并用预先生成的 t 时点收益预测及其估计标准误校准**同一情景集合的收益漂移和回撤**：

```
log_return_scenario[d] = bootstrapped_log_return[d]
                         - historic_mean_log_return
                         + log(1 + forecast_annual) / 252
                         + one_forecast_error_per_scenario / 252
```

其中收益预测误差对情景中全部持有天数使用相同随机扰动，保留“预测错估持续整个持有期”风险。预测标准误可由 M1 模型提供；若没有，则用过去收益序列的 Newey–West 长期方差估计作研究代理。这里仍是假设性分布，**不能由NAV推断隐藏信用风险和摊余成本影子市值**。

### A3. 收益预测：可复现类别基准与经验贝叶斯，而非主观权重

`fundkit/return_models.py` 中 `fit_category_forecasts`：
- M0：t 以前历史类别总收益基准的年化历史 log 均值；
- M1：M0 + 同类别基金超额 log 收益的正态–正态经验贝叶斯收缩（类别先验和类内方差仅来自 t 前的样本）；
- 同时输出 `forecast_se_annual`（类别基准与主动成分的抽样不确定性保守相加）。**输出用于形成研究候选，不代表未来收益可信。**
- 外部模型预测必须注明 `lookback_horizon ∈ {5y,3y,1y,6m}`，不得把一次6个月拟合的收益预测同时应用于四张榜。

### A4. 特殊估值：URL+任意“经济风险上界”不再通关

`fundkit/pricing_audit.py` 只接受明确可重算的 `BOND_DURATION_STRESS_v1` 债券因子压力模型，输入：`modified_duration, rate_shock_bp, spread_duration, spread_shock_bp, convexity, default_loss, risk_input_source_url, risk_input_known_at`。形式：

```
scenario_loss = max(0, modified_duration * rate_shock_bp/10000
                      - 0.5 * convexity * (rate_shock_bp/10000)^2
                      + spread_duration * spread_shock_bp/10000
                      + default_loss)
```

只在风险因素披露时间不晚于风险文件形成时间、所有参数可计算、产生有效正压力损失时供情景计算。用户手填的 `economic_mdd_upper` **被忽略且重算**。即使通过重算，也只标记为 `REPRODUCIBLE_STRESS_PROXY_NOT_VALIDATED`，不是保证未来回撤的经济上界，不是来源真实性认证。摊余成本产品的净值本身始终额外提示 `AMORTIZED_COST_NAV_NOT_MARKET_RISK`。**不认可未知估值基金凭0回撤获风险认证。**

### A5. 交易资格与研究分离

基金合同、开放日、持有锁定、销售渠道申赎状态、限额、往返费率分别有来源和 `known_at`；名称正则只发出人工合同核验提示。未知交易资格不删除基金历史研究；但没有合同/费用/赎回核验不允许进入已证实可执行榜。当前 `buyable_now` 是保守的**投资预检通过**，不代表真实订单成交。今天抓到的销售状态不能回填到2026-11-20。

- `research_{h}.csv`：全目录及历史信息与所有缺失原因；
- `top10_historical_return_DIAGNOSTIC_{h}.csv`：纯历史收益诊断榜，**不代表“优秀投资基金TOP10”**；
- `smoothing_review_{h}.csv`：极平滑NAV需进一步核验的高历史收益样本；
- `top10_experimental_{h}.csv`：仅模型、现金、成本、经济风险情景证据完整后可能出现，**实验分数**；
- `investable_{h}.csv`：合同和交易限制证据完整且实验分数为正时的预检候选；真实成交和独立OOS仍未证实。

### A6. 时间序列滚动实验：自动重建历史点，不能喂今天的存活基金目录

新增 `scripts/run_discovery_walkforward.py`。

- 对每个形成日 t，**必须**提供当时基金目录 `historical-catalogs/YYYY-MM-DD.csv`，其中每行 `code,name,type,known_at,source_url`，所有记录 known_at<=t；拒绝缺失目录、未来时间、重复代码；
- 仅以 t 可见的基准价格 `group,date,nav,known_at,source_url` 与历史基金NAV拟合 M0/M1；仅以当时现金总回报序列 `date,nav,known_at,source_url` 计算现金历史替代基线；
- 自动生成 `HIST_RETURN_ONLY`、`M0_CATEGORY_BASELINE`、`M1_EB_FORECAST`、`M1_NAV_RISK_PROXY` 候选，四个方法在**可共同比较的同一份额候选池**上进行比较；
- 在选择完成之后才能读取持有期后半段的真实NAV作为研究标签。无真实费用、发行时交易资格或经济风险源，**结果只能叫NAV研究评价**；
- 验证期仅使用评价终点在 `validation-end` 之前的样本，先在验证期选定固定方法，再要求 `test-start > validation-end+H` 实现持有期隔离；测试期只评价冻结的方法，使用非重叠同日期配对区间估计；
- 对当下实际历史档案不完整的情形，状态为 `EVIDENCE_INSUFFICIENT`，永不把今天全部存活基金当成历史总体，`oos_certified=false`。

**限制**：这还不是已验证可执行交易回测；亦无法自动修复历史已清盘基金缺失、评级源历史回填等数据问题。模型在未来表现是否更好，必须用上述回测实际评价。

## B. 本地已经拥有的11月20日完整缓存：立刻执行

```powershell
git pull --ff-only
python -m unittest discover -s tests -p "test_*.py" -v

python scripts/run_fund_discovery.py --asof 2026-11-20 --snapshot reports/all_funds/industry_rank_20261120 --data-dir data/all_funds --output reports/all_funds/fund_discovery_v3_diagnostic_20261120
```

输出四套描述性研究榜：`top10_historical_return_DIAGNOSTIC_*.csv`，且原来被错误筛除的基金仍保留在总目录中。这个命令故意**不伪造**缺失的未来收益预测、利率风险上界和交易费用。没有这些数据，`top10_experimental_*.csv` 仍会为空，这是正确但不同于旧版的结果：现在已有独立的纯历史收益诊断榜供研究。

## C. 如果已收集基准、现金、合同与风险来源：启用预测实验

字段如下：

- `--benchmark-history data/evidence/category_benchmark_history.csv`：`group,date,nav,known_at,source_url`；
- `--cash-rate-annual <真实有据年化>`：同时传 `--cash-source URL --cash-known-at YYYY-MM-DD`；
- `--valuation-evidence ...`：`code,known_at,source_url,pricing_method,risk_model,risk_input_source_url,risk_input_known_at,modified_duration,rate_shock_bp,spread_duration,spread_shock_bp,convexity,default_loss`；未知必须缺值，不能手动填极小风险；
- `--liquidity-evidence ...`：`code,known_at,source_url,subscription_status,redemption_status,next_exit,lock_until,total_fee_rate,max_subscription_amount,contract_source_url,contract_known_at`；
- 可单独 `--forecast-evidence ...`，但必须含 `lookback_horizon` 和 `forecast_mu_annual,forecast_se_annual,trained_through,known_at,decision_months,source_url,model_id`。

```powershell
python scripts/run_fund_discovery.py --asof 2026-11-20 --snapshot reports/all_funds/industry_rank_20261120 --data-dir data/all_funds --benchmark-history data/evidence/category_benchmark_history.csv --valuation-evidence data/evidence/valuation_stress_20261120.csv --liquidity-evidence data/evidence/liquidity_20261120.csv --cash-rate-annual <真实数值> --cash-source <已核验URL> --cash-known-at <真正公开日期> --decision-months 6 --order-amount 1000 --output reports/all_funds/fund_discovery_v3_experiment_20261120
```

以上 `<> ` 仅表示必填参数占位符，请替换实际数值，否则命令不可直接执行。**不能用11月23日的新交易快照冒充11月20日证据。** 缺少来源，保留“未知”而非硬凑榜单。

## D. 历史滚动研究实验命令

前置条件：存在每个形成日的真实当期目录（不是重新用现存基金逆推）、原始全历史净值、类别基准及现金总回报历史，并且各自有可审计的 `known_at`。

```powershell
python scripts/run_discovery_walkforward.py --historical-catalogs data/pit_catalogs --data-dir data/all_funds --benchmark-history data/evidence/category_benchmark_history.csv --cash-nav data/evidence/cash_total_return_history.csv --valuation-evidence data/evidence/dated_valuation.csv --dates data/evidence/formation_dates.csv --lookback 1y --decision-months 6 --validation-end 2024-12-31 --test-start 2025-08-01 --output reports/fund_discovery_v3_oos_1y_6m
```

必须先确认上述时间分段在真实数据范围内；此命令只示范输入契约，不代表现有用户数据已满足条件。不同h需独立跑4次。结果 `historical_folds.csv,test_paired_deltas.csv,audit.json`；`research_status=NAV_RESEARCH_TEST_ONLY` 时不能改写成“已通过投资回测”。

## E. 非拍脑袋验收条件

1. 对 007589、008693、007680、008338、003528 的名字仅触发合同审核，不能据此颁发可交易资格；价格异常平滑基金不能拥有经济零风险。
2. 新旧数据四个期限的历史特征必须真正由自己的 h 决定；外部收益模型没有 `lookback_horizon` 不能混用。
3. 对市场估值普通基金，即使缺申赎/费用证据，也能得到纯历史诊断排名；特殊估值和未知估值不会被默认标为低风险。
4. 所有收益/风险模拟在同一条件路径中运行；对收益不确定性和历史窗口进行敏感性检验；算法不能宣称无条件防回撤。
5. **KimiCode仅执行已写好的命令、抓取当日真实数据并回传来源；不自主更改评分权重、阈值或算法。** 如果某种证据（历史目录、模型输入）尚缺，应返回明确缺失清单，不允许手工捏造。
6. 测试通过只是工程验收；正式方法的投资发现有效性，必须由独立样本外成对研究证明。

## F. 实际状态（不可夸大）

原 legacy 三轴排名保留；新v3核心研究/风险/预测/时间序列滚动驱动器与反例测试已提交，2026-11-20 已发布数据可由 GitHub Actions 做脱离本地的完整快照诊断。真正的可执行榜单仍依赖可核实的当期合同、费用、历史数据及OOS验证。**不能根据代码通过测试宣称“新榜单已经科学有效”。**
