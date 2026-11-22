# 全市场基金发现 v2：证据驱动算法与运行说明

**目标**：四个历史观察期限 5y、3y、1y、6m 的研究候选；未来拟持有期限 H 与观察期限严格区分。此版本是可执行的证据核验/情景决策实验框架，不是已经通过 OOS 的正式基金评级。原 `run_industry_fund_ranking.py` 的三轴 Pareto 保留为基线，**不覆盖历史 CSV**。

## 核心设计（不会再用正则确定交易资格或零回撤奖励）

1. `fundkit/fund_master.py`：交易资格从有 `known_at,source_url` 的申赎状态和合同记录取得。名称规则只输出 `contract_review_hint`。未知或证据冲突均不能标为“可立即交易”；申购、赎回、持有期、费用、限额分别校验。
2. `fundkit/pricing_audit.py`：估值方法 `market_value/amortized_cost/mixed/unknown`；零回撤/净值平滑是审计触发条件。无法取得可核实经济压力风险上界时输出 `RISK_UNVERIFIED`，不把历史净值零回撤当真实风险零。
3. `fundkit/return_models.py`：同类经验贝叶斯的正态先验收缩函数；完整预测结果由截止时点已知的训练数据与模型记录供给。未验证的预测不能冒充预测准确。
4. `fundkit/opportunity_rank.py`：固定现金与单基金的初始买入持有比例 `w`，通过圆周平稳区块 Bootstrap 产生未来 NAV 路径，以 `max(Q95(MDD), 独立经济冲击路径 MDD)` 给出保守*情景估计*。在 `B={1%,2%,4%,8%}` 各风险预算内网格优化年化现金超额预期收益减真实往返费用，不使用原 Pareto 的绝对优先、也不再使用 1/3 等权。**情景估计不是未来硬回撤上限**。
5. `fundkit/discovery_oos.py` + `scripts/evaluate_fund_discovery_oos.py`：只接受 PIT 形成日证据时间戳，先检查未来信息泄漏，再用真实后续 NAV/现金 NAV 验证；只对不重叠形成期进行成对差异和区间估计。没有真实历史快照及嵌套训练验证记录时，永不宣布正式 OOS 通过。

## 运行一次完整已发布快照审计（无需网络、无需伪造分数）

```powershell
python scripts/run_fund_discovery.py --asof 2026-11-20 --snapshot reports/all_funds/industry_rank_20261120 --output reports/all_funds/fund_discovery_v2_20261120
```

输出（每个期限一份）：`research_*.csv`、`investable_*.csv`、`top10_experimental_*.csv`、`budget_{1,2,4,8}pct_*.csv`、`audit.json`。

**重要**：原发布快照已因旧正则筛选排除896个“周期未核实”基金，部分原始历史特征已不存在。未传 `--data-dir` 时会在 `research_*.csv` 中保留这些代码与原因，但不能恢复其历史收益，不可声称完成完整全市场重新评估。要恢复全目录候选、包括原来被旧筛选排除的产品，请使用完整本地缓存：

```powershell
python scripts/run_fund_discovery.py --asof 2026-11-20 --snapshot reports/all_funds/industry_rank_20261120 --data-dir data/all_funds --output reports/all_funds/fund_discovery_v2_allfunds_20261120
```

即使只有历史净值，新程序也不会凭空编造未来预期收益、经济风险和往返费用。**缺证据的投资价值分数保持空值，不能强行凑出“优质基金TOP10”。**

## 真实交易状态抓取

官方 AKShare 公募基金 `fund_purchase_em()` 基于天天基金公示数据，提供基金代码、当前申购/赎回状态、下一开放日、限额、前端申购费等。它仅能证明“平台当时显示什么”，**不能替代合同实际赎回权、真实全渠道可用额度和完整申赎费用**。抓取后记录实际捕获时间，禁止回填到过去日期。

```powershell
python scripts/fetch_fund_trade_status.py --output data/evidence/purchase_20261122_1600.csv
```

文件名应换成真正抓取时间；已有文件不可覆盖。`next_reported_open_day` 是平台的原始字段，不自动填入 `next_exit`（实际持有人退出资格需要合同及最近开放公告）。`front_fee_percent` 不是 `total_fee_rate`，也不会被自动解释为完整往返成本。场内交易基金的“场内买入”不能替代交易所停牌/价差核验。

## 证据数据文件格式

所有证据文件都是逐基金份额 `code`，必须带 `known_at`（记录当时已公开的时间）及 `source_url`（官方基金公司/交易所或平台原始页面）。已晚于形成日才知道的信息一律剔除。

### 交易状态 `--liquidity-evidence`

```csv
code,known_at,source_url,subscription_status,redemption_status,next_exit,lock_until,total_fee_rate,max_subscription_amount,contract_source_url,contract_known_at
```

状态只能为 `OPEN/LIMITED/PAUSED/CLOSED/UNKNOWN`。要发放 `buyable_now` 预检标记，必须同时提供 `contract_source_url` 和不晚于评价日的 `contract_known_at`，以及资金限额、完整交易费率、可执行退出日；仅有销售平台状态会保留 `STATUS_SOURCE_OBSERVED` 但买入预检为否。`next_exit` 是该投资者在该拟持有期限内可以退出的已核验日期，不是简单基金开放日。交易费用为买卖和价差合计的小数比例，额度以实际申购金额相同币种表示。记录矛盾时 `CONFLICT`，超过7天 `STALE`（适用于现时状态，历史回放需用当时快照）。

### 估值与风险 `--valuation-evidence`

```csv
code,known_at,source_url,pricing_method,economic_mdd_upper
```

`economic_mdd_upper` 为有明确来源、窗口及测算方法的保守经济压力损失比例（0–1），不是纯净值波动或猜测风险底线。对摊余成本法债基，没有足够持仓、久期、信用风险及影子估值数据时**应留空**。有数值仅表示有外部证据输入，不代表模型已被样本外证实。

### 预测 `--forecast-evidence`

```csv
code,known_at,trained_through,source_url,model_id,forecast_mu_annual,decision_months
```

`forecast_mu_annual` 为预期年化收益率小数，`trained_through` 不得晚于 `known_at`，且预测持有期限要与 `--decision-months` 一致。经验贝叶斯只是候选模型，未在历史前推证明其优越性前不可直接宣布最优。

### 可复现的 M0/M1 类别基准预测（未通过 OOS）

使用 `--data-dir data/all_funds --benchmark-history data/evidence/category_benchmark_history.csv` 生成截止日可知的 M0（历史类别基准）和 M1（同类经验贝叶斯收缩）预测，按5y/3y/1y/6m分别估计。基准CSV字段：`group,date,nav,known_at,source_url`，`nav` 必须是同类可比较的总回报价格序列，`known_at` 为该条价格实际已知时间；不能把最新版基准回填到过去。输出 `model_diagnostics_{h}.csv` 包含训练截止日、月度样本数、同类先验和收缩分量。这是实验预测候选模型，不能在没有独立前推验证时直接升版。

### 启用实验分数（不是已证实榜）

```powershell
python scripts/run_fund_discovery.py --asof 2026-11-20 --snapshot reports/all_funds/industry_rank_20261120 --data-dir data/all_funds --liquidity-evidence data/evidence/liquidity_20261120.csv --valuation-evidence data/evidence/pricing_20261120.csv --forecast-evidence data/evidence/forecast_20261120.csv --cash-rate-annual 0.015 --cash-source https://your-verifiable-historical-cash-source --cash-known-at 2026-11-19 --decision-months 6 --order-amount 1000 --output reports/all_funds/fund_discovery_v2_experiment_20261120
```

其中 `0.015` 仅为命令示范，不代表2026年11月20日的实际现金利率；真正运行需用当时可得的有据数据。未提供本地完整历史净值则不得基于这条命令启用Bootstrap。

`top10_experimental_*.csv` 必须带 `EXPERIMENTAL_NOT_OOS_VERIFIED` 标记，投资执行表仅是条件预检查、不是真实下单成功或经验证的投资建议。预测比现金低的产品机会增量为0，不强行填满前十。

## 样本外验证

```powershell
python scripts/evaluate_fund_discovery_oos.py --selections data/pit_selections.csv --fund-nav-dir data/all_funds/total_return --cash-nav data/actual_cash_nav.csv --output reports/oos_fund_discovery_v2
```

`pit_selections.csv` 至少含 `formation_date,evaluation_end,code,method,horizon,weight,fee_rate,catalog_known_at,forecast_known_at,valuation_known_at,liquidity_known_at,price_known_at,fees_known_at`。必须使用**历史形成日当时**可见的名单和预测；当下的交易快照不能填充历史。结果含 `oos_results.csv, oos_paired_deltas.csv, oos_status.json`；即使研究期指标改善，没有独立的历史候选集、执行记录、冻结选参及完整无偏数据时，也明确写 `oos_certified=false`。

## 验收与已知边界

- 测试包括007589/008693/007680/008338/003528名称必须触发合同核验、无凭证不能认定可交易、前端费用不能冒充往返费用、摊余成本法净值零回撤不可等同经济零风险。
- 同样风险约束下高净超额收益模型不得劣于低净超额收益模型；现金收益高于基金预测时净增量不大于0。
- 对路径模拟、费用、影子冲击、缺失数据、时间泄漏及非重叠检验有独立测试。
- **CI自动执行四期限真实全量快照审计**，产物 `fund-discovery-20261120-diagnostic` 包含覆盖与未核验清单。本轮尚没有真实合同、经济风险和预测训练数据，不应出现有误导性的正式Top10。
- 保留 `fundkit/defensive_screen.py` 原Pareto完整逻辑作为基线；未声称当前新模型改善了样本外结果。

参考：AKShare `fund_purchase_em` 文档 https://akshare-hh.readthedocs.io/en/latest/data/fund/fund_public.html；Lo 2002 Sharpe统计 https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios；Politis/Romano 1994 Stationary Bootstrap https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870
