> **算法联合优化 v0.9（2026-11-10）**：在v0.8基础上新增目标权重与B2/B3/B4/B5退出规则的外层联合搜索、滚动训练/验证/测试隔离、费用敏感性及按资产类别的基金级止损阈值。当前上传包没有 `data/nav/`，真实策略绩效仍为 DATA_BLOCKED。65项本地测试通过，仅验证数学与代码逻辑，不代表基金实盘业绩。

# 场外基金日K线抓取与区间指标排序

抓取全部场外基金的单位净值日K线，按自定义时间区间计算主流绩效指标并排序。**多数据源自动故障转移**，规避单一源限流。

## 数据源（按默认优先级）

| 源 | 提供方 | 特点 |
|---|---|---|
| `danjuan` | 蛋卷基金（雪球系） | JSON，单请求拉全部历史，限流宽松，默认主源 |
| `em_f10` | 天天基金 F10 | JSON 分页（每页 20 条），含累计净值，反爬较轻，全量需翻页，适合兜底 |
| `em_js` | 东财 pingzhongdata（akshare） | 限流最严重，最后兜底 |

说明：

- **Baostock 不可用**：只覆盖股票/指数和部分场内 ETF，不支持场外基金。
- **同花顺不可用**：无公开的历史净值接口（只有最新净值快照）。
- em_f10 的 `sdate/edate` 区间参数实测已被服务端忽略，无法用于增量拉取；增量靠"全量拉取 + 本地按日期去重合并"实现（danjuan 单请求成本极低）。

## 安装

```bash
pip install -r requirements.txt
```

## 用法

```bash
# 1. 更新基金列表 + 交易状态快照（约 2.8 万只）
python main.py list

# 1.5 抓取分红送配记录（复权必需，逐只抓 F10 页面，建议每周跑一次）
python main.py dividends

# 2. 抓取日净值（建议先小批量试跑）
python main.py fetch --limit 100                    # 先抓 100 只
python main.py fetch --types 股票型,混合型           # 按类型过滤
python main.py fetch --workers 8                    # 全量抓取
python main.py fetch --sources em_f10,em_js         # 自定义数据源及优先级
python main.py fetch --refresh-days 0               # 强制重抓（默认 1 天内已抓过的跳过）

# 3. 按区间指标排序（默认排除货币型 + 不可交易基金，默认分红复权口径）
python main.py rank --start 2025-01-01 --end 2026-10-01 --metric sharpe --top 30 --with-name
python main.py rank --metric max_drawdown --types 债券型        # 最大回撤默认升序（越小越好）
python main.py rank --metric sharpe --max-avg-gap 3             # 过滤低频披露基金（定开等）
python main.py rank --metric sharpe --no-adjust                 # 关闭分红复权
python main.py rank --metric ann_return --csv result.csv        # 完整结果导出 CSV
```

## 支持的指标（`--metric`）

| 指标 | 说明 | 排序方向 |
|---|---|---|
| `sharpe` | 夏普比率（默认） | 降序 |
| `sortino` | 索提诺比率 | 降序 |
| `calmar` | 卡玛比率 | 降序 |
| `ann_return` | 年化收益率 | 降序 |
| `total_return` | 区间收益率 | 降序 |
| `ann_vol` | 年化波动率 | 升序 |
| `max_drawdown` | 最大回撤 | 升序 |

其他参数：`--rf` 年化无风险利率（默认 0.02）、`--min-days` 区间内最少交易日数（默认 60）、`--max-avg-gap` 净值平均披露间隔天数上限（过滤周频披露的定开基金，避免夏普虚高）、`--include-money` 包含货币基金（默认排除，其净值字段为万份收益口径，不能用于净值指标）、`--include-closed` 包含当前不可交易的基金（默认排除）、`--min-daily-limit` "限大额"基金的日累计限定金额下限（默认 10000 元，低于此额度视为不可交易，可挡住 0 元/10 元每天的象征性限额）、`--asc/--desc` 强制排序方向、`--rank-workers` 排序计算进程数（默认 CPU 核数）。

**可交易判定规则**（数据来自天天基金申购状态批量快照，含日累计限定金额，随 `python main.py list` 更新）：赎回状态须为「开放赎回」，且申购状态为「开放申购」，或「限大额」且日累计限额达到 `--min-daily-limit`。注意快照中"开放申购"的限额字段为 0 表示未设限。

## 项目结构

```
main.py             CLI 入口（list / fetch / dividends / rank / audit 子命令）
fundkit/sources.py  三个数据源的净值抓取实现（danjuan / em_f10 / em_js）
fundkit/fetcher.py  多源调度：顺序尝试、自动故障转移、并发批量抓取、增量合并
fundkit/storage.py  本地 CSV 存储：fund_list / fund_status / dividends / nav/{code}.csv
fundkit/metrics.py  区间指标计算（日收益率年化，252 个交易日，支持分红复权）
fundkit/adapters.py 算法层 as-of 适配接口（六表契约只读适配，published_at 规则推断）
fundkit/audit.py    数据质量与公开时点审核（python main.py audit）
fundkit/simulator.py 组合现金流模拟器（TWR/XIRR/CDaR，限额/锁定/分红/到账延迟）
fundkit/policies.py 策略族 B0-B5（定投基线/现金流再平衡/阈值/风格风控/回撤守卫）
fundkit/style.py    风格识别（披露断点+滚动因子回归+三套风险视图）
fundkit/weight_optimizer.py 训练期目标权重候选生成与情景风险筛选
fundkit/joint_optimizer.py 目标权重×止盈止损规则外层联合搜索
fundkit/collect_extra.py 补充采集（季报持仓披露 + 市场风险因子）
config/account_snapshot.example.json  本地账户快照模板（需复制并填写为 account_snapshot.json）
tests/              数学核验单元测试（MDD/CDaR/单位化现金流/as-of，9 条）
docs/               算法层交接适配文档（step0 映射表）
```

## 本地配置与数据保护

公开仓库不包含个人账户快照、基金净值缓存、申赎流水、回测报告或历史决策文件。
首次运行前，请复制 `config/account_snapshot.example.json` 为 `config/account_snapshot.json`，
并按实际账户情况填写；再根据自己的基金列表和独立样本外验证结果调整 `config/policy_params.json`。
`data/`、`reports/`、`decisions/` 及 `config/account_snapshot.json` 已加入 `.gitignore`，
不要将个人账户数据或访问凭据提交到公共仓库。

## 算法层（定投+止盈止损联合优化，交接包 algorithm-handoff-v1.0）

在数据底座之上按 `docs/step0_adapter_mapping.md` 的映射分阶段建设，当前进度：

- 第0步 适配映射表：`docs/step0_adapter_mapping.md`
- 第1步 数据质量审核：`python main.py audit` → `data/quality/step1_audit.{json,md}`
- 数据补齐：`python main.py extra` → 季报持仓披露（`data/disclosures/`，含 published_at
  规则推断、重仓债券分类 lower_bound）+ 市场风险因子（`data/factors/daily.csv`，
  国债收益率曲线/沪深300价格指数/上海金；信用利差与汇率无可靠源，按契约留 MISSING）
- 第2步 风格识别：`python main.py style` → `reports/style_report.json`（披露断点 +
  滚动加权岭回归 beta/R²/条件数 + 双证据合并状态机 + 三套风险视图反事实压力）
- 第3步 现金流模拟器 + 第4步 策略族 B0-B5：`python main.py sim --policy B0|...|B5` →
  `reports/sim_*_{daily,events}.csv`（TWR 单位净值、CAGR/Sharpe/Sortino/MDD/CDaR95/XIRR；
  申购限额、30天滚动锁定、分红到账、赎回到账延迟、反未来信息 as-of；
  B2 现金流注水再平衡 / B3 阈值再平衡（默认赎回仅提案）/ B4 风格风控状态机 /
  B5 组合回撤守卫+逐基金暂停再入场；参数见 `config/policy_params.json`，
  均为候选值，待第5步样本外验证冻结）
- 第5步 样本外验证：`python main.py validate` → `reports/oos_trials.csv`（全部试验）、
  `reports/oos_summary.json`（风险优先选参、冻结 test 折、可行性判定、帕累托前沿、
  块自助压力测试 P(MDD>4%) 与 2022-11 债灾回放）
- 第6步 研究建议：`python main.py decide` → `decisions/YYYY-MM-DD.json`
  （jsonschema 校验，RESEARCH_ONLY）、`reports/fund_decisions.csv`、
  `reports/strategy_report.md`、`reports/full_research_audit.md`

测试：`python -m unittest discover -s tests`（25 条：数学核验 9 + 模拟器 9 + 策略 5 + 决策 2）

当前所有算法层输出限 `RESEARCH_ONLY`：账户批次为模拟（`ACCOUNT_DRAWDOWN_UNVERIFIABLE`），
季报持仓与市场风险因子未接入（`MISSING`），不提供任何交易建议。

## 已知注意事项

- 全量抓取约 2.7 万只基金（不含货币型），danjuan 源实测约 8 只/秒（8 线程），预计 1 小时左右。
- 个别基金（后端份额、已清盘等）在所有源均无数据，属正常现象，失败列表会列出。
- 定开债/定开混合等基金按周披露净值（avg_gap ≈ 6），收益率曲线被"熨平"，夏普/索提诺会系统性虚高，建议排序时加 `--max-avg-gap 3`。
- **分红/拆分复权**：`rank` 默认用后复权口径（除息日分红加回收益、份额折算日按比例放大），`--no-adjust` 可关闭。分红数据（`data/dividends.csv`）由 `python main.py dividends` 逐只抓取东财 F10 分红页生成，拆分数据存于 `data/splits.json`。**不要用 akshare 的批量接口 fund_fh_em，它只覆盖近期几个月的记录，用于复权会严重失真**。完整性经"累计净值−单位净值=累计分红"恒等式全量对账：无拆分基金 96.2% 完全一致；其余约 260 只为 2007-2015 年间的老基金，东财 F10 对其早期分红/拆分事件记录不全（数据源侧残缺，清单见 `data/dividend_mismatch.csv`），影响仅限于覆盖那些远古事件的超长区间。
- 数据缓存在 `data/nav/` 下，`rank` 完全离线计算。

---

## v0.8（2026-11-10）目标权重优化升级

**最新账户目标：年化净收益超过5%，组合最大回撤目标不超过4%**（此前文档中任何4%年化/5%回撤均为旧目标，已废弃）。

本版新增 `fundkit/weight_optimizer.py`（有约束的联合收益情景目标权重搜索），`python main.py optimize-weights` 命令，和 `--weights-report` 的历史模拟时间冻结入口。真实净值不足时只返回 `DATA_BLOCKED`，不出交易指令。原`decide`在目标权重尚未验证时不再展示硬编码的45%/35%为正式目标。

具体数学公式、超参数、局限与数据需求：**`docs/WEIGHT_OPTIMIZATION_V08.md`**。本版本的数学单元测试为53项；**测试通过不代表真实基金策略通过收益或最大回撤验收**。


## v0.9（2026-11-10）目标权重 + 止盈止损联合优化

正式目标统一为：**长期年化净收益严格超过5%，组合最大回撤风险预算不超过4%**。风险预算优先；找不到可行策略就返回不可行，不修改目标。

新增：

- `fundkit/joint_optimizer.py`：目标权重候选与 B2/B3/B4/B5 规则的联合外层搜索；验证期选参、测试期只跑冻结候选。
- `fundkit/weight_optimizer.py::generate_weight_candidates`：生成风险—收益多样化权重候选，保留可能由动态退出规则改善的静态高风险候选。
- B4 支持每21个工作日更新一次 as-of 风格信号；不读取未来公告。
- B5 基金级止损信号改用分红调整后的总回报净值，并支持按基金代码设置不同暂停/恢复阈值。债券与黄金候选阈值区间不同，但都只是待验证参数。
- 新 CLI：`python main.py optimize-strategy`。默认训练504日、验证252日、测试252日；费用压力情景默认为0bp/30bp。

示例：

```powershell
python main.py optimize-strategy --asof 2026-11-10
python main.py optimize-strategy --asof 2026-11-10 --folds 2 --step-days 252
python main.py optimize-strategy --asof 2026-11-10 --weight-pool 1200 --weight-candidates 8 --weight-paths 128 --fee-stress-bps 0,10,30
```

完整公式、搜索空间、数据契约和局限见 `docs/JOINT_OPTIMIZATION_V09.md`。所有输出保持 `RESEARCH_ONLY`；当前账户逐笔批次、真实费率、初始持仓转换成本及信用利差因子仍未完整接入，不能自动交易或声称达到5%/4%。

## v1.0 risk-budgeted strategy upgrade

See docs/UPGRADE_V10.md for robust return shrinkage, concentration guardrails, B5 stop-loss/trailing take-profit, cost-basis accounting, covariance-risk de-risking, index-fund scope, and non-overlapping OOS rules. This release remains research-only and does not claim the 5% net-return / 4% maximum-drawdown objective has been achieved.


## v1.0.1 validation hardening (2026-11-11)

- Candidate ranking now reports historical Sharpe and block-bootstrap Sharpe Q25; among candidates meeting the 5% return and 4% MDD research constraints, Sharpe is the first tie-break/selection objective. The annual risk-free assumption defaults to 2% and is configurable.
- Counterfactual initial purchases now pay the same subscription-fee stress assumption as subsequent buys. The initial fee is included in starting unit NAV, TWR, CAGR, Sharpe, XIRR, fees and turnover.
- B5 blocks new purchases when fewer than two published total-return NAV observations are available and no observed high-water mark exists.
- DATA_BLOCKED joint-optimization and OOS-validation runs now write explicit structured reports rather than leaving a stale prior report in place.
- All current fund categories, including CSI 300 / S&P 500 / Nasdaq index funds, remain in the investable fund universe; no direct stocks are introduced.

### 一键本地验证（v1.0.1）

在已具备 `config/account_snapshot.json`、`data/nav/`、`data/factors/` 和交易状态缓存的机器上运行：

```bash
python scripts/run_v10_research.py --asof YYYY-MM-DD
```

该脚本依次执行数据审计、组合风险分析、目标权重优化、权重/止盈止损联合搜索、滚动样本外验证和报告导出。结果写入 `reports/v10_research_run.json`。若返回码为2，表示数据阻断、样本外证据不足或某一步失败；不能视为策略通过。脚本不会联网补造本地缓存，也不会下单。

v1.0.1 的Sharpe默认使用2%年化无风险利率假设，可用 `--risk-free-rate` 调整；首次反事实建仓申购费也计入回测净值。


**Live monitoring limitation:** B5's pause/recovery/cooldown state is currently in-memory inside a simulation. Persistent state storage, restart recovery, idempotent order submission and actual-position reconciliation are not implemented; do not run this as a live auto-trading monitor.
