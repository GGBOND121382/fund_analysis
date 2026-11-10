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
fundkit/collect_extra.py 补充采集（季报持仓披露 + 市场风险因子）
config/account_snapshot.json  账户持仓快照（模拟批次口径，SIMULATED）
tests/              数学核验单元测试（MDD/CDaR/单位化现金流/as-of，9 条）
docs/               算法层交接适配文档（step0 映射表）
```

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
  块自助压力测试 P(MDD>5%) 与 2022-11 债灾回放）
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
