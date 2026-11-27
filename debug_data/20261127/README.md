# debug_data/20261127：ChatGPT 算法审计数据包

生成脚本：`scripts/export_debug_data_20261127.py`（只搬运数据，不含任何算法改动）。
基准提交见 `MANIFEST.json` 的 `git_sha`；数据截止 asof=2026-11-20；抽样种子 42。
逐文件的路径/字段/行数/时间范围/SHA256 见 `MANIFEST.json`。

## 数据来源与口径

- **净值序列**：`data/all_funds/{total_return,money,listed}/` 本地缓存。
  - `total_return/`（fund_nav 类基金）：东财净值+事件调整后的**总收益/复权序列**（source=`eastmoney_nav_and_event_return`），采集时已用分红/拆分事件完成复权；**原始单位净值未单独缓存**。
  - `listed/`（场内 ETF/LOF/REITs）：腾讯后复权收盘价（source=`tencent_hfq_close`）。
  - `money/`（货币基金）：万份收益口径（source 见各行），非单位净值。
- **分红/拆分事件**：`data/dividends.csv`（code, ex_date, div）与 `data/splits.json`（code → [date, ratio]）；本包导出样本基金子集。
- **交易状态**：天天基金申购/赎回状态与限额（`data/all_funds/fund_status.csv`，核查 2026-11-17）；同花顺渠道状态（仅核查过前 200 候选，核查 2026-11-24）。
- **锁定期**：名称正则（滚动/持有/定开/封闭）+ 目录的定开分类字段（periodic_status 等）。
- **费用**：申购/赎回费率**未缓存**——全部实验统一假设 roundtrip 0.1%（进/出各半）。这是已知缺口。
- **成立时间**：未缓存；样本表提供 `first_nav_date_in_cache`（缓存内首条净值日）作下限参考。
- **基准/无风险利率**：`data/factors/daily.csv`（沪深300价格指数、国债收益率曲线、risk_free_annual）；注意沪深300为价格指数非全收益，信用利差/汇率缺失（源文件 quality_flag 已注明）。

## 目录结构

```
sample/      496 只代表性基金
  sample_funds.csv       元数据 + selection_tags（M0-M3 各期限 TOP50、异常、关注、同产品份额、
                         恒定净值/负收益/低波动、类别分层随机）
  nav_shard_*.csv        净值序列分片（单片 <500KB）：code,date,nav,source
  trade_lock_status.csv  交易/锁定状态（THS 仅核查过的子集有值）
  events_dividends.csv / events_splits.csv
features/    四期限全市场特征
  features_{5y,3y,1y,6m}.csv.gz   全部基金（含 SCORED 之外的状态与原因）+
                                  D95/L_daily/annual_se/observed_log_cagr/EB 字段 +
                                  交易状态/锁定期；未经任何评分预筛
  monthly_{5y,3y,1y,6m}.csv.gz    各期限窗口内完整月度收益（仅 SCORED 基金；
                                  质检失败者无有效窗口，原因见 features 文件）
oos/         7 个形成日（2023-05-19 ~ 2026-05-20，间隔半年）样本外实验数据
  fold_{date}_pre_features.csv.gz  形成日候选特征（仅用形成日前 1y 数据，与实验同口径）
  fold_{date}_pre_monthly.csv.gz   形成前月度收益（1y 窗）
  fold_{date}_post_daily.csv.gz    形成后 6 个月日度净值（验证专用，禁止用于打分；
                                   首行=形成日或之前最后一条净值作基准）
  market_calendar.csv              沪深交易日历（沪深300有行情的日期）
  factors_daily.csv                基准与无风险利率全量
```

## 复现

```bash
git checkout 6281035   # 或 MANIFEST.json 中的 git_sha
python3 scripts/export_debug_data_20261127.py
```

## 已知缺失（如实声明，未补数）

1. 原始单位净值未缓存（仅存复权/总收益序列）；分红拆分事件见事件文件。
2. 申购/赎回费率无缓存；成立时间无缓存。
3. 历史（PIT）交易状态无存档，OOS 折中标记 `PIT_UNAVAILABLE`。
4. THS 渠道状态只覆盖每期前 200 候选；天天基金状态为 2026-11-17 快照。
5. 完整净值缓存 2.3GB（28,346 只基金全历史），超出 git 合理体积未推送；本包已含审计所需的全部切片。如确需完整缓存，可另行打包走 GitHub Release。
