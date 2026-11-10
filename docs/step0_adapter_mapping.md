# 第0步｜现有工程 → 算法层接口契约 最小差异适配方案

日期：2026-11-09｜依据：`05_AKSHARE_ADAPTER_CONTRACT.md`、`06_CODEX_TASK.md` 第0步
范围：只读审查，**未修改任何采集/存储/排序代码**；本步只产出映射表与缺口清单。

## 0. 一句话结论

现有工程的净值/分红/拆分/交易状态数据底座可以直接复用，算法层只需新增一个 `adapters` 转换层；
契约要求的 5 张输入表中，**1 张基本就绪（净值）、1 张部分就绪（交易规则）、3 张整体缺失
（季报持仓披露、市场风险因子、账户逐笔流水）**。缺失项不阻塞开工，但按契约必须标注
`MISSING`/`DATA_BLOCKED`，在补齐前相关输出只能为 `RESEARCH_ONLY`。

## 1. 现有资产盘点（实测）

### 代码

| 文件 | 职责 |
|---|---|
| `main.py` | CLI：`list` / `fetch` / `dividends` / `rank` |
| `fundkit/sources.py` | 三数据源：danjuan（默认，单请求全历史）、em_f10（含累计净值）、em_js（兜底） |
| `fundkit/fetcher.py` | 多源调度/重试/故障转移、基金列表、分红、拆分、交易状态 |
| `fundkit/storage.py` | 本地 CSV 缓存 |
| `fundkit/metrics.py` | 后复权 `adjust_nav` + sharpe/sortino/calmar/MDD 等区间指标 |

### 数据缓存（data/）

| 文件 | 内容 | 规模 |
|---|---|---|
| `fund_list.csv` | code, name, type | 28172 只 |
| `fund_status.csv` | code, sub_status, red_status, daily_limit | 27884 只（当前快照，无历史） |
| `nav/{code}.csv` | date, nav（em_f10 源时含 acc_nav） | 26613 只 |
| `dividends.csv` | code, ex_date, div（每份现金分红） | 7161 只有记录 |
| `splits.json` | {code: [(date, 折算比例k)]} | 223 只 |
| `dividend_mismatch.csv` | 复权对账不符名单（数据源侧残缺） | 260 只 |

### 账户 13 只持仓基金的数据就绪情况（2026-11-09 核查）

全部有本地净值。710301 自 2012 年起 3435 行；004400 自 2017 年起 2301 行且有完整分红记录；
8 只股指基金净值至 2026-10-23，其余至 2026-10-26。
**注意：净值普遍落后今日约 2 周**（`fetch` 增量更新即可，无需重拉历史）。
本地净值均为 danjuan 源，**无 acc_nav 列**；004400 在 `fund_status.csv` 中为
"限大额 500 元/日"，与截图每日定投 500 一致。

## 2. 字段映射表（现有字段 → 接口契约 → 缺失字段）

### 表1 `nav_total_return_daily` — 基本就绪（2 个关键缺口）

| 契约字段 | 现有来源 | 状态 | 说明 |
|---|---|---|---|
| `fund_code` | nav 文件名 / fund_list.code | ✅ | 六位代码，zfill 已保证 |
| `nav_date` | nav CSV `date` | ✅ | |
| `unit_nav` | nav CSV `nav` | ✅ | |
| `cumulative_nav` | nav CSV `acc_nav` | ⚠️ 部分 | 仅 em_f10 源有；danjuan 源文件无此列。契约允许缺（非默认总回报口径） |
| `adj_nav_total_return` | `metrics.adjust_nav` 由分红+拆分事件重建 | ✅ 可重建 | 契约允许"或拥有可重建事件"；分红/拆分事件表已有。260 只对账不符基金见 dividend_mismatch.csv，须排除或标记 |
| `cash_dividend_per_unit` | `dividends.csv` (ex_date, div) | ✅ | |
| `split_ratio` | `splits.json` | ✅ | |
| **`published_at`** | **无** | ❌ **关键缺失** | 净值 CSV 只有估值日，无披露时间戳。无偏回测要求"决策时点 ≤ 公布时点"。实测可行的近似：中国场外基金 T 日净值通常在 T 日晚 20:00 后公布，可用"T 日净值 T+1 日可用"规则构造保守 published_at，但须在证据状态中标注为规则推断而非真实时间戳 |
| `source` / `fetched_at` / `quality_flag` | 无逐行记录 | ❌ | 合并后丢失来源；文件 mtime 只能当 fetched_at 弱代理。适配层需补记录 |

### 表2 `fund_holdings_disclosures` — ❌ 整体缺失（第2步风格识别的前置）

现有工程从未抓过季报持仓。契约要求 report_asof + **published_at 双时间戳**、
券种结构（利率债/信用债/银行二级资本债/永续）、久期、前十大债券、基金规模等。

候选数据源（**未经实测，第1步验证**）：
- 东财 F10 资产组合/债券持仓页（fundf10 `zcpz`/`zqcc` 类接口，akshare `fund_portfolio_hold_em` 只覆盖重仓股）
- 季报 PDF 公告日期作为 published_at 来源
- 只有前五大/前十大重仓披露时按契约记 `lower_bound`，不得伪造全基金权重

### 表3 `market_risk_factors_daily` — ❌ 整体缺失（风格回归与压力测试的前置）

国债收益率（1/3/5/10/30Y）、政金债、AAA 信用利差、二级资本债/永续利差、转债/沪深300/
黄金全收益指数、USDCNY、日频无风险利率——现工程均无。

候选数据源（**未经实测，第1步验证**）：
- 国债收益率曲线：akshare `bond_china_yield`（中国货币网）
- 信用利差：若无可靠直接来源，契约禁止"造一条利差"，只能用可溯源券种收益率差自行计算并标注
- 沪深300全收益：中证指数 H00300；黄金：上海金或 000218 自身净值仅作 C 级代理
- `risk_free_daily`：可用 1Y 国债或货基收益，单位必须统一为小数

### 表4 `account_lots_and_cashflows` — ❌ 整体缺失（硬性 BLOCKED 项）

只有 2026-11-05 截图市值（13 只共 195123.58 元），**无逐笔确认份额、成本、费率、
分红方式、锁定期批次**。契约明确：未获取流水前 `redeemable_shares`/`tax_lot_gain`/
`next_redeem_window` 不得假造；历史模拟只能以**明确标注的模拟批次**进行。

→ 后果：真实账户的历史 TWR/MDD 无法验证，输出须带 `ACCOUNT_DRAWDOWN_UNVERIFIABLE`；
需要用户提供渠道导出的交易确认流水（或接受全模拟口径）。

### 表5 `fund_trade_rules` — ⚠️ 部分就绪

| 契约字段 | 现有来源 | 状态 | 说明 |
|---|---|---|---|
| 申购/赎回状态、日累计限额 | `fund_status.csv` | ⚠️ 当前快照 | **无 rule_effective_from 历史**，回测历史时点限购状态不可得 |
| 申购下限/费率分档/持有期费率 | 无 | ❌ | 候选：东财 F10 交易规则/费率页（第1步验证） |
| 锁定期/滚动持有窗口 | 无 | ❌ | 021559"30天滚动持有"必须逐笔批次窗口，不能用全局 T+30 代替 |
| 确认时点/到账延迟/截止时点/节假日 | 无 | ❌ | 模拟器需要；可用通用规则（T+1 确认、15:00 截止）作默认假设并标注 |

## 3. 数据可用性分级（按契约 §4）

| 数据 | 等级 | 理由 |
|---|---|---|
| danjuan 净值 | B/SINGLE_SOURCE | 单源；可对抽样基金用 em_f10（含累计净值）交叉对账升至 A |
| 分红/拆分事件 | B | 东财 F10 单源；260 只远古事件残缺已在对账名单 |
| 交易状态快照 | B | 单源且无历史 |
| published_at（拟构造） | C | 规则推断，非真实时间戳 |
| 季报持仓/风险因子/账户流水 | D/MISSING | 未接入 |

## 4. 现有代码观察（不改，仅记录）

1. `fundkit/metrics.py:56-61`：`adjust_nav` 尾部有一段不可达的死代码（`slice_by_date` 的重复体），
   无功能影响，属清理项而非错误。
2. `storage.save_nav` 只落 date/nav/acc_nav，不记录数据源与抓取时间——适配层需补溯源字段。
3. 净值缓存落后约 2 周（最新 2026-10-23/26），下次 `fetch` 增量补齐即可，无需重拉。

## 5. 最小修改清单（后续步骤新增，不动旧代码）

| 新增模块 | 内容 | 对应契约 |
|---|---|---|
| `fundkit/adapters.py` | `nav(codes, until)` / `disclosures` / `risk_factors` / `account_snapshot` / `cashflows_and_lots` / `trade_rules` 六个只读接口，内部复用 storage；强制 `as-of` 过滤 | 05 §3 |
| `data/quality/` | 逐表完整性审核输出（含证据状态） | 第1步 |
| `tests/test_finance_math.py` | 直接引入交接包 reference 的 10 条确定性测试 | 04 |

## 6. 第0步验收

- [x] 未修改采集/存储/排序实现
- [x] 未要求重拉已缓存数据（仅指出净值有 2 周自然滞后，走既有增量逻辑）
- [x] 5 张契约表逐字段映射完成；3 项整体缺失 + 2 项关键字段缺失已如实标注
- [x] 待用户确认：账户流水（表4）是提供真实导出，还是接受"全模拟批次 + 不可验证"标注
  → 用户已选择**模拟批次口径**（2026-11-09），后续接入真实流水可升级；
  落地于 `config/account_snapshot.json`（lots_mode=SIMULATED）与 `fundkit/adapters.py`

## 7. 第1步已落地（2026-11-09）

- `fundkit/adapters.py`：六接口只读适配层（nav/disclosures/risk_factors/account_snapshot/
  cashflows_and_lots/trade_rules），强制 as-of 过滤 + `assert_no_lookahead` 自检；
  published_at 按"净值日+2个自然日 23:59"规则推断并标注 C 级证据
- `fundkit/audit.py` + `python main.py audit`：五表完整性审核 → `data/quality/step1_audit.{json,md}`
- `tests/`：交接包 reference 数学核验测试 9 条全部通过
- 审核结论：`RESEARCH_ONLY`（披露/因子 MISSING，账户流水 SIMULATED）

## 8. 第2-6步进展（2026-11-09）

- 数据补齐：`fundkit/collect_extra.py`（`main.py extra`）→ 季报持仓披露（13只×8-10期，
  含债券分类 lower_bound）+ 风险因子（国债曲线 1708 交易日/沪深300/上海金）；
  信用利差与汇率无可靠源，留 MISSING
- 第2步 `fundkit/style.py`（`main.py style`）：710301 确认 2025 年股票+转债退出断点
  （2024 末 31.3%→连续两期≤0.5%），三视图风险口径分离
- 第3步 `fundkit/simulator.py`：TWR/限额/锁定/分红/到账延迟；9 条确定性测试
- 第4步 `fundkit/policies.py`：B0-B5 全族（参数 `config/policy_params.json` 候选值）
- 第5步 `fundkit/evaluation.py`（`main.py validate`）：滚动 train/val/test + 全试验记录
  + 块自助压力 + 2022 债灾回放 + 可行性判定
- 第6步 `fundkit/decision.py`（`main.py decide`）：schema 校验的 decisions/*.json +
  strategy_report.md + full_research_audit.md