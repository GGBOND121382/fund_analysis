# 基金发现 v3 执行报告（2026-11-20 截面）

**本报告是研究诊断，不是投资推荐。** 历史收益诊断榜不代表"优秀基金 TOP10"；实验榜单为空是证据闸门的设计行为，不是运行失败。

- 代码版本：分支 `codex/fund-analysis-all-fund-ranking-20261114`，基线提交 `5615b92`（+ 本次 1 个本地提交：walkforward 缺输入闸门修复）
- 工程验收：`python3 -m unittest discover -s tests -p "test_*.py"` → **206 条全部通过**（仅工程验收，不构成算法有效性证明）
- 数据截止：2026-11-20；净值缓存 28346 目录份额 / 25954 只有效 / 24002 只截止日精确匹配

## 1. 执行命令（可复现）

```bash
# 四期限历史诊断（离线，复用本地缓存）
python scripts/run_fund_discovery.py --asof 2026-11-20 \
  --snapshot reports/all_funds/industry_rank_20261120 \
  --data-dir data/all_funds \
  --output reports/all_funds/fund_discovery_v3_diagnostic_20261120

# 交易状态快照（真实抓取，时间戳即 known_at，不可回填）
python scripts/fetch_fund_trade_status.py --output data/evidence/purchase_20261123_2141.csv

# 现金利率就位后的对照实验（其余证据缺失，用于记录闸门逐级状态）
python scripts/run_fund_discovery.py --asof 2026-11-20 \
  --snapshot reports/all_funds/industry_rank_20261120 --data-dir data/all_funds \
  --cash-rate-annual 0.0125 \
  --cash-source "https://finance.caixin.com/2026-10-30/102499195.html" \
  --cash-known-at 2026-10-30 --decision-months 6 --order-amount 1000 \
  --output reports/all_funds/fund_discovery_v3_cash_only_20261120

# 滚动样本外（前置证据缺失，闸门明确拒绝并落审计文件）
python scripts/run_discovery_walkforward.py --historical-catalogs data/pit_catalogs \
  --data-dir data/all_funds --benchmark-history data/evidence/category_benchmark_history.csv \
  --cash-nav data/evidence/cash_total_return_history.csv --dates data/evidence/formation_dates.csv \
  --lookback 1y --decision-months 6 --validation-end 2024-12-31 --test-start 2025-08-01 \
  --output reports/fund_discovery_v3_oos_1y_6m
```

## 2. 数据覆盖率（四期限历史诊断）

| 期限 | 目录特征行 | 历史候选 | 缺历史/不可比 | 平滑复核触发 |
|---|---|---|---|---|
| 5y | 28346 | 6481 | 21865 | 202 |
| 3y | 28346 | 8903 | 19443 | 266 |
| 1y | 28346 | 11758 | 16588 | 405 |
| 6m | 28346 | 12173 | 16173 | 444 |

5y 候选受限于数据源约 3 年历史；`smoothing_review_*.csv` 列出净值异常平滑、需人工核验估值口径的样本（摊余成本法等）。

## 3. 证据来源审计（known_at / source_url 纪律）

| 证据 | 状态 | known_at | 来源 |
|---|---|---|---|
| 现金利率 1.25%（央行隔夜逆回购操作利率） | ✅ 已核验，≤asof | 2026-10-30 | [央行公开市场业务公告（财新转引）](https://finance.caixin.com/2026-10-30/102499195.html) |
| 基金交易状态快照 | ✅ 已抓取，28109 码 | 2026-11-23T21:42+08 | AKShare `fund_purchase_em`（天天基金渠道公示） |
| 类别基准历史 | ❌ 缺口 | — | 无带 PIT known_at 的逐组总回报序列 |
| 估值/经济风险输入（久期、利差久期、凸性、违约损失） | ❌ 缺口 | — | 无逐基金可重算输入及来源 |
| 逐形成日历史目录 | ❌ 缺口 | — | 无存档，不能用今日存活目录逆推 |
| 现金总回报历史序列 | ❌ 缺口 | — | 仅有单点利率，无逐日总回报 NAV |

**关键限制**：今日抓取的交易快照 known_at=2026-11-23 晚于 asof=2026-11-20，按 PIT 纪律不能用于 11-20 实验（文档明确禁止），仅存档供未来截面使用。

## 4. 结果

- **描述性诊断榜**：`top10_historical_return_DIAGNOSTIC_{5y,3y,1y,6m}.csv` 已生成（6m 前列为先锋聚优A、大摩优悦安和、银行ETF 系列等——纯历史收益事实，非推荐）。
- **实验榜**：现金闸门通过后状态从 `CASH_FORECAST_MISSING` 变为 `FORECAST_MISSING`；因无合规预测与风险证据，`top10_experimental_*` 与 `budget_*` 全部为空，`oos_acceptance=EVIDENCE_INSUFFICIENT`。缺证据不凑榜。
- **异常基金**：003528/003529（66 个月合同周期，已核验排除）；710301 五年期单日涨跌超 3% 待核验；202~444 只平滑复核样本待人工核验估值口径；896 只定开周期未核实 fail-closed 排除。
- **样本外**：walkforward 闸门输出见 `reports/fund_discovery_v3_oos_1y_6m/audit.json`（`EVIDENCE_INSUFFICIENT`，列出缺失输入）。既有 12 折回放（`industry_rank_20261120`）为当前目录回放，有幸存者/分类偏差，不构成无偏验证；v3 主榜不在其前推范围内。

## 5. 本轮代码修复（最小复现 → 修复 → 测试）

- **问题**：`run_discovery_walkforward.py` 在证据文件缺失时以 `FileNotFoundError` 裸崩，而非文档承诺的 `EVIDENCE_INSUFFICIENT`。
- **复现**：缺 `data/evidence/formation_dates.csv` 直接执行 D 命令即崩（pandas 内部堆栈）。
- **修复**：仅增加输入文件存在性 preflight，缺失时写 `audit.json`（列出缺失清单、`oos_certified=false`）并以退出码 2 结束；不改任何评分/权重/阈值；目录缺失仍走 fold 级既有闸门（保持原测试语义）。
- **测试**：新增 `test_missing_inputs_produce_evidence_gate_not_traceback`；全套 206 条通过。

## 6. 后续解锁条件

1. 类别基准历史：需逐组、逐日、带真实 `known_at` 的总回报序列（如基金公司/指数公司公告存档），拿到后 M0/M1 预测与 walkforward 即可运行。
2. 估值风险输入：逐基金久期/利差久期/凸性/违约损失及来源日期（季报披露），拿到后 `BOND_DURATION_STRESS_v1` 可重算。
3. PIT 历史目录：需交易方/数据商的历史快照存档；在此之前任何 OOS 都只是诊断。
