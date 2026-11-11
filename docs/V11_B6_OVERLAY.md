# fund_analysis v1.1 B6 — 可安装算法增量（2026-11-11）

**基线**：GitHub `GGBOND121382/fund_analysis`，分支 `codex/fund-analysis-v1.0-risk-exits-20261111`，提交 `12e4fdc3cc6a5ffe2a8235db59b886151c605efa`。**本交付是该分支的集成增量，不是全量仓库快照。**

**真实目标**：费后年化严格超过5%；完整组合最大回撤预算4%。4%只能作为研究/场景约束，不能保证未来不突破。缺少足够样本不得声明 `OOS_ACCEPTED`。本增量无交易所/基金平台交易功能。

## 交付内容

- `fundkit/v11_cash.py`：真实外部入金 `F_t` 与当日申购 `a_t` 分离；延期现金**只是可用现金的子账**，不增加资产；实际成交回执 `BUY` 才扣账；不同日交易回执必须逐笔对齐。
- `fundkit/v11_signals.py`：已公开的黄金/权益126/252日趋势；债券不能依赖净值趋势追加；联合移动块Bootstrap情景风险门；缺数据拒绝战术加仓。
- `fundkit/v11_forecast.py`：使用当时公开净值构造的同样本保守收益代理。没有据此获得95%覆盖率统计保证、也不等于严格未来收益预测。
- `fundkit/v11_exit.py`：持仓批次级移动止盈激活锁存、未知成本/锁定/FIFO拦截；在有风险模型时比较合法退出份额候选；否则保留原B5退出量，不虚构最优。
- `fundkit/v11_policy.py`：新增B6策略；B0—B5保持不变；动态投入、选择性延期释放、真实赎回风险门、止盈止损分批处理。
- `fundkit/v11_ablation.py`：A1—A6六个预注册消融候选，增加到现有联合策略网格；原B0由原项目评估代码承担A0基线。
- `scripts/install_v11_overlay.py`：向完整 v1.0.1 本地仓库安装B6及其连接点：`policies`、`simulator`、`joint_optimizer`、配置。重复运行幂等；遇到基线锚点变化拒绝修改。
- `tests/test_v11.py`：新增19个数学/审计/集成接口测试。

## 安装（仅在 v1.0.1 分支上操作）

先确保工作目录中的分支和提交对应上面版本，并创建新工作分支，不要直接修改 main：

```powershell
# 在自己的 fund_analysis 本地仓库目录
# git checkout codex/fund-analysis-v1.0-risk-exits-20261111
# git checkout -b codex/fund-analysis-v1.1-dynamic-b6

# 将 ZIP 解压到本地某个目录，例如 .\\v11_overlay
python .\\v11_overlay\\scripts\\install_v11_overlay.py --repo .
python -m unittest discover -s tests -p 'test_*.py' -q
python .\\v11_overlay\\tests\\test_v11.py
```

首次执行脚本复制6个新增模块，并将B6接入三个原文件。使用 Git 检查变更：

```powershell
git status --short
git diff --check
git diff -- fundkit/policies.py fundkit/simulator.py fundkit/joint_optimizer.py config/policy_params.json
```

## 测试/运行入口

现有 `main.py sim --policy B6` 使用原模拟器运行B6，但**必须先提供研究用途、符合时序的 `target_weights`**。公开安全模板的 `target_weights` 是 `{"cash":1.0}`，因此默认不会自动购买基金。不要为了绕过不合格报告，篡改权重报告验证状态。

```powershell
python main.py sim --policy B6 --start 2026-11-05 --end 2026-11-11
```

本地净值缺失时原模拟器会返回 `DATA_BLOCKED`；不能将其视为回测完成。若要运行联合优化：

```powershell
python main.py optimize-strategy --asof 2026-11-11
```

该命令通过原 `joint_optimizer` 的注册表纳入A1—A6，但目前13基金共同历史约507日、默认最低要求1008日，依旧会返回 `DATA_BLOCKED`。不能缩短测试窗口或复制日期来骗过校验。

## B6 参数说明

在`config/policy_params.json`配置：

| 参数 | 默认 | 语义 |
|--|--:|--|
| `base_fraction` | 1.0 | 当日外部入金用于基础定投的最大份额 |
| `release_fraction` | 0.50 | 信号及联合风险门均通过时，允许释放先前延期资金的比例 |
| `deferred_horizon_days` | 10 | 延期现金等待天数（交易日），超期仅报告，不会冒险强制买入 |
| `cash_reserve_weight` | .05 | 必须保留的账户现金比例（研究参数，非最优） |
| `entry_timing_enabled` | True | 仅黄金及权益指数允许已公布趋势择机追加；还需保守收益代理与4%风险门通过 |
| `exit_mode` | both | none / take_profit_only / stop_only / both |
| `optimize_exit_fraction` | True | 风险已知时评估25%、50%、100%可赎份额候选 |
| `simulate_redeem` | True | **仅研究模拟**执行假设赎回并计费，绝不连接实盘交易 |
| `risk_paths` | 32 | 每种20/60日块长的场景数；不足以保证统计精度 |

## 仍未完成的边界

1. 远程仓库完整代码没有在本容器被克隆；新增模块和补丁安装器通过19项测试，**没有在真实 v1.0.1 全仓库运行原有测试**。用户本地安装后必须执行全量回归测试。
2. 收益预测仍使用历史净值的Bootstrap代理；债券Carry/信用利差/久期因子与真实交易权限数据的接入尚未完成。债券默认禁用额外趋势加仓。
3. 批次止盈锁存状态目前在模拟进程内有效，`policy.snapshot()`提供导出；生产级重启持久化、历史披露时间精确确认和异常重放尚未实现。因此不应用于实盘账户自动决策。
4. A1—A6消融规则已加入研究网格，但联合实验因历史不足尚未完成，也未证明收益>5%或MDD<4%。
5. `risk_gate`只复用历史与联合移动块Bootstrap，**未纳入可核实信用利差和利率冲击的完整压力测试**；战术加仓结论严格限定为研究，不得作为4%未来硬风控保证。
6. 固定 `target_weights` 的更新仍使用 v1.0.1 原优化器；尚未实现逐21个交易日自动滚动更新战略权重。

**审计结论：v1.1核心 B6 资金/择时/退出集成增量已实现并通过自有接口测试；v1.1全量的实证与生产级实现还没有闭环。**
