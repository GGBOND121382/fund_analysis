#!/usr/bin/env python3
"""场外基金日K线抓取与区间指标排序 CLI。

用法示例：
  python main.py list                                  # 更新基金列表
  python main.py fetch --limit 200 --workers 8         # 抓取日净值（先小批量试跑）
  python main.py fetch                                 # 全量抓取
  python main.py rank --start 2025-01-01 --metric sharpe --top 30
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from fundkit import fetcher, metrics, storage


def cmd_list(_args):
    df = fetcher.fetch_fund_list()
    storage.save_fund_list(df)
    print(f"基金列表已更新: {len(df)} 只 -> {storage.FUND_LIST_FILE}")
    print(df["type"].value_counts().head(15).to_string())

    status = fetcher.fetch_trade_status()
    storage.save_trade_status(status)
    tradable = fetcher.tradable_codes(status)
    print(f"\n交易状态快照已更新: {len(status)} 只 -> {storage.FUND_STATUS_FILE}")
    print(f"当前可交易（开放/限大额申购 且 开放赎回）: {len(tradable)} 只")
    print(status["sub_status"].value_counts().to_string())

    print("\n提示: 分红复权数据由 `python main.py dividends` 单独抓取维护")


def cmd_fetch(args):
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    invalid = [s for s in sources if s not in fetcher.SOURCE_FUNCS]
    if invalid:
        sys.exit(f"未知数据源 {invalid}，可选: {', '.join(fetcher.SOURCE_FUNCS)}")

    fund_list = storage.load_fund_list()
    if not args.include_money:
        n0 = len(fund_list)
        fund_list = fund_list[~fund_list["type"].astype(str).str.contains("货币型")]
        print(f"已排除货币型基金 {n0 - len(fund_list)} 只（其净值字段为万份收益口径，--include-money 可关闭）")
    if args.types:
        keys = [t.strip() for t in args.types.split(",") if t.strip()]
        mask = fund_list["type"].astype(str).apply(lambda s: any(k in s for k in keys))
        fund_list = fund_list[mask]
        print(f"按类型 {keys} 过滤后: {len(fund_list)} 只")

    codes = fund_list["code"].tolist()
    todo = [c for c in codes if not storage.is_fresh(c, args.refresh_days)]
    print(f"共 {len(codes)} 只，需更新 {len(todo)} 只（--refresh-days {args.refresh_days}）")

    if args.limit:
        todo = todo[: args.limit]
        print(f"本次仅抓取前 {len(todo)} 只（--limit）")

    results, failed, used = fetcher.fetch_nav_batch(
        todo, sources=sources, workers=args.workers)
    for code, (df, _src) in results.items():
        if storage.has_nav(code):
            df = fetcher.merge_nav(storage.load_nav(code), df)
        storage.save_nav(code, df)
    print(f"完成：成功 {len(results)}，失败 {len(failed)}，数据源使用 {used}，目录 {storage.NAV_DIR}")
    if failed:
        sample = list(failed.items())[:10]
        for code, err in sample:
            print(f"  失败 {code}: {err}")
        if len(failed) > len(sample):
            print(f"  ... 其余 {len(failed) - len(sample)} 条失败略")


def cmd_dividends(args):
    import akshare as ak

    codes = storage.local_codes()
    if not codes:
        sys.exit("本地无净值数据，请先运行: python main.py fetch --limit N")

    # 用天天基金批量快照的"累计净值 > 单位净值"筛出分过红的基金，大幅减少请求数
    print("获取净值快照以筛选分红基金...")
    snap = ak.fund_open_fund_daily_em()
    nav_col = next(c for c in snap.columns if c.endswith("单位净值"))
    acc_col = next(c for c in snap.columns if c.endswith("累计净值"))
    snap["code"] = snap["基金代码"].astype(str).str.zfill(6)
    snap_codes = set(snap["code"])
    payers = set(snap.loc[pd.to_numeric(snap[acc_col], errors="coerce")
                          > pd.to_numeric(snap[nav_col], errors="coerce"), "code"])
    targets = [c for c in codes if c in payers or c not in snap_codes]
    print(f"本地 {len(codes)} 只，疑似/无法确认分红 {len(targets)} 只，开始逐只抓取分红历史")

    div_map = fetcher.fetch_dividends_batch(targets, workers=args.workers)
    storage.save_dividends(div_map)
    print(f"完成: {len(div_map)} 只基金有分红记录 -> {storage.DIVIDENDS_FILE}")


def _rank_one(task):
    """单只基金的区间指标计算（供多进程调用，必须是模块级函数）。"""
    code, start, end, rf, min_days, max_avg_gap, divs, splits = task
    try:
        df = storage.load_nav(code)
    except Exception:  # noqa: BLE001 - 单个文件损坏不影响整体
        return None
    df = metrics.slice_by_date(df, start, end)
    df = metrics.adjust_nav(df, divs, splits)  # divs/splits 为空时原样返回（未复权）
    m = metrics.compute_metrics(df, rf=rf, min_days=min_days)
    if m is None:
        return None
    if max_avg_gap and m["avg_gap"] == m["avg_gap"] and m["avg_gap"] > max_avg_gap:
        return None  # 净值披露间隔过大（如定开基金周频披露），指标失真
    m["code"] = code
    return m


def cmd_rank(args):
    if args.metric not in metrics.METRICS:
        sys.exit(f"未知指标 {args.metric}，可选: {', '.join(metrics.METRICS)}")

    codes = storage.local_codes()
    if not codes:
        sys.exit("本地无净值数据，请先运行: python main.py fetch --limit N")

    fund_list = None
    try:
        fund_list = storage.load_fund_list()
    except FileNotFoundError:
        if args.types or not args.include_money:
            sys.exit("类型过滤/货币基金排除需要基金列表，请先运行: python main.py list")
    if fund_list is not None and not args.include_money:
        money = set(fund_list.loc[fund_list["type"].astype(str).str.contains("货币型"), "code"])
        codes = [c for c in codes if c not in money]
    if args.types and fund_list is not None:
        keys = [t.strip() for t in args.types.split(",") if t.strip()]
        mask = fund_list["type"].astype(str).apply(lambda s: any(k in s for k in keys))
        allow = set(fund_list.loc[mask, "code"])
        codes = [c for c in codes if c in allow]
    if not args.include_closed:
        try:
            status = storage.load_trade_status()
        except FileNotFoundError:
            sys.exit("交易状态过滤需要状态快照，请先运行: python main.py list（--include-closed 可关闭该过滤）")
        n0 = len(codes)
        allow = fetcher.tradable_codes(status, min_daily_limit=args.min_daily_limit)
        codes = [c for c in codes if c in allow]
        print(f"已过滤不可交易基金 {n0 - len(codes)} 只，剩余 {len(codes)} 只"
              f"（限大额阈值 {args.min_daily_limit:,.0f} 元/日，--include-closed 可关闭）")

    div_map, split_map = {}, {}
    if not args.no_adjust:
        div_map = storage.load_dividends()
        split_map = storage.load_splits()
        if not div_map:
            print("警告: 无分红数据（python main.py dividends 可生成），本次按未复权计算")
    else:
        print("已关闭分红复权（--no-adjust）")

    tasks = [(c, args.start, args.end, args.rf, args.min_days, args.max_avg_gap,
              div_map.get(c, ()), split_map.get(c, ())) for c in codes]
    workers = min(args.rank_workers, os.cpu_count() or 1)
    if workers > 1 and len(tasks) > 200:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [m for m in pool.map(_rank_one, tasks, chunksize=64) if m is not None]
    else:
        rows = [m for m in map(_rank_one, tasks) if m is not None]

    if not rows:
        sys.exit("区间内没有满足条件的样本（检查 --start/--end/--min-days）")

    result = pd.DataFrame(rows)
    if args.min_ann_return:
        n0 = len(result)
        result = result[result["ann_return"] >= args.min_ann_return]
        print(f"年化收益 >= {args.min_ann_return:.1%} 过滤: {n0} -> {len(result)} 只")
        if result.empty:
            sys.exit("过滤后无样本，放宽 --min-ann-return 试试")
    if args.min_sharpe:
        n0 = len(result)
        result = result[result["sharpe"] >= args.min_sharpe]
        print(f"Sharpe >= {args.min_sharpe:g} 过滤: {n0} -> {len(result)} 只")
        if result.empty:
            sys.exit("过滤后无样本，放宽 --min-sharpe 试试")
    if args.max_drawdown:
        n0 = len(result)
        result = result[result["max_drawdown"] <= args.max_drawdown]
        print(f"最大回撤 <= {args.max_drawdown:.1%} 过滤: {n0} -> {len(result)} 只")
        if result.empty:
            sys.exit("过滤后无样本，放宽 --max-drawdown 试试")
    if args.max_daily_return:
        n0 = len(result)
        result = result[result["max_abs_daily"] <= args.max_daily_return]
        print(f"单日涨跌幅 <= {args.max_daily_return:.1%} 过滤: {n0} -> {len(result)} 只")
        if result.empty:
            sys.exit("过滤后无样本，放宽 --max-daily-return 试试")
    if fund_list is not None and args.with_name:
        names = fund_list.set_index("code")["name"]
        result.insert(1, "name", result["code"].map(names))

    ascending = (args.asc) or (args.metric in metrics.ASCENDING_METRICS and not args.desc)
    result = result.sort_values(args.metric, ascending=ascending, na_position="last")

    show_cols = ["code"] + (["name"] if "name" in result.columns else []) + [
        args.metric, "total_return", "ann_return", "max_drawdown", "ann_vol", "days", "avg_gap", "start", "end",
    ]
    show_cols = list(dict.fromkeys(c for c in show_cols if c in result.columns))

    if args.csv:
        result.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"完整结果已写入 {args.csv}（共 {len(result)} 行）")

    top = result.head(args.top)
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}",
                           "display.max_colwidth", 24, "display.width", 200):
        print(f"\n区间 [{args.start or '最早'} ~ {args.end or '最新'}] 按 {args.metric}"
              f"（{metrics.METRICS[args.metric]}）排序，Top {len(top)}：")
        print(top[show_cols].to_string(index=False))


def cmd_audit(args):
    from fundkit import audit

    report = audit.run_audit(today=args.today)
    d = report["decision"]
    print("五表状态:")
    for k, v in report["tables"].items():
        print(f"  {k}: {v['status']}")
    chk = report["asof_selfcheck"]
    print(f"as-of 自检通过（决策时点 {chk['decision_at']}，最迟净值 {chk['max_nav_date']}）")
    print(f"结论: {d['recommendation_status']} - {d['message']}")
    print("报告: data/quality/step1_audit.{json,md}")


def cmd_sim(args):
    from fundkit import policies, simulator
    from fundkit.adapters import LocalDataAdapter, load_account_snapshot

    snap = load_account_snapshot()
    codes = [f["code"] for f in snap["funds"]]
    equity = frozenset(f["code"] for f in snap["funds"] if f["category"] == "equity_index")
    limits = {}
    try:
        status = storage.load_trade_status()
        lim = status.set_index("code")["daily_limit"]
        limits = {c: float(lim[c]) for c in codes
                  if c in lim.index and pd.notna(lim[c]) and 0 < lim[c] < 1e9}
    except FileNotFoundError:
        print("警告: 无交易状态快照，申购限额不生效（python main.py list 可生成）")

    adapter = LocalDataAdapter()
    try:
        lots, mode = simulator.lots_from_snapshot(adapter)
    except ValueError as exc:
        print(f"{exc}；已停止模拟，不允许用空批次生成绩效")
        return
    params = policies.load_params()
    if args.weights_report:
        from fundkit.weight_optimizer import load_frozen_research_weights
        try:
            frozen = load_frozen_research_weights(args.weights_report,
                                                  args.start or snap["snapshot_date"], codes)
            if args.policy in ("B0", "B1"):
                raise ValueError('只有B2—B5会使用优化目标权重，B0/B1基线不得覆盖')
            params["target_weights"] = frozen
            print("已加载 as-of 冻结研究权重；仅用于历史模拟，不可自动交易")
        except ValueError as exc:
            print(f"DATA_BLOCKED: {exc}")
            return
    params["daily_limits"] = limits
    params["equity_index_codes"] = sorted(equity)
    config = simulator.SimConfig(
        budget_per_day=snap["new_cash_budget_cny_per_execution_day"],
        allow_new_equity_index=args.allow_new_equity or args.policy != "B1",
        equity_index_codes=equity, daily_limits=limits)
    sim = simulator.Simulator(codes, config, adapter)

    start = args.start or snap["snapshot_date"]
    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    policy = policies.build_policy(args.policy, params)
    try:
        result = sim.run(start, end, initial_lots=lots, policy=policy)
    except (FileNotFoundError, ValueError) as exc:
        print(f"DATA_BLOCKED: {exc}；必须补齐原始净值，不能产生模拟绩效")
        return

    m = result["metrics"]
    print(f"\n策略 {args.policy} 区间 [{start} ~ {end}]（{m['days']} 个披露日，批次口径 {mode}）：")
    print(f"  期末市值      {result['daily']['value'].iloc[-1]:,.2f} 元")
    print(f"  TWR区间收益   {m['total_return_twr']:.4%}")
    print(f"  年化(CAGR)    {m['cagr']:.4%}")
    print(f"  最大回撤      {m['max_drawdown']:.4%}")
    print(f"  CDaR95        {m['cdar95']:.4%}")
    print(f"  Sharpe        {m['sharpe'] if m['sharpe'] is not None else float('nan'):.4f}")
    print(f"  XIRR          {m['xirr'] if m['xirr'] is not None else float('nan'):.4%}")
    os.makedirs("reports", exist_ok=True)
    tag = f"{args.policy}_{start}_{end}".replace("-", "")
    result["daily"].to_csv(f"reports/sim_{tag}_daily.csv", index=False)
    if not result["events"].empty:
        result["events"].to_csv(f"reports/sim_{tag}_events.csv", index=False)
    print(f"  明细: reports/sim_{tag}_daily.csv / _events.csv")
    print("  状态: RESEARCH_ONLY（模拟批次口径，ACCOUNT_DRAWDOWN_UNVERIFIABLE）")


def cmd_extra(args):
    from fundkit import collect_extra
    from fundkit.adapters import load_account_snapshot

    codes = args.codes.split(",") if args.codes else [f["code"] for f in load_account_snapshot()["funds"]]
    if args.part in ("all", "disclosures"):
        print("采集季报持仓披露（资产配置+债券持仓）...")
        summary = collect_extra.build_disclosures(codes)
        for c, s in summary.items():
            print(f"  {c}: {s}")
    if args.part in ("all", "factors"):
        print("采集市场风险因子（国债曲线/沪深300/黄金）...")
        path = collect_extra.build_risk_factors(start=args.start)
        print(f"  -> {path}")


def cmd_style(args):
    from fundkit import style

    codes = args.codes.split(",") if args.codes else ["710301", "004400", "021559", "014846"]
    report = style.run_style_report(codes)
    for c, r in report.items():
        print(f"\n{c}: 当前风格 {r['current_style_state']}（{r['style_state_reason']}）")
        print(f"  迁移证据: {r['combined_shift_evidence']['state']} - {r['combined_shift_evidence']['reason']}")
        for s in r["disclosure_shifts"]:
            print(f"  披露断点: {s['confirmed_at']} {s['evidence']}")
        if r["latest_betas"]:
            bs = " ".join(f"{k}={v:.4f}" for k, v in r["latest_betas"].items())
            print(f"  最新beta: {bs}  R²={r['latest_r2']:.3f}  cond={r['regression_cond_latest']:.0f}")
        if r["factor_unidentified"]:
            print("  回归不可辨识: FACTOR_UNIDENTIFIED（有效样本不足）")
        for view, st in r["risk_views"].items():
            if isinstance(st, dict) and st.get("mdd") is not None:
                print(f"  {view}: MDD={st['mdd']:.2%} CDaR95={st['cdar95']:.2%} (n={st['n']})")
    print("\n报告: reports/style_report.json  状态: RESEARCH_ONLY")


def cmd_validate(args):
    from fundkit import evaluation

    print("滚动样本外验证（train→val 选参→test 冻结复跑；全部试验记录）...")
    summary = evaluation.run_validation()
    print(f"\n可行性判定: {summary['decision']}")
    if summary.get("status") == "DATA_BLOCKED":
        print("DATA_BLOCKED：" + str(summary.get("reason") or summary.get("note") or "数据前置条件未满足"))
        if summary.get("missing_nav_codes"):
            print("缺少基金净值：" + ", ".join(summary["missing_nav_codes"]))
        print("没有执行有效回测；旧版报告不得作为当前算法的验证结果。")
        return
    for fam, v in summary["feasibility"].items():
        rows = v.get("primary_12m_windows", []) + v.get("diagnostic_overlapping_windows", [])
        folds_text = "  ".join(
            f"fold{f['fold']}[{f.get('test_role','primary')}]: CAGR {f['cagr']:.2%} MDD {f['max_drawdown']:.2%}"
            for f in rows)
        ok = "✓" if v.get("single_primary_12m_window_meets_targets", False) else "✗"
        print(f"  {ok} {fam}: {folds_text}")
    print("\n帕累托前沿（test 均值）:")
    for p in summary["pareto_front_test_mean"]:
        print(f"  {p['family']}: CAGR {p['cagr']:.2%} / MDD {p['max_drawdown']:.2%}")
    boot = summary["stress"]["bootstrap"]
    if "p_mdd_exceed_4pct" in boot:
        ci = boot["ci95"]
        print(f"\n压力测试（块自助 {boot['n_paths']} 条×{boot['horizon_days']}日）: "
              f"P(MDD>4%)={boot['p_mdd_exceed_4pct']:.1%} [{ci[0]:.1%},{ci[1]:.1%}]，"
              f"MDD中位 {boot['mdd_median']:.2%}，P95 {boot['mdd_p95']:.2%}")
    crash = summary["stress"]["bond_crash_2022"]
    if crash.get("portfolio_return") is None:
        print("2022-11 债灾回放：DATA_BLOCKED（缺失历史序列，不补零）")
    else:
        print(f"2022-11 债灾回放（当前权重）: 组合 {crash['portfolio_return']:.2%}")
    print("\n明细: reports/oos_trials.csv, reports/oos_summary.json  状态: RESEARCH_ONLY")


def cmd_optimize_weights(args):
    """仅研究优化：复用本地AKShare缓存，不写回账户目标权重或下单。"""
    import json
    from dataclasses import asdict
    from fundkit.adapters import LocalDataAdapter, CST, load_account_snapshot
    from fundkit.weight_optimizer import (WeightSearchConfig, build_returns_asof,
                                          optimize_target_weights)

    snap = load_account_snapshot()
    all_codes = [f["code"] for f in snap["funds"]]
    codes = args.codes.split(",") if args.codes else all_codes
    omitted_held = [f["code"] for f in snap["funds"]
                    if f["code"] not in codes and float(f["value_cny"]) > 0]
    if args.asof:
        until = pd.Timestamp(args.asof)
        if until.tzinfo is None:
            until = until.tz_localize(CST)
        if len(args.asof.strip()) == 10:  # 仅日期默认按15:00的决策时点截断
            until += pd.Timedelta(hours=15)
    else:
        until = pd.Timestamp.now(tz=CST)
    cfg = WeightSearchConfig(candidate_count=args.candidates,
                             bootstrap_paths=args.paths,
                             lookback_days=args.lookback,
                             min_cash_weight=args.min_cash,
                             max_single_fund_weight=args.max_fund_weight,
                             annual_cash_return=args.cash_return,
                             annual_risk_free_rate=args.risk_free_rate,
                             turnover_cost_rate=args.turnover_cost,
                             seed=args.seed)
    total = sum(float(f["value_cny"]) for f in snap["funds"])
    current = {f["code"]: float(f["value_cny"]) / total for f in snap["funds"] if f["code"] in codes}
    current["cash"] = 1 - sum(current.values())
    try:
        if omitted_held:
            raise ValueError("DATA_BLOCKED: 组合中持有却未纳入的基金: " + ",".join(omitted_held))
        if pd.Timestamp(until).tz_localize(None) < pd.Timestamp(snap['snapshot_date']):
            raise ValueError('DATA_BLOCKED: 不能用当前持仓快照拟合早于快照的历史权重；请通过Python接口注入当时账户状态')
        R = build_returns_asof(LocalDataAdapter(), codes, until, cfg)
        if args.regime_start:
            R = R[R.index >= pd.Timestamp(args.regime_start)]
        categories = {str(f["code"]): str(f.get("category", "other")) for f in snap["funds"]}
        caps = {str(f["code"]): args.max_fund_weight for f in snap["funds"]}
        out = optimize_target_weights(R, current_weights=current,
                                      caps=caps, cfg=cfg, categories=categories)
    except ValueError as exc:
        out = {"status": "DATA_BLOCKED", "reason": str(exc), "trade_ready": False}
    out["config"] = asdict(cfg)
    out["asof"] = until.isoformat()
    out["regime_start"] = args.regime_start
    out["data_quality"] = "PUBLISHED_AT_RULE_INFERRED; DIVIDEND_SOURCE_REQUIRES_AUDIT"
    out["account_history"] = "SIMULATED_LOTS_NO_REAL_ACCOUNT_MDD"
    out["portfolio_scope"] = "ALL_CURRENT_HOLDINGS_REQUIRED; EXISTING_INDEX_EXPOSURE_FROZEN"
    out["style_regime_verification"] = "USER_PROVIDED_CUTOFF_NOT_AUTOMATICALLY_VERIFIED" if args.regime_start else "MISSING"
    os.makedirs("reports", exist_ok=True)
    path = "reports/weight_optimization_research.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print(f"目标权重研究状态：{out['status']}；输出 {path}")
    if "weights" in out:
        for c, w in out["weights"].items():
            print(f"  {c}: {w:.2%}")
        print(f"场景MDD P95={out['scenario_mdd_p95']:.2%}；场景一年收益Q25={out['scenario_return_q25']:.2%}")
        print("重要：权重仅研究候选；未进行独立OOS、未校验真实费用/持仓批次，不能转为实盘申购金额")



def cmd_optimize_strategy(args):
    """外层联合搜索：目标权重 + 再平衡/止盈/止损/恢复规则。

    不采集网络数据，只复用现有本地净值缓存。历史起点为标准化目标组合，
    不是用户真实账户流水；所有结果 RESEARCH_ONLY。
    """
    import json
    from dataclasses import asdict
    from fundkit import joint_optimizer, policies, simulator, style
    from fundkit.adapters import CST, LocalDataAdapter, load_account_snapshot
    from fundkit.weight_optimizer import (WeightSearchConfig,
                                          build_returns_asof,
                                          generate_weight_candidates)

    def write_blocked(reason):
        os.makedirs("reports", exist_ok=True)
        payload = {
            "model_version": ("joint-strategy-v1.1-exploratory"
                              if getattr(args, "exploratory", False) else
                              "joint-strategy-v1.0.1"),
            "asof": getattr(args, "asof", None),
            "objective": {"annualized_net_return_strictly_gt": 0.05,
                          "portfolio_mdd_lte": 0.04, "priority": "drawdown_first"},
            "status": "DATA_BLOCKED", "reason": str(reason),
            "trade_ready": False, "research_only": True,
            "limits": ["未运行有效回测；不得复用旧版报告作为本次结果"],
        }
        with open("reports/joint_strategy_optimization.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        print(f"DATA_BLOCKED: {reason}；已写入 reports/joint_strategy_optimization.json")

    exploratory = bool(getattr(args, "exploratory", False))
    strategy_scope = getattr(args, "strategy_scope", "all")
    if exploratory:
        # Explicit partial-year research protocol; not full annual OOS validation.
        if strategy_scope != "b6-ablation":
            write_blocked("EXPLORATORY_SCOPE_REQUIRED: choose b6-ablation")
            return
        if args.train_days < 252 or args.validation_days < 63 or args.test_days < 63:
            write_blocked("EXPLORATORY_MINIMUM: train >=252, validation/test >=63 sessions")
            return
        if getattr(args, "embargo_days", 0) < 3:
            write_blocked("EXPLORATORY_EMBARGO_REQUIRED: T+2 inferred publication requires 3 trading-day gaps")
            return
    elif min(args.train_days, args.validation_days, args.test_days) < 252:
        write_blocked("INSUFFICIENT_OOS_HORIZON: full validation requires >=252 in all three splits")
        return
    snap = load_account_snapshot()
    all_funds = snap["funds"]
    all_codes = [str(f["code"]) for f in all_funds]
    equity_codes = {str(f["code"]) for f in all_funds
                    if f.get("category") == "equity_index"}
    # 沪深300、标普500、纳斯达克等指数基金属于用户明确纳入的资产类别；
    # 不买个股不等于排除指数基金。默认全部基金代码参与组合优化。
    investable_codes = list(all_codes)
    bond_codes = [str(f["code"]) for f in all_funds
                  if f.get("category") == "bond"]
    adapter = LocalDataAdapter()

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now(tz=CST)
    if asof.tzinfo is None:
        asof = asof.tz_localize(CST)
    if len(str(args.asof or "").strip()) == 10:
        asof += pd.Timedelta(hours=23, minutes=59)

    # 用全部持仓代码构造共同交易日期，缺失任一当前持仓即阻断。
    all_cfg = WeightSearchConfig(
        target_annual_return=0.05, max_drawdown=0.04,
        lookback_days=10000, horizon=252, bootstrap_paths=32,
        block_lengths=(20, 60), historical_windows=8,
        candidate_count=0, min_observations=252,
        max_single_fund_weight=args.max_fund_weight,
        min_cash_weight=args.min_cash, annual_risk_free_rate=args.risk_free_rate,
        seed=args.seed)
    try:
        returns_all = build_returns_asof(adapter, all_codes, asof, all_cfg)
    except Exception as exc:  # noqa: BLE001
        write_blocked(f"无法建立完整账户共同收益矩阵：{exc}")
        return

    try:
        folds = joint_optimizer.build_rolling_folds(
            returns_all.index, train_days=args.train_days,
            validation_days=args.validation_days, test_days=args.test_days,
            step_days=args.step_days, folds=args.folds,
            embargo_days=getattr(args, "embargo_days", 0))
    except ValueError as exc:
        write_blocked(str(exc))
        return

    # 研究范围包含用户当前选定的指数基金类别；不涉及直接个股交易。
    # 回测起点为标准化反事实目标组合，不代表用户当前存量已实际迁移。
    caps = {c: args.max_fund_weight for c in investable_codes}
    current_for_search = {c: 0.0 for c in all_codes}
    current_for_search["cash"] = 1.0  # 避免把2026快照权重倒灌到历史训练期

    # 交易限额来自本地快照；若缺失只允许研究模拟并在报告中标注。
    daily_limits = {}
    trade_limit_quality = "MISSING"
    try:
        st = storage.load_trade_status()
        if len(st) and {"code", "daily_limit"}.issubset(st.columns):
            for _, row in st.iterrows():
                if pd.notna(row["daily_limit"]) and float(row["daily_limit"]) > 0:
                    daily_limits[str(row["code"])] = float(row["daily_limit"])
            trade_limit_quality = "CURRENT_SNAPSHOT_ONLY"
    except Exception:  # noqa: BLE001
        pass

    def make_style_timeline(window):
        start, end = map(pd.Timestamp, window)
        anchors = list(pd.bdate_range(start, end, freq="B"))[::21]
        if not anchors or anchors[0].date() != start.date():
            anchors.insert(0, start)
        timeline = {}
        usable, total = 0, 0
        for anchor in anchors:
            until = pd.Timestamp(anchor).tz_localize(CST) + pd.Timedelta(hours=15)
            day_signals = {}
            for code in bond_codes:
                total += 1
                try:
                    result = style.analyze_fund(adapter, code, until)
                    views = result.get("risk_views", {})
                    stress = views.get("CURRENT_STYLE_REWEIGHTED") or {}
                    if stress.get("cdar95") is not None:
                        usable += 1
                    day_signals[code] = {
                        "combined_shift_evidence": result.get("combined_shift_evidence", {}),
                        "risk_views": views,
                        "current_style_state": result.get("current_style_state"),
                    }
                except Exception as exc:  # noqa: BLE001
                    day_signals[code] = {
                        "combined_shift_evidence": {"state": "STYLE_UNCERTAIN",
                                                     "reason": str(exc)[:120]},
                        "risk_views": {},
                    }
            timeline[pd.Timestamp(anchor).strftime("%Y-%m-%d")] = day_signals
        quality = {"usable_current_style_stress_observations": usable,
                   "total_fund_anchor_observations": total,
                   "usable_fraction": usable / total if total else 0.0,
                   "style_signal_update_every_business_days": 21}
        return timeline, quality

    def make_initial_target_lots(weights, window_start, capital, fee_rate):
        """用候选权重建立反事实组合；首次申购费在期初净值中扣除。"""
        lots, fee_by_code, gross_by_code = {}, {}, {}
        start_ts = pd.Timestamp(window_start)
        for code in investable_codes:
            weight = float(weights.get(code, 0.0))
            if weight <= 1e-12:
                continue
            visible = adapter.nav([code], start_ts.tz_localize(CST) + pd.Timedelta(hours=15))
            visible = visible[visible["nav_date"] <= start_ts]
            if visible.empty:
                raise ValueError(f"DATA_BLOCKED: {code} 在研究窗口起点没有可用已披露净值")
            nav0 = float(visible.sort_values("nav_date").iloc[-1]["unit_nav"])
            if not (nav0 > 0):
                raise ValueError(f"DATA_BLOCKED: {code} 起点单位净值非法")
            gross = capital * weight
            fee = gross * fee_rate
            net = gross - fee
            # 021559 初始批次的实际申请日未知，simulated=True 会阻止猜测其赎回日期。
            lots[code] = [simulator.Lot(
                code, net / nav0, start_ts, simulated=True, order_date=None,
                cost_basis_cny=gross, cost_basis_known=True,
                basis_source="counterfactual_start",
            )]
            fee_by_code[code] = fee
            gross_by_code[code] = gross
        return (lots, capital * float(weights.get("cash", 0.0)),
                fee_by_code, gross_by_code)

    style_cache = {}
    def evaluator(weights, rule, window, fee_rate):
        family = rule["family"]
        params = policies.load_params()
        params["target_weights"] = dict(weights)
        params["daily_limits"] = daily_limits
        params["equity_index_codes"] = sorted(equity_codes)
        params["asset_categories"] = asset_categories
        for section, values in (rule.get("params") or {}).items():
            params[section] = {**params.get(section, {}), **values}
        if family in ("B3", "B5"):
            # 允许在历史模拟器中执行赎回，以检验退出规则；不代表授权实盘自动赎回。
            if family == "B3":
                params["B3"]["allow_redeem"] = True
            if family == "B5":
                params["B5"]["allow_redeem"] = True
        if family == "B4":
            key = tuple(window)
            if key not in style_cache:
                style_cache[key] = make_style_timeline(window)
            params["_style_signals_by_date"] = style_cache[key][0]
        policy = policies.build_policy(family, params)
        start, end = window
        lots, initial_cash, initial_fee_by_code, initial_buy_by_code = make_initial_target_lots(
            weights, start, float(sum(f["value_cny"] for f in all_funds)), float(fee_rate))
        sim_params = dict(simulator.SIM_DEFAULTS)
        sim_params["subscribe_fee_rate"] = float(fee_rate)
        sim_params["redeem_fee_rate"] = float(fee_rate)
        sim_params["initial_investment_fee_cny"] = sum(initial_fee_by_code.values())
        sim_params["initial_investment_fee_by_code"] = initial_fee_by_code
        sim_params["initial_buy_amount_by_code"] = initial_buy_by_code
        # 统一费率只是敏感性假设；没有真实费率表时不能视为实际净收益。
        config = simulator.SimConfig(
            budget_per_day=float(snap["new_cash_budget_cny_per_execution_day"]),
            allow_new_equity_index=True,
            equity_index_codes=frozenset(equity_codes),
            daily_limits=daily_limits,
            params=sim_params)
        sim_codes = all_codes
        engine = simulator.Simulator(sim_codes, config, adapter)
        result = engine.run(start, end, initial_lots=lots,
                            initial_cash=initial_cash, policy=policy)
        m = result["metrics"]
        events = result["events"]
        daily = result["daily"]
        avg_value = float(daily["value"].mean()) if len(daily) else 0.0
        years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-9)
        buys = sells = fees_total = 0.0
        # B6 策略事件只有 action 列：event 列缺失时用对齐的空串序列兜底，
        # 避免未对齐布尔索引抛 Unalignable boolean Series。
        ev_col = (events["event"] if "event" in events.columns
                  else pd.Series("", index=events.index, dtype=str))
        if not events.empty:
            buy_rows = events[ev_col == "BUY"]
            sell_rows = events[ev_col == "REDEEM"]
            buys = float(pd.to_numeric(buy_rows.get("amount", pd.Series(dtype=float)),
                                       errors="coerce").fillna(0).sum())
            sells = float((pd.to_numeric(sell_rows.get("shares", pd.Series(dtype=float)),
                                          errors="coerce").fillna(0) *
                           pd.to_numeric(sell_rows.get("nav", pd.Series(dtype=float)),
                                         errors="coerce").fillna(0)).sum())
            if "fee_cny" in events:
                fees_total = float(pd.to_numeric(events["fee_cny"], errors="coerce").fillna(0).sum())
        return {
            "cagr": float(m["cagr"]),
            "max_drawdown": float(m["max_drawdown"]),
            "sharpe": None if m["sharpe"] is None else float(m["sharpe"]),
            "sortino": None if m["sortino"] is None else float(m["sortino"]),
            "cdar95": float(m["cdar95"]),
            "xirr": None if m["xirr"] is None else float(m["xirr"]),
            "days": int(m["days"]),
            "redeem_events": int((ev_col == "REDEEM").sum()) if not events.empty else 0,
            "buy_events": int((ev_col == "BUY").sum()) if not events.empty else 0,
            "turnover_ann": ((buys + sells) / avg_value / years if avg_value > 0 else 0.0),
            "fees_total": fees_total,
            "fee_assumption": "UNIFORM_STRESS_PROXY_NOT_ACTUAL_FUND_FEE_TABLE",
            "start_portfolio": "COUNTERFACTUAL_TARGET_WEIGHT_START",
        }

    all_results = []
    asset_categories = {str(f["code"]): str(f.get("category", "unknown")) for f in all_funds}
    rules = joint_optimizer.build_rule_grid(
        asset_categories,include_v13=(strategy_scope=="b6-ablation"))
    if strategy_scope == "b6-ablation":
        rules = [r for r in rules if r["family"] == "B6" or
                 (r["family"] == "B2" and r["rule_id"] == "B2_cashflow_only")]
    # CLI 费率按 bp 输入，转换成小数；例如 0,30 -> 0,0.003。
    fee_rates = tuple(float(x.strip()) / 10000.0
                      for x in args.fee_stress_bps.split(",") if x.strip())
    joint_cfg = joint_optimizer.JointSearchConfig(
        target_annual_return=0.05, max_drawdown=0.04,
        min_validation_days=args.validation_days,
        min_test_days=args.test_days,
        fee_stress_rates=fee_rates, seed=args.seed,
        evaluate_ablation_on_test=(exploratory and strategy_scope == "b6-ablation"))

    def evaluate_fixed_bond_benchmarks(validation_window, test_window):
        """No test-set model selection; predetermined simple bond references."""
        base_rule={"family":"B2","rule_id":"B2_cashflow_only","params":{}}
        definitions=[]
        if "021559" in all_codes:
            definitions.append(("GRINN_100_REFERENCE_ONLY",{"021559":1.0},
                                "violates single-fund 40% cap; research only"))
        if bond_codes:
            definitions.append(("BOND_EQUAL_WEIGHT",
                                {c:1/len(bond_codes) for c in bond_codes},
                                "no equity/gold; cashflow top-ups"))
        other_bonds=[c for c in bond_codes if c!="021559"]
        if "021559" in bond_codes and other_bonds:
            reference={"021559":.4}
            reference.update({c:.6/len(other_bonds) for c in other_bonds})
            definitions.append(("GRINN40_BOND60",reference,
                                "40% Grinn + equal weight other bond funds"))
        out=[]
        for name,nonzero,note in definitions:
            weights={c:float(nonzero.get(c,0.)) for c in all_codes}
            weights["cash"]=max(0.,1.-sum(weights.values()))
            row={"name":name,"weights":weights,"note":note,
                 "role":"FIXED_BENCHMARK_NOT_ELIGIBLE_FOR_TEST_SELECTION"}
            for label,window in (("validation",validation_window),("test",test_window)):
                try:
                    fee_rows=[{"uniform_fee_rate_assumption":float(fee),
                               **evaluator(weights,base_rule,window,float(fee))}
                              for fee in fee_rates]
                    row[label]=joint_optimizer._aggregate_fee_metrics(fee_rows)
                except Exception as exc:
                    row[label]={"status":"BENCHMARK_BLOCKED","reason":str(exc)}
            out.append(row)
        return out

    for fold in folds:
        style_quality = None
        tr_start, tr_end = fold["train"]
        va_start, va_end = fold["validation"]
        te_start, te_end = fold["test"]
        # B4只有在验证期和测试期均存在可计算的当前风格压力CDaR时才进入搜索；
        # 否则剔除，避免把无信号的空操作误当成风格风控策略。
        style_val_key, style_test_key = tuple(fold["validation"]), tuple(fold["test"])
        style_cache[style_val_key] = make_style_timeline(fold["validation"])
        style_cache[style_test_key] = make_style_timeline(fold["test"])
        style_quality = {"validation": style_cache[style_val_key][1],
                         "test": style_cache[style_test_key][1]}
        b4_usable = (style_quality["validation"]["usable_current_style_stress_observations"] > 0 and
                     style_quality["test"]["usable_current_style_stress_observations"] > 0)
        rules_for_fold = rules if b4_usable else [r for r in rules if r["family"] != "B4"]
        # 训练净值必须在验证起点之前已公开；T+2推断披露时滞留出3天缓冲。
        train_cutoff = pd.Timestamp(va_start).tz_localize(CST) + pd.Timedelta(hours=15)
        # 探索模式：embargo(3)+披露滞后(≈2)消耗数日，情景长度与最小观测相应放宽；
        # 正式验证维持 252 不变。
        train_relax = args.embargo_days + 3 if exploratory else 0
        train_cfg = WeightSearchConfig(
            target_annual_return=0.05, max_drawdown=0.04,
            lookback_days=args.train_days,
            horizon=min(252, args.train_days) - train_relax,
            bootstrap_paths=args.weight_paths, block_lengths=(20, 60),
            historical_windows=12, candidate_count=args.weight_pool,
            min_observations=min(252, args.train_days) - train_relax,
            max_single_fund_weight=args.max_fund_weight,
            min_cash_weight=args.min_cash, annual_risk_free_rate=args.risk_free_rate,
            seed=args.seed + fold["fold"])
        try:
            train_asof = build_returns_asof(adapter, all_codes, train_cutoff, train_cfg)
            train_returns = train_asof.loc[(train_asof.index >= pd.Timestamp(tr_start)) &
                                           (train_asof.index <= pd.Timestamp(tr_end))]
            if args.regime_start:
                train_returns = train_returns[train_returns.index >= pd.Timestamp(args.regime_start)]
            # 探索模式：embargo 明确消耗训练窗末尾 N 个交易日，训练样本下限相应放宽
            # （另留 3 日余量，252-3-3=246，仍约等于一个完整年度）；正式验证维持 252 不变。
            min_train_sessions = (min(train_cfg.horizon, args.train_days) - args.embargo_days - 3
                                  if exploratory else
                                  max(252, min(train_cfg.horizon, args.train_days)))
            if len(train_returns) < min_train_sessions:
                raise ValueError(f"DATA_BLOCKED: 训练期可用共同收益只有 {len(train_returns)} 日"
                                 f"（要求 {min_train_sessions}）")
            candidate_pack = generate_weight_candidates(
                train_returns, current_weights=current_for_search,
                caps=caps, fixed_weights={}, cfg=train_cfg,
                top_k=args.weight_candidates, categories=asset_categories)
            if not candidate_pack["candidates"]:
                # A genuine finite-scenario infeasibility is not a missing-data
                # error. Never replace research portfolios with all cash.
                all_results.append({
                    "fold": fold,
                    "status": "NO_NONCASH_RISK_FEASIBLE_CANDIDATE",
                    "validation_status": "NO_NONCASH_RISK_FEASIBLE_CANDIDATE",
                    "reason": "LP frontiers could not provide a distinct noncash "
                              "portfolio satisfying 4% historical/scenario risk",
                    "weight_candidate_generation": {
                        k:v for k,v in candidate_pack.items()
                        if k not in ("candidates","cash_benchmark")},
                    "weight_candidates": [],
                    "trade_ready": False,"research_only":True,
                    "fixed_benchmarks":evaluate_fixed_bond_benchmarks(
                        (va_start,va_end),(te_start,te_end)),
                })
                continue
            result = joint_optimizer.optimize_joint_fold(
                candidate_pack["candidates"], rules_for_fold, evaluator,
                (va_start, va_end), (te_start, te_end), joint_cfg)
            result["fold"] = fold
            result["fixed_benchmarks"] = evaluate_fixed_bond_benchmarks(
                (va_start,va_end),(te_start,te_end))
            result["style_signal_quality"] = style_quality
            result["rules_included"] = [r["rule_id"] for r in rules_for_fold]
            result["weight_candidate_generation"] = {
                k: v for k, v in candidate_pack.items() if k != "candidates"}
            result["weight_candidates"] = candidate_pack["candidates"]
            result["training_data_end"] = candidate_pack["training_data_end"]
            result["training_data_start"] = candidate_pack["training_data_start"]
            all_results.append(result)
        except Exception as exc:  # noqa: BLE001
            import traceback
            all_results.append({"fold": fold, "status": "DATA_BLOCKED",
                                "reason": str(exc),
                                "traceback_tail": traceback.format_exc().splitlines()[-6:],
                                "style_signal_quality": style_quality,
                                "trade_ready": False})

    os.makedirs("reports", exist_ok=True)
    blocked_count = sum(x["status"] == "DATA_BLOCKED" for x in all_results)
    no_portfolio_count = sum(x["status"] == "NO_NONCASH_RISK_FEASIBLE_CANDIDATE"
                             for x in all_results)
    overall_status = ("DATA_BLOCKED" if blocked_count == len(all_results) else
                      "NO_NONCASH_RISK_FEASIBLE_CANDIDATE"
                      if no_portfolio_count == len(all_results) else
                      "PARTIAL_DATA_BLOCKED" if blocked_count else
                      "EXPLORATORY_NOT_LONG_TERM_VALIDATED" if exploratory else "RESEARCH_ONLY")
    payload = {
        "model_version": "joint-strategy-v1.1-exploratory" if exploratory else "joint-strategy-v1.0.1",
        "protocol": "EXP_252_120_120_EMBARGO3_OR_252_63_63_EMBARGO3" if exploratory else "FULL_YEAR_252_252_252",
        "strategy_scope": strategy_scope,
        "evidence_level": "PARTIAL_YEAR_OOS_NOT_LONG_TERM" if exploratory else "FULL_YEAR_PROTOCOL",
        "asof": asof.isoformat(),
        "objective": {"annualized_net_return_strictly_gt": 0.05,
                      "portfolio_mdd_lte": 0.04, "priority": "drawdown_first"},
        "portfolio_scope": "ALL_CURRENT_FUND_ASSET_CLASSES_INCLUDED; INDEX_FUNDS_ARE_INVESTABLE; NO_DIRECT_STOCKS",
        "initialization": "COUNTERFACTUAL_TARGET_WEIGHT_START; NOT_REAL_ACCOUNT_HISTORY",
        "style_signal_update": "ASOF_EVERY_21_BUSINESS_DAYS_FOR_B4",
        "trade_limit_quality": trade_limit_quality,
        "fee_assumptions": list(fee_rates),
        "rules_tested": rules,
        "folds": all_results,
        "trade_ready": False,
        "status": overall_status,
        "limits": [
            "当前基金池中的债券、黄金、沪深300、标普500和纳斯达克指数基金均纳入候选；不涉及直接个股",
            "起点按候选目标权重建立标准化反事实组合，不是用户真实账户历史",
            "首次建仓申购费、后续申购费和赎回费均按统一费率压力情景计入；并非真实基金费率表",
            "021559 初始批次赎回日未知，模拟器按不可赎回处理；后续新增批次按合同近似到期规则处理",
            "B4 风格信号每21个工作日按当时可见信息更新；published_at 仍使用本地T+2规则推断",
            "5%收益/4%回撤为历史样本与场景筛选条件，不构成未来保证",
        ],
    }
    out_path = "reports/joint_strategy_optimization.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    print(f"联合优化研究结果: {out_path}")
    for r in all_results:
        print(f"  fold{r.get('fold', {}).get('fold')}: {r['status']}")
        if r.get("selected"):
            s = r["selected"]
            print(f"    {s['family']} / {s['rule_id']} | 验证 CAGR={s['validation']['cagr_worst_fee']:.2%} "
                  f"MDD={s['validation']['mdd_worst_fee']:.2%} | 测试 CAGR={r['test']['cagr_worst_fee']:.2%} "
                  f"MDD={r['test']['mdd_worst_fee']:.2%}")
    print("状态: RESEARCH_ONLY；不自动交易，不把反事实回测称为真实账户历史")

def cmd_decide(args):
    from fundkit import decision

    d = decision.build_decision(asof=args.asof)
    path = decision.write_decision(d)
    p1, p2 = decision.write_reports(d)
    print(f"决策状态: {d['decision_status']}｜证据: {d['evidence_status']}")
    print(f"每日预算 {d['cash_budget_cny']:.0f} 元，建议留现金 {d['recommended_cash_hold_cny']:.0f} 元")
    for fd in d["fund_decisions"]:
        tw_str = (f"{fd['target_weight']:.0%}"
                  if fd["target_weight"] is not None else "未验证")
        print(f"  {fd['fund_code']} 权重{fd['current_weight']:.1%}→目标"
              f"{tw_str} 申购{fd['subscribe_cny']:.0f} {fd['action']}")
    print(f"\n输出: {path}, reports/fund_decisions.csv, {p1}, {p2}")


def main():
    parser = argparse.ArgumentParser(description="场外基金日K线抓取与区间指标排序")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="更新全部场外基金列表")

    p_audit = sub.add_parser("audit", help="第1步：数据质量与公开时点审核（算法层）")
    p_audit.add_argument("--today", default=None, help="审核基准日 YYYY-MM-DD（默认今天）")

    p_sim = sub.add_parser("sim", help="第3步：组合现金流模拟（RESEARCH_ONLY，模拟批次口径）")
    p_sim.add_argument("--policy", default="B1", help="策略族（当前实现 B0/B1，默认 B1）")
    p_sim.add_argument("--weights-report", default="", help="训练期生成的目标权重报告；必须严格早于模拟开始日")
    p_sim.add_argument("--start", default=None, help="模拟起始日（默认账户快照日）")
    p_sim.add_argument("--end", default=None, help="模拟结束日（默认今天）")
    p_sim.add_argument("--allow-new-equity", action="store_true",
                     help="允许新增股指基金申购（默认禁止）")

    p_extra = sub.add_parser("extra", help="采集季报持仓披露与市场风险因子（第1/2步前置）")
    p_extra.add_argument("--part", default="all", choices=["all", "disclosures", "factors"])
    p_extra.add_argument("--codes", default="", help="逗号分隔基金代码（默认账户快照13只）")
    p_extra.add_argument("--start", default="2019-01-01", help="因子起始日（默认 2019-01-01）")

    p_style = sub.add_parser("style", help="第2步：风格识别与风险估计（RESEARCH_ONLY）")
    p_style.add_argument("--codes", default="", help="逗号分隔基金代码（默认 4 只持仓债基）")

    sub.add_parser("validate", help="第5步：滚动样本外验证与压力测试（RESEARCH_ONLY）")

    p_wo = sub.add_parser("optimize-weights", help="第2.5步：研究用目标权重联合情景优化（不自动交易）")
    p_wo.add_argument("--codes", default="", help="基金代码，默认全部当前持仓；遗漏任何非零持仓则阻断")
    p_wo.add_argument("--asof", default=None, help="数据公开截止日 YYYY-MM-DD")
    p_wo.add_argument("--regime-start", default=None, help="当前持仓风格阶段起点；必须有披露证据")
    p_wo.add_argument("--lookback", type=int, default=252)
    p_wo.add_argument("--candidates", type=int, default=2400)
    p_wo.add_argument("--paths", type=int, default=192)
    p_wo.add_argument("--min-cash", type=float, default=0.0)
    p_wo.add_argument("--max-fund-weight", type=float, default=0.40,
                      help="单只基金治理上限；默认40%%，不是经验收益参数")
    p_wo.add_argument("--cash-return", type=float, default=0.0)
    p_wo.add_argument("--risk-free-rate", type=float, default=0.02,
                      help="年化无风险利率假设，用于Sharpe计算；默认2%%")
    p_wo.add_argument("--turnover-cost", type=float, default=0.003,
                      help="单边换仓成本压力代理（默认30bp；真实费率应逐基金替换）")
    p_wo.add_argument("--seed", type=int, default=42)

    p_js = sub.add_parser("optimize-strategy", help="目标权重+止盈止损+再入场联合优化（RESEARCH_ONLY）")
    p_js.add_argument("--asof", default=None, help="数据公开截止时点")
    p_js.add_argument("--strategy-scope", choices=["all", "b6-ablation"], default="all",
                      help="B2 baseline and registered B6 parameter ablations")
    p_js.add_argument("--exploratory", action="store_true",
                      help="explicit short-history research only: train >=252, val/test >=63")
    p_js.add_argument("--train-days", type=int, default=504)
    p_js.add_argument("--validation-days", type=int, default=252)
    p_js.add_argument("--test-days", type=int, default=252)
    p_js.add_argument("--step-days", type=int, default=252)
    p_js.add_argument("--folds", type=int, default=1)
    p_js.add_argument("--embargo-days", type=int, default=0,
                      help="信息隔离天数，探索性研究应>=3交易日以保护T+2可见性")
    p_js.add_argument("--weight-pool", type=int, default=600)
    p_js.add_argument("--weight-candidates", type=int, default=5)
    p_js.add_argument("--weight-paths", type=int, default=64)
    p_js.add_argument("--min-cash", type=float, default=0.0)
    p_js.add_argument("--max-fund-weight", type=float, default=0.40,
                      help="单只基金治理上限；默认40%%，不是经验收益参数")
    p_js.add_argument("--regime-start", default=None, help="可选：经季报证据确认的当前风格起点")
    p_js.add_argument("--fee-stress-bps", default="30,60",
                       help="统一单边费用压力情景，单位bp；真实基金费率需另行接入")
    p_js.add_argument("--risk-free-rate", type=float, default=0.02,
                       help="年化无风险利率假设，用于训练候选Sharpe；默认2%%")
    p_js.add_argument("--seed", type=int, default=42)

    p_decide = sub.add_parser("decide", help="第6步：输出研究建议（schema 校验，RESEARCH_ONLY）")
    p_decide.add_argument("--asof", default=None, help="决策时点（默认现在）")

    p_div = sub.add_parser("dividends", help="抓取全部分红送配记录（复权数据，建议每周运行）")
    p_div.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")

    p_fetch = sub.add_parser("fetch", help="抓取基金日净值（日K线）")
    p_fetch.add_argument("--limit", type=int, default=0, help="只抓取前 N 只（调试用）")
    p_fetch.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8，过大易被限流）")
    p_fetch.add_argument("--types", default="", help="基金类型过滤，逗号分隔子串，如 股票型,混合型")
    p_fetch.add_argument("--refresh-days", type=int, default=1,
                         help="距上次更新不足 N 天则跳过（默认 1，0 表示强制全量重抓）")
    p_fetch.add_argument("--sources", default="danjuan,em_f10,em_js",
                         help="数据源及优先级，逗号分隔，可选: danjuan,em_f10,em_js")
    p_fetch.add_argument("--include-money", action="store_true",
                         help="包含货币型基金（默认排除，其净值字段为万份收益口径）")

    p_rank = sub.add_parser("rank", help="按时间区间的指标排序")
    p_rank.add_argument("--start", default=None, help="区间起始日 YYYY-MM-DD（默认最早）")
    p_rank.add_argument("--end", default=None, help="区间结束日 YYYY-MM-DD（默认最新）")
    p_rank.add_argument("--metric", default="sharpe",
                        help=f"排序指标，可选: {', '.join(metrics.METRICS)}")
    p_rank.add_argument("--top", type=int, default=30, help="输出前 N 名")
    p_rank.add_argument("--rf", type=float, default=0.02, help="年化无风险利率（默认 0.02）")
    p_rank.add_argument("--min-days", type=int, default=60, help="区间内最少交易日数（默认 60）")
    p_rank.add_argument("--min-ann-return", type=float, default=0,
                        help="年化收益率下限，小数（如 0.05 表示 5%%，默认 0 不过滤）")
    p_rank.add_argument("--min-sharpe", type=float, default=0,
                        help="夏普比率下限（如 7，默认 0 不过滤）")
    p_rank.add_argument("--max-drawdown", type=float, default=0,
                        help="最大回撤上限，小数（如 0.02 表示 2%%，默认 0 不过滤）")
    p_rank.add_argument("--max-daily-return", type=float, default=0,
                        help="单日涨跌幅上限，小数（如 0.03 表示 3%%，过滤巨额赎回等净值异动，默认 0 不过滤）")
    p_rank.add_argument("--types", default="", help="基金类型过滤，逗号分隔子串")
    p_rank.add_argument("--include-money", action="store_true", help="包含货币型基金（默认排除）")
    p_rank.add_argument("--include-closed", action="store_true",
                        help="包含封闭期/暂停申赎等当前不可交易的基金（默认排除）")
    p_rank.add_argument("--min-daily-limit", type=float, default=10000,
                        help="限大额基金的日累计限定金额下限，元（默认 10000，低于此额度视为不可交易）")
    p_rank.add_argument("--no-adjust", action="store_true",
                        help="不做分红复权（默认用后复权口径，除息日分红加回收益）")
    p_rank.add_argument("--max-avg-gap", type=float, default=0,
                        help="净值平均披露间隔天数上限，过滤定开等低频披露基金（默认 0 不过滤）")
    p_rank.add_argument("--rank-workers", type=int, default=os.cpu_count() or 1,
                        help="排序计算进程数（默认 CPU 核数）")
    p_rank.add_argument("--with-name", action="store_true", help="输出中附带基金名称")
    p_rank.add_argument("--asc", action="store_true", help="强制升序")
    p_rank.add_argument("--desc", action="store_true", help="强制降序")
    p_rank.add_argument("--csv", default="", help="将完整排序结果写入 CSV 文件")

    args = parser.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch, "rank": cmd_rank, "dividends": cmd_dividends,
     "audit": cmd_audit, "sim": cmd_sim, "extra": cmd_extra, "style": cmd_style,
     "validate": cmd_validate, "decide": cmd_decide, "optimize-weights": cmd_optimize_weights,
     "optimize-strategy": cmd_optimize_strategy}[args.cmd](args)


if __name__ == "__main__":
    main()