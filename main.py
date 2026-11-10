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
    lots, mode = simulator.lots_from_snapshot(adapter)
    params = policies.load_params()
    params["daily_limits"] = limits
    params["equity_index_codes"] = sorted(equity)
    config = simulator.SimConfig(
        budget_per_day=snap["new_cash_budget_cny_per_execution_day"],
        allow_new_equity_index=args.allow_new_equity or args.policy == "B0",
        equity_index_codes=equity, daily_limits=limits)
    sim = simulator.Simulator(codes, config, adapter)

    start = args.start or snap["snapshot_date"]
    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")
    policy = policies.build_policy(args.policy, params)
    result = sim.run(start, end, initial_lots=lots, policy=policy)

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
    for fam, v in summary["feasibility"].items():
        folds = "  ".join(f"fold{f['fold']}: CAGR {f['cagr']:.2%} MDD {f['max_drawdown']:.2%}"
                          for f in v["folds"])
        ok = "✓" if v["all_folds_cagr_ge_4pct_and_mdd_le_5pct"] else "✗"
        print(f"  {ok} {fam}: {folds}")
    print("\n帕累托前沿（test 均值）:")
    for p in summary["pareto_front_test_mean"]:
        print(f"  {p['family']}: CAGR {p['cagr']:.2%} / MDD {p['max_drawdown']:.2%}")
    boot = summary["stress"]["bootstrap"]
    if "p_mdd_exceed_5pct" in boot:
        ci = boot["ci95"]
        print(f"\n压力测试（块自助 {boot['n_paths']} 条×{boot['horizon_days']}日）: "
              f"P(MDD>5%)={boot['p_mdd_exceed_5pct']:.1%} [{ci[0]:.1%},{ci[1]:.1%}]，"
              f"MDD中位 {boot['mdd_median']:.2%}，P95 {boot['mdd_p95']:.2%}")
    crash = summary["stress"]["bond_crash_2022"]
    print(f"2022-11 债灾回放（当前权重）: 组合 {crash['portfolio_return']:.2%}")
    print("\n明细: reports/oos_trials.csv, reports/oos_summary.json  状态: RESEARCH_ONLY")


def cmd_decide(args):
    from fundkit import decision

    d = decision.build_decision(asof=args.asof)
    path = decision.write_decision(d)
    p1, p2 = decision.write_reports(d)
    print(f"决策状态: {d['decision_status']}｜证据: {d['evidence_status']}")
    print(f"每日预算 {d['cash_budget_cny']:.0f} 元，建议留现金 {d['recommended_cash_hold_cny']:.0f} 元")
    for fd in d["fund_decisions"]:
        print(f"  {fd['fund_code']} 权重{fd['current_weight']:.1%}→目标"
              f"{(fd['target_weight'] or 0):.0%} 申购{fd['subscribe_cny']:.0f} {fd['action']}")
    print(f"\n输出: {path}, reports/fund_decisions.csv, {p1}, {p2}")


def main():
    parser = argparse.ArgumentParser(description="场外基金日K线抓取与区间指标排序")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="更新全部场外基金列表")

    p_audit = sub.add_parser("audit", help="第1步：数据质量与公开时点审核（算法层）")
    p_audit.add_argument("--today", default=None, help="审核基准日 YYYY-MM-DD（默认今天）")

    p_sim = sub.add_parser("sim", help="第3步：组合现金流模拟（RESEARCH_ONLY，模拟批次口径）")
    p_sim.add_argument("--policy", default="B1", help="策略族（当前实现 B0/B1，默认 B1）")
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
                        help="年化收益率下限，小数（如 0.05 表示 5%，默认 0 不过滤）")
    p_rank.add_argument("--min-sharpe", type=float, default=0,
                        help="夏普比率下限（如 7，默认 0 不过滤）")
    p_rank.add_argument("--max-drawdown", type=float, default=0,
                        help="最大回撤上限，小数（如 0.02 表示 2%，默认 0 不过滤）")
    p_rank.add_argument("--max-daily-return", type=float, default=0,
                        help="单日涨跌幅上限，小数（如 0.03 表示 3%，过滤巨额赎回等净值异动，默认 0 不过滤）")
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
     "validate": cmd_validate, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
