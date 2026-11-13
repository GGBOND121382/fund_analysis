#!/usr/bin/env python3
"""Local research runner, never writes account config or submits trades."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundkit import simulator, storage  # noqa: E402
from fundkit.adapters import CST, LocalDataAdapter  # noqa: E402
from fundkit.v14_core_satellite import (CORE, SATELLITES, STRESS, Limits,  # noqa: E402
    combined_drawdown, fixed_weights, make_policy, optimized_weights)
from fundkit.weight_optimizer import WeightSearchConfig, build_returns_asof  # noqa: E402


def trade_constraints(codes, allow_missing=False):
    if not Path(storage.FUND_STATUS_FILE).is_file():
        if allow_missing:
            return {}, set(), set(), "MISSING_UNRESTRICTED_DIAGNOSTIC_OVERRIDE"
        raise ValueError("DATA_BLOCKED: missing data/fund_status.csv; refresh with python main.py list")
    st = storage.load_trade_status()
    if not {"code", "sub_status", "red_status", "daily_limit"}.issubset(st.columns):
        raise ValueError("DATA_BLOCKED: missing trade status columns")
    st = st[st.code.isin(codes)]
    if st.code.duplicated().any():
        raise ValueError("DATA_BLOCKED: duplicate trade status")
    buy_block, sell_block, caps = set(codes), set(codes), {}
    for row in st.itertuples(index=False):
        limit = pd.to_numeric(row.daily_limit, errors="coerce")
        if row.sub_status == "开放申购" or (row.sub_status == "限大额" and
                                         pd.notna(limit) and 0 < limit < math.inf):
            buy_block.discard(row.code)
            if pd.notna(limit) and 0 < limit < math.inf:
                caps[row.code] = float(limit)
        if row.red_status == "开放赎回":
            sell_block.discard(row.code)
    return caps, buy_block, sell_block, "CURRENT_SNAPSHOT_ONLY_NOT_HISTORICAL"


def load_universe(path, start, diagnostic):
    meta = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    groups = meta["funds"]
    if not groups or any(len(c) != 6 or not c.isascii() or not c.isdigit() for c in groups):
        raise ValueError("invalid universe codes")
    if set(groups.values()) - (CORE | SATELLITES | {"enhanced"}):
        raise ValueError("invalid fund risk group")
    eligible_date = pd.Timestamp(meta["candidate_asof"])
    if eligible_date >= start and not diagnostic:
        raise ValueError("LOOKAHEAD_BLOCKED: shortlist is not prior to the test start. "
                         "Use --diagnostic for this retrospective replay, or provide a dated prior universe.")
    return groups, eligible_date


def summarize(result, groups, contributed_cash):
    at = result["attribution"]
    daily = result["daily"]
    events = result["events"]
    counts = Counter(events["event"].dropna()) if "event" in events else Counter()
    decisions = Counter(events["action"].dropna()) if "action" in events else Counter()
    fixed = events.loc[events["action"] == "FIXED_INFLOW"] if "action" in events else pd.DataFrame()
    stress_blocked = int(fixed["stress_blocked"].eq(True).sum()) if "stress_blocked" in fixed else 0
    group_map = {c: g for c, g in groups.items()}
    group_map.update({"cash_interest": "cash"})
    dd_groups = {}
    for key, value in at["drawdown"]["loss_contribution"].items():
        group = group_map.get(key, "fees")
        dd_groups[group] = dd_groups.get(group, 0.0) + value
    fees = pd.to_numeric(events.get("fee_cny", pd.Series(dtype=float)), errors="coerce").fillna(0)
    buys = pd.to_numeric(events.get("amount", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if "event" in events and not buys.empty:
        buys = buys.loc[events["event"] == "BUY"]
    sells = pd.to_numeric(events.get("gross_proceeds_cny", pd.Series(dtype=float)),
                          errors="coerce").fillna(0)
    years = (daily.date.iloc[-1] - daily.date.iloc[0]).days / 365.25
    from fundkit.finance_math import max_drawdown
    if abs(at["drawdown"]["max_drawdown"] - max_drawdown([1.0] + daily.unit_nav.tolist())) > 1e-9:
        raise ArithmeticError("drawdown attribution does not reconcile")
    return {**result["metrics"], "buys": counts["BUY"], "redemptions": counts["REDEEM"],
            "fees_fraction_of_paid_in_capital": float(fees.sum()/contributed_cash),
            "annualized_gross_turnover": float((buys.sum()+sells.sum())/daily.value.mean()/years),
            "average_cash_weight": float((daily.cash/daily.value).mean()),
            "average_pending_weight": float((daily.pending/daily.value).mean()),
            "events_by_reason": dict(counts), "decisions_by_reason": dict(decisions),
            "stress_blocked_sessions": stress_blocked,
            "attribution": {"drawdown": {**at["drawdown"], "group_loss_contribution": dd_groups},
                            "total_return_contribution": at["total_return_contribution"]}}


def run_case(groups, target, limits, args, *, benchmark=False):
    caps, blocked_buy, blocked_sell, status = trade_constraints(
        groups, args.allow_missing_trade_status)
    params = dict(simulator.SIM_DEFAULTS, subscribe_fee_rate=args.fee_bps/10000,
                  redeem_fee_rate=args.fee_bps/10000, cash_annual_rate=args.cash_rate,
                  # T+2 23:59 is not visible at T+2 14:55.
                  published_lag_days=3)
    cfg = simulator.SimConfig(
        budget_per_day=args.daily_inflow, allow_new_equity_index=True,
        equity_index_codes=frozenset(c for c, g in groups.items() if g.endswith("equity")),
        daily_limits=caps, params=params, track_attribution=True,
        include_signal_history=False, track_exposures=True,
    )
    policy = make_policy(groups, target, limits, trade_limits=caps, blocked_buy=blocked_buy,
                         blocked_sell=blocked_sell, fee_rate=args.fee_bps/10000,
                         risk_gate=not benchmark, rebalance=not benchmark,
                         group_rules=not benchmark)
    result = simulator.Simulator(args.common_codes, cfg).run(args.start, args.end, policy=policy,
                                                  initial_cash=args.initial_cash)
    report = summarize(result, groups, args.initial_cash + args.daily_inflow*len(result["daily"]))
    marks = result["weights"]
    max_weights = {}
    breaches = {}
    for name, members, cap in limits.constraints(groups):
        w = marks[list(members)].sum(axis=1)
        max_weights[name] = float(w.max())
        breaches[name] = int((w > cap+1e-7).sum())
    report.update(target=target, limits=asdict(limits), trade_status=status,
                  execution_status=("TRADES_OBSERVED" if report["buys"] or report["redemptions"]
                                    else "NO_FILLED_TRADES"),
                  snapshot_buy_blocked=sorted(blocked_buy),
                  snapshot_sell_blocked=sorted(blocked_sell),
                  actual_post_trade_max_weights=max_weights, actual_cap_breach_days=breaches,
                  test_start=str(result["daily"].date.iloc[0].date()),
                  test_end=str(result["daily"].date.iloc[-1].date()),
                  historical_finite_window_target_met=bool(
                      report["cagr"] >= .05 and report["max_drawdown"] <= limits.drawdown))
    return report, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="config/v14_core_satellite.json")
    p.add_argument("--diagnostic", action="store_true",
                   help="allow a retrospective replay using a later shortlist (not OOS)")
    p.add_argument("--satellite-caps", type=float, nargs="+", default=[0., .05, .10, .15])
    p.add_argument("--enhancement-target", type=float, default=0.,
                   help="optional extra run with this enhancement weight, capped at 10%%")
    p.add_argument("--optimizer", action="store_true", help="also run robust SOCP targets")
    p.add_argument("--initial-cash", type=float, default=0.,
                   help="uninvested starting cash, not an initial holding and not bulk invested")
    p.add_argument("--daily-inflow", type=float, default=1000.)
    p.add_argument("--fee-bps", type=float, default=60.)
    p.add_argument("--cash-rate", type=float, default=0.)
    p.add_argument("--allow-missing-trade-status", action="store_true",
                   help="explicit research-only override when no local status file exists")
    p.add_argument("--allow-missing-dividends", action="store_true",
                   help="allow missing cash dividend file; results are explicitly incomplete")
    p.add_argument("--output", default="reports/v14_core_satellite")
    a = p.parse_args()
    try:
        a.start, a.end = pd.Timestamp(a.start), pd.Timestamp(a.end)
        if (pd.isna(a.start) or pd.isna(a.end) or a.end <= a.start or
                a.start.tzinfo is not None or a.end.tzinfo is not None or
                a.start != a.start.normalize() or a.end != a.end.normalize()):
            raise ValueError("start/end must be date-only and end must follow start")
        if (any(not math.isfinite(x) or x < 0 for x in
                (a.daily_inflow, a.initial_cash, a.cash_rate, a.fee_bps, a.enhancement_target))
                or a.daily_inflow <= 0 or a.fee_bps >= 10000 or a.enhancement_target > .1):
            raise ValueError("invalid cash, fee or enhancement settings")
        universe, selection_date = load_universe(a.universe, a.start, a.diagnostic)
        grids = sorted({0., *a.satellite_caps})
        for cap in grids:
            Limits(cap)
        # Match the actual evaluation interval across all compared cases.
        eligible = {c: g for c, g in universe.items() if g in CORE or
                    (g in SATELLITES and max(grids) > 0) or
                    (g == "enhanced" and (a.optimizer or a.enhancement_target > 0))}
        missing = [c for c in eligible if not storage.has_nav(c)]
        if missing:
            raise ValueError("DATA_BLOCKED: missing NAV files: " + ",".join(missing) +
                             ". Fetch these codes with python main.py fetch --codes " +
                             ",".join(missing))
        first_days, last_days = [], []
        for c in eligible:
            d = storage.load_nav(c)
            if (d.empty or d.date.isna().any() or d.date.duplicated().any() or
                    not d.nav.map(lambda x: math.isfinite(x) and x > 0).all()):
                raise ValueError(f"DATA_BLOCKED: invalid NAV history for {c}")
            d = d.loc[d.date <= a.end]
            if d.empty:
                raise ValueError(f"DATA_BLOCKED: {c} did not exist in requested window")
            first_days.append(d.date.iloc[0])
            last_days.append(d.date.iloc[-1])
        if max(first_days) >= a.start:
            raise ValueError("DATA_BLOCKED: at least one included fund lacks pre-window NAV")
        requested_end = a.end
        a.end = min(last_days)
        if a.end <= a.start or (requested_end-a.end).days > 7:
            raise ValueError("DATA_BLOCKED: NAV coverage does not reach the requested end")
        if (a.end-a.start).days < 28:
            raise ValueError("DATA_BLOCKED: test window must span at least 28 calendar days")
        a.common_codes = list(eligible)
        trade_constraints(eligible, a.allow_missing_trade_status)
        if not Path(storage.DIVIDENDS_FILE).exists() and not a.allow_missing_dividends:
            raise ValueError("DATA_BLOCKED: missing data/dividends.csv; run python main.py dividends, "
                             "or use --allow-missing-dividends for an incomplete diagnostic")
        if not Path(storage.DIVIDENDS_FILE).exists():
            a.diagnostic = True
    except (ValueError, KeyError, OSError, TypeError) as exc:
        p.exit(2, str(exc)+"\n")

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=False).stdout.strip() or None
    except OSError:
        git_sha = None
    results = {"research_only": True, "trade_ready": False, "git_sha": git_sha,
               "candidate_asof": str(selection_date.date()),
               "evidence": ("RETROSPECTIVE_SHORTLIST_LOOKAHEAD_DIAGNOSTIC" if selection_date >= a.start
                            else "DATED_UNIVERSE_NOT_INDEPENDENTLY_VERIFIED"),
               "diagnostic_override": a.diagnostic,
               "objective": {"net_cagr_at_least": .05, "max_drawdown_at_most": .04},
               "requested_start": str(a.start.date()), "requested_end": str(requested_end.date()),
               "common_latest_test_date": str(a.end.date()), "fee_bps_proxy": a.fee_bps,
               "stress_terminal_returns": STRESS, "fund_risk_groups": eligible,
               "dividend_source": ("LOCAL_DIVIDEND_CACHE_UNVERIFIED_COVERAGE"
                                   if Path(storage.DIVIDENDS_FILE).exists()
                                   else "MISSING_DIAGNOSTIC_ONLY"),
               "nav_publication": "T+2_CALENDAR_DAYS_23_59_PROXY",
               "observed_portfolio_drawdown_visibility": "T+3_CALENDAR_DAY_AT_14_55_PROXY",
               "initialization": "CASH_AND_FIXED_DAILY_INFLOW_NO_REAL_ACCOUNT",
               "cases": {}}

    def record(name, groups, target, limits, audit=None, benchmark=False):
        report, result = run_case(groups, target, limits, a, benchmark=benchmark)
        _, dd = combined_drawdown({c: target[c] for c in groups}, 1., 0., groups)
        report["target_prescreen_stress_mdd"] = dd
        report["target_stress_feasible"] = dd <= limits.drawdown
        report["allocation_audit"] = audit
        report["strategy"] = "FIXED_BUY_ONLY_NO_GUARDS" if benchmark else "CORE_SATELLITE"
        results["cases"][name] = report
        result["attribution"]["daily"].to_csv(out/f"{name}_daily_attribution.csv", index=False)
        result["weights"].to_csv(out/f"{name}_weights.csv", index=False)
        # Never export account cash amounts. Decision events here are already ratios.
        ev = result["events"].reindex(columns=["date", "code", "action", "event", "scope",
                       "planned_pct", "unfilled_pct", "observed_drawdown", "pre_stress_drawdown",
                       "post_stress_drawdown", "stress_case", "accepted_buy_pct", "stress_blocked"])
        ev.to_csv(out/f"{name}_execution_audit.csv", index=False)
        print(f"{name:24s} CAGR={report['cagr']:7.2%} MDD={report['max_drawdown']:6.2%} "
              f"Sharpe={report['sharpe']!s} buy={report['buys']} sell={report['redemptions']}",
              flush=True)

    for cap in grids:
        limits = Limits(cap)
        groups = {c:g for c,g in eligible.items() if g in CORE or (cap and g in SATELLITES)}
        label = f"sat_{cap*100:g}".replace(".","p")
        try:
            target = fixed_weights(groups, limits)
            record(f"fixed_{label}", groups, target, limits)
            if cap:
                record(f"buy_only_{label}", groups, target, limits, benchmark=True)
        except (ValueError, ArithmeticError) as exc:
            results["cases"][f"fixed_{label}"] = {"status":"DATA_BLOCKED", "reason":str(exc)}
        enhanced_groups = {c:g for c,g in eligible.items() if c in groups or g == "enhanced"}
        if a.enhancement_target > 0 and cap:
            try:
                target = fixed_weights(enhanced_groups, limits, a.enhancement_target)
                record(f"enhanced_{label}", enhanced_groups, target, limits)
            except (ValueError, ArithmeticError) as exc:
                results["cases"][f"enhanced_{label}"] = {"status":"DATA_BLOCKED", "reason":str(exc)}
        if a.optimizer and cap:
            name = f"socp_{label}"
            try:
                cut = a.start.tz_localize(CST) + pd.Timedelta(hours=14, minutes=55)
                cfg = WeightSearchConfig(lookback_days=756)
                ret = build_returns_asof(LocalDataAdapter(), list(enhanced_groups), cut, cfg)
                target, audit = optimized_weights(ret, enhanced_groups, limits, fee_rate=a.fee_bps/10000)
                record(name, enhanced_groups, target, limits, audit)
            except (ValueError, ArithmeticError) as exc:
                results["cases"][name] = {"status":"DATA_BLOCKED", "reason":str(exc)}

    blocked = [name for name, r in results["cases"].items() if "status" in r]
    results["historical_finite_window_passes"] = [
        name for name, row in results["cases"].items()
        if row.get("historical_finite_window_target_met") and row["execution_status"] == "TRADES_OBSERVED"]
    results["long_term_target_validated"] = False
    results["status"] = ("DATA_BLOCKED" if len(blocked) == len(results["cases"]) else
                         "PARTIAL_DATA_BLOCKED" if blocked else "RESEARCH_ONLY_COMPLETE")
    (out/"summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                               allow_nan=False), encoding="utf-8")
    for name in blocked:
        print(f"{name}: {results['cases'][name]['reason']}")
    print(f"{results['status']}: {out/'summary.json'}")
    return 2 if results["status"] == "DATA_BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
