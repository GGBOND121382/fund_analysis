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
from fundkit import simulator, storage, policies  # noqa: E402
from fundkit.adapters import CST, LocalDataAdapter  # noqa: E402
from fundkit.v14_core_satellite import (CORE, SATELLITES, STRESS, Limits,  # noqa: E402
    combined_drawdown, fixed_weights, make_policy, optimized_weights)
from fundkit.weight_optimizer import WeightSearchConfig, build_returns_asof
from fundkit.v15_signals import VARIANTS, IndexHistory, ValuationHistory  # noqa: E402


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


def resolve_fees(args):
    """Return subscription bps, redemption bps, and fee scenario provenance."""
    symmetric = getattr(args, "fee_bps", None)
    buy = getattr(args, "subscribe_fee_bps", None)
    sell = getattr(args, "redeem_fee_bps", None)
    if symmetric is not None and (buy is not None or sell is not None):
        raise ValueError("--fee-bps cannot be combined with directional fee flags")
    if symmetric is not None:
        buy, sell, mode = symmetric, symmetric, "LEGACY_SYMMETRIC_OVERRIDE"
    elif not getattr(args, "layered", False) and buy is None and sell is None:
        buy, sell, mode = 60., 60., "V14_LEGACY_SYMMETRIC_DEFAULT"
    else:
        buy, sell, mode = 0. if buy is None else buy, 60. if sell is None else sell, "DIRECTIONAL"
    if any(not math.isfinite(x) or not 0 <= x < 10000 for x in (buy, sell)):
        raise ValueError("fee bps must be finite and in [0, 10000)")
    return buy, sell, mode


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
    buy_fees = fees[events["event"].isin(("BUY", "INITIAL_BUY", "INITIAL_BUY_FEE"))].sum() if len(fees) else 0.
    sell_fees = fees[events["event"].eq("REDEEM")].sum() if len(fees) else 0.
    if not math.isclose(buy_fees + sell_fees, fees.sum(), abs_tol=1e-8):
        raise ArithmeticError("unclassified transaction fees")
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
            "subscription_fees_fraction_of_paid_in_capital": float(buy_fees/contributed_cash),
            "redemption_fees_fraction_of_paid_in_capital": float(sell_fees/contributed_cash),
            "ending_gain_fraction_of_paid_in_capital": float(daily.value.iloc[-1]/contributed_cash-1),
            "annualized_gross_turnover": float((buys.sum()+sells.sum())/daily.value.mean()/years),
            "average_cash_weight": float((daily.cash/daily.value).mean()),
            "average_pending_weight": float((daily.pending/daily.value).mean()),
            "events_by_reason": dict(counts), "decisions_by_reason": dict(decisions),
            "stress_blocked_sessions": stress_blocked,
            "attribution": {"drawdown": {**at["drawdown"], "group_loss_contribution": dd_groups},
                            "total_return_contribution": at["total_return_contribution"]}}


def run_case(groups, target, limits, args, *, benchmark=False, variant="fixed"):
    legacy = variant in ("legacy_b2", "legacy_a5")
    use_signal = variant != "fixed" and not legacy
    valuation = getattr(args, "valuation_history", None)
    index_history = getattr(args, "index_history", None)
    caps, blocked_buy, blocked_sell, status = trade_constraints(
        groups, args.allow_missing_trade_status)
    buy_bps, sell_bps, _ = resolve_fees(args)
    buy_fee, sell_fee = buy_bps/10000, sell_bps/10000
    params = dict(simulator.SIM_DEFAULTS, subscribe_fee_rate=buy_fee,
                  redeem_fee_rate=sell_fee, cash_annual_rate=args.cash_rate,
                  # T+2 23:59 is not visible at T+2 14:55.
                  published_lag_days=3)
    cfg = simulator.SimConfig(
        budget_per_day=args.daily_inflow, allow_new_equity_index=True,
        equity_index_codes=frozenset(c for c, g in groups.items() if g.endswith("equity")),
        daily_limits=caps, params=params, track_attribution=True,
        include_signal_history=(use_signal and index_history is None) or variant == "legacy_a5",
        track_exposures=True, signal_history_limit=756 if legacy else 126,
    )
    if legacy:
        params = policies.load_params()
        params.update(target_weights=target, daily_limits=caps, disabled_new_buy_codes=sorted(blocked_buy),
                      asset_categories={c: "bond" if g in CORE or g == "enhanced" else
                          "gold" if g == "gold" else "equity_index" for c,g in groups.items()},
                      equity_index_codes=sorted(cfg.equity_index_codes))
        if variant == "legacy_a5":
            from fundkit.v11_ablation import build_ablation_rules
            from fundkit.v11_policy import make_b6
            rule = next(x for x in build_ablation_rules() if x["rule_id"] == "A5_stop")
            params["B6"] = {**params["B6"], **rule["params"]["B6"],
                            "subscription_fee_proxy": buy_fee, "redeem_fee_proxy": sell_fee}
            base = make_b6(params)
        else:
            base = policies.make_b2(params)
        def policy(state):
            order = base(state) or {}
            order["subscribe"] = {c:v for c,v in order.get("subscribe", {}).items()
                                  if c in groups and c not in blocked_buy}
            order["redeem"] = {c:v for c,v in order.get("redeem", {}).items()
                               if c in groups and c not in blocked_sell}
            return order
        policy.on_execution = getattr(base, "on_execution", None)
    else:
        policy = make_policy(groups, target, limits, trade_limits=caps, blocked_buy=blocked_buy,
                             blocked_sell=blocked_sell, subscribe_fee_rate=buy_fee,
                             redeem_fee_rate=sell_fee,
                             risk_gate=not benchmark, rebalance=not benchmark,
                             group_rules=not benchmark, variant=variant, valuation_history=valuation,
                             index_history=index_history)
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
    decisions = report["decisions_by_reason"]
    if variant == "valuation":
        active = decisions.get("VALUATION_MONTH", 0)
        missing = decisions.get("VALUATION_NOT_AVAILABLE", 0)
        factors = result["events"].get("valuation_factor", pd.Series(dtype=float)).dropna()
        report["valuation_coverage"] = {"months_with_signal": active, "months_missing": missing,
                                         "non_neutral_months": int((factors != 1.).sum()),
                                         "neutral_fallback_in_missing_months": bool(missing)}
        if not active:
            raise ValueError("VALUATION_DATA_BLOCKED: no decision month has fresh PE/PB and 36 prior monthly snapshots")
    report["market_signal_missing_asset_days"] = decisions.get("MARKET_SIGNAL_NOT_AVAILABLE", 0)
    report["market_signal_candidate_asset_days"] = (
        len(result["daily"]) * sum(g in SATELLITES and target[c] > 1e-9
                                  for c,g in groups.items()) if use_signal else 0)
    if (use_signal and report["market_signal_candidate_asset_days"] and
        report["market_signal_missing_asset_days"] == report["market_signal_candidate_asset_days"]):
        raise ValueError("INDEX_DATA_BLOCKED: no usable market signal within the test window")
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
    p.add_argument("--fee-bps", type=float, help="legacy explicit symmetric fee override")
    p.add_argument("--subscribe-fee-bps", type=float, help="v1.5 subscription fee, default 0")
    p.add_argument("--redeem-fee-bps", type=float,
                   help="research redemption fee proxy, default 60; use 0 for sensitivity")
    p.add_argument("--cash-rate", type=float, default=0.)
    p.add_argument("--allow-missing-trade-status", action="store_true",
                   help="explicit research-only override when no local status file exists")
    p.add_argument("--allow-missing-dividends", action="store_true",
                   help="allow missing cash dividend file; results are explicitly incomplete")
    p.add_argument("--layered", action="store_true", help="run fixed, dip, trend-only, dip+trend and valuation comparisons")
    p.add_argument("--legacy", action="store_true", help="also compare legacy B2 and A5 on identical fixed targets")
    p.add_argument("--valuation-csv", default="data/factors/csi300_valuation.csv")
    p.add_argument("--index-csv", default=None, help="optional point-in-time total-return benchmarks, keyed by fund code")
    p.add_argument("--output", default="reports/v14_core_satellite")
    a = p.parse_args()
    try:
        a.start, a.end = pd.Timestamp(a.start), pd.Timestamp(a.end)
        if (pd.isna(a.start) or pd.isna(a.end) or a.end <= a.start or
                a.start.tzinfo is not None or a.end.tzinfo is not None or
                a.start != a.start.normalize() or a.end != a.end.normalize()):
            raise ValueError("start/end must be date-only and end must follow start")
        buy_bps, sell_bps, fee_mode = resolve_fees(a)
        if (any(not math.isfinite(x) or x < 0 for x in
                (a.daily_inflow, a.initial_cash, a.cash_rate, a.enhancement_target))
                or a.daily_inflow <= 0 or a.enhancement_target > .1):
            raise ValueError("invalid cash or enhancement settings")
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
        a.valuation_history, a.index_history = None, None
        valuation_blocked = None
        if a.layered:
            try:
                a.valuation_history = ValuationHistory(a.valuation_csv)
                if a.valuation_history.is_proxy and not a.diagnostic:
                    a.valuation_history = None
                    raise ValueError("VALUATION_DATA_BLOCKED: proxy history requires --diagnostic")
            except (ValueError, OSError, TypeError) as exc:
                valuation_blocked = str(exc)
            if a.index_csv:
                a.index_history = IndexHistory(a.index_csv)
                missing_bench = [c for c,g in eligible.items()
                                 if g in SATELLITES and c not in a.index_history.frames]
                if missing_bench:
                    raise ValueError("INDEX_DATA_BLOCKED: missing benchmark series: " + ",".join(missing_bench))
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
               "common_latest_test_date": str(a.end.date()),
               "fee_bps_proxy": buy_bps if buy_bps == sell_bps else None,
               "fees": {"mode": fee_mode, "subscribe_fee_bps": buy_bps,
                        "redeem_fee_bps": sell_bps, "redeem_fee_is_research_proxy": True,
                        "redeem_fee_holding_period_schedule_verified": False},
               "stress_terminal_returns": STRESS, "fund_risk_groups": eligible,
               "dividend_source": ("LOCAL_DIVIDEND_CACHE_UNVERIFIED_COVERAGE"
                                   if Path(storage.DIVIDENDS_FILE).exists()
                                   else "MISSING_DIAGNOSTIC_ONLY"),
               "nav_publication": "T+2_CALENDAR_DAYS_23_59_PROXY",
               "observed_portfolio_drawdown_visibility": "T+3_CALENDAR_DAY_AT_14_55_PROXY",
               "initialization": "CASH_AND_FIXED_DAILY_INFLOW_NO_REAL_ACCOUNT",
               "valuation_source_quality": (a.valuation_history.source_quality
                                            if a.valuation_history else None),
               "signal_source": ("USER_SUPPLIED_TOTAL_RETURN_BENCHMARKS" if a.index_history else
                                 "PUBLISHED_FUND_TOTAL_RETURN_NAV_PROXY"),
               "cases": {}}

    def record(name, groups, target, limits, audit=None, benchmark=False, variant="fixed"):
        report, result = run_case(groups, target, limits, a, benchmark=benchmark, variant=variant)
        _, dd = combined_drawdown({c: target[c] for c in groups}, 1., 0., groups)
        report["target_prescreen_stress_mdd"] = dd
        report["target_stress_feasible"] = dd <= limits.drawdown
        report["allocation_audit"] = audit
        report["strategy"] = "FIXED_BUY_ONLY_NO_GUARDS" if benchmark else variant
        report["signal_parameter_version"] = "V15_PREDECLARED" if a.layered else None
        results["cases"][name] = report
        result["attribution"]["daily"].to_csv(out/f"{name}_daily_attribution.csv", index=False)
        result["weights"].to_csv(out/f"{name}_weights.csv", index=False)
        # Never export account cash amounts. Decision events here are already ratios.
        ev = result["events"].reindex(columns=["date", "code", "action", "event", "scope",
                       "planned_pct", "unfilled_pct", "observed_drawdown", "pre_stress_drawdown",
                       "post_stress_drawdown", "stress_case", "accepted_buy_pct", "stress_blocked",
                       "signal_date", "trend_level", "valuation_rank", "valuation_factor"])
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
        if a.layered and cap:
            for variant in VARIANTS:
                if variant == "fixed":
                    continue
                name = f"{variant}_{label}"
                try:
                    if variant == "valuation":
                        if valuation_blocked:
                            raise ValueError(valuation_blocked)
                        if groups.get("110020") != "china_equity":
                            raise ValueError("VALUATION_DATA_BLOCKED: no CSI 300 fund in universe")
                    record(name, groups, fixed_weights(groups, limits), limits, variant=variant)
                except (ValueError, ArithmeticError) as exc:
                    results["cases"][name] = {"status":"DATA_BLOCKED", "reason":str(exc)}
        if a.legacy and cap:
            for variant in ("legacy_b2", "legacy_a5"):
                name = f"{variant}_{label}"
                try:
                    record(name, groups, fixed_weights(groups, limits), limits, variant=variant)
                except (ValueError, ArithmeticError, OSError, KeyError) as exc:
                    results["cases"][name] = {"status":"DATA_BLOCKED", "reason":str(exc)}
        enhanced_groups = {c:g for c,g in eligible.items() if c in groups or g == "enhanced"}
        if a.enhancement_target > 0 and (cap or a.layered):
            try:
                target = fixed_weights(enhanced_groups, limits, a.enhancement_target)
                record(f"enhanced_{label}", enhanced_groups, target, limits)
            except (ValueError, ArithmeticError) as exc:
                results["cases"][f"enhanced_{label}"] = {"status":"DATA_BLOCKED", "reason":str(exc)}
        if a.optimizer and (cap or a.layered):
            name = f"socp_{label}"
            try:
                cut = a.start.tz_localize(CST) + pd.Timedelta(hours=14, minutes=55)
                cfg = WeightSearchConfig(lookback_days=756)
                ret = build_returns_asof(LocalDataAdapter(), list(enhanced_groups), cut, cfg)
                fee_kwargs = ({"directional_fees": (buy_bps/10000, sell_bps/10000)}
                              if fee_mode == "DIRECTIONAL" else {"fee_rate": buy_bps/10000})
                target, audit = optimized_weights(ret, enhanced_groups, limits, **fee_kwargs)
                record(name, enhanced_groups, target, limits, audit)
            except (ValueError, ArithmeticError) as exc:
                results["cases"][name] = {"status":"DATA_BLOCKED", "reason":str(exc)}

    blocked = [name for name, r in results["cases"].items() if "status" in r]
    results["historical_finite_window_passes"] = [
        name for name, row in results["cases"].items()
        if row.get("historical_finite_window_target_met") and row["execution_status"] == "TRADES_OBSERVED"]
    results["long_term_target_validated"] = False
    if a.layered:
        pairs = {"dip": "fixed", "trend_only": "fixed", "trend": "dip", "valuation": "trend"}
        cases = results["cases"]
        results["layer_deltas"] = []
        columns = ("total_return_twr", "cagr", "xirr", "ending_gain_fraction_of_paid_in_capital",
                   "max_drawdown", "sharpe", "annualized_gross_turnover",
                   "fees_fraction_of_paid_in_capital", "subscription_fees_fraction_of_paid_in_capital",
                   "redemption_fees_fraction_of_paid_in_capital", "average_cash_weight")
        for name, case in cases.items():
            layer, _, suffix = name.partition("_sat_")
            if layer not in pairs:
                continue
            previous = pairs[layer] + "_sat_" + suffix
            other = cases[previous]
            row = {"case": name, "against": previous}
            if "status" in case or "status" in other:
                row["status"] = "DATA_BLOCKED"
            elif any(r["execution_status"] == "NO_FILLED_TRADES" for r in (case, other)):
                row["status"] = "NO_FILLED_TRADES_NOT_COMPARABLE"
            else:
                row["status"] = ("SAME_AGGREGATE_OUTCOME" if
                                 all(abs(case[k]-other[k]) < 1e-10 for k in
                                     ("cagr", "max_drawdown", "annualized_gross_turnover"))
                                 else "DIFFERENT_OUTCOME")
                row.update({k: case[k] - other[k] if case[k] is not None and other[k] is not None
                            else None for k in columns})
            results["layer_deltas"].append(row)
        pd.DataFrame(results["layer_deltas"]).to_csv(out/"layer_deltas.csv", index=False)
        rows = []
        for name, case in cases.items():
            rows.append({"case":name, "status":case.get("status", case.get("execution_status")),
                         **{k:case.get(k) for k in columns}, "reason":case.get("reason")})
        pd.DataFrame(rows).to_csv(out/"comparison.csv", index=False)
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
