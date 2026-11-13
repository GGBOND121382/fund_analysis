"""Research-only core/satellite weights, stress paths and cash-flow rebalancing."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

EPS = 1e-9
SATELLITES = frozenset({"china_equity", "gold", "us_equity"})
CORE = frozenset({"core_short", "core_policy", "core_rate"})

# Predeclared deterministic shocks, not probabilities, estimates or forecasts.
# Entries are cumulative returns before cash/trading costs.
STRESS = {
    "china_equity_gold_down": dict(core_short=-.003, core_policy=-.01, core_rate=-.015,
                                   enhanced=-.04, china_equity=-.20, gold=-.16, us_equity=-.10),
    "us_equity_fx_down": dict(core_short=-.003, core_policy=-.01, core_rate=-.015,
                              enhanced=-.04, china_equity=-.12, gold=-.08,
                              us_equity=(1-.22)*(1-.06)-1),
    "rates_and_credit_up": dict(core_short=-.008, core_policy=-.018, core_rate=-.024,
                                enhanced=-.06, china_equity=-.08, gold=-.06, us_equity=-.10),
}


@dataclass(frozen=True)
class Limits:
    satellite: float
    enhancement: float = .10
    fund: float = .25
    minimum_cash: float = .02
    drawdown: float = .04

    def __post_init__(self):
        if not all(isfinite(v) and 0 <= v < 1 for v in
                   (self.satellite, self.enhancement, self.fund,
                    self.minimum_cash, self.drawdown)) or not self.fund or not self.drawdown:
            raise ValueError("invalid research limits")

    def constraints(self, groups):
        # Multiple shares of the same index cannot bypass the common budget.
        rows = [(c, (c,), self.fund) for c in groups]
        rows.extend((g, tuple(c for c in groups if groups[c] == g), cap) for g, cap in (
            ("china_equity", min(.075, self.satellite)),
            ("gold", min(.075, self.satellite)),
            ("us_equity", min(.075, self.satellite)),
            ("enhanced", self.enhancement),
            ("core_short", .55),
            ("core_policy", .55),
            ("core_rate", .55),
        ))
        rows.append(("satellite_total", tuple(c for c in groups if groups[c] in SATELLITES),
                     self.satellite))
        return [row for row in rows if row[1]]


def fixed_weights(groups, limits, enhancement_target=0.0):
    """Equal within core and within each satellite sleeve. Cash is residual."""
    if set(groups.values()) - (CORE | SATELLITES | {"enhanced"}):
        raise ValueError("unclassified fund")
    if not 0 <= enhancement_target <= limits.enhancement:
        raise ValueError("enhancement target exceeds sleeve cap")
    core = [c for c in groups if groups[c] in CORE]
    enhanced = [c for c in groups if groups[c] == "enhanced"]
    sleeves = sorted(set(groups.values()) & SATELLITES)
    if not core or (enhancement_target and not enhanced):
        raise ValueError("DATA_BLOCKED: required core/enhancement funds missing")
    if limits.satellite and not sleeves:
        raise ValueError("DATA_BLOCKED: satellite cap requested without satellite assets")
    target = dict.fromkeys(groups, 0.0)
    for c in enhanced:
        target[c] = min(limits.fund, enhancement_target / len(enhanced))
    # Strategic target is below the trigger limit, leaving room for market drift.
    for sleeve in sleeves:
        members = [c for c in groups if groups[c] == sleeve]
        sleeve_target = min(.8 * limits.satellite / len(sleeves), .8 * .075)
        for c in members:
            target[c] = min(limits.fund, sleeve_target / len(members))
    core_target = (1.0 - limits.minimum_cash - sum(target.values())) / len(core)
    if core_target < -EPS or core_target > limits.fund + EPS:
        raise ValueError("DATA_BLOCKED: fixed core target cannot fit fund caps")
    for c in core:
        target[c] = max(0.0, core_target)
    if any(sum(target[c] for c in members) > cap + EPS
           for _, members, cap in limits.constraints(groups)):
        raise ValueError("DATA_BLOCKED: fixed core target cannot fit group caps")
    return {**target, "cash": max(0.0, 1 - sum(target.values()))}


def stress_paths(codes, groups, horizon=20):
    """Joint monotone stress paths, independent of dates and training returns."""
    if horizon < 2 or set(groups.values()) - (CORE | SATELLITES | {"enhanced"}):
        raise ValueError("invalid horizon or unclassified fund")
    terminal = np.array([[v[groups[c]] for c in codes] for v in STRESS.values()])
    daily = np.expm1(np.log1p(terminal) / horizon)
    return np.repeat(daily[:, None, :], horizon, axis=1)


def combined_drawdown(marked, total, observed_dd, groups, *, fee=0.0):
    if total <= 0 or fee < 0 or fee >= total:
        raise ValueError("invalid stress-account denominator")
    values = [(name, (1 - observed_dd) * (fee - sum(
        marked[c] * shock[groups[c]] for c in groups)) / total + observed_dd)
              for name, shock in STRESS.items()]
    return max(values, key=lambda x: x[1])


def make_policy(groups, target, limits, *, trade_limits=None, blocked_buy=(),
                blocked_sell=(), fee_rate=.006, risk_gate=True, rebalance=True,
                group_rules=True, variant="fixed", valuation_history=None, index_history=None):
    """Fixed inflow, with opt-in, bounded satellite signals."""
    from fundkit.v15_signals import VARIANTS, TrendState, market_signal
    use_dip, use_trend, use_value = VARIANTS[variant]
    if use_value and valuation_history is None:
        raise ValueError("VALUATION_DATA_BLOCKED: a historical valuation CSV is required")
    groups = dict(groups)
    if set(target) != set(groups) | {"cash"} or abs(sum(target.values()) - 1) > EPS:
        raise ValueError("invalid target weights")
    if not isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("invalid fee rate")
    rules = limits.constraints(groups) if group_rules else []
    blocked_buy, blocked_sell = set(blocked_buy), set(blocked_sell)
    trade_limits = trade_limits or {}
    if any(not isfinite(x) or x < 0 for x in target.values()) or any(
        sum(target[c] for c in codes) > cap + EPS for _, codes, cap in rules
    ):
        raise ValueError("target breaches group or fund caps")
    satellites = [c for c,g in groups.items() if g in SATELLITES and target[c] > EPS]
    trends = {c: TrendState() for c in satellites}
    valuation_cache = {}

    def policy(state):
        notes = []
        def note(action, **kwargs):
            notes.append({"action": action, **kwargs})
        def orders(buys=None, sells=None):
            return {"subscribe": buys or {}, "redeem": sells or {}, "events": notes}

        total, cash = float(state["value"]), float(state["cash"])
        if total <= 0 or state["value_quality"] != "COMPLETE":
            note("MISSING_ASOF_MARK")
            return orders()
        h = state["holdings"]
        marked = {c: float(h.get(c, {}).get("value_pub", 0.0)) for c in groups}
        hist = state.get("unit_nav_history") or []
        observed = max(0.0, 1 - hist[-1] / max(1.0, max(hist))) if hist else 0.0
        _, pre_stress = combined_drawdown(marked, total, observed, groups)

        active_target = dict(target)
        signal = {}
        trend_sell = {}
        valuation_trim = set()
        paused = set()
        if use_value:
            period = str(state["date"])[:7]
            if period not in valuation_cache:
                valuation_cache[period] = valuation_history.signal(state["date"])
                value = valuation_cache[period]
                note("VALUATION_MONTH" if value else "VALUATION_NOT_AVAILABLE",
                     valuation_rank=value["rank"] if value else None,
                     valuation_factor=value["factor"] if value else None,
                     signal_date=value["asof"] if value else None)
            value = valuation_cache[period]
            if value:
                for c in satellites:
                    if c == "110020":
                        active_target[c] *= value["factor"]
                        if value["factor"] < 1.0:
                            valuation_trim.add(c)
        # A valuation tilt cannot exceed either its sleeve or the overall satellite cap.
        for name, codes, cap in rules:
            if name in SATELLITES or name == "satellite_total":
                total_target = sum(active_target[c] for c in codes)
                if total_target > cap and total_target > EPS:
                    for c in codes:
                        active_target[c] *= cap/total_target
        for c in satellites:
            active_target[c] = min(active_target[c], limits.fund)
            if use_dip or use_trend:
                history = (index_history.rows(c, state["date"]) if index_history is not None else
                           state.get("fund_nav_history", {}).get(c, []))
                signal[c] = market_signal(history, state["date"])
                if signal[c] is None:
                    note("MARKET_SIGNAL_NOT_AVAILABLE", code=c)
                    if use_trend and trends[c].scale < 1:
                        paused.add(c)
            if use_trend:
                z = trends[c]
                risk_pressure = pre_stress >= .8*limits.drawdown
                if z.update(signal[c], allow_exit=risk_pressure):
                    note("TREND_LEVEL_CHANGED", code=c, trend_level=z.level,
                         signal_date=str(z.last.date()))
                active_target[c] *= z.scale
                if z.pause:
                    paused.add(c)
                if z.scale < 1 and z.direction < 0 and risk_pressure and signal[c] is not None:
                    trend_sell[c] = active_target[c]
        # Finance a cheap-valuation overweight with future core subscriptions,
        # never by assuming an extra deposit or spending the cash floor.
        overflow = max(0.0, sum(active_target[c] for c in groups) + target["cash"] - 1.0)
        if overflow:
            core_total = sum(active_target[c] for c,g in groups.items() if g in CORE)
            if overflow > core_total:
                raise ValueError("valuation tilt exceeds core funding")
            for c,g in groups.items():
                if g in CORE:
                    active_target[c] *= (core_total-overflow)/core_total
        # Risk removed by signals remains unallocated, not forced into other risky sleeves.
        reserve_weight = max(target["cash"], 1 - sum(active_target[c] for c in groups))

        sells = {}
        sell_denominator = total
        sell_exposure = marked.copy()
        def reduce(name, codes, dest, reason):
            nonlocal sell_denominator
            need = max(0.0, (sum(sell_exposure[c] for c in codes) -
                             dest * sell_denominator) / (1 - dest * fee_rate))
            if need <= EPS:
                return
            planned = 0.0
            for c in sorted(codes, key=lambda c: (active_target[c]*total - sell_exposure[c], c)):
                if need <= EPS or c in blocked_sell or marked[c] <= EPS:
                    continue
                item = h[c]
                nav = marked[c] / float(item["shares"])
                available = max(0.0, float(item["redeemable_shares"]) * nav - sells.get(c, 0.0)*nav)
                amount = min(available, need)
                if amount > EPS:
                    sells[c] = sells.get(c, 0.0) + amount/nav
                    sell_exposure[c] -= amount
                    sell_denominator -= amount * fee_rate
                    planned += amount
                    need -= amount
            note(reason if planned > EPS else ("CAP_REDEMPTION_BLOCKED" if reason == "CAP_REBALANCE" else reason + "_BLOCKED"),
                 scope=name, planned_pct=planned/total, unfilled_pct=need/total)

        if rebalance:
            for name, codes, cap in rules:
                if (sum(sell_exposure[c] for c in codes) > cap * sell_denominator + EPS
                        and sum(marked[c] for c in codes) > cap * total + EPS):
                    dest = .5 * (sum(active_target[c] for c in codes) + cap)
                    reduce(name, codes, dest, "CAP_REBALANCE")
            for c, dest in trend_sell.items():
                if sell_exposure[c] > (dest+.001)*sell_denominator:
                    reduce(c, (c,), dest, "TREND_REBALANCE")
            for c in sorted(valuation_trim):
                dest = active_target[c]
                # No trade for small valuation-induced drifts. Move halfway
                # across a 0.5 pp band, not to a zero allocation.
                if sell_exposure[c] > (dest+.005)*sell_denominator:
                    reduce(c, (c,), dest+.0025, "VALUATION_REBALANCE")
        if sells:
            scenario, post_stress = combined_drawdown(
                sell_exposure, total, observed, groups, fee=total - sell_denominator)
            note("SALE_ONLY", observed_drawdown=observed, pre_stress_drawdown=pre_stress,
                 post_stress_drawdown=post_stress, stress_case=scenario)
            # Explicitly separate sale and buy days, no dependence on unsettled proceeds.
            return orders(sells=sells)

        reserve = reserve_weight * total
        budget = min(max(0.0, cash - reserve), float(state["external_cash_inflow_today"]))
        denominator_floor = total - budget * fee_rate
        buys = {}
        projected = marked.copy()
        def allocate(c, ceiling):
            nonlocal budget
            if (c in blocked_buy or c in paused or c not in state["nav"] or budget <= EPS or
                    (risk_gate and not hist and marked[c] > EPS)):
                return 0.
            if risk_gate and groups[c] not in CORE and (not hist or pre_stress > limits.drawdown):
                return 0.
            amount = min(ceiling, budget, active_target[c]*denominator_floor - projected[c],
                         trade_limits.get(c, float("inf")) - buys.get(c, 0.0))
            for _, codes, cap in rules:
                if c in codes:
                    amount = min(amount, cap*denominator_floor -
                                 sum(projected[x] for x in codes))
            if amount <= EPS:
                return 0.
            if risk_gate:
                if pre_stress > limits.drawdown:
                    return 0.
                fees_so_far = fee_rate * sum(buys.values())
                for shock in STRESS.values():
                    base_loss = -sum(projected[x]*shock[groups[x]] for x in groups)
                    available = total * (limits.drawdown - observed)/(1 - observed) - base_loss - fees_so_far
                    marginal = -(1 - fee_rate)*shock[groups[c]] + fee_rate
                    amount = min(amount, available/marginal)
            if amount <= EPS:
                return 0.
            buys[c] = buys.get(c, 0.0) + amount
            projected[c] += (1-fee_rate)*amount
            budget -= amount
            return amount

        if use_dip:
            priority = .25*budget
            dipped = [c for c in satellites if signal[c] is not None and signal[c]["dip"]]
            for c in sorted(dipped, key=lambda c: (-signal[c]["drawdown20"], c)):
                amount = allocate(c, priority)
                priority -= amount
                note("DIP_PRIORITY" if amount > EPS else "DIP_NO_FILL_PLANNED",
                     code=c, planned_pct=amount/total,
                     signal_date=str(signal[c]["date"].date()))
        for c in sorted(groups, key=lambda c: (projected[c] - active_target[c]*total, c)):
            allocate(c, budget)
        stress_case, post_stress = combined_drawdown(
            projected, total, observed, groups, fee=fee_rate*sum(buys.values()))
        note("FIXED_INFLOW", observed_drawdown=observed, pre_stress_drawdown=pre_stress,
             post_stress_drawdown=post_stress, stress_case=stress_case,
             accepted_buy_pct=sum(buys.values())/total,
             stress_blocked=bool(risk_gate and pre_stress > limits.drawdown))
        return orders(buys)

    return policy


def optimized_weights(returns, groups, limits, *, fee_rate, seed=42):
    """As-of-only robust allocation; fixed baseline remains the primary comparator."""
    from fundkit.v13_model import fit_one_year_model
    from fundkit.v13_portfolio import RobustSOCPConfig, solve_robust_socp
    from fundkit.weight_optimizer import WeightSearchConfig, _scenarios, _risk_metrics

    codes = list(groups)
    returns = returns[codes]
    if len(returns) < 252:
        raise ValueError("DATA_BLOCKED: robust allocation needs 252 joint published returns")
    r = returns.to_numpy()
    cfg = WeightSearchConfig(lookback_days=len(r), bootstrap_paths=32, seed=seed,
                             turnover_cost_rate=fee_rate,
                             min_cash_weight=limits.minimum_cash)
    paths = _scenarios(r, cfg)
    categories = {c: "bond" if g in CORE or g == "enhanced" else
                  "gold" if g == "gold" else "equity_index" for c, g in groups.items()}
    fit = fit_one_year_model(returns.tail(252), categories=categories, seed=seed)
    reference = fixed_weights(groups, limits)
    current = np.array([reference[c] for c in [*codes, "cash"]])
    # Target bounds below the execution triggers leave room for market drift.
    bounds = [(tuple(codes.index(c) for c in members), .8*cap)
              for name, members, cap in limits.constraints(groups) if name not in groups]
    ans = solve_robust_socp(
        r, paths, fit["mean_annual"], fit["return_cov_annual"],
        fit["forecast_error_cov_annual"], fund_upper=np.full(len(codes), limits.fund),
        current_weights=current, group_upper=bounds,
        hard_stress_returns=stress_paths(codes, groups),
        config=RobustSOCPConfig(drawdown_budget=limits.drawdown, turnover_cost=fee_rate,
                                min_cash_weight=limits.minimum_cash,
                                min_risky_weight=.85, uncertainty_penalty=1.0,
                                volatility_penalty=1.0),
    )
    if ans["status"] != "OPTIMAL":
        raise ValueError("NO_RISK_FEASIBLE_TARGET: " + str(ans["status"]) + " " +
                         str(ans.get("reason", "")))
    w = np.minimum(limits.fund, np.maximum(np.asarray(ans["weights"][:-1]), 0.0))
    w[w < 1e-8] = 0.
    for idx, cap in bounds:
        ids = list(idx)
        if w[ids].sum() > cap:
            w[ids] *= cap/w[ids].sum()
    if w.sum() > 1-limits.minimum_cash:
        w *= (1-limits.minimum_cash)/w.sum()
    if w.sum() < .85-1e-6:
        raise ValueError("NO_RISK_FEASIBLE_TARGET: numerical invested floor")
    target = {**dict(zip(codes, map(float, w))), "cash": 1 - float(w.sum())}
    ordered = np.array([*w, target["cash"]])
    historical, q95, q25, *_ = _risk_metrics(r, ordered[None, :], current, cfg, paths)
    _, stress_dd = combined_drawdown({c: target[c] for c in codes}, 1., 0., groups)
    if max(historical[0], q95[0], stress_dd) > limits.drawdown + 1e-6:
        raise ValueError("NO_RISK_FEASIBLE_TARGET: rebalanced-path independent risk check")
    audit = {"historical_mdd": float(historical[0]), "scenario_mdd_p95": float(q95[0]),
             "scenario_return_q25": float(q25[0]), "hard_stress_mdd": dict(zip(STRESS, ans["hard_stress_mdd"])),
             "joint_observations": len(returns), "training_start": str(returns.index[0].date()),
             "training_end": str(returns.index[-1].date()),
             "forecast_method": "V13_PRIOR_SHRINKAGE_NOT_BENCHMARK_CALIBRATED",
             "turnover_reference": "FIXED_RESEARCH_WEIGHTS_NOT_ACCOUNT_HOLDINGS"}
    return target, audit
