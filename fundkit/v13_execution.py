"""v1.3 risk-budgeted *amount* choice for a fixed target portfolio.

Choose the best target-deficit reducing subscription plan among 9 bounded,
deterministically enumerated budgets. Each candidate is rechecked using the
same as-of portfolio risk gate, cash ledger, daily limits and locked lots.
This is a constrained finite-grid decision, NOT a global multi-period optimum.
"""
from __future__ import annotations
import math


def choose_daily_buys(*,budget_cny,deficits_cny,proposal_for_budget,
                      risk_checker,fee_rate=.003,steps=8):
    if (not math.isfinite(float(budget_cny)) or budget_cny<0 or
        not 0<=fee_rate<1 or steps<1):
        raise ValueError("DATA_BLOCKED: invalid amount optimizer inputs")
    if not isinstance(deficits_cny,dict):
        raise ValueError("DATA_BLOCKED: missing target deficits")
    if any(not math.isfinite(float(v)) or v<0 for v in deficits_cny.values()):
        raise ValueError("DATA_BLOCKED: invalid asset deficits")

    baseline=risk_checker({})
    if baseline.get("status")!="PASS":
        return {},{"status":"RISK_BLOCKED","risk_status":baseline.get("status"),
                   "reason":baseline.get("reason"),"missing":baseline.get("missing"),
                   "joint_days":baseline.get("joint_days"),
                   "tested":0,"budget_cny":float(budget_cny)}
    best={};best_gain=0.;best_cash=0.
    accepted=0;rejected=0;unknown=0
    for k in range(1,steps+1):
        cap=float(budget_cny)*k/steps
        buys=proposal_for_budget(cap)
        if (not isinstance(buys,dict) or
            any(not math.isfinite(float(v)) or v<0 for v in buys.values())):
            raise ValueError("DATA_BLOCKED: invalid proposed orders")
        amount=float(sum(buys.values()))
        if amount>cap+.02 or amount>budget_cny+.02:
            raise ValueError("CASH_LEDGER_OVERSPEND: proposal exceeds budget")
        if not amount:
            continue
        risk=risk_checker(buys)
        if risk.get("status")!="PASS":
            rejected+=1
            unknown+=int(risk.get("status")=="RISK_UNKNOWN")
            continue
        accepted+=1
        # The amount score is *tracking-deficit reduction minus fee proxy*;
        # no fabricated market timing or uncalibrated excess-return forecast.
        gain=sum(min(float(v),float(deficits_cny.get(c,0.)))
                 for c,v in buys.items())-fee_rate*amount
        if gain>best_gain+1e-8:
            best=dict(buys);best_gain=gain;best_cash=amount
    return best,{
        "status":"APPROVED_RESEARCH_GRID" if best else "NO_FEASIBLE_POSITIVE_BUY",
        "amount_optimizer":"FINITE_TARGET_DEFICIT_GRID_NOT_GLOBAL_MPC",
        "budget_cny":float(budget_cny),
        "approved_buy_cny":best_cash,
        "estimated_deficit_reduction_net_fee":best_gain,
        "tested":steps,"accepted":accepted,"rejected":rejected,
        "unknown_rejections":unknown,
    }
