"""Research-only continuous robust SOCP portfolio optimizer.

The finite-scenario *buy-and-hold* problem is convex. Historical 4% MDD
is enforced by separating affine cuts. Scenario signed-gap CVaR<=0 is a
conservative empirical chance-constraint surrogate. Mean-forecast uncertainty
and portfolio covariance risk are both penalized explicitly.
No random weight sampling and no global optimality claim for dynamic trading.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from fundkit.convex_portfolio import wealth_coefficients, worst_violation, realized_mdd, empirical_cvar


@dataclass(frozen=True)
class RobustSOCPConfig:
    drawdown_budget: float = .04
    cvar_alpha: float = .95
    uncertainty_penalty: float = 1.
    volatility_penalty: float = 1.
    turnover_cost: float = .003
    cash_return_annual: float = 0.
    min_cash_weight: float = 0.
    min_risky_weight: float = 0.
    tolerance: float = 2e-6
    max_rounds: int = 60

    def validate(self):
        if (not 0<self.drawdown_budget<1 or not 0<self.cvar_alpha<1 or
            min(self.uncertainty_penalty,self.volatility_penalty,
                self.turnover_cost,self.min_cash_weight,self.min_risky_weight)<0 or
            self.min_cash_weight+self.min_risky_weight>1+1e-9 or
            self.max_rounds<1):
            raise ValueError("invalid robust SOCP settings")


def _psd(x,n):
    a=np.asarray(x,float)
    if a.shape!=(n,n) or not np.isfinite(a).all():
        raise ValueError("DATA_BLOCKED: invalid covariance matrix")
    a=(a+a.T)/2.
    e,v=np.linalg.eigh(a)
    if e.min() < -1e-7:
        raise ValueError("DATA_BLOCKED: non-PSD covariance")
    return (v*np.maximum(e,0.))@v.T, (v*np.sqrt(np.maximum(e,0.)))@v.T


def solve_robust_socp(historical_returns, scenario_returns,
                      mean_annual, covariance_annual, mean_error_covariance,
                      *, fund_lower=None, fund_upper=None, current_weights=None,
                      config=None, group_upper=(), hard_stress_returns=None, directional_fees=None):
    import cvxpy as cp
    cfg=config or RobustSOCPConfig()
    cfg.validate()
    r=np.asarray(historical_returns,float)
    paths=np.asarray(scenario_returns,float)
    mu=np.asarray(mean_annual,float)
    if (r.ndim!=2 or paths.ndim!=3 or paths.shape[1]<2 or
        paths.shape[2]!=r.shape[1] or mu.shape!=(r.shape[1],) or
        not np.isfinite(r).all() or not np.isfinite(paths).all() or
        not np.isfinite(mu).all() or (r<=-1).any() or (paths<=-1).any()):
        raise ValueError("DATA_BLOCKED: invalid input returns or forecasts")
    n=len(mu);m=len(paths)
    sigma,_=_psd(covariance_annual,n)
    omega,root=_psd(mean_error_covariance,n)
    lo=np.zeros(n) if fund_lower is None else np.asarray(fund_lower,float)
    hi=np.ones(n) if fund_upper is None else np.asarray(fund_upper,float)
    current=np.r_[np.zeros(n),1.] if current_weights is None else np.asarray(current_weights,float)
    if (lo.shape!=(n,) or hi.shape!=(n,) or current.shape!=(n+1,) or
        not np.isfinite(lo).all() or not np.isfinite(hi).all() or
        not np.isfinite(current).all() or (lo<0).any() or (hi<lo-1e-10).any() or
        (hi>1+1e-10).any() or (current<0).any() or abs(current.sum()-1)>1e-6):
        raise ValueError("DATA_BLOCKED: invalid asset bounds")
    if lo.sum()+cfg.min_cash_weight>1+cfg.tolerance or hi.sum()<cfg.min_risky_weight-cfg.tolerance:
        return {"status":"INFEASIBLE","reason":"ASSET_BOUNDS","weights":None}

    h=wealth_coefficients(r[None],cfg.cash_return_annual)[0]
    scenarios=wealth_coefficients(paths,cfg.cash_return_annual)
    group_upper = tuple((tuple(indices), float(cap)) for indices, cap in group_upper)
    if any(not indices or len(set(indices)) != len(indices) or min(indices) < 0
           or max(indices) >= n or not 0 <= cap <= 1 for indices, cap in group_upper):
        raise ValueError("invalid group upper bounds")
    hard = (wealth_coefficients(hard_stress_returns,cfg.cash_return_annual)
            if hard_stress_returns is not None else ())
    w=cp.Variable(n+1)
    z=cp.Variable(m)
    eta=cp.Variable()
    u=cp.Variable(m,nonneg=True)
    cons=[cp.sum(w)==1, w[:n]>=lo,w[:n]<=hi,w[n]>=cfg.min_cash_weight,
          w[n]<=1,cp.sum(w[:n])>=cfg.min_risky_weight,
          u>=z-eta,eta+cp.sum(u)/(m*(1-cfg.cvar_alpha))<=0]
    cons += [cp.sum(w[list(indices)]) <= cap for indices, cap in group_upper]
    # Trivial peak at t=0 supplies a valid epigraph lower bound.
    cons += [z>=-cfg.drawdown_budget]
    lower=mu@w[:n] + cfg.cash_return_annual*w[n]
    objective=lower-cfg.uncertainty_penalty*cp.norm(root@w[:n],2)
    objective-=.5*cfg.volatility_penalty*cp.quad_form(w[:n],cp.psd_wrap(sigma))
    delta = w[:n] - current[:n]
    if directional_fees is None:
        reference_cost = cfg.turnover_cost * cp.norm1(delta)
    else:
        buy_fee, sell_fee = directional_fees
        if any(not np.isfinite(x) or not 0 <= x < 1 for x in (buy_fee, sell_fee)):
            raise ValueError("invalid directional fees")
        reference_cost = 0.
        if buy_fee:
            reference_cost += buy_fee * cp.sum(cp.pos(delta))
        if sell_fee:
            reference_cost += sell_fee * cp.sum(cp.pos(-delta))
    objective -= reference_cost
    prob=cp.Problem(cp.Maximize(objective),cons)
    added=set()
    for it in range(1,cfg.max_rounds+1):
        try:
            prob.solve(solver=cp.CLARABEL,verbose=False,
                       max_iter=300,tol_gap_abs=1e-8,
                       tol_feas=1e-8,tol_gap_rel=1e-8)
        except (cp.SolverError,ValueError) as exc:
            return {"status":"SOLVER_FAILED","reason":str(exc),
                    "weights":None,"rounds":it}
        if prob.status not in ("optimal","optimal_inaccurate") or w.value is None:
            return {"status":"INFEASIBLE" if prob.status=="infeasible" else "SOLVER_FAILED",
                    "reason":prob.status,"weights":None,"rounds":it}
        ww=np.asarray(w.value,float)
        worst,pair=worst_violation(h,ww,cfg.drawdown_budget)
        new=[]
        if worst>cfg.tolerance and (-1,*pair) not in added:
            gap=(1-cfg.drawdown_budget)*h[pair[0]]-h[pair[1]]
            new.append(gap@w<=0.)
            added.add((-1,*pair))
        for k,g in enumerate(hard):
            violation,pair=worst_violation(g,ww,cfg.drawdown_budget)
            key=(-2-k,*pair)
            if violation>cfg.tolerance and key not in added:
                gap=(1-cfg.drawdown_budget)*g[pair[0]]-g[pair[1]]
                new.append(gap@w<=0.)
                added.add(key)
        for j,g in enumerate(scenarios):
            violation,pair=worst_violation(g,ww,cfg.drawdown_budget)
            if violation>z.value[j]+cfg.tolerance and (j,*pair) not in added:
                gap=(1-cfg.drawdown_budget)*g[pair[0]]-g[pair[1]]
                new.append(gap@w<=z[j])
                added.add((j,*pair))
        if not new:
            violations=np.array([worst_violation(g,ww,cfg.drawdown_budget)[0]
                                  for g in scenarios])
            cvar=empirical_cvar(violations,cfg.cvar_alpha)
            hist_mdd=realized_mdd(h,ww)
            hard_dd=[realized_mdd(g,ww) for g in hard]
            if (worst>cfg.tolerance or cvar>cfg.tolerance or
                any(worst_violation(g,ww,cfg.drawdown_budget)[0]>cfg.tolerance for g in hard) or
                any(ww[list(idx)].sum()>cap+cfg.tolerance for idx,cap in group_upper) or
                np.max(violations-np.asarray(z.value))>cfg.tolerance or
                ww.sum()<1-1e-5 or ww.sum()>1+1e-5 or
                np.min(ww[:n]-lo)<-1e-5 or np.max(ww[:n]-hi)>1e-5):
                return {"status":"NUMERICAL_REJECTED","reason":"independent path/risk certificate failed",
                        "weights":None,"rounds":it}
            change = ww[:n] - current[:n]
            turnover_cost = (cfg.turnover_cost * np.abs(change).sum()
                             if directional_fees is None else
                             buy_fee * np.maximum(change, 0.).sum()
                             + sell_fee * np.maximum(-change, 0.).sum())
            return {
                "status":"OPTIMAL","weights":ww,
                "model_score":float(prob.value),
                "estimated_mean_annual":float(mu@ww[:n]+cfg.cash_return_annual*ww[n]),
                "forecast_uncertainty_penalty":float(cfg.uncertainty_penalty*np.linalg.norm(root@ww[:n])),
                "variance_penalty":float(.5*cfg.volatility_penalty*ww[:n]@sigma@ww[:n]),
                "turnover_cost":float(turnover_cost),
                "historical_mdd":hist_mdd,
                "hard_stress_mdd":hard_dd,
                "scenario_mdd_p95":float(np.quantile([realized_mdd(g,ww) for g in scenarios],cfg.cvar_alpha)),
                "scenario_cvar_signed_gap":float(cvar),
                "cut_count":len(added),"rounds":it,
                "solver":"CVXPY/CLARABEL",
                "model":"ONE_YEAR_ROBUST_SOCP_BUY_AND_HOLD",
            }
        cons.extend(new)
        prob=cp.Problem(cp.Maximize(objective),cons)
    return {"status":"CUT_LIMIT","weights":None,
            "reason":"maximum conic constraint generation iterations exceeded"}
