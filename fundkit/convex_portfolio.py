"""Deterministic continuous portfolio LP, research only.

For fixed buy-and-hold weights w, scenario wealth is V[t] = G[t] @ w.
MDD <= d iff V[t] >= (1-d)*V[s], for all s<=t. These are affine cuts.
We enforce hard historical MDD and empirical scenario CVaR of the signed
drawdown violation (a conservative chance-constraint surrogate).
Uses HiGHS and exact linear-constraint separation, NOT random weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class PortfolioLPConfig:
    drawdown_budget: float = 0.04
    cvar_alpha: float = 0.95
    turnover_cost: float = 0.003
    cash_annual_return: float = 0.
    min_risky_weight: float = 0.
    min_cash_weight: float = 0.
    max_cuts: int = 100
    tolerance: float = 1e-8

    def validate(self):
        if not 0 < self.drawdown_budget < 1:
            raise ValueError('invalid drawdown budget')
        if not 0 < self.cvar_alpha < 1:
            raise ValueError('invalid CVaR level')
        if not 0 <= self.min_risky_weight <= 1 or not 0 <= self.min_cash_weight <= 1:
            raise ValueError('invalid cash/risky bounds')
        if self.min_risky_weight+self.min_cash_weight>1+1e-10:
            raise ValueError('incompatible cash/risky exposure minimums')
        if not 0 <= self.turnover_cost < 1:
            raise ValueError('invalid turnover cost')
        if self.max_cuts < 1 or not 0 < self.tolerance <= 1e-3:
            raise ValueError('invalid cuts/tolerance')
        if self.cash_annual_return <= -1:
            raise ValueError('invalid cash return')


def wealth_coefficients(returns, cash_annual_return):
    """[scenarios, dates, funds] -> [scenarios, dates+1, funds+cash]."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 3 or min(r.shape) < 1 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('DATA_BLOCKED: invalid joint scenario returns')
    s, t, n = r.shape
    risky = np.concatenate([np.ones((s, 1, n)), np.cumprod(1+r, axis=1)], axis=1)
    cash = (1+cash_annual_return)**(np.arange(t+1)/252.)
    return np.concatenate((risky, np.broadcast_to(cash[None, :, None], (s,t+1,1))), axis=2)


def worst_violation(g, weights, drawdown):
    """Return max_{s<=t} ((1-d)V_s-V_t) and witness (s,t) in O(T)."""
    wealth = g @ weights
    peak = float(wealth[0]); peak_index = 0
    worst = -np.inf; witness = (0,0)
    for t, value in enumerate(wealth):
        if value > peak:
            peak, peak_index = float(value), t
        violation = (1-drawdown)*peak-float(value)
        if violation > worst:
            worst, witness = violation, (peak_index,t)
    return float(worst), witness


def realized_mdd(g, w):
    v = g @ w
    return float(np.max(1-v/np.maximum.accumulate(v)))


def empirical_cvar(values, alpha):
    """Exact finite-sample upper CVaR, not interpolated quantile."""
    x=np.sort(np.asarray(values,float))
    return float(np.min(x + np.maximum(x[None,:]-x[:,None],0.).mean(axis=1)/(1-alpha)))


def solve_continuous_weights(
    historical_returns, scenario_returns, annual_lower_returns,
    *, current_weights=None, fund_lower=None, fund_upper=None,
    config=None,
):
    """Global LP under finite buy-and-hold path model; never invent a solution.

    Scenario CVaR is applied to signed wealth deficits; CVaR<=0 implies
    empirical VaR(deficit)<=0, sufficient for <=4% MDD on >=95% paths.
    Results are NOT a future drawdown guarantee or a dynamic-cashflow optimum.
    """
    cfg = config or PortfolioLPConfig()
    cfg.validate()
    h=np.asarray(historical_returns,float)
    paths=np.asarray(scenario_returns,float)
    mu=np.asarray(annual_lower_returns,float)
    if h.ndim!=2 or paths.ndim!=3 or paths.shape[2]!=h.shape[1] or mu.shape!=(h.shape[1],):
        raise ValueError('DATA_BLOCKED: incompatible returns/scenarios/forecasts')
    if not np.isfinite(mu).all():
        raise ValueError('DATA_BLOCKED: invalid return estimates')
    n=len(mu); m=len(paths)
    hist=wealth_coefficients(h[None],cfg.cash_annual_return)[0]
    scenarios=wealth_coefficients(paths,cfg.cash_annual_return)
    lo=np.zeros(n) if fund_lower is None else np.asarray(fund_lower,float)
    hi=np.ones(n) if fund_upper is None else np.asarray(fund_upper,float)
    current=np.r_[np.zeros(n),1.] if current_weights is None else np.asarray(current_weights,float)
    if (lo.shape!=(n,) or hi.shape!=(n,) or current.shape!=(n+1,) or
        not np.isfinite(lo).all() or not np.isfinite(hi).all() or
        not np.isfinite(current).all() or (lo<0).any() or (hi<lo).any() or
        (hi>1).any() or (current<0).any() or abs(current.sum()-1)>1e-6):
        raise ValueError('DATA_BLOCKED: invalid portfolio bounds/current weights')
    if (lo.sum()+cfg.min_cash_weight>1+cfg.tolerance or
        hi.sum()<cfg.min_risky_weight-cfg.tolerance):
        return {'status':'INFEASIBLE','reason':'ASSET_BOUNDS_INFEASIBLE','weights':None}

    # Variables: w_funds,w_cash, |delta w_funds|, z_scenario, eta, u_scenario
    w_end=n+1; d_start=w_end; z_start=d_start+n; eta=z_start+m
    u_start=eta+1; nvar=u_start+m
    objective=np.zeros(nvar)
    objective[:n]=-mu; objective[n]=-cfg.cash_annual_return
    objective[d_start:z_start]=cfg.turnover_cost
    bounds=[(float(a),float(b)) for a,b in zip(lo,hi)]+[(cfg.min_cash_weight,1.)]
    bounds += [(0.,None)]*n + [(None,None)]*m + [(None,None)]+[(0.,None)]*m
    aeq=np.zeros((1,nvar)); aeq[0,:w_end]=1.

    structural=[]
    for i in range(n):
        structural += [((i,d_start+i),(1.,-1.),float(current[i])),
                       ((i,d_start+i),(-1.,-1.),-float(current[i]))]
    structural.append((tuple(range(n)),tuple([-1.]*n),-cfg.min_risky_weight))
    for j in range(m):
        structural.append(((z_start+j,eta,u_start+j),(1.,-1.,-1.),0.))
    structural.append((tuple([eta]+list(range(u_start,nvar))),
                       tuple([1.]+[1./((1-cfg.cvar_alpha)*m)]*m),0.))
    cuts=[]; seen=set()
    def add_cut(scenario_idx, peak, trough):
        key=(scenario_idx,peak,trough)
        if key in seen:
            return False
        seen.add(key)
        g=hist if scenario_idx==-1 else scenarios[scenario_idx]
        coef=(1-cfg.drawdown_budget)*g[peak]-g[trough]
        indices=tuple(range(w_end))
        values=tuple(float(a) for a in coef)
        if scenario_idx!=-1:
            indices+=(z_start+scenario_idx,)
            values+=(-1.,)
        cuts.append((indices,values,0.))
        return True
    add_cut(-1,0,0)
    for j in range(m):
        add_cut(j,0,0)

    def assemble():
        rows=structural+cuts
        rid=[];cid=[];vals=[];rhs=np.empty(len(rows))
        for k,(indices,coefs,right) in enumerate(rows):
            rhs[k]=right
            for idx,val in zip(indices,coefs):
                if val!=0:
                    rid.append(k);cid.append(idx);vals.append(val)
        return coo_matrix((vals,(rid,cid)),shape=(len(rows),nvar)).tocsr(),rhs

    for iteration in range(1,cfg.max_cuts+1):
        A,b=assemble()
        sol=linprog(objective,A_ub=A,b_ub=b,A_eq=aeq,b_eq=[1.],bounds=bounds,
                    method='highs',options={'primal_feasibility_tolerance':1e-8,
                                           'dual_feasibility_tolerance':1e-8})
        if not sol.success:
            return {'status':'INFEASIBLE' if sol.status==2 else 'SOLVER_FAILED',
                    'reason':str(sol.message),'weights':None,
                    'solver_status':int(sol.status),'iterations':iteration}
        w=sol.x[:w_end]
        hard,witness=worst_violation(hist,w,cfg.drawdown_budget)
        added=int(hard>cfg.tolerance and add_cut(-1,*witness))
        for j in range(m):
            violation,pair=worst_violation(scenarios[j],w,cfg.drawdown_budget)
            if violation>sol.x[z_start+j]+cfg.tolerance and add_cut(j,*pair):
                added+=1
        if not added:
            violations=np.array([worst_violation(g,w,cfg.drawdown_budget)[0]
                                 for g in scenarios])
            exact_cvar=empirical_cvar(violations,cfg.cvar_alpha)
            epigraph=float(np.max(violations-sol.x[z_start:eta]))
            hist_mdd=realized_mdd(hist,w)
            primal=float(np.max(A@sol.x-b))
            if (max(hard,exact_cvar,epigraph,primal)>max(cfg.tolerance,3e-7) or
                abs(w.sum()-1)>1e-7):
                return {'status':'NUMERICAL_REJECTED','reason':'failed primal/risk check',
                        'historical_mdd':hist_mdd,'scenario_cvar_violation':exact_cvar,
                        'weights':None,'iterations':iteration}
            return {
                'status':'OPTIMAL','weights':w,
                'objective_robust_annual_net':float(-sol.fun),
                'robust_return_before_cost':float(mu@w[:n]+cfg.cash_annual_return*w[-1]),
                'estimated_turnover_cost':float(cfg.turnover_cost*np.abs(w[:n]-current[:n]).sum()),
                'historical_mdd':hist_mdd,
                'scenario_mdd_quantile':float(np.quantile(
                    [realized_mdd(g,w) for g in scenarios],cfg.cvar_alpha)),
                'scenario_cvar_violation':exact_cvar,'scenario_count':m,
                'iteration_count':iteration,'cut_count':len(cuts),
                'solver':'scipy.optimize.linprog/HiGHS',
                'model':'FINITE_SCENARIO_BUY_AND_HOLD_CVAR'}
    return {'status':'CUT_LIMIT','reason':'constraint generation did not converge',
            'weights':None,'iterations':cfg.max_cuts}
