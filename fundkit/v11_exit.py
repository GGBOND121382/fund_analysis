"""B6 lot-aware trailing take profit and constrained exit-size selection."""
import math
import pandas as pd


class ArmedTakeProfit:
    """Per-lot armed state. Missing cost basis or unknown dates never arm.

    Because the existing simulator redeems FIFO by fund, profit orders are
    constrained to the *first redeemable FIFO lot*, preventing mismatch.
    """
    def __init__(self):
        self._arms = {}

    def candidates(self, state, categories, config):
        candidates = {}
        info = []
        date = pd.Timestamp(state['date'])
        for code, lots in (state.get('lot_states') or {}).items():
            if not lots:
                continue
            navrec = (state.get('nav') or {}).get(code)
            if not navrec:
                continue
            nav = float(navrec.get('total_return_nav') or navrec.get('nav') or 0.)
            raw = float(navrec.get('nav') or 0.)
            if nav <= 0 or raw <= 0:
                continue
            category = categories.get(code,'other')
            activation = float(config.get('take_profit_activation_by_category',{}).get(category,.0))
            trailing = float(config.get('take_profit_trailing_drawdown_by_category',{}).get(category,.0))
            min_profit = float(config.get('min_profit_after_fee',0.))
            ratio = float(config.get('take_profit_redeem_fraction',.25))
            if activation <= 0 or trailing <= 0 or ratio <= 0:
                continue
            valid_fifo = [lot for lot in lots if float(lot.get('shares',0))>1e-10 and lot.get('redeemable')]
            oldest = valid_fifo[0] if valid_fifo else None
            for lot in lots:
                key = str(lot['lot_id'])
                shares=float(lot.get('shares',0))
                if shares <= 1e-10:
                    self._arms.pop(key,None)
                    continue
                if not lot.get('cost_basis_known') or lot.get('cost_basis_cny') is None:
                    info.append({'code':code,'event':'TAKE_PROFIT_BLOCKED_UNKNOWN_BASIS','lot_id':key})
                    continue
                basis = float(lot['cost_basis_cny'])
                if basis <= 0:
                    continue
                ret = shares * raw / basis -1
                z=self._arms.setdefault(key,{'armed':False,'peak':nav})
                if ret >= activation and not z['armed']:
                    z['armed']=True
                    z['peak']=nav
                    info.append({'code':code,'event':'TAKE_PROFIT_ARMED','lot_id':key})
                if not z['armed']:
                    continue
                z['peak']=max(z['peak'],nav)
                dd = max(0.,1-nav/z['peak'])
                if dd < trailing or ret <= min_profit:
                    continue
                if not lot.get('redeemable'):
                    info.append({'code':code,'event':'TAKE_PROFIT_BLOCKED_LOCKED','lot_id':key})
                    continue
                # Existing FIFO simulator cannot target a later lot; fail closed.
                if oldest is None or oldest['lot_id'] != lot['lot_id']:
                    info.append({'code':code,'event':'TAKE_PROFIT_BLOCKED_NON_FIFO','lot_id':key})
                    continue
                shares_to_sell = min(shares,float(oldest['shares']))*ratio
                if shares_to_sell>1e-8:
                    candidates[code]=max(candidates.get(code,0.),shares_to_sell)
                    info.append({'code':code,'event':'TAKE_PROFIT_CANDIDATE','lot_id':key,
                                 'trailing_drawdown':dd,'profit_rate':ret,'shares':shares_to_sell})
        return candidates,info

    def snapshot(self):
        return {k:dict(v) for k,v in self._arms.items()}


def select_fraction(state, code, requested_shares, *, fractions=(.25,.50,1.0),
                    target_weight=0., fee_rate=.003, max_drawdown=.04,
                    risk_checker=None):
    """Select redeem fraction of available shares with risk-first lexicographic sorting.

    If no quantitative risk model is available, return original requested amount;
    do not claim optimality. Never sell locked shares or use unknown NAV.
    """
    h=(state.get('holdings') or {}).get(code) or {}
    eligible=float(h.get('redeemable_shares') or 0.)
    shares=float(h.get('shares') or 0.)
    published_value=h.get('value_pub')
    if eligible<=0 or shares<=0 or published_value is None:
        return 0., 'REDEEM_BLOCKED_NO_ELIGIBLE_LOTS'
    requested=min(max(0.,float(requested_shares)),eligible)
    if risk_checker is None:
        return requested,'RETAIN_ORIGINAL_RISK_MODEL_UNAVAILABLE'
    unit=float(published_value)/shares
    total=float(state.get('value') or 0.)
    if total<=0 or unit<=0:
        return requested,'RETAIN_ORIGINAL_INVALID_PORTFOLIO_VALUE'
    opts=[]
    for frac in fractions:
        amount=min(eligible,eligible*float(frac))
        if amount+1e-8<requested:
            continue
        risk = risk_checker({code:amount})
        if risk.get('status') not in ('PASS','FAIL'):
            continue
        w_new=max(0.,float(published_value)-amount*unit)/total
        cost=amount*unit*float(fee_rate)/total
        dev=(w_new-float(target_weight))**2
        # Risk first; trading cost and target deviation only break risk ties.
        opts.append((risk['status']!='PASS',max(risk['mdd_p95'],risk['historical_mdd']),
                     cost+.05*dev,amount))
    if not opts:
        return requested,'RETAIN_ORIGINAL_NO_VALID_POST_EXIT_SCENARIO'
    best=min(opts)
    return float(best[-1]),'SCENARIO_EXIT_FRACTION_SELECTED'
