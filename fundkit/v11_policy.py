"""B6: dynamic contributions, cautious AS-OF timing, lot-aware trailing exits.

Research only. Existing B0-B5 policies are left intact. The policy requires
simulator's fill acknowledgements to keep deferred-cash book consistent.
"""
from copy import deepcopy
from fundkit.v11_cash import DeferredCashLedger, TimingConfig
from fundkit.v11_signals import score_entries, scenario_risk_gate
from fundkit.v11_forecast import conservative_excess_return
from fundkit.v11_exit import ArmedTakeProfit, select_fraction


def make_b6(params):
    from fundkit import policies
    cfg = dict(params.get('B6') or {})
    timing = TimingConfig(
        base_fraction=float(cfg.get('base_fraction',1.0)),
        release_fraction=float(cfg.get('release_fraction',.5)),
        deferred_horizon_days=int(cfg.get('deferred_horizon_days',10)),
        cash_reserve_weight=float(cfg.get('cash_reserve_weight',.05)),
    )
    timing.validate()
    ledger=DeferredCashLedger()
    arms=ArmedTakeProfit()
    categories=dict(params.get('asset_categories') or {})
    if not categories:
        try:
            categories = {str(p['code']):str(p.get('category','other')) for p in policies._plans()}
        except (FileNotFoundError,KeyError,ValueError):
            categories={}
    target={k:float(v) for k,v in (params.get('target_weights') or {}).items()
            if k!='cash' and not k.startswith('_')}
    allowed=set(target)-set(params.get('disabled_new_buy_codes',[]))
    limits=dict(params.get('daily_limits') or {})
    risk_budget=float((params.get('B5') or {}).get('mdd_budget',.04))

    # Keep B5's pre-existing portfolio/fund STOP logic, disable the aggregate
    # TAKE PROFIT component; B6 performs per-lot armed trailing profit instead.
    p=deepcopy(params)
    p.setdefault('B5',{})
    # In a research-only simulator, execute hypothetical redemptions so their
    # fees, locks and realized effects can actually be measured.
    p['B5']['allow_redeem'] = bool(cfg.get('simulate_redeem',True))
    p['B5']['take_profit_activation_by_code']={c:0.0 for c in target}
    p['B5']['take_profit_activation_by_category']={x:0.0 for x in ('bond','gold','equity_index','other')}
    exit_mode=str(cfg.get('exit_mode','both'))
    if exit_mode not in ('none','take_profit_only','stop_only','both'):
        raise ValueError('B6: invalid exit_mode')
    stop_enabled=exit_mode in ('stop_only','both')
    tp_enabled=exit_mode in ('take_profit_only','both')
    legacy_stop=policies.make_b5(p) if stop_enabled else policies.make_b2(p)

    def policy(state):
        orders=legacy_stop(state) or {}
        orders.setdefault('subscribe',{})
        orders.setdefault('events',[])
        # Exact effective external deposit is required. No guessing from cash delta:
        # dividends, receipts and withdrawals may also change available cash.
        if 'external_cash_inflow_today' not in state:
            orders['subscribe']={}
            orders['events'].append({'code':'PORTFOLIO','action':'B6_DATA_BLOCKED_MISSING_INFLOW'})
            return orders
        ledger.begin(state['date'],state['external_cash_inflow_today'])
        if state.get('value_quality')!='COMPLETE':
            orders['subscribe']={}
            orders['events'].append({'code':'PORTFOLIO','action':'B6_DATA_BLOCKED_MISSING_MARK'})
            return orders
        # Explicit lot ids/fees/locks required for profit-taking; otherwise the
        # original B5 stop-loss remains active and TP is fail-closed.
        if tp_enabled and state.get('lot_states'):
            take,detail=arms.candidates(state,categories,cfg)
            for event in detail:
                orders['events'].append({'action':event.pop('event'),**event})
            for code,shares in take.items():
                if code not in (orders.get('redeem') or {}):
                    if p['B5'].get('allow_redeem',False):
                        orders.setdefault('redeem',{})[code]=shares
                    else:
                        orders.setdefault('redeem_review',{})[code]=shares
        elif tp_enabled:
            orders['events'].append({'code':'PORTFOLIO','action':'B6_TP_BLOCKED_NO_LOT_STATES'})
        # If a stop or profit order is placed, don't buy the same fund back.
        excludes=set(orders.get('redeem',{}))|set(orders.get('redeem_review',{}))
        risk_seed=int(cfg.get('risk_seed',42))
        def checker(redemptions=None,buys=None):
            return scenario_risk_gate(state,proposed_buys=buys,
                                      proposed_redemptions=redemptions,
                                      budget=risk_budget,
                                      n_paths=int(cfg.get('risk_paths',32)),seed=risk_seed)
        # Allowed to maintain legacy STOP order if scenario model is unavailable,
        # but never invent a risk improvement or override locked lots.
        for code,original in list((orders.get('redeem') or {}).items()):
            if not cfg.get('optimize_exit_fraction',True):
                continue
            selected,why=select_fraction(
                state,code,original,
                target_weight=target.get(code,0),
                fee_rate=float(cfg.get('redeem_fee_proxy',.003)),
                max_drawdown=risk_budget,
                risk_checker=(lambda red, _code=code:checker(redemptions=red)))
            if selected>0:
                orders['redeem'][code]=selected
            else:
                del orders['redeem'][code]
            orders['events'].append({'code':code,'action':'B6_EXIT_FRACTION',
                                     'shares':selected,'reason':why})
        excludes |= set(orders.get('redeem',{}))
        risk_pre=checker()
        risk_ok=risk_pre['status']=='PASS'
        orders['events'].append({'code':'PORTFOLIO','action':'B6_PRE_RISK',
                                 'status':risk_pre['status'],'mdd_p95':risk_pre.get('mdd_p95')})
        signal=score_entries(state,categories,annual_cash_return=float(cfg.get('annual_cash_return',0)))
        candidates=[]
        # Signal must have BOTH positive published trend and conservative
        # same-history excess-return score. Bond trend alone is never enough.
        for code in sorted(allowed) if cfg.get('entry_timing_enabled',True) else []:
            if not signal.get(code,{}).get('eligible'):
                continue
            forecast=conservative_excess_return(
                (state.get('fund_nav_history') or {}).get(code),state['date'],
                annual_cash_return=float(cfg.get('annual_cash_return',0)),
                seed=risk_seed+sum(ord(ch) for ch in code))
            orders['events'].append({'code':code,'action':'B6_TIMING_FORECAST',
                                     'status':forecast['status'],
                                     'excess_q05':forecast.get('excess_q05')})
            if (forecast['status']=='ESTIMATED_RESEARCH_ONLY' and
                    forecast['excess_q05']>0):
                candidates.append(code)
        # No speculative tactical purchases during a risk warning or if any
        # nonzero position lacks known published NAV or scenario coverage.
        cash_plan=ledger.plan(timing,available_cash=state['cash'],total_value=state['value'],
                              risk_pass=risk_ok,eligible_signal=bool(candidates))
        orders['events'].append({'code':'PORTFOLIO','action':'B6_CASH_PLAN',**cash_plan})
        # B5 may have entered PAUSE_BUY. Never bypass risk-state restrictions.
        if any(ev.get('action') in ('DE_RISK_PAUSE_BUYS','RISK_DATA_BLOCKED')
               for ev in orders['events']):
            orders['subscribe']={}
            return orders
        if not target:
            orders['subscribe']={}
            orders['events'].append({'code':'PORTFOLIO','action':'B6_BLOCKED_NO_TARGET_WEIGHTS'})
            return orders
        weights,total=policies._weights(state)
        if total<=0:
            orders['subscribe']={}
            return orders
        base_allowed=sorted(allowed-excludes)
        if cfg.get('amount_mode')=='risk_budgeted':
            # Independent A7 experimental rule. Pick the trade amount only
            # after the ledger has approved it, and check EACH candidate with
            # the identical as-of exposure risk gate. Never bypass B5 pauses.
            from fundkit.v13_execution import choose_daily_buys
            paused={ev.get('code') for ev in orders['events']
                    if ev.get('action') in ('PAUSE_NEW_BUY','FUND_RISK_DATA_BLOCKED')}
            eligible=sorted(set(base_allowed)-paused)
            gaps={c:max(0.,target.get(c,0.)-weights.get(c,0.))*total
                  for c in eligible}
            buys,decision=choose_daily_buys(
                budget_cny=cash_plan['approved_buy_cny'],
                deficits_cny=gaps,
                proposal_for_budget=lambda budget: policies._water_filling(
                    budget,weights,total,target,limits,eligible),
                risk_checker=lambda proposed:checker(buys=proposed),
                fee_rate=float(cfg.get('subscription_fee_proxy',.003)),
                steps=int(cfg.get('amount_grid_steps',8)))
            orders['events'].append({
                'code':'PORTFOLIO','action':'V13_AMOUNT_DECISION',**decision})
            orders['subscribe']=buys
            return orders
        base_budget=cash_plan['base_budget_cny']
        if risk_pre['status']!='PASS':
            # Fail closed for additional risk when as-of scenario evidence is unknown.
            # Keep B2 as the explicit unguarded reference comparator.
            base_budget=0.
            orders['events'].append({'code':'PORTFOLIO','action':'B6_BASE_BLOCKED_RISK',
                                     'status':risk_pre['status']})
        buys=policies._water_filling(base_budget,weights,total,target,limits,base_allowed)
        # Use the *remaining* deficit and daily buy-limit, not an independent
        # budget that could double-subscribe the same target.
        extra_cash=cash_plan['approved_buy_cny']-base_budget
        if extra_cash>1e-8 and risk_ok and candidates:
            adjusted=dict(weights)
            for code,amt in buys.items():
                adjusted[code]=adjusted.get(code,0)+amt/total
            residual_limits={c:max(0.,float(limits.get(c,float('inf')))-buys.get(c,0))
                             for c in candidates}
            tactical=policies._water_filling(extra_cash,adjusted,total,target,
                                             residual_limits,sorted(set(candidates)-excludes))
            for code,amount in tactical.items():
                buys[code]=buys.get(code,0)+amount
        # Post-order risk gate: reject tactical if it increases estimated MDD
        # beyond 4%. When risk is unknown, base is research baseline only.
        if buys and checker(buys=buys)['status']!='PASS':
            buys={}
            orders['events'].append({'code':'PORTFOLIO','action':'B6_BUY_BLOCKED_POST_RISK'})
        # B5 may have paused individual funds. Do not revive those buys.
        paused={ev.get('code') for ev in orders['events']
                if ev.get('action') in ('PAUSE_NEW_BUY','FUND_RISK_DATA_BLOCKED')}
        buys={c:a for c,a in buys.items() if c not in paused and c not in excludes}
        orders['subscribe']=buys
        return orders

    def on_execution(day,events):
        return ledger.ack_execution(day,events)

    policy.on_execution=on_execution
    policy.snapshot=lambda:{'cash_ledger':ledger.snapshot(),'take_profit_arms':arms.snapshot(),
                           'research_only':True}
    policy.policy_family='B6'
    return policy
