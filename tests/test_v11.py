import importlib
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fundkit.v11_cash import DeferredCashLedger, TimingConfig
from fundkit.v11_signals import scenario_risk_gate, score_entries
from fundkit.v11_exit import ArmedTakeProfit, select_fraction


class TestCashLedger(unittest.TestCase):
    def test_deferred_not_double_counted_and_fill_ack(self):
        x=DeferredCashLedger(); cfg=TimingConfig(base_fraction=.5,cash_reserve_weight=0)
        x.begin('2026-01-01',1000)
        plan=x.plan(cfg,available_cash=1000,total_value=100000,risk_pass=False,eligible_signal=False)
        self.assertEqual(plan['approved_buy_cny'],500)
        with self.assertRaisesRegex(RuntimeError,'FILL_ACK_REQUIRED'):
            x.begin('2026-01-02',1000)
        x.ack_execution('2026-01-01',[{'event':'BUY','date':'2026-01-01','amount':300}])
        self.assertAlmostEqual(x.total,700)
        x.begin('2026-01-02',1000)
        plan=x.plan(cfg,available_cash=1700,total_value=100000,risk_pass=True,eligible_signal=True)
        self.assertAlmostEqual(plan['base_budget_cny'],500)
        self.assertAlmostEqual(plan['tactical_release_cny'],350)
        x.ack_execution('2026-01-02',[])
        self.assertEqual(x.total,1700)

    def test_rejected_buys_do_not_reduce_cash(self):
        x=DeferredCashLedger(); x.begin('2026-01-01',200)
        x.ack_execution('2026-01-01',[{'event':'BUY_REJECTED_FORBIDDEN','date':'2026-01-01','amount':200}])
        self.assertEqual(x.total,200)

    def test_redeem_pending_not_added_as_spendable(self):
        x=DeferredCashLedger();x.begin('2026-01-01',100)
        x.ack_execution('2026-01-01',[{'event':'REDEEM','date':'2026-01-01','amount':500}])
        self.assertEqual(x.total,100)

    def test_cash_floor_and_bounds(self):
        x=DeferredCashLedger();x.begin('2026-01-01',1000)
        b=x.plan(TimingConfig(base_fraction=1,cash_reserve_weight=.1),
                 available_cash=1000,total_value=15000,risk_pass=True,eligible_signal=True)
        self.assertEqual(b['approved_buy_cny'],0)
        x.ack_execution('2026-01-01',[])

    def test_budgets_fail_closed(self):
        with self.assertRaises(ValueError):TimingConfig(base_fraction=1.5).validate()
        x=DeferredCashLedger()
        with self.assertRaises(ValueError):x.begin('2026-01-01',float('nan'))


class TestSignals(unittest.TestCase):
    def frame(self,code,vals):
        dates=pd.bdate_range('2024-01-01',periods=len(vals))
        return [{'date':str(d.date()),'total_return_nav':v} for d,v in zip(dates,vals)]

    def test_signal_uses_only_published_data(self):
        vals=np.linspace(1,1.2,255)
        hist=self.frame('E',vals)
        state={'date':pd.bdate_range('2024-01-01',periods=255)[-1],
               'fund_nav_history':{'E':hist}}
        res=score_entries(state,{'E':'equity_index'})
        self.assertTrue(res['E']['eligible'])
        state['date']=pd.bdate_range('2024-01-01',periods=255)[-2]
        with self.assertRaisesRegex(ValueError,'LOOKAHEAD_BLOCKED'):
            score_entries(state,{'E':'equity_index'})

    def test_bond_no_technical_trend_only(self):
        state={'date':pd.Timestamp('2025-01-01'), 'fund_nav_history':{'B':self.frame('B',[1]*255)}}
        s=score_entries(state,{'B':'bond'})
        self.assertFalse(s['B']['eligible'])

    def test_gate_missing_history_fail_closed(self):
        s={'date':pd.Timestamp('2026-01-01'),'value':1000,'cash':100,
           'value_quality':'COMPLETE','holdings':{'F':{'value_pub':900,'shares':90}},
           'fund_nav_history':{}}
        self.assertEqual(scenario_risk_gate(s)['status'],'RISK_UNKNOWN')

    def test_gate_zero_vol_pass_and_shock_fail(self):
        days=pd.bdate_range('2024-01-01',periods=280)
        def scenario(v):
            nav=np.cumprod(1+np.asarray(v,float))
            return {'date':days[-1], 'value':1000,'cash':100,
                    'value_quality':'COMPLETE','holdings':{'F':{'value_pub':900,'shares':90}},
                    'fund_nav_history':{'F':[{'date':str(t.date()),'total_return_nav':float(x)} for t,x in zip(days,nav)]}}
        self.assertEqual(scenario_risk_gate(scenario([0]*280),n_paths=2)['status'],'PASS')
        r=scenario_risk_gate(scenario([0]*5+[-.10]+[0]*274),n_paths=3)
        self.assertEqual(r['status'],'FAIL')


class TestProfitArm(unittest.TestCase):
    def state(self,nav,raw,lotid='B:0',basis=100,redeemable=True):
        return {'date':pd.Timestamp('2026-01-02'), 'nav':{'B':{'total_return_nav':nav,'nav':raw}},
                'lot_states':{'B':[{'lot_id':lotid,'shares':100,'cost_basis_cny':basis,
                                     'cost_basis_known':True,'redeemable':redeemable}]}}
    def cfg(self):
        return {'take_profit_activation_by_category':{'bond':.05},
                'take_profit_trailing_drawdown_by_category':{'bond':.02},
                'take_profit_redeem_fraction':.25,'min_profit_after_fee':.002}
    def test_armed_latch_after_profit_drop_below_threshold(self):
        arms=ArmedTakeProfit()
        a,events=arms.candidates(self.state(1.07,1.07),{'B':'bond'},self.cfg())
        self.assertEqual(a,{})
        self.assertTrue(arms.snapshot()['B:0']['armed'])
        a,events=arms.candidates(self.state(1.03,1.03),{'B':'bond'},self.cfg())
        self.assertAlmostEqual(a['B'],25)

    def test_unknown_basis_never_arm(self):
        arms=ArmedTakeProfit()
        s=self.state(1.09,1.09);s['lot_states']['B'][0]['cost_basis_known']=False
        a,e=arms.candidates(s,{'B':'bond'},self.cfg())
        self.assertFalse(a)
        self.assertTrue(any(x['event']=='TAKE_PROFIT_BLOCKED_UNKNOWN_BASIS' for x in e))

    def test_locked_profit_never_sells(self):
        arms=ArmedTakeProfit()
        s=self.state(1.07,1.07,redeemable=False)
        arms.candidates(s,{'B':'bond'},self.cfg())
        s['nav']['B']['total_return_nav']=1.03;s['nav']['B']['nav']=1.03
        buys,e=arms.candidates(s,{'B':'bond'},self.cfg())
        self.assertFalse(buys)
        self.assertTrue(any(x['event']=='TAKE_PROFIT_BLOCKED_LOCKED' for x in e))

    def test_no_non_fifo_redeem(self):
        arms=ArmedTakeProfit()
        s=self.state(1.07,1.07)
        s['lot_states']['B'].insert(0,{'lot_id':'B:old','shares':10,'cost_basis_cny':10,
                                      'cost_basis_known':True,'redeemable':True})
        arms.candidates(s,{'B':'bond'},self.cfg())
        s['nav']['B']['nav']=1.03;s['nav']['B']['total_return_nav']=1.03
        buys,e=arms.candidates(s,{'B':'bond'},self.cfg())
        self.assertLessEqual(buys.get('B',0),10*.25)

    def test_fraction_failure_retains_baseline(self):
        s={'holdings':{'F':{'shares':100,'redeemable_shares':50,'value_pub':1000}},'value':2000}
        v,reason=select_fraction(s,'F',20,risk_checker=lambda _: {'status':'RISK_UNKNOWN'})
        self.assertEqual(v,20)
        self.assertIn('RETAIN_ORIGINAL',reason)


class TestInstaller(unittest.TestCase):
    def test_idempotent_on_v101_anchors(self):
        from scripts.install_v11_overlay import install
        with tempfile.TemporaryDirectory() as folder:
            base=pathlib.Path(folder)
            for d in ('fundkit','config'): (base/d).mkdir()
            (base/'fundkit/policies.py').write_text('table = {\n        "B5": lambda: make_b5(params),\n    }\n')
            (base/'fundkit/simulator.py').write_text('''state = {
                "unit_nav_history": visible_hist,
            }
            orders = policy(state) if policy else {}
            # 估值与单位净值
''')
            (base/'config/policy_params.json').write_text('{}')
            overlay=pathlib.Path(__file__).resolve().parents[1]
            install(base,overlay);install(base,overlay)
            p=(base/'fundkit/policies.py').read_text()
            s=(base/'fundkit/simulator.py').read_text()
            self.assertEqual(p.count('"B6": lambda'),1)
            self.assertEqual(s.count('policy.on_execution(day,'),1)
            self.assertIn('B6',json.loads((base/'config/policy_params.json').read_text()))


if __name__=='__main__':unittest.main()

class TestPolicyGlue(unittest.TestCase):
    def setUp(self):
        # Isolated policy glue tests: the actual legacy B5 remains unchanged.
        import types
        self.fake=types.ModuleType('fundkit.policies')
        self.fake.make_b5=lambda p: (lambda state: {'subscribe':{'F':999},'events':[]})
        self.fake._plans=lambda:[{'code':'F','category':'bond'}]
        self.fake._weights=lambda state:({'F':0.1},float(state['value']))
        self.fake._water_filling=lambda cash,weights,total,target,limits,allowed: ({allowed[0]:cash}
                             if cash>0 and allowed else {})
        import fundkit
        self.old=getattr(fundkit,'policies',None)
        self.old_mod=sys.modules.get('fundkit.policies')
        fundkit.policies=self.fake;sys.modules['fundkit.policies']=self.fake

    def tearDown(self):
        import fundkit
        if self.old is None: delattr(fundkit,'policies')
        else: fundkit.policies=self.old
        if self.old_mod is None:sys.modules.pop('fundkit.policies',None)
        else:sys.modules['fundkit.policies']=self.old_mod

    def test_no_fill_ack_blocks_next_day_and_base_buy_integrates(self):
        from fundkit.v11_policy import make_b6
        # This test is for the cash-ledger callback; model PASS is explicit.
        # Unknown model evidence must block new B6 buys (separate test below).
        risk_patch = mock.patch('fundkit.v11_policy.scenario_risk_gate',
                                return_value={'status':'PASS','mdd_p95':0.,
                                              'historical_mdd':0.})
        risk_patch.start()
        self.addCleanup(risk_patch.stop)
        params={'target_weights':{'F':.8,'cash':.2},'B5':{'mdd_budget':.04},
                'asset_categories':{'F':'bond'},'B6':{'cash_reserve_weight':0,'base_fraction':.5}}
        policy=make_b6(params)
        d1=pd.Timestamp('2026-01-02')
        s={'date':d1,'external_cash_inflow_today':100,
           'cash':100,'value':1000,'value_quality':'COMPLETE',
           'holdings':{'F':{'shares':100,'value_pub':900,'redeemable_shares':100}},
           'fund_nav_history':{},'lot_states':{}}
        orders=policy(s)
        self.assertEqual(orders['subscribe'],{'F':50})
        with self.assertRaisesRegex(RuntimeError,'FILL_ACK_REQUIRED'):
            policy({**s,'date':d1+pd.Timedelta(days=1)})
        policy.on_execution(d1,[{'event':'BUY','date':str(d1.date()),'amount':50}])
        orders2=policy({**s,'date':d1+pd.Timedelta(days=3),'cash':150})
        self.assertEqual(orders2['subscribe'],{'F':50})
        policy.on_execution(d1+pd.Timedelta(days=3),[])

    def test_no_misleading_buy_when_risk_state_paused(self):
        from fundkit.v11_policy import make_b6
        self.fake.make_b5=lambda p: lambda s:{'subscribe':{},'events':[{'code':'PORTFOLIO','action':'DE_RISK_PAUSE_BUYS'}]}
        policy=make_b6({'target_weights':{'F':.8,'cash':.2},'B5':{'mdd_budget':.04},
                       'asset_categories':{'F':'bond'},'B6':{'cash_reserve_weight':0}})
        s={'date':pd.Timestamp('2026-01-02'),'external_cash_inflow_today':100,
           'cash':100,'value':1000,'value_quality':'COMPLETE','holdings':{},
           'fund_nav_history':{},'lot_states':{}}
        self.assertEqual(policy(s)['subscribe'],{})
        policy.on_execution(s['date'],[])


    def test_unknown_portfolio_risk_blocks_new_base_investment(self):
        from fundkit.v11_policy import make_b6
        policy=make_b6({'target_weights':{'F':.8,'cash':.2},
                        'B5':{'mdd_budget':.04},
                        'asset_categories':{'F':'bond'},
                        'B6':{'cash_reserve_weight':0.,'base_fraction':1.0}})
        day=pd.Timestamp('2026-01-08')
        state={'date':day,'external_cash_inflow_today':100,
               'cash':100,'value':1000,'value_quality':'COMPLETE',
               'holdings':{'F':{'shares':100,'value_pub':900,
                                'redeemable_shares':100}},
               'fund_nav_history':{},'lot_states':{}}
        orders=policy(state)
        self.assertEqual(orders['subscribe'],{})
        self.assertTrue(any(e.get('action')=='B6_BASE_BLOCKED_RISK'
                            for e in orders['events']))
        policy.on_execution(day,[])

class TestReturnForecast(unittest.TestCase):
    def test_no_future_and_no_historical_data_fail_closed(self):
        from fundkit.v11_forecast import conservative_excess_return
        self.assertEqual(conservative_excess_return([],pd.Timestamp('2025-01-01'))['status'],'NO_FORECAST')
        rows=[{'date':'2026-01-01','total_return_nav':1.2},
              {'date':'2026-01-02','total_return_nav':1.3}]
        with self.assertRaisesRegex(ValueError,'LOOKAHEAD_BLOCKED'):
            conservative_excess_return(rows,'2026-01-01')

    def test_flat_price_cannot_generate_positive_excess(self):
        from fundkit.v11_forecast import conservative_excess_return
        ds=pd.bdate_range('2024-01-01',periods=300)
        rows=[{'date':str(d.date()),'total_return_nav':1.0} for d in ds]
        x=conservative_excess_return(rows,ds[-1],annual_cash_return=.01,paths=6)
        self.assertEqual(x['status'],'ESTIMATED_RESEARCH_ONLY')
        self.assertLess(x['excess_q05'],0)


class TestV11ResearchCalibration(unittest.TestCase):
    def test_grid_small_registered_and_cash_comparable(self):
        from fundkit.v11_ablation import build_ablation_rules
        rows = build_ablation_rules()
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(set(r['rule_id'] for r in rows)), len(rows))
        self.assertTrue(all(
            r['params']['B6']['cash_reserve_weight'] == 0.0 for r in rows))
        self.assertEqual({r['params']['B6']['base_fraction'] for r in rows},
                         {0.5, 0.75, 1.0})

    def test_expired_cash_release_requires_asof_risk_pass(self):
        cash = DeferredCashLedger()
        cfg = TimingConfig(base_fraction=.5, release_fraction=0.0,
                           deferred_horizon_days=2, cash_reserve_weight=0)
        cash.begin('2026-01-05', 100.)
        cash.plan(cfg, available_cash=100., total_value=1000.,
                  risk_pass=False, eligible_signal=False)
        cash.ack_execution('2026-01-05', [])
        cash.begin('2026-01-06', 0.)
        cash.plan(cfg, available_cash=100., total_value=1000.,
                  risk_pass=False, eligible_signal=False)
        cash.ack_execution('2026-01-06', [])
        cash.begin('2026-01-07', 0.)
        blocked = cash.plan(cfg, available_cash=100., total_value=1000.,
                            risk_pass=False, eligible_signal=False)
        self.assertTrue(blocked['expired_review_required'])
        self.assertEqual(blocked['approved_buy_cny'], 0.)
        cash.ack_execution('2026-01-07', [])
        cash.begin('2026-01-08', 0.)
        approved = cash.plan(cfg, available_cash=100., total_value=1000.,
                             risk_pass=True, eligible_signal=False)
        self.assertEqual(approved['approved_buy_cny'], 100.)
        self.assertEqual(approved['expired_release_approved_cny'], 100.)
        cash.ack_execution('2026-01-08',
                           [{'date':'2026-01-08','event':'BUY','amount':100.}])
        self.assertAlmostEqual(cash.total, 0.)

    def test_protocols_are_multi_month_and_non_overlapping(self):
        from scripts.run_v11_exploratory import PROTOCOLS
        from fundkit.joint_optimizer import build_rolling_folds
        d = pd.bdate_range('2024-08-16', periods=507)
        halfyear = build_rolling_folds(d, **PROTOCOLS['halfyear'])
        self.assertEqual(halfyear[0]['n_test'], 126)
        quarter = build_rolling_folds(d, **PROTOCOLS['quarterly'])
        self.assertEqual(len(quarter), 2)
        self.assertLess(quarter[0]['test'][1], quarter[1]['test'][0])
        self.assertEqual(quarter[1]['n_test'], 63)
