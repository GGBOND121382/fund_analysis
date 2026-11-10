"""修复验收用的手工可验证计算；无数据源、无联网依赖。"""
import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from fundkit import metrics, policies, style
from fundkit.simulator import Lot, SimConfig
from tests.test_simulator import DAYS, make_sim
from tests.test_policies import make_state, PARAMS


class AccountingIndependentTests(unittest.TestCase):
    def test_price_doubles_before_new_money_joined(self):
        # 期初100份，首日投100元，次日先翻倍后再投100元：净值2.0，单位数250。
        sim = make_sim({'004400': [1, 2] + [2] * 38}, SimConfig(budget_per_day=100))
        x = sim._execute(initial_lots={'004400': [Lot('004400', 100, DAYS[0])]},
                         policy=lambda _: {'subscribe': {'004400': 100}})
        d = x['daily'].iloc[1]
        self.assertAlmostEqual(d['unit_nav'], 2.0)
        self.assertAlmostEqual(d['units'], 250.0)
        self.assertAlmostEqual(d['value'], 500.0)

    def test_cash_only_portfolio_dilutes_stock_return_correctly(self):
        sim = make_sim({'004400': [1, 2] + [2] * 38}, SimConfig(budget_per_day=100))
        x = sim._execute(initial_lots={'004400': [Lot('004400', 100, DAYS[0])]},
                         policy=lambda _: {})
        d = x['daily'].iloc[1]
        # 股票100->200，第一日现金100，第二天入金100，TWR净值1.5而非2。
        self.assertAlmostEqual(d['unit_nav'], 1.5)
        self.assertAlmostEqual(d['value'], 400.0)

    def test_ex_dividend_published_valuation_uses_raw_nav(self):
        sim = make_sim({'004400': [1.] * 20 + [.9] * 20}, SimConfig(),
                       dividends={'004400': [(str(DAYS[20].date()), .1)]})
        sim._published_nav['004400']['adj_nav_total_return'] = 1.0
        seen = {}
        def check(s):
            if s['date'] == DAYS[20]:
                seen['pub'] = s['holdings']['004400']['value_pub']
                seen['value'] = s['value']
            return {}
        d = sim._execute(initial_lots={'004400': [Lot('004400', 100, DAYS[0])]},
                         policy=check)['daily'].iloc[20]
        self.assertAlmostEqual(seen['pub'], 90.)
        self.assertAlmostEqual(seen['value'], 100.)
        self.assertAlmostEqual(d['unit_nav'], 1.)

    def test_split_adjusts_shares_without_creating_loss(self):
        sim = make_sim({'004400': [1.] * 20 + [.5] * 20})
        sim._splits = {'004400': [(str(DAYS[20].date()), 2.)]}
        r = sim._execute(initial_lots={'004400': [Lot('004400', 100, DAYS[0])]})
        self.assertAlmostEqual(r['daily'].iloc[20]['unit_nav'], 1.)
        self.assertEqual(r['metrics']['max_drawdown'], 0.)

    def test_simulated_lot_cannot_invent_rolling_unlock(self):
        sim = make_sim({'021559': [1.] * 40})
        lot = Lot('021559', 100, DAYS[0], simulated=True)
        self.assertFalse(sim._redeemable('021559', lot, DAYS[31]))
        lot2 = Lot('021559', 100, DAYS[0], order_date=DAYS[0])
        maturity = DAYS[0] + pd.Timedelta(days=30)
        self.assertTrue(sim._redeemable('021559', lot2, maturity))
        self.assertFalse(sim._redeemable('021559', lot2, maturity + pd.Timedelta(days=1)))

    def test_sortino_is_downside_deviation_not_selected_subsample_sd(self):
        vals = [1., 1.1, 1.045, 1.07635, 1.1086405]
        df = pd.DataFrame({'date': DAYS[:5], 'nav': vals})
        stats = metrics.compute_metrics(df, rf=0., min_days=5)
        r = np.diff(np.log(vals))  # only to show expected standard is on arithmetic returns below
        arithmetic = np.diff(vals) / vals[:-1]
        expected_d = np.sqrt(np.mean(np.minimum(arithmetic, 0.)**2))
        expected_s = arithmetic.mean() / expected_d * np.sqrt(252)
        self.assertAlmostEqual(stats['sortino'], expected_s)


class PolicyIndependentTests(unittest.TestCase):
    def test_target_deficits_do_not_count_cash_twice(self):
        total = 100.
        ans = policies._water_filling(
            40, {'A': .2, 'B': .2}, total, {'A': .4, 'B': .4}, {}, ['A', 'B'])
        self.assertAlmostEqual(ans['A'], 20.)
        self.assertAlmostEqual(ans['B'], 20.)
        self.assertAlmostEqual(sum(ans.values()), 40.)

    def test_closed_form_waterfill_caps_and_budget(self):
        ans = policies._water_filling(
            15, {'A': 0, 'B': 0}, 100, {'A': .5, 'B': .5},
            {'A': 5}, ['A', 'B'])
        self.assertAlmostEqual(sum(ans.values()), 15, places=7)
        self.assertAlmostEqual(ans['A'], 5, places=7)

    def test_no_missing_nav_position_is_treated_as_zero(self):
        state = make_state(10000, {'004400': 10000}, {'004400': 1.})
        state['holdings']['004400']['value_pub'] = None
        policy = policies.make_b2(PARAMS)
        self.assertEqual(policy(state), {})

    def test_mdd_guard_can_recover_after_drawdown(self):
        p = {**PARAMS, 'B5': {**PARAMS['B5'], 'reenter_cooldown_days': 2}}
        with mock.patch.object(policies, '_plans', return_value=[]):
            policy = policies.make_b5(p)
        for h in ([1., .95], [1., .99], [1., .99]):
            state = make_state(100000, {'004400': 10000}, {'004400': 1.}, unit_hist=h)
            out = policy(state)
        self.assertFalse(any(x.get('action') == 'DE_RISK_CASH' for x in out.get('events', [])))

    def test_b4_cannot_read_future_style_file_by_default(self):
        with mock.patch.object(policies, '_load_style_signals', side_effect=RuntimeError('leak')):
            p = policies.make_b4(PARAMS)
            state = make_state(10000, {'004400': 10000}, {'004400': 1.})
            p(state)


class StyleIndependentTests(unittest.TestCase):
    def test_rolling_factor_scaler_is_causal(self):
        rng = np.random.default_rng(1)
        f = rng.normal(scale=.005, size=(100, 2))
        r = .001 + f @ np.array([.8, -.3])
        a = style.rolling_regression(r[:80], f[:80], lookback=50,
                                     min_samples=20, ridge=0, smooth=0)
        f2 = np.vstack([f[:80], np.array([[100., 200.]])] )
        r2 = np.r_[r[:80], 1000.]
        b = style.rolling_regression(r2, f2, lookback=50,
                                     min_samples=20, ridge=0, smooth=0)
        for col in ('alpha', 'beta_0', 'beta_1'):
            self.assertAlmostEqual(a.iloc[-1][col], b.iloc[79][col], places=10)
        self.assertAlmostEqual(a.iloc[-1]['beta_0'], .8, places=6)
        self.assertAlmostEqual(a.iloc[-1]['beta_1'], -.3, places=6)

    def test_missing_disclosure_not_treated_as_zero(self):
        rows = pd.DataFrame({
            'equity_nav_pct': [10., np.nan, 0.],
            'convertible_nav_pct_lower_bound': [10., 0., 0.],
            'report_asof': pd.to_datetime(['2024-12-31','2025-03-31','2025-06-30']),
            'published_at': pd.to_datetime(['2025-01-20','2025-04-20','2025-07-20'])})
        self.assertEqual(style.disclosure_breakpoints(rows, confirm_reports=2), [])


if __name__ == '__main__':
    unittest.main()

class ExecutionAndStressTests(unittest.TestCase):
    def test_b3_redeem_is_actual_transaction_in_simulator(self):
        params = {'target_weights': {'004400': .1, 'cash': .9},
                  'daily_limits': {}, 'B3': {'check_frequency': 'weekly',
                  'weight_trigger': .02, 'destination_fraction': .5,
                  'allow_redeem': True}}
        with mock.patch.object(policies, '_plans', return_value=[]):
            policy = policies.make_b3(params)
        sim = make_sim({'004400': [1.] * 40}, SimConfig())
        results = sim._execute(initial_lots={'004400': [Lot('004400', 100, DAYS[0])]},
                               initial_cash=100, policy=policy)
        self.assertIn('REDEEM', set(results['events']['event']))
        self.assertAlmostEqual(results['daily'].iloc[0]['value'], 200.)

    def test_b5_redeem_guard_executes_and_stays_research_only(self):
        params = {**PARAMS, 'B5': {**PARAMS['B5'], 'allow_redeem': True,
                                  'portfolio_redeem_fraction': .5}}
        sim = make_sim({'004400': [1.] + [.94] * 39}, SimConfig())
        with mock.patch.object(policies, '_plans', return_value=[]):
            policy = policies.make_b5(params)
        result = sim._execute(initial_lots={'004400': [Lot('004400', 100, DAYS[0])]},
                              policy=policy)
        redeem = result['events'][result['events']['event'] == 'REDEEM']
        self.assertFalse(redeem.empty)
        self.assertAlmostEqual(redeem['shares'].iloc[0], 50., places=6)

    def test_bootstrap_aligns_funds_by_calendar_and_wilson_zero_upper_bound(self):
        from fundkit.evaluation import stress_bootstrap
        from fundkit.adapters import CST
        d = pd.date_range('2025-01-01', periods=200, freq='B')
        class Fake:
            def nav(self, codes, _until):
                code = codes[0]
                idx = d if code == 'A' else d[10:]
                # 上涨序列不会触发5%回撤，但置信区间上界不能是0。
                return pd.DataFrame({'nav_date': idx,
                                     'adj_nav_total_return': 1.001 ** np.arange(len(idx))})
        snap = {'funds': [{'code': 'A', 'value_cny': 50},
                          {'code': 'B', 'value_cny': 50}]}
        x = stress_bootstrap(Fake(), snap, horizon=40, block=5,
                             n_paths=100, lookback_days=200)
        self.assertEqual(x['n_common_dates'], 189)
        self.assertEqual(x['p_mdd_exceed_4pct'], 0)
        self.assertGreater(x['ci95'][1], 0)

    def test_missing_historical_series_blocks_stress_instead_of_zero_return(self):
        from fundkit.evaluation import stress_2022_bond_crash
        class NoneAdapter:
            def nav(self, codes, _until):
                return pd.DataFrame(columns=['nav_date', 'adj_nav_total_return'])
        result = stress_2022_bond_crash(NoneAdapter(),
                    {'funds': [{'code': '004400', 'value_cny': 100}]})
        self.assertIsNone(result['portfolio_return'])
        self.assertEqual(result['missing_codes'], ['004400'])