"""独立合成价格/约束测试，不作为真实基金业绩证据。"""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from fundkit.weight_optimizer import (WeightSearchConfig, optimize_target_weights,
                                      build_returns_asof)


DAYS = pd.bdate_range('2024-01-02', periods=340)
CFG = WeightSearchConfig(bootstrap_paths=7, block_lengths=(10, 30),
                         candidate_count=105, historical_windows=4,
                         min_observations=252, batch_size=50, lookback_days=340)


class TargetWeightOptimizerTests(unittest.TestCase):
    def test_no_fake_5pct_return_when_market_flat(self):
        df = pd.DataFrame({'F1': np.zeros(340), 'F2': np.zeros(340)}, index=DAYS)
        r = optimize_target_weights(df, cfg=CFG)
        self.assertEqual(r['status'], 'RETURN_TARGET_UNMET')
        self.assertEqual(r['scenario_return_q25'], 0)
        self.assertTrue(r['scenario_mdd_p95'] <= .04)
        self.assertFalse(r['trade_ready'])

    def test_stable_synthetic_returns_can_be_scenario_feasible(self):
        df = pd.DataFrame({'F1': [.0005] * 340, 'F2': [0.] * 340}, index=DAYS)
        r = optimize_target_weights(df, cfg=CFG)
        self.assertEqual(r['status'], 'SCENARIO_CRITERIA_MET_UNVALIDATED')
        self.assertLessEqual(r['weights']['F1'], .6 + 1e-8)
        self.assertGreater(r['scenario_return_q25'], .05)
        self.assertAlmostEqual(sum(r['weights'].values()), 1, places=7)

    def test_no_risky_position_when_drawdown_unacceptable(self):
        df = pd.DataFrame({'F1': [-.10] * 10 + [.001] * 330}, index=DAYS)
        r = optimize_target_weights(df, cfg=CFG)
        self.assertLessEqual(r['scenario_mdd_p95'], .04 + 1e-8)
        self.assertNotEqual(r['status'], 'SCENARIO_CRITERIA_MET_UNVALIDATED')
        self.assertFalse(r['trade_ready'])

    def test_frozen_weight_handoff_rejects_lookahead(self):
        import tempfile, json, os
        from fundkit.weight_optimizer import load_frozen_research_weights
        result = {'status': 'SCENARIO_CRITERIA_MET_UNVALIDATED', 'trade_ready': False,
                  'data_end': '2025-12-01',
                  'weights': {'F1': .6, 'cash': .4}}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'weights.json')
            with open(path, 'w') as fh:
                json.dump(result, fh)
            with self.assertRaisesRegex(ValueError, 'LOOKAHEAD_BLOCKED'):
                load_frozen_research_weights(path, '2025-11-01', ['F1'])
            w = load_frozen_research_weights(path, '2026-01-01', ['F1'])
            self.assertAlmostEqual(w['F1'], .6)

    def test_nonaligned_missing_nav_must_block(self):
        df = pd.DataFrame({'F1': [0.] * 340, 'F2': [0.] * 340}, index=DAYS)
        df.iloc[5, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'DATA_BLOCKED'):
            optimize_target_weights(df, cfg=CFG)

    def test_no_lookahead_from_asof_adapter(self):
        days = DAYS
        class FakeAdapter:
            def nav(self, codes, until):
                rows = []
                for code in codes:
                    for i, d in enumerate(days):
                        if d <= pd.Timestamp(until).tz_localize(None) - pd.Timedelta(days=2):
                            rows.append((code, d, 1.0003 ** i))
                return pd.DataFrame(rows, columns=['fund_code', 'nav_date',
                                                     'adj_nav_total_return'])
        r = build_returns_asof(FakeAdapter(), ['F1', 'F2'], DAYS[-1], CFG)
        self.assertLessEqual(r.index[-1], DAYS[-1] - pd.Timedelta(days=2))
        self.assertFalse(r.isna().any().any())

    def test_excluded_equity_weight_must_stay_zero(self):
        df = pd.DataFrame({'BOND': [.0005] * 340, 'EQUITY': [.0008] * 340}, index=DAYS)
        r = optimize_target_weights(df, caps={'EQUITY': 0.0}, cfg=CFG)
        self.assertEqual(r['weights']['EQUITY'], 0.0)
        self.assertAlmostEqual(sum(r['weights'].values()), 1, places=8)

    def test_frozen_equity_not_assumed_sold_or_reinvested(self):
        df = pd.DataFrame({'BOND': [.0005] * 340, 'EQUITY': [-.0001] * 340}, index=DAYS)
        r = optimize_target_weights(df, fixed_weights={'EQUITY': .13},
                                     current_weights={'BOND': .6, 'EQUITY': .13, 'cash': .27},
                                     cfg=CFG)
        self.assertAlmostEqual(r['weights']['EQUITY'], .13, places=8)
        self.assertAlmostEqual(sum(r['weights'].values()), 1, places=7)

    def test_same_seed_reproduces(self):
        df = pd.DataFrame({'F1': [.0005] * 340, 'F2': [0.] * 340}, index=DAYS)
        a = optimize_target_weights(df, cfg=CFG)
        b = optimize_target_weights(df, cfg=CFG)
        self.assertEqual(a['weights'], b['weights'])
        self.assertEqual(a['status'], b['status'])

    def test_enforce_5pct_return_4pct_drawdown_target(self):
        from fundkit.evaluation import MDD_BUDGET, CAGR_TARGET
        from fundkit import policies
        self.assertEqual(MDD_BUDGET, .04)
        self.assertEqual(CAGR_TARGET, .05)
        self.assertEqual(policies.load_params()['B5']['mdd_budget'], .04)
        self.assertEqual(CFG.max_drawdown, .04)
        self.assertEqual(CFG.target_annual_return, .05)

    def test_out_of_scope_cash_is_not_automatically_an_optimized_portfolio(self):
        df = pd.DataFrame({'F1': [0.] * 340}, index=DAYS)
        with self.assertRaisesRegex(ValueError, 'unknown|未知'):
            optimize_target_weights(df, caps={'NOT_IN_MATRIX': 0}, cfg=CFG)

if __name__ == '__main__':
    unittest.main()