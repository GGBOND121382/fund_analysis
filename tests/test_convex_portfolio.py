"""Numerical certificates and adversarial cases for continuous portfolio LP.

These tests use synthetic returns exclusively, NOT historical fund performance.
"""
import unittest
import numpy as np
import pandas as pd

from fundkit.convex_portfolio import (
    PortfolioLPConfig, solve_continuous_weights, wealth_coefficients,
    worst_violation, empirical_cvar,
)
from fundkit.weight_optimizer import WeightSearchConfig, generate_weight_candidates


class ContinuousPortfolioSolverTests(unittest.TestCase):
    def test_exact_long_only_optimum_not_random(self):
        data=np.zeros((280,2))
        data[:,0]=.0004
        sc=np.stack([data[:252],data[-252:]])
        ans=solve_continuous_weights(
            data,sc,np.array([.08,.01]),fund_upper=np.array([.6,.6]))
        self.assertEqual(ans['status'],'OPTIMAL')
        self.assertAlmostEqual(ans['weights'][0],.6,places=7)
        self.assertAlmostEqual(sum(ans['weights']),1,places=8)
        self.assertLessEqual(ans['historical_mdd'],.04+1e-8)
        self.assertLessEqual(ans['scenario_cvar_violation'],1e-7)

    def test_historical_crash_limits_continuous_exposure(self):
        returns=np.zeros((280,1))
        returns[50,0]=-.2
        paths=np.stack([returns[:252],returns[-252:]])
        ans=solve_continuous_weights(
            returns,paths,np.array([.30]),fund_upper=np.array([1.]))
        self.assertEqual(ans['status'],'OPTIMAL')
        self.assertLessEqual(ans['weights'][0],.2+1e-7)
        g=wealth_coefficients(returns[None],0.)[0]
        self.assertLessEqual(worst_violation(g,ans['weights'],.04)[0],1e-8)

    def test_incompatible_minimum_exposure_is_infeasible_not_cash(self):
        returns=np.zeros((280,1))
        returns[50,0]=-.2
        paths=np.stack([returns[:252],returns[-252:]])
        ans=solve_continuous_weights(
            returns,paths,np.array([.20]),
            config=PortfolioLPConfig(min_risky_weight=.7))
        self.assertEqual(ans['status'],'INFEASIBLE')
        self.assertIsNone(ans['weights'])

    def test_permanent_frozen_weight_is_exact(self):
        r=np.zeros((280,2))
        r[:,0]=.0003
        sc=np.stack([r[:252],r[-252:]])
        ans=solve_continuous_weights(
            r,sc,np.array([.08,.03]),fund_lower=np.array([.15,0]),
            fund_upper=np.array([.15,.6]),
            current_weights=np.array([.15,.0,.85]))
        self.assertEqual(ans['status'],'OPTIMAL')
        self.assertAlmostEqual(ans['weights'][0],.15,places=8)

    def test_cvar_uses_exact_discrete_tail_not_quantile_interpolation(self):
        v=np.array([-2.,-1.,4.])
        self.assertAlmostEqual(empirical_cvar(v,.95),4.0,places=10)
        self.assertAlmostEqual(empirical_cvar(np.zeros(20),.95),0.,places=10)

    def test_weight_search_respects_count_and_separates_cash(self):
        df=pd.DataFrame({
            'BOND_A':[.0003]*340,
            'BOND_B':[.0002]*340,
            'STOCK':[.0005]*340,
        },index=pd.bdate_range('2024-01-01',periods=340))
        cfg=WeightSearchConfig(
            min_observations=252,horizon=252,
            bootstrap_paths=4,block_lengths=(10,),
            historical_windows=2,candidate_count=12,
            max_single_fund_weight=.6)
        result=generate_weight_candidates(
            df,cfg=cfg,caps={'STOCK':0.},top_k=3)
        self.assertEqual(result['solver'].split()[0],'HiGHS')
        self.assertGreaterEqual(len(result['candidates']),1)
        self.assertLessEqual(len(result['candidates']),3)
        self.assertEqual(result['cash_benchmark']['weights']['cash'],1.)
        for row in result['candidates']:
            self.assertEqual(row['weights']['STOCK'],0.)
            self.assertGreater(sum(row['weights'][k] for k in ('BOND_A','BOND_B')),0)
            self.assertTrue(row['static_risk_feasible'])

    def test_all_negative_alpha_not_falsely_called_investment_optimum(self):
        df=pd.DataFrame({'BAD':[-.0002]*340},
                        index=pd.bdate_range('2024-01-01',periods=340))
        cfg=WeightSearchConfig(
            min_observations=252,horizon=252,
            bootstrap_paths=2,block_lengths=(10,),
            historical_windows=1,candidate_count=12)
        r=generate_weight_candidates(df,cfg=cfg,top_k=2)
        self.assertLessEqual(len(r['candidates']),2)
        self.assertTrue(all(row['weights']['cash']<1 for row in r['candidates']))
        self.assertTrue(all(row['static_risk_feasible'] for row in r['candidates']))


if __name__=='__main__':
    unittest.main()
