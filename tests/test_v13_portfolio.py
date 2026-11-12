"""Synthetic acceptance tests; never interpreted as actual fund performance."""
import unittest

import numpy as np
import pandas as pd

from fundkit.v13_model import fit_one_year_model
from fundkit.v13_portfolio import RobustSOCPConfig,solve_robust_socp


class RecentModelTests(unittest.TestCase):
    def test_only_latest_year_affects_forecast(self):
        rng=np.random.default_rng(9)
        returns=rng.normal(.0002,.001,(352,3))
        x=pd.DataFrame(returns,index=pd.bdate_range("2024-01-01",periods=352),
                       columns=["A","B","C"])
        y=x.copy();y.iloc[:100]+=0.2
        a=fit_one_year_model(x,categories={"A":"bond","B":"bond","C":"bond"},
                             bootstrap_draws=12)
        b=fit_one_year_model(y,categories={"A":"bond","B":"bond","C":"bond"},
                             bootstrap_draws=12)
        self.assertTrue(np.allclose(a["mean_annual"],b["mean_annual"]))
        self.assertEqual(a["n_common_observations"],252)
        self.assertTrue(a["mean_shrinkage_validation"]["n_folds"]>=2)

    def test_invalid_returns_fail_closed(self):
        df=pd.DataFrame({"A":[0.001]*251+[float("nan")]},
                        index=pd.bdate_range("2024-01-01",periods=252))
        with self.assertRaisesRegex(ValueError,"DATA_BLOCKED"):
            fit_one_year_model(df)

    def test_covariance_and_uncertainty_are_psd(self):
        rng=np.random.default_rng(8)
        r=rng.normal(0,.002,(252,3))
        df=pd.DataFrame(r,index=pd.bdate_range("2024-01-01",periods=252),
                        columns=["A","B","C"])
        m=fit_one_year_model(df,bootstrap_draws=16)
        for key in ("return_cov_annual","forecast_error_cov_annual"):
            self.assertGreaterEqual(np.linalg.eigvalsh(m[key]).min(),-1e-9)


class PortfolioSOCPTests(unittest.TestCase):
    def test_positive_alpha_direct_weight_solution(self):
        R=np.column_stack((np.full(260,.0005),np.full(260,.0001)))
        s=np.stack([R[:252],R[-252:]])
        ans=solve_robust_socp(
            R,s,np.array([.12,.02]),np.zeros((2,2)),np.zeros((2,2)),
            fund_upper=np.array([.6,.6]),
            config=RobustSOCPConfig(uncertainty_penalty=0.,
                                    volatility_penalty=0.,turnover_cost=0.))
        self.assertEqual(ans["status"],"OPTIMAL",ans)
        self.assertAlmostEqual(ans["weights"][0],.6,places=4)
        self.assertLessEqual(ans["scenario_mdd_p95"],.04+1e-7)

    def test_injected_forecast_uncertainty_reduces_concentration(self):
        R=np.column_stack((np.full(260,.0005),np.full(260,.0004)))
        s=np.stack([R[:252],R[-252:]])
        omega=np.diag([.08**2,.001**2])
        kw=dict(fund_upper=np.array([.6,.6]))
        baseline=solve_robust_socp(R,s,[.11,.08],np.zeros((2,2)),omega,
                                  config=RobustSOCPConfig(
                                      uncertainty_penalty=0.,volatility_penalty=0.,
                                      turnover_cost=0.),**kw)
        robust=solve_robust_socp(R,s,[.11,.08],np.zeros((2,2)),omega,
                                config=RobustSOCPConfig(
                                    uncertainty_penalty=1.5,volatility_penalty=0.,
                                    turnover_cost=0.),**kw)
        self.assertEqual(baseline["status"],"OPTIMAL",baseline)
        self.assertEqual(robust["status"],"OPTIMAL",robust)
        self.assertLess(robust["weights"][0],baseline["weights"][0]-0.05)

    def test_historical_crash_hard_cut_and_frozen_weight(self):
        R=np.zeros((260,2))
        R[30,0]=-.20
        s=np.stack([R[:252],R[-252:]])
        ans=solve_robust_socp(
            R,s,[.30,.01],np.zeros((2,2)),np.zeros((2,2)),
            fund_lower=[0.,.1],fund_upper=[1.,.1],
            current_weights=[0.,.1,.9],
            config=RobustSOCPConfig(uncertainty_penalty=0.,
                                    volatility_penalty=0.,turnover_cost=0.))
        self.assertEqual(ans["status"],"OPTIMAL",ans)
        self.assertLessEqual(ans["weights"][0],.2+1e-5)
        self.assertAlmostEqual(ans["weights"][1],.1,places=5)

    def test_high_exposure_floor_not_replaced_with_cash(self):
        R=np.zeros((260,1));R[50,0]=-.20
        s=np.stack([R[:252],R[-252:]])
        ans=solve_robust_socp(R,s,[.2],[[0]],[[0]],
                              config=RobustSOCPConfig(
                                  min_risky_weight=.8,turnover_cost=0.))
        self.assertEqual(ans["status"],"INFEASIBLE")
        self.assertIsNone(ans["weights"])


if __name__=="__main__":
    unittest.main()
