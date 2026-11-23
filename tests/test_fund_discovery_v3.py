"""Fund-discovery v3 regression tests: window isolation, factor risk and research flow."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fundkit.industry_rank import NavPath
from fundkit.discovery_scoring_v3 import (windowed_values, nav_mean_uncertainty,
    joint_forecast_paths,descriptive_research)
from fundkit.pricing_audit import audit_valuation, duration_stress_loss
from fundkit.return_models import resolve_forecasts
from fundkit.fund_master import resolve_liquidity
from fundkit.opportunity_rank import OpportunityInput,optimize_opportunity,risk_upper,cash_paths
from scripts.run_fund_discovery import analyze_horizon


class FundDiscoveryV3Test(unittest.TestCase):
    def test_four_observation_windows_are_distinct_no_old_data_leak(self):
        dates=pd.bdate_range("2020-01-01","2026-11-20")
        daily=np.where(dates < pd.Timestamp("2026-01-01"),.001,-.0004)
        path=NavPath(dates,np.exp(np.cumsum(daily)))
        w={h:windowed_values(path,"2026-11-20",h)
           for h in ("6m","1y","3y","5y")}
        self.assertLess(len(w["6m"]),len(w["1y"]))
        self.assertLess(len(w["1y"]),len(w["3y"]))
        self.assertLess(len(w["3y"]),len(w["5y"]))
        self.assertLess(w["6m"][-1]/w["6m"][0],
                        w["5y"][-1]/w["5y"][0])

    def test_explicit_horizon_forecast_cannot_cross_apply(self):
        e=pd.DataFrame([dict(code="000001",known_at="2026-11-19",
           trained_through="2026-11-18",source_url="https://example.com/forecast",
           model_id="M",forecast_mu_annual=.12,decision_months=6,
           lookback_horizon="6m",forecast_se_annual=.03)])
        a=resolve_forecasts(["000001"],e,"2026-11-20",6,lookback_horizon="6m")
        b=resolve_forecasts(["000001"],e,"2026-11-20",6,lookback_horizon="5y")
        self.assertEqual(a.forecast_status.iloc[0],"EVIDENCE_SUPPLIED_UNVALIDATED")
        self.assertEqual(b.forecast_status.iloc[0],"FORECAST_MISSING")
        self.assertAlmostEqual(a.forecast_se_annual.iloc[0],.03)

    def test_joint_paths_share_returns_and_risk_assumptions(self):
        rng=np.random.default_rng(123)
        hist=np.exp(np.cumsum(rng.normal(.0001,.004,size=1500)))
        lo=joint_forecast_paths(hist,.02,steps=126,scenarios=64,seed=8,
                                forecast_se_annual=0)
        hi=joint_forecast_paths(hist,.12,steps=126,scenarios=64,seed=8,
                                forecast_se_annual=0)
        self.assertEqual(lo.shape,(64,127))
        self.assertTrue(np.array_equal(lo[:,0],np.ones(64)))
        self.assertTrue(np.all(hi[:,-1]>lo[:,-1]))
        self.assertTrue(np.isfinite(nav_mean_uncertainty(hist)))

    def test_user_supplied_tiny_loss_is_not_passed_as_economic_risk(self):
        frame=pd.DataFrame([dict(code="007680",period_return=.03,max_drawdown=0)])
        e=pd.DataFrame([dict(code="007680",known_at="2026-11-19",
            pricing_method="amortized_cost",source_url="https://example.com/valuation",
            risk_model="BOND_DURATION_STRESS_v1",
            risk_input_source_url="https://example.com/exposures",
            risk_input_known_at="2026-11-18",
            economic_mdd_upper=.00001,modified_duration=3.,rate_shock_bp=100.,
            spread_duration=2.,spread_shock_bp=50.,convexity=0.,default_loss=0.)])
        d=audit_valuation(frame,e)
        self.assertAlmostEqual(duration_stress_loss(3,100,2,50,0,0),.04)
        self.assertAlmostEqual(d.economic_mdd_upper.iloc[0],.04)
        self.assertEqual(d.valuation_warning.iloc[0],"AMORTIZED_COST_NAV_NOT_MARKET_RISK")
        self.assertEqual(d.risk_quality.iloc[0],"REPRODUCIBLE_STRESS_NOT_OOS_VALIDATED")
        e.loc[0,"risk_input_source_url"]=""
        d=audit_valuation(frame,e)
        self.assertFalse(d.economic_risk_verified.iloc[0])
        self.assertTrue(d.economic_mdd_upper.isna().all())

    def test_no_contract_or_fee_cannot_suppress_historical_research(self):
        d=pd.DataFrame([dict(code="000001",name="普通纯债",group="中长纯债",
           feature_status="HISTORICAL_FEATURE_AVAILABLE",start="2025-11-20",
           end="2026-11-20",period_return=.06,max_drawdown=.02)])
        l=resolve_liquidity(["000001"],pd.DataFrame(),"2026-11-20")
        forecast=resolve_forecasts(["000001"],pd.DataFrame(),"2026-11-20",6,"1y")
        args=SimpleNamespace(cash_rate_annual=None,scenarios=40,block_length=5,
                             seed=42,grid_points=21,order_amount=1000)
        out,investable,_,stats=analyze_horizon(
            d,l,pd.DataFrame(),forecast,pd.Timestamp("2026-11-20"),
            6,False,args,None,lookback="1y")
        self.assertAlmostEqual(out.historical_return_cagr.iloc[0],.06,places=2)
        self.assertEqual(out.historical_return_rank.iloc[0],1)
        self.assertEqual(stats["historical_return_candidates"],1)
        self.assertEqual(stats["experimentally_scored"],0)
        self.assertTrue(investable.empty)

    def test_monotone_risk_search_equals_exhaustive_grid(self):
        rng=np.random.default_rng(7)
        f=np.concatenate([np.ones((36,1)),
              np.exp(np.cumsum(rng.normal(.0001,.009,size=(36,36)),axis=1))],axis=1)
        c=cash_paths(.02,steps=36,scenarios=36)
        setting=OpportunityInput(.12,.02,.004,.03,decision_months=6,
                                 grid_points=71)
        fast=optimize_opportunity(f,c,setting)
        weights=np.linspace(0,1,71)
        risks=[risk_upper(f,c,float(w),.004,.03) for w in weights]
        self.assertTrue(np.all(np.diff(risks)>=-1e-10))
        for result in fast:
            feasible=np.flatnonzero(np.asarray(risks)<=result["budget"]+1e-12)
            self.assertTrue(len(feasible)>0)
            self.assertAlmostEqual(result["weight"],weights[feasible[-1]])

    def test_falling_cash_uses_general_grid_not_false_monotonicity(self):
        rng=np.random.default_rng(11)
        f=np.concatenate([np.ones((16,1)),
             np.exp(np.cumsum(rng.normal(0,.01,size=(16,9)),axis=1))],axis=1)
        c=np.tile(np.array([1.,.99,1.01,1.03,.98,1.02,1.03,1.01,1.05,1.03]),(16,1))
        setting=OpportunityInput(.11,-.005,.001,.02,decision_months=3,
                                 grid_points=31)
        got=optimize_opportunity(f,c,setting,budgets=(.04,))
        risks=[risk_upper(f,c,w,.001,.02) for w in np.linspace(0,1,31)]
        feasible=np.flatnonzero(np.asarray(risks)<=.04+1e-12)
        if len(feasible):
            self.assertAlmostEqual(got[0]["weight"],np.linspace(0,1,31)[feasible[-1]])
        else:
            self.assertEqual(got[0]["status"],"CASH_BASELINE_OVER_BUDGET")

    def test_not_validated_forecast_never_promoted_by_historical_win(self):
        d=pd.DataFrame([dict(code="000001",name="示例",group="中长纯债",
            feature_status="HISTORICAL_FEATURE_AVAILABLE",start="2025-11-20",
            end="2026-11-20",period_return=.14,max_drawdown=0.)])
        q=descriptive_research(d.assign(economic_risk_verified=False,
             zero_or_tiny_nav_mdd=True),"2026-11-20","1y")
        self.assertEqual(q.risk_comparability.iloc[0],"NAV_SMOOTHING_REVIEW")
        self.assertEqual(q.descriptive_research_status.iloc[0],"NAV_HISTORY_ONLY")
        self.assertEqual(q.historical_return_rank.iloc[0],1)


if __name__=="__main__":
    unittest.main()
