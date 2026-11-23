"""Required user-facing risk-adjusted ranking acceptance tests.

High raw return/huge MDD cannot win solely from return; smooth NAV and
unverified lockups cannot exploit near-zero observed MDD; index ETF clones
occupy at most one diversified rank. All rely on existing dated NAVs.
"""
import unittest

import numpy as np
import pandas as pd

from fundkit.industry_rank import NavPath
from fundkit.discovery_risk_rank import (maximum_weight_for_budget,
    monthly_stationary_paths,rank_historical_risk_adjusted,theme_bucket)
from fundkit.pricing_audit import audit_valuation


class RiskAdjustedDiscoveryTest(unittest.TestCase):
    def test_closed_form_budget_matches_exhaustive_grid_on_any_path(self):
        rng=np.random.default_rng(8)
        for _ in range(15):
            v=np.exp(np.concatenate([[0],np.cumsum(rng.normal(.0001,.02,60))]))
            for budget in (.01,.02,.04,.08):
                w=maximum_weight_for_budget(v,budget)
                wealth=(1-w)+w*v
                mdd=np.max(1-wealth/np.maximum.accumulate(wealth))
                self.assertLessEqual(mdd,budget+1e-9)
                if w<1-.001:
                    high_w=min(w+.001,1)
                    new=(1-high_w)+high_w*v
                    self.assertGreater(np.max(1-new/np.maximum.accumulate(new)),
                                       budget-1e-9)

    def test_equal_return_higher_drawdown_gets_smaller_weight(self):
        safe=np.array([1.,1.04,1.06,1.2])
        risky=np.array([1.,1.9,1.05,1.2])
        a=maximum_weight_for_budget(safe,.04)
        b=maximum_weight_for_budget(risky,.04)
        self.assertAlmostEqual(a,1.)
        self.assertLess(b,.1)
        self.assertLess(np.expm1(np.log1p(b*.2)),
                        np.expm1(np.log1p(a*.2)))

    def test_bootstrap_is_deterministic(self):
        x=np.array([.02,-.03,.05,-.01,.02,-.04])
        a=monthly_stationary_paths(x,scenarios=48,months=6,seed=42)
        b=monthly_stationary_paths(x,scenarios=48,months=6,seed=42)
        np.testing.assert_array_equal(a,b)

    def test_no_future_data_or_external_prediction_needed_for_four_ranks(self):
        dates=pd.bdate_range("2020-10-01","2026-11-20")
        n=len(dates)
        rng=np.random.default_rng(123)
        paths={}
        rows=[]
        for i in range(16):
            code=f"{i+1:06d}"
            # Moderate positive growth with occasional losses, ample history.
            returns=rng.normal(.00019 + i*.000005,.0016,size=n)
            values=np.exp(np.cumsum(returns))
            paths[code]=NavPath(dates,values)
            rows.append(dict(code=code,name=f"普通基金{i}",group="中长纯债",
                   product=f"fund_{i}",product_key=f"fund_{i}",
                   feature_status="HISTORICAL_FEATURE_AVAILABLE",
                   period_return=.06,max_drawdown=.02,
                   is_long_periodic=False,is_periodic_unverified=False,
                   contract_review_hint=False))
        frame=pd.DataFrame(rows)
        frame.loc[0,"name"]="摊余成本型异常平滑"
        frame.loc[0,"max_drawdown"]=0.
        # We do not let a fake NAV zero MDD exploit the research score.
        for h in ("5y","3y","1y","6m"):
            d=audit_valuation(frame,pd.DataFrame())
            result,div,raw,audit=rank_historical_risk_adjusted(
                  d,paths,"2026-11-20",h,scenarios=32,top=10)
            self.assertTrue(len(div)>0,h)
            self.assertTrue(div.risk_adjusted_score.gt(0).all())
            self.assertTrue(div.risk_weight_4pct.between(0,1).all())
            self.assertTrue(div.risk_standalone_mdd.notna().all())
            self.assertEqual(result.loc[0,"risk_adjusted_status"],
                             "NAV_SMOOTHING_REVIEW")
            self.assertNotIn("000001",div.code.values)

    def test_index_theme_clones_are_not_repeated_in_diversified_selection(self):
        self.assertEqual(theme_bucket("银行ETF华宝","指数型-股票"),"银行")
        self.assertEqual(theme_bucket("银行ETF华夏","指数型-股票"),"银行")
        self.assertEqual(theme_bucket("权益成长主动","混合型-灵活"),"")
        # End-to-end diversified selection from actual risk-ranked NAV paths.
        dates=pd.bdate_range("2025-11-01","2026-11-20")
        n=len(dates)
        rows=[];paths={}
        for i in range(14):
            code=f"{i+1:06d}"
            name=f"银行ETF厂商{i}" if i<7 else f"其他指数{i}"
            ret=.0005 if i<7 else .00025
            noise=.0015*np.sin(np.arange(n)*.17+i*.1)
            values=np.exp(np.cumsum(np.full(n,ret)+noise))
            paths[code]=NavPath(dates,values)
            rows.append(dict(code=code,name=name,
                 group="指数型-股票",product=f"p_{i}",product_key=f"p_{i}",
                 feature_status="HISTORICAL_FEATURE_AVAILABLE",
                 period_return=.07,max_drawdown=.01,
                 is_long_periodic=False,is_periodic_unverified=False,
                 contract_review_hint=False))
        frame=audit_valuation(pd.DataFrame(rows),pd.DataFrame())
        _,div,raw,_=rank_historical_risk_adjusted(
               frame,paths,"2026-11-20","6m",scenarios=32,top=10)
        self.assertLessEqual((div.exposure_bucket=="银行").sum(),1)
        self.assertGreater(len(div),0)
        self.assertGreater(len(raw),0)


if __name__=="__main__":
    unittest.main()
