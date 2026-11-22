"""Contract, pricing, risk and anti-lookahead acceptance tests for fund discovery."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fundkit.fund_master import (product_review_hint,read_timestamped_evidence,
                                 resolve_liquidity)
from fundkit.pricing_audit import audit_valuation
from fundkit.return_models import empirical_bayes_alpha, resolve_forecasts
from fundkit.opportunity_rank import (OpportunityInput,buy_hold_wealth,
    cash_paths,max_drawdown,optimize_opportunity,risk_upper,stationary_bootstrap)
from scripts.run_fund_discovery import main


class FundDiscoveryContractsTest(unittest.TestCase):
    def test_contract_names_are_review_triggers_never_trade_permission(self):
        for name in ("嘉实致元42个月定期债券", "民生加银39个月定期纯债",
                     "中加享利三年债券", "嘉实安元39个月定期纯债",
                     "汇添富长添利定期开放债券A"):
            self.assertTrue(product_review_hint(name),name)
        result=resolve_liquidity(["007589","008693","007680","008338","003528"],
                                pd.DataFrame(),pd.Timestamp("2026-11-20"))
        self.assertTrue(result.liquidity_status.eq("UNKNOWN").all())
        self.assertFalse(result.buyable_now.any())
        self.assertFalse(result.executable_horizon.any())

    def test_future_known_liquidity_is_not_available_today(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"liquidity.csv"
            pd.DataFrame([dict(code="007589", known_at="2026-11-21",
                source_url="https://example.com/future",
                subscription_status="OPEN",redemption_status="OPEN")]).to_csv(p,index=False)
            e=read_timestamped_evidence(p,"2026-11-20")
            self.assertTrue(e.empty)
            l=resolve_liquidity(["007589"],e,"2026-11-20")
            self.assertFalse(l.executable_horizon.iloc[0])

    def test_conflict_and_stale_are_not_verified_open(self):
        items=[dict(code="007589",known_at=pd.Timestamp("2026-11-20"),
                    source_url=f"https://example.com/{i}",
                    subscription_status=s,redemption_status="OPEN",
                    total_fee_rate=0.01,max_subscription_amount=10000,
                    next_exit="2026-12-20",lock_until="")
               for i,s in enumerate(["OPEN","PAUSED"])]
        r=resolve_liquidity(["007589"],pd.DataFrame(items),"2026-11-20")
        self.assertEqual(r.liquidity_status.iloc[0],"CONFLICT")
        self.assertFalse(r.buyable_now.iloc[0])
        old=pd.DataFrame(items[:1])
        old["known_at"]=pd.Timestamp("2026-11-01")
        r=resolve_liquidity(["007589"],old,"2026-11-20")
        self.assertEqual(r.liquidity_status.iloc[0],"STALE")
        self.assertFalse(r.executable_horizon.iloc[0])

    def test_contract_exit_and_fee_are_required(self):
        row=dict(code="000001",known_at=pd.Timestamp("2026-11-20"),
                 source_url="https://example.com/official",
                 subscription_status="OPEN",redemption_status="OPEN",
                 total_fee_rate=0.005,max_subscription_amount=10000,
                 next_exit="2027-12-20",lock_until="")
        bad=resolve_liquidity(["000001"],pd.DataFrame([row]),"2026-11-20",6)
        self.assertTrue(bad.buyable_now.iloc[0])
        self.assertFalse(bad.executable_horizon.iloc[0])
        row["next_exit"]="2027-01-20"
        good=resolve_liquidity(["000001"],pd.DataFrame([row]),"2026-11-20",6)
        self.assertTrue(good.executable_horizon.iloc[0])

    def test_valuation_unknown_and_amortized_not_zero_economic_risk(self):
        frame=pd.DataFrame([dict(code="007680",period_return=.02,max_drawdown=0),
                            dict(code="000001",period_return=.10,max_drawdown=.03)])
        d=audit_valuation(frame,pd.DataFrame())
        self.assertTrue(d.risk_quality.eq("RISK_UNVERIFIED").all())
        self.assertTrue(d.loc[d.code.eq("007680"),"zero_or_tiny_nav_mdd"].iloc[0])
        evidence=pd.DataFrame([dict(code="007680",pricing_method="amortized_cost",
                     economic_mdd_upper=np.nan,source_url="https://example.com/report",
                     known_at="2026-11-01")])
        d=audit_valuation(frame,evidence)
        self.assertEqual(d.loc[d.code.eq("007680"),"valuation_warning"].iloc[0],
                         "AMORTIZED_COST_NAV_NOT_MARKET_RISK")
        self.assertFalse(d.loc[d.code.eq("007680"),"economic_risk_verified"].iloc[0])

    def test_forecast_requires_horizon_and_training_cutoff(self):
        data=pd.DataFrame([dict(code="000001",known_at="2026-11-20",
                                trained_through="2026-11-21",
                                source_url="https://example.com/model",
                                model_id="M1",forecast_mu_annual=.15,decision_months=6)])
        r=resolve_forecasts(["000001"],data,"2026-11-20",6)
        self.assertEqual(r.forecast_status.iloc[0],"FORECAST_MISSING")
        data.loc[0,"trained_through"]="2026-11-19"
        r=resolve_forecasts(["000001"],data,"2026-11-20",6)
        self.assertEqual(r.forecast_status.iloc[0],"EVIDENCE_SUPPLIED_UNVALIDATED")

    def test_empirical_bayes_shrinks_without_inventing_alpha(self):
        self.assertAlmostEqual(empirical_bayes_alpha(.10,.2,0,.1),.02)
        self.assertAlmostEqual(empirical_bayes_alpha(.10,0,0,.1),.10)
        self.assertAlmostEqual(empirical_bayes_alpha(.10,1,0,0),0)

    def test_pathwise_mdd_not_weight_times_fund_mdd(self):
        fund=np.array([[1.,1.4,0.9,1.2]])
        cash=np.array([[1.,1.,1.,1.]])
        v=buy_hold_wealth(fund,cash,.5)
        self.assertAlmostEqual(max_drawdown(v)[0],(1.2-.95)/1.2)
        self.assertNotAlmostEqual(max_drawdown(v)[0],.5*max_drawdown(fund)[0])

    def test_budget_optimizer_monotonicity(self):
        fund=np.tile(np.array([1.,1.01,1.02,1.04]),(64,1))
        cash=np.tile(np.array([1.,1.001,1.002,1.003]),(64,1))
        b=OpportunityInput(.02,.015,0.,.02,decision_months=6,grid_points=101)
        a=OpportunityInput(.05,.015,0.,.02,decision_months=6,grid_points=101)
        scores_a=optimize_opportunity(fund,cash,a)
        scores_b=optimize_opportunity(fund,cash,b)
        for u,v in zip(scores_a,scores_b):
            self.assertGreaterEqual(u["opportunity_score_annual"],v["opportunity_score_annual"])
        self.assertGreater(scores_a[2]["opportunity_score_annual"],scores_b[2]["opportunity_score_annual"])

    def test_money_below_cash_has_zero_increment(self):
        f=np.tile(np.array([1.,1.01,1.02]),(64,1))
        c=np.tile(np.array([1.,1.,1.]),(64,1))
        score=optimize_opportunity(f,c,OpportunityInput(.01,.02,0.,.03))
        self.assertTrue(all(x["opportunity_score_annual"]==0 for x in score))

    def test_valuation_shock_prevents_zero_mdd_hijacking(self):
        f=np.ones((64,5))
        c=np.ones((64,5))
        # NAV is monotone/flat, but supplied independent economic shock is 20%.
        self.assertGreater(risk_upper(f,c,1.,0.,.2),.19)
        x=optimize_opportunity(f,c,OpportunityInput(.05,.015,0.,.20,grid_points=101))
        self.assertLess(x[2]["weight"],.25)

    def test_stationary_bootstrap_deterministic(self):
        v=np.exp(np.arange(130)*.001)
        a=stationary_bootstrap(v,steps=30,scenarios=40,block_length=5,seed=3)
        b=stationary_bootstrap(v,steps=30,scenarios=40,block_length=5,seed=3)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(a[:,0]==1))

    def test_snapshot_without_evidence_does_not_publish_fake_top10(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); source=base/"snapshot"; out=base/"output"
            source.mkdir()
            (source/"audit.json").write_text(json.dumps(
                {"asof":"2026-11-20","periodic_unverified_excluded":896}),encoding="utf-8")
            for h in ("5y","3y","1y","6m"):
                pd.DataFrame([dict(code="007589",name="嘉实致元42个月定期债券",
                            group="利率债",period_return=.03,max_drawdown=0.,
                            return_for_rank=.03)]).to_csv(source/f"ranking_{h}.csv",index=False)
                pd.DataFrame([dict(code="003528",reason="长周期定开")]).to_csv(
                     source/f"unranked_{h}.csv",index=False)
            audit=main(["--asof","2026-11-20","--snapshot",str(source),
                        "--output",str(out)])
            self.assertEqual(audit["oos_acceptance"],"EVIDENCE_INSUFFICIENT")
            self.assertFalse(audit["certified_investable_ranking"])
            for h in ("5y","3y","1y","6m"):
                r=pd.read_csv(out/f"research_{h}.csv",dtype={"code":str})
                self.assertEqual(set(r.code),{"007589","003528"})
                self.assertFalse(r.score_annual.notna().any())
                self.assertTrue(pd.read_csv(out/f"top10_experimental_{h}.csv").empty)
                self.assertTrue(pd.read_csv(out/f"investable_{h}.csv").empty)


if __name__=="__main__":
    unittest.main()
