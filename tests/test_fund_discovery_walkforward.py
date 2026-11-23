"""Walk-forward formation archives, cash point-in-time and deterministic fold tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundkit.industry_rank import NavPath
from fundkit.pricing_audit import audit_valuation
from scripts.run_discovery_walkforward import (date_catalog,read_cash_history,
     cash_formation_rate, fold_signals,realized_signals,main as wf_main)
from scripts.run_industry_fund_ranking import load_catalog


class FundWalkForwardTest(unittest.TestCase):
    def test_no_historical_catalog_snapshots_cannot_reuse_current_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            t="2026-11-20"
            pd.DataFrame([dict(formation_date=t)]).to_csv(p/"dates.csv",index=False)
            pd.DataFrame([dict(date="2025-01-01",nav=1.,
                known_at="2025-01-01",source_url="https://example.com/cash")]).to_csv(
                p/"cash.csv",index=False)
            pd.DataFrame([dict(group="中长纯债",date="2025-01-01",nav=1.,
                known_at="2025-01-01",source_url="https://example.com/bench")]).to_csv(
                p/"bench.csv",index=False)
            with self.assertRaisesRegex(ValueError,"missing as-of historical catalog"):
                wf_main(["--dates",str(p/"dates.csv"),"--historical-catalogs",str(p/"archive"),
                   "--benchmark-history",str(p/"bench.csv"),"--cash-nav",str(p/"cash.csv"),
                   "--data-dir",str(p/"data"),"--output",str(p/"out"),
                   "--validation-end","2025-01-01","--test-start","2026-01-02"])

    def test_future_catalog_record_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            pd.DataFrame([dict(code="000001",name="TEST A",type="债券型",
                known_at="2026-11-21",source_url="https://example.com/catalog")]).to_csv(
                p/"2026-11-20.csv",index=False)
            with self.assertRaisesRegex(ValueError,"lookahead"):
                date_catalog(p,pd.Timestamp("2026-11-20"))

    def test_cash_historical_forecast_ignores_future_observations(self):
        dates=pd.bdate_range("2025-01-01","2027-06-01")
        v=(1.0001)**np.arange(len(dates))
        cash=pd.DataFrame(dict(date=dates,nav=v,known_at=dates,
            source_url="https://example.com/cash"))
        r=cash_formation_rate(cash,pd.Timestamp("2026-11-20"))
        cash.loc[cash.date.gt("2026-11-20"),"nav"]*=100
        self.assertAlmostEqual(cash_formation_rate(cash,pd.Timestamp("2026-11-20")),r)

    def test_actual_fold_selects_before_label_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            funds=[dict(code=str(i+1).zfill(6),name=f"模拟纯债{i}A",
                        type="债券型-中长期纯债") for i in range(6)]
            pd.DataFrame(funds).to_csv(p/"catalog.csv",index=False)
            catalog=load_catalog(p/"catalog.csv",periodic_overrides={})
            grp=catalog.group.iloc[0]
            t=pd.Timestamp("2026-11-20")
            dates=pd.bdate_range("2024-01-01","2027-07-01")
            paths={}
            for k,code in enumerate(catalog.code):
                rng=np.random.default_rng(100+k)
                logret=.00035+k*.00003+rng.normal(0,.004,size=len(dates))
                paths[code]=NavPath(dates,np.exp(np.cumsum(logret)))
            benchmark=pd.DataFrame(dict(group=grp,date=dates,
                   nav=np.exp(.0003*np.arange(len(dates))),
                   known_at=dates,source_url="https://example.com/market-index"))
            cash=pd.DataFrame(dict(date=dates,known_at=dates,
                   nav=np.exp(.00008*np.arange(len(dates))),
                   source_url="https://example.com/cash-index"))
            val=pd.DataFrame([dict(code=c,known_at="2026-01-01",
                pricing_method="market_value",source_url="https://example.com/official")
                for c in catalog.code])
            a=fold_signals(catalog,paths,benchmark,cash,val,t,"1y",6,
                           top=2,scenarios=32,grid_points=11,seed=9)
            b=fold_signals(catalog,paths,benchmark,cash,val,t,"1y",6,
                           top=2,scenarios=32,grid_points=11,seed=9)
            self.assertFalse(a.empty)
            pd.testing.assert_frame_equal(a,b)
            self.assertTrue(set(a.method).issuperset({"HIST_RETURN_ONLY",
                "M0_CATEGORY_BASELINE","M1_EB_FORECAST","M1_NAV_RISK_PROXY"}))
            # A 4%-risk-budget model cannot be compared with unscaled
            # 100%-fund historical-return baselines.
            self.assertTrue(a.weight.between(0,1).all())
            self.assertTrue(a.groupby("code").weight.nunique().le(1).all())
            realized=realized_signals(a,paths,cash,t+pd.DateOffset(months=6))
            self.assertFalse(realized.empty)
            self.assertTrue(realized.evaluation_status.eq("NAV_ONLY_NO_FEES").all())


if __name__=="__main__":
    unittest.main()
