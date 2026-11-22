"""PIT forward diagnostic and trade status adapter regression tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundkit.discovery_oos import validate_pit_selections,nonoverlapping_pairs,realized_buy_hold
from scripts.evaluate_fund_discovery_oos import main as oos_main
from scripts.fetch_fund_trade_status import map_status,transform_purchase_status


def signal(date, method="opportunity_experimental", code="000001", next_months=6):
    formed=pd.Timestamp(date)
    end=formed+pd.DateOffset(months=next_months)
    d=dict(formation_date=str(formed.date()),evaluation_end=str(end.date()),
        code=code,method=method,horizon="6m",weight=1.,
        fee_rate=0.01)
    for key in ("catalog_known_at","forecast_known_at","valuation_known_at",
                "liquidity_known_at","price_known_at","fees_known_at"):
        d[key]=str((formed-pd.Timedelta(days=1)).date())
    return d


class TradeAndOOSTest(unittest.TestCase):
    def test_akshare_snapshot_is_not_a_contract_or_total_fee(self):
        self.assertEqual(map_status("开放申购","purchase"),"OPEN")
        self.assertEqual(map_status("开放赎回","redeem"),"OPEN")
        self.assertEqual(map_status("限大额","purchase"),"LIMITED")
        self.assertEqual(map_status("暂停申购","purchase"),"PAUSED")
        self.assertEqual(map_status("封闭期","redeem"),"CLOSED")
        self.assertEqual(map_status("场内买入","purchase"),"UNKNOWN")
        df=pd.DataFrame([{"基金代码":"00123","基金简称":"示例",
                          "申购状态":"开放申购","赎回状态":"开放赎回",
                          "下一开放日":"2026-11-24","手续费":0.12,
                          "日累计限定金额":5000}])
        r=transform_purchase_status(df,captured_at="2026-11-22T08:00:00+00:00")
        self.assertEqual(r.code.iloc[0],"000123")
        self.assertEqual(r.next_exit.iloc[0],"")
        self.assertEqual(r.total_fee_rate.iloc[0],"")
        self.assertFalse(r.official_contract_verified.iloc[0])

    def test_oos_rejects_future_known_and_missing_evidence(self):
        d=signal("2026-01-31")
        d["forecast_known_at"]="2026-02-01"
        p=validate_pit_selections(pd.DataFrame([d]))
        self.assertIn("LOOKAHEAD_FORECAST_KNOWN_AT",p.pit_status.iloc[0])
        self.assertEqual(p.execution_status.iloc[0],"EVIDENCE_INSUFFICIENT")
        d["forecast_known_at"]=""
        p=validate_pit_selections(pd.DataFrame([d]))
        self.assertIn("MISSING_FORECAST_KNOWN_AT",p.pit_status.iloc[0])

    def test_actual_wealth_includes_cost_and_cash(self):
        r=realized_buy_hold([1,1.2,1.1,1.3],[1,1,1,1],.5,.02)
        self.assertAlmostEqual(r["net_return"],.5*(1.3*.98)+.5-1)
        self.assertGreater(r["max_drawdown"],0)

    def test_oos_paired_requires_nonoverlap_and_min_folds(self):
        frames=[]
        for dt in pd.date_range("2024-01-31",periods=6,freq="6ME"):
            end=dt+pd.DateOffset(months=6)
            for method,value,dd in (("opportunity_experimental",.04,.02),
                                    ("legacy_pareto",.01,.03)):
                frames.append(dict(formation_date=dt,evaluation_end=end,
                    horizon="6m",method=method,cash_excess=value,
                    max_drawdown=dd,pit_status="PIT_OK"))
        r=nonoverlapping_pairs(pd.DataFrame(frames),
              "opportunity_experimental","legacy_pareto",min_folds=5)
        self.assertEqual(r.folds.iloc[0],6)
        self.assertEqual(r.decision.iloc[0],"RESEARCH_EVIDENCE_POSITIVE")
        r=nonoverlapping_pairs(pd.DataFrame(frames).iloc[:6],
              "opportunity_experimental","legacy_pareto",min_folds=5)
        self.assertEqual(r.decision.iloc[0],"EVIDENCE_INSUFFICIENT")

    def test_missing_historical_inputs_cannot_claim_oos(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            out=oos_main(["--selections",str(p/"unknown.csv"),
               "--fund-nav-dir",str(p/"nav"),"--cash-nav",str(p/"cash.csv"),
               "--output",str(p/"out")])
            self.assertFalse(out["oos_certified"])
            self.assertEqual(out["status"],"EVIDENCE_INSUFFICIENT")

    def test_actual_oos_is_still_nav_research_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/"nav").mkdir()
            formation=pd.Timestamp("2025-12-31")
            pd.DataFrame([signal(formation)]).to_csv(p/"selection.csv",index=False)
            dates=pd.bdate_range("2026-01-01","2026-07-01")
            pd.DataFrame({"date":dates,"nav":1+np.arange(len(dates))*.001}).to_csv(
                p/"nav"/"000001.csv",index=False)
            pd.DataFrame({"date":dates,"nav":1+np.arange(len(dates))*.0001}).to_csv(
                p/"cash.csv",index=False)
            out=oos_main(["--selections",str(p/"selection.csv"),
               "--fund-nav-dir",str(p/"nav"),"--cash-nav",str(p/"cash.csv"),
               "--output",str(p/"out")])
            self.assertEqual(out["forward_evaluated"],1)
            self.assertFalse(out["oos_certified"])
            self.assertEqual(out["status"],"PIT_RESEARCH_DIAGNOSTIC_ONLY")
            self.assertTrue((p/"out"/"oos_results.csv").is_file())


if __name__=="__main__":
    unittest.main()
