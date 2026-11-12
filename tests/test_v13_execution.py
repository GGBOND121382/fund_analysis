"""Synthetic tests of v1.3 daily amount and date-aligned bond ranking."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from fundkit.v13_execution import choose_daily_buys
from fundkit.v11_signals import scenario_risk_gate
from scripts.bond_resilience import screen_bond_resilience
from fundkit.v11_ablation import build_ablation_rules


class TestDailyAmountDecision(unittest.TestCase):
    def test_choose_maximum_feasible_subscription_not_fixed_75pct(self):
        buys,audit=choose_daily_buys(
            budget_cny=1000,deficits_cny={"B":1000},
            proposal_for_budget=lambda x: {"B":x},
            risk_checker=lambda b:{"status":"PASS" if b.get("B",0)<=400 else "FAIL"})
        self.assertEqual(buys,{"B":375.})
        self.assertEqual(audit["status"],"APPROVED_RESEARCH_GRID")
        self.assertEqual(audit["tested"],8)
        self.assertGreater(audit["rejected"],0)

    def test_unknown_risk_blocks_with_reason_not_fake_no_signal(self):
        buys,audit=choose_daily_buys(
            budget_cny=100,deficits_cny={"B":100},
            proposal_for_budget=lambda x: {"B":x},
            risk_checker=lambda b:{"status":"RISK_UNKNOWN","reason":"MISSING_MARK"})
        self.assertEqual(buys,{})
        self.assertEqual(audit["reason"],"MISSING_MARK")
        self.assertEqual(audit["tested"],0)

    def test_b6_grid_includes_a7_only_when_explicit(self):
        old=build_ablation_rules()
        new=build_ablation_rules(include_v13=True)
        self.assertEqual(len(old),9)
        self.assertEqual(len(new),10)
        self.assertEqual(new[-1]["rule_id"],"A7_risk_budgeted_amount")
        self.assertEqual(new[-1]["params"]["B6"]["amount_mode"],"risk_budgeted")

    def test_risk_history_only_held_assets(self):
        days=pd.bdate_range("2024-01-01",periods=300)
        def history(ds):
            return [{"date":d,"total_return_nav":1.} for d in ds]
        state={"date":days[-1],"cash":.0,"value":1000,
               "value_quality":"COMPLETE",
               "holdings":{"B":{"value_pub":1000.,"shares":100}},
               "fund_nav_history":{"B":history(days),"STALE":history(days[:5])}}
        result=scenario_risk_gate(state,n_paths=2,seed=3)
        self.assertEqual(result["status"],"PASS",result)


class TestBondResilienceDates(unittest.TestCase):
    def test_exact_end_required_and_dividend_adjusted(self):
        days=pd.bdate_range("2026-10-29",periods=10)
        d1=pd.DataFrame({"date":days,"nav":np.full(10,1.)})
        d2=pd.DataFrame({"date":days[:-1],"nav":np.full(9,1.02)})
        funds=pd.DataFrame([
            {"code":"021559","name":"Test Bond A","type":"债券型-长债"},
            {"code":"012345","name":"Stale Bond B","type":"债券型-长债"}])
        def load(code):return d1 if code=="021559" else d2
        with (patch("fundkit.storage.load_fund_list",return_value=funds),
              patch("fundkit.storage.load_nav",side_effect=load),
              patch("fundkit.storage.has_nav",return_value=True),
              patch("fundkit.storage.load_dividends",return_value={
                  "021559":[(str(days[2].date()),0.01)]}),
              patch("fundkit.storage.load_splits",return_value={}),
              patch("fundkit.storage.load_trade_status",side_effect=FileNotFoundError)):
            ranked,audit=screen_bond_resilience(
                baseline=str(days[0].date()),end=str(days[-1].date()),
                min_days=6)
        self.assertEqual(len(ranked),1)
        self.assertEqual(ranked.iloc[0]["code"],"021559")
        self.assertAlmostEqual(ranked.iloc[0]["return_since_shock"],.01,places=6)
        self.assertEqual(audit["rejection_reasons"]["missing_end"],1)


if __name__=="__main__":
    unittest.main()
