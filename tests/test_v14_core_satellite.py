import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fundkit import simulator
from fundkit.v13_portfolio import RobustSOCPConfig, solve_robust_socp
from fundkit.v14_core_satellite import (
    Limits, combined_drawdown, fixed_weights, make_policy, stress_paths,
)


GROUPS = {"c1": "core_short", "c2": "core_policy", "c3": "core_rate",
          "c4": "core_policy", "a": "china_equity", "g": "gold",
          "u": "us_equity", "v": "us_equity", "e": "enhanced"}


def state(marked, *, cash=None, flow=10., hist=(1.,), locked=(), pending=0.):
    cash = 100. - sum(marked.values()) - pending if cash is None else cash
    return {"value": sum(marked.values()) + cash + pending, "cash": cash,
            "value_quality": "COMPLETE", "unit_nav_history": list(hist),
            "external_cash_inflow_today": flow,
            "nav": dict.fromkeys(GROUPS, {"nav": 1.}),
            "holdings": {c: {"value_pub": x, "shares": x,
                             "redeemable_shares": 0. if c in locked else x}
                         for c, x in marked.items() if x > 0}}


class CoreSatelliteTest(unittest.TestCase):
    def test_shared_satellite_cap_and_cash(self):
        lim = Limits(.15)
        target = fixed_weights(GROUPS, lim, .05)
        self.assertAlmostEqual(sum(target.values()), 1.)
        self.assertAlmostEqual(sum(target[c] for c in "aguv"), .12)
        self.assertAlmostEqual(target["u"], target["v"])
        self.assertAlmostEqual(target["cash"], .02)
        for _, codes, cap in lim.constraints(GROUPS):
            self.assertLessEqual(sum(target[c] for c in codes), cap + 1e-9)
        with self.assertRaisesRegex(ValueError, "required core"):
            fixed_weights({"a": "china_equity"}, lim)

    def test_sale_only_rebalances_cap_and_respects_locks(self):
        lim = Limits(.10)
        target = fixed_weights(GROUPS, lim)
        p = make_policy(GROUPS, target, lim, fee_rate=.006)
        order = p(state({"a": 12., "g": 1., "c1": 20.}, locked=("g",)))
        self.assertEqual(order["subscribe"], {})
        self.assertGreater(order["redeem"]["a"], 4.)
        self.assertNotIn("g", order["redeem"])
        blocked = p(state({"a": 12., "g": 1., "c1": 20.}, locked=("a", "g")))
        self.assertNotIn("a", blocked["subscribe"])
        self.assertNotIn("g", blocked["subscribe"])
        self.assertIn("CAP_REDEMPTION_BLOCKED", [e["action"] for e in blocked["events"]])

    def test_historical_drawdown_not_reset(self):
        groups = {"a": "china_equity"}
        pre = combined_drawdown({"a": 10.}, 100., .02, groups)[1]
        self.assertAlmostEqual(pre, .02 + .98*.02)
        lim = Limits(.10)
        target = fixed_weights(GROUPS, lim)
        order = make_policy(GROUPS, target, lim)(state(
            {"c1": 21., "c2": 21., "c3": 21., "c4": 21.}, hist=(1.05, 1.0)))
        self.assertEqual(order["subscribe"], {})
        self.assertTrue(order["events"][-1]["stress_blocked"])

    def test_only_available_cash_and_fixed_inflow(self):
        lim = Limits(.10)
        target = fixed_weights(GROUPS, lim)
        p = make_policy(GROUPS, target, lim, trade_limits={"c1": .4})
        order = p(state({}, cash=5., pending=95., flow=1.))
        self.assertLessEqual(sum(order["subscribe"].values()), 1. + 1e-9)
        self.assertLessEqual(order["subscribe"].get("c1", 0), .4)
        none = p(state({}, cash=1., pending=99., flow=10.))
        self.assertEqual(none["subscribe"], {})

    def test_socp_has_independent_hard_stress_and_group_constraints(self):
        g = {"c1": "core_short", "a": "china_equity", "g": "gold"}
        r = np.tile([.00005, .0003, .00035], (40, 1))
        cov = np.zeros((3, 3))
        ans = solve_robust_socp(r, r[None], [.01, .2, .21], cov, cov,
              group_upper=[((1, 2), .15)], hard_stress_returns=stress_paths(list(g), g),
              config=RobustSOCPConfig(drawdown_budget=.025, turnover_cost=0.))
        self.assertEqual(ans["status"], "OPTIMAL")
        self.assertLessEqual(ans["weights"][1:3].sum(), .15 + 1e-6)
        self.assertLessEqual(max(ans["hard_stress_mdd"]), .025 + 1e-6)
        # With no group cap, the independent stress still binds.
        ans = solve_robust_socp(r, r[None], [.01, .2, .21], cov, cov,
              hard_stress_returns=stress_paths(list(g), g),
              config=RobustSOCPConfig(drawdown_budget=.025, turnover_cost=0.))
        self.assertEqual(ans["status"], "OPTIMAL")
        self.assertLess(ans["weights"][1:3].sum(), .16)
        self.assertGreater(max(ans["hard_stress_mdd"]), .024)


class AttributionTest(unittest.TestCase):
    def test_reconciles_price_dividend_split_interest_and_fees(self):
        days = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07",
                               "2026-01-08", "2026-01-09"])
        p = dict(simulator.SIM_DEFAULTS, subscribe_fee_rate=.01, redeem_fee_rate=.01,
                 cash_annual_rate=.02, initial_investment_fee_cny=1.,
                 initial_investment_fee_by_code={"a": 1.}, initial_buy_amount_by_code={"a": 51.})
        with patch.object(simulator.storage, "load_dividends", return_value={"a":[("2026-01-06", .05)]}), \
             patch.object(simulator.storage, "load_splits", return_value={"a":[("2026-01-07", 2.)]}):
            s = simulator.Simulator(["a"], simulator.SimConfig(
                budget_per_day=5., params=p, track_attribution=True,
                track_exposures=True, include_signal_history=False))
        s.calendar = days
        s._actual_nav = {"a": pd.Series([1., .95, .475, .4, .4], index=days)}

        def policy(x):
            if x["date"] == days[1]:
                return {"subscribe": {"a": 5.}}
            if x["date"] == days[3]:
                return {"redeem": {"a": 10.}}
            return {}

        result = s._execute({"a": [simulator.Lot("a", 50., days[0])]}, 50., policy)
        at = result["attribution"]
        self.assertAlmostEqual(sum(at["total_return_contribution"].values()),
                               result["metrics"]["total_return_twr"])
        self.assertAlmostEqual(sum(at["drawdown"]["loss_contribution"].values()),
                               result["metrics"]["max_drawdown"])
        self.assertEqual(at["drawdown"]["trough_date"], "2026-01-08")
        self.assertLess(at["total_return_contribution"]["fee_a"], 0.)
        self.assertGreater(at["total_return_contribution"]["cash_interest"], 0.)
        self.assertTrue(np.allclose(result["weights"][["a", "cash", "pending"]].sum(axis=1), 1.))
        self.assertAlmostEqual(at["daily"].loc[1, "a"], 0.)
        self.assertAlmostEqual(at["daily"].loc[2, "a"], 0.)


class RunnerInputsTest(unittest.TestCase):
    def test_postdated_shortlist_requires_diagnostic(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_v14_research.py"
        spec = importlib.util.spec_from_file_location("runner14", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "pool.json"
            config.write_text(json.dumps({"candidate_asof": "2026-11-12",
                                          "funds": {"000218": "gold"}}))
            with self.assertRaisesRegex(ValueError, "LOOKAHEAD_BLOCKED"):
                runner.load_universe(config, pd.Timestamp("2026-05-12"), False)
            self.assertEqual(runner.load_universe(config, pd.Timestamp("2026-05-12"), True)[0],
                             {"000218": "gold"})

    def test_cli_replays_local_nav_and_optimizes_without_account_snapshot(self):
        root = Path(__file__).resolve().parents[1]
        runner = root/"scripts/run_v14_research.py"
        config = json.loads((root/"config/v14_core_satellite.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/"data/nav").mkdir(parents=True)
            (base/"pool.json").write_text(json.dumps(config))
            days = pd.bdate_range("2024-01-01", "2026-06-30")
            codes = list(config["funds"])
            for i, code in enumerate(codes):
                r = .0001 + .0001*np.sin(np.arange(len(days))/7+i)
                if config["funds"][code] in {"china_equity", "gold"}:
                    r[days >= pd.Timestamp("2026-06-01")] -= .006
                pd.DataFrame({"date":days, "nav":np.cumprod(1+r)}).to_csv(
                    base/f"data/nav/{code}.csv", index=False)
            pd.DataFrame({"code":codes, "sub_status":"开放申购", "red_status":"开放赎回",
                          "daily_limit":0}).to_csv(base/"data/fund_status.csv", index=False)
            (base/"data/dividends.csv").write_text("code,ex_date,div\n")
            cmd = [sys.executable, str(runner), "--start", "2026-05-01", "--end", "2026-06-30",
                   "--universe", "pool.json", "--diagnostic", "--optimizer",
                   "--satellite-caps", ".10", "--enhancement-target", ".05"]
            result = subprocess.run(cmd, cwd=base, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            out = base/"reports/v14_core_satellite"
            doc = json.loads((out/"summary.json").read_text())
            self.assertEqual(doc["status"], "RESEARCH_ONLY_COMPLETE", result.stdout+result.stderr)
            self.assertEqual(doc["evidence"], "RETROSPECTIVE_SHORTLIST_LOOKAHEAD_DIAGNOSTIC")
            self.assertEqual(set(doc["cases"]), {"fixed_sat_0", "fixed_sat_10", "buy_only_sat_10",
                                                  "socp_sat_10", "enhanced_sat_10"})
            for k, case in doc["cases"].items():
                self.assertGreater(case["buys"], 0)
                self.assertAlmostEqual(sum(case["attribution"]["drawdown"]["loss_contribution"].values()),
                                       case["max_drawdown"])
                audit = pd.read_csv(out/f"{k}_execution_audit.csv")
                self.assertNotIn("amount", audit.columns)
            self.assertLess(doc["cases"]["socp_sat_10"]["allocation_audit"]["training_end"], "2026-05-01")


if __name__ == "__main__":
    unittest.main()
