"""Regression tests for exact OOS/P3 accounting repairs (2026-11-25)."""
import unittest

import numpy as np
import pandas as pd

from fundkit.industry_rank import NavPath
from scripts.run_oos_model_comparison import future_perf, ROUNDTRIP_FEE
from scripts.run_portfolio_risk_budget import optimize_weights, walk_forward


class OOSAccounting(unittest.TestCase):
    def make_paths(self, missing_middle=False, stale_end=False):
        dates = pd.bdate_range("2025-01-02", periods=150)
        paths = {}
        for i in range(10):
            # Returns depend on actual dates, not array position.
            values = (1.0 + 0.001 * np.arange(len(dates)) * (i + 1) / 10)
            d, v = dates, values
            if i == 2 and missing_middle:
                d, v = d.delete(20), np.delete(v, 20)
            if i == 3 and stale_end:
                d, v = d[:80], v[:80]
            paths[f"{i:06d}"] = NavPath(d, v)
        return paths, dates

    def test_common_calendar_intersection_not_positional_mean(self):
        paths, dates = self.make_paths(missing_middle=True)
        end = dates[-1]
        # One dated fund observation removed, common calendar still supports
        # the entire requested period. All funds share identical first/last.
        perf = future_perf(paths, list(paths), dates[0], 7)
        self.assertIsNotNone(perf)
        last_values = [p.value[-1] / p.value[0] for p in paths.values()]
        expected = ((1 - ROUNDTRIP_FEE / 2) ** 2
                    * np.mean(last_values) - 1)
        self.assertAlmostEqual(perf["net_return"], expected, places=12)
        self.assertEqual(perf["n_funds"], 10)

    def test_stale_horizon_is_not_silently_used(self):
        paths, dates = self.make_paths(stale_end=True)
        self.assertIsNone(future_perf(paths, list(paths), dates[0], 7))

    def test_missing_fund_is_not_silently_removed(self):
        paths, dates = self.make_paths()
        paths.pop("000002")
        self.assertIsNone(future_perf(paths, [f"{i:06d}" for i in range(10)], dates[0], 7))


class PortfolioRiskAccounting(unittest.TestCase):
    def test_no_lookahead_walk_forward(self):
        rows = walk_forward(pd.DataFrame({"code": ["123456"]}), {},
                            0.01, ["2025-05-20", "2025-11-20"])
        self.assertTrue((rows.status == "PIT_SELECTION_DATA_BLOCKED").all())

    def test_roundtrip_fee_affects_scenario_drawdown_and_forecast(self):
        paths = np.ones((4, 3, 1))
        cash = np.ones((4, 3))
        dates = pd.bdate_range("2025-01-01", periods=3)
        historical = pd.DataFrame({"123456": np.ones(3)}, index=dates)
        out = optimize_weights(paths, cash, np.array([0.0, 0.10]),
                               historical, budget=0.04, samples=5, seed=42)
        self.assertIsNotNone(out)
        self.assertGreater(out["p95_mdd"], 0.0)
        self.assertGreater(out["hist_mdd"], 0.0)
        self.assertLess(out["expected_annual"], 0.10)


if __name__ == "__main__":
    unittest.main()
