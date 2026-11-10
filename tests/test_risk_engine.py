import unittest

import numpy as np
import pandas as pd

from fundkit.risk_engine import (
    estimate_robust_returns,
    portfolio_return_lcb,
    risk_contributions,
)


class RobustRiskEngineTests(unittest.TestCase):
    def test_short_sample_is_shrunk_toward_cross_sectional_prior(self):
        idx = pd.bdate_range("2024-01-01", periods=280)
        df = pd.DataFrame({
            "B1": np.full(280, 0.0008),
            "B2": np.full(280, 0.0004),
            "B3": np.full(280, 0.0003),
        }, index=idx)
        out = estimate_robust_returns(df, {"B1": "bond", "B2": "bond", "B3": "bond"})
        self.assertLess(out["B1"]["shrunk_mean_annual"],
                        out["B1"]["sample_mean_annual"])
        self.assertGreater(out["B1"]["robust_lcb_annual"], 0.05)

    def test_uncertainty_penalty_reduces_lower_bound(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2024-01-01", periods=260)
        df = pd.DataFrame({"A": rng.normal(0.0004, 0.002, 260),
                           "B": rng.normal(0.0004, 0.002, 260)}, index=idx)
        a = estimate_robust_returns(df, uncertainty_z=0.0)
        b = estimate_robust_returns(df, uncertainty_z=2.0)
        self.assertLess(b["A"]["robust_lcb_annual"], a["A"]["robust_lcb_annual"])

    def test_portfolio_lcb_requires_complete_weights(self):
        estimates = {"A": {"robust_lcb_annual": 0.06}}
        self.assertAlmostEqual(portfolio_return_lcb({"A": .5, "cash": .5}, estimates), .03)
        with self.assertRaisesRegex(ValueError, "权重之和"):
            portfolio_return_lcb({"A": .5, "cash": .4}, estimates)

    def test_risk_contributions_sum_to_one(self):
        rng = np.random.default_rng(8)
        idx = pd.bdate_range("2024-01-01", periods=120)
        common = rng.normal(0, .001, 120)
        df = pd.DataFrame({
            "A": common + rng.normal(0, .0005, 120),
            "B": common + rng.normal(0, .0007, 120),
        }, index=idx)
        out = risk_contributions(df, {"A": .6, "B": .4})
        total = sum(x["fraction"] for x in out["contributions"].values())
        self.assertAlmostEqual(total, 1.0, places=7)

    def test_missing_returns_are_blocked(self):
        idx = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame({"A": np.zeros(30)}, index=idx)
        df.iloc[3, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "DATA_BLOCKED"):
            estimate_robust_returns(df)


if __name__ == "__main__":
    unittest.main()
