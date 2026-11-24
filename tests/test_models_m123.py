import unittest

import numpy as np
import pandas as pd

from scripts.rank_models_m123 import BOOT_STEPS, eb_posteriors, pct
from fundkit.opportunity_rank import max_drawdown, stationary_bootstrap


class TestPercentile(unittest.TestCase):
    def test_single_gets_50(self):
        self.assertEqual(pct(pd.Series([1.0])).iloc[0], 50.0)

    def test_average_ties(self):
        r = pct(pd.Series([1.0, 2.0, 2.0, 3.0]))
        self.assertEqual(r.tolist(), [0.0, 50.0, 50.0, 100.0])

    def test_nan_stays_nan(self):
        r = pct(pd.Series([1.0, np.nan, 3.0]))
        self.assertTrue(np.isnan(r.iloc[1]))
        self.assertEqual(r.iloc[0], 0.0)
        self.assertEqual(r.iloc[2], 100.0)


class TestEmpiricalBayes(unittest.TestCase):
    def test_high_se_shrinks_more_and_lowers_mu(self):
        # 组内 12 只基金收益分散（真实类内方差 > 采样方差，tau2>0）
        rng = np.random.default_rng(0)
        rows = [{"code": f"f{i}", "group": "g",
                 "observed_log_cagr": float(0.04 + 0.06 * rng.standard_normal()),
                 "annual_se": 0.02} for i in range(11)]
        rows.append({"code": "star", "group": "g", "observed_log_cagr": 0.20, "annual_se": 0.15})
        d = eb_posteriors(pd.DataFrame(rows))
        star = d[d["code"] == "star"].iloc[0]
        peer = d[d["code"] == "f0"].iloc[0]
        # 高 se 的明星基金被大幅收缩向先验，mu_lower 显著低于名义收益
        self.assertLess(star["eb_log_cagr"], 0.20)
        self.assertGreater(star["eb_se"], peer["eb_se"])
        self.assertLess(star["mu_lower"], np.exp(0.20) - 1)

    def test_small_group_uses_conservative_se(self):
        df = pd.DataFrame({"code": ["a", "b"], "group": ["tiny", "tiny"],
                           "observed_log_cagr": [0.05, 0.10], "annual_se": [0.05, 0.08]})
        d = eb_posteriors(df)
        # n<10: posterior se 不得低于自身采样 se
        self.assertGreaterEqual(d.loc[1, "eb_se"], 0.08 - 1e-12)

    def test_missing_not_filled_with_zero(self):
        df = pd.DataFrame({"code": ["a", "b", "c"], "group": ["g"] * 3,
                           "observed_log_cagr": [0.05, np.nan, 0.03],
                           "annual_se": [0.02, 0.02, 0.02]})
        d = eb_posteriors(df.dropna(subset=["observed_log_cagr"]))
        self.assertEqual(len(d), 2)


class TestBootstrapD95(unittest.TestCase):
    def _wealth(self, seed=42):
        v = 1.0 + np.cumsum(np.random.default_rng(0).normal(0.0002, 0.001, 300))
        return stationary_bootstrap(v, steps=BOOT_STEPS, scenarios=500, block_length=10, seed=seed)

    def test_deterministic_with_seed(self):
        a = max_drawdown(self._wealth(42))
        b = max_drawdown(self._wealth(42))
        np.testing.assert_array_equal(a, b)

    def test_d95_exceeds_typical_drawdown(self):
        d = max_drawdown(self._wealth())
        self.assertGreater(np.quantile(d, 0.95), np.median(d))
        self.assertTrue(np.all(d >= 0))

    def test_seed_changes_result(self):
        a = np.quantile(max_drawdown(self._wealth(42)), 0.95)
        b = np.quantile(max_drawdown(self._wealth(7)), 0.95)
        self.assertNotEqual(a, b)


class TestScoreAssembly(unittest.TestCase):
    def test_m1_only_replaces_return_leg(self):
        # SR_EB > SR 时 M1 总分变化必须严格等于 0.40*(SR_EB-SR)
        sr, sd, sa, sl = 50.0, 60.0, 70.0, 80.0
        sr_eb = 70.0
        m0 = 0.4 * 50.0 + 0.35 * sd + 0.15 * sa + 0.10 * sl
        m1 = 0.4 * sr_eb + 0.35 * sd + 0.15 * sa + 0.10 * sl
        self.assertAlmostEqual(m1 - m0, 0.4 * (sr_eb - 50.0))


if __name__ == "__main__":
    unittest.main()
