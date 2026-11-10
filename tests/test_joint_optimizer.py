import unittest

import numpy as np
import pandas as pd

from fundkit.joint_optimizer import (
    JointSearchConfig,
    build_rolling_folds,
    build_rule_grid,
    optimize_joint_fold,
)
from fundkit.weight_optimizer import (
    WeightSearchConfig,
    generate_weight_candidates,
)


class JointOptimizerTests(unittest.TestCase):
    def test_rule_grid_contains_baseline_rebalance_style_and_drawdown_rules(self):
        rules = build_rule_grid({"710301": "bond", "004400": "bond", "000218": "gold"})
        families = {x["family"] for x in rules}
        self.assertEqual(families, {"B2", "B3", "B4", "B5"})
        self.assertEqual(len(rules), 21)
        self.assertTrue(all(x["rule_id"] for x in rules))
        for rule in rules:
            if rule["family"] == "B5":
                p = rule["params"]["B5"]
                self.assertLess(p["fund_reenter_drawdown"], p["fund_pause_drawdown"])
                self.assertEqual(p["mdd_budget"], .04)
                self.assertTrue(p["allow_redeem"])
                self.assertLess(p["fund_pause_drawdown_by_code"]["710301"],
                                p["fund_pause_drawdown_by_code"]["000218"])

    def test_validation_selects_joint_weight_and_exit_rule_then_tests_once(self):
        weights = [
            {"weights": {"A": .6, "cash": .4}, "turnover_proxy": .2},
            {"weights": {"A": .4, "cash": .6}, "turnover_proxy": .1},
        ]
        rules = [
            {"family": "B2", "params": {}, "rule_id": "B2"},
            {"family": "B5", "params": {"B5": {"internal_drawdown_margin": .01}},
             "rule_id": "B5"},
        ]
        calls = []
        def evaluator(w, rule, window, fee):
            calls.append((tuple(sorted(w.items())), rule["rule_id"], window, fee))
            if window[0] == "2024-01-01":
                if w["A"] == .6 and rule["family"] == "B2":
                    cagr, mdd = .075 - fee, .046
                elif w["A"] == .6 and rule["family"] == "B5":
                    cagr, mdd = .064 - fee, .035
                elif w["A"] == .4 and rule["family"] == "B2":
                    cagr, mdd = .052 - fee, .025
                else:
                    cagr, mdd = .049 - fee, .020
            else:
                cagr, mdd = .048 - fee, .041
            return {"cagr": cagr, "max_drawdown": mdd, "sharpe": 1.0,
                    "sortino": 1.2, "cdar95": mdd * .7, "xirr": cagr,
                    "days": 252, "redeem_events": 1}
        cfg = JointSearchConfig(fee_stress_rates=(0.0, .003),
                                min_validation_days=252, min_test_days=252)
        out = optimize_joint_fold(weights, rules, evaluator,
                                  ("2024-01-01", "2024-12-31"),
                                  ("2025-01-01", "2025-12-31"), cfg)
        self.assertEqual(out["selected"]["family"], "B5")
        self.assertEqual(out["selected"]["weights"]["A"], .6)
        self.assertEqual(out["status"], "OOS_TARGET_UNMET")
        # 2 weights × 2 rules × 2 fee assumptions for validation, then only
        # the selected pair × 2 fees for test.
        self.assertEqual(len(calls), 10)
        test_calls = [x for x in calls if x[2][0] == "2025-01-01"]
        self.assertEqual(len(test_calls), 2)
        self.assertEqual({x[1] for x in test_calls}, {"B5"})

    def test_no_validation_feasible_candidate_is_explicit(self):
        weights = [{"weights": {"A": 1.0, "cash": 0.0}}]
        rules = [{"family": "B2", "params": {}, "rule_id": "B2"}]
        def evaluator(w, r, window, fee):
            return {"cagr": .03, "max_drawdown": .02, "days": 252}
        out = optimize_joint_fold(weights, rules, evaluator,
                                  ("2024-01-01", "2024-12-31"),
                                  ("2025-01-01", "2025-12-31"))
        self.assertEqual(out["status"], "NO_VALIDATION_FEASIBLE_CANDIDATE")
        self.assertFalse(out["trade_ready"])

    def test_short_validation_window_is_blocked_before_candidate_selection(self):
        weights = [{"weights": {"A": 1.0, "cash": 0.0}}]
        rules = [{"family": "B2", "params": {}, "rule_id": "B2"}]
        def evaluator(w, r, window, fee):
            return {"cagr": .06, "max_drawdown": .03, "days": 100}
        cfg = JointSearchConfig(min_validation_days=252, min_test_days=252,
                                fee_stress_rates=(0.0,))
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_VALIDATION_HORIZON"):
            optimize_joint_fold(weights, rules, evaluator,
                                ("2024-01-01", "2024-05-01"),
                                ("2025-01-01", "2025-12-31"), cfg)

    def test_short_test_window_cannot_be_declared_validated(self):
        weights = [{"weights": {"A": 1.0, "cash": 0.0}}]
        rules = [{"family": "B2", "params": {}, "rule_id": "B2"}]
        def evaluator(w, r, window, fee):
            return {"cagr": .06, "max_drawdown": .03, "days": 100}
        cfg = JointSearchConfig(min_validation_days=100, min_test_days=252,
                                fee_stress_rates=(0.0,))
        out = optimize_joint_fold(weights, rules, evaluator,
                                  ("2024-01-01", "2024-05-01"),
                                  ("2025-01-01", "2025-05-01"), cfg)
        self.assertEqual(out["status"], "INSUFFICIENT_TEST_HORIZON")
        self.assertFalse(out["test_pass"])

    def test_rolling_folds_are_strictly_forward_and_nonoverlapping(self):
        dates = pd.bdate_range("2018-01-01", periods=1260)
        folds = build_rolling_folds(dates, train_days=504, validation_days=252,
                                    test_days=252, step_days=252, folds=2)
        self.assertEqual(len(folds), 2)
        for f in folds:
            self.assertLess(f["train"][1], f["validation"][0])
            self.assertLess(f["validation"][1], f["test"][0])
        self.assertLess(folds[0]["test"][1], folds[1]["test"][0])

    def test_weight_candidate_generation_respects_caps_and_freeze(self):
        dates = pd.bdate_range("2024-01-01", periods=340)
        df = pd.DataFrame({"BOND": .00025, "GOLD": .0001, "EQUITY": .0005},
                          index=dates)
        cfg = WeightSearchConfig(lookback_days=340, horizon=252,
                                  min_observations=252, candidate_count=120,
                                  bootstrap_paths=8, block_lengths=(10, 30),
                                  historical_windows=4, seed=9)
        out = generate_weight_candidates(
            df, current_weights={"cash": 1.0},
            caps={"BOND": .6, "GOLD": .6, "EQUITY": 0.0},
            fixed_weights={}, cfg=cfg, top_k=4)
        self.assertGreaterEqual(len(out["candidates"]), 1)
        self.assertLessEqual(len(out["candidates"]), 4)
        for item in out["candidates"]:
            w = item["weights"]
            self.assertAlmostEqual(sum(w.values()), 1.0, places=7)
            self.assertEqual(w["EQUITY"], 0.0)
            self.assertLessEqual(w["BOND"], .6 + 1e-9)
            self.assertLessEqual(w["GOLD"], .6 + 1e-9)

    def test_weight_candidate_generation_is_reproducible(self):
        dates = pd.bdate_range("2024-01-01", periods=340)
        df = pd.DataFrame({"A": .0003, "B": .0001}, index=dates)
        cfg = WeightSearchConfig(lookback_days=340, horizon=252,
                                  min_observations=252, candidate_count=90,
                                  bootstrap_paths=6, block_lengths=(10, 30),
                                  historical_windows=3, seed=11)
        a = generate_weight_candidates(df, current_weights={"cash": 1.0}, cfg=cfg, top_k=3)
        b = generate_weight_candidates(df, current_weights={"cash": 1.0}, cfg=cfg, top_k=3)
        self.assertEqual([x["weights"] for x in a["candidates"]],
                         [x["weights"] for x in b["candidates"]])


if __name__ == "__main__":
    unittest.main()