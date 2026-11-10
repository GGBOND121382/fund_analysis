import unittest

import pandas as pd

from fundkit.evaluation import _select_risk_first


class EvaluationSelectionTests(unittest.TestCase):
    def test_feasible_candidates_rank_sharpe_before_cagr(self):
        trials = pd.DataFrame([
            {"max_drawdown": .03, "cagr": .065, "sharpe": 1.4},
            {"max_drawdown": .035, "cagr": .055, "sharpe": 1.9},
            {"max_drawdown": .05, "cagr": .09, "sharpe": 3.0},
        ])
        selected = _select_risk_first(trials)
        self.assertAlmostEqual(selected["sharpe"], 1.9)

    def test_when_no_candidate_meets_drawdown_budget_choose_lowest_drawdown(self):
        trials = pd.DataFrame([
            {"max_drawdown": .05, "cagr": .08, "sharpe": 2.0},
            {"max_drawdown": .045, "cagr": .04, "sharpe": 1.0},
        ])
        selected = _select_risk_first(trials)
        self.assertAlmostEqual(selected["max_drawdown"], .045)


if __name__ == "__main__":
    unittest.main()
