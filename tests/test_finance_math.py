"""Run: python -m unittest discover -s reference -p 'test_*.py' -v"""
import unittest
from finance_math import (
    max_drawdown, drawdown_path, empirical_cdar, external_cash_unitization,
    single_fund_stress_impact, check_new_cash_plan, signal_observable,
)


class FinanceMathTests(unittest.TestCase):
    def test_mdd_is_five_percent(self):
        self.assertAlmostEqual(max_drawdown([1.0, 1.1, 1.045]), 0.05)

    def test_mdd_no_loss(self):
        self.assertEqual(max_drawdown([1.0, 1.001, 1.003]), 0.0)

    def test_unitization_flow_not_return(self):
        units, nav = external_cash_unitization(100, 100, 20)
        self.assertAlmostEqual(units, 120)
        self.assertAlmostEqual(nav, 1.0)
        self.assertEqual(max_drawdown([1.0, nav]), 0.0)

    def test_invalid_drawdown_path_rejected(self):
        with self.assertRaises(ValueError):
            max_drawdown([1.0, -0.5])

    def test_discrete_cdar(self):
        draws = [0.0, 0.02, 0.04, 0.1]
        self.assertAlmostEqual(empirical_cdar(draws, alpha=0.75), 0.1)
        self.assertAlmostEqual(max(drawdown_path([1.0, 1.1, 1.045])), 0.05)

    def test_fuan_stress(self):
        self.assertAlmostEqual(single_fund_stress_impact(109056.19, 195123.58, .10),
                               0.0558908313, places=7)

    def test_forbidden_equity_new_buys(self):
        with self.assertRaises(ValueError):
            check_new_cash_plan({'110020': 50}, 955, 1005, {'110020'})

    def test_cash_budget_and_fund_limit(self):
        check_new_cash_plan({'004400': 500, '021559': 200, '000218': 100},
                            205, 1005, {'110020'}, {'004400': 3000})
        with self.assertRaises(ValueError):
            check_new_cash_plan({'004400': 3010}, 0, 3010, set(), {'004400': 3000})

    def test_asof_no_lookahead(self):
        self.assertFalse(signal_observable(100.1, 100.0))
        self.assertTrue(signal_observable(100.0, 100.0))


if __name__ == '__main__':
    unittest.main()