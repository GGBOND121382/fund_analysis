import unittest
import pandas as pd

from fundkit.policies import make_b4


class B4StyleTimelineTests(unittest.TestCase):
    def test_uses_latest_style_signal_not_future_signal(self):
        params = {
            "target_weights": {"F1": .5, "cash": .5},
            "equity_index_codes": [],
            "daily_limits": {},
            "B4": {"pause_cdar95": .02,
                   "pause_shift_states": ["POSSIBLE_SHIFT", "CONFIRMED_SHIFT"],
                   "reenter_cooldown_days": 1,
                   "reenter_cdar95_low": .01},
            "_style_signals_by_date": {
                "2025-01-10": {"F1": {
                    "combined_shift_evidence": {"state": "POSSIBLE_SHIFT"},
                    "risk_views": {"CURRENT_STYLE_REWEIGHTED": {"cdar95": .03}}}},
                "2025-01-20": {"F1": {
                    "combined_shift_evidence": {"state": "NO_SHIFT"},
                    "risk_views": {"CURRENT_STYLE_REWEIGHTED": {"cdar95": .005}}}},
            },
        }
        policy = make_b4(params)
        state = {"cash": 1000.0, "pending_value": 0.0,
                 "value_quality": "COMPLETE", "holdings": {}, "nav": {},
                 "date": pd.Timestamp("2025-01-15")}
        first = policy(state)
        self.assertEqual(first.get("subscribe", {}), {})
        state["date"] = pd.Timestamp("2025-01-25")
        second = policy(state)
        self.assertGreater(second.get("subscribe", {}).get("F1", 0), 0)


if __name__ == "__main__":
    unittest.main()