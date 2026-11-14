import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from fundkit.defensive_screen import Shock, adjusted_asof, calculate_fund, category, rank_candidates
from scripts.screen_defensive_funds import screen, write_reports


class DefensiveScreenTest(unittest.TestCase):
    def setUp(self):
        self.end = pd.Timestamp("2026-11-12")
        self.shocks = [
            Shock("rates", "rates", pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-11")),
            Shock("credit", "credit", pd.Timestamp("2026-10-29"), self.end),
        ]
        self.dates = pd.bdate_range("2023-10-01", self.end)

    def test_classification(self):
        self.assertEqual(category("东财启和混合A", "混合型-偏债"), "偏债混合")
        self.assertEqual(category("汇添富稳宏6个月持有债券A", "债券型-混合一级"), "一级债基")
        self.assertEqual(category("可转债", "债券型-混合二级"), "可转债敞口")
        self.assertIsNone(category("海外债券", "QDII-纯债"))
        self.assertIsNone(category("灵活配置", "混合型-灵活"))

    def test_asof_and_dividends(self):
        dates = pd.to_datetime(["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"])
        nav = pd.DataFrame(dict(date=dates, nav=[1.00, .95, .97, 99.], acc_nav=[1., 1.00, 1.02, 99.]))
        adj = adjusted_asof(nav, self.end, [("2026-11-11", .05)])
        self.assertAlmostEqual(adj.iloc[-1], .97/.95)
        self.assertEqual(len(adj), 3)
        with self.assertRaisesRegex(ValueError, "unreconciled"):
            adjusted_asof(nav, self.end)

    def test_equal_dates_and_horizon_label(self):
        dates = self.dates[self.dates >= "2025-12-26"]
        nav = pd.Series(np.linspace(1, 1.08, len(dates)), index=dates)
        stats = calculate_fund(nav, self.end, self.shocks)
        self.assertEqual(stats["horizon_tier"], "6m")
        self.assertNotIn("return_1y", stats)
        self.assertAlmostEqual(stats["available_history_return"], .08)
        self.assertEqual(stats["shock_observations"], 2)
        missing = nav.drop(pd.Timestamp("2026-06-10"))
        self.assertEqual(calculate_fund(missing, self.end, self.shocks)["shock_observations"], 1)
        stale = calculate_fund(nav.iloc[:-1], self.end, self.shocks)
        self.assertFalse(stale["asof_exact"])
        self.assertNotIn("return_6m", stale)

    def test_scoring_composes_axes_and_keeps_locked_fund(self):
        def row(code, r, mdd, w, locked=False):
            return dict(code=code, group="一级债基", horizon_tier="1y", asof_exact=True,
                        shock_observations=2, selection_status="", product=code, share="A",
                        nav_observations=250, return_1y=r, mdd_1y=mdd,
                        worst_shock_return=-w, worst_shock_mdd=w, median_shock_mdd=w,
                        has_lock=locked, sub_status="暂停申购", daily_limit=0)
        data = pd.DataFrame([row("A", .07, .009, .005, True),
                             row("B", .035, .008, .007),
                             row("C", .026, .030, .020)])
        full, short = rank_candidates(data, shock_count=2, top=1)
        self.assertEqual(short.code.tolist(), ["A"])
        self.assertTrue(bool(short.iloc[0].has_lock))
        self.assertFalse(bool(full.loc[2, "pareto"]))
        self.assertEqual(full.loc[2, "selection_status"], "未入围")

    def test_full_run_duplicate_and_separate_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/"nav").mkdir()
            funds = [("000001", "示例6个月持有债券A", "债券型-混合一级"),
                     ("000002", "示例6个月持有债券C", "债券型-混合一级"),
                     ("000003", "示例混合A", "混合型-偏债"),
                     ("000004", "无数据混合A", "混合型-偏债")]
            pd.DataFrame(funds, columns=["code", "name", "type"]).to_csv(p/"fund_list.csv", index=False)
            for c, _, typ in funds[:3]:
                dates = self.dates if typ.startswith("债券") else self.dates[-196:]
                nav = np.linspace(1., 1.1, len(dates))
                pd.DataFrame(dict(date=dates, nav=nav, acc_nav=nav)).to_csv(p/"nav"/f"{c}.csv", index=False)
            full, short = screen(p, self.end, self.shocks, top=3, min_peers=1)
            status = full.set_index("code")
            self.assertEqual(status.loc["000002", "selection_status"], "同基金其他份额")
            self.assertEqual(status.loc["000004", "selection_status"], "无净值缓存")
            self.assertEqual(short.set_index("code").horizon_tier.to_dict(), {"000001":"3y","000003":"6m"})
            audit = write_reports(p/"out", full, short, self.shocks, self.end, ["000001","999999"])
            self.assertEqual(audit["exact_asof_share_codes"], 3)
            self.assertIn("清单中无该代码", (p/"out"/"summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
