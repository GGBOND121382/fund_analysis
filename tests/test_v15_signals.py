from pathlib import Path
import tempfile
import json
import subprocess
import sys

import numpy as np
import unittest

import pandas as pd

from fundkit.v15_signals import IndexHistory, TrendState, ValuationHistory, market_signal
from fundkit.v14_core_satellite import Limits, fixed_weights, make_policy


class V15SignalTest(unittest.TestCase):
    def test_asof_vintage_and_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"valuation.csv"
            d = pd.date_range("2019-01-01", periods=40, freq="MS")
            rows = pd.DataFrame({"index_code": "000300", "observed_at": d,
                                 "published_at": d, "pe": 10., "pb": 2.})
            rows.loc[38, ["pe", "pb"]] = [5., 1.]
            rows.loc[39, ["pe", "pb"]] = [20., 5.]
            rows.loc[39, "published_at"] = d[-1]+pd.Timedelta(hours=16)
            rows.to_csv(path, index=False)
            h = ValuationHistory(path)
            self.assertEqual(h.signal(d[-1])["factor"], 1.25)
            self.assertEqual(h.signal(d[-1]+pd.Timedelta(days=1))["factor"], .75)
            self.assertIsNone(h.signal(d[-1]+pd.Timedelta(days=70)))

    def test_index_asof_not_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"index.csv"
            p.write_text("fund_code,observed_at,published_at,total_return_index\n"
                         "110020,2026-05-01,2026-05-01 15:00:00,100\n"
                         "110020,2026-05-02,2026-05-02 15:00:00,1\n")
            h = IndexHistory(p)
            self.assertEqual(len(h.rows("110020", "2026-05-02")), 1)
            self.assertEqual(h.rows("110020", "2026-05-02")[0]["total_return_nav"], 100)

    def test_trend_steps_need_unique_marks_and_risk_for_second_exit(self):
        z = TrendState()
        days = pd.bdate_range("2026-01-01", periods=25)
        for i in range(5):
            s = {"date": days[i], "ratio": .95}
            z.update(s, allow_exit=False)
            for _ in range(6):
                z.update(s, allow_exit=False)
            self.assertEqual(z.level, 2 if i<4 else 1)
        for day in days[5:10]:
            z.update({"date": day, "ratio": .95}, allow_exit=False)
        self.assertEqual(z.level, 1)
        z.update({"date": days[10], "ratio": .95}, allow_exit=True)
        self.assertEqual(z.level, 0)
        for day in days[11:21]:
            z.update({"date": day, "ratio": 1.02}, allow_exit=True)
        self.assertEqual(z.level, 2)

    def test_dip_priority_does_not_add_money_or_override_pressure(self):
        g = {"c1":"core_short", "c2":"core_rate", "c3":"core_policy",
             "c4":"core_rate", "110020":"china_equity"}
        lim = Limits(.10)
        target = fixed_weights(g, lim)
        days = pd.bdate_range("2025-01-01", periods=127)
        x = list(range(100, 225)) + [216.]
        hist = [{"date": d.isoformat(), "total_return_nav": v} for d,v in zip(days[1:],x)]
        self.assertTrue(market_signal(hist, days[-1])["dip"])
        p = make_policy(g, target, lim, variant="dip")
        state = {"date":days[-1], "value":100., "cash":100., "value_quality":"COMPLETE",
                 "unit_nav_history":[1.], "external_cash_inflow_today":10.,
                 "nav":dict.fromkeys(g, 1.), "holdings":{},
                 "fund_nav_history":{"110020":hist}}
        o = p(state)
        self.assertGreater(o["subscribe"]["110020"], 0.)
        self.assertLessEqual(sum(o["subscribe"].values()), 10.+1e-9)
        state["unit_nav_history"] = [1., .95]
        self.assertEqual(p(state)["subscribe"], {})

    def test_high_valuation_can_trim_without_breaching_core_cap(self):
        class Expensive:
            def signal(self, day):
                return {"rank":.95, "factor":.75, "asof":"2026-01-02"}
        g = {"c1":"core_short", "c2":"core_rate", "c3":"core_policy",
             "c4":"core_rate", "110020":"china_equity"}
        lim = Limits(.10)
        target = fixed_weights(g, lim)
        p = make_policy(g,target,lim,variant="valuation",valuation_history=Expensive())
        state = {"date":pd.Timestamp("2026-01-05"),"value":100.,"cash":94.,
                 "value_quality":"COMPLETE", "unit_nav_history":[1.],
                 "external_cash_inflow_today":10., "nav":dict.fromkeys(g,1.),
                 "holdings":{"110020":{"value_pub":6.,"shares":6.,"redeemable_shares":6.}}}
        o = p(state)
        self.assertIn("110020", o["redeem"])
        self.assertEqual(o["subscribe"], {})
        self.assertIn("VALUATION_REBALANCE", [e["action"] for e in o["events"]])

    def test_layered_cli_values_all_cases_on_common_dates(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base/"data"
            (data/"nav").mkdir(parents=True)
            pool = {"000001":"core_short", "000002":"core_policy", "000003":"core_rate",
                    "000004":"core_policy", "110020":"china_equity", "000218":"gold"}
            (base/"pool.json").write_text(json.dumps(
                {"candidate_asof":"2026-11-12","funds":pool}))
            days = pd.bdate_range("2025-01-01", "2026-03-31")
            for c,g in pool.items():
                rates = np.where(days < "2026-02-01", .002, -.01) if g in ("china_equity","gold") else .00012
                pd.DataFrame({"date":days, "nav":np.cumprod(np.broadcast_to(1+rates, len(days)))}).to_csv(
                    data/"nav"/(c+".csv"), index=False)
            pd.DataFrame({"code":list(pool), "sub_status":"开放申购","red_status":"开放赎回",
                          "daily_limit":0}).to_csv(data/"fund_status.csv", index=False)
            (data/"dividends.csv").write_text("code,ex_date,div\n")
            months = pd.date_range("2019-01-01","2026-03-01",freq="MS")
            pd.DataFrame({"index_code":"000300","observed_at":months,"published_at":months,
                          "pe":np.linspace(8.,18.,len(months)),"pb":2.}).to_csv(
                base/"valuation.csv", index=False)
            cmd = [sys.executable,str(root/"scripts/run_v15_research.py"),"--start","2026-01-02",
                   "--end","2026-03-31","--diagnostic","--universe","pool.json",
                   "--valuation-csv","valuation.csv","--satellite-caps",".10"]
            result = subprocess.run(cmd,cwd=base,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            out = base/"reports/v15_layered"
            doc = json.loads((out/"summary.json").read_text())
            self.assertEqual(doc["status"],"RESEARCH_ONLY_COMPLETE",result.stdout)
            self.assertEqual(len(doc["layer_deltas"]),4)
            self.assertEqual(len({(c["test_start"],c["test_end"]) for c in doc["cases"].values()}),1)
            self.assertFalse(doc["long_term_target_validated"])
            self.assertTrue((out/"comparison.csv").exists())
            blocked = subprocess.run([*cmd,"--valuation-csv","missing.csv","--output","blocked"],
                                     cwd=base,capture_output=True,text=True,timeout=60)
            self.assertEqual(blocked.returncode,0,blocked.stderr+blocked.stdout)
            other = json.loads((base/"blocked/summary.json").read_text())
            self.assertEqual(other["cases"]["valuation_sat_10"]["status"],"DATA_BLOCKED")
            self.assertEqual(other["status"],"PARTIAL_DATA_BLOCKED")


if __name__ == "__main__":
    unittest.main()
