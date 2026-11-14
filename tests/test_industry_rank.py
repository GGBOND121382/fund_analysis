import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fundkit.defensive_screen import Shock
from fundkit.industry_rank import (NavPath, assemble, compare_forward, effective_group,
                                   feature, forward, legacy_events, pareto, ranking, stress_metrics)
from scripts.fetch_all_fund_rank_data import get_one
from scripts.run_industry_fund_ranking import load_catalog, main


def path(dates, vals):
    return NavPath(pd.DatetimeIndex(dates),np.asarray(vals,dtype=float))


def catalog_rows():
    return [("025368","示例偏债A","混合型-偏债"),
            ("000001","甲债券A","债券型-长债"),
            ("000002","乙债券A","债券型-长债"),
            ("000003","甲债券C","债券型-长债"),
            ("000004","丙债券A","债券型-长债")]


class IndustryRankTest(unittest.TestCase):
    def test_november_13_extends_instead_of_double_counting(self):
        e = [Shock(key,key,pd.Timestamp(start),pd.Timestamp(end))
             for key,start,end in (
                 ("credit_20261030","2026-10-29","2026-11-06"),
                 ("bonds_202611","2026-11-09","2026-11-12"),
                 ("bonds_202611_through_13","2026-11-09","2026-11-13"),
                 ("bonds_20261113_daily","2026-11-12","2026-11-13"))]
        start = pd.Timestamp("2026-05-13")
        self.assertEqual([x.id for x in legacy_events(e,pd.Timestamp("2026-11-12"),start,True,"domestic")],
                         ["credit_20261030","bonds_202611"])
        for current in (True,False):
            self.assertEqual([x.id for x in legacy_events(e,pd.Timestamp("2026-11-13"),start,current,"domestic")],
                             ["credit_20261030","bonds_202611_through_13"])

    def test_midmonth_no_future_or_incomplete_month(self):
        ds = pd.bdate_range("2025-12-26","2026-12-31")
        x = np.linspace(1,1.1,len(ds))
        x[ds > "2026-11-12"] *= 10
        nav = path(ds,x)
        end = pd.Timestamp("2026-11-12")
        f,_ = feature(nav,end,"6m",True)
        self.assertEqual(len(f["_monthly"]),5)
        self.assertLess(f["period_return"],.1)
        self.assertEqual(feature(nav,end,"1y",True)[1],"期限起点缺历史")
        weekend = pd.Timestamp("2026-10-31")
        self.assertIsNone(feature(nav,weekend,"6m",True)[0])
        f,_ = feature(nav,weekend,"6m",True,exact=False)
        self.assertEqual(len(f["_monthly"]),6)
        self.assertLessEqual(f["end"],weekend)

    def test_stress_recovers_peak_not_only_baseline(self):
        ds = pd.bdate_range("2026-11-09","2026-11-16")
        nav = path(ds,[100,110,99,101,109,110])
        e = Shock("a","a",ds[0],ds[3])
        r = stress_metrics(nav,e,ds[-1])
        self.assertAlmostEqual(r["stress_drawdown"],.1)
        self.assertEqual(r["recovery_date"],ds[-1])
        self.assertEqual(r["recovery_calendar_days"],5)
        self.assertEqual(stress_metrics(nav,e,ds[4])["recovery_status"],"未观察到修复")
        self.assertIsNone(stress_metrics(path(ds.delete(1),[100,99,101,109,110]),e,ds[-1]))
        self.assertIsNone(stress_metrics(nav,e,ds[2]))
        self.assertEqual(legacy_events([e],ds[2],ds[0],False,"domestic"),[])

    def test_pareto_ties_and_incomplete(self):
        z = pd.DataFrame([(1,.2,.2,-.1),(1,.2,.2,-.1),(.5,.3,.2,-.1),(2,.3,.2,-.1),(1,.2,np.nan,-.1)],
                         columns=["return_for_rank","max_drawdown","worst_stress_drawdown","worst_stress_return"])
        p = pareto(z)
        self.assertEqual(p[:4].tolist(),[True,True,False,True])
        self.assertTrue(np.isnan(p[4]))

    def test_monthly_hits_and_pure_bond_weights(self):
        z = pd.DataFrame({"code":["01","02","03"],"group":"中长纯债",
                          "_monthly":[np.array([.01,.01]),np.array([-.01,.04]),np.array([0.,0.])],
                          "return_for_rank":[.02,.03,0.],"max_drawdown":[0.,.01,0.],
                          "negative_month_loss":[0.,.01,0.],
                          "worst_stress_drawdown":[.01,.02,np.nan],
                          "worst_stress_return":[.01,-.01,np.nan]})
        x = ranking(z,min_peers=2).set_index("code")
        self.assertAlmostEqual(x.loc["01","research_score"],.50*.5+.25*.75+.25*.75)
        self.assertAlmostEqual(x.loc["02","industry_proxy_score"],.67+.33*.75)
        self.assertEqual(x.loc["01","legacy_rank"],1)
        self.assertTrue(pd.isna(x.loc["03","legacy_rank"]))
        self.assertEqual(effective_group("示例63个月开放债券","中长纯债","债券型-长债")[1],True)
        self.assertEqual(effective_group("示例6个月持有债券","中长纯债","债券型-长债")[1],False)

    def test_forward_entry_after_signal_and_missing_not_zero(self):
        ds = pd.bdate_range("2026-01-01","2026-01-09")
        nav = path(ds,[1,2,4,8,16,32,64])
        r = forward(nav,ds[1],ds[4])
        self.assertEqual(r[2],ds[2])
        self.assertAlmostEqual(r[0],3)
        self.assertIsNone(forward(nav,ds[-1],ds[-1]+pd.Timedelta(days=4)))

    def test_long_history_refresh_does_not_trust_current_tail_only(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            old = d/"total_return"/"000001.csv"
            old.parent.mkdir()
            pd.DataFrame({"date":["2023-11-11","2026-11-11","2026-11-13"],"nav":[1.,1.1,1.2]}).to_csv(old,index=False)
            longer = pd.DataFrame({"date":["2020-11-11","2023-11-11","2026-11-11","2026-11-12"],
                                   "nav":[.9,1.,1.1,1.15]})
            row = SimpleNamespace(code="000001",name="示例债券A",type="债券型-长债")
            with patch("scripts.fetch_all_fund_rank_data.read_url",return_value="x"), patch(
                "scripts.fetch_all_fund_rank_data.parse_nav",return_value=longer) as f:
                get_one(row,d,pd.Timestamp("2026-11-12"),False,5)
                self.assertEqual(f.call_args.args[-1],5)
                get_one(row,d,pd.Timestamp("2026-11-12"),False,5)
                self.assertEqual(f.call_count,1)
            self.assertEqual(pd.read_csv(old).date.tolist(),
                             ["2020-11-11","2023-11-11","2026-11-11","2026-11-13"])

    def test_pipeline_keeps_short_history_and_comparable_shares(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame(catalog_rows(),columns=["code","name","type"]).to_csv(d/"fund_list.csv",index=False)
            (d/"total_return").mkdir()
            for i,(code,_,_) in enumerate(catalog_rows()):
                ds = pd.bdate_range("2025-12-26" if not i else "2019-01-01","2026-11-12")
                v = np.exp((.001+i*.001)*np.arange(len(ds))/10)
                pd.DataFrame({"date":ds,"nav":v}).to_csv(d/"total_return"/f"{code}.csv",index=False)
            out = d/"out"
            main(["--asof","2026-11-12","--data-dir",str(d),"--output",str(out),
                  "--min-peers","2","--oos-folds","1"])
            a = pd.read_csv(out/"ranking_6m.csv",dtype={"code":str})
            b = pd.read_csv(out/"ranking_5y.csv",dtype={"code":str})
            self.assertIn("025368",a.code.tolist())
            self.assertNotIn("025368",b.code.tolist())
            self.assertNotIn("000003",a.code.tolist())
            self.assertEqual(len(pd.read_csv(out/"matched_6m.csv")),3)
            f = pd.read_csv(out/"oos_folds.csv")
            self.assertGreater(len(f),0)
            self.assertTrue(f.paired_complete.all())
            self.assertTrue((f.signal < f.evaluation_end).all())
            self.assertFalse((f.method.eq("legacy") & f.horizon.eq("6m")).any())
            self.assertTrue((f.method.eq("legacy") & f.horizon.eq("3y")).any())


if __name__=="__main__":
    unittest.main()
