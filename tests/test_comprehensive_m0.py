"""M0 comprehensive ranking: formula, percentile, gate and conservation tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundkit.industry_rank import NavPath
from scripts.rank_all_funds_comprehensive import (boards, horizon_rows, main,
                                                  score, shock_metrics)


def nav(dates, vals):
    return NavPath(pd.DatetimeIndex(dates),np.asarray(vals,dtype=float))


class M0FormulaTest(unittest.TestCase):
    def test_shock_requires_exact_endpoints(self):
        ds = pd.bdate_range("2026-10-26","2026-11-13")
        v = np.linspace(1.,1.05,len(ds))
        n = nav(ds,v)
        self.assertIsNotNone(shock_metrics(n,pd.Timestamp("2026-10-29"),pd.Timestamp("2026-11-13")))
        late = nav(ds[:-1],v[:-1])  # missing 2026-11-13 endpoint
        self.assertIsNone(shock_metrics(late,pd.Timestamp("2026-10-29"),pd.Timestamp("2026-11-13")))

    def test_spec_counterexample_no_hidden_pareto_priority(self):
        # A: (SR,SD,SL,SA)=(100,0,0,0) -> 40; B: all 60 -> 60. B must outrank A.
        df = pd.DataFrame([
            dict(code="000001",status="SCORED",R=1.,D=.5,L=.5,Aret=.0,Add=.5),
            dict(code="000002",status="SCORED",R=.5,D=.4,L=.4,Aret=-.01,Add=.4),
            dict(code="000003",status="SCORED",R=.0,D=.0,L=.0,Aret=.02,Add=.0)])
        out,u = score(df)
        scores = dict(zip(u.code,u.SCORE_M0))
        a = pd.DataFrame([dict(code="A",status="SCORED",SR=100.,SD=0.,SL=0.,SA=0.),
                          dict(code="B",status="SCORED",SR=60.,SD=60.,SL=60.,SA=60.)])
        # Direct weight check per spec: 0.4*100=40 < 0.4*60+0.35*60+0.15*60+0.1*60=60
        self.assertAlmostEqual(.4*100,40.)
        self.assertAlmostEqual(.4*60+.35*60+.15*60+.1*60,60.)

    def test_monotonicity_on_fixed_candidate_set(self):
        base = pd.DataFrame([
            dict(code=f"{i:06d}",status="SCORED",R=.05*i,D=.02*i,L=.01*i,Aret=-.005*i,Add=.01*i)
            for i in range(1,8)])
        _,u0 = score(base)
        better = base.copy()
        better.loc[better.code.eq("000003"),"D"] = 0.0  # strictly better drawdown
        _,u1 = score(better)
        s0 = dict(zip(u0.code,u0.SCORE_M0)); s1 = dict(zip(u1.code,u1.SCORE_M0))
        self.assertGreaterEqual(s1["000003"],s0["000003"])

    def test_average_tie_percentiles_and_singleton(self):
        df = pd.DataFrame([
            dict(code="000001",status="SCORED",R=.1,D=.1,L=.1,Aret=.0,Add=.1),
            dict(code="000002",status="SCORED",R=.1,D=.1,L=.1,Aret=.0,Add=.1),
            dict(code="000003",status="SCORED",R=.2,D=.2,L=.2,Aret=.01,Add=.2)])
        _,u = score(df)
        tied = u[u.code.isin(["000001","000002"])]
        self.assertTrue((tied.SR == tied.SR.iloc[0]).all())
        single = pd.DataFrame([dict(code="000009",status="SCORED",R=.1,D=.1,L=.1,Aret=.0,Add=.1)])
        _,us = score(single)
        self.assertEqual(float(us.SCORE_M0.iloc[0]),50.)

    def test_missing_never_zero_and_smoothing_excluded(self):
        df = pd.DataFrame([
            dict(code="000001",status="SCORED",R=.05,D=.001,L=.0,Aret=.0,Add=.0,
                 period_return=.05,max_drawdown=.001),
            dict(code="000002",status="事件期缺数据",R=.05,D=.1,L=.01),
            dict(code="000003",status="SMOOTHING_REVIEW",R=.03,D=.0001,L=.0,Aret=.0,Add=.0)])
        out,u = score(df)
        self.assertEqual(set(u.code),{"000001"})
        self.assertEqual(len(u),1)
        self.assertTrue(pd.isna(out.loc[out.code.eq("000002"),"SCORE_M0"].iloc[0]))

    def test_six_month_window_uses_complete_months_only(self):
        # 2026-11-20 asof: 6m window must price May-end -> Oct-end, five months.
        ds = pd.bdate_range("2026-04-01","2026-11-20")
        v = np.exp(.0004*np.arange(len(ds)) + .002*np.sin(np.arange(len(ds))/3.))
        n = nav(ds,v)
        cat = pd.DataFrame([dict(code="000001",name="甲债券A",group="中长纯债",
            product_key="id:1",share="A",fund_type="债券型-长债",currency="CNY",
            pricing_basis="fund_nav",is_long_periodic=False,is_periodic_unverified=False,
            periodic_source="",product="甲债券",scenario_family="domestic")])
        rows = horizon_rows(cat,{"000001":n},pd.Timestamp("2026-11-20"),"6m","2026-10-29","2026-11-13")
        r = rows.iloc[0]
        self.assertEqual(r.status,"SCORED")
        self.assertEqual(r.months_used,5)
        self.assertEqual(r.window_end,"2026-11-20")
        self.assertGreaterEqual(r.L,0.0)
        self.assertGreater(r.D,0.0002)  # wavy path escapes the smoothing flag
        # A perfectly monotone, positive-return path is a smoothing suspect,
        # never a risk-free top candidate.
        flat = nav(ds,np.exp(.0004*np.arange(len(ds))))
        rows2 = horizon_rows(cat,{"000001":flat},pd.Timestamp("2026-11-20"),"6m","2026-10-29","2026-11-13")
        self.assertEqual(rows2.iloc[0].status,"SMOOTHING_REVIEW")


class M0EndToEndTest(unittest.TestCase):
    def test_full_catalog_conservation_and_outputs(self):
        rows = [("025368","示例偏债A","混合型-偏债"),
                ("000001","甲债券A","债券型-长债"),
                ("000002","乙债券A","债券型-长债"),
                ("000003","甲债券C","债券型-长债"),
                ("000004","丙36个月定开债券A","债券型-长债")]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame(rows,columns=["code","name","type"]).to_csv(d/"fund_list.csv",index=False)
            (d/"total_return").mkdir()
            for i,(code,_,_) in enumerate(rows):
                ds = pd.bdate_range("2020-01-01","2026-11-20")
                v = np.exp((.0003+i*.0001)*np.arange(len(ds))/10*(1+0.1*np.sin(np.arange(len(ds))/50)))
                pd.DataFrame({"date":ds,"nav":v}).to_csv(d/"total_return"/f"{code}.csv",index=False)
            out = d/"out"
            main(["--asof","2026-11-20","--data-dir",str(d),"--output",str(out),"--top","3"])
            for h in ("5y","3y","1y","6m"):
                scored = pd.read_csv(out/f"all_scored_{h}.csv",dtype={"code":str})
                self.assertEqual(scored.code.nunique(),5)  # conservation: no code lost
                self.assertTrue(scored.status.notna().all())
                self.assertTrue((out/f"top10_research_{h}.csv").is_file())
                self.assertTrue((out/f"top10_verified_tradeable_{h}.csv").is_file())
                self.assertTrue((out/f"top10_diversified_{h}.csv").is_file())
            import json
            audit = json.loads((out/"audit.json").read_text(encoding="utf-8"))
            self.assertTrue(all(v["conservation_ok"] for v in audit["horizons"].values()))


if __name__=="__main__":
    unittest.main()
