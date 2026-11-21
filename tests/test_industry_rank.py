import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fundkit.defensive_screen import Shock, pareto as historical_pareto, share_identity
from fundkit.industry_rank import (HORIZONS, NavPath, add_main_ranking, add_global_ranking, assemble, compare_forward,
                                   effective_group, feature, forward, legacy_events, main_stress_features,
                                   pareto, ranking, stress_metrics)
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
    def test_historical_pareto_matches_strict_dominance_with_ties(self):
        rng = np.random.default_rng(12)
        x = rng.integers(0, 5, size=(120, 3))
        expected = [not ((x >= a).all(axis=1) & (x > a).any(axis=1)).any() for a in x]
        np.testing.assert_array_equal(historical_pareto(x), expected)
        self.assertTrue(historical_pareto(np.ones((5, 3))).all())

    def test_main_counts_only_applicable_completed_configured_events(self):
        end = pd.Timestamp("2026-11-16")
        events = [Shock(key,key,pd.Timestamp(start),pd.Timestamp(finish),families=family)
                  for key,start,finish,family in (
                      ("first","2026-03-20","2026-03-23",()),
                      ("second","2026-11-09","2026-11-12",()),
                      ("overseas","2026-10-29","2026-11-04",("overseas",)),
                      ("future","2026-11-16","2026-11-17",()))]
        f = pd.DataFrame(dict(code=["000001","000002","000003","000004","000005"],
                              group=["domestic"]*4+["overseas"],
                              scenario_family=["domestic"]*4+["overseas"],
                              return_for_rank=[.10,.20,.12,.99,.30],
                              max_drawdown=[.05,.30,.10,.01,.10],
                              end=end, research_rank=[4,2,3,1,1]))
        raw = pd.DataFrame([(c,e.id,e.baseline,e.end, -loss, loss)
                            for c,loss in zip(f.code,[.01,.03,.02,.001,.05]) for e in events
                            if not (c == "000004" and e.id == "first")],
                           columns=["code","event","baseline","stress_end","stress_return","stress_drawdown"])
        metrics = main_stress_features(raw,events,end)
        self.assertNotIn("future",metrics.index.get_level_values("event"))
        z = add_main_ranking(f,metrics,events,end,2)
        self.assertEqual(z.loc[z.main_rank.eq(1),"code"].tolist(),["000001"])
        self.assertEqual(z.main_rank.iloc[:3].tolist(),[1,3,2])
        self.assertEqual(z.research_rank.tolist(),f.research_rank.tolist())
        self.assertEqual(z.main_stress_required.tolist(),[2,2,2,2,3])
        self.assertEqual(z.main_missing_stress_ids.iloc[3],"first")
        self.assertEqual(z.main_status.iloc[3],"主榜冲击净值覆盖不足")
        self.assertEqual(z.main_status.iloc[4],"主榜同类样本不足")

    def test_global_ranking_uses_one_cross_asset_pool_and_common_shocks(self):
        end = pd.Timestamp("2026-11-20")
        events = [
            Shock("market_a","market_a",pd.Timestamp("2026-03-20"),pd.Timestamp("2026-03-23")),
            Shock("market_b","market_b",pd.Timestamp("2026-06-05"),pd.Timestamp("2026-06-11")),
            Shock("commodity_only","commodity_only",pd.Timestamp("2026-01-29"),
                  pd.Timestamp("2026-02-02"),families=("commodity",)),
        ]
        codes = ["000001","000002","000003","000004","000005","000006"]
        frame = pd.DataFrame({
            "code":codes,
            "name":[f"fund-{c}" for c in codes],
            "group":["债券"]*3+["股票"]*3,
            "fund_type":["债券型-长债"]*3+["股票型"]*3,
            "scenario_family":["domestic"]*3+["commodity"]*3,
            "product_key":[f"id:{c}" for c in codes],
            "product":[f"fund-{c}" for c in codes],
            "share":["A"]*6,
            "currency":["CNY"]*6,
            "pricing_basis":["fund_nav"]*6,
            "obs":[250]*6,
            "return_for_rank":[.01,.02,.03,.04,.05,.06],
            "max_drawdown":[.06,.05,.04,.03,.02,.01],
        })
        records = []
        for i,code in enumerate(codes):
            for j,event in enumerate(events[:2]):
                records.append((code,event.id,event.baseline,event.end,
                                .001*(i+j+1),.001*(i+j+1)))
            if i >= 3:
                event = events[2]
                records.append((code,event.id,event.baseline,event.end,.01,.01))
        raw = pd.DataFrame(records,columns=[
            "code","event","baseline","stress_end","stress_return","stress_drawdown"])
        stress = main_stress_features(raw,events,end)
        ranked = add_global_ranking(frame,stress,events,end,min_peers=5)
        self.assertEqual(ranked.global_rank.notna().sum(),6)
        self.assertEqual(ranked.global_peer_count.unique().tolist(),[6])
        self.assertEqual(sorted(ranked.global_rank.astype(int).tolist()),list(range(1,7)))
        self.assertTrue(ranked.global_status.eq("全市场已排名").all())
        self.assertEqual(ranked.global_stress_required.unique().tolist(),[2])
        self.assertTrue(ranked.global_stress_ids.eq("market_a,market_b").all())
        self.assertTrue(ranked.global_pareto.notna().all())

    def test_fof_parenthesis_share_identity(self):
        pairs = [
            ("025029", "025030", "宝盈淳悦稳健配置3个月持有混合发起(FOF)"),
            ("024495", "024496", "平安盈轩90天持有债券(ETF-FOF)"),
        ]
        rows = [(code, name+share, "FOF-稳健型") for a,c,name in pairs
                for code,share in ((a,"A"),(c,"C"))]
        self.assertEqual(share_identity("示例（FOF） C类 "),("示例（FOF）","C"))
        self.assertEqual(share_identity("示例ETF"),("示例ETF",""))
        self.assertEqual(share_identity("示例(FOF)"),("示例(FOF)",""))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"fund_list.csv"
            pd.DataFrame(rows,columns=["code","name","type"]).to_csv(p,index=False)
            catalog = load_catalog(p)
            ds = pd.bdate_range("2019-01-01","2026-11-16")
            paths = {code:path(ds,np.linspace(1,1.3,len(ds))) for code,_,_ in rows}
            for horizon in HORIZONS:
                frame,rejected = assemble(catalog,paths,ds[-1],horizon,[])
                self.assertEqual(set(frame.code),{"025029","024495"})
                self.assertEqual({(r[0],r[2]) for r in rejected},
                                 {("025030","同基金其他份额"),("024496","同基金其他份额")})

    def test_catalog_product_ids_take_precedence_without_merging_conflicts(self):
        rows = [
            ("100001","甲债券A","债券型-长债","001"),
            ("100002","改名债券C","债券型-长债","001"),
            ("100003","甲债券E","债券型-长债",""),
            ("100004","乙债券A","债券型-长债","002"),
            ("100005","乙债券C","债券型-长债","003"),
            ("100006","乙债券E","债券型-长债",""),
            ("100007","甲债券美元A","债券型-长债","001"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"fund_list.csv"
            pd.DataFrame(rows,columns=["code","name","type","product_id"]).to_csv(p,index=False)
            catalog = load_catalog(p)
            self.assertEqual(catalog.product_key.tolist(),
                             ["id:001","id:001","id:001","id:002","id:003","name:乙债券","id:001"])
            self.assertEqual(load_catalog(p,["100003"]).product_key.tolist(),["id:001"])
            ds = pd.bdate_range("2025-01-01","2026-11-16")
            paths = {code:path(ds,np.linspace(1,1.1,len(ds))) for code, *_ in rows}
            frame,_ = assemble(catalog,paths,ds[-1],"6m",[])
            self.assertEqual(set(frame.code),{"100001","100004","100005","100006","100007"})

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
            for horizon in HORIZONS:
                top = pd.read_csv(out/f"top10_legacy_{horizon}.csv",dtype={"code":str})
                self.assertEqual(len(top),3)
                self.assertTrue(top.legacy_peer_count.eq(3).all())
                self.assertTrue(top.shock_1_id.eq("credit_20261030").all())
                self.assertTrue(top.shock_2_id.eq("bonds_202611").all())
                np.testing.assert_allclose(top[["shock_1_return","shock_2_return"]].min(axis=1),
                                           top.worst_stress_return)
                np.testing.assert_allclose(top[["shock_1_mdd","shock_2_mdd"]].max(axis=1),
                                           top.worst_stress_drawdown)
            replay = d/"replay"
            main(["--asof","2026-11-12","--replay-snapshot",str(out),"--output",str(replay),
                  "--min-peers","2"])
            for h in HORIZONS:
                for label in ("ranking", "ranking_global", "top10", "top10_global", "unranked_global",
                              "top10_stress", "top10_research", "top10_legacy"):
                    left = pd.read_csv(out/f"{label}_{h}.csv",dtype={"code":str})
                    right = pd.read_csv(replay/f"{label}_{h}.csv",dtype={"code":str})
                    pd.testing.assert_frame_equal(left,right)
                global_top = pd.read_csv(out/f"top10_global_{h}.csv",dtype={"code":str})
                self.assertLessEqual(len(global_top),10)
                self.assertTrue(global_top.global_rank.notna().all())
                self.assertEqual(global_top.global_rank.tolist(),
                                 sorted(global_top.global_rank.tolist()))
                t = pd.read_csv(out/f"top10_{h}.csv",dtype={"code":str})
                self.assertTrue(t.main_rank.notna().all())
            with self.assertRaisesRegex(ValueError,"cutoff"):
                main(["--asof","2026-11-13","--replay-snapshot",str(out),"--output",str(replay)])
            f = pd.read_csv(out/"oos_folds.csv")
            self.assertGreater(len(f),0)
            self.assertTrue(f.paired_complete.all())
            self.assertTrue((f.signal < f.evaluation_end).all())
            self.assertFalse((f.method.eq("legacy") & f.horizon.eq("6m")).any())
            self.assertTrue((f.method.eq("legacy") & f.horizon.eq("3y")).any())

    def test_long_periodic_excluded_by_default_but_audited(self):
        rows = catalog_rows()+[("000005","戊36个月定开债券A","债券型-长债"),
                               ("000006","己3个月定开债券A","债券型-长债")]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame(rows,columns=["code","name","type"]).to_csv(d/"fund_list.csv",index=False)
            (d/"total_return").mkdir()
            for i,(code,_,_) in enumerate(rows):
                ds = pd.bdate_range("2019-01-01","2026-11-12")
                v = np.exp((.001+i*.0005)*np.arange(len(ds))/10)
                pd.DataFrame({"date":ds,"nav":v}).to_csv(d/"total_return"/f"{code}.csv",index=False)
            out = d/"out"
            main(["--asof","2026-11-12","--data-dir",str(d),"--output",str(out),
                  "--min-peers","2","--oos-folds","1"])
            a = pd.read_csv(out/"ranking_6m.csv",dtype={"code":str})
            self.assertNotIn("000005",a.code.tolist())
            self.assertIn("000006",a.code.tolist())
            u = pd.read_csv(out/"unranked_6m.csv",dtype={"code":str})
            hit = u[u.code.eq("000005")]
            self.assertEqual(len(hit),1)
            self.assertIn("长封闭定开",hit.reason.iloc[0])
            lp = pd.read_csv(out/"long_periodic_audit.csv",dtype={"code":str})
            self.assertIn("000005",lp.code.tolist())
            t = pd.read_csv(out/"top10_6m.csv",dtype={"code":str})
            self.assertGreater(len(t),0)
            audit = json.loads((out/"audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["long_periodic_excluded"],1)
            self.assertIn("nav_recency",audit)
            # Opt-in keeps the previous behaviour.
            out2 = d/"out2"
            main(["--asof","2026-11-12","--data-dir",str(d),"--output",str(out2),
                  "--min-peers","2","--oos-folds","1","--include-long-periodic"])
            b = pd.read_csv(out2/"ranking_6m.csv",dtype={"code":str})
            self.assertIn("000005",b.code.tolist())

    def test_recency_audit_flags_missing_recent_days(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame(catalog_rows(),columns=["code","name","type"]).to_csv(d/"fund_list.csv",index=False)
            (d/"total_return").mkdir()
            for i,(code,_,_) in enumerate(catalog_rows()):
                end = "2026-11-11" if code=="000001" else "2026-11-12"
                ds = pd.bdate_range("2025-12-26",end)
                v = np.exp((.001+i*.001)*np.arange(len(ds))/10)
                pd.DataFrame({"date":ds,"nav":v}).to_csv(d/"total_return"/f"{code}.csv",index=False)
            out = d/"out"
            main(["--asof","2026-11-12","--data-dir",str(d),"--output",str(out),
                  "--min-peers","2","--oos-folds","1"])
            r = pd.read_csv(out/"nav_recency_audit.csv",dtype={"code":str})
            lag = r[r.code.eq("000001")].iloc[0]
            self.assertEqual(lag.nav_end,"2026-11-11")
            self.assertFalse(lag.exact_end)
            self.assertIn("2026-11-12",str(lag.missing_recent_days))
            ok = r[r.code.eq("000002")].iloc[0]
            self.assertTrue(ok.exact_end)
            self.assertTrue(pd.isna(ok.missing_recent_days) or ok.missing_recent_days=="")


if __name__=="__main__":
    unittest.main()
