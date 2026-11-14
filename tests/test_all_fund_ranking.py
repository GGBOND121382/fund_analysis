import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundkit.defensive_screen import Shock, market_category, money_index
from scripts.fetch_all_fund_rank_data import parse_catalog, parse_money, parse_nav, listed_history
from scripts.screen_defensive_funds import screen, write_reports
from unittest.mock import patch


def epoch(d):
    return int(pd.Timestamp(d, tz="Asia/Shanghai").timestamp() * 1000)


class AllFundRankingTest(unittest.TestCase):
    def test_catalog_and_type_dispatch(self):
        d = parse_catalog('var r = [["000009","XX","货币A","货币型-普通货币","XX"]];')
        self.assertEqual(d.code.tolist(), ["000009"])
        self.assertEqual(market_category("示例货币", d.type.iloc[0])[-1], "income_per_10000")
        self.assertEqual(market_category("示例货币ETF", d.type.iloc[0])[-1], "income_per_10000")
        self.assertEqual(market_category("示例REIT", "Reits")[-1], "adjusted_close")
        self.assertEqual(market_category("美股美元现汇", "QDII-普通股票")[2], "USD")
        self.assertEqual(market_category("原油", "QDII-商品")[1], "commodity")
        self.assertEqual(market_category("未知", None)[0], "类型待核验")

    def test_income_not_seven_day_yield_and_gap_no_fill(self):
        data = [[epoch("2026-11-06"), 1.0], [epoch("2026-11-09"), 2.0],
                [epoch("2026-11-10"), 3.0], [epoch("2026-11-11"), -1.0]]
        js = "var Data_millionCopiesIncome=" + json.dumps(data) + ";var Data_sevenDaysYearIncome=[];"
        nav = money_index(parse_money(js), pd.Timestamp("2026-11-11"))
        self.assertEqual(len(nav), 3)
        self.assertAlmostEqual(nav.iloc[-1] / nav.iloc[0], (1 + 3/10000) * (1 - 1/10000))

    def test_event_adjustment_and_unreconciled_jump(self):
        def make(event):
            dates = [epoch(d) for d in ("2026-11-10", "2026-11-11", "2026-11-12")]
            nav = [dict(x=t, y=y, equityReturn=r, unitMoney=m) for t,y,r,m in
                   zip(dates, [1.00, .96, .97], [0., 1., 1.04], ["", event, ""])]
            acc = list(zip(dates, [1., 1.01, 1.02]))
            return f"var Data_netWorthTrend={json.dumps(nav, ensure_ascii=False)};var Data_ACWorthTrend={json.dumps(acc)};"
        out = parse_nav(make("分红：每10份派现金元"), pd.Timestamp("2026-11-12"))
        self.assertAlmostEqual(out.nav.iloc[-1], 1.01*.97/.96)
        with self.assertRaisesRegex(ValueError, "unreconciled"):
            parse_nav(make(""), pd.Timestamp("2026-11-12"))

    def test_reits_never_uses_unadjusted_close(self):
        data = '{"data":{"sh508000":{"day":[["2026-11-12","2","2","2","2"]]}}}'
        with patch("scripts.fetch_all_fund_rank_data.read_url", return_value=data):
            with self.assertRaisesRegex(ValueError, "no_valid_hfq"):
                listed_history("508000", pd.Timestamp("2026-11-12"))

    def test_all_scope_ranking_not_truncated_to_shortlist(self):
        end = pd.Timestamp("2026-11-12")
        events = [Shock("a", "a", pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-11")),
                  Shock("b", "b", pd.Timestamp("2026-10-29"), end),
                  Shock("c", "c", pd.Timestamp("2026-11-04"), pd.Timestamp("2026-11-06"),
                        families=("overseas",))]
        with tempfile.TemporaryDirectory() as t:
            data = Path(t)
            (data/"nav").mkdir()
            (data/"money").mkdir()
            codes = [("000001","甲股票A","股票型"),("000002","乙股票A","股票型"),
                     ("000003","海外股票A","QDII-普通股票"),
                     ("000004","货币A","货币型-普通货币"),("000005","未分类",None)]
            pd.DataFrame(codes,columns=["code","name","type"]).to_csv(data/"fund_list.csv",index=False)
            ds = pd.bdate_range("2024-01-01",end)
            for i,(code,*_) in enumerate(codes[:3]):
                pd.DataFrame({"date":ds,"nav":np.linspace(1.,1.1+i*.05,len(ds))}).to_csv(data/"nav"/f"{code}.csv",index=False)
            pd.DataFrame({"date":pd.date_range("2024-01-01",end),"income_per_10000":.34}).to_csv(data/"money"/"000004.csv",index=False)
            full,short = screen(data,end,events,top=1,min_peers=1,universe="all")
            out = data/"out"
            audit = write_reports(out,full,short,events,end,["000001","000004"])
            self.assertEqual(audit["universe_share_codes"],5)
            self.assertEqual(len(pd.read_csv(out/"ranking.csv")),4)
            self.assertEqual(len(pd.read_csv(out/"unranked.csv")),1)
            self.assertEqual(len(short),3)
            self.assertEqual(full.set_index("code").required_shocks.to_dict(),
                             {"000001":2,"000002":2,"000003":3,"000004":2,"000005":2})


if __name__ == "__main__":
    unittest.main()
