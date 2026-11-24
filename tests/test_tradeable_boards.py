import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_tradeable_boards import classify, load_channel_status, process_period

CH_CSV = """code,name,fund_type,group,has_lock,sub_status,red_status,daily_limit,channel_purchase_status,channel_status_source,channel_status_checked_at,selection_status
000001,开放基,债券型,债券型,False,开放申购,开放赎回,100000000000.0,该渠道开放申购,src,2026-11-17T15:25:49+0800,未入围
000002,限额基,债券型,债券型,False,限大额,开放赎回,1000.0,该渠道限额申购,src,2026-11-17T15:25:49+0800,未入围
000003,低限额基,债券型,债券型,False,限大额,开放赎回,100.0,该渠道限额申购,src,2026-11-17T15:25:49+0800,未入围
000004,暂停基,债券型,债券型,True,暂停申购,暂停赎回,0.0,该渠道暂停申购,src,2026-11-17T15:25:49+0800,未入围
000005,场内基,债券型,债券型,False,场内交易,场内交易,0.0,该渠道状态待核实：场内交易,src,2026-11-17T15:25:49+0800,未入围
000006,零限额基,债券型,债券型,False,限大额,开放赎回,0.0,该渠道限额申购,src,2026-11-17T15:25:49+0800,未入围
000010,产品甲A(暂停),债券型,债券型,False,暂停申购,开放赎回,0.0,该渠道暂停申购,src,2026-11-17T15:25:49+0800,未入围
000011,产品甲C(开放),债券型,债券型,False,开放申购,开放赎回,100000000000.0,该渠道开放申购,src,2026-11-17T15:25:49+0800,未入围
000012,产品乙A(暂停),债券型,债券型,False,暂停申购,开放赎回,0.0,该渠道暂停申购,src,2026-11-17T15:25:49+0800,未入围
000013,产品乙C(暂停),债券型,债券型,False,暂停申购,暂停赎回,0.0,该渠道暂停申购,src,2026-11-17T15:25:49+0800,未入围
"""


def _ch():
    tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
    tmp.write_text(CH_CSV)
    return load_channel_status(tmp)


def _row(code, status="SCORED"):
    return pd.Series({"code": code, "status": status})


class TestTradeableFilter(unittest.TestCase):
    def setUp(self):
        self.ch = _ch()

    def test_open_is_tradeable(self):
        ok, note = classify(_row("000001"), self.ch, 500.0)
        self.assertTrue(ok)

    def test_limit_above_threshold_tradeable(self):
        ok, _ = classify(_row("000002"), self.ch, 500.0)
        self.assertTrue(ok)

    def test_limit_below_threshold_excluded(self):
        for code in ("000003", "000006"):
            ok, note = classify(_row(code), self.ch, 500.0)
            self.assertFalse(ok)
            self.assertIn("限额过低", note)

    def test_suspended_excluded(self):
        ok, note = classify(_row("000004"), self.ch, 500.0)
        self.assertFalse(ok)
        self.assertEqual(note, "暂停申购")

    def test_exchange_traded_excluded_with_note(self):
        ok, note = classify(_row("000005"), self.ch, 500.0)
        self.assertFalse(ok)
        self.assertIn("证券账户", note)

    def test_unknown_code_excluded(self):
        ok, note = classify(_row("999999"), self.ch, 500.0)
        self.assertFalse(ok)
        self.assertIn("无渠道", note)

    def test_unscored_passthrough(self):
        ok, note = classify(_row("000001", status="期限起点缺历史"), self.ch, 500.0)
        self.assertIsNone(ok)


def _scored_row(code, product_key, score):
    return {"code": code, "name": code, "product_key": product_key,
            "status": "SCORED", "SCORE_M0": score}


class TestDedupOrderP0T1(unittest.TestCase):
    """P0-T1: filter by tradeability BEFORE product_key dedup."""

    def setUp(self):
        self.ch = _ch()

    def test_suspended_best_share_does_not_kill_product(self):
        # 产品甲：A份额(高分，暂停) + C份额(低分，开放) -> C 必须入榜且标记切换
        df = pd.DataFrame([_scored_row("000010", "prodA", 90.0),
                           _scored_row("000011", "prodA", 80.0),
                           _scored_row("000001", "prodB", 85.0)])
        trad, excl, stats = process_period(df, self.ch, 500.0, 10)
        self.assertEqual(len(trad), 2)
        row_a = trad[trad["product_key"] == "prodA"].iloc[0]
        self.assertEqual(row_a["code"], "000011")
        self.assertTrue(row_a["share_switched"])
        self.assertEqual(row_a["best_share_code"], "000010")
        self.assertEqual(len(excl), 0)
        self.assertEqual(stats["share_switched"], 1)

    def test_product_with_no_tradeable_share_excluded(self):
        # 产品乙：两个份额都暂停 -> 产品级排除
        df = pd.DataFrame([_scored_row("000012", "prodC", 95.0),
                           _scored_row("000013", "prodC", 94.0),
                           _scored_row("000001", "prodB", 85.0)])
        trad, excl, stats = process_period(df, self.ch, 500.0, 10)
        self.assertEqual(len(trad), 1)
        self.assertEqual(len(excl), 1)
        self.assertEqual(excl.iloc[0]["product_key"], "prodC")

    def test_dedup_keeps_highest_tradeable_share(self):
        # 同一产品两个可交易份额 -> 保留高分份额，不标记切换
        df = pd.DataFrame([_scored_row("000011", "prodA", 80.0),
                           _scored_row("000001", "prodA", 85.0)])
        trad, excl, stats = process_period(df, self.ch, 500.0, 10)
        self.assertEqual(len(trad), 1)
        self.assertEqual(trad.iloc[0]["code"], "000001")
        self.assertFalse(trad.iloc[0]["share_switched"])


if __name__ == "__main__":
    unittest.main()
