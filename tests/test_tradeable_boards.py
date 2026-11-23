import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_tradeable_boards import classify, load_channel_status

CH_CSV = """code,name,fund_type,group,has_lock,sub_status,red_status,daily_limit,channel_purchase_status,channel_status_source,channel_status_checked_at,selection_status
000001,开放基,债券型,债券型,False,开放申购,开放赎回,100000000000.0,该渠道开放申购,src,2026-11-17T15:25:49+0800,未入围
000002,限额基,债券型,债券型,False,限大额,开放赎回,1000.0,该渠道限额申购,src,2026-11-17T15:25:49+0800,未入围
000003,低限额基,债券型,债券型,False,限大额,开放赎回,100.0,该渠道限额申购,src,2026-11-17T15:25:49+0800,未入围
000004,暂停基,债券型,债券型,True,暂停申购,暂停赎回,0.0,该渠道暂停申购,src,2026-11-17T15:25:49+0800,未入围
000005,场内基,债券型,债券型,False,场内交易,场内交易,0.0,该渠道状态待核实：场内交易,src,2026-11-17T15:25:49+0800,未入围
000006,零限额基,债券型,债券型,False,限大额,开放赎回,0.0,该渠道限额申购,src,2026-11-17T15:25:49+0800,未入围
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


if __name__ == "__main__":
    unittest.main()
