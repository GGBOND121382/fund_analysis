import unittest

from scripts.check_ths_tradeable import ths_tradeable


def row(buy=None, sgstat=None, error=None):
    return {"code": "000001", "ths_name": "x", "sgstat": sgstat, "shstat": "开放",
            "buy": buy, "zdsg": "100", "error": error}


class TestThsTradeable(unittest.TestCase):
    def test_buy1_open_tradeable(self):
        ok, _ = ths_tradeable(row(buy="1", sgstat="开放"))
        self.assertTrue(ok)

    def test_buy1_limited_tradeable(self):
        ok, note = ths_tradeable(row(buy="1", sgstat="限制大额"))
        self.assertTrue(ok)
        self.assertIn("限制大额", note)

    def test_buy0_unsupported(self):
        ok, note = ths_tradeable(row(buy="0", sgstat="开放"))
        self.assertFalse(ok)
        self.assertEqual(note, "同花顺暂不支持交易")

    def test_buy2_suspended(self):
        ok, note = ths_tradeable(row(buy="2", sgstat="暂停"))
        self.assertFalse(ok)
        self.assertIn("暂停", note)

    def test_sgstat_paused_even_if_buy1(self):
        ok, _ = ths_tradeable(row(buy="1", sgstat="暂停"))
        self.assertFalse(ok)

    def test_error_passthrough(self):
        ok, note = ths_tradeable(row(error="timeout"))
        self.assertIsNone(ok)
        self.assertIn("查询失败", note)

    def test_unknown_state(self):
        ok, note = ths_tradeable(row(buy="9", sgstat="封闭期"))
        self.assertFalse(ok)
        self.assertIn("状态未知", note)


if __name__ == "__main__":
    unittest.main()
