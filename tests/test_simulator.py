"""模拟器确定性测试（合成净值，无网络）。

运行: python -m unittest discover -s tests -p 'test_*.py' -v
"""
import unittest

import pandas as pd

from fundkit.adapters import CST
from fundkit.simulator import Lot, SimConfig, Simulator
from tests import finance_math as fm

DAYS = pd.date_range("2026-01-05", periods=40, freq="B")  # 40 个工作日


def make_sim(nav_map, config: SimConfig | None = None, dividends=None):
    """注入合成净值。nav_map: {code: [每日净值...]}（长度<=40，缺日沿用前值）。"""
    sim = Simulator(list(nav_map), config or SimConfig())
    sim._actual_nav = {c: pd.Series(v, index=DAYS[: len(v)]) for c, v in nav_map.items()}
    sim._published_nav = {}
    for c, v in nav_map.items():
        idx = DAYS[: len(v)]
        sim._published_nav[c] = pd.DataFrame({
            "nav_date": idx,
            "published_at": idx.tz_localize(CST),  # 测试假设当日凌晨即公开
            "adj_nav_total_return": v,
        })
    sim.calendar = sorted({d for s in sim._actual_nav.values() for d in s.index})
    sim._dividends = dividends or {}
    return sim


def buy(policy_map):
    return lambda state: {"subscribe": dict(policy_map)}


class SimulatorTests(unittest.TestCase):
    def test_deposit_does_not_change_unit_nav(self):
        sim = make_sim({"004400": [1.0] * 40},
                       SimConfig(budget_per_day=1000))
        r = sim._execute(policy=buy({}))
        navs = r["daily"]["unit_nav"].to_numpy()
        self.assertTrue(abs(navs - 1.0).max() < 1e-12)
        self.assertAlmostEqual(r["metrics"]["total_return_twr"], 0.0)
        self.assertEqual(fm.max_drawdown(navs.tolist()), 0.0)

    def test_price_move_drives_twr_not_cashflow(self):
        nav = [1.0] * 20 + [2.0] * 20  # 中途净值翻倍
        # 无现金流：TWR 完全由价格变动决定
        r0 = make_sim({"004400": nav}, SimConfig())._execute(
            initial_lots={"004400": [Lot("004400", 100, DAYS[0])]}, policy=buy({}))
        self.assertAlmostEqual(r0["metrics"]["total_return_twr"], 1.0, places=9)
        # 大额每日入金：涨幅被现金摊薄但方向不变，且单位净值路径不受入金额扭曲
        r1 = make_sim({"004400": nav}, SimConfig(budget_per_day=100000))._execute(
            initial_lots={"004400": [Lot("004400", 100, DAYS[0])]}, policy=buy({}))
        twr = r1["metrics"]["total_return_twr"]
        self.assertGreater(twr, 0)
        self.assertLess(twr, 0.001)
        self.assertAlmostEqual(r1["metrics"]["max_drawdown"], 0.0)

    def test_xirr_single_deposit(self):
        days = pd.date_range("2025-01-06", periods=252, freq="B")

        def long_sim(budget):
            sim = make_sim({"004400": [1.0] * 40}, SimConfig(budget_per_day=budget))
            sim.calendar = list(days)
            sim._actual_nav = {"004400": pd.Series([1.0] * len(days), index=days)}
            return sim

        r = long_sim(0)._execute(initial_lots={"004400": [Lot("004400", 1000, days[0])]})
        self.assertAlmostEqual(r["metrics"]["xirr"], 0.0, places=6)  # 期初市值计为首笔流出，净值不变则 XIRR=0
        r2 = long_sim(100)._execute(policy=buy({"004400": 100}))
        self.assertAlmostEqual(r2["metrics"]["xirr"], 0.0, places=6)

    def test_buy_limit_clipped(self):
        sim = make_sim({"004400": [1.0] * 40},
                       SimConfig(budget_per_day=1000, daily_limits={"004400": 500}))
        r = sim._execute(policy=buy({"004400": 800}))
        ev = r["events"]
        self.assertIn("BUY_CLIPPED_LIMIT", set(ev["event"]))
        buys = ev[ev["event"] == "BUY"]
        self.assertTrue((buys["amount"] <= 500).all())

    def test_forbidden_equity_buy_rejected(self):
        sim = make_sim({"110020": [1.0] * 40, "004400": [1.0] * 40},
                       SimConfig(budget_per_day=1000, allow_new_equity_index=False,
                                 equity_index_codes=frozenset({"110020"})))
        r = sim._execute(policy=buy({"110020": 100, "004400": 100}))
        ev = set(r["events"]["event"])
        self.assertIn("BUY_REJECTED_FORBIDDEN", ev)
        rej = r["events"][r["events"]["event"] == "BUY_REJECTED_FORBIDDEN"]
        self.assertTrue((rej["code"] == "110020").all())

    def test_rolling_lock_enforced(self):
        sim = make_sim({"021559": [1.0] * 40}, SimConfig())
        lot = Lot("021559", 100, DAYS[0])  # 锁 30 天
        # 第 10 个交易日（约14自然日）赎回被拒
        early = sim._execute(initial_lots={"021559": [Lot("021559", 100, DAYS[0])]},
                             policy=lambda s: {"redeem": {"021559": 100}})
        self.assertIn("REDEEM_REJECTED_LOCKED", set(early["events"]["event"]))
        # 第 31 自然日后首个周期窗口内赎回成功
        late_day = DAYS[0] + pd.Timedelta(days=31)
        redeemable = sim._redeemable("021559", lot, late_day)
        self.assertTrue(redeemable)
        self.assertFalse(sim._redeemable("021559", lot, DAYS[0] + pd.Timedelta(days=10)))

    def test_dividend_keeps_unit_nav_continuous(self):
        nav = [1.0] * 20 + [0.9] * 20   # 除息日净值从 1.0 跌到 0.9
        divs = {"004400": [(str(DAYS[20].date()), 0.1)]}  # 每份派 0.1
        sim = make_sim({"004400": nav}, SimConfig(), dividends=divs)
        r = sim._execute(initial_lots={"004400": [Lot("004400", 100, DAYS[0])]})
        navs = r["daily"]["unit_nav"].to_numpy()
        self.assertTrue(abs(navs - 1.0).max() < 1e-9)  # 分红到账后价值连续
        self.assertIn("DIVIDEND", set(r["events"]["event"]))

    def test_redeem_cash_lag(self):
        sim = make_sim({"004400": [2.0] * 40}, SimConfig())
        policy_calls = {"done": False}

        def policy(state):
            if not policy_calls["done"]:
                policy_calls["done"] = True
                return {"redeem": {"004400": 50}}
            return {}

        r = sim._execute(initial_lots={"004400": [Lot("004400", 100, DAYS[0])]},
                         policy=policy)
        d0 = r["daily"].iloc[0]
        self.assertAlmostEqual(d0["pending"], 100.0)  # 50份×2元 在途
        self.assertAlmostEqual(d0["value"], 200.0)    # 在途计入组合价值
        self.assertGreater(r["daily"]["cash"].iloc[-1], 99.9)  # 2日后到账

    def test_mdd_matches_reference(self):
        nav = [1.0, 1.2, 1.0, 1.1, 0.9] + [1.0] * 35
        sim = make_sim({"004400": nav}, SimConfig())
        r = sim._execute(initial_lots={"004400": [Lot("004400", 100, DAYS[0])]})
        navs = r["daily"]["unit_nav"].to_numpy()
        self.assertAlmostEqual(r["metrics"]["max_drawdown"], fm.max_drawdown(navs.tolist()))
        self.assertAlmostEqual(r["metrics"]["max_drawdown"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
