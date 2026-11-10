"""策略族 B2-B5 确定性测试（手工构造 state，无网络）。

运行: python -m unittest discover -s tests -p 'test_*.py' -v
"""
import unittest
from unittest import mock

import pandas as pd

from fundkit import policies

D0 = pd.Timestamp("2026-11-02")


def make_state(cash, holdings, navs, unit_hist=None, day=D0):
    return {
        "date": day, "cash": cash,
        "nav": {c: {"nav": n, "nav_date": str(day.date())} for c, n in navs.items()},
        "holdings": {c: {"shares": s, "value_pub": s * navs[c],
                         "redeemable_shares": s}
                     for c, s in holdings.items()},
        "value": cash + sum(s * navs[c] for c, s in holdings.items()),
        "unit_nav": 1.0,
        "unit_nav_history": unit_hist or [1.0],
    }


PARAMS = {
    "target_weights": {"710301": 0.45, "004400": 0.35, "021559": 0.05,
                       "000218": 0.05, "cash": 0.10},
    "daily_limits": {"004400": 500},  # 金信限大额 500 元/日
    "B3": {"check_frequency": "weekly", "weight_trigger": 0.02,
           "destination_fraction": 0.5, "allow_redeem": False},
    "B4": {"pause_cdar95": 0.02, "pause_shift_states": ["POSSIBLE_SHIFT", "CONFIRMED_SHIFT"],
           "reenter_cooldown_days": 3, "reenter_cdar95_low": 0.01},
    "B5": {"mdd_budget": 0.05, "internal_drawdown_margin": 0.02,
           "reenter_mdd_fraction": 0.6, "fund_pause_drawdown": 0.03,
           "fund_reenter_drawdown": 0.015, "reenter_cooldown_days": 3},
}


class PolicyTests(unittest.TestCase):
    def test_b2_water_filling_moves_toward_target(self):
        # 富安达超重（约 55.9% > 45% 目标），金信低配：现金应流向低配基金
        # 且金信受 500 元/日限额截断；现金下限 10% 必须保留
        state = make_state(
            cash=30000,
            holdings={"710301": 73315, "004400": 20000, "000218": 364},
            navs={"710301": 1.4875, "004400": 1.0806, "000218": 3.1049})
        with mock.patch.object(policies, "_plans", return_value=[
                {"code": "004400", "daily_plan_cny": 500},
                {"code": "021559", "daily_plan_cny": 200},
                {"code": "000218", "daily_plan_cny": 100}]):
            policy = policies.make_b2(PARAMS)
            orders = policy(state)
        buys = orders["subscribe"]
        self.assertNotIn("710301", buys)          # 超配基金不再买入
        self.assertLessEqual(buys.get("004400", 0), 500 + 1e-9)  # 受日限额
        investable = 30000 - 0.10 * state["value"]
        self.assertLessEqual(sum(buys.values()), investable + 1e-6)
        self.assertGreater(sum(buys.values()), 500)  # 限额之外的低配基金吸收剩余

    def test_b2_cash_floor_respected(self):
        state = make_state(cash=100, holdings={"004400": 90000},
                           navs={"004400": 1.0806})
        with mock.patch.object(policies, "_plans", return_value=[]):
            policy = policies.make_b2(PARAMS)
            orders = policy(state)
        # 现金下限 10%×(总资产)≈1 万，100 元现金远低于下限：不投放
        self.assertEqual(sum((orders.get("subscribe") or {}).values()), 0.0)

    def _b4_with_signals(self, cdar, shift="POSSIBLE_SHIFT"):
        fake = {"710301": {
            "combined_shift_evidence": {"state": shift},
            "risk_views": {"CURRENT_STYLE_REWEIGHTED": {"cdar95": cdar, "mdd": cdar}}}}
        with mock.patch.object(policies, "_plans", return_value=[]):
            return policies.make_b4({**PARAMS, "_style_signals": fake})

    def test_b4_pause_and_reenter_with_hysteresis(self):
        policy = self._b4_with_signals(cdar=0.03)  # 高于 pause 阈值 0.02
        state = make_state(cash=100000,
                           holdings={"710301": 73315, "004400": 90000},
                           navs={"710301": 1.4875, "004400": 1.0806})
        orders = policy(state)
        self.assertNotIn("710301", orders.get("subscribe", {}))  # 被 PAUSE
        self.assertTrue(any(e.get("action") == "PAUSE_NEW_BUY"
                            for e in orders.get("events", [])))
        # 风险信号持续高位：保持 PAUSE
        for i in range(1, 6):
            s2 = dict(state, date=D0 + pd.Timedelta(days=i))
            self.assertNotIn("710301", policy(s2).get("subscribe", {}))

    def test_b5_portfolio_mdd_guard(self):
        with mock.patch.object(policies, "_load_style_signals", return_value={}), \
             mock.patch.object(policies, "_plans", return_value=[]):
            policy = policies.make_b5(PARAMS)
        # 组合单位净值从历史高点回撤 4%（> 软预算 3%）
        hist = [1.0] * 20 + [0.96] * 5
        state = make_state(cash=100000, holdings={"004400": 90000},
                           navs={"004400": 1.0806}, unit_hist=hist)
        orders = policy(state)
        self.assertEqual(orders.get("subscribe"), {})  # 全部新增留现金
        self.assertTrue(any(e.get("action") == "DE_RISK_PAUSE_BUYS"
                            for e in orders.get("events", [])))
        # 回撤未恢复：保持防守
        s2 = dict(state, date=D0 + pd.Timedelta(days=1))
        self.assertEqual(policy(s2).get("subscribe"), {})

    def test_b5_fund_level_pause_and_reenter(self):
        with mock.patch.object(policies, "_load_style_signals", return_value={}), \
             mock.patch.object(policies, "_plans", return_value=[
                 {"code": "004400", "daily_plan_cny": 500}]):
            policy = policies.make_b5(PARAMS)
        # 004400 净值从峰值回撤 4%（> 3% 暂停线）
        state = make_state(cash=100000, holdings={"004400": 20000},
                           navs={"004400": 0.96}, unit_hist=[1.0, 1.001])
        policy(make_state(cash=100000, holdings={"004400": 20000},
                          navs={"004400": 1.0}, unit_hist=[1.0, 1.001]))  # 先建立峰值
        orders = policy(state)
        self.assertNotIn("004400", orders.get("subscribe", {}))
        # 净值恢复到距峰值 <1.5% 并持续冷却期：再入场
        for i in range(1, 5):
            s = make_state(cash=100000, holdings={"004400": 20000},
                           navs={"004400": 0.995}, unit_hist=[1.0, 1.001],
                           day=D0 + pd.Timedelta(days=i))
            orders = policy(s)
        self.assertIn("004400", orders.get("subscribe", {}))


class PolicyFundSpecificRiskTests(unittest.TestCase):
    def test_b5_uses_fund_specific_thresholds(self):
        from fundkit.policies import make_b5
        params = {
            "target_weights": {"BOND": .4, "GOLD": .4, "cash": .2},
            "equity_index_codes": [], "daily_limits": {},
            "B3": {"check_frequency": "weekly", "weight_trigger": .02,
                   "destination_fraction": .5, "allow_redeem": False},
            "B4": {"pause_cdar95": .02, "pause_shift_states": ["POSSIBLE_SHIFT", "CONFIRMED_SHIFT"],
                   "reenter_cooldown_days": 10, "reenter_cdar95_low": .01},
            "B5": {"mdd_budget": .04, "internal_drawdown_margin": .01,
                   "reenter_mdd_fraction": .6, "fund_pause_drawdown": .03,
                   "fund_reenter_drawdown": .015,
                   "fund_pause_drawdown_by_code": {"BOND": .01, "GOLD": .04},
                   "fund_reenter_drawdown_by_code": {"BOND": .005, "GOLD": .02},
                   "fund_redeem_fraction_by_code": {"BOND": 0.0, "GOLD": 0.0},
                   "reenter_cooldown_days": 10, "allow_redeem": False,
                   "portfolio_redeem_fraction": 0.0, "fund_redeem_fraction": 0.0,
                   "cdar_lookback": 126, "cdar_guard_threshold": 1.0},
        }
        policy = make_b5(params)
        def state(day, bond_tr, gold_tr):
            return {"date": pd.Timestamp(day), "cash": 1000.0, "pending_value": 0.0,
                    "value_quality": "COMPLETE", "holdings": {},
                    "nav": {"BOND": {"nav": .9, "total_return_nav": bond_tr},
                            "GOLD": {"nav": .9, "total_return_nav": gold_tr}},
                    "unit_nav_history": [1.0] * 20}
        policy(state("2025-01-01", 1.0, 1.0))  # establish peaks
        out = policy(state("2025-01-02", .98, .98))
        events = out.get("events", [])
        paused = {e["code"] for e in events if e["action"] == "PAUSE_NEW_BUY"}
        self.assertIn("BOND", paused)
        self.assertNotIn("GOLD", paused)

    def test_b5_uses_total_return_nav_not_ex_dividend_raw_nav(self):
        from fundkit.policies import make_b5
        params = {
            "target_weights": {"F1": .5, "cash": .5},
            "equity_index_codes": [], "daily_limits": {},
            "B3": {"check_frequency": "weekly", "weight_trigger": .02,
                   "destination_fraction": .5, "allow_redeem": False},
            "B4": {"pause_cdar95": .02, "pause_shift_states": ["POSSIBLE_SHIFT", "CONFIRMED_SHIFT"],
                   "reenter_cooldown_days": 10, "reenter_cdar95_low": .01},
            "B5": {"mdd_budget": .04, "internal_drawdown_margin": .01,
                   "reenter_mdd_fraction": .6, "fund_pause_drawdown": .03,
                   "fund_reenter_drawdown": .015,
                   "reenter_cooldown_days": 10, "allow_redeem": False,
                   "portfolio_redeem_fraction": 0.0, "fund_redeem_fraction": 0.0,
                   "cdar_lookback": 126, "cdar_guard_threshold": 1.0},
        }
        policy = make_b5(params)
        base = {"cash": 1000.0, "pending_value": 0.0, "value_quality": "COMPLETE",
                "holdings": {}, "unit_nav_history": [1.0] * 20}
        s1 = {**base, "date": pd.Timestamp("2025-01-01"),
              "nav": {"F1": {"nav": 1.0, "total_return_nav": 1.0}}}
        s2 = {**base, "date": pd.Timestamp("2025-01-02"),
              "nav": {"F1": {"nav": .9, "total_return_nav": 1.0}}}
        policy(s1)
        out = policy(s2)
        self.assertFalse(any(e.get("action") == "PAUSE_NEW_BUY" for e in out.get("events", [])))


    def test_b5_executes_real_stop_loss_redemption(self):
        params = {
            **PARAMS,
            "asset_categories": {"004400": "bond"},
            "B5": {**PARAMS["B5"], "mdd_budget": .04,
                   "internal_drawdown_margin": .01,
                   "fund_pause_drawdown": .03, "fund_reenter_drawdown": .01,
                   "fund_redeem_fraction": .5, "allow_redeem": True,
                   "fund_stop_vol_multiplier_by_category": {"bond": 0.0},
                   "fund_stop_drawdown_cap": .12},
        }
        with mock.patch.object(policies, "_load_style_signals", return_value={}), \
             mock.patch.object(policies, "_plans", return_value=[]):
            policy = policies.make_b5(params)
        s0 = make_state(1000, {"004400": 100}, {"004400": 1.0},
                        unit_hist=[1.0] * 25)
        s0["nav"]["004400"]["total_return_nav"] = 1.0
        policy(s0)
        state = make_state(1000, {"004400": 100}, {"004400": .96},
                           unit_hist=[1.0] * 25)
        state["nav"]["004400"]["total_return_nav"] = .96
        out = policy(state)
        self.assertAlmostEqual(out["redeem"]["004400"], 50.0)
        self.assertTrue(any(e.get("action") == "FUND_STOP_LOSS_REDEEM"
                            for e in out["events"]))

    def test_b5_take_profit_requires_known_cost_basis_and_redeems_partial(self):
        params = {
            **PARAMS,
            "asset_categories": {"004400": "bond"},
            "B5": {**PARAMS["B5"], "mdd_budget": .04,
                   "internal_drawdown_margin": .01,
                   "fund_pause_drawdown": .20, "fund_reenter_drawdown": .10,
                   "fund_stop_vol_multiplier_by_category": {"bond": 0.0},
                   "fund_stop_drawdown_cap": .25,
                   "allow_redeem": True,
                   "take_profit_activation_by_code": {"004400": .04},
                   "take_profit_trailing_drawdown_by_code": {"004400": .03},
                   "take_profit_redeem_fraction_by_code": {"004400": .25}},
        }
        with mock.patch.object(policies, "_load_style_signals", return_value={}), \
             mock.patch.object(policies, "_plans", return_value=[]):
            policy = policies.make_b5(params)
        state = make_state(1000, {"004400": 100}, {"004400": 1.05},
                           unit_hist=[1.0] * 25)
        state["nav"]["004400"]["total_return_nav"] = 1.05
        state["holdings"]["004400"].update({
            "redeemable_shares": 80,
            "redeemable_position_return": .06,
            "redeemable_cost_basis_known": True,
        })
        policy(state)
        state2 = make_state(1000, {"004400": 100}, {"004400": 1.008},
                            unit_hist=[1.0] * 25, day=D0 + pd.Timedelta(days=1))
        state2["nav"]["004400"]["total_return_nav"] = 1.008
        state2["holdings"]["004400"].update({
            "redeemable_shares": 80,
            "redeemable_position_return": .06,
            "redeemable_cost_basis_known": True,
        })
        out = policy(state2)
        self.assertAlmostEqual(out["redeem"]["004400"], 20.0)
        self.assertTrue(any(e.get("action") == "TAKE_PROFIT_REDEEM"
                            for e in out["events"]))
        state3 = make_state(1000, {"004400": 100}, {"004400": 1.008},
                            unit_hist=[1.0] * 25, day=D0 + pd.Timedelta(days=15))
        state3["nav"]["004400"]["total_return_nav"] = 1.008
        state3["holdings"]["004400"].update({
            "redeemable_shares": 60,
            "redeemable_position_return": .06,
            "redeemable_cost_basis_known": True,
        })
        out3 = policy(state3)
        self.assertNotIn("004400", out3.get("redeem", {}))

    def test_b5_retries_stop_when_locked_lot_becomes_redeemable(self):
        params = {
            **PARAMS,
            "asset_categories": {"004400": "bond"},
            "B5": {**PARAMS["B5"], "mdd_budget": .04,
                   "internal_drawdown_margin": .01,
                   "fund_pause_drawdown": .03, "fund_reenter_drawdown": .01,
                   "fund_redeem_fraction": .5, "allow_redeem": True,
                   "fund_stop_vol_multiplier_by_category": {"bond": 0.0},
                   "fund_stop_drawdown_cap": .12},
        }
        with mock.patch.object(policies, "_load_style_signals", return_value={}), \
             mock.patch.object(policies, "_plans", return_value=[]):
            policy = policies.make_b5(params)
        s0 = make_state(1000, {"004400": 100}, {"004400": 1.0},
                        unit_hist=[1.0] * 25)
        s0["nav"]["004400"]["total_return_nav"] = 1.0
        s0["holdings"]["004400"]["redeemable_shares"] = 0
        policy(s0)
        s1 = make_state(1000, {"004400": 100}, {"004400": .96},
                        unit_hist=[1.0] * 25, day=D0 + pd.Timedelta(days=1))
        s1["nav"]["004400"]["total_return_nav"] = .96
        s1["holdings"]["004400"]["redeemable_shares"] = 0
        policy(s1)
        s2 = make_state(1000, {"004400": 100}, {"004400": .96},
                        unit_hist=[1.0] * 25, day=D0 + pd.Timedelta(days=2))
        s2["nav"]["004400"]["total_return_nav"] = .96
        s2["holdings"]["004400"]["redeemable_shares"] = 100
        out = policy(s2)
        self.assertAlmostEqual(out["redeem"]["004400"], 50.0)

    def test_b5_reconstructs_fund_peak_from_published_history(self):
        params = {
            **PARAMS,
            "asset_categories": {"004400": "bond"},
            "B5": {**PARAMS["B5"], "mdd_budget": .04,
                   "internal_drawdown_margin": .01,
                   "fund_pause_drawdown": .03, "fund_reenter_drawdown": .01,
                   "fund_redeem_fraction": .5, "allow_redeem": True,
                   "fund_stop_vol_multiplier_by_category": {"bond": 0.0},
                   "fund_stop_drawdown_cap": .12,
                   "fund_peak_lookback_days": 252},
        }
        with mock.patch.object(policies, "_load_style_signals", return_value={}), \
             mock.patch.object(policies, "_plans", return_value=[]):
            policy = policies.make_b5(params)
        state = make_state(1000, {"004400": 100}, {"004400": .96},
                           unit_hist=[1.0] * 25)
        state["nav"]["004400"]["total_return_nav"] = .96
        state["fund_nav_history"] = {"004400": [
            {"date": "2026-01-01", "total_return_nav": 1.0},
            {"date": "2026-02-01", "total_return_nav": 1.05},
            {"date": "2026-03-01", "total_return_nav": .96},
        ]}
        out = policy(state)
        self.assertAlmostEqual(out["redeem"]["004400"], 50.0)

    def test_b2_can_allocate_to_index_fund_asset_class(self):
        params = {
            **PARAMS,
            "target_weights": {"BOND": .5, "INDEX": .3, "cash": .2},
            "equity_index_codes": ["INDEX"],
            "daily_limits": {},
        }
        state = make_state(cash=1000, holdings={"BOND": 1000},
                           navs={"BOND": 1.0})
        with mock.patch.object(policies, "_plans", return_value=[]):
            out = policies.make_b2(params)(state)
        self.assertIn("INDEX", out["subscribe"])
        self.assertGreater(out["subscribe"]["INDEX"], 0)



if __name__ == "__main__":
    unittest.main()