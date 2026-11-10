"""严格现金流/基金交易模拟器（第3步）。

处理：外部每日现金流、现金分红、费率、确认时点、赎回到账延迟、份额锁定
（30天滚动持有逐笔批次）、申购限额、节假日（以真实净值披露日为交易日历）、
现金收益。业绩口径为现金流调整后的 TWR 单位净值（入金按当时单位净值折算份额，
不改变单位净值本身），并输出 CAGR/Sharpe/Sortino/MDD/CDaR95/XIRR。

反未来信息：策略决策只能看到 published_at <= 决策时点的净值（经 adapters 适配层）；
成交按真实规则——T 日 15:00 前提交的申赎以 T 日真实净值成交（决策时该净值未知）。

未知项一律不伪造，走可配置假设并显式标注：
- 申赎费率默认 0（未知），可逐基金配置
- 确认/到账延迟：申购 T 日净值确认，赎回 T 日净值确认、T+2 自然日到账（可配）
- 现金收益率默认 0
- 021559 滚动持有：批次自确认日起锁 30 个自然日，其后每 30 天开放 5 天窗口（近似，可配）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fundkit import storage
from fundkit.adapters import CST, LocalDataAdapter
from tests import finance_math as fm

SIM_DEFAULTS = {
    "redeem_cash_lag_days": 2,      # 赎回确认后到账延迟（自然日；QDII 实际更长，可配）
    "dividend_cash_lag_days": 2,    # 现金分红到账延迟（假设）
    "subscribe_fee_rate": 0.0,      # 申购费率（未知，默认 0，逐基金可配）
    "redeem_fee_rate": 0.0,
    "cash_annual_rate": 0.0,        # 现金年化收益（小数）
    "rolling_lock_days": 30,        # 滚动持有锁定天数（021559）
    "rolling_open_window_days": 5,  # 每个滚动周期的开放窗口（近似假设）
    "rf": 0.02,                     # 无风险利率（年化小数，用于 Sharpe/Sortino）
    "published_lag_days": 2,        # 与 adapters 的披露延迟规则一致（决策可见性）
}

# 滚动持有期基金（逐笔批次锁定）。其余基金默认随时可赎回。
ROLLING_LOCK_FUNDS = {"021559"}


@dataclass
class Lot:
    code: str
    shares: float
    confirm_date: pd.Timestamp
    simulated: bool = False


@dataclass
class SimConfig:
    budget_per_day: float = 0.0                 # 每交易日外部新增预算（元）
    allow_new_equity_index: bool = False        # 默认禁止新增股指申购
    equity_index_codes: frozenset = frozenset()
    daily_limits: dict = field(default_factory=dict)   # code -> 日申购上限（元）
    fee_overrides: dict = field(default_factory=dict)  # code -> {"subscribe": r, "redeem": r}
    params: dict = field(default_factory=lambda: dict(SIM_DEFAULTS))

    def fee(self, code: str, kind: str) -> float:
        return self.fee_overrides.get(code, {}).get(kind, self.params[f"{kind}_fee_rate"])


class Simulator:
    """以 adapters 适配层为唯一数据入口的组合模拟器。"""

    def __init__(self, codes, config: SimConfig, adapter: LocalDataAdapter | None = None):
        self.codes = list(codes)
        self.config = config
        self.adapter = adapter or LocalDataAdapter()
        self._actual_nav: dict[str, pd.Series] = {}
        self._published_nav: dict[str, pd.DataFrame] = {}
        self._dividends = storage.load_dividends()

    # -- 数据准备 ---------------------------------------------------------

    def load(self, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        for code in self.codes:
            df = storage.load_nav(code)
            self._actual_nav[code] = df.set_index("date")["nav"]
        pub = self.adapter.nav(self.codes, pd.Timestamp(end).tz_localize(CST))
        for code, grp in pub.groupby("fund_code"):
            self._published_nav[code] = grp[["nav_date", "published_at",
                                             "adj_nav_total_return"]].reset_index(drop=True)
        dates = set()
        for s in self._actual_nav.values():
            dates |= set(s.loc[start:end].index)
        self.calendar = sorted(dates)
        if not self.calendar:
            raise ValueError("区间内无任何净值披露日")

    def _nav_at(self, code: str, date: pd.Timestamp) -> float | None:
        """真实净值（成交/估值用）：date 当日或之前最近披露。"""
        s = self._actual_nav[code]
        v = s.loc[:date]
        return float(v.iloc[-1]) if len(v) else None

    def _published_state(self, decision_ts: pd.Timestamp) -> dict:
        """决策时点可见的净值（published_at <= 决策时点）。"""
        out = {}
        for code, grp in self._published_nav.items():
            vis = grp[grp["published_at"] <= decision_ts]
            if len(vis):
                r = vis.iloc[-1]
                out[code] = {"nav_date": r["nav_date"].date().isoformat(),
                             "nav": float(r["adj_nav_total_return"])}
        return out

    # -- 主循环 -----------------------------------------------------------

    def run(self, start, end, initial_lots: dict[str, list[Lot]] | None = None,
            initial_cash: float = 0.0, policy=None) -> dict:
        """policy(state) -> {"subscribe": {code: cny}, "redeem": {code: shares}}"""
        self.load(start, end)
        return self._execute(initial_lots, initial_cash, policy)

    def _execute(self, initial_lots=None, initial_cash: float = 0.0, policy=None) -> dict:
        """主循环（与数据加载分离，测试可注入合成净值后直接调用）。"""
        cfg, p = self.config, self.config.params
        cash = float(initial_cash)
        lots: dict[str, list[Lot]] = {c: list(v) for c, v in (initial_lots or {}).items()}
        for c in self.codes:
            lots.setdefault(c, [])
        pending_cash: dict[pd.Timestamp, float] = {}   # 到账日 -> 金额（赎回/分红在途）
        external_flows: list[tuple[pd.Timestamp, float]] = []
        units = 0.0
        unit_nav = 1.0
        # 初始持仓按期初单位净值 1.0 折算为单位数；XIRR 口径下期初市值计为首笔流出
        init_value = self._value(lots, cash, {}, self.calendar[0])
        if init_value > 0:
            units = init_value
            external_flows.append((self.calendar[0], -init_value))
        records, events = [], []
        unit_hist: list[tuple[pd.Timestamp, float]] = []
        prev_date = None

        for day in self.calendar:
            # 现金收益（按自然日间隔计息）
            if prev_date is not None and p["cash_annual_rate"] > 0 and cash > 0:
                cash *= (1 + p["cash_annual_rate"]) ** ((day - prev_date).days / 365)
            prev_date = day
            # 在途资金到账
            cash += pending_cash.pop(day, 0.0)
            # 外部预算注入：按注入前单位净值折算份额（TWR 单位化，入金不改变净值）
            if cfg.budget_per_day > 0:
                if units > 0:
                    units += cfg.budget_per_day / unit_nav
                else:
                    units = cfg.budget_per_day / 1.0
                cash += cfg.budget_per_day
                external_flows.append((day, -cfg.budget_per_day))

            # 分红（现金分红模式；红利再投资未配置，属未知项）
            for code, lot_list in lots.items():
                held = sum(l.shares for l in lot_list)
                if held <= 0:
                    continue
                for ex_date, div in self._dividends.get(code, ()):
                    if pd.Timestamp(ex_date) == day:
                        amt = held * div
                        arrive = self._next_day(day, p["dividend_cash_lag_days"])
                        pending_cash[arrive] = pending_cash.get(arrive, 0.0) + amt
                        events.append({"date": str(day.date()), "code": code,
                                       "event": "DIVIDEND", "amount": round(amt, 2)})

            # 策略决策（只见已公布净值）
            decision_ts = day.tz_localize(CST) + pd.Timedelta(hours=15)
            pub_navs = self._published_state(decision_ts)
            holdings = {}
            for code, lot_list in lots.items():
                shares = sum(l.shares for l in lot_list)
                if shares <= 0:
                    continue
                pub = pub_navs.get(code)
                holdings[code] = {
                    "shares": shares,
                    "value_pub": shares * pub["nav"] if pub else None,
                    "redeemable_shares": sum(l.shares for l in lot_list
                                             if self._redeemable(code, l, day)),
                }
            state = {
                "date": day, "cash": cash,
                "nav": pub_navs,
                "holdings": holdings,
                "value": self._value(lots, cash, pending_cash, day),
                "unit_nav": unit_nav,
                # 已公开口径的组合单位净值历史（lag 内的最近净值决策时不可得）
                "unit_nav_history": [u for d, u in unit_hist
                                     if d + pd.Timedelta(days=p.get("published_lag_days", 2)) <= day],
            }
            orders = policy(state) if policy else {}

            # 策略附带事件（PAUSE/DE_RISK 等）与赎回审议（不执行）
            for ev in (orders.get("events") or []):
                events.append({"date": str(day.date()), **ev})
            for code, shares in (orders.get("redeem_review") or {}).items():
                events.append({"date": str(day.date()), "code": code,
                               "event": "REDUCE_REVIEW", "shares": shares,
                               "note": "提案不执行（allow_redeem=false）"})

            # 申购：T 日真实净值成交（决策时未知）
            for code, amt in (orders.get("subscribe") or {}).items():
                amt = float(amt)
                if amt <= 0:
                    continue
                if code in cfg.equity_index_codes and not cfg.allow_new_equity_index:
                    events.append({"date": str(day.date()), "code": code,
                                   "event": "BUY_REJECTED_FORBIDDEN", "amount": amt})
                    continue
                cap = cfg.daily_limits.get(code)
                if cap is not None and amt > cap:
                    events.append({"date": str(day.date()), "code": code,
                                   "event": "BUY_CLIPPED_LIMIT", "amount": amt,
                                   "filled": cap})
                    amt = cap
                amt = min(amt, cash)
                if amt <= 0:
                    continue
                nav_t = self._nav_at(code, day)
                if nav_t is None:
                    continue
                net = amt * (1 - cfg.fee(code, "subscribe"))
                lots[code].append(Lot(code, net / nav_t, day))
                cash -= amt
                units += 0.0  # 组合内部调仓不改变单位数
                events.append({"date": str(day.date()), "code": code,
                               "event": "BUY", "amount": round(amt, 2), "nav": nav_t})

            # 赎回：逐笔 FIFO，锁定批次不可赎；T 日净值确认，T+lag 到账
            for code, want in (orders.get("redeem") or {}).items():
                remain = float(want)
                for lot in lots[code]:
                    if remain <= 1e-9:
                        break
                    if not self._redeemable(code, lot, day):
                        events.append({"date": str(day.date()), "code": code,
                                       "event": "REDEEM_REJECTED_LOCKED"})
                        continue
                    take = min(lot.shares, remain)
                    nav_t = self._nav_at(code, day)
                    if nav_t is None:
                        break
                    proceeds = take * nav_t * (1 - cfg.fee(code, "redeem"))
                    arrive = self._next_day(day, p["redeem_cash_lag_days"])
                    pending_cash[arrive] = pending_cash.get(arrive, 0.0) + proceeds
                    lot.shares -= take
                    remain -= take
                    events.append({"date": str(day.date()), "code": code,
                                   "event": "REDEEM", "shares": round(take, 2), "nav": nav_t})

            # 估值与单位净值
            value = self._value(lots, cash, pending_cash, day)
            if units > 0:
                unit_nav = value / units
            records.append({"date": day, "value": value, "cash": cash,
                            "pending": sum(pending_cash.values()),
                            "units": units, "unit_nav": unit_nav})
            unit_hist.append((day, unit_nav))

        daily = pd.DataFrame(records)
        return {"daily": daily, "events": pd.DataFrame(events),
                "metrics": self._metrics(daily, external_flows)}

    # -- 内部工具 -----------------------------------------------------------

    def _next_day(self, day: pd.Timestamp, lag_days: int) -> pd.Timestamp:
        """lag 个自然日之后的首个日历交易日（无则顺延到区间末日后，仍在到账队列）。"""
        target = day + pd.Timedelta(days=lag_days)
        for d in self.calendar:
            if d >= target:
                return d
        return target

    def _redeemable(self, code: str, lot: Lot, day: pd.Timestamp) -> bool:
        if code not in ROLLING_LOCK_FUNDS:
            return True
        p = self.config.params
        elapsed = (day - lot.confirm_date).days
        if elapsed < p["rolling_lock_days"]:
            return False
        return (elapsed - p["rolling_lock_days"]) % p["rolling_lock_days"] < p["rolling_open_window_days"]

    def _value(self, lots, cash, pending_cash, day) -> float:
        v = cash + sum(pending_cash.values())
        for code, lot_list in lots.items():
            nav = self._nav_at(code, day)
            if nav is not None:
                v += sum(l.shares for l in lot_list) * nav
        return v

    def _metrics(self, daily: pd.DataFrame, flows) -> dict:
        nav = daily["unit_nav"].to_numpy(dtype=float)
        rets = pd.Series(nav).pct_change().dropna().to_numpy()
        days = max((daily["date"].iloc[-1] - daily["date"].iloc[0]).days, 1)
        rf, td = self.config.params["rf"], 252
        total = nav[-1] / nav[0] - 1 if len(nav) > 1 else 0.0
        cagr = (1 + total) ** (365 / days) - 1 if total > -1 else -1.0
        vol = float(np.std(rets, ddof=1) * np.sqrt(td)) if len(rets) > 1 else 0.0
        ex = rets - rf / td
        std = np.std(rets, ddof=1) if len(rets) > 1 else 0.0
        sharpe = float(np.mean(ex) / std * np.sqrt(td)) if std > 0 else None
        dn = rets[rets < rf / td] - rf / td
        dstd = np.std(dn, ddof=1) if len(dn) > 1 else 0.0
        sortino = float(np.mean(ex) / dstd * np.sqrt(td)) if dstd > 0 else None
        nav_list = nav.tolist()
        mdd = fm.max_drawdown(nav_list) if nav_list else 0.0
        cdar = fm.empirical_cdar(fm.drawdown_path(nav_list), alpha=0.95) if nav_list else 0.0
        xirr = self._xirr(flows, daily["value"].iloc[-1], daily["date"].iloc[-1])
        return {"total_return_twr": total, "cagr": cagr, "ann_vol": vol,
                "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd,
                "cdar95": cdar, "xirr": xirr, "days": len(daily)}

    @staticmethod
    def _xirr(flows, final_value: float, final_date) -> float | None:
        """外部现金流 + 期末市值 的年化内部收益率（二分法）。"""
        if not flows:
            return None
        t0 = flows[0][0]
        cf = [(d, a) for d, a in flows] + [(final_date, final_value)]

        def npv(r):
            return sum(a / (1 + r) ** ((d - t0).days / 365) for d, a in cf)

        lo, hi = -0.9999, 10.0
        if npv(lo) * npv(hi) > 0:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            if npv(lo) * npv(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2


def lots_from_snapshot(adapter: LocalDataAdapter | None = None) -> tuple[dict[str, list[Lot]], str]:
    """从模拟批次配置生成初始持仓（全部 simulated=True）。"""
    adapter = adapter or LocalDataAdapter()
    df = adapter.cashflows_and_lots()
    lots: dict[str, list[Lot]] = {}
    for _, r in df.iterrows():
        lots.setdefault(r["fund_code"], []).append(Lot(
            r["fund_code"], float(r["shares_confirmed"]),
            pd.Timestamp(r["confirmed_at"]), simulated=True))
    return lots, json.dumps(df["order_id"].tolist())[:0] or "SIMULATED"
