"""算法层 as-of 数据适配接口（05_AKSHARE_ADAPTER_CONTRACT.md §3 的最小实现）。

只读复用现有 storage/metrics 缓存，不重做采集。引擎只依赖本模块的六个接口，
不感知 akshare 具体函数名。

关键约定：
- published_at 缺失是本地缓存的最大缺口。本适配层按规则推断：
  净值披露统一假设为估值日后 PUBLISHED_LAG_DAYS 个自然日 23:59（ Asia/Shanghai ），
  默认 2 天以覆盖 QDII 等延迟披露。该字段是**规则推断（C/PROXY 级证据）**，
  不是真实披露时间戳，所有输出须保留 quality_flag 说明。
- as-of 强制：任何接口只返回 published_at <= until_timestamp 的行；
  `assert_no_lookahead` 用于引擎侧自检。
- 缺失表（季报持仓/风险因子）返回带 MISSING 状态的空 DataFrame，不伪造。
- 账户批次为模拟批次（config/account_snapshot.json lots_mode=SIMULATED），
  输出常量 ACCOUNT_DRAWDOWN_UNVERIFIABLE 供报告层引用。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from fundkit import metrics, storage

CST = timezone(timedelta(hours=8))

ACCOUNT_DRAWDOWN_UNVERIFIABLE = (
    "ACCOUNT_DRAWDOWN_UNVERIFIABLE: 无逐笔确认流水，批次为快照市值反推的模拟批次；"
    "真实历史 MDD/TWR/可赎回份额不可验证"
)

PUBLISHED_LAG_DAYS = 2  # 规则推断的披露延迟（自然日），QDII 常见 T+1/T+2，取保守值

SNAPSHOT_FILE = os.path.join("config", "account_snapshot.json")

NAV_COLUMNS = [
    "fund_code", "nav_date", "published_at", "unit_nav",
    "cumulative_nav", "adj_nav_total_return", "source", "fetched_at", "quality_flag",
]

TRADE_RULE_DEFAULTS = {
    "settlement_lag": "T+1(assumption)",
    "cutoff_time": "15:00",
    "timezone": "Asia/Shanghai",
    "verified": False,
}


def _to_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(CST)
    return ts


def load_account_snapshot() -> dict:
    with open(SNAPSHOT_FILE, encoding="utf-8") as f:
        return json.load(f)


class LocalDataAdapter:
    """对接现有本地缓存的只读适配器。"""

    def __init__(self, published_lag_days: int = PUBLISHED_LAG_DAYS):
        self.published_lag_days = published_lag_days

    # -- 表1: nav_total_return_daily -------------------------------------

    def nav(self, codes: list[str], until_timestamp) -> pd.DataFrame:
        """总回报口径日净值长表，as-of 过滤后返回。

        published_at = nav_date + published_lag_days 天 23:59（规则推断，C 级证据）。
        """
        until = _to_ts(until_timestamp)
        div_map = storage.load_dividends()
        split_map = storage.load_splits()
        frames = []
        for code in codes:
            try:
                df = storage.load_nav(code)
            except Exception:  # noqa: BLE001 - 缺文件/损坏按缺失处理
                continue
            adj = metrics.adjust_nav(df, div_map.get(code, ()), split_map.get(code, ()))
            out = pd.DataFrame({
                "fund_code": code,
                "nav_date": df["date"],
                "unit_nav": df["nav"],
                "cumulative_nav": df["acc_nav"] if "acc_nav" in df.columns else pd.NA,
                "adj_nav_total_return": adj["nav"],
            })
            out["published_at"] = out["nav_date"].map(
                lambda d: d + timedelta(days=self.published_lag_days)
            ).dt.tz_localize(CST) + pd.Timedelta(hours=23, minutes=59)
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=NAV_COLUMNS)
        nav = pd.concat(frames, ignore_index=True)
        nav = nav[nav["published_at"] <= until].reset_index(drop=True)
        nav["source"] = "local_cache(origin_unrecorded)"
        nav["fetched_at"] = None  # 现有缓存未记录逐行抓取时间
        nav["quality_flag"] = "B/SINGLE_SOURCE; PUBLISHED_AT_RULE_INFERRED"
        return nav[NAV_COLUMNS]

    # -- 表2/表3: 季报持仓披露 / 市场风险因子 -------------------------------

    def disclosures(self, codes: list[str], until_timestamp) -> pd.DataFrame:
        """季报持仓披露（data/disclosures/{code}.csv，由 main.py extra 采集）。

        as-of 强制：只返回 published_at <= until 的披露。
        未采集的基金静默跳过（调用方按 coverage 判定 MISSING）。
        """
        until = _to_ts(until_timestamp)
        frames = []
        for code in codes:
            path = os.path.join("data", "disclosures", f"{code}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, dtype={"fund_code": str})
            df["published_at"] = pd.to_datetime(df["published_at"]).dt.tz_localize(CST)
            frames.append(df[df["published_at"] <= until])
        if not frames:
            return pd.DataFrame(columns=["fund_code", "report_asof", "published_at",
                                         "disclosure_completeness", "quality_flag"])
        return pd.concat(frames, ignore_index=True)

    def risk_factors(self, until_timestamp) -> pd.DataFrame:
        """市场风险因子（data/factors/daily.csv）。未采集返回 MISSING 空表。"""
        path = os.path.join("data", "factors", "daily.csv")
        if not os.path.exists(path):
            return pd.DataFrame(columns=["date", "published_at", "quality_flag"])
        until = _to_ts(until_timestamp)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["published_at"] = pd.to_datetime(df["published_at"])
        return df[df["published_at"] <= until].reset_index(drop=True)

    # -- 表4: account_lots_and_cashflows（模拟批次） ------------------------

    def account_snapshot(self, snapshot_timestamp=None) -> pd.DataFrame:
        """账户快照（截图转录）。snapshot_timestamp 仅作一致性校验提示。"""
        snap = load_account_snapshot()
        df = pd.DataFrame(snap["funds"])
        df["snapshot_date"] = snap["snapshot_date"]
        df["lots_mode"] = snap["lots_mode"]
        return df

    def cashflows_and_lots(self, until_timestamp=None) -> pd.DataFrame:
        """模拟批次：按快照市值 / 快照日前最新单位净值反推份额。

        每基金一个模拟批次，confirmed_at = 快照日；021559 滚动持有批次
        next_redeem_date 按 快照日+30天 的保守假设。全部标记 simulated=True。
        """
        snap = load_account_snapshot()
        snap_date = pd.Timestamp(snap["snapshot_date"])
        rows = []
        for f in snap["funds"]:
            code = f["code"]
            try:
                nav_df = storage.load_nav(code)
                hist = nav_df[nav_df["date"] <= snap_date]
                if hist.empty:
                    continue
                nav0 = float(hist["nav"].iloc[-1])
            except Exception:  # noqa: BLE001
                continue
            shares = f["value_cny"] / nav0
            lot = {
                "fund_code": code,
                "order_id": f"SIM-{snap['snapshot_date']}-{code}",
                "order_type": "SIMULATED_LOT",
                "confirmed_at": snap["snapshot_date"],
                "shares_confirmed": shares,
                "nav_confirmed": nav0,
                "gross_amount_cny": f["value_cny"],
                "total_fees_cny": None,  # 未知，不得假造
                "dividend_mode": None,   # 未知
                "simulated": True,
                "next_redeem_date": None,
            }
            if code == "021559":  # 30天滚动持有：保守假设快照日起锁定30天
                lot["next_redeem_date"] = (snap_date + timedelta(days=30)).strftime("%Y-%m-%d")
            rows.append(lot)
        return pd.DataFrame(rows)

    # -- 表5: fund_trade_rules ------------------------------------------

    def trade_rules(self, codes: list[str], at_timestamp=None) -> pd.DataFrame:
        """交易规则：仅有当前申赎状态快照，无生效时间历史。

        at_timestamp 早于快照抓取日时结果不代表当时规则，调用方须自行判断。
        """
        status = storage.load_trade_status()
        df = status[status["code"].isin(codes)].copy()
        df = df.rename(columns={"sub_status": "subscribe_status",
                                "red_status": "redeem_status",
                                "daily_limit": "subscribe_daily_max_cny"})
        df["rule_effective_from"] = None  # 无历史
        df["snapshot_only"] = True
        for k, v in TRADE_RULE_DEFAULTS.items():
            df[k] = v
        df["fees_by_holding_period"] = None  # 未接入
        df["unlock_calendar"] = None         # 未接入（021559 需逐笔窗口）
        return df.reset_index(drop=True)


def assert_no_lookahead(df: pd.DataFrame, decision_timestamp,
                        col: str = "published_at") -> None:
    """引擎侧自检：决策时点之前不得有未公开数据进入。"""
    ts = _to_ts(decision_timestamp)
    if col in df.columns and len(df):
        bad = df[pd.to_datetime(df[col]) > ts]
        if len(bad):
            raise AssertionError(
                f"未来信息泄露: {len(bad)} 行 {col} 晚于决策时点 {ts}")
