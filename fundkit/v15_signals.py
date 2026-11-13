"""Predeclared research signals. Inputs must already be decision-time visible."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fundkit.adapters import CST


VARIANTS = {
    "fixed": (False, False, False),
    "dip": (True, False, False),
    "trend_only": (False, True, False),
    "trend": (True, True, False),
    "valuation": (True, True, True),
}


class ValuationHistory:
    """Monthly trailing percentiles using only snapshots released by the cutoff."""
    def __init__(self, path):
        if not Path(path).is_file():
            raise ValueError("VALUATION_DATA_BLOCKED: missing " + str(path))
        f = pd.read_csv(path, dtype={"index_code": str})
        required = {"index_code", "observed_at", "published_at", "pe", "pb"}
        if not required.issubset(f.columns):
            raise ValueError("VALUATION_DATA_BLOCKED: expected " + ", ".join(sorted(required)))
        f = f.loc[f.index_code == "000300"].copy()
        if f.empty:
            raise ValueError("VALUATION_DATA_BLOCKED: no CSI 300 (000300) rows")
        try:
            f["observed_at"] = f.observed_at.map(self.timestamp)
            f["published_at"] = f.published_at.map(self.timestamp)
            for c in ["pe", "pb"]:
                f[c] = pd.to_numeric(f[c], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("VALUATION_DATA_BLOCKED: invalid dates or valuations") from exc
        if (f[["observed_at", "published_at"]].isna().any().any()
                or (f.published_at < f.observed_at).any()
                or f.duplicated(["observed_at", "published_at"]).any()
                or not np.isfinite(f[["pe", "pb"]]).all().all()
                or (f[["pe", "pb"]] <= 0).any().any()):
            raise ValueError("VALUATION_DATA_BLOCKED: invalid or duplicate valuation vintage")
        self.source_quality = sorted(set(f.get("source_quality", pd.Series(
            ["USER_SUPPLIED_UNVERIFIED_VINTAGES"])).dropna().astype(str)))
        self.is_proxy = any("PROXY" in x for x in self.source_quality)
        self.frame = f.sort_values(["observed_at", "published_at"])

    @staticmethod
    def timestamp(value):
        ts = pd.Timestamp(value)
        return ts.tz_localize(CST) if ts.tzinfo is None else ts.tz_convert(CST)

    def signal(self, day):
        cutoff = self.timestamp(day).normalize() + pd.Timedelta(hours=14, minutes=55)
        f = self.frame.loc[self.frame.published_at <= cutoff].drop_duplicates(
            "observed_at", keep="last").copy()
        if f.empty or cutoff - f.iloc[-1].observed_at > pd.Timedelta(days=45):
            return None
        f["month"] = f.observed_at.dt.strftime("%Y-%m")
        month = f.drop_duplicates("month", keep="last").tail(121)
        if len(month) < 37:
            return None
        hist, now = month.iloc[:-1], month.iloc[-1]
        ranks = []
        for c in ["pe", "pb"]:
            ranks.append(float(((hist[c] < now[c]).sum() + .5*(hist[c] == now[c]).sum())/len(hist)))
        rank = sum(ranks)/2
        factor = 1.25 if rank <= .20 else .75 if rank >= .80 else 1.
        return {"rank": rank, "factor": factor, "asof": now.observed_at.date().isoformat()}


def market_signal(rows, day):
    # Input is an ordered, published total-return NAV sequence, not raw NAV.
    if len(rows) < 126:
        return None
    tail = rows[-126:]
    last = pd.Timestamp(tail[-1]["date"])
    age = (pd.Timestamp(day).normalize() - last.normalize()).days
    if not 0 <= age <= 10:
        return None
    prices = np.asarray([r["total_return_nav"] for r in tail], float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        return None
    ratio = prices[-1]/prices.mean()
    dd20 = 1-prices[-1]/prices[-20:].max()
    # Pullback within a still-intact slow trend, not an attempt to catch a crash.
    return {"date": last, "ratio": ratio, "drawdown20": dd20,
            "dip": .02 <= dd20 <= .08 and ratio >= 1.0}


class TrendState:
    """Two-step exit/re-entry with five fresh marks per step and a deadband."""
    def __init__(self):
        self.level = 2
        self.last = None
        self.streak = 0
        self.direction = 0

    def update(self, signal, *, allow_exit):
        if signal is None or (self.last is not None and signal["date"] <= self.last):
            return False
        self.last = signal["date"]
        ratio = signal["ratio"]
        direction = -1 if ratio < .99 else 1 if ratio > 1.01 else 0
        self.streak = self.streak+1 if direction and direction == self.direction else int(bool(direction))
        self.direction = direction
        if self.streak < 5:
            return False
        previous = self.level
        if direction == -1:
            if allow_exit or self.level == 2:
                # First risk-off state pauses buys even if the stress trigger has
                # not fired. Only the caller's risk gate can turn it into a sale.
                self.level = max(0, self.level-1)
        elif direction == 1:
            self.level = min(2, self.level+1)
        if self.level != previous:
            self.streak = 0
        return self.level != previous

    @property
    def scale(self):
        return self.level/2

    @property
    def pause(self):
        return self.direction < 0 and (self.level < 2 or self.streak >= 2)


class IndexHistory:
    """Optional long total-return benchmark series by fund code, with release times."""
    def __init__(self, path):
        if not Path(path).is_file():
            raise ValueError("INDEX_DATA_BLOCKED: missing " + str(path))
        f = pd.read_csv(path, dtype={"fund_code": str})
        required = {"fund_code", "observed_at", "published_at", "total_return_index"}
        if not required.issubset(f.columns):
            raise ValueError("INDEX_DATA_BLOCKED: expected " + ", ".join(sorted(required)))
        f["observed_at"] = f.observed_at.map(ValuationHistory.timestamp)
        f["published_at"] = f.published_at.map(ValuationHistory.timestamp)
        f["total_return_index"] = pd.to_numeric(f.total_return_index, errors="raise")
        if (f.empty or f[["fund_code", "observed_at", "published_at"]].isna().any().any()
                or (f.published_at < f.observed_at).any()
                or f.duplicated(["fund_code", "observed_at"]).any()
                or not np.isfinite(f.total_return_index).all()
                or (f.total_return_index <= 0).any()):
            raise ValueError("INDEX_DATA_BLOCKED: invalid benchmark series")
        self.frames = {code: g.sort_values("observed_at") for code, g in f.groupby("fund_code")}

    def rows(self, code, day):
        f = self.frames.get(code)
        if f is None:
            return []
        cutoff = ValuationHistory.timestamp(day).normalize() + pd.Timedelta(hours=14, minutes=55)
        f = f.loc[f.published_at <= cutoff].tail(126)
        return [{"date": x.observed_at.date().isoformat(), "total_return_nav": x.total_return_index}
                for x in f.itertuples(index=False)]
