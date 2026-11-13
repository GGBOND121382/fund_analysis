"""Exact daily unit-NAV attribution for simulated fund orders and cash flows."""
from __future__ import annotations

import math

import pandas as pd


class Attribution:
    def __init__(self):
        self.rows = []

    def add(self, day, nav, effects):
        previous = self.rows[-1]["unit_nav"] if self.rows else 1.0
        actual = nav / previous - 1.0
        if abs(math.fsum(effects.values()) - actual) > 1e-9:
            raise ArithmeticError(f"ATTRIBUTION_RECONCILIATION_FAILED: {day.date()}")
        self.rows.append({"date": day, "unit_nav": nav, "daily_return": actual,
                          "contributions": dict(effects)})

    def result(self):
        peak, peak_idx, best, interval = 1.0, -1, 0.0, (-1, -1)
        for idx, row in enumerate(self.rows):
            nav = row["unit_nav"]
            if nav >= peak:
                peak, peak_idx = nav, idx
            dd = 1.0 - nav / peak
            if dd > best:
                best, interval = dd, (peak_idx, idx)
        p, t = interval

        def linked(start, stop):
            anchor = self.rows[start]["unit_nav"] if start >= 0 else 1.0
            prev = anchor
            total = {}
            for row in self.rows[start + 1:stop + 1]:
                for code, value in row["contributions"].items():
                    total[code] = total.get(code, 0.0) + value * prev / anchor
                prev = row["unit_nav"]
            if abs(math.fsum(total.values()) - (prev / anchor - 1)) > 1e-9:
                raise ArithmeticError("ATTRIBUTION_LINK_RECONCILIATION_FAILED")
            return total

        def date(i):
            return self.rows[i]["date"].date().isoformat() if i >= 0 else None

        rows = [{**{k: v for k, v in r.items() if k != "contributions"},
                 **r["contributions"]} for r in self.rows]
        return {
            "daily": pd.DataFrame(rows).fillna(0.0),
            "total_return_contribution": linked(-1, len(self.rows) - 1),
            "drawdown": {"peak_date": date(p), "trough_date": date(t),
                         "max_drawdown": best,
                         "loss_contribution": {k: -v for k, v in linked(p, t).items()}},
        }
