"""Same-category discovery ranking and forward validation from daily NAV paths."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np
import pandas as pd

from fundkit.defensive_screen import Shock

HORIZONS = {"5y": (60, 1050), "3y": (36, 645), "1y": (12, 215), "6m": (6, 105)}
LOW_VOL = re.compile("债|固收|绝对收益|货币|FOF-稳健")
PERIODIC = re.compile(r"(?<!\d)(\d{1,3})\s*(个月|月|年)\s*(?:定期开放|定开|开放|封闭)")
EXTENDED_NOVEMBER = "bonds_202611_through_13"
METHODS = ("research", "industry_proxy", "legacy", "return_only", "drawdown_only", "preservation")
RANK_COLS = tuple(f"{m}_rank" for m in METHODS)
PARETO_AXES = ("return_for_rank", "max_drawdown", "worst_stress_drawdown", "worst_stress_return")
FEATURE_COLS = ("code", "group", "product", "share", "name", "currency", "pricing_basis",
                "scenario_family", "fund_type", "is_long_periodic", "obs", "start", "end",
                "period_return", "return_for_rank", "max_drawdown", "negative_month_loss",
                "negative_month_frequency", "negative_month_mean", "worst_stress_drawdown",
                "worst_stress_return", "stress_count", "stress_ids", "_monthly")


def effective_group(name, group, fund_type):
    """Separate long closed cycles without treating low liquidity as a penalty."""
    if LOW_VOL.search(group) or str(fund_type).startswith("债券"):
        for m in PERIODIC.finditer(name):
            if int(m[1]) * (12 if m[2] == "年" else 1) >= 24:
                return "债券长封闭定开（≥24月）", True
    return group, False


@lru_cache(maxsize=512)
def month_days(period, n, end):
    anchors = pd.period_range(end=period, periods=n+1, freq="M").to_timestamp(how="end").normalize()
    return np.minimum(anchors.values.astype("datetime64[D]").astype("int64"),NavPath.day(end))


@lru_cache(maxsize=512)
def window_dates(end, n):
    period = end.to_period("M")
    month_end = pd.offsets.BDay().rollback(period.end_time.normalize())
    return end-pd.DateOffset(months=n),period if end>=month_end else period-1,n if end>=month_end else n-1


@dataclass
class NavPath:
    dates: pd.DatetimeIndex
    value: np.ndarray
    money: bool = False

    def __post_init__(self):
        self.days = self.dates.values.astype("datetime64[D]").astype("int64")
        self.jumps = np.abs(self.value[1:] / self.value[:-1] - 1)
        self.gaps = np.diff(self.days)

    @staticmethod
    def day(t):
        return pd.Timestamp(t).to_datetime64().astype("datetime64[D]").astype("int64")

    def left(self, t):
        return int(np.searchsorted(self.days, self.day(t), side="left"))

    def right(self, t):
        return int(np.searchsorted(self.days, self.day(t), side="right")) - 1

    def stats(self, i, j):
        v = self.value[i:j+1]
        return float(v[-1] / v[0] - 1), float(max(0., np.max(1. - v / np.maximum.accumulate(v))))

    def monthly(self, end, period, n):
        # Complete calendar months only. Endpoints never come from after the
        # formation date, and the same months are compared within a peer group.
        days = month_days(period,n,end)
        idx = np.searchsorted(self.days, days, side="right") - 1
        if np.any(idx < 0) or np.any(days - self.days[np.maximum(idx, 0)] > 10) or np.any(np.diff(idx) <= 0):
            return None
        v = self.value[idx]
        return v[1:] / v[:-1] - 1


def feature(nav, end, horizon, low_vol, exact=True):
    n, obs_min = HORIZONS[horizon]
    begin,period,complete_months = window_dates(end,n)
    i, j = nav.right(begin), nav.right(end)
    if j < 0 or (exact and nav.dates[j] != end) or nav.day(end) - nav.days[j] > 7:
        return None, "缺截止日净值"
    if i < 0 or nav.day(begin) - nav.days[i] > 7:
        return None, "期限起点缺历史"
    gap_max = 1 if nav.money else 14
    if j-i+1 < obs_min or np.any(nav.gaps[i:j] > gap_max):
        return None, "期限中间覆盖不足"
    if low_vol and np.any(nav.jumps[i:j] > .03):
        return None, "低波类别单日涨跌超过3%待核验"
    # If the cutoff is mid-month, the six-month horizon contains five fully
    # contained calendar months. Do not pull an extra month from before begin.
    monthly = nav.monthly(end,period,complete_months)
    if monthly is None:
        return None, "完整月度样本不足"
    ret, mdd = nav.stats(i, j)
    years = (nav.days[j]-nav.days[i])/365.2425
    return {"obs": j-i+1, "start": nav.dates[i], "end": nav.dates[j], "period_return": ret,
            "return_for_rank": ret if n <= 12 else (1+ret)**(1/years)-1,
            "max_drawdown": mdd, "negative_month_loss": float(-np.minimum(monthly, 0).sum()),
            "negative_month_frequency": float(np.mean(monthly < 0)),
            "negative_month_mean": float(-monthly[monthly < 0].mean()) if np.any(monthly < 0) else 0.,
            "_monthly": monthly}, "可比较"


def stress_metrics(nav, event: Shock, asof, recovery_limit=None):
    i, j = nav.left(event.baseline), nav.right(event.end)
    if (event.end > asof or i >= j or nav.dates[i] != event.baseline or nav.dates[j] != event.end
            or not pd.bdate_range(event.baseline, event.end).isin(nav.dates[i:j+1]).all()):
        return None
    v = nav.value[i:j+1]
    dd = 1. - v / np.maximum.accumulate(v)
    trough = int(np.argmax(dd))
    # Recovery of the *peak-to-trough* drawdown, not simply back to window start.
    peak = int(np.argmax(v[:trough+1]))
    last = nav.right(asof if recovery_limit is None else min(asof, recovery_limit))
    if dd[trough] <= 0:
        state, recovery, days = "无回撤", pd.NaT, 0
    else:
        rec = np.flatnonzero(nav.value[i+trough+1:last+1] >= v[peak] * (1-1e-12))
        k = i+trough+1+int(rec[0]) if len(rec) else None
        state = "已修复" if k is not None else "未观察到修复"
        recovery = nav.dates[k] if k is not None else pd.NaT
        days = int(nav.days[k]-nav.days[i+trough]) if k is not None else None
    return {"stress_return": float(v[-1]/v[0]-1), "stress_drawdown": float(dd[trough]),
            "peak_date": nav.dates[i+peak], "trough_date": nav.dates[i+trough],
            "recovery_date": recovery, "recovery_calendar_days": days,
            "recovery_status": state, "last_observed": nav.dates[last]}


def percentile(s, higher=True):
    if len(s) < 2:
        return pd.Series(.5, index=s.index)
    return (s.rank(ascending=higher, method="average")-1)/(len(s)-1)


def legacy_events(events, asof, begin, current, family):
    selected = [e for e in events if begin <= e.baseline < e.end <= asof
                and (not e.families or family in e.families)]
    # The 13-Nov extension is the same episode, not a second independent
    # stress event. Keep the original variant available for 12-Nov reruns.
    november = (EXTENDED_NOVEMBER if any(e.id == EXTENDED_NOVEMBER for e in selected)
                else "bonds_202611")
    legacy_ids = {"credit_20261030", november}
    if current:
        return [e for e in selected if e.id in legacy_ids]
    # Historical baseline uses the last two completed broad/bond shocks.
    # As-of dates prevent the 2026 selloff from leaking into earlier folds.
    return sorted([e for e in selected if e.id in legacy_ids or e.id.startswith(("rates_", "bond_"))],
                  key=lambda e:e.end)[-2:]


def assemble(catalog, paths, end, horizon, events, current=True, stress_cache=None):
    features, rejected = [], []
    stress_cache = {} if stress_cache is None else stress_cache
    begin = end - pd.DateOffset(months=HORIZONS[horizon][0])
    selected_by_family = {}
    for row in catalog.itertuples(index=False):
        nav = paths.get(row.code)
        if nav is None:
            rejected.append((row.code, row.group, "缺有效净值缓存"))
            continue
        f, status = feature(nav, end, horizon, bool(LOW_VOL.search(row.group)), exact=current)
        if f is None:
            rejected.append((row.code, row.group, status))
            continue
        if row.scenario_family not in selected_by_family:
            selected_by_family[row.scenario_family] = legacy_events(events, end, begin, current, row.scenario_family)
        selected = selected_by_family[row.scenario_family]
        st = []
        for e in selected:
            key = (row.code, e.id)
            if key not in stress_cache:
                r = stress_metrics(nav, e, e.end)
                stress_cache[key] = None if r is None else (r["stress_drawdown"], r["stress_return"])
            st.append(stress_cache[key])
        valid = [x for x in st if x is not None]
        complete = len(selected) == 2 and len(valid) == 2
        features.append(dict(row._asdict(), **f,
                             worst_stress_drawdown=max(x[0] for x in valid) if complete else np.nan,
                             worst_stress_return=min(x[1] for x in valid) if complete else np.nan,
                             stress_count=len(valid), stress_ids=",".join(e.id for e in selected)))
    if not features:
        return pd.DataFrame(columns=FEATURE_COLS), rejected
    df = pd.DataFrame(features)
    df["_a"] = ~df.share.isin(["A", ""])
    df = df.sort_values(["_a","obs","code"], ascending=[True,False,True])
    duplicate = df.duplicated(["group","product","currency","pricing_basis"])
    rejected.extend((r.code,r.group,"同基金其他份额") for r in df.loc[duplicate].itertuples())
    return df.loc[~duplicate].drop(columns="_a"), rejected


def pareto(rows):
    x = rows[list(PARETO_AXES)].to_numpy(dtype=float)
    out = np.full(len(x), np.nan, dtype=object)
    ok = np.isfinite(x).all(axis=1)
    # Sort by desirability. A dominator must precede the candidate. The
    # retained front suffices because dominance is transitive.
    values = x[ok] * np.array([1,-1,-1,1])
    order = np.lexsort(tuple(-values[:,i] for i in (3,2,1,0)))
    front = []
    marks = np.zeros(len(values), dtype=bool)
    for k in order:
        v = values[k]
        if front:
            prev = values[front]
            if np.any(np.all(prev >= v, axis=1) & np.any(prev > v, axis=1)):
                continue
        front.append(k)
        marks[k] = True
    out[ok] = marks
    return out


def ranking(group, min_peers=5, mark_pareto=True):
    """Ranks within a single universe. No score across unlike fund categories."""
    z = group.copy().reset_index(drop=True)
    for m in METHODS:
        z[m+"_score"] = np.nan
        z[m+"_rank"] = np.nan
    z["monthly_hit_rate"] = np.nan
    z["pareto"] = np.nan
    z["peer_count"] = len(z)
    if len(z) < min_peers:
        return z
    monthly = np.stack(z["_monthly"].to_numpy())
    z["monthly_hit_rate"] = (monthly > monthly.mean(axis=0)).mean(axis=1)
    def perc(col, high=True):
        return percentile(z[col], high)
    a, b, c = perc("return_for_rank"), perc("monthly_hit_rate"), perc("negative_month_loss", False)
    z["research_score"] = .50*a + .25*b + .25*c
    group_name = z.group.iloc[0]
    if group_name in {"短债/中短债","中长纯债","利率债","信用债","一级债基"}:
        z["industry_proxy_score"] = .67*a + .33*b
    elif group_name == "二级债基":
        z["industry_proxy_score"] = .56*a + .22*b + .11*perc("negative_month_frequency", False) + .11*perc("negative_month_mean", False)
    # This is a transparent industry-inspired proxy, not the agency's own stars.
    z["return_only_score"] = a
    z["drawdown_only_score"] = perc("max_drawdown", False)
    z["preservation_score"] = c
    ok = z[list(PARETO_AXES)].notna().all(axis=1)
    z["legacy_peer_count"] = int(ok.sum())
    if ok.sum() >= min_peers:
        v = z.loc[ok]
        z.loc[ok,"legacy_score"] = (
            .35*percentile(v.return_for_rank) + .35*percentile(v.max_drawdown,False)
            + .20*percentile(v.worst_stress_drawdown,False) + .10*percentile(v.worst_stress_return))
    if mark_pareto:
        z["pareto"] = pareto(z)
    for m in METHODS:
        s = m+"_score"
        t = z[z[s].notna()].sort_values([s,"code"],ascending=[False,True])
        if m == "legacy":
            t = t.sort_values([s,"worst_stress_drawdown","max_drawdown","return_for_rank","code"],
                              ascending=[False,True,True,False,True])
        z.loc[t.index, m+"_rank"] = np.arange(1,len(t)+1)
    return z


def forward(nav, signal, end):
    # Selection uses NAV through signal; first subsequent observed NAV is
    # the entry proxy. No extra next-day return is booked.
    i, j = nav.right(signal)+1, nav.right(end)
    if i >= j or nav.days[i]-nav.day(signal) > 10 or nav.day(end)-nav.days[j] > 7:
        return None
    if np.any(nav.gaps[i:j] > (1 if nav.money else 14)):
        return None
    ret, dd = nav.stats(i,j)
    return ret, dd, nav.dates[i], nav.dates[j]


def fold_dates(asof, folds, months):
    anchor = asof - pd.offsets.MonthEnd(0)
    if anchor > asof:
        anchor -= pd.offsets.MonthEnd(1)
    return [anchor-pd.offsets.MonthEnd(months*k) for k in range(folds,0,-1)]


def compare_forward(catalog, paths, events, asof, folds=12, months=3, top=10, min_peers=5,
                    historical_catalogs=None):
    """Fixed-rule historical selections, evaluated on subsequent nonoverlapping windows."""
    selections, summary = [], []
    stress_cache = {}
    for signal in fold_dates(asof,folds,months):
        end = signal + pd.offsets.MonthEnd(months)
        eligible_catalog = catalog if historical_catalogs is None else historical_catalogs[signal]
        for h in HORIZONS:
            frame,_ = assemble(eligible_catalog, paths, signal, h, events, current=False, stress_cache=stress_cache)
            for group,g in frame.groupby("group", sort=True):
                for cohort in ("all","stress_matched"):
                    pool = g if cohort == "all" else g[g.stress_count.eq(2)]
                    if len(pool) < min_peers:
                        continue
                    z = ranking(pool,min_peers,mark_pareto=False)
                    k = min(top, max(1, int(np.ceil(.20*len(z)))))
                    available = [m for m in METHODS if (m != "legacy" or cohort == "stress_matched")
                                 and z[m+"_rank"].notna().sum() >= k]
                    fold_rows = []
                    for m in available:
                        selected = z.loc[z[m+"_rank"].le(k)].sort_values(m+"_rank")
                        results = []
                        for code,name in selected[["code","name"]].itertuples(index=False,name=None):
                            row = {"signal":signal,"evaluation_end":end,"horizon":h,"group":group,
                                   "cohort":cohort,"method":m,"code":code,"name":name}
                            ret = forward(paths[code],signal,end)
                            if ret is not None:
                                r,dd,entry,exit_ = ret
                                row.update(fwd_return=r,fwd_mdd=dd,entry=entry,exit=exit_)
                                results.append((r,dd))
                            row["observed"] = ret is not None
                            selections.append(row)
                        a = np.asarray(results)
                        fold_rows.append({"signal":signal,"evaluation_end":end,"horizon":h,"group":group,
                                          "cohort":cohort,"method":m,"candidate_count":len(z),
                                          "selected":k,"observed":len(a),"coverage":len(a)/k,
                                          "median_fwd_return":float(np.median(a[:,0])) if len(a)==k else np.nan,
                                          "median_fwd_mdd":float(np.median(a[:,1])) if len(a)==k else np.nan,
                                          "worst_selected_mdd":float(a[:,1].max()) if len(a)==k else np.nan})
                    all_complete = bool(fold_rows) and all(r["observed"]==k for r in fold_rows)
                    for r in fold_rows:
                        r["paired_complete"] = all_complete
                        summary.append(r)
    return pd.DataFrame(selections),pd.DataFrame(summary)
