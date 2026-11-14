"""Research screen for domestic bond and bond-biased funds, using as-of NAVs."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from fundkit.metrics import adjust_nav

GROUPS = {
    "债券型-中短债": "短债/中短债", "债券型-长债": "中长纯债",
    "债券型-利率债": "利率债", "债券型-信用债": "信用债",
    "债券型-混合一级": "一级债基", "债券型-混合二级": "二级债基",
    "混合型-偏债": "偏债混合",
}
LOCK = re.compile(r"定期开放|定开|封闭|持有期|滚动持有|(?:个月|月|日|天|年)持有")
SHARE = re.compile(r"(?<=[\u4e00-\u9fff])\s*([A-Z])(?:类)?$")
WINDOWS = {"6m": (6, 105), "1y": (12, 215), "2y": (24, 430), "3y": (36, 645)}
AXES = ("return_score", "drawdown_score", "shock_score")
TIER_WINDOWS = {
    "6m": ("6m",), "1y": ("1y",), "2y": ("1y", "2y"),
    "3y": ("1y", "2y", "3y"),
}


@dataclass(frozen=True)
class Shock:
    id: str
    label: str
    baseline: pd.Timestamp
    end: pd.Timestamp
    source: str = ""
    families: tuple[str, ...] = ()

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", self.id) or self.baseline >= self.end:
            raise ValueError(f"invalid shock: {self.id}")


def category(name, fund_type):
    if fund_type in ("混合型-偏债", "偏债混合型"):
        return "偏债混合"
    if str(fund_type).startswith("债券型"):
        if re.search(r"转债|可转换", name):
            return "可转债敞口"
        return GROUPS.get(fund_type, "其他债券型")
    return None


def market_category(name, fund_type):
    """Keep vendor subtypes, disambiguate currencies, and preserve unknowns."""
    typ = str(fund_type).strip() if pd.notna(fund_type) else ""
    group = category(name, typ) or typ or "类型待核验"
    if typ.lower() in {"reits", "reit", "公募reits"}:
        return "公募REITs", "reits", "CNY", "adjusted_close"
    if typ in {"货币型-普通货币", "货币型"}:
        return group, "money", "CNY", "income_per_10000"
    family = ("commodity" if "商品" in typ or (typ == "指数型-其他" and
                  re.search(r"黄金|白银|原油|豆粕|有色金属|能源化工|商品期货", name))
              else "overseas" if typ.startswith("QDII") or "海外" in typ or "QDII" in name.upper()
              else "domestic")
    currency = ("USD" if re.search(r"美元|USD", name, re.I) else
                "HKD" if re.search(r"港元|港币|HKD", name, re.I) else
                "EUR" if re.search(r"欧元|EUR", name, re.I) else "CNY")
    if family == "commodity" and "商品" not in typ:
        group += " · 商品主题"
    elif family == "overseas" and not (typ.startswith("QDII") or "海外" in typ):
        group += " · 海外"
    if currency != "CNY":
        group += f" · {currency}"
    return group, family, currency, "fund_nav"


def money_index(raw, end):
    """Reinvest each explicitly reported cash income; never compound 7-day yields."""
    d = raw[["date", "income_per_10000"]].copy()
    d["date"] = pd.to_datetime(d.date, errors="raise").dt.normalize()
    d = d.loc[d.date <= end].sort_values("date")
    income = pd.to_numeric(d.income_per_10000, errors="coerce") / 10000.
    if (d.empty or d.date.duplicated().any() or not np.isfinite(income).all()
            or not income.gt(-1.).all()):
        raise ValueError("invalid_money_income")
    # A gap might be a lump-sum income disclosure or a missing day. Without a
    # source flag we cannot distinguish them, so use only the final continuous run.
    gaps = np.flatnonzero(d.date.diff().gt(pd.Timedelta(days=1)))
    i = int(gaps[-1]) if len(gaps) else 0
    return pd.Series(np.cumprod(1. + income.iloc[i:].to_numpy()), index=d.date.iloc[i:], name="nav")


def share_identity(name):
    m = SHARE.search(name)
    return (name[:m.start()], m.group(1)) if m else (name, "")


def adjusted_asof(raw, end, divs=(), splits=()):
    if not {"date", "nav"} <= set(raw.columns):
        raise ValueError("missing_date_or_nav")
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise").dt.normalize()
    d = d.loc[d["date"] <= end].sort_values("date").reset_index(drop=True)
    d["nav"] = pd.to_numeric(d["nav"], errors="coerce")
    if d.empty or d.date.duplicated().any() or not np.isfinite(d.nav).all() or not d.nav.gt(0).all():
        raise ValueError("empty_duplicate_or_invalid_nav")
    dividends = {pd.Timestamp(k): float(v) for k, v in divs}
    split_map = {pd.Timestamp(k): float(v) for k, v in splits}
    if "acc_nav" in d:
        # A change in cumulative-minus-unit NAV is a useful check for missing
        # corporate actions. Check amounts on dividend dates as well.
        acc = pd.to_numeric(d["acc_nav"], errors="coerce")
        delta = (acc - d.nav).diff()
        cash = d.date.map(dividends).fillna(0.)
        unexplained = (delta - cash).abs().gt(.00021) & ~d.date.isin(split_map)
        if unexplained.any():
            raise ValueError("unreconciled_dividend_or_split")
    adj = adjust_nav(d, tuple(dividends.items()), tuple(split_map.items())).set_index("date").nav
    if not np.isfinite(adj).all() or not adj.gt(0).all():
        raise ValueError("invalid_adjusted_nav")
    return adj


def path_stats(s):
    v = s.to_numpy(dtype=float)
    return float(v[-1] / v[0] - 1), float(np.max(1. - v / np.maximum.accumulate(v)))


def calculate_fund(nav, end, shocks):
    row = {"nav_start": nav.index[0].date().isoformat(),
           "nav_end": nav.index[-1].date().isoformat(), "nav_observations": len(nav)}
    row["large_move_review"] = bool(nav.pct_change().abs().gt(.03).any())
    row["asof_exact"] = end in nav.index
    row["horizon_tier"] = ""
    row["shock_observations"] = 0
    if row["asof_exact"]:
        row["available_history_return"], row["available_history_mdd"] = path_stats(nav)
        for key, (months, minimum) in WINDOWS.items():
            begin = end - pd.DateOffset(months=months)
            i = nav.index.searchsorted(begin, side="right") - 1
            if i < 0 or begin - nav.index[i] > pd.Timedelta(days=7):
                continue
            s = nav.iloc[i:]
            if len(s) < minimum or s.index.to_series().diff().max() > pd.Timedelta(days=14):
                continue
            ret, mdd = path_stats(s)
            years = (s.index[-1] - s.index[0]).days / 365.2425
            row[f"return_{key}"], row[f"mdd_{key}"] = ret, mdd
            if key != "6m":
                row[f"cagr_{key}"] = (1 + ret) ** (1 / years) - 1
        for tier, windows in TIER_WINDOWS.items():
            if all(f"return_{key}" in row for key in windows):
                row["horizon_tier"] = tier

    observed = []
    for event in shocks:
        window = nav.loc[event.baseline:event.end]
        # Do not turn a missing stress day into apparent low volatility.
        if (event.baseline not in window.index or event.end not in window.index
                or not pd.bdate_range(event.baseline, event.end).isin(window.index).all()):
            continue
        ret, mdd = path_stats(window)
        row[f"shock_{event.id}_return"] = ret
        row[f"shock_{event.id}_mdd"] = mdd
        observed.append((ret, mdd))
    row["shock_observations"] = len(observed)
    if observed:
        row["worst_shock_return"] = min(x[0] for x in observed)
        row["worst_shock_mdd"] = max(x[1] for x in observed)
        row["median_shock_mdd"] = float(np.median([x[1] for x in observed]))
    return row


def percentile(s, higher):
    if len(s) < 2:
        return pd.Series(.5, index=s.index)
    return (s.rank(method="average", ascending=higher) - 1) / (len(s) - 1)


def pareto(values):
    # Vectorized dominance comparisons avoid nested Python loops on large groups.
    return np.array([not ((values >= a).all(axis=1) & (values > a).any(axis=1)).any()
                     for a in values], dtype=bool)


def rank_candidates(data, *, shock_count, top, min_peers=3):
    """Select one representative per product, then compare within category and history tier."""
    if shock_count < 2 or top < 1 or min_peers < 1:
        raise ValueError("need at least two shocks, a positive top and a positive peer minimum")
    rows = data.copy()
    if rows.empty:
        return rows.assign(selection_status=pd.Series(dtype=str)), rows
    rows["selection_status"] = rows["selection_status"].fillna("")
    required = rows["required_shocks"] if "required_shocks" in rows else shock_count
    eligible = (rows.selection_status.eq("") & rows.asof_exact.fillna(False).eq(True)
                & rows.horizon_tier.fillna("").ne("") & rows.shock_observations.eq(required))
    rows.loc[rows.selection_status.eq("") & ~eligible, "selection_status"] = "样本或冲击覆盖不足"
    reps = rows.loc[eligible].copy()
    reps["a_priority"] = ~reps.share.isin(["", "A"])
    reps["tier_priority"] = reps.horizon_tier.map({k: -i for i, k in enumerate(TIER_WINDOWS)})
    reps = reps.sort_values(["tier_priority", "a_priority", "nav_observations", "code"],
                           ascending=[True, True, False, True])
    keep = reps.drop_duplicates(["group", "product"])
    rows.loc[reps.index.difference(keep.index), "selection_status"] = "同基金其他份额"
    for _, group in keep.groupby(["group", "horizon_tier"], sort=True):
        if "shock_set" in group and group.shock_set.nunique() != 1:
            raise ValueError("funds in same peer group have different stress scenarios")
        if len(group) < min_peers:
            rows.loc[group.index, "selection_status"] = "同类样本不足"
            continue
        win = TIER_WINDOWS[group.horizon_tier.iloc[0]]
        scores = pd.DataFrame(index=group.index)
        scores["return_score"] = pd.concat([percentile(group[f"return_{w}"], True) for w in win], axis=1).mean(axis=1)
        scores["drawdown_score"] = pd.concat([percentile(group[f"mdd_{w}"], False) for w in win], axis=1).mean(axis=1)
        scores["shock_score"] = pd.concat([
            percentile(group.worst_shock_return, True), percentile(group.worst_shock_mdd, False),
            percentile(group.median_shock_mdd, False),
        ], axis=1).mean(axis=1)
        scores["pareto"] = pareto(scores[list(AXES)].to_numpy())
        scores["balance_score"] = scores[list(AXES)].mean(axis=1)
        scores["weakest_score"] = scores[list(AXES)].min(axis=1)
        scores["peer_count"] = len(group)
        scores["code"] = group.code
        scores = scores.sort_values(["pareto", "balance_score", "weakest_score", "code"],
                                    ascending=[False, False, False, True])
        scores["peer_rank"] = np.arange(1, len(scores) + 1)
        scores["selection_status"] = np.where(scores.peer_rank <= top, "入围", "未入围")
        for c in scores.columns.drop("code"):
            if c not in rows:
                rows[c] = pd.Series(index=rows.index, dtype="boolean" if c == "pareto" else "float64")
            rows.loc[scores.index, c] = scores[c]
    short = rows[rows.selection_status.eq("入围")].copy()
    if len(short):
        short = short.sort_values(["group", "horizon_tier", "peer_rank"])
    return rows, short
