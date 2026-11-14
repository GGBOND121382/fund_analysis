#!/usr/bin/env python3
"""Resumable public-market data cache for the all-fund research ranking."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundkit.defensive_screen import market_category  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
FUND_URL = "https://fund.eastmoney.com"
LIST_URL = FUND_URL + "/js/fundcode_search.js"
BUCKET = {"fund_nav": "total_return", "income_per_10000": "money", "adjusted_close": "listed"}


def read_url(url, attempts=2):
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=UA), timeout=15) as response:
                return response.read().decode("utf-8-sig")
        except (OSError, UnicodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(1.5 * (attempt + 1))


def atomically_write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def parse_catalog(js):
    s = json.loads(js[js.index("["):js.rindex("]") + 1])
    frame = pd.DataFrame([(c, n, t) for c, _, n, t, *_ in s if re.fullmatch(r"\d{6}", c)],
                         columns=["code", "name", "type"])
    if frame.empty or frame.code.duplicated().any():
        raise ValueError("empty_or_duplicate_catalog")
    return frame


def array(js, key):
    m = re.search(r"\b" + key + r"\s*=\s*(\[.*?\]);", js, re.S)
    return json.loads(m.group(1)) if m else []


def local_date(series):
    return pd.to_datetime(series, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")


def parse_money(js):
    frame = pd.DataFrame(array(js, "Data_millionCopiesIncome"), columns=["epoch", "income_per_10000"])
    if frame.empty:
        raise ValueError("missing_money_income")
    frame["date"] = local_date(frame.epoch)
    return frame[["date", "income_per_10000"]].sort_values("date")


def parse_nav(js, asof, history_years=3):
    d = pd.DataFrame(array(js, "Data_netWorthTrend"))
    a = pd.DataFrame(array(js, "Data_ACWorthTrend"), columns=["x", "acc_nav"])
    if d.empty or not {"x", "y", "unitMoney", "equityReturn"} <= set(d.columns):
        raise ValueError("missing_daily_nav")
    d = d.merge(a, on="x", how="left", validate="one_to_one").sort_values("x")
    d["date"] = local_date(d.x)
    floor = (asof - pd.DateOffset(years=history_years, months=1)).date().isoformat()
    d = d.loc[d.date.ge(floor) & d.date.le(asof.date().isoformat())].copy()
    if d.empty:
        raise ValueError("no_recent_nav")
    n = pd.to_numeric(d.y, errors="coerce")
    a = pd.to_numeric(d.acc_nav, errors="coerce")
    reported = pd.to_numeric(d.equityReturn, errors="coerce") / 100
    event = d.unitMoney.fillna("").astype(str)
    if not np.isfinite(n).all() or not n.gt(0).all():
        raise ValueError("invalid_daily_nav")
    daily = n / n.shift() - 1
    # Some ETF cumulative NAVs use a different share-denomination scale.
    # Unmarked changes are checked against the published return rather than
    # treating every change in (accumulated NAV - unit NAV) as a distribution.
    delta = (a - n).diff()
    valid_event = event.str.contains(r"分红|折算|拆分|合并", regex=True)
    if (event.ne("") & ~valid_event).iloc[1:].any():
        raise ValueError("unknown_corporate_action")
    for j in np.flatnonzero(valid_event.to_numpy()):
        if j == 0:
            continue
        val = reported.iloc[j]
        if not math.isfinite(val) or val <= -1. or abs(val) > .5:
            raise ValueError(f"event_return_invalid:{d.date.iloc[j]}")
        # For ordinary distribution arithmetic use the higher precision NAV
        # amount when it reconciles with the published return. Otherwise use
        # the provider's adjusted event-day return, not an inferred split.
        guessed = (n.iloc[j] + delta.iloc[j]) / n.iloc[j - 1] - 1
        implied = n.iloc[j - 1] * (1 + val) - n.iloc[j]
        is_cash = "分红" in event.iloc[j] and not re.search(r"拆分|折算|合并", event.iloc[j])
        if is_cash and implied < -.00025 * n.iloc[j - 1]:
            raise ValueError(f"negative_implied_dividend:{d.date.iloc[j]}")
        daily.iloc[j] = (guessed if is_cash and delta.iloc[j] > 0
                         and abs(guessed - val) <= .00016 else val)
    # 3.5bp is generous for a published daily percentage rounded to 2 decimals.
    # Reject potentially missed events, but retain normal NAV arithmetic.
    raw_jump = daily.sub(reported).abs().gt(.00035) & ~valid_event
    inconsistent = raw_jump & (daily.abs().gt(.008) | delta.abs().gt(.00021))
    if inconsistent.iloc[1:].any():
        where = d.loc[inconsistent, "date"].iloc[0]
        raise ValueError(f"unreconciled_nav_return:{where}")
    path = (1 + daily.fillna(0.)).cumprod()
    if not np.isfinite(path).all() or not path.gt(0).all():
        raise ValueError("invalid_total_return_index")
    return pd.DataFrame({"date": d.date, "nav": path,
                         "source": "eastmoney_nav_and_event_return"})


def listed_history(code, asof, history_years=3):
    symbol = ("sh" if code.startswith("5") else "sz") + code
    # Tencent returns at most 640 bars for a request. Bounded segments preserve
    # the full horizon, and adjusted levels must reconcile on their overlap.
    anchor = asof - pd.DateOffset(years=history_years, months=1)
    frames = []
    for n in range(history_years + 1):
        start = anchor + pd.DateOffset(years=n)
        if start > asof:
            break
        end = min(start + pd.DateOffset(years=1, days=5), asof)
        q = urlencode({"param": f"{symbol},day,{start.date()},{end.date()},500,hfq"})
        payload = json.loads(read_url("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?" + q))
        rows = (payload.get("data") or {}).get(symbol, {}).get("hfqday", [])
        if rows:
            frames.extend([(r[0], float(r[2])) for r in rows if start.date().isoformat() <= r[0] <= end.date().isoformat()])
    d = pd.DataFrame(frames, columns=["date", "nav"])
    if d.empty or not np.isfinite(d.nav).all() or not d.nav.gt(0).all():
        raise ValueError("no_valid_hfq_close")
    if d.groupby("date").nav.nunique().gt(1).any():
        raise ValueError("hfq_overlap_mismatch")
    return d.drop_duplicates("date").sort_values("date").assign(source="tencent_hfq_close")


def tail_date(path):
    with path.open("rb") as fh:
        size = fh.seek(0, 2)
        fh.seek(max(0, size - 512))
        tail = fh.read().splitlines()
    if not tail:
        return ""
    return tail[-1].split(b",", 1)[0].decode("utf-8", errors="ignore")


def get_one(row, data, asof, refresh, history_years=3):
    code = row.code
    group, _, _, basis = market_category(row.name, row.type)
    dest = data / BUCKET[basis] / f"{code}.csv"
    asof_text = asof.date().isoformat()
    cached = tail_date(dest) if dest.is_file() else ""
    marker = dest.with_suffix(".history.json")
    head = ""
    if cached:
        with dest.open(encoding="utf-8-sig") as fh:
            next(fh, "")
            head = next(fh, "").split(",", 1)[0]
    if not refresh and cached >= asof_text:
        if history_years <= 3 or head <= (asof - pd.DateOffset(years=history_years)).date().isoformat():
            return code, group, basis, "cached", cached, ""
        if marker.is_file():
            try:
                m = json.loads(marker.read_text(encoding="utf-8"))
                if (m["head"], m["tail"]) == (head, cached) and m["years"] >= history_years:
                    return code, group, basis, "cached", cached, m["note"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
    try:
        note = ""
        if basis == "adjusted_close":
            values = listed_history(code, asof, history_years)
        else:
            js = read_url(f"{FUND_URL}/pingzhongdata/{code}.js")
            if basis == "income_per_10000":
                values = parse_money(js)
            else:
                # An old unreconciled dividend must not erase usable recent
                # history. Record the first rejection and keep coverage explicit.
                for depth in sorted({history_years, *(x for x in (5, 3, 1) if x < history_years)}, reverse=True):
                    try:
                        values = parse_nav(js, asof, depth)
                        break
                    except ValueError as exc:
                        if depth == 1 or history_years == 3:
                            raise
                        if not note:
                            note = f"long_history_rejected:{exc}"
                if depth != history_years:
                    note += f";retained_years={depth}"
        final = values.date.max()
        first = values.date.min()
        # Keep a newer cached endpoint when backfilling a historical run. Join
        # only if proportional adjusted paths agree throughout their overlap.
        if cached and (first > head or final < cached) and first <= cached and final >= head:
            existing = pd.read_csv(dest, dtype={"date": str})
            old = existing.set_index("date")
            new = values.set_index("date")
            common = old.index.intersection(new.index)
            column = "income_per_10000" if basis == "income_per_10000" else "nav"
            if len(common) < 2:
                raise ValueError("backfill_cache_no_overlap")
            if column == "nav":
                ratio = old.loc[common, column] / new.loc[common, column]
                if not np.allclose(ratio, ratio.iloc[0], rtol=1e-6, atol=0.):
                    raise ValueError("backfill_cache_overlap_mismatch")
                new[column] *= ratio.iloc[0]
            elif not np.allclose(old.loc[common, column], new.loc[common, column], rtol=0, atol=1e-10):
                raise ValueError("backfill_cash_overlap_mismatch")
            values = pd.concat([new.loc[new.index < head],old,new.loc[new.index > cached]]).sort_index().reset_index()
            first,final = values.date.min(),values.date.max()
        if final >= cached and (not cached or first <= head or final > cached):
            atomically_write(values, dest)
            if history_years > 3:
                marker.write_text(json.dumps({"head": first, "tail": final, "years": history_years,
                                              "note": note}, ensure_ascii=False), encoding="utf-8")
        return code, group, basis, "updated" if final >= asof_text else "stale", max(final, cached), note
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return code, group, basis, "failed", cached, str(exc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--history-years", type=int, default=3)
    p.add_argument("--codes", help="comma-separated debug subset")
    p.add_argument("--refresh-catalog", action="store_true")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= 16:
        p.error("workers must be between 1 and 16")
    if not 1 <= a.history_years <= 25:
        p.error("history-years must be between 1 and 25")
    end = pd.Timestamp(a.asof).normalize()
    cat = a.data_dir / "fund_list.csv"
    if a.refresh_catalog or not cat.is_file():
        atomically_write(parse_catalog(read_url(LIST_URL)), cat)
    fund_list = pd.read_csv(cat, dtype={"code": str})
    if a.codes:
        codes = set(c.strip() for c in a.codes.split(",") if c.strip())
        if not codes or not codes <= set(fund_list.code):
            p.error("code absent from catalog")
        fund_list = fund_list.loc[fund_list.code.isin(codes)]
    rows = []
    audit_file = a.data_dir / ("fetch_subset_audit.csv" if a.codes else "fetch_audit.csv")
    columns = ["code", "group", "pricing_basis", "state", "latest_date", "reason"]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(get_one, r, a.data_dir, end, a.refresh, a.history_years)
                   for r in fund_list.itertuples(index=False)]
        for i, task in enumerate(as_completed(futures), 1):
            rows.append(task.result())
            if i % 250 == 0 or i == len(futures):
                atomically_write(pd.DataFrame(rows, columns=columns).sort_values("code"), audit_file)
                print(f"{i}/{len(futures)}", flush=True)
    audit = pd.DataFrame(rows, columns=columns)
    print(audit.state.value_counts().to_string())


if __name__ == "__main__":
    main()
