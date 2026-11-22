"""Evidence-backed product and trading-state resolution. Names are NEVER execution evidence."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TRADE_STATES = {"OPEN", "LIMITED", "PAUSED", "CLOSED", "UNKNOWN"}
VALID_PRICING = {"market_value", "amortized_cost", "mixed", "unknown"}
# This is an audit trigger, not a classifier of liquidity or contractual cycles.
CONTRACT_REVIEW_HINT = re.compile(r"定开|定期|封闭|持有期|滚动持有|\d+个月|三年债券")


def product_review_hint(name: str) -> bool:
    return bool(CONTRACT_REVIEW_HINT.search(str(name)))


def read_timestamped_evidence(path: str | Path | None, asof, required=()):
    """Reject malformed, unsourced evidence; discard future-known facts without leakage."""
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=["code", "known_at", "source_url", *required])
    data = pd.read_csv(path, dtype={"code": str}, keep_default_na=False)
    required = set(required) | {"code", "known_at", "source_url"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing required evidence fields {sorted(missing)}")
    if not data.code.str.fullmatch(r"\d{6}").all():
        raise ValueError(f"{path}: invalid share codes")
    if data.source_url.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: evidence must name its source URL")
    when = pd.to_datetime(data.known_at, errors="coerce", utc=True)
    if when.isna().any():
        raise ValueError(f"{path}: invalid known_at timestamp")
    data["known_at"] = when.dt.tz_convert(None)
    cutoff = pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tzinfo else pd.Timestamp(asof)
    return data.loc[data.known_at <= cutoff].copy()


def resolve_liquidity(codes, evidence: pd.DataFrame, asof, decision_months=6, max_age_days=7):
    """No evidence => UNKNOWN; no historical information from a future record."""
    end = pd.Timestamp(asof) + pd.DateOffset(months=int(decision_months))
    cutoff = pd.Timestamp(asof)
    columns = ["code", "subscription_status", "redemption_status", "next_exit",
               "lock_until", "total_fee_rate", "max_subscription_amount",
               "liquidity_source", "liquidity_known_at", "liquidity_status",
               "buyable_now", "executable_horizon"]
    empty = dict(subscription_status="UNKNOWN", redemption_status="UNKNOWN",
                 next_exit="", lock_until="", total_fee_rate=float("nan"),
                 max_subscription_amount=float("nan"),
                 liquidity_source="", liquidity_known_at="", liquidity_status="UNKNOWN",
                 buyable_now=False, executable_horizon=False)
    rows = []
    data = evidence.copy()
    for col in ("code","known_at","source_url","subscription_status", "redemption_status",
                "next_exit", "lock_until", "total_fee_rate", "max_subscription_amount"):
        if col not in data:
            data[col] = pd.Series(dtype=object)
    if len(data):
        data = data.loc[pd.to_datetime(data.known_at) <= cutoff]
    # Group once: a full-fund catalog cannot spend O(N^2) scanning the
    # complete evidence table separately for every share code.
    by_code = {str(code): g.sort_values("known_at")
               for code,g in data.groupby("code",sort=False)}
    for code in codes:
        code = str(code)
        result = dict(code=code, **empty)
        matching = by_code.get(code)
        if matching is None or matching.empty:
            rows.append(result)
            continue
        latest = matching.iloc[-1]

        latest_time = pd.Timestamp(latest.known_at)
        if (cutoff-latest_time).total_seconds() > max_age_days*86400:
            result.update(liquidity_status="STALE", liquidity_source=str(latest.source_url),
                          liquidity_known_at=latest_time.isoformat())
            rows.append(result)
            continue
        # Contradictory current evidence (same date, different sources) is not OPEN.
        recent = matching.loc[matching.known_at.eq(latest.known_at)]
        state_pairs = set(zip(recent.subscription_status, recent.redemption_status))
        if len(state_pairs) > 1:
            result.update(liquidity_status="CONFLICT", liquidity_known_at=latest_time.isoformat(),
                          liquidity_source=";".join(sorted(set(recent.source_url.astype(str)))))
            rows.append(result)
            continue
        sub, redeem = str(latest.subscription_status).strip().upper(), str(latest.redemption_status).strip().upper()
        if sub not in TRADE_STATES or redeem not in TRADE_STATES:
            result.update(liquidity_status="UNKNOWN", liquidity_source=str(latest.source_url),
                          liquidity_known_at=latest_time.isoformat())
            rows.append(result)
            continue
        def parse_date(value):
            if value is None or str(value).strip() in ("", "nan", "NaT"):
                return pd.NaT
            return pd.to_datetime(value, errors="coerce")
        next_exit, lock_until = parse_date(latest.next_exit), parse_date(latest.lock_until)
        fee = pd.to_numeric(pd.Series([latest.total_fee_rate]), errors="coerce").iloc[0]
        maximum = pd.to_numeric(pd.Series([latest.max_subscription_amount]), errors="coerce").iloc[0]
        # A known OPEN status is necessary, not sufficient for an executable transaction.
        buy = sub in ("OPEN", "LIMITED")
        exits = redeem == "OPEN" and pd.notna(next_exit) and next_exit <= end
        exits = exits and (pd.isna(lock_until) or lock_until <= end)
        executable = bool(buy and exits and pd.notna(fee) and fee >= 0
                          and pd.notna(maximum) and maximum > 0)
        result.update(subscription_status=sub, redemption_status=redeem,
                      next_exit=str(next_exit.date()) if pd.notna(next_exit) else "",
                      lock_until=str(lock_until.date()) if pd.notna(lock_until) else "",
                      total_fee_rate=fee, max_subscription_amount=maximum,
                      liquidity_source=str(latest.source_url),liquidity_known_at=latest_time.isoformat(),
                      liquidity_status="VERIFIED", buyable_now=bool(buy),
                      executable_horizon=executable)
        rows.append(result)
    return pd.DataFrame(rows, columns=columns)
