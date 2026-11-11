"""Low-parameter AS-OF timing signals and independently assessed scenario gate.

No fill-forward of missing fund returns. Signals never use unpublished T-day NAV.
A signal is NOT a prediction guarantee; use for research ablations only.
"""
import numpy as np
import pandas as pd


def published_returns(state, min_days=252):
    data = {}
    day = pd.Timestamp(state['date']).tz_localize(None)
    for code, rows in (state.get('fund_nav_history') or {}).items():
        if not rows:
            continue
        f = pd.DataFrame(rows)
        if not {'date', 'total_return_nav'}.issubset(f):
            continue
        f['date'] = pd.to_datetime(f['date']).dt.tz_localize(None)
        if (f['date'] > day).any():
            raise ValueError('LOOKAHEAD_BLOCKED: future published NAV in state')
        f = f.drop_duplicates('date').sort_values('date')
        x = pd.to_numeric(f.total_return_nav, errors='coerce')
        if x.isna().any() or (x <= 0).any():
            continue
        s = pd.Series(x.to_numpy(float), index=pd.DatetimeIndex(f.date), name=code)
        data[code] = s.pct_change(fill_method=None).dropna()
    if not data:
        return pd.DataFrame()
    # No cross-asset filling. Missing dates are intentionally excluded.
    out = pd.concat(data.values(), axis=1, join='inner').dropna()
    if len(out) < min_days:
        return pd.DataFrame()
    return out


def score_entries(state, categories, *, min_observations=253, annual_cash_return=0.):
    out = {}
    for code, rows in (state.get('fund_nav_history') or {}).items():
        category = categories.get(code, 'other')
        if category not in ('gold', 'equity_index'):
            out[code] = {'score': 0., 'eligible': False, 'quality': 'BOND_NO_UNVERIFIED_TREND'}
            continue
        if not rows:
            out[code] = {'score': 0., 'eligible': False, 'quality': 'MISSING_HISTORY'}
            continue
        f = pd.DataFrame(rows).sort_values('date')
        if len(f) < min_observations:
            out[code] = {'score': 0., 'eligible': False, 'quality': 'SHORT_HISTORY'}
            continue
        nav = pd.to_numeric(f['total_return_nav'], errors='coerce').to_numpy(float)
        if (not np.isfinite(nav).all()) or np.any(nav <= 0):
            out[code] = {'score': 0., 'eligible': False, 'quality': 'BAD_HISTORY'}
            continue
        dates = pd.to_datetime(f['date'])
        if (dates > pd.Timestamp(state['date'])).any():
            raise ValueError('LOOKAHEAD_BLOCKED: unpublished data in entry signal')
        tr126 = nav[-1] / nav[-127] - 1
        tr252 = nav[-1] / nav[-253] - 1
        score = 0.5 * tr126 + 0.5 * tr252 - annual_cash_return * 0.75
        out[code] = {'score': float(score), 'eligible': bool(score > 0),
                     'quality': 'PUBLISHED_TREND_126_252', 'tr126': float(tr126),
                     'tr252': float(tr252)}
    return out


def scenario_risk_gate(state, proposed_buys=None, proposed_redemptions=None,
                       *, budget=.04, n_paths=96, horizon=252, seed=42):
    """Joint moving block bootstrap; conservative tactical gate only.

    Requires every nonzero position's series. Not an independent OOS check.
    Redemption values use published NAV and cannot make pending cash spendable.
    """
    if state.get('value_quality') != 'COMPLETE':
        return {'status': 'RISK_UNKNOWN', 'reason': 'MISSING_MARK'}
    total = float(state.get('value', 0.))
    if total <= 0:
        return {'status': 'RISK_UNKNOWN', 'reason': 'MISSING_VALUE'}
    holdings = state.get('holdings') or {}
    held = {k: float(h.get('value_pub') or 0) for k, h in holdings.items()
            if float(h.get('value_pub') or 0) > 0}
    proposed_buys = proposed_buys or {}
    proposed_redemptions = proposed_redemptions or {}
    exposure = dict(held)
    for code, amt in proposed_buys.items():
        exposure[code] = exposure.get(code, 0) + max(0., float(amt))
    for code, shares in proposed_redemptions.items():
        h = holdings.get(code) or {}
        units = float(h.get('shares') or 0.)
        value = float(h.get('value_pub') or 0.)
        if units > 0:
            exposure[code] = max(0., exposure.get(code, 0) - float(shares) * value / units)
    matrix = published_returns(state, min_days=horizon)
    absent = [code for code, amt in exposure.items() if amt > 1e-8 and code not in matrix.columns]
    if absent or len(matrix) < horizon:
        return {'status': 'RISK_UNKNOWN', 'reason': 'JOINT_HISTORY_INCOMPLETE',
                'missing': absent, 'joint_days': len(matrix)}
    codes = [c for c in matrix.columns if exposure.get(c, 0) > 1e-8]
    if not codes:
        return {'status': 'PASS', 'mdd_p95': 0., 'historical_mdd': 0.}
    R = matrix[codes].to_numpy(float)
    weights = np.array([exposure[c] / total for c in codes])
    if (weights.sum() > 1 + 1e-6):
        return {'status': 'RISK_UNKNOWN', 'reason': 'POST_TRADE_VALUE_NOT_CONSERVED'}
    r = R @ weights
    nav = np.cumprod(1 + r)
    hdd = float(np.max(1 - nav / np.maximum(1., np.maximum.accumulate(nav))))
    rng = np.random.default_rng(seed)
    paths = []
    for block in (20, 60):
        if len(R) < block:
            return {'status': 'RISK_UNKNOWN', 'reason': 'HISTORY_SHORTER_THAN_BLOCK'}
        for _ in range(n_paths):
            starts = rng.integers(0, len(R)-block+1, size=int(np.ceil(horizon / block)))
            idx = np.concatenate([np.arange(s,s+block) for s in starts])[:horizon]
            rr = R[idx] @ weights
            nav = np.cumprod(1 + rr)
            paths.append(float(np.max(1 - nav / np.maximum(1., np.maximum.accumulate(nav)))))
    q = float(np.quantile(paths,.95))
    return {'status': 'PASS' if max(hdd,q) <= budget else 'FAIL',
            'historical_mdd': hdd,'mdd_p95': q, 'source':'BOOTSTRAP_NOT_OOS'}
