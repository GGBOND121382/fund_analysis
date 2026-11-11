"""As-of conservative return screen for tactical additions (not an OOS LCB).

Training-only historical joint-block resampling is used for a low-parameter
sensitivity screen. No equilibrium premium, Black-Litterman view or expected
fund profit is invented. An additional strict scenario risk gate is required.
"""
import numpy as np
import pandas as pd


def conservative_excess_return(rows, decision_date, *, annual_cash_return=0.,
                               min_observations=252, paths=64, seed=42):
    if not rows:
        return {'status':'NO_FORECAST','reason':'MISSING_PUBLISHED_NAV'}
    f=pd.DataFrame(rows).sort_values('date')
    if not {'date','total_return_nav'}.issubset(f.columns):
        return {'status':'NO_FORECAST','reason':'MISSING_REQUIRED_FIELDS'}
    dates=pd.to_datetime(f['date'])
    if (dates > pd.Timestamp(decision_date)).any():
        raise ValueError('LOOKAHEAD_BLOCKED: future NAV in return forecast')
    x=pd.to_numeric(f['total_return_nav'],errors='coerce').to_numpy(float)
    if not np.isfinite(x).all() or (x<=0).any():
        return {'status':'NO_FORECAST','reason':'BAD_PUBLISHED_NAV'}
    r=x[1:]/x[:-1]-1.
    if len(r)<min_observations:
        return {'status':'NO_FORECAST','reason':'INSUFFICIENT_HISTORY'}
    # Historical distribution is NOT a calibrated confidence interval.
    rng=np.random.default_rng(seed)
    outcomes=[]
    horizon=252
    for block in (20,60):
        if len(r)<block:
            return {'status':'NO_FORECAST','reason':'BLOCK_HISTORY_MISSING'}
        for _ in range(paths):
            starts=rng.integers(0,len(r)-block+1,size=int(np.ceil(horizon/block)))
            idx=np.concatenate([np.arange(s,s+block) for s in starts])[:horizon]
            outcomes.append(float(np.prod(1+r[idx])-1))
    q05=float(np.quantile(outcomes,.05))
    return {'status':'ESTIMATED_RESEARCH_ONLY',
            'annual_return_q05':q05,
            'excess_q05':q05-float(annual_cash_return),
            'quality':'SAME_HISTORY_BOOTSTRAP_NOT_OOS',
            'historical_days':len(r)}
