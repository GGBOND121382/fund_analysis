"""One-year, as-of predictive model for fund allocation (research only).

Uses at most 252 mutually observed trading-day returns. A blocked *past-only*
rolling evaluation selects the shrinkage of sample means toward peer medians.
The last 63/126 sessions are reported as regime diagnostics and increase the
forecast-uncertainty estimate when they diverge from the one-year sample.
There is NO extrapolation of ten-day bond-market performance into annual alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _peer_prior(train, categories):
    means=252*np.asarray(train).mean(axis=0)
    n=len(means)
    out=np.empty(n)
    for i,cat in enumerate(categories):
        peer=[j for j,c in enumerate(categories) if c==cat]
        vals=means[peer] if len(peer)>=3 else means
        out[i]=float(np.median(vals))
    return out


def _shrink_covariance(x):
    """Diagonal Ledoit-Wolf-style shrink with intensity estimated from data."""
    x=np.asarray(x,float)
    n,d=x.shape
    centered=x-x.mean(axis=0)
    s=centered.T@centered/n
    target=np.diag(np.diag(s))
    deviation=np.sum((s-target)**2)
    if deviation < 1e-25:
        rate=1.
    else:
        # E ||S-Sigma||_F^2 estimated from within-sample fourth moments.
        phi=np.mean(np.sum(
            ((centered[:,:,None]*centered[:,None,:])-s)**2,
            axis=(1,2)))
        rate=float(np.clip(phi/(n*deviation),0.,1.))
    out=(1-rate)*s+rate*target
    return (out+out.T)/2.,rate


def _mean_cv_weights(R, cats, candidate_shrinkages=(0.,.25,.5,.75,1.)):
    """Chronological blocked CV; never score a prediction on its input data."""
    n,d=R.shape
    # Four or more fixed-origin predictions, evaluated over *future* 21-day blocks.
    folds=[k for k in range(126,n-20,21)]
    if len(folds)<2:
        return .5, {"status":"INSUFFICIENT_CV_FOLDS","n_folds":len(folds)}
    loss=np.zeros(len(candidate_shrinkages))
    for k in folds:
        train=R[:k]
        test=R[k:k+21]
        peer=_peer_prior(train,cats)
        sample=train.mean(axis=0)*252.
        truth=test.mean(axis=0)*252.
        vol=np.std(train,axis=0,ddof=1)*np.sqrt(252)
        scale=np.maximum(vol,.02)
        for j,a in enumerate(candidate_shrinkages):
            prediction=a*sample+(1-a)*peer
            loss[j]+=np.mean(((prediction-truth)/scale)**2)
    best=int(np.argmin(loss))
    return float(candidate_shrinkages[best]),{
        "status":"BLOCKED_PAST_ONLY_CV","n_folds":len(folds),
        "candidate_shrinkages":list(candidate_shrinkages),
        "relative_validation_mse":list((loss/len(folds)).astype(float)),
    }


def fit_one_year_model(returns, categories=None, *, lookback=252,
                       bootstrap_draws=64, block_days=10, seed=42):
    """Estimate model at the last published return date, no future observations.

    Returns annualized mean forecasts and PSD covariance/error covariance in
    the fund column order. Error covariance is a finite-sample uncertainty
    estimate, not a guaranteed confidence region.
    """
    if not isinstance(returns,pd.DataFrame) or returns.empty:
        raise ValueError("DATA_BLOCKED: expected nonempty return DataFrame")
    if returns.columns.duplicated().any() or not returns.index.is_monotonic_increasing:
        raise ValueError("DATA_BLOCKED: duplicate asset or unsorted returns")
    if lookback<126 or bootstrap_draws<8 or block_days<1:
        raise ValueError("DATA_BLOCKED: invalid one-year model parameters")
    df=returns.tail(lookback).copy()
    if len(df)<min(lookback,189):
        raise ValueError("DATA_BLOCKED: insufficient common published NAV")
    R=df.to_numpy(dtype=float)
    if not np.isfinite(R).all() or (R<=-1).any():
        raise ValueError("DATA_BLOCKED: invalid return observations")
    codes=[str(c) for c in df.columns]
    cats=[(categories or {}).get(c,"other") for c in codes]
    alpha,cv=_mean_cv_weights(R,cats)
    sample=252*R.mean(axis=0)
    prior=_peer_prior(R,cats)
    mu=alpha*sample+(1-alpha)*prior
    daily_cov,cov_shrink=_shrink_covariance(R)
    cov=252*daily_cov
    rng=np.random.default_rng(seed)
    n,d=R.shape
    m=int(np.ceil(n/block_days))
    estimates=[]
    for _ in range(bootstrap_draws):
        starts=rng.integers(0,n-block_days+1,size=m)
        idx=np.concatenate([np.arange(s,s+block_days) for s in starts])[:n]
        re=R[idx]
        estimates.append(alpha*(252*re.mean(axis=0))+
                         (1-alpha)*_peer_prior(re,cats))
    samples=np.array(estimates)
    uncertainty=np.atleast_2d(np.cov(samples,rowvar=False))
    recent63=252*R[-min(63,n):].mean(axis=0)
    recent126=252*R[-min(126,n):].mean(axis=0)
    # Uncertain regime shifts widen the *error bar*, not the mean forecast.
    drift=(recent63-recent126)
    uncertainty+=.1*np.outer(drift,drift)
    uncertainty=(uncertainty+uncertainty.T)/2.
    e,v=np.linalg.eigh(uncertainty)
    uncertainty=(v*np.maximum(e,0.))@v.T
    lcb=mu-np.sqrt(np.maximum(np.diag(uncertainty),0.))
    return {
        "asset_codes":codes,
        "asof_return_date":str(df.index[-1].date()) if hasattr(df.index[-1],"date") else str(df.index[-1]),
        "n_common_observations":n,
        "window_max_trading_days":lookback,
        "mean_annual":mu,
        "lower_annual":lcb,
        "return_cov_annual":cov,
        "forecast_error_cov_annual":uncertainty,
        "mean_shrinkage_weight":alpha,
        "mean_shrinkage_validation":cv,
        "covariance_shrinkage_weight":cov_shrink,
        "diagnostics":{
            c:{"sample_mean_annual":float(sample[i]),
               "peer_prior_annual":float(prior[i]),
               "forecast_annual":float(mu[i]),
               "lower_proxy_annual":float(lcb[i]),
               "recent_63d_mean_annual":float(recent63[i]),
               "recent_126d_mean_annual":float(recent126[i]),
               "forecast_error_sd_annual":float(np.sqrt(max(0.,uncertainty[i,i]))),
               "category":cats[i]}
            for i,c in enumerate(codes)},
        "limitations":[
            "63/126-day observations only widen uncertainty; no momentum extrapolation",
            "forecast error covariance is estimated from same-year blocks, not calibrated coverage",
            "fund NAV publication, dividends, vintage and exposure-change require independent audit",
        ],
    }
