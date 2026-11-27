"""Fixed-weight ablation helpers for M4-M7.

M5 uses an internally constructed, leave-one-product-out category factor,
with unit exposure, not a fitted external asset-pricing model. All inputs
must be available at the formation date. Its lower score is a conservative
research statistic, not a calibrated predictive quantile.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from fundkit.defensive_screen import share_identity

Z = .67448975
MIN_PRODUCTS = 10
CURRENCY_SUFFIX = re.compile(
    r"(?:[（(]?(?:人民币|美元|港元|港币|欧元|CNY|RMB|USD|HKD|EUR)(?:现汇|现钞|份额)?[）)]?)$",
    re.I)
FOREIGN_GROUP = re.compile(r"\s·\s(?:USD|HKD|EUR|JPY|GBP|AUD|CAD|SGD)$", re.I)


def canonical_product_name(name):
    """Normalize explicit terminal currency and share markers only."""
    s = str(name).strip()
    s, _ = share_identity(s)
    s = CURRENCY_SUFFIX.sub("", s).strip()
    s, _ = share_identity(s)
    return s.strip()


def identity_keys(d):
    """Prefer vendor IDs; repair name-based identities conservatively."""
    names = d.name.map(canonical_product_name)
    old = d.product_key.fillna("").astype(str)
    return old.where(old.str.startswith("id:"), "name:" + names).where(
        names.ne("") | old.str.startswith("id:"), "share:" + d.code.astype(str))


def cny_comparable(d):
    """Fail closed on known non-CNY denominations; no FX imputation."""
    base = d.name.map(lambda n: share_identity(str(n).strip())[0])
    suffix = base.str.extract(CURRENCY_SUFFIX, expand=False) if CURRENCY_SUFFIX.groups else base.map(
        lambda n: CURRENCY_SUFFIX.search(n).group(0) if CURRENCY_SUFFIX.search(n) else "")
    foreign = suffix.fillna("").str.contains(r"美元|港元|港币|欧元|USD|HKD|EUR", case=False, regex=True)
    foreign |= d.group.fillna("").str.contains(FOREIGN_GROUP, regex=True)
    if "currency" in d:
        foreign |= ~d.currency.fillna("CNY").astype(str).str.upper().isin(["CNY","RMB"])
    return ~foreign


def category_residual_features(d, monthly):
    """One vote per product/month; exclude the evaluated product itself.

    monthly contains code, month, monthly_return, only formation-time data.
    Complete-month sample per fund must match the raw mean and standard error.
    A missing factor for any observed month is not imputed.
    """
    out = []
    m = monthly.loc[monthly.code.isin(d.code), ["code","month","monthly_return"]].copy()
    if m.duplicated(["code","month"]).any():
        raise ValueError("duplicate monthly observations")
    m["x"] = np.log1p(m.monthly_return)
    m = m.loc[np.isfinite(m.x)]
    if m.empty:
        return pd.DataFrame(columns=["code","factor_mean","factor_se","residual_mean","residual_se","factor_months","factor_peer_count"])
    matrix = m.pivot(index="code",columns="month",values="x")
    for _, g in d.groupby("group",dropna=False):
        g = g.loc[g.code.isin(matrix.index)].sort_values("code")
        reps = g.drop_duplicates("product_key")
        if len(reps)<MIN_PRODUCTS:
            continue
        v = matrix.loc[g.code].to_numpy(dtype=float)
        p = matrix.loc[reps.code].to_numpy(dtype=float)
        lookup = {key:i for i,key in enumerate(reps.product_key)}
        own = p[[lookup[k] for k in g.product_key]]
        count = np.isfinite(p).sum(axis=0)[None,:]-np.isfinite(own)
        total = np.nansum(p,axis=0)[None,:]-np.nan_to_num(own,nan=0.)
        f = np.divide(total,count,out=np.full_like(total,np.nan),where=count>=MIN_PRODUCTS-1)
        for i,c in enumerate(g.code):
            mask=np.isfinite(v[i])
            n=int(mask.sum())
            if n<4 or not np.isfinite(f[i,mask]).all():
                continue
            b=f[i,mask]; e=v[i,mask]-b
            out.append(dict(code=c,factor_mean=float(12*b.mean()),
                factor_se=float(12*b.std(ddof=1)/np.sqrt(n)),
                residual_mean=float(12*e.mean()),residual_se=float(12*e.std(ddof=1)/np.sqrt(n)),
                factor_months=n,factor_peer_count=len(reps)-1))
    return pd.DataFrame(out,columns=["code","factor_mean","factor_se","residual_mean","residual_se","factor_months","factor_peer_count"])


def add_return_ablation(d, monthly, eb_fit):
    """M4: raw complete-month mean minus SE; M5: factor-residual EB.

    When positive residual dispersion is identified, add the historical
    factor mean to residual EB posterior mean. Use sum of factor SE and
    conditional residual posterior SE (Cauchy upper bound on combined SD,
    avoiding an unsupported independence assumption). Otherwise fall back
    exactly to M4. Neither score claims a future-return confidence level.
    """
    d=d.copy()
    d["return_M4"]=np.expm1(d.eb_observed_log_mean-Z*d.annual_se)
    d["return_M5"]=d.return_M4
    d["factor_status"]="FALLBACK_NO_FACTOR"
    d["factor_eb_weight"]=np.nan
    d=d.merge(category_residual_features(d,monthly),on="code",how="left",validate="one_to_one")
    valid=np.isfinite(d[["factor_mean","factor_se","residual_mean","residual_se"]]).all(axis=1)
    if "complete_months" in d:
        valid &= d.factor_months.eq(d.complete_months)
    sub=d.loc[valid,["code","group","product_key","residual_mean","residual_se"]].copy()
    if sub.empty:
        return d
    sub=sub.rename(columns={"residual_mean":"eb_observed_log_mean","residual_se":"annual_se"})
    fitted=eb_fit(sub).set_index("code")
    d.loc[valid,"factor_status"]=d.loc[valid,"code"].map(fitted.eb_status).map(
        lambda s:"FACTOR_EB" if s=="EB_CONDITIONAL" else "FALLBACK_"+str(s))
    use=d.factor_status.eq("FACTOR_EB")
    d.loc[valid,"factor_eb_weight"]=d.loc[valid,"code"].map(fitted.eb_weight)
    center=d.loc[use,"factor_mean"]+d.loc[use,"code"].map(fitted.eb_log_cagr)
    uncertainty=d.loc[use,"factor_se"]+d.loc[use,"code"].map(fitted.eb_se)
    d.loc[use,"return_M5"]=np.expm1(center-Z*uncertainty)
    return d
