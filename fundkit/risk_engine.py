"""稳健收益估计与组合风险贡献工具。

仅使用调用方提供的、按可用时点对齐的日收益矩阵。这里的收缩与风险贡献
是估计工具，不是未来收益或回撤保证。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("DATA_BLOCKED: 缺少收益矩阵")
    if returns.columns.duplicated().any() or not returns.index.is_unique:
        raise ValueError("DATA_BLOCKED: 重复资产或日期")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("DATA_BLOCKED: 收益日期未排序")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("DATA_BLOCKED: 收益矩阵含缺失或非法值；禁止填零")
    if len(returns) < 20:
        raise ValueError("INSUFFICIENT_HISTORY: 风险估计至少需要20个共同观测")
    return returns.astype(float)


def estimate_robust_returns(
    returns: pd.DataFrame,
    categories: Mapping[str, str] | None = None,
    prior_shrinkage_days: int = 504,
    uncertainty_z: float = 1.0,
) -> dict[str, dict]:
    """以类别中位数/全体中位数为先验，收缩样本均值并扣除估计误差。"""
    df = _validate_returns(returns)
    if prior_shrinkage_days < 0 or uncertainty_z < 0:
        raise ValueError("收缩强度和不确定性系数不得为负")
    n = len(df)
    means = df.mean() * 252.0
    vols = df.std(ddof=1) * np.sqrt(252.0)
    global_prior = float(means.median())
    cats = dict(categories or {})
    out: dict[str, dict] = {}
    shrink = n / (n + float(prior_shrinkage_days)) if n + prior_shrinkage_days else 1.0
    for code in df.columns:
        members = [c for c in df.columns if cats.get(c) == cats.get(code)] if cats.get(code) else []
        prior = float(means[members].median()) if len(members) >= 3 else global_prior
        sample = float(means[code])
        vol = float(vols[code]) if np.isfinite(vols[code]) else 0.0
        se = vol * np.sqrt(252.0 / n)
        shrunk = shrink * sample + (1.0 - shrink) * prior
        lcb = shrunk - uncertainty_z * se
        out[str(code)] = {
            "n": int(n),
            "sample_mean_annual": sample,
            "prior_mean_annual": prior,
            "shrinkage_weight": float(shrink),
            "annual_volatility": vol,
            "annual_mean_se": float(se),
            "shrunk_mean_annual": float(shrunk),
            "robust_lcb_annual": float(lcb),
            "category": cats.get(code),
        }
    return out


def portfolio_return_lcb(
    weights: Mapping[str, float],
    estimates: Mapping[str, Mapping],
    cash_return: float = 0.0,
) -> float:
    """按权重汇总单资产保守收益估计；缺估计资产直接阻断。"""
    total = 0.0
    weight_sum = 0.0
    for code, weight in weights.items():
        w = float(weight)
        if not np.isfinite(w) or w < -1e-10:
            raise ValueError("非法组合权重")
        if code == "cash":
            total += w * float(cash_return)
        elif w > 0:
            if code not in estimates:
                raise ValueError(f"DATA_BLOCKED: {code} 缺少稳健收益估计")
            total += w * float(estimates[code]["robust_lcb_annual"])
        weight_sum += w
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError("组合权重之和必须为1")
    return float(total)


def risk_contributions(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    shrinkage: float = 0.25,
    min_observations: int = 20,
) -> dict:
    """对齐共同日期后计算对角收缩协方差及波动率风险贡献。"""
    df = _validate_returns(returns)
    if not 0 <= shrinkage <= 1:
        raise ValueError("协方差收缩强度必须在[0,1]")
    codes = [c for c in df.columns if float(weights.get(c, 0.0)) > 0]
    if not codes:
        raise ValueError("DATA_BLOCKED: 没有正权重风险资产")
    x = df[codes].tail(756)
    if len(x) < min_observations:
        raise ValueError("INSUFFICIENT_HISTORY: 风险贡献共同观测不足")
    w = np.asarray([float(weights[c]) for c in codes], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("非法风险权重")
    cov = np.atleast_2d(np.cov(x.to_numpy(float), rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(cov))
    cov = (1.0 - shrinkage) * cov + shrinkage * diagonal
    cov = (cov + cov.T) / 2.0
    variance = float(w @ cov @ w)
    if not np.isfinite(variance) or variance <= 1e-18:
        raise ValueError("INSUFFICIENT_RISK: 组合波动率不可识别")
    sigma = float(np.sqrt(variance))
    component = w * (cov @ w) / sigma
    fractions = component / sigma
    return {
        "status": "ESTIMATED_COVARIANCE_RISK_CONTRIBUTION",
        "n_observations": int(len(x)),
        "portfolio_daily_volatility": float(sigma),
        "contributions": {
            code: {"absolute": float(component[i]), "fraction": float(fractions[i])}
            for i, code in enumerate(codes)
        },
        "limits": [
            "协方差风险贡献不等于CDaR/预期损失贡献",
            "输入必须是决策时点可见且共同日期对齐的收益",
            "估计值不构成未来风险保证",
        ],
    }
