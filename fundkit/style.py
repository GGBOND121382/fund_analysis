"""第2步：债基风格识别与风险估计（02_STYLE_MODEL.md 实现）。

三部分：
1. 披露证据：持仓状态向量时间轴 + 候选断点（股票/转债仓位连续归零等）
2. 收益证据：滚动 EWMA 加权岭回归（含平滑项），输出 beta 路径/R²/条件数/残差诊断；
   样本不足输出 FACTOR_UNIDENTIFIED；双证据合并状态机
   （CONFIRMED_SHIFT / POSSIBLE_SHIFT / STYLE_UNCERTAIN）
3. 三套风险视图：AS_TRADED_HISTORY / CURRENT_STYLE_REWEIGHTED（反事实，不得冒充实绩）
   / CURRENT_STYLE_REALIZED

因子集（可辨识优先）：-Δy3Y, -Δy10Y, -Δy30Y, r_eq(沪深300价格指数), r_gold(上海金)。
信用利差/二级资本债利差无可靠数据源，**留 MISSING 不伪造**；转债因子缺失时
转债敞口风险由披露端 lower_bound 表达。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from fundkit.adapters import CST, LocalDataAdapter

FACTOR_COLS = ["rate_3y", "rate_10y", "eq", "gold"]  # 回归因子集（可辨识优先）
# rate_30y 保留在因子表中供压力情景使用，但数据源每月仅约5条，不进回归（02 C：不可辨识则降维）

STATE_LABELS = ["EQUITY_CB_ENHANCED", "RATE_TRADING", "CREDIT_BANK_CAPITAL", "UNKNOWN"]

DEFAULTS = {
    "lookback": 126,          # 滚动窗口（交易日）
    "ewma_halflife": 60,      # EWMA 半衰期
    "ridge_lambda": 1.0,      # 标准化岭惩罚
    "smooth_lambda": 0.1,     # 相邻 beta 平滑惩罚
    "min_effective_samples": 60,
    "shift_confirm_reports": 2,  # 连续 N 期披露确认断点
    "cond_warn": 30.0,        # 设计矩阵条件数告警
}


# -- 因子与收益 -------------------------------------------------------------

def build_factor_frame(adapter: LocalDataAdapter, until) -> pd.DataFrame:
    """日频因子表：收益率差分（小数）与指数日收益。"""
    f = adapter.risk_factors(until)
    if f.empty:
        return f
    f = f.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": f["date"]})
    for k, c in (("rate_3y", "gov_yield_3y"), ("rate_10y", "gov_yield_10y"),
                 ("rate_30y", "gov_yield_30y")):
        out[k] = -f[c].diff()  # 利率下行 -> 正收益（久期敞口方向）
    for k, c in (("eq", "csi300_price_index"), ("gold", "cny_gold_price")):
        out[k] = f[c].pct_change(fill_method=None)  # 不用前值填充伪造 0 收益
    return out


def fund_returns(adapter: LocalDataAdapter, code: str, until) -> pd.DataFrame:
    """基金总回报日收益（复权净值一阶差分），as-of 过滤后。"""
    nav = adapter.nav([code], until)
    if nav.empty:
        return pd.DataFrame(columns=["date", "ret"])
    s = nav.sort_values("nav_date")
    return pd.DataFrame({"date": s["nav_date"].values,
                         "ret": s["adj_nav_total_return"].pct_change().values}).dropna()


# -- 滚动加权岭回归 ---------------------------------------------------------

def rolling_regression(returns: np.ndarray, factors: np.ndarray,
                       lookback=126, halflife=60, ridge=1.0, smooth=0.1,
                       min_samples=60) -> pd.DataFrame:
    """逐日拟合 r = alpha + beta'f，EWMA 加权 + ridge + 相邻平滑。

    returns: (T,), factors: (T, K)。返回逐日行：alpha, beta_k, r2, cond, n_eff。
    有效样本不足时该行 beta 为 NaN（FACTOR_UNIDENTIFIED 由调用方判定）。
    """
    T, K = factors.shape
    # 因子标准化（全样本矩仅用于尺度，方向由滚动拟合决定）
    f_mean = np.nanmean(factors, axis=0)
    f_std = np.nanstd(factors, axis=0)
    f_std[f_std == 0] = 1.0
    F = (factors - f_mean) / f_std
    rows = []
    prev_beta = np.zeros(K)
    for t in range(T):
        lo = max(0, t - lookback + 1)
        idx = slice(lo, t + 1)
        y, X = returns[idx], F[idx]
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        y, X = y[valid], X[valid]
        if len(y) < min_samples:
            rows.append({"alpha": np.nan, "r2": np.nan, "cond": np.nan, "n_eff": len(y),
                         **{f"beta_{c}": np.nan for c in range(K)}})
            continue
        w = 0.5 ** ((t - (lo + np.where(valid)[0])) / halflife)
        W = w / w.sum()
        Xa = np.column_stack([np.ones(len(y)), X])
        A = (Xa * W[:, None]).T @ Xa
        P = np.diag([0.0] + [ridge] * K) + np.diag([0.0] + [smooth] * K)
        b_smooth = np.concatenate([[0.0], smooth * prev_beta])
        try:
            coef = np.linalg.solve(A + P, (Xa * W[:, None]).T @ y + b_smooth)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A + P, (Xa * W[:, None]).T @ y + b_smooth,
                                   rcond=None)[0]
        resid = y - Xa @ coef
        tss = float(np.sum(W * (y - np.average(y, weights=W)) ** 2))
        r2 = 1 - float(np.sum(W * resid ** 2)) / tss if tss > 0 else np.nan
        cond = float(np.linalg.cond(A + P))
        prev_beta = coef[1:]
        rows.append({"alpha": coef[0], "r2": r2, "cond": cond, "n_eff": len(y),
                     **{f"beta_{k}": b for k, b in enumerate(coef[1:])}})
    return pd.DataFrame(rows)


# -- 披露端断点与状态 -------------------------------------------------------

def disclosure_timeline(adapter: LocalDataAdapter, code: str, until) -> pd.DataFrame:
    """持仓状态向量时间轴（含 published_at）。"""
    d = adapter.disclosures([code], until)
    if d.empty:
        return d
    return d.sort_values("report_asof").reset_index(drop=True)


def disclosure_breakpoints(timeline: pd.DataFrame, confirm_reports=2) -> list[dict]:
    """披露证据候选断点：股票/转债从有仓位转为连续 N 期（未披露或≤0.5%）。

    断点确认时点 = 第 N 期报告的 published_at（不用未来信息）。
    """
    out = []
    if timeline.empty:
        return out
    eq = timeline["equity_nav_pct"].fillna(0.0)
    cb = timeline["convertible_nav_pct_lower_bound"].fillna(0.0)
    risky = (eq + cb) > 0.5
    for i in range(confirm_reports, len(timeline)):
        window = risky.iloc[i - confirm_reports + 1: i + 1]
        # 只在"前一报告期仍有风险仓位、随后连续 N 期归零"的转折点报告一次
        if not window.any() and bool(risky.iloc[i - confirm_reports]):
            prev = timeline.iloc[i - confirm_reports]
            cur = timeline.iloc[i]
            out.append({
                "type": "EQUITY_CB_EXIT",
                "confirmed_at": str(cur["published_at"].date()),
                "evidence": (f"股票+转债占净值比 {prev['report_asof']} 期 "
                             f"{(prev['equity_nav_pct'] or 0) + (prev['convertible_nav_pct_lower_bound'] or 0):.1f}% "
                             f"→ 连续 {confirm_reports} 期 ≤0.5%（截至 {cur['report_asof']}）"),
            })
    return out


def classify_state(timeline_row: pd.Series | None, betas: dict | None,
                   r2: float | None) -> tuple[str, str]:
    """三状态 + UNKNOWN 分类标签（不输出伪精确概率）。

    返回 (state, reason)。规则保守：证据不足一律 UNKNOWN/STYLE_UNCERTAIN。
    """
    if timeline_row is None:
        return "UNKNOWN", "无披露数据"
    eq = timeline_row.get("equity_nav_pct")
    cb = timeline_row.get("convertible_nav_pct_lower_bound") or 0
    tier2 = timeline_row.get("bank_tier2_nav_pct_lower_bound") or 0
    perp = timeline_row.get("bank_perpetual_nav_pct_lower_bound") or 0
    credit = timeline_row.get("credit_bond_nav_pct_lower_bound") or 0
    if (eq or 0) + cb > 3:
        return "EQUITY_CB_ENHANCED", f"股票{eq or 0:.1f}%+转债{cb:.1f}%"
    if tier2 + perp + credit > 10:
        return "CREDIT_BANK_CAPITAL", f"信用{tier2 + perp + credit:.1f}%（重仓下限）"
    if betas and r2 is not None and r2 > 0.3:
        b10 = betas.get("rate_10y", 0) or 0
        b30 = betas.get("rate_30y", 0) or 0
        if abs(b10) + abs(b30) > 0.3:
            return "RATE_TRADING", f"利率beta显著（10Y {b10:.2f}, 30Y {b30:.2f}），R²={r2:.2f}"
    return "UNKNOWN", "披露与回归证据均不足，按 STYLE_UNCERTAIN 处理"


# -- 双证据合并状态机 ---------------------------------------------------------

def combined_style_state(disc_shifts: list[dict], beta_path: pd.DataFrame,
                         lookback: int) -> tuple[str, str]:
    """披露与收益证据合并：CONFIRMED_SHIFT / POSSIBLE_SHIFT / NO_SHIFT。

    Z 统计量：beta_t 与 beta_{t-L} 的标准化差异（对角协方差近似，不可逆时降维标记）。
    """
    if beta_path.empty or "beta_0" not in beta_path:
        return ("CONFIRMED_SHIFT" if disc_shifts else "NO_SHIFT",
                "无收益回归证据" if not disc_shifts else "仅披露证据")
    b = beta_path[[c for c in beta_path.columns if c.startswith("beta_")]].to_numpy(float)
    valid = ~np.isnan(b).any(axis=1)
    if valid.sum() < lookback + 10:
        return ("POSSIBLE_SHIFT" if disc_shifts else "NO_SHIFT", "回归有效样本不足")
    bv = b[valid]
    z_series = np.full(len(bv), np.nan)
    for t in range(lookback, len(bv)):
        diff = bv[t] - bv[t - lookback]
        ref = bv[max(0, t - 2 * lookback): t - lookback]
        if len(ref) < 20:
            continue
        sd = ref.std(axis=0)
        sd[sd == 0] = np.nan
        z_series[t] = float(np.nansum((diff / sd) ** 2))
    z_now = z_series[-1] if len(z_series) else np.nan
    if not np.isfinite(z_now):
        return ("POSSIBLE_SHIFT" if disc_shifts else "NO_SHIFT", "Z 统计量不可计算（协方差降维）")
    # 阈值用样本内分布校准（训练期分位数），不硬编码卡方临界值
    calib = z_series[: max(len(z_series) // 2, 30)]
    calib = calib[np.isfinite(calib)]
    thr = float(np.quantile(calib, 0.99)) if len(calib) >= 30 else np.inf
    return_drift = z_now > thr
    if disc_shifts and return_drift:
        return "CONFIRMED_SHIFT", f"披露断点+收益漂移一致（Z={z_now:.1f} > 校准阈值{thr:.1f}）"
    if disc_shifts:
        return "POSSIBLE_SHIFT", f"披露断点存在，收益漂移未确认（Z={z_now:.1f}≤{thr:.1f}）"
    if return_drift:
        return "POSSIBLE_SHIFT", f"收益漂移（Z={z_now:.1f}>{thr:.1f}）但无披露佐证"
    return "NO_SHIFT", f"Z={z_now:.1f}≤{thr:.1f}，无披露断点"


# -- 三套风险视图 -------------------------------------------------------------

def risk_views(adapter: LocalDataAdapter, code: str, until, factor_df: pd.DataFrame,
               reg: pd.DataFrame, shift_date: str | None) -> dict:
    """AS_TRADED_HISTORY / CURRENT_STYLE_REWEIGHTED / CURRENT_STYLE_REALIZED 的 MDD/CDaR95。"""
    from tests import finance_math as fm

    rets = fund_returns(adapter, code, until)
    out = {}
    if rets.empty:
        return {"error": "no_returns"}

    def stats(r: np.ndarray) -> dict:
        if len(r) < 20:
            return {"n": len(r), "mdd": None, "cdar95": None}
        nav = np.cumprod(1 + r)
        dd = fm.drawdown_path(nav.tolist())
        return {"n": len(r), "mdd": fm.max_drawdown(nav.tolist()),
                "cdar95": fm.empirical_cdar(dd, 0.95)}

    out["AS_TRADED_HISTORY"] = stats(rets["ret"].to_numpy())
    if shift_date:
        realized = rets[rets["date"] >= pd.Timestamp(shift_date)]
        out["CURRENT_STYLE_REALIZED"] = {
            **stats(realized["ret"].to_numpy()), "since": shift_date,
            "note": "风格确认后公开净值样本，样本短、证据弱"}
    # 反事实：当前 beta × 历史因子冲击 + 残差自助
    # 02 E：残差不得沿用旧股票时代的厚尾分布，取风格确认后（或最近一年）样本
    if not reg.empty and not factor_df.empty and len(factor_df) > 60:
        last = reg.iloc[-1]
        beta_cols = [c for c in reg.columns if c.startswith("beta_")]
        if not last[beta_cols].isna().any():
            beta = last[beta_cols].to_numpy(float)
            alpha = float(last["alpha"]) * 0.5  # alpha 收缩 0.5（候选超参）
            F = factor_df[FACTOR_COLS].to_numpy(float)
            ok = ~np.isnan(F).any(axis=1)
            F = F[ok]
            resid_pool = rets
            if shift_date:
                resid_pool = rets[rets["date"] >= pd.Timestamp(shift_date)]
            elif len(rets) > 252:
                resid_pool = rets.tail(252)
            resid = resid_pool["ret"].to_numpy()
            resid = resid[np.isfinite(resid)]
            rng = np.random.default_rng(42)
            eps = rng.choice(resid, size=len(F), replace=True) if len(resid) > 30 else 0.0
            cf = alpha + F @ beta + eps
            out["CURRENT_STYLE_REWEIGHTED"] = {
                **stats(cf),
                "note": "COUNTERFACTUAL_STRESS：当前敞口×历史因子冲击+近期残差自助，非真实历史业绩"}
    return out


# -- 主流程 -------------------------------------------------------------------

def analyze_fund(adapter: LocalDataAdapter, code: str, until, params=None) -> dict:
    """单基金完整风格分析（02 G 节结构化输出）。"""
    p = {**DEFAULTS, **(params or {})}
    timeline = disclosure_timeline(adapter, code, until)
    shifts = disclosure_breakpoints(timeline, p["shift_confirm_reports"])
    factor_df = build_factor_frame(adapter, until)
    rets = fund_returns(adapter, code, until)
    reg = pd.DataFrame()
    if not rets.empty and not factor_df.empty:
        merged = pd.merge(rets, factor_df, on="date", how="inner")
        if len(merged) >= p["min_effective_samples"]:
            reg = rolling_regression(
                merged["ret"].to_numpy(), merged[FACTOR_COLS].to_numpy(),
                p["lookback"], p["ewma_halflife"], p["ridge_lambda"],
                p["smooth_lambda"], p["min_effective_samples"])
            reg.insert(0, "date", merged["date"].values)
    state_evidence = combined_style_state(shifts, reg, p["lookback"])
    last_row = timeline.iloc[-1] if not timeline.empty else None
    last_beta = None
    r2 = None
    if not reg.empty:
        last = reg.iloc[-1]
        beta_cols = [c for c in reg.columns if c.startswith("beta_")]
        if not last[beta_cols].isna().all():
            last_beta = {FACTOR_COLS[i]: float(last[f"beta_{i}"]) for i in range(len(FACTOR_COLS))}
            r2 = float(last["r2"]) if pd.notna(last["r2"]) else None
    state, state_reason = classify_state(last_row, last_beta, r2)
    shift_date = shifts[-1]["confirmed_at"] if shifts else None
    views = risk_views(adapter, code, until, factor_df, reg, shift_date)

    result = {
        "fund_code": code,
        "disclosure_periods": int(len(timeline)),
        "disclosure_shifts": shifts,
        "combined_shift_evidence": {"state": state_evidence[0], "reason": state_evidence[1]},
        "current_style_state": state,
        "style_state_reason": state_reason,
        "style_confidence": None,  # 不输出伪精确概率（02 F）
        "latest_betas": last_beta,
        "latest_r2": r2,
        "regression_cond_latest": float(reg["cond"].iloc[-1]) if not reg.empty else None,
        "factor_unidentified": bool(reg.empty or last_beta is None),
        "risk_views": views,
        "caveats": [
            "alpha 与 carry 分离：不将风格确认后收益自动外推（02 G）",
            "信用利差/二级资本债利差因子 MISSING：信用敞口以披露 lower_bound 表达",
            "沪深300 为价格指数（非全收益），权益 beta 略偏低估",
        ],
    }
    return result


def run_style_report(codes, until=None, out_dir="reports") -> dict:
    adapter = LocalDataAdapter()
    until = until or pd.Timestamp.now().tz_localize(CST)
    report = {c: analyze_fund(adapter, c, until) for c in codes}
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "style_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return report
