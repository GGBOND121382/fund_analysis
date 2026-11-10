"""基于日净值序列计算区间绩效指标。

约定：日收益率由单位净值一阶差分得到；年化因子取 252 个交易日；
无风险利率 rf 为年化小数（如 0.02 表示 2%）。
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 指标中文名 -> (是否越大越好)。排序时按 metric 方向决定降序/升序。
METRICS = {
    "sharpe": "夏普比率",
    "sortino": "索提诺比率",
    "calmar": "卡玛比率",
    "ann_return": "年化收益率",
    "total_return": "区间收益率",
    "ann_vol": "年化波动率",
    "max_drawdown": "最大回撤",
}

# 默认排序方向：这些指标越小越好，其余越大越好
ASCENDING_METRICS = {"ann_vol", "max_drawdown"}


def slice_by_date(nav_df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    df = nav_df
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def adjust_nav(nav_df: pd.DataFrame, divs, splits=()) -> pd.DataFrame:
    """后复权：除息日分红加回收益、份额折算日按比例放大收益，重建净值序列。

    divs:   [(ex_date, 每份分红金额), ...]，当日收益按 (nav + div) / nav_prev 计算
    splits: [(date, 折算比例k), ...]，1 份变 k 份（净值除 k），当日收益按 nav * k / nav_prev 计算
    """
    if nav_df.empty or (not divs and not splits):
        return nav_df
    div_map = {pd.Timestamp(d): float(a) for d, a in divs}
    split_map = {pd.Timestamp(d): float(k) for d, k in splits}
    dates = nav_df["date"]
    nav = nav_df["nav"].to_numpy(dtype=float)
    rets = np.ones(len(nav))
    for i in range(1, len(nav)):
        d = div_map.get(dates.iloc[i], 0.0)
        k = split_map.get(dates.iloc[i], 1.0)
        rets[i] = (nav[i] + d) * k / nav[i - 1]
    out = nav_df.copy()
    out["nav"] = nav[0] * np.cumprod(rets)
    return out
    df = nav_df
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def compute_metrics(nav_df: pd.DataFrame, rf: float = 0.02, min_days: int = 60) -> dict | None:
    """对给定区间的净值序列计算全部指标。

    样本点不足 min_days 时返回 None（样本太短指标不可靠）。
    """
    nav = nav_df["nav"].astype(float).to_numpy()
    if len(nav) < max(min_days, 2):
        return None

    rets = pd.Series(nav).pct_change().dropna().to_numpy()
    n = len(rets)
    if n == 0:
        return None

    total_return = nav[-1] / nav[0] - 1.0
    ann_return = (1.0 + total_return) ** (TRADING_DAYS / n) - 1.0
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0

    rf_daily = rf / TRADING_DAYS
    excess = rets - rf_daily

    std = np.std(rets, ddof=1)
    sharpe = (np.mean(excess) / std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan

    downside = rets[rets < rf_daily] - rf_daily
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = (np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS)) if downside_std > 0 else np.nan

    cummax = np.maximum.accumulate(nav)
    drawdown = 1.0 - nav / cummax
    max_dd = float(np.max(drawdown))
    calmar = (ann_return / max_dd) if max_dd > 0 else np.nan

    gaps = nav_df["date"].diff().dt.days.dropna()
    avg_gap = float(gaps.mean()) if len(gaps) else np.nan
    max_abs_daily = float(np.max(np.abs(rets))) if n > 0 else np.nan

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": max_dd,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "sortino": float(sortino) if sortino == sortino else np.nan,
        "calmar": float(calmar) if calmar == calmar else np.nan,
        "days": n,
        "avg_gap": avg_gap,
        "max_abs_daily": max_abs_daily,
        "start": nav_df["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": nav_df["date"].iloc[-1].strftime("%Y-%m-%d"),
    }
