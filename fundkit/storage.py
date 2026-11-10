"""本地 CSV 存储：基金列表 + 每只基金的日净值文件（date, nav[, acc_nav]）。"""

import os
from datetime import datetime, timedelta

import pandas as pd

DATA_DIR = "data"
NAV_DIR = os.path.join(DATA_DIR, "nav")
FUND_LIST_FILE = os.path.join(DATA_DIR, "fund_list.csv")
FUND_STATUS_FILE = os.path.join(DATA_DIR, "fund_status.csv")
DIVIDENDS_FILE = os.path.join(DATA_DIR, "dividends.csv")


def _ensure_dirs():
    os.makedirs(NAV_DIR, exist_ok=True)


def save_fund_list(df: pd.DataFrame):
    _ensure_dirs()
    df.to_csv(FUND_LIST_FILE, index=False, encoding="utf-8-sig")


def load_fund_list() -> pd.DataFrame:
    if not os.path.exists(FUND_LIST_FILE):
        raise FileNotFoundError(f"未找到 {FUND_LIST_FILE}，请先运行: python main.py list")
    df = pd.read_csv(FUND_LIST_FILE, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df


def save_trade_status(df: pd.DataFrame):
    _ensure_dirs()
    df.to_csv(FUND_STATUS_FILE, index=False, encoding="utf-8-sig")


def load_trade_status() -> pd.DataFrame:
    if not os.path.exists(FUND_STATUS_FILE):
        raise FileNotFoundError(f"未找到 {FUND_STATUS_FILE}，请先运行: python main.py list")
    df = pd.read_csv(FUND_STATUS_FILE, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df


def save_dividends(div_map: dict):
    """保存 {code: [(ex_date_str, 每份分红), ...]} 到 CSV。"""
    _ensure_dirs()
    rows = [(c, d, a) for c, divs in div_map.items() for d, a in divs]
    df = pd.DataFrame(rows, columns=["code", "ex_date", "div"])
    df.to_csv(DIVIDENDS_FILE, index=False, float_format="%.4f")


def load_dividends() -> dict:
    """返回 {code: [(ex_date_str, div), ...]}，无文件时返回空 dict。"""
    if not os.path.exists(DIVIDENDS_FILE):
        return {}
    df = pd.read_csv(DIVIDENDS_FILE, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    out = {}
    for code, grp in df.groupby("code"):
        out[code] = list(zip(grp["ex_date"], grp["div"]))
    return out


def nav_path(code: str) -> str:
    return os.path.join(NAV_DIR, f"{code}.csv")


def save_nav(code: str, df: pd.DataFrame):
    _ensure_dirs()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    float_cols = [c for c in ("nav", "acc_nav") if c in out.columns]
    out.to_csv(nav_path(code), index=False,
               columns=["date"] + float_cols, float_format="%.4f")


def load_nav(code: str) -> pd.DataFrame:
    df = pd.read_csv(nav_path(code))
    df["date"] = pd.to_datetime(df["date"])
    for col in ("nav", "acc_nav"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def has_nav(code: str) -> bool:
    return os.path.exists(nav_path(code))


def last_date(code: str) -> datetime | None:
    """本地净值文件中的最新日期，无文件或为空返回 None。"""
    if not has_nav(code):
        return None
    try:
        df = pd.read_csv(nav_path(code), usecols=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max().to_pydatetime()
    except Exception:  # noqa: BLE001 - 损坏文件按无数据处理
        return None


def is_fresh(code: str, refresh_days: int) -> bool:
    """净值文件存在且修改时间距今不超过 refresh_days 天，则视为新鲜可跳过。"""
    path = nav_path(code)
    if not os.path.exists(path):
        return False
    if refresh_days <= 0:
        return False
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime < timedelta(days=refresh_days)
    except OSError:
        return False


def load_splits() -> dict:
    """返回 {code: [(date_str, 折算比例), ...]}，无文件时返回空 dict。"""
    import json

    path = os.path.join(DATA_DIR, "splits.json")
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    return {c: [tuple(x) for x in v] for c, v in raw.items()}


def local_codes():
    """已下载到本地的基金代码列表。"""
    if not os.path.isdir(NAV_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(NAV_DIR) if f.endswith(".csv"))
