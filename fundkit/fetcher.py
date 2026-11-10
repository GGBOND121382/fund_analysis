"""基金列表与日净值抓取：多数据源调度、自动故障转移。"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from fundkit.sources import SOURCE_FUNCS

DEFAULT_SOURCES = ("danjuan", "em_f10", "em_js")


def fetch_fund_list() -> pd.DataFrame:
    """获取全部场外基金列表。

    返回列: code, name, type
    """
    import akshare as ak

    df = ak.fund_name_em()
    df = df.rename(columns={"基金代码": "code", "基金简称": "name", "基金类型": "type"})
    return df[["code", "name", "type"]]


def fetch_dividend_detail(code: str) -> list:
    """单只基金的完整历史分红（东财 F10 分红送配页）。

    返回 [(ex_date_str, 每份分红金额_元), ...]。无分红返回空列表。
    注意: akshare 的批量接口 fund_fh_em 只覆盖近期记录，不可用于复权。
    """
    import re

    import akshare as ak

    df = ak.fund_open_fund_info_em(symbol=code, indicator="分红送配详情")
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        m = re.search(r"每10份派现金([\d.]+)元", str(row.get("每10份分红", "")))
        ex = str(row.get("除息日", ""))[:10]
        if m and ex and ex != "None":
            out.append((ex, float(m.group(1)) / 10.0))
    return out


def fetch_split_detail(code: str) -> list:
    """单只基金的份额拆分/折算历史（东财 F10 分红送配页）。

    返回 [(date_str, 折算比例k), ...]，1:k 表示 1 份变 k 份（净值相应除以 k）。
    """
    import akshare as ak

    df = ak.fund_open_fund_info_em(symbol=code, indicator="拆分详情")
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        ratio = str(row.get("拆分折算比例", ""))
        date = str(row.get("拆分折算日", ""))[:10]
        if ":" in ratio and date and date != "None":
            try:
                out.append((date, float(ratio.split(":")[1])))
            except (ValueError, IndexError):
                continue
    return out


def fetch_dividends_batch(codes, workers: int = 8, progress_every: int = 200):
    """并发抓取多只基金的分红历史，返回 {code: [(ex_date, div), ...]}（仅含有分红的基金）。"""
    results, failed = {}, 0
    total = len(codes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_dividend_detail, c): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            code = futures[fut]
            try:
                divs = fut.result()
                if divs:
                    results[code] = divs
            except Exception:  # noqa: BLE001 - 单只失败不影响整体
                failed += 1
            if i % progress_every == 0 or i == total:
                print(f"  进度 {i}/{total}，有分红 {len(results)}，失败 {failed}")
    return results


def fetch_trade_status() -> pd.DataFrame:
    """获取全部开放式基金的申购/赎回状态快照（天天基金批量接口，单次请求）。

    返回列: code, sub_status, red_status, daily_limit(日累计限定金额，元)
    """
    import akshare as ak

    df = ak.fund_purchase_em()
    df = df.rename(columns={"基金代码": "code", "申购状态": "sub_status",
                            "赎回状态": "red_status", "日累计限定金额": "daily_limit"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["daily_limit"] = pd.to_numeric(df["daily_limit"], errors="coerce")
    return df[["code", "sub_status", "red_status", "daily_limit"]]


def tradable_codes(status_df: pd.DataFrame, min_daily_limit: float = 10000) -> set:
    """可交易 = 赎回开放，且（申购开放 或 限大额但日限额 >= min_daily_limit）。

    注意快照中"开放申购"的日限额字段 0 或 1e10+ 均表示未设限，只对"限大额"应用阈值。
    """
    red_ok = status_df["red_status"] == "开放赎回"
    sub_open = status_df["sub_status"] == "开放申购"
    sub_limited = (status_df["sub_status"] == "限大额") & (
        status_df["daily_limit"].fillna(0) >= min_daily_limit)
    return set(status_df.loc[red_ok & (sub_open | sub_limited), "code"])


def fetch_fund_nav(
    code: str,
    sources=DEFAULT_SOURCES,
    max_retry: int = 2,
    retry_sleep: float = 2.0,
) -> tuple[pd.DataFrame, str]:
    """按顺序尝试多个数据源抓取日净值，返回 (DataFrame, 实际使用的源)。

    全部源失败时抛出 RuntimeError，附带各源错误信息。
    """
    errors = {}
    for src in sources:
        func = SOURCE_FUNCS[src]
        for attempt in range(max_retry):
            try:
                df = func(code)
                if df.empty:
                    raise ValueError("empty nav data")
                return df, src
            except Exception as e:  # noqa: BLE001 - 网络/接口异常统一重试并转移
                errors[src] = str(e).splitlines()[0][:120]
                if attempt < max_retry - 1:
                    time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"{code} all sources failed: {errors}")


def merge_nav(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """与本地历史合并：按 date 去重，新数据优先（可修正历史净值）。"""
    if old is None or old.empty:
        return new.sort_values("date").reset_index(drop=True)
    cols_new = set(new.columns)
    keep = [c for c in old.columns if c in cols_new or c == "date"]
    merged = pd.concat([old[keep], new], ignore_index=True)
    merged = merged.drop_duplicates("date", keep="last").sort_values("date")
    return merged.reset_index(drop=True)


def fetch_nav_batch(
    codes,
    sources=DEFAULT_SOURCES,
    workers: int = 8,
    max_retry: int = 2,
    progress_every: int = 100,
):
    """并发抓取多只基金的日净值。

    返回 (results, failed, used_sources):
    results 为 {code: (DataFrame, 数据源)}, failed 为 {code: 异常信息}。
    """
    results, failed, used = {}, {}, {}
    total = len(codes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_fund_nav, code, sources, max_retry): code
            for code in codes
        }
        for i, fut in enumerate(as_completed(futures), 1):
            code = futures[fut]
            try:
                results[code] = fut.result()
                used[results[code][1]] = used.get(results[code][1], 0) + 1
            except Exception as e:  # noqa: BLE001
                failed[code] = str(e)
            if i % progress_every == 0 or i == total:
                print(f"  进度 {i}/{total}，成功 {len(results)}，失败 {len(failed)}")
    return results, failed, used
