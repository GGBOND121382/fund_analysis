"""基金日净值的多数据源实现。

三个来源，按限流程度互补：
- danjuan: 蛋卷基金（雪球系），JSON，单请求可拉全部历史（size<=10000），非东财系，限流宽松
- em_f10:  天天基金 F10 历史净值 JSON 接口（api.fund.eastmoney.com/f10/lsjz），
           每页固定 20 条、含累计净值，反爬远轻于 pingzhongdata，但全量需多次翻页，适合兜底
           （注：实测其 sdate/edate 参数已被服务端忽略，无法区间查询）
- em_js:   akshare fund_open_fund_info_em（东财 pingzhongdata JS 页面），限流最严重，最后兜底

统一返回 DataFrame[date(datetime64), nav(float), acc_nav(float|可选)]
"""

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
EM_F10_HEADERS = {**UA, "Referer": "https://fundf10.eastmoney.com/"}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("nav", "acc_nav"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_danjuan(code: str, timeout: float = 15) -> pd.DataFrame:
    """蛋卷基金：单请求拉全部历史净值。"""
    url = f"https://danjuanfunds.com/djapi/fund/nav/history/{code}"
    r = requests.get(url, params={"page": 1, "size": 10000}, headers=UA, timeout=timeout)
    r.raise_for_status()
    data = r.json().get("data") or {}
    items = data.get("items") or []
    if not items:
        raise ValueError(f"danjuan: empty nav data for {code}")
    return _normalize(pd.DataFrame(items)[["date", "nav"]])


def fetch_em_f10(code: str, timeout: float = 15, max_pages: int = 400) -> pd.DataFrame:
    """天天基金 F10：JSON 分页（每页固定 20 条），含累计净值。"""
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    params = {"fundCode": code, "pageIndex": 1, "pageSize": 20}

    rows, page = [], 1
    while page <= max_pages:
        params["pageIndex"] = page
        r = requests.get(url, params=params, headers=EM_F10_HEADERS, timeout=timeout)
        r.raise_for_status()
        batch = (r.json().get("Data") or {}).get("LSJZList") or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 20:  # 不足一页说明已到最后一页
            break
        page += 1
    if not rows:
        raise ValueError(f"em_f10: empty nav data for {code}")
    df = pd.DataFrame(rows)
    df = df.rename(columns={"FSRQ": "date", "DWJZ": "nav", "LJJZ": "acc_nav"})
    return _normalize(df[["date", "nav", "acc_nav"]])


def fetch_em_js(code: str, timeout: float = 15) -> pd.DataFrame:
    """akshare 东财 JS 页面源（限流重，仅作兜底）。"""
    import akshare as ak

    df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    if df is None or df.empty:
        raise ValueError(f"em_js: empty nav data for {code}")
    df = df.rename(columns={"净值日期": "date", "单位净值": "nav"})
    return _normalize(df[["date", "nav"]])


SOURCE_FUNCS = {
    "danjuan": fetch_danjuan,
    "em_f10": fetch_em_f10,
    "em_js": fetch_em_js,
}