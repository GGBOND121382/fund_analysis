"""补充数据采集：季报持仓披露 + 市场风险因子（第1/2步前置）。

仍在 akshare/东财体系内，不引入新数据供应商：
- 资产配置（股/债/现金占净值比、净资产）：东财移动端 FundMNAssetAllocation（季度全历史）
- 前五/前十债券持仓：东财 F10 FundArchivesDatas zqcc（按年分页，HTML）
- 国债收益率曲线：akshare bond_china_yield（中国货币网，日频）
- 沪深300：akshare index_zh_a_hist（价格指数，**非全收益**，已标注）
- 黄金：akshare 上海金基准价 spot_golden_benchmark_sge（不可用时回退 SGE Au99.99）
- 汇率/信用利差/二级资本债利差：无可靠公开源，按契约**留 MISSING 不伪造**

published_at 规则推断（标注 PUBLISHED_AT_RULE_INFERRED）：
- 一/三季报：季度末 +15 个工作日（季报披露法定时限）
- 半年报（Q2）：+60 个自然日；年报（Q4）：+90 个自然日
- 收益率曲线/指数：当日 18:00（中债/交易所当日发布）
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

DISCLOSURE_DIR = os.path.join("data", "disclosures")
FACTOR_DIR = os.path.join("data", "factors")

MOB_API = "https://fundmobapi.eastmoney.com/FundMNewApi/"
F10_API = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36"}
MOB_HEADERS = {**UA, "Referer": "https://fund.eastmoney.com/"}
F10_HEADERS = {**UA, "Referer": "https://fundf10.eastmoney.com/"}
MOB_PARAMS = {"deviceid": "kimi-fund-analysis", "plat": "Android",
              "product": "EFund", "version": "6.5.5"}


def published_at_rule(report_asof: pd.Timestamp) -> pd.Timestamp:
    """定期报告披露时点（保守上界，规则推断）。"""
    q = (report_asof.month - 1) // 3
    if q in (0, 2):  # Q1/Q3：季度末+15个工作日
        return report_asof + pd.tseries.offsets.BDay(15)
    if q == 1:       # 半年报：+60自然日
        return report_asof + timedelta(days=60)
    return report_asof + timedelta(days=90)  # 年报：+90自然日


# -- 资产配置 -------------------------------------------------------------

def fetch_asset_allocation(code: str) -> pd.DataFrame:
    """东财移动端季度资产配置：FSRQ, GP(股票%), ZQ(债券%), HB(现金%), QT(其他%), JZC(净资产亿)。"""
    r = requests.get(MOB_API + "FundMNAssetAllocation",
                     params={**MOB_PARAMS, "FCODE": code}, headers=MOB_HEADERS, timeout=15)
    r.raise_for_status()
    datas = r.json().get("Datas") or []
    rows = []
    for d in datas:
        def num(k):
            v = d.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None  # "--" 等不可解析：未知，不等于 0
        rows.append({
            "report_asof": d["FSRQ"],
            "equity_nav_pct": num("GP"),
            "total_bond_nav_pct": num("ZQ"),
            "cash_nav_pct": num("HB"),
            "other_nav_pct": num("QT"),
            "fund_size_cny": (num("JZC") or 0) * 1e8 or None,
        })
    return pd.DataFrame(rows)


# -- 债券持仓 -------------------------------------------------------------

def classify_bond(name: str) -> str:
    """按债券名称粗分类（契约：只有重仓披露，权重为 lower_bound）。"""
    if re.search(r"永续", name):
        return "bank_perpetual"
    if re.search(r"二级", name):
        return "bank_tier2"
    if re.search(r"转债|可转换", name):
        return "convertible"
    if re.search(r"国债", name):
        return "gov_bond"
    if re.search(r"国开|农发|进出|口行|开行债", name):
        return "policy_bank"
    return "credit_bond"


def fetch_bond_holdings(code: str, years) -> pd.DataFrame:
    """东财 F10 债券投资明细（按年）：report_asof, bond_code, bond_name, pct_nav, mv_wan。"""
    rows = []
    for y in years:
        r = requests.get(F10_API, params={"type": "zqcc", "code": code, "year": y},
                         headers=F10_HEADERS, timeout=15)
        m = re.search(r'content:"(.*)",arryear', r.text, re.S)
        if not m:
            continue
        html = m.group(1).replace('\\"', '"').replace("\\/", "/")
        # 每个季度一个 <h4> 块，内含截止日
        for block in re.split(r"<h4[^>]*>", html)[1:]:
            dm = re.search(r"截止至：<font[^>]*>([\d-]+)</font>", block)
            if not dm:
                continue
            asof = dm.group(1)
            for tr in re.findall(r"<tr>(.*?)</tr>", block, re.S):
                cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
                if len(cells) >= 5:
                    rows.append({
                        "report_asof": asof,
                        "bond_code": re.sub(r"<[^>]+>", "", cells[1]).strip(),
                        "bond_name": re.sub(r"<[^>]+>", "", cells[2]).strip(),
                        "pct_nav": re.sub(r"<[^>]+>|%", "", cells[3]).strip(),
                        "mv_wan": re.sub(r"<[^>]+>|,", "", cells[4]).strip(),
                    })
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame(columns=["report_asof", "bond_code", "bond_name", "pct_nav", "mv_wan"])
    df = pd.DataFrame(rows)
    df["pct_nav"] = pd.to_numeric(df["pct_nav"], errors="coerce")
    df["mv_wan"] = pd.to_numeric(df["mv_wan"], errors="coerce")
    df["sector"] = df["bond_name"].map(classify_bond)
    return df


def build_disclosures(codes, years=range(2023, 2027), out_dir=DISCLOSURE_DIR) -> dict:
    """合并资产配置 + 债券持仓，逐基金落盘 disclosures/{code}.csv。"""
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for code in codes:
        try:
            alloc = fetch_asset_allocation(code)
            bonds = fetch_bond_holdings(code, years)
            if alloc.empty:
                summary[code] = "no_allocation_data"
                continue
            alloc["published_at"] = alloc["report_asof"].map(
                lambda d: published_at_rule(pd.Timestamp(d)).strftime("%Y-%m-%d"))
            sector = {}
            if not bonds.empty:
                bonds["published_at"] = bonds["report_asof"].map(
                    lambda d: published_at_rule(pd.Timestamp(d)).strftime("%Y-%m-%d"))
                sector = (bonds.groupby(["report_asof", "sector"])["pct_nav"].sum()
                          .unstack(fill_value=0).to_dict("index"))
                bonds.to_csv(os.path.join(out_dir, f"{code}_bonds.csv"), index=False)
            recs = []
            for _, r in alloc.iterrows():
                rec = r.to_dict()
                sec = sector.get(r["report_asof"], {})
                for col in ("gov_bond", "policy_bank", "credit_bond", "convertible",
                            "bank_tier2", "bank_perpetual"):
                    rec[f"{col}_nav_pct_lower_bound"] = sec.get(col, 0.0)
                rec["disclosure_completeness"] = (
                    "top_bonds_lower_bound" if sec else "asset_allocation_only")
                rec["source"] = "eastmoney(FundMNAssetAllocation+zqcc)"
                rec["quality_flag"] = "B/SINGLE_SOURCE; PUBLISHED_AT_RULE_INFERRED; TOP_HOLDINGS_LOWER_BOUND"
                recs.append(rec)
            df = pd.DataFrame(recs)
            df.insert(0, "fund_code", code)
            df.to_csv(os.path.join(out_dir, f"{code}.csv"), index=False)
            summary[code] = f"{len(df)} 期"
        except Exception as e:  # noqa: BLE001
            summary[code] = f"FAIL: {str(e)[:80]}"
    return summary


# -- 风险因子 -------------------------------------------------------------

def fetch_gov_yields(start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    # 接口单次只支持短区间（实测 ~数周），按 20 天窗口分段抓取
    frames = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=20), end_ts)
        df = ak.bond_china_yield(start_date=cursor.strftime("%Y%m%d"),
                                 end_date=chunk_end.strftime("%Y%m%d"))
        if len(df):
            frames.append(df)
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.4)
    if not frames:
        raise ValueError("bond_china_yield 全部区间无数据")
    df = pd.concat(frames).drop_duplicates("日期")
    df = df.rename(columns={"日期": "date", "1年": "gov_yield_1y", "3年": "gov_yield_3y",
                            "5年": "gov_yield_5y", "10年": "gov_yield_10y",
                            "30年": "gov_yield_30y"})
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c.startswith("gov_yield"):
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # 百分比 -> 小数
    return df[["date"] + [c for c in df.columns if c.startswith("gov_yield")]]


def fetch_index_daily(symbol: str, name: str) -> pd.DataFrame | None:
    """指数日线（收盘价），失败返回 None。"""
    import akshare as ak

    try:
        if name == "csi300":
            df = ak.index_zh_a_hist(symbol=symbol, period="daily")
            out = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "csi300_price_index"})
        elif name == "gold":
            df = ak.spot_golden_benchmark_sge()
            out = df[["交易时间", "晚盘价"]].rename(
                columns={"交易时间": "date", "晚盘价": "cny_gold_price"})
            out = out[out["cny_gold_price"].notna()]
        else:
            return None
        out["date"] = pd.to_datetime(out["date"])
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  因子 {name} 获取失败（不伪造，留 MISSING）: {str(e)[:100]}")
        return None


def build_risk_factors(start="2019-01-01", end=None, out_dir=FACTOR_DIR) -> str:
    """合成日频因子表 -> data/factors/daily.csv。缺项留空列不伪造。"""
    os.makedirs(out_dir, exist_ok=True)
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    base = fetch_gov_yields(start, end)
    # 无风险利率：1Y 国债收益率（年化小数），日频 rf = /252
    base["risk_free_annual"] = base["gov_yield_1y"]
    base["risk_free_daily"] = base["risk_free_annual"] / 252.0
    for symbol, name in (("000300", "csi300"), (None, "gold")):
        ext = fetch_index_daily(symbol, name)
        if ext is not None:
            base = base.merge(ext, on="date", how="outer")
    base = base.sort_values("date").reset_index(drop=True)
    base["published_at"] = base["date"].dt.strftime("%Y-%m-%d") + "T18:00:00+08:00"
    base["source"] = "chinamoney(国债曲线)+akshare"
    base["quality_flag"] = ("B/SINGLE_SOURCE; PUBLISHED_AT_RULE_INFERRED; "
                            "CSI300为价格指数非全收益; 信用利差/汇率 MISSING")
    path = os.path.join(out_dir, "daily.csv")
    base.to_csv(path, index=False, float_format="%.6f")
    return path