"""Capture ACTUAL present-day fund purchase statuses via AKShare / Eastmoney.

Never backdate a capture. These are third-party distribution-channel
indications, NOT evidence of a fund contract or complete roundtrip fees.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path
import re

import pandas as pd

SOURCE="https://fund.eastmoney.com/Fund_sgzt_bzdm.html"


def map_status(text, action):
    s=str(text).strip().replace(" ","")
    if not s or s in ("nan","None","--","-"):
        return "UNKNOWN"
    if "场内" in s:
        return "UNKNOWN" # Exchange tradeability needs exchange prices/halts.
    if "限" in s and ("申购" in s or "购买" in s or "大额" in s):
        return "LIMITED"
    if "暂停" in s or "停止" in s:
        return "PAUSED"
    if "封闭" in s or "未开放" in s:
        return "CLOSED"
    if ("开放" in s or "正常" in s) and ("申购" in s or "赎回" in s or s=="正常"):
        return "OPEN"
    return "UNKNOWN"


def transform_purchase_status(data, captured_at=None):
    cols={"基金代码","申购状态","赎回状态"}
    if not cols.issubset(data.columns):
        raise ValueError(f"unexpected AKShare schema: {sorted(cols-set(data.columns))}")
    captured_at=captured_at or datetime.now(timezone.utc).isoformat()
    results=[]
    for _,r in data.iterrows():
        code=str(r["基金代码"]).strip().zfill(6)
        if not re.fullmatch(r"\d{6}",code):
            continue
        sub=map_status(r["申购状态"],"purchase")
        redeem=map_status(r["赎回状态"],"redeem")
        next_day=r.get("下一开放日","")
        limit=pd.to_numeric(pd.Series([r.get("日累计限定金额","")]),errors="coerce").iloc[0]
        front_fee=pd.to_numeric(pd.Series([r.get("手续费","")]),errors="coerce").iloc[0]
        # Published front-end fee % cannot certify total purchase+redemption cost.
        results.append(dict(code=code,known_at=captured_at,source_url=SOURCE,
            source_channel="eastmoney_fund_purchase_em",
            subscription_status=sub,redemption_status=redeem,
            next_reported_open_day=str(next_day) if pd.notna(next_day) else "",
            next_exit="",lock_until="",
            max_subscription_amount=limit,front_fee_percent=front_fee,
            total_fee_rate="",official_contract_verified=False,
            fund_name=str(r.get("基金简称","")),
            raw_subscription=str(r["申购状态"]),raw_redemption=str(r["赎回状态"])))
    d=pd.DataFrame(results)
    if not d.empty:
        d=d.sort_values("code").drop_duplicates("code")
    return d


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True,
                   help="immutable timestamped CSV (new pathname per capture)")
    a=p.parse_args(argv)
    if a.output.exists():
        p.error("evidence snapshot already exists; use a unique timestamped filename")
    import akshare as ak
    raw=ak.fund_purchase_em()
    if raw.empty:
        raise RuntimeError("fund_purchase_em returned no rows; refusing empty evidence")
    now=datetime.now(timezone.utc).isoformat()
    normalized=transform_purchase_status(raw,captured_at=now)
    if normalized.empty:
        raise RuntimeError("AKShare schema produced no valid fund codes")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    normalized.to_csv(a.output,index=False,encoding="utf-8-sig")
    print(f"Captured {len(normalized)} fund share codes at {now}; tradeability still needs contract/fee corroboration")


if __name__=="__main__":
    main()
