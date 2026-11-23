#!/usr/bin/env python3
"""同花顺爱基金渠道可交易核查：对 M0 可交易榜前列基金逐只查询同花顺交易状态。

数据源：https://fund.10jqka.com.cn/data/client/myfund/{code}（同花顺爱基金公开接口）
判定：buy=="1" 且 sgstat in {开放, 限制大额} -> 同花顺支持申购
     buy=="0" -> 同花顺暂不支持交易（未代销/渠道不支持）
     buy=="2" / sgstat==暂停 -> 产品层面停售
证据时间：运行时刻（晚于榜单 asof，属"当前可交易性"核查，不回填历史榜）。
"""
import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

API = "https://fund.10jqka.com.cn/data/client/myfund/{}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://fund.10jqka.com.cn/public/ifundout/dist/detail.html",
}
TRADEABLE_SGSTAT = {"开放", "限制大额"}


def fetch_status(code: str, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            r = requests.get(API.format(code), headers=HEADERS, timeout=15)
            data = r.json().get("data")
            if data:
                d = data[0]
                return {
                    "code": code,
                    "ths_name": d.get("name"),
                    "sgstat": d.get("sgstat"),
                    "shstat": d.get("shstat"),
                    "buy": str(d.get("buy")),
                    "zdsg": d.get("zdsg"),
                    "error": None,
                }
            return {"code": code, "ths_name": None, "sgstat": None, "shstat": None,
                    "buy": None, "zdsg": None, "error": "empty data"}
        except Exception as e:
            if attempt == retries:
                return {"code": code, "ths_name": None, "sgstat": None, "shstat": None,
                        "buy": None, "zdsg": None, "error": str(e)}
            time.sleep(1.0)


def ths_tradeable(row) -> tuple:
    if row["error"]:
        return None, f"查询失败({row['error']})"
    if row["buy"] == "1" and row["sgstat"] in TRADEABLE_SGSTAT:
        note = "同花顺可申购" + (f"({row['sgstat']})" if row["sgstat"] != "开放" else "")
        return True, note
    if row["buy"] == "2" or row["sgstat"] == "暂停":
        return False, "产品停售/暂停申购"
    if row["buy"] == "0":
        return False, "同花顺暂不支持交易"
    return False, f"状态未知(buy={row['buy']},sgstat={row['sgstat']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/all_funds/comprehensive_20261120")
    ap.add_argument("--candidates", type=int, default=200,
                    help="每期限核查的候选数（渠道可交易榜前列）")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    inp = Path(args.input)
    checked_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    summary = {}
    for period in ("5y", "3y", "1y", "6m"):
        cand = pd.read_csv(inp / f"top_candidates_channel_tradeable_{period}.csv", dtype={"code": str})
        cand["code"] = cand["code"].str.zfill(6)
        cand = cand.head(args.candidates).reset_index(drop=True)
        rows = []
        for i, code in enumerate(cand["code"]):
            st = fetch_status(code)
            rows.append(st)
            if (i + 1) % 20 == 0:
                print(f"{period}: {i+1}/{len(cand)}", flush=True)
            time.sleep(args.sleep)
        sdf = pd.DataFrame(rows)
        sdf["checked_at"] = checked_at
        sdf.to_csv(inp / f"ths_tradeable_status_{period}.csv", index=False)
        merged = cand.merge(sdf, on="code", how="left")
        verdicts = merged.apply(ths_tradeable, axis=1)
        merged["ths_tradeable"] = [v[0] for v in verdicts]
        merged["ths_note"] = [v[1] for v in verdicts]
        trad = merged[merged["ths_tradeable"] == True].reset_index(drop=True)  # noqa: E712
        excl = merged[merged["ths_tradeable"] != True].reset_index(drop=True)  # noqa: E712
        trad.insert(0, "ths_rank", range(1, len(trad) + 1))
        trad.head(args.top).to_csv(inp / f"top10_ths_tradeable_{period}.csv", index=False)
        excl.to_csv(inp / f"excluded_ths_untradeable_{period}.csv", index=False)
        summary[period] = {
            "candidates_checked": int(len(merged)),
            "ths_tradeable": int(len(trad)),
            "excluded": int(len(excl)),
            "excluded_reasons": excl["ths_note"].value_counts().to_dict(),
            "errors": int(sdf["error"].notna().sum()),
        }
        print(period, summary[period], flush=True)
    audit = {"source": API, "evidence_time": checked_at,
             "note": "同花顺爱基金渠道交易状态，晚于榜单 asof=2026-11-20，仅作当前可交易性过滤，不回填历史榜",
             "rule": 'buy=="1" and sgstat in {开放,限制大额}', "periods": summary}
    with open(inp / "ths_tradeable_audit.json", "w") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
