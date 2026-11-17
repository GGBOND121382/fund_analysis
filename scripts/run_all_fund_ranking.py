#!/usr/bin/env python3
"""Fetch (resumable) then export the complete scored and unranked fund lists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copyfile
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_all_fund_rank_data as fetch  # noqa: E402
import screen_defensive_funds as rank  # noqa: E402
from fundkit.fetcher import fetch_trade_status  # noqa: E402
STATUS_SOURCE = "天天基金/东方财富申购状态批量接口（AkShare fund_purchase_em）"


def refresh_channel_status(data_dir: Path, *, skip: bool = False) -> dict:
    """Refresh a source-specific purchase-status snapshot without gating rankings."""
    data_dir.mkdir(parents=True, exist_ok=True)
    status_path = data_dir / "fund_status.csv"
    audit_path = data_dir / "fund_status_audit.json"
    checked_at = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%dT%H:%M:%S%z")
    state, error, records = "skipped_no_cache", "", 0

    if skip:
        if status_path.is_file():
            state = "skipped_cached"
            try:
                records = len(pd.read_csv(status_path, dtype={"code": str}))
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                state, error = "skipped_cache_unreadable", str(exc)[:500]
    else:
        try:
            status = fetch_trade_status()
            if status.empty:
                raise ValueError("empty_purchase_status_snapshot")
            status = status.copy()
            status["status_source"] = STATUS_SOURCE
            status["status_checked_at"] = checked_at
            fetch.atomically_write(status, status_path)
            state, records = "updated", len(status)
        except Exception as exc:  # noqa: BLE001 - status failure must not block historical ranking
            error = f"{type(exc).__name__}: {exc}"[:500]
            if status_path.is_file():
                state = "failed_using_previous_cache"
                try:
                    records = len(pd.read_csv(status_path, dtype={"code": str}))
                except (OSError, ValueError, pd.errors.ParserError):
                    records = 0
            else:
                state = "failed_no_cache"

    audit = {
        "state": state,
        "source": STATUS_SOURCE,
        "checked_at": checked_at if state == "updated" else None,
        "records": records,
        "cache_path": str(status_path),
        "error": error,
        "ranking_filter": False,
        "scope_note": "该状态仅代表天天基金/东方财富这一销售渠道；未查询到不等于其他平台不可购买。申购与赎回状态分开记录，赎回状态不决定能否申购。",
    }
    tmp = audit_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(audit_path)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asof", required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "all_funds")
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "all_funds")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--refresh-catalog", action="store_true")
    p.add_argument("--skip-status", action="store_true",
                   help="不联网更新天天基金渠道申购状态；使用已有快照（若存在）")
    p.add_argument("--shocks", type=Path, default=ROOT / "config" / "all_fund_shocks.json")
    a = p.parse_args()
    status_audit = refresh_channel_status(a.data_dir, skip=a.skip_status)
    print(f"渠道申购状态快照: {status_audit['state']}，记录 {status_audit['records']} 条；"
          "该快照仅作参考，不参与基金排名筛选。")
    if status_audit["error"]:
        print(f"渠道状态更新说明: {status_audit['error']}")
    if not a.skip_fetch:
        fetch.main(["--asof", a.asof, "--data-dir", str(a.data_dir), "--workers", str(a.workers)]
                   + (["--refresh-catalog"] if a.refresh_catalog else []))
    rank.main(["--universe", "all", "--asof", a.asof, "--data-dir", str(a.data_dir),
               "--output", str(a.output), "--shocks", str(a.shocks)])
    for name in ("fetch_audit.csv", "fund_status_audit.json"):
        source = a.data_dir / name
        dest = a.output / name
        if source.is_file() and source.resolve() != dest.resolve():
            copyfile(source, dest)


if __name__ == "__main__":
    main()
