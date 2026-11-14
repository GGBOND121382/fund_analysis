#!/usr/bin/env python3
"""Fetch (resumable) then export the complete scored and unranked fund lists."""
from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile

import fetch_all_fund_rank_data as fetch
import screen_defensive_funds as rank

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asof", required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "all_funds")
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "all_funds")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--refresh-catalog", action="store_true")
    p.add_argument("--shocks", type=Path, default=ROOT / "config" / "all_fund_shocks.json")
    a = p.parse_args()
    if not a.skip_fetch:
        fetch.main(["--asof", a.asof, "--data-dir", str(a.data_dir), "--workers", str(a.workers)]
                   + (["--refresh-catalog"] if a.refresh_catalog else []))
    rank.main(["--universe", "all", "--asof", a.asof, "--data-dir", str(a.data_dir),
               "--output", str(a.output), "--shocks", str(a.shocks)])
    log = a.data_dir / "fetch_audit.csv"
    dest = a.output / "fetch_audit.csv"
    if log.is_file() and log.resolve() != dest.resolve():
        copyfile(log, dest)


if __name__ == "__main__":
    main()
