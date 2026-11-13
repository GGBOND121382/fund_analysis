#!/usr/bin/env python3
"""Optional retrospective CSI 300 PE/PB cache, not verified publication vintages."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/factors/csi300_valuation_proxy.csv")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        p.exit(2, "Output already exists. Choose another path to preserve the prior cache.\n")
    import akshare as ak
    pe = ak.stock_index_pe_lg(symbol="沪深300")[["日期", "滚动市盈率"]].rename(
        columns={"日期": "observed_at", "滚动市盈率": "pe"})
    pb = ak.stock_index_pb_lg(symbol="沪深300")[["日期", "市净率"]].rename(
        columns={"日期": "observed_at", "市净率": "pb"})
    f = pe.merge(pb, on="observed_at", validate="one_to_one").dropna()
    f = f.loc[(f.pe > 0) & (f.pb > 0)].copy()
    if f.empty:
        p.exit(2, "VALUATION_DATA_BLOCKED: no valid common PE/PB observations.\n")
    f["published_at"] = pd.to_datetime(f.observed_at) + pd.Timedelta(days=1, hours=23, minutes=59)
    f["index_code"] = "000300"
    f["source_quality"] = "LEGULEGU_RETROSPECTIVE_PUBLICATION_PROXY"
    out.parent.mkdir(parents=True, exist_ok=True)
    f.sort_values("observed_at").to_csv(out, index=False)
    print(f"{len(f)} rows -> {out} (diagnostic only; inferred publication timestamp)")


if __name__ == "__main__":
    main()
