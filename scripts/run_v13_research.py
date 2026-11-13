#!/usr/bin/env python3
"""v1.3 研究流水线一键入口（P0→P5），支持断点续跑。

用法:
  python scripts/run_v13_research.py --asof 2026-11-12 \
      --shock-baseline 2026-10-29 --shock-end 2026-11-12 [--skip-fetch] \
      [--mode halfyear|quarterly|both] [--pool A|B|both]

每个阶段完成后写入 reports/.v13_checkpoints.json；已完成的阶段自动跳过
（除非 --force）。所有子命令日志落盘 reports/v13_pipeline_<stage>.log。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CKPT = REPORTS / ".v13_checkpoints.json"


def sh(cmd: list[str], log: str) -> int:
    print(f"[v13] $ {' '.join(cmd)}")
    with open(REPORTS / log, "w", encoding="utf-8") as fh:
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[v13] rc={rc} elapsed={time.time()-t0:.0f}s log=reports/{log}")
    return rc


def load_ckpt() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text(encoding="utf-8"))
    return {}


def save_ckpt(c: dict) -> None:
    CKPT.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="v1.3 research pipeline P0-P5")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--shock-baseline", default="2026-10-29")
    ap.add_argument("--shock-end", default=None, help="默认 asof 当日")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--mode", choices=["halfyear", "quarterly", "both"], default="both")
    ap.add_argument("--pool", choices=["A", "B", "both"], default="both")
    ap.add_argument("--force", action="store_true", help="忽略断点全部重跑")
    a = ap.parse_args()
    shock_end = a.shock_end or a.asof
    ckpt = {} if a.force else load_ckpt()

    def done(stage):
        ckpt[stage] = {"asof": a.asof, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_ckpt(ckpt)

    py = sys.executable

    if "P0" not in ckpt:
        rc = sh([py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
                "v13_pipeline_P0_tests.log")
        if rc != 0:
            print("P0 FAILED: unit tests"); return 1
        rc = sh([py, "scripts/data_audit.py"], "v13_pipeline_P0_audit.log")
        if rc != 0:
            print("P0 FAILED: data audit"); return 1
        done("P0")

    if "P1_fetch" not in ckpt:
        if a.skip_fetch:
            print("[v13] --skip-fetch: 沿用本地净值缓存")
        else:
            rc = sh([py, "main.py", "fetch", "--types", "债券型", "--workers", "8"],
                    "v13_pipeline_P1_fetch.log")
            if rc != 0:
                print("P1 fetch FAILED"); return 1
        done("P1_fetch")

    if "P1_rank" not in ckpt:
        rc = sh([py, "-m", "scripts.bond_resilience", "--baseline", a.shock_baseline,
                 "--end", shock_end, "--min-days", "6", "--top", "100",
                 "--output", "reports/bond_resilience_20261030_latest.csv"],
                "v13_pipeline_P1_rank.log")
        if rc != 0:
            print("P1 rank FAILED"); return 1
        rc = sh([py, "scripts/v13_bond_shortlist.py"], "v13_pipeline_P1_shortlist.log")
        if rc != 0:
            print("P1 shortlist FAILED"); return 1
        done("P1_rank")

    if "P2" not in ckpt:
        codes = (ROOT / "reports/v13_shortlist_codes.txt").read_text().strip()
        rc = sh([py, "main.py", "extra", "--part", "disclosures", "--codes", codes],
                "v13_pipeline_P2_disclosures.log")
        if rc != 0:
            print("P2 disclosures FAILED"); return 1
        rc = sh([py, "scripts/v13_due_diligence.py"], "v13_pipeline_P2_dd.log")
        if rc != 0:
            print("P2 due diligence FAILED"); return 1
        done("P2")

    if "P3" not in ckpt:
        rc = sh([py, "main.py", "optimize-weights", "--asof", a.asof,
                 "--lookback", "252", "--paths", "96"], "v13_pipeline_P3_252.log")
        if rc != 0:
            print("P3 252 FAILED"); return 1
        (ROOT / "reports/v13_weights_recent252.json").write_bytes(
            (ROOT / "reports/weight_optimization_research.json").read_bytes())
        rc = sh([py, "main.py", "optimize-weights", "--asof", a.asof,
                 "--lookback", "756", "--paths", "96"], "v13_pipeline_P3_756.log")
        (ROOT / "reports/v13_weights_stress756.json").write_bytes(
            (ROOT / "reports/weight_optimization_research.json").read_bytes())
        if a.pool in ("B", "both"):
            pool = json.loads((ROOT / "config/v13_candidate_pool_research.json")
                              .read_text(encoding="utf-8"))
            codes = ",".join([f["code"] for f in pool["holdings"]] +
                             [c["code"] for c in pool["candidates"]])
            rc = sh([py, "main.py", "optimize-weights", "--asof", a.asof,
                     "--lookback", "252", "--paths", "96", "--codes", codes],
                    "v13_pipeline_P3_poolB.log")
            (ROOT / "reports/v13_weights_recent252_poolB.json").write_bytes(
                (ROOT / "reports/weight_optimization_research.json").read_bytes())
        # 恢复规范输出为池A 252结果
        (ROOT / "reports/weight_optimization_research.json").write_bytes(
            (ROOT / "reports/v13_weights_recent252.json").read_bytes())
        done("P3")

    if "P4" not in ckpt:
        rc = sh([py, "scripts/v13_execution_audit.py"], "v13_pipeline_P4_execution.log")
        if rc != 0:
            print("P4 FAILED"); return 1
        done("P4")

    if "P5" not in ckpt:
        modes = ["halfyear", "quarterly"] if a.mode == "both" else [a.mode]
        for m in modes:
            rc = sh([py, "scripts/run_v11_exploratory.py", "--asof", a.asof,
                     "--mode", m], f"v13_pipeline_P5_{m}.log")
            if rc != 0:
                print(f"P5 {m} FAILED"); return 1
            (ROOT / f"reports/v13_oos_{m}.json").write_bytes(
                (ROOT / "reports/joint_strategy_optimization.json").read_bytes())
        done("P5")

    print("[v13] ALL STAGES DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
