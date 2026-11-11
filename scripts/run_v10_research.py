#!/usr/bin/env python3
"""One-command v1.0.1 local research pipeline.

Uses the user's existing local data/nav cache and account snapshot. Never fetches
network data and never submits fund orders. DATA_BLOCKED is a valid research
outcome and is preserved in structured reports rather than replaced by stale files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import date, datetime

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], records: list[dict]) -> int:
    print("\n" + "=" * 78)
    print(f"[RUN] {name}")
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT, text=True)
    records.append({
        "name": name,
        "command": command,
        "returncode": int(result.returncode),
    })
    print(f"[{'OK' if result.returncode == 0 else 'FAILED'}] {name} (exit={result.returncode})")
    return int(result.returncode)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "REPORT_READ_ERROR", "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete local fund-strategy research pipeline")
    parser.add_argument("--asof", default=date.today().isoformat(), help="Decision cutoff date YYYY-MM-DD")
    parser.add_argument("--regime-start", default=None, help="Optional verified current-style start date")
    parser.add_argument("--max-fund-weight", type=float, default=0.40)
    parser.add_argument("--turnover-cost", type=float, default=0.003)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--weight-candidates", type=int, default=2400)
    parser.add_argument("--weight-paths", type=int, default=192)
    parser.add_argument("--joint-weight-pool", type=int, default=600)
    parser.add_argument("--joint-candidates", type=int, default=5)
    parser.add_argument("--joint-paths", type=int, default=64)
    parser.add_argument("--fee-stress-bps", default="30,60")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    if not (ROOT / "config/account_snapshot.json").exists():
        print("DATA_BLOCKED: missing config/account_snapshot.json")
        return 2
    if not (ROOT / "data/nav").is_dir():
        print("DATA_BLOCKED: missing data/nav/ local fund NAV cache")
        return 2

    # 先归档上一轮生成物，避免本轮DATA_BLOCKED时旧报告冒充新结果。
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    archive_dir = reports_dir / "archive" / datetime.now().strftime("previous_%Y%m%d_%H%M%S")
    generated = [
        "DATA_COVERAGE.md", "DATA_QUALITY_AUDIT.md", "data_coverage.json",
        "PORTFOLIO_RISK_REPORT.md", "portfolio_risk.json",
        "weight_optimization_research.json", "joint_strategy_optimization.json",
        "oos_trials.csv", "oos_summary.json", "validate.log",
        "OOS_BACKTEST_REPORT.md", "OPTIMIZATION_RESULT.json",
        "PARAMETER_SENSITIVITY.md", "v10_research_run.json",
    ]
    archived = []
    for name in generated:
        source = reports_dir / name
        if source.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, archive_dir / name)
            source.unlink()
            archived.append(name)
    sensitivity_dir = reports_dir / "sensitivity"
    if sensitivity_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sensitivity_dir, archive_dir / "sensitivity", dirs_exist_ok=True)
        shutil.rmtree(sensitivity_dir)

    py = sys.executable
    records: list[dict] = []
    failures = 0

    if not args.skip_audit:
        failures += run_step("account as-of audit",
                             [py, "main.py", "audit", "--today", args.asof], records) != 0
        failures += run_step("fund NAV / disclosure / factor data audit",
                             [py, "scripts/data_audit.py"], records) != 0
        if (ROOT / "data/factors/daily.csv").exists():
            failures += run_step("portfolio risk report",
                                 [py, "scripts/portfolio_risk.py", "--asof", args.asof], records) != 0
        else:
            print("[SKIP] portfolio risk report: data/factors/daily.csv is missing")

    weight_cmd = [
        py, "main.py", "optimize-weights", "--asof", args.asof,
        "--candidates", str(args.weight_candidates), "--paths", str(args.weight_paths),
        "--max-fund-weight", str(args.max_fund_weight),
        "--turnover-cost", str(args.turnover_cost),
        "--risk-free-rate", str(args.risk_free_rate),
    ]
    if args.regime_start:
        weight_cmd += ["--regime-start", args.regime_start]
    failures += run_step("robust target-weight search", weight_cmd, records) != 0

    joint_cmd = [
        py, "main.py", "optimize-strategy", "--asof", args.asof,
        "--train-days", "504", "--validation-days", "252", "--test-days", "252",
        "--step-days", "252", "--folds", "1",
        "--weight-pool", str(args.joint_weight_pool),
        "--weight-candidates", str(args.joint_candidates),
        "--weight-paths", str(args.joint_paths),
        "--max-fund-weight", str(args.max_fund_weight),
        "--fee-stress-bps", args.fee_stress_bps,
        "--risk-free-rate", str(args.risk_free_rate),
    ]
    if args.regime_start:
        joint_cmd += ["--regime-start", args.regime_start]
    failures += run_step("joint weight / rebalance / stop / take-profit search",
                         joint_cmd, records) != 0

    failures += run_step("rolling out-of-sample validation",
                         [py, "main.py", "validate"], records) != 0

    if (ROOT / "reports/oos_trials.csv").exists() and (ROOT / "reports/oos_summary.json").exists():
        summary = load_json(ROOT / "reports/oos_summary.json") or {}
        if summary.get("status") != "DATA_BLOCKED" and "grid_size" in summary:
            failures += run_step("OOS report generation",
                                 [py, "scripts/oos_report.py"], records) != 0
        else:
            print("[SKIP] OOS report formatting: validation did not produce trial rows")
    else:
        print("[SKIP] OOS report formatting: no fresh OOS trial output")

    if (ROOT / "reports/weight_optimization_research.json").exists() and \
       (ROOT / "reports/joint_strategy_optimization.json").exists():
        failures += run_step("optimization and sensitivity report generation",
                             [py, "scripts/optimization_report.py"], records) != 0

    if not args.skip_export and (ROOT / "reports/DATA_COVERAGE.md").exists() and \
       (ROOT / "reports/data_coverage.json").exists():
        failures += run_step("sanitized analysis export",
                             [py, "scripts/export_analysis.py"], records) != 0

    weight = load_json(ROOT / "reports/weight_optimization_research.json") or {}
    joint = load_json(ROOT / "reports/joint_strategy_optimization.json") or {}
    oos = load_json(ROOT / "reports/oos_summary.json") or {}
    status = {
        "pipeline_version": "v1.0.1",
        "asof": args.asof,
        "research_only": True,
        "trade_ready": False,
        "weight_optimization": weight.get("status", "NOT_RUN"),
        "joint_optimization": joint.get("status", "NOT_RUN"),
        "oos_validation": oos.get("decision", oos.get("status", "NOT_RUN")),
        "independent_full_year_test_windows": oos.get("independent_full_year_test_windows"),
        "steps": records,
        "command_failures": int(failures),
        "previous_reports_archived_to": str(archive_dir.relative_to(ROOT)) if archived else None,
        "previous_report_files_archived": archived,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/v10_research_run.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(json.dumps(status, ensure_ascii=False, indent=2))

    blocked = any(x in {"DATA_BLOCKED", "PARTIAL_DATA_BLOCKED",
                        "RISK_CONSTRAINT_UNMET", "INDEPENDENT_SCENARIO_REJECTED",
                        "RETURN_TARGET_UNMET"}
                  for x in (status["weight_optimization"], status["joint_optimization"]))
    joint_folds = joint.get("folds") or []
    if joint_folds and any(f.get("status") not in (
            "TEST_WINDOW_CRITERIA_MET_NOT_GENERALIZED",) for f in joint_folds):
        blocked = True
    blocked = blocked or status["oos_validation"] in {
        "NOT_VALIDATED", "INSUFFICIENT_INDEPENDENT_OOS_WINDOWS",
        "NO_FEASIBLE_STRATEGY", "DATA_BLOCKED"
    }
    if failures or blocked:
        print("\n研究流程已执行，但数据/样本外证据不足或存在命令失败；不得视为策略验收通过。")
        return 2
    print("\n研究流程完成。仍需人工审阅报告并确认目标约束；本程序不下单。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
