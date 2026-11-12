#!/usr/bin/env python3
"""Multi-month v1.1 exploratory walk-forward and fixed-weight B6 ablations.

Runs against LOCAL NAV cache. No downloads, no trading, no fabricated NAVs.
A successful execution is not full-year OOS acceptance.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    # 507 common dates -> 252+3 embargo+120+3 embargo+120 = 498.
    "halfyear": dict(train_days=252, validation_days=120, test_days=120,
                     step_days=120, folds=1, embargo_days=3),
    "quarterly": dict(train_days=252, validation_days=63, test_days=63,
                      step_days=63, folds=2, embargo_days=3),
}


def _fmt_pct(value):
    return "N/A" if value is None else f"{float(value):.2%}"


def _fmt_int(value):
    return "N/A" if value is None else str(int(value))


def run(argv=None):
    p = argparse.ArgumentParser(description="v1.1 exploratory research only, no orders")
    p.add_argument("--asof", default=date.today().isoformat())
    p.add_argument("--mode", choices=PROTOCOLS, default="halfyear")
    p.add_argument("--weight-pool", type=int, default=180)
    p.add_argument("--weight-candidates", type=int, default=5,
                   help="up to five DISTINCT risk-feasible continuous LP portfolios; cash baseline stays separate")
    p.add_argument("--weight-paths", type=int, default=48)
    p.add_argument("--fee-stress-bps", default="30,60")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-fund-weight", type=float, default=0.40)
    args = p.parse_args(argv)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / "joint_strategy_optimization.json"
    summary_file = reports / "v11_exploratory_summary.json"
    md_file = reports / "V11_EXPLORATORY_REPORT.md"
    for prior in (summary_file, md_file):
        if prior.exists():
            prior.unlink()
    if path.exists():
        archive = reports / "archive" / datetime.now().strftime("v11_%Y%m%d_%H%M%S")
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
        path.unlink()

    mandatory = [
        ROOT / "config" / "account_snapshot.json",
        ROOT / "data" / "nav",
    ]
    missing = [str(x.relative_to(ROOT)) for x in mandatory if not x.exists()]
    if missing:
        output = {"status": "DATA_BLOCKED", "reason": "MISSING_LOCAL_INPUT",
                  "missing": missing, "trade_ready": False}
        summary_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print("DATA_BLOCKED: missing local inputs:", ", ".join(missing))
        return 2

    protocol = PROTOCOLS[args.mode]
    cmd = [sys.executable, "main.py", "optimize-strategy",
           "--asof", args.asof, "--exploratory", "--strategy-scope", "b6-ablation"]
    for key, value in protocol.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    for key in ("weight_pool", "weight_candidates", "weight_paths",
                "fee_stress_bps", "seed", "max_fund_weight"):
        cmd += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    print("[RESEARCH_ONLY] Running", " ".join(cmd), flush=True)
    process = subprocess.run(cmd, cwd=ROOT, check=False)
    if not path.exists():
        output = {"status": "RUN_FAILED", "returncode": process.returncode,
                  "trade_ready": False}
        summary_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_version") != "joint-strategy-v1.1-exploratory":
        raise RuntimeError("STALE_REPORT_BLOCKED: not an exploratory v1.1 result")
    folds = payload.get("folds", [])
    output = {
        "status": payload.get("status"),
        "model_version": payload.get("model_version"),
        "protocol": protocol,
        "asof": args.asof,
        "strategy_scope": "b6-ablation",
        "candidate_params_pre_registered": True,
        "test_used_to_choose_parameters": False,
        "has_independent_full_year_oos": False,
        "target_net_cagr_strict_gt": .05,
        "max_drawdown_budget": .04,
        "trade_ready": False,
        "folds": [],
    }
    lines = [
        "# v1.1 B6 多月探索性样本外研究",
        "",
        "> RESEARCH_ONLY / EXPLORATORY_NOT_LONG_TERM_VALIDATED."
        " 不能依据126/63日测试判断长期年化收益达标，不能自动交易。",
        "",
        "本轮不是2026年11月的一周冒烟测试。训练权重只用训练期；"
        "验证期选参数；测试期仅诊断已冻结候选及预注册消融，不用测试结果再调参。",
        "",
        f"- 截止：{args.asof}；窗口方案：{args.mode}；参数：{protocol}",
        f"- 费用压力：{args.fee_stress_bps} bp；单基金上限：{args.max_fund_weight:.0%}",
        "- 所有试验都使用相同验证期选出的研究权重与初始反事实账户，"
        "不是用户真实持仓历史。",
        "- 5%净年化为目标，4%最大回撤为研究风险预算，均非未来保证。",
        "",
    ]
    for item in folds:
        fold = item.get("fold", {})
        summary = {
            "fold": fold.get("fold"),
            "train": fold.get("train"),
            "validation": fold.get("validation"),
            "test": fold.get("test"),
            "status": item.get("status"),
            "selected_rule": (item.get("selected") or {}).get("rule_id"),
            "ablation": [],
        }
        lines += [
            f"## Fold {fold.get('fold', '?')} — {item.get('status', 'UNKNOWN')}",
            "",
            f"训练：{fold.get('train')}；验证：{fold.get('validation')}；"
            f"测试：{fold.get('test')}",
            "",
        ]
        tests = item.get("ablation_test", [])
        if tests:
            lines += [
                "| 规则 | 费后CAGR(较差费率) | MDD(较差费率) | Sharpe(保守) | "
                "申购次数 | 赎回次数 | 费用上界(元) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for t in tests:
                m = t["test"]
                summary["ablation"].append({"rule_id": t["rule_id"], "test": m})
                sh = m.get("sharpe_conservative")
                sh_text = "N/A" if sh is None else f"{float(sh):.2f}"
                lines.append(
                    f"| {t['rule_id']} | {_fmt_pct(m.get('cagr_worst_fee'))}"
                    f" | {_fmt_pct(m.get('mdd_worst_fee'))} | {sh_text}"
                    f" | {_fmt_int(m.get('buy_events_max'))}"
                    f" | {_fmt_int(m.get('redeem_events_max'))}"
                    f" | {float(m.get('fees_total_max', 0)):.2f} |"
                )
        else:
            lines.append("无有效消融结果。原因详见JSON里的状态与阻断原因。")
        lines.append("")
        output["folds"].append(summary)
    lines += [
        "## 方法与局限", "",
        "- B2原始现金流水位法作为独立参照；A1 B6仍包含其自身的风险门，"
        "两者不应假定行为完全相同。",
        "- A1—A6对照共享同一冻结权重，且持仓初始化为反事实建仓；"
        "原始持仓的真实交易成本及完整批次不可复核。",
        "- 数据窗口仅约507个共同交易日，不具备多个完整年度独立样本外窗口。",
        "- 当前债券Carry/信用利差择时尚未接入，债券默认不进行净值趋势加仓。",
        "- 请同时检查各fold的申赎事件计数，若止盈/止损触发0次则没有退出增益证据。",
        "- 不可根据本次测试列在事后挑选最佳参数并宣称其已经样本外验证。",
    ]
    summary_file.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Report:", md_file)
    print("Raw:", path)
    if process.returncode != 0 or output["status"] in ("DATA_BLOCKED", "PARTIAL_DATA_BLOCKED"):
        print("NOT_VALIDATED:", output["status"])
        return 2
    print("EXPLORATORY_ONLY: no full-year OOS acceptance")
    return 0


if __name__ == "__main__":
    sys.exit(run())
