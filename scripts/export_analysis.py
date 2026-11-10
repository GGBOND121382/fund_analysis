#!/usr/bin/env python3
"""把 reports/ 的分析交付物脱敏导出到 analysis/（可提交到公开仓库）。

脱敏规则：
- 删除一切绝对持仓金额（value_cny / 快照金额列），替换为组合权重百分比；
- 保留全部指标、比例、日期、判定与方法论内容；
- 每日新增预算（流量参数，非存量财富）保留，GPT 复算需要。
"""
import json
import os
import re
import shutil

REPORTS = "reports"
OUT = "analysis"

COPY_ASIS = [
    "DATA_QUALITY_AUDIT.md",
    "PORTFOLIO_RISK_REPORT.md",
    "OOS_BACKTEST_REPORT.md",
    "PARAMETER_SENSITIVITY.md",
    "OPTIMIZATION_RESULT.json",
    "portfolio_risk.json",
    "oos_trials.csv",
    "oos_summary.json",
    "validate.log",
    "weight_optimization_research.json",
]


def sanitize_coverage_md(text: str, weights: dict) -> str:
    """DATA_COVERAGE.md：快照金额列 → 权重。"""
    text = text.replace("| 快照金额(元) |", "| 组合权重 |")
    def repl(m):
        code = m.group(1)
        w = weights.get(code)
        return m.group(0).replace(m.group(2), f"{w:.2%}" if w is not None else "-")
    # 表格行：| code | name | cat | type | 金额 | ...
    return re.sub(r"\| (\d{6}) \|[^|]*\|[^|]*\|[^|]*\| ([\d,]+\.\d{2}) \|", repl, text)


def sanitize_coverage_json(obj: dict, weights: dict) -> dict:
    for f in obj.get("funds", []):
        f.pop("value_cny", None)
        f["weight"] = weights.get(f["code"])
    return obj


def main():
    snap = json.load(open("config/account_snapshot.json", encoding="utf-8"))
    total = sum(float(f["value_cny"]) for f in snap["funds"])
    weights = {f["code"]: float(f["value_cny"]) / total for f in snap["funds"]}

    os.makedirs(OUT, exist_ok=True)
    for name in COPY_ASIS:
        src = os.path.join(REPORTS, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, name))

    with open(f"{REPORTS}/DATA_COVERAGE.md", encoding="utf-8") as fh:
        md = fh.read()
    with open(f"{OUT}/DATA_COVERAGE.md", "w", encoding="utf-8") as fh:
        fh.write(sanitize_coverage_md(md, weights))

    with open(f"{REPORTS}/data_coverage.json", encoding="utf-8") as fh:
        cov = json.load(fh)
    with open(f"{OUT}/data_coverage.json", "w", encoding="utf-8") as fh:
        json.dump(sanitize_coverage_json(cov, weights), fh, ensure_ascii=False, indent=2)

    # 残留检查：精确金额（带两位小数）与字段名不得出现；指标小数的子串不算泄漏
    amounts = [f"{float(f['value_cny']):.2f}" for f in snap["funds"]]
    leak_patterns = amounts + [f"{total:.2f}", "value_cny"]
    leaks = []
    for root, _dirs, files in os.walk(OUT):
        for fn in files:
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8") as fh:
                content = fh.read()
            for pat in leak_patterns:
                if pat in content:
                    leaks.append(f"{p}: {pat}")
    if leaks:
        raise SystemExit("脱敏残留检查失败:\n" + "\n".join(leaks))
    print(f"导出完成: {OUT}/（{len(COPY_ASIS) + 2} 个文件，残留检查通过）")


if __name__ == "__main__":
    main()
