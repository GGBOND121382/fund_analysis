"""第1步：数据质量与公开时点审核。

对契约 5 张输入表做完整性审核，产出 data/quality/step1_audit.{json,md}。
数据不可靠时给出 DATA_BLOCKED / RESEARCH_ONLY 结论，不输出交易建议。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from fundkit import metrics, storage
from fundkit.adapters import (ACCOUNT_DRAWDOWN_UNVERIFIABLE, LocalDataAdapter,
                              assert_no_lookahead)

QUALITY_DIR = os.path.join("data", "quality")


def _fund_nav_check(code: str, today: pd.Timestamp) -> dict:
    row = {"code": code, "status": "MISSING"}
    try:
        df = storage.load_nav(code)
    except Exception:  # noqa: BLE001
        return row
    if df.empty:
        return row
    rets = df["nav"].pct_change().dropna()
    gaps = df["date"].diff().dt.days.dropna()
    row.update({
        "status": "OK",
        "rows": len(df),
        "first": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "last": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "stale_days": int((today - df["date"].iloc[-1]).days),
        "avg_gap": round(float(gaps.mean()), 2) if len(gaps) else None,
        "max_abs_daily": round(float(rets.abs().max()), 4) if len(rets) else None,
        "has_acc_nav": bool("acc_nav" in df.columns),
    })
    return row


def run_audit(today: str | None = None, out_dir: str = QUALITY_DIR) -> dict:
    today_ts = pd.Timestamp(today) if today else pd.Timestamp(datetime.now().date())
    adapter = LocalDataAdapter()
    snap = adapter.account_snapshot()
    codes = snap["code"].tolist()

    report: dict = {
        "audit_date": today_ts.strftime("%Y-%m-%d"),
        "holdings_snapshot_date": snap["snapshot_date"].iloc[0],
        "lots_mode": snap["lots_mode"].iloc[0],
        "tables": {},
        "holdings_nav": [],
        "special_checks": [],
        "decision": None,
    }

    # 表1 净值：持仓 13 只逐只核查 + as-of 过滤自检
    nav_rows = []
    for code in codes:
        r = _fund_nav_check(code, today_ts)
        divs = storage.load_dividends().get(code, [])
        r["dividend_events"] = len(divs)
        nav_rows.append(r)
    mismatch = set()
    if os.path.exists(os.path.join("data", "dividend_mismatch.csv")):
        mismatch = set(pd.read_csv("data/dividend_mismatch.csv", dtype={"code": str})["code"])
    for r in nav_rows:
        r["dividend_mismatch"] = r["code"] in mismatch
    report["holdings_nav"] = nav_rows
    ok = sum(1 for r in nav_rows if r["status"] == "OK")
    report["tables"]["nav_total_return_daily"] = {
        "status": "PARTIAL" if ok == len(codes) else "MISSING",
        "holdings_ok": f"{ok}/{len(codes)}",
        "gaps": ["published_at 规则推断（C级证据）", "无逐行 source/fetched_at 溯源",
                 "danjuan 源无 cumulative_nav"],
    }

    # as-of 自检：以快照日为决策时点，确认无未来净值混入
    nav_asof = adapter.nav(codes, f"{snap['snapshot_date'].iloc[0]}T15:00:00+08:00")
    assert_no_lookahead(nav_asof, f"{snap['snapshot_date'].iloc[0]}T15:00:00+08:00")
    report["asof_selfcheck"] = {
        "decision_at": f"{snap['snapshot_date'].iloc[0]}T15:00+08:00",
        "rows_passed": len(nav_asof),
        "max_nav_date": str(nav_asof["nav_date"].max().date()) if len(nav_asof) else None,
        "published_lag_days_rule": adapter.published_lag_days,
    }

    # 表2/表3/表4/表5
    disc = adapter.disclosures(codes, f"{report['audit_date']}T23:59:00+08:00")
    disc_codes = disc["fund_code"].nunique() if len(disc) else 0
    report["tables"]["fund_holdings_disclosures"] = (
        {"status": "PARTIAL", "coverage": f"{disc_codes}/{len(codes)}",
         "periods": int(len(disc)),
         "note": "东财资产配置+前五债券持仓；published_at 规则推断；重仓权重为 lower_bound；久期/全持仓 MISSING"}
        if disc_codes else
        {"status": "MISSING", "note": "季报持仓未采集（python main.py extra --part disclosures）"})
    factors = adapter.risk_factors(f"{report['audit_date']}T23:59:00+08:00")
    report["tables"]["market_risk_factors_daily"] = (
        {"status": "PARTIAL", "rows": int(len(factors)),
         "range": f"{factors['date'].min().date()}~{factors['date'].max().date()}",
         "note": "国债曲线/沪深300（价格非全收益）/上海金；信用利差、二级资本债利差、汇率 MISSING（无可靠源，不伪造）"}
        if len(factors) else
        {"status": "MISSING", "note": "风险因子未采集（python main.py extra --part factors）"})
    lots = adapter.cashflows_and_lots()
    report["tables"]["account_lots_and_cashflows"] = {
        "status": "SIMULATED", "simulated_lots": len(lots),
        "note": ACCOUNT_DRAWDOWN_UNVERIFIABLE}
    rules = adapter.trade_rules(codes)
    report["tables"]["fund_trade_rules"] = {
        "status": "PARTIAL",
        "coverage": f"{len(rules)}/{len(codes)}",
        "gaps": ["仅当前快照无生效时间历史", "费率分档/锁定期窗口/确认时点未接入"]}

    # 专项核查（数据层面；风格回归属第2步）
    for code, label in (("021559", "格林：异常NAV/规模"),
                        ("014846", "博时：持仓收益异常"),
                        ("710301", "富安达：风格迁移前置核查")):
        row = next((r for r in nav_rows if r["code"] == code), {})
        note = ""
        if row.get("status") == "OK":
            if code == "021559":
                note = ("净值日频披露但样本仅约2年；规模与持仓无数据（MISSING），"
                        "单日异动见 max_abs_daily；滚动锁定批次窗口仅能模拟假设")
            elif code == "014846":
                note = ("净值序列正常；'账户收益异常'无法用流水核对（无真实流水），"
                        "模拟批次口径下无法区分数据错误与真实申赎时点效应")
            else:
                try:
                    disc = pd.read_csv(os.path.join("data", "disclosures", f"{code}.csv"))
                    disc = disc.sort_values("report_asof")
                    eq = disc[["report_asof", "equity_nav_pct",
                               "convertible_nav_pct_lower_bound"]].dropna(how="all", subset=["equity_nav_pct"])
                    note = ("披露数据已接入："
                            + "; ".join(f"{r.report_asof} 股票{('%.1f%%' % r.equity_nav_pct) if pd.notna(r.equity_nav_pct) else '未披露/趋零'}"
                                        f" 转债下限{r.convertible_nav_pct_lower_bound:.1f}%"
                                        for r in eq.tail(5).itertuples())
                            + "；风格回归定量判定见第2步 style 模块")
                except FileNotFoundError:
                    note = ("净值序列完整；风格迁移判定需季报持仓+因子回归（第2步，"
                            "当前 disclosures MISSING，暂不可辨识）")
        report["special_checks"].append({"code": code, "item": label,
                                         "nav": row, "finding": note})

    # 结论
    blocking = [k for k, v in report["tables"].items() if v["status"] == "MISSING"]
    partial = [k for k, v in report["tables"].items() if v["status"] in ("PARTIAL", "SIMULATED")]
    if blocking:
        msg = f"存在缺失表 {blocking}：所有输出限 RESEARCH_ONLY，不提供交易建议"
    else:
        msg = ("五表均有数据，但账户流水为模拟批次（真实MDD不可验证）、披露/因子为"
               "规则推断时点与单源口径：所有输出限 RESEARCH_ONLY，不提供交易建议")
    report["decision"] = {
        "recommendation_status": "RESEARCH_ONLY",
        "blocked_tables": blocking,
        "partial_tables": partial,
        "message": msg,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "step1_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    _write_md(report, os.path.join(out_dir, "step1_audit.md"))
    return report


def _write_md(report: dict, path: str) -> None:
    lines = [f"# 第1步数据质量审核（{report['audit_date']}）", ""]
    lines.append("## 五张契约输入表状态")
    lines.append("")
    lines.append("| 表 | 状态 | 说明 |")
    lines.append("|---|---|---|")
    for k, v in report["tables"].items():
        note = v.get("note") or "; ".join(v.get("gaps", [])) or v.get("holdings_ok", "")
        lines.append(f"| {k} | {v['status']} | {note} |")
    lines.append("")
    lines.append(f"## 持仓 13 只净值核查（快照日 {report['holdings_snapshot_date']}，"
                 f"批次模式 {report['lots_mode']}）")
    lines.append("")
    lines.append("| 代码 | 行数 | 区间 | 滞后天数 | 平均披露间隔 | 单日最大异动 | 分红事件 | 对账不符 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in report["holdings_nav"]:
        if r["status"] != "OK":
            lines.append(f"| {r['code']} | MISSING | - | - | - | - | - | - |")
            continue
        lines.append(f"| {r['code']} | {r['rows']} | {r['first']}~{r['last']} "
                     f"| {r['stale_days']} | {r['avg_gap']} | {r['max_abs_daily']} "
                     f"| {r['dividend_events']} | {'是' if r['dividend_mismatch'] else '否'} |")
    lines.append("")
    chk = report["asof_selfcheck"]
    lines.append(f"## as-of 自检：决策时点 {chk['decision_at']}，通过 {chk['rows_passed']} 行，"
                 f"最迟净值日 {chk['max_nav_date']}（披露延迟规则 {chk['published_lag_days_rule']} 天）")
    lines.append("")
    lines.append("## 专项核查")
    lines.append("")
    for s in report["special_checks"]:
        lines.append(f"- **{s['item']}（{s['code']}）**：{s['finding']}")
    lines.append("")
    d = report["decision"]
    lines.append(f"## 结论：`{d['recommendation_status']}`——{d['message']}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))