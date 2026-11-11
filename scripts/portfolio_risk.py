#!/usr/bin/env python3
"""第三步：统一收益与全账户风险分析 → reports/PORTFOLIO_RISK_REPORT.md

口径：
- 单基金/组合收益一律用分红拆分后复权净值（metrics.adjust_nav）；
- 组合单位净值 = 快照权重买入持有（无外部现金流、不再平衡），
  MDD 在该单位净值上计算，天然剔除外部入金影响；
- 外部现金流调整的组合单位净值（TWR 单位化）与 XIRR 由 fundkit.simulator
  在模拟批次口径下实现（finance_math.external_cash_unitization）；
  真实账户现金流不可得，XIRR 只能对模拟批次计算，标注 UNVERIFIABLE；
- 共同日期取 13 只基金交集（不做任何填充）。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundkit import storage
from fundkit.adapters import load_account_snapshot
from fundkit.finance_math import drawdown_path, empirical_cdar
from fundkit.metrics import adjust_nav, compute_metrics

REPORTS = "reports"
AUDIT_DATE = pd.Timestamp("2026-11-06")  # 本地净值共同最新日（见 DATA_COVERAGE）
STYLE_SWITCH_710301 = pd.Timestamp("2025-08-29")  # 710301 纯债切换披露确认日

STRESS_WINDOWS = {
    "2022-11 债灾（理财赎回潮）": ("2022-11-01", "2022-12-15"),
    "2024-09 政策行情股债跷跷板": ("2024-09-24", "2024-10-14"),
    "2025-04 关税冲击（权益/QDII）": ("2025-04-02", "2025-04-10"),
    "2026 年中债市调整": ("2026-06-01", "2026-08-03"),
}


def total_return_series(code, div_map, split_map):
    df = adjust_nav(storage.load_nav(code), div_map.get(code, ()), split_map.get(code, ()))
    return df.dropna(subset=["nav"]).drop_duplicates("date").set_index("date")["nav"].astype(float)


def portfolio_unit_nav(nav_panel: pd.DataFrame, weights: dict) -> pd.Series:
    """快照权重买入持有的组合单位净值（nav_panel 索引=共同日期，列=基金）。"""
    first = nav_panel.iloc[0]
    shares = {c: weights[c] / first[c] for c in nav_panel.columns}
    value = sum(nav_panel[c] * s for c, s in shares.items())
    return value / value.iloc[0]


def cdar95_of(unit_nav: pd.Series) -> float:
    return empirical_cdar(drawdown_path(list(unit_nav.to_numpy())), 0.95)


def window_metrics(unit_nav: pd.Series, rf: float) -> dict:
    df = unit_nav.reset_index()
    df.columns = ["date", "nav"]
    m = compute_metrics(df, rf=rf, min_days=20)
    if m is None:
        return None
    m["cdar95"] = cdar95_of(unit_nav)
    return m


def fmt_pct(x, digits=2):
    return "NA" if x is None or x != x else f"{x:.{digits}%}"


def main():
    global AUDIT_DATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", default=None, help="分析截止日 YYYY-MM-DD；默认全部持仓共同最新净值日")
    args = parser.parse_args()
    os.makedirs(REPORTS, exist_ok=True)
    snap = load_account_snapshot()
    funds = snap["funds"]
    codes = [f["code"] for f in funds]
    if args.asof:
        AUDIT_DATE = pd.Timestamp(args.asof)
    else:
        latest = [storage.last_date(c) for c in codes]
        if any(x is None for x in latest):
            missing = [c for c, x in zip(codes, latest) if x is None]
            raise SystemExit("DATA_BLOCKED: 缺少净值文件 " + ",".join(missing))
        AUDIT_DATE = pd.Timestamp(min(latest)).normalize()
    total = sum(float(f["value_cny"]) for f in funds)
    weights = {f["code"]: float(f["value_cny"]) / total for f in funds}
    names = {f["code"]: f.get("name", "") for f in funds}
    cats = {f["code"]: f["category"] for f in funds}
    div_map, split_map = storage.load_dividends(), storage.load_splits()

    series = {c: total_return_series(c, div_map, split_map) for c in codes}

    factors = pd.read_csv("data/factors/daily.csv", parse_dates=["date"])
    rf_rows = factors.loc[(factors["date"] <= AUDIT_DATE) &
                          factors["risk_free_annual"].notna()]
    if rf_rows.empty:
        raise SystemExit(f"DATA_BLOCKED: {AUDIT_DATE.date()} 之前缺少无风险利率因子")
    rf_latest = float(rf_rows.iloc[-1]["risk_free_annual"])

    windows = {
        "近1年": (AUDIT_DATE - pd.Timedelta(days=365), AUDIT_DATE),
        "近2年": (AUDIT_DATE - pd.Timedelta(days=730), AUDIT_DATE),
        "当前风格区间(2025-08-29起)": (STYLE_SWITCH_710301, AUDIT_DATE),
    }

    out = {"audit_date": str(AUDIT_DATE.date()), "rf_latest": rf_latest,
           "weights": weights, "windows": {}}
    L = ["# PORTFOLIO_RISK_REPORT — 全账户风险分析", "",
         f"数据截止: {AUDIT_DATE.date()}｜组合: 13 只基金，快照权重（{snap['snapshot_date']}）"
         f"买入持有口径｜无风险利率: {rf_latest:.2%}（中债1Y国债最新）", "",
         "## 1. 当前组合权重", "",
         "| 代码 | 名称 | 类别 | 权重 |", "|---|---|---|---|"]
    for c in codes:
        L.append(f"| {c} | {names[c]} | {cats[c]} | {weights[c]:.2%} |")
    L.append(f"| cash | 每日新增预算 {snap['new_cash_budget_cny_per_execution_day']:.0f} 元（增量资金，"
             f"不计入存量权重） | cash | - |")

    panels, port_metrics = {}, {}
    for wname, (s, e) in windows.items():
        panel = pd.DataFrame({c: series[c] for c in codes})
        panel = panel[(panel.index >= s) & (panel.index <= e)].dropna()
        if len(panel) < 20:
            continue
        panels[wname] = panel
        pu = portfolio_unit_nav(panel, weights)
        pm = window_metrics(pu, rf_latest)
        port_metrics[wname] = pm
        # 单基金指标
        rows = []
        for c in codes:
            fm = window_metrics(panel[c] / panel[c].iloc[0], rf_latest)
            fm["code"] = c
            rows.append(fm)
        out["windows"][wname] = {"start": str(panel.index[0].date()),
                                 "end": str(panel.index[-1].date()),
                                 "common_days": len(panel),
                                 "portfolio": pm, "funds": rows}
        L += ["", f"## 2.{list(windows).index(wname) + 1} {wname} "
              f"（{panel.index[0].date()} ~ {panel.index[-1].date()}，共同交易日 {len(panel)} 天）", "",
              "| 代码 | 区间收益 | 年化 | 年化波动 | Sharpe | Sortino | MDD | CDaR95 | Calmar |",
              "|---|---|---|---|---|---|---|---|---|"]
        for fm in rows:
            L.append(f"| {fm['code']} | {fmt_pct(fm['total_return'])} | {fmt_pct(fm['ann_return'])} "
                     f"| {fmt_pct(fm['ann_vol'])} | {fm['sharpe']:.2f} | {fm['sortino']:.2f} "
                     f"| {fmt_pct(fm['max_drawdown'])} | {fmt_pct(fm['cdar95'])} "
                     f"| {fmt_pct(fm['calmar'], 2) if fm['calmar']==fm['calmar'] else 'NA'} |")
        L.append(f"| **组合** | **{fmt_pct(pm['total_return'])}** | **{fmt_pct(pm['ann_return'])}** "
                 f"| **{fmt_pct(pm['ann_vol'])}** | **{pm['sharpe']:.2f}** | **{pm['sortino']:.2f}** "
                 f"| **{fmt_pct(pm['max_drawdown'])}** | **{fmt_pct(pm['cdar95'])}** | - |")

    # 相关性矩阵（近1年共同日期日收益）
    L += ["", "## 3. 相关性矩阵（近1年，日收益，共同交易日）", ""]
    w1 = panels.get("近1年")
    if w1 is not None:
        rets = w1.pct_change().dropna()
        corr = rets.corr()
        out["correlation_1y"] = corr.round(4).to_dict()
        L.append("| 基金 | " + " | ".join(codes) + " |")
        L.append("|---|" + "---|" * len(codes))
        for c in codes:
            L.append(f"| {c} | " + " | ".join(f"{corr.loc[c, x]:.2f}" for x in codes) + " |")
        # 高相关对
        pairs = []
        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                if abs(corr.loc[a, b]) > 0.8:
                    pairs.append(f"{a}~{b}: {corr.loc[a, b]:.2f}")
        L.append("")
        L.append("高相关（|ρ|>0.8）：" + ("；".join(pairs) if pairs else "无"))

    # 滚动风险（近2年组合，60日窗口）
    L += ["", "## 4. 滚动风险（组合单位净值，近2年，60交易日窗口）", ""]
    w2 = panels.get("近2年")
    if w2 is not None:
        pu2 = portfolio_unit_nav(w2, weights)
        r2 = pu2.pct_change().dropna()
        roll_vol = r2.rolling(60).std() * np.sqrt(252)
        roll_mdd = pu2.rolling(60).apply(lambda x: float(1 - np.min(x) / np.max(x)), raw=True)
        out["rolling"] = {"vol_now": float(roll_vol.iloc[-1]), "vol_max": float(roll_vol.max()),
                          "vol_median": float(roll_vol.median()),
                          "mdd60_max": float(roll_mdd.max())}
        L.append(f"- 60日年化波动：当前 {roll_vol.iloc[-1]:.2%}｜中位 {roll_vol.median():.2%}"
                 f"｜区间最大 {roll_vol.max():.2%}")
        L.append(f"- 60日窗口内最大回撤的历史最差值: {roll_mdd.max():.2%}")

    # 压力情景（历史窗口回放，当前权重买入持有）
    L += ["", "## 5. 压力情景（历史窗口真实净值回放，当前权重）", "",
          "| 情景 | 窗口 | 组合区间收益 | 期间最大回撤 |", "|---|---|---|---|"]
    stress_out = {}
    for label, (s, e) in STRESS_WINDOWS.items():
        # 只用窗口内有数据的基金，权重归一化；剔除的基金如实标注
        panel_all = pd.DataFrame({c: series[c] for c in codes})
        panel_all = panel_all[(panel_all.index >= s) & (panel_all.index <= e)]
        usable = [c for c in codes if panel_all[c].notna().sum() >= 3]
        dropped = [c for c in codes if c not in usable]
        panel = panel_all[usable].dropna()
        if len(panel) < 3:
            L.append(f"| {label} | {s}~{e} | 数据不足 | - |")
            continue
        w = {c: weights[c] / sum(weights[x] for x in usable) for c in usable}
        pu = portfolio_unit_nav(panel, w)
        mdd = float((1 - pu / pu.cummax()).max())
        tr = float(pu.iloc[-1] / pu.iloc[0] - 1)
        note = f"（剔除 {','.join(dropped)}，权重已归一）" if dropped else ""
        stress_out[label] = {"window": [s, e], "return": tr, "mdd": mdd, "excluded": dropped}
        L.append(f"| {label} | {s}~{e} | {tr:+.2%}{note} | {mdd:.2%} |")
    out["stress"] = stress_out
    L += ["", "注：2022-11 与 2024-09 窗口早于 710301 风格切换（2025-08-29 确认），"
          "彼时其含权益/转债敞口，回放结果不代表当前风格下的冲击；"
          "当前风格反事实压力见 `main.py style` 的 CURRENT_STYLE_REWEIGHTED 视图。"]

    # 债券敞口与利率敏感度
    L += ["", "## 6. 债券敞口识别（最新季报披露，lower_bound 口径）", "",
          "| 代码 | 披露期 | 股票% | 可转债% | 信用债% | 国债% | 政金债% | 二级资本债% | 永续债% |",
          "|---|---|---|---|---|---|---|---|---|"]
    bond_exposure = {}
    for c in codes:
        if cats[c] != "bond":
            continue
        path = f"data/disclosures/{c}.csv"
        if not os.path.exists(path):
            L.append(f"| {c} | 无披露 | - | - | - | - | - | - | - |")
            continue
        d = pd.read_csv(path)
        r = d[d["report_asof"] == d["report_asof"].max()].iloc[0]
        g = lambda col: None if pd.isna(r.get(col)) else float(r[col])
        bond_exposure[c] = {"report_asof": r["report_asof"],
                            **{k: g(k2) for k, k2 in [
                                ("equity", "equity_nav_pct"),
                                ("convertible", "convertible_nav_pct_lower_bound"),
                                ("credit", "credit_bond_nav_pct_lower_bound"),
                                ("gov", "gov_bond_nav_pct_lower_bound"),
                                ("policy_bank", "policy_bank_nav_pct_lower_bound"),
                                ("tier2", "bank_tier2_nav_pct_lower_bound"),
                                ("perpetual", "bank_perpetual_nav_pct_lower_bound")]}}
        eq_s = "NA" if g("equity_nav_pct") is None else f"{g('equity_nav_pct'):.1f}"
        cb_s = f"{g('convertible_nav_pct_lower_bound') or 0:.1f}"
        cr_s = f"{g('credit_bond_nav_pct_lower_bound') or 0:.1f}"
        gv_s = f"{g('gov_bond_nav_pct_lower_bound') or 0:.1f}"
        pb_s = f"{g('policy_bank_nav_pct_lower_bound') or 0:.1f}"
        t2_s = f"{g('bank_tier2_nav_pct_lower_bound') or 0:.1f}"
        pp_s = f"{g('bank_perpetual_nav_pct_lower_bound') or 0:.1f}"
        L.append(f"| {c} | {r['report_asof']} | {eq_s} | {cb_s} | {cr_s} "
                 f"| {gv_s} | {pb_s} | {t2_s} | {pp_s} |")
    out["bond_exposure"] = bond_exposure
    L += ["", "### 隐含久期（当前风格区间日收益对5Y国债收益率变动回归）", "",
          "| 代码 | 样本天数 | beta(收益/Δy) | 隐含久期(年) | R² |", "|---|---|---|---|---|"]
    fy = factors.set_index("date")["gov_yield_5y"].dropna()
    dy = fy.diff()
    dur_out = {}
    for c in codes:
        if cats[c] != "bond":
            continue
        s = series[c]
        s = s[s.index >= STYLE_SWITCH_710301]
        r = s.pct_change().dropna()
        j = pd.concat([r, dy], axis=1, keys=["r", "dy"]).dropna()
        if len(j) < 60:
            L.append(f"| {c} | {len(j)} | 样本不足 | - | - |")
            continue
        beta = float(np.polyfit(j["dy"], j["r"], 1)[0])
        r2 = float(np.corrcoef(j["dy"], j["r"])[0, 1] ** 2)
        # dy 单位为百分点，ret ≈ -D × Δy_decimal = -(D/100) × Δy_pp → D = -beta × 100
        dur_out[c] = {"n": len(j), "beta": beta, "implied_duration": -beta * 100, "r2": r2}
        L.append(f"| {c} | {len(j)} | {beta:.2f} | {-beta * 100:.1f} | {r2:.3f} |")
    out["implied_duration"] = dur_out
    L += ["", "注：日频债基收益以票息 carry 为主，回归 R² 普遍很低，隐含久期仅为量级参考；"
          "久期/凸性精确值需要持仓债券明细（MISSING，季报仅前五）。",
          "信用利差冲击无法参数化（因子 MISSING），信用敞口以披露 lower_bound 表达；"
          "二级资本债+永续债属于信用敏感品种，710301 合计 lower_bound 15%+。", ""]

    # XIRR / TWR 口径声明
    L += ["## 7. TWR 与 XIRR 口径声明", "",
          "- 组合单位净值（本报告）：买入持有、无外部现金流，MDD 不受入金摊薄影响；",
          "- 外部现金流单位化（TWR）：`fundkit.finance_math.external_cash_unitization`，"
          "在 `main.py sim` 的模拟批次口径下逐日应用；",
          "- XIRR：真实账户申购流水不可得（仅快照金额），**真实 XIRR 不可验证"
          "（ACCOUNT_DRAWDOWN_UNVERIFIABLE）**；模拟器对模拟批次输出 XIRR，"
          "快照日至今样本过短（<10 个披露日），不具统计意义，本报告不引用。", ""]
    L += ["---", "状态: RESEARCH_ONLY。随机性：无（全部确定性计算）。"]

    with open(f"{REPORTS}/portfolio_risk.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    open(f"{REPORTS}/PORTFOLIO_RISK_REPORT.md", "w", encoding="utf-8").write("\n".join(L))
    print("完成: reports/PORTFOLIO_RISK_REPORT.md, reports/portfolio_risk.json")


if __name__ == "__main__":
    main()
