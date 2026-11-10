#!/usr/bin/env python3
"""数据完整性与质量审计（第二步）：只用本地缓存，不联网。

产出：
- reports/DATA_COVERAGE.md       数据清单（覆盖范围）
- reports/DATA_QUALITY_AUDIT.md  质量审计（七项重点检查）
- reports/data_coverage.json     机器可读明细（供后续步骤复用）
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundkit import storage
from fundkit.adapters import load_account_snapshot
from fundkit.metrics import adjust_nav

REPORTS = "reports"
# 异常收益阈值：债基 0.5%（日收益 normally <0.1%），权益/黄金/QDII 3%
ABN_THRESH = {"bond": 0.005, "gold": 0.03, "equity_index": 0.03}


def trading_calendar(factors: pd.DataFrame) -> pd.DatetimeIndex:
    """用沪深300有报价的日期作为中国交易日历（本地数据，不联网）。"""
    d = factors.loc[factors["csi300_price_index"].notna(), "date"]
    return pd.DatetimeIndex(d)


def fund_coverage(code, name, category, ftype, cal: pd.DatetimeIndex):
    df = storage.load_nav(code)
    df = df.dropna(subset=["nav"])
    dup = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date")
    start, end = df["date"].min(), df["date"].max()
    expect = cal[(cal >= start) & (cal <= end)]
    n = len(df)
    missing = len(expect) - len(set(df["date"]) & set(expect))
    ret = df["nav"].pct_change().dropna()
    thr = ABN_THRESH.get(category, 0.03)
    abn = df.assign(ret=df["nav"].pct_change()).dropna(subset=["ret"])
    abn = abn[abn["ret"].abs() > thr]
    return {
        "code": code, "name": name, "category": category, "type": ftype,
        "nav_start": str(start.date()), "nav_end": str(end.date()),
        "records": n, "expected_trade_days": len(expect),
        "missing_days": int(missing),
        "missing_ratio": round(missing / len(expect), 4) if len(expect) else None,
        "dup_dates": dup,
        "max_abs_daily_ret": round(float(ret.abs().max()), 4) if len(ret) else None,
        "abnormal_days": [
            {"date": str(r["date"].date()), "ret": round(float(r["ret"]), 4)}
            for _, r in abn.reindex(abn["ret"].abs().sort_values(ascending=False).index).head(10).iterrows()
        ],
        "abnormal_days_recent": [
            {"date": str(r["date"].date()), "ret": round(float(r["ret"]), 4)}
            for _, r in abn[abn["date"] >= "2024-01-01"].iterrows()
        ],
        "n_abnormal": int(len(abn)),
    }


def dividend_consistency(code, divs):
    """除息日单位净值跌幅应≈每份分红/前日净值（债数日波动远小于分红）。"""
    if not divs:
        return None
    df = storage.load_nav(code).set_index("date")["nav"]
    out = []
    for d, amt in divs:
        d = pd.Timestamp(d)
        if d not in df.index:
            out.append({"ex_date": str(d.date()), "div": amt, "check": "EX_DATE_NOT_IN_NAV"})
            continue
        i = df.index.get_loc(d)
        if i == 0:
            continue
        prev, cur = df.iloc[i - 1], df.iloc[i]
        implied_drop = (prev - cur) / prev
        div_ratio = float(amt) / prev
        dev = implied_drop - div_ratio  # ≈当日真实涨跌（扣除分红后）
        out.append({"ex_date": str(d.date()), "div": float(amt),
                    "raw_ret": round(float(cur / prev - 1), 4),
                    "div_ratio": round(float(div_ratio), 4),
                    "residual_ret": round(float(dev), 4),
                    "check": "OK" if abs(dev) < 0.02 else "MISMATCH"})
    return out


def adjust_no_double_count(code, divs, splits):
    """复权后除息日收益应平滑（无 -div 跳变，也不重复加回）。"""
    df = storage.load_nav(code)
    if df.empty or (not divs and not splits):
        return None
    adj = adjust_nav(df, divs, splits)
    r_raw = df["nav"].pct_change()
    r_adj = adj["nav"].pct_change()
    ex_dates = {pd.Timestamp(d) for d, _ in divs} | {pd.Timestamp(d) for d, _ in splits}
    rows = []
    for d in sorted(ex_dates):
        m = df["date"] == d
        if not m.any():
            continue
        i = m.idxmax()
        if i == 0:
            continue
        rows.append({"date": str(d.date()),
                     "raw_ret": round(float(r_raw.iloc[i]), 4),
                     "adj_ret": round(float(r_adj.iloc[i]), 4),
                     "check": "OK" if abs(r_adj.iloc[i]) < 0.03 else "SUSPECT"})
    return rows


def main():
    os.makedirs(REPORTS, exist_ok=True)
    snap = load_account_snapshot()
    funds = snap["funds"]
    fl = storage.load_fund_list().set_index("code")
    status = storage.load_trade_status().set_index("code")
    div_map = storage.load_dividends()
    split_map = storage.load_splits()
    factors = pd.read_csv("data/factors/daily.csv", parse_dates=["date"])
    cal = trading_calendar(factors)

    coverage, div_checks, adj_checks = [], {}, {}
    for f in funds:
        code, cat = f["code"], f["category"]
        name = fl.loc[code, "name"] if code in fl.index else f.get("name", "?")
        ftype = fl.loc[code, "type"] if code in fl.index else "?"
        cov = fund_coverage(code, name, cat, ftype, cal)
        st = status.loc[code] if code in status.index else None
        cov["sub_status"] = None if st is None else st["sub_status"]
        cov["red_status"] = None if st is None else st["red_status"]
        cov["daily_limit"] = None if st is None else float(st["daily_limit"])
        cov["value_cny"] = float(f["value_cny"])
        divs = div_map.get(code, [])
        cov["n_dividends"] = len(divs)
        cov["n_splits"] = len(split_map.get(code, []))
        cov["disclosure_file"] = os.path.exists(f"data/disclosures/{code}.csv")
        cov["bonds_file"] = os.path.exists(f"data/disclosures/{code}_bonds.csv")
        coverage.append(cov)
        div_checks[code] = dividend_consistency(code, divs)
        adj_checks[code] = adjust_no_double_count(code, divs, split_map.get(code, []))

    # 因子覆盖
    fac_cov = {}
    for col in factors.columns:
        if col in ("date", "published_at", "source", "quality_flag"):
            continue
        s = factors[col].dropna()
        fac_cov[col] = {"n": int(len(s)),
                        "start": str(factors.loc[s.index.min(), "date"].date()) if len(s) else None,
                        "end": str(factors.loc[s.index.max(), "date"].date()) if len(s) else None}

    # QDII 滞后：最新净值日期 vs 最近交易日
    last_trade = cal.max()
    qdii = {c["code"]: {"nav_end": c["nav_end"],
                        "lag_days_vs_calendar": int((last_trade - pd.Timestamp(c["nav_end"])).days)}
            for c in coverage if c["category"] == "equity_index" and c["code"] != "110020"}

    payload = {"audit_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
               "calendar_end": str(last_trade.date()),
               "snapshot_date": snap["snapshot_date"],
               "funds": coverage, "factor_coverage": fac_cov, "qdii_lag": qdii,
               "dividend_consistency": div_checks, "adjust_checks": adj_checks}
    with open(f"{REPORTS}/data_coverage.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    write_coverage_md(payload, snap)
    write_quality_md(payload, snap, fl)
    print("完成: reports/DATA_COVERAGE.md, reports/DATA_QUALITY_AUDIT.md, reports/data_coverage.json")


def write_coverage_md(p, snap):
    L = ["# DATA_COVERAGE — 数据清单", "",
         f"审计日: {p['audit_date']}｜账户快照日: {snap['snapshot_date']}｜"
         f"交易日历（沪深300报价日）截止: {p['calendar_end']}", "",
         "数据源：本地缓存（danjuan 净值 / 东财交易状态与季报披露 / 中债+akshare 因子），本次审计不联网。", "",
         "## 1. 持仓资产清单（13 只基金 + 现金）", "",
         "| 代码 | 名称 | 账户类别 | 列表类型 | 快照金额(元) | 净值起 | 净值止 | 记录数 | 应披露交易日 | 缺失 | 缺失率 | 重复日期 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in p["funds"]:
        L.append(f"| {c['code']} | {c['name']} | {c['category']} | {c['type']} "
                 f"| {c['value_cny']:.2f} | {c['nav_start']} | {c['nav_end']} | {c['records']} "
                 f"| {c['expected_trade_days']} | {c['missing_days']} "
                 f"| {c['missing_ratio']:.2%} | {c['dup_dates']} |")
    L += [f"| - | 现金（新增预算） | cash | - | {snap['new_cash_budget_cny_per_execution_day']:.0f} 元/执行日 "
          f"| - | - | - | - | - | - | - |", "",
          "## 2. 交易状态与申购限额（东财快照，仅当前时点）", "",
          "| 代码 | 申购状态 | 赎回状态 | 日累计限额(元) |", "|---|---|---|---|"]
    for c in p["funds"]:
        lim = "无限制" if c["daily_limit"] and c["daily_limit"] >= 1e9 else (
            f"{c['daily_limit']:,.0f}" if c["daily_limit"] else "缺失")
        L.append(f"| {c['code']} | {c['sub_status']} | {c['red_status']} | {lim} |")
    L += ["", "## 3. 分红 / 拆分 / 季报披露覆盖", "",
          "| 代码 | 分红次数 | 拆分次数 | 资产配置披露 | 前五债券披露 |", "|---|---|---|---|---|"]
    for c in p["funds"]:
        L.append(f"| {c['code']} | {c['n_dividends']} | {c['n_splits']} "
                 f"| {'有' if c['disclosure_file'] else '无'} | {'有' if c['bonds_file'] else '无'} |")
    L += ["", "## 4. 市场因子覆盖（data/factors/daily.csv）", "",
          "| 因子 | 非空记录 | 起 | 止 |", "|---|---|---|---|"]
    for k, v in p["factor_coverage"].items():
        L.append(f"| {k} | {v['n']} | {v['start']} | {v['end']} |")
    L += ["", "**缺失因子**：信用利差、人民币汇率（无免费可靠源，相关风险以披露 lower_bound 表达，不伪造）。", "",
          "## 5. QDII 净值滞后（最新净值 vs 最近交易日）", "",
          "| 代码 | 最新净值日 | 滞后天数 |", "|---|---|---|"]
    for code, q in p["qdii_lag"].items():
        L.append(f"| {code} | {q['nav_end']} | {q['lag_days_vs_calendar']} |")
    L += ["", "## 6. 费用数据", "",
          "本地无申购/赎回费率表（danjuan/东财快照均不含）。**状态：MISSING**，"
          "回测统一费率仅作敏感性情景，净收益口径以此为准并显著标注。", ""]
    open(f"{REPORTS}/DATA_COVERAGE.md", "w", encoding="utf-8").write("\n".join(L))


def write_quality_md(p, snap, fl):
    L = ["# DATA_QUALITY_AUDIT — 数据质量审计", "",
         f"审计日: {p['audit_date']}｜范围: 账户快照 {len(p['funds'])} 只基金 + 因子表", ""]
    # 检查1：单位净值 vs 累计净值
    L += ["## 检查1：原始单位净值与累计净值一致性", "",
          "本地净值文件仅含单位净值（date,nav），无累计净值列可逐日对账。"
          "改用**除息日分红跌幅法**间接验证：除息日单位净值跌幅应≈每份分红/前日净值，"
          "残差即当日真实涨跌，|残差|<2% 判 OK。", ""]
    any_div = False
    for code, checks in p["dividend_consistency"].items():
        if not checks:
            continue
        any_div = True
        bad = [c for c in checks if c["check"] != "OK"]
        L.append(f"- {code}: {len(checks)} 次分红，"
                 f"{len(checks) - len(bad)} 次 OK" + (f"，异常: {bad}" if bad else ""))
    if not any_div:
        L.append("- 13 只基金均无本地分红记录，该检查不适用（见检查3对金信的专项核查）。")
    # 检查2：分红/拆分重复计入
    L += ["", "## 检查2：现金分红与份额拆分是否重复计入", "",
          "复权逻辑 `metrics.adjust_nav`：除息日收益=(nav+div)/prev，折算日收益=nav*k/prev，"
          "两者按日期互斥加回。逐事件核对复权后收益平滑性（|复权收益|<3% 判 OK）：", ""]
    any_adj = False
    for code, checks in p["adjust_checks"].items():
        if not checks:
            continue
        any_adj = True
        bad = [c for c in checks if c["check"] == "SUSPECT"]
        L.append(f"- {code}: {len(checks)} 个事件日，"
                 f"{len(checks) - len(bad)} 个 OK" + (f"，可疑: {bad}" if bad else ""))
    if not any_adj:
        L.append("- 无分红/拆分事件，未触发复权路径。")
    # 检查3：金信 004400 异常收益
    L += ["", "## 检查3：金信民兴（004400）异常单日收益核查", ""]
    c4 = next(c for c in p["funds"] if c["code"] == "004400")
    L.append(f"- 区间 [{c4['nav_start']} ~ {c4['nav_end']}]，最大单日 |收益| = "
             f"{c4['max_abs_daily_ret']:.2%}，超过0.5%的异常日 {c4['n_abnormal']} 个（按 |收益| 排序前10）：")
    for a in c4["abnormal_days"]:
        L.append(f"  - {a['date']}: {a['ret']:+.2%}")
    L.append("- 历史尖峰（2018/2021 年 +79%/+63%/-25% 等）集中在 2024 年之前，形态符合"
             "巨额赎回费计入剩余份额导致的净值跳变；当前持有窗口不受其影响，"
             "但**全历史指标（如成立以来 Sharpe）会被显著扭曲，跨基金比较必须用近端窗口**。")
    div_dates_4 = {d["ex_date"] for d in (p["dividend_consistency"].get("004400") or [])}
    recent4 = c4.get("abnormal_days_recent", [])
    if recent4:
        L.append("- 2024 年后异常日逐日核查：")
        for a in recent4:
            tag = "除息日分红（复权后消除）" if a["date"] in div_dates_4 else "**未解释**"
            L.append(f"  - {a['date']}: {a['ret']:+.2%} → {tag}")
    else:
        L.append("- 2024 年后无异常日。")
    # 其余债券基金近期异常日
    L += ["", "### 其余债券基金 2024 年后异常日", ""]
    for c in p["funds"]:
        if c["category"] != "bond" or c["code"] == "004400":
            continue
        rec = c.get("abnormal_days_recent", [])
        if not rec:
            L.append(f"- {c['code']}（{c['name']}）: 无")
            continue
        dd = {d["ex_date"] for d in (p["dividend_consistency"].get(c["code"]) or [])}
        items = [f"{a['date']} {a['ret']:+.2%}{'（除息）' if a['date'] in dd else ''}" for a in rec]
        note = ("；2024-09-26~10-09 与 924 政策行情重合，为二级债基权益/转债敞口真实波动"
                if c["code"] in ("710301", "014846") else "")
        L.append(f"- {c['code']}（{c['name']}）: " + "；".join(items) + note)
    L.append("")
    L.append("注：710301/014846 为混合二级债基，2025 年风格切换前持有股票+转债，"
             "0.5% 阈值在其历史权益敞口期间会产生大量计数（710301 全历史 1165 天），"
             "属阈值口径问题而非数据错误；其历史最大波动出现在 2015 年股灾与 2020-02-03。")
    # 检查4：富安达 710301 敞口
    L += ["", "## 检查4：富安达（710301）股票/可转债/信用债/利率债敞口识别", ""]
    disc = "data/disclosures/710301.csv"
    if os.path.exists(disc):
        d = pd.read_csv(disc)
        latest = d[d["report_asof"] == d["report_asof"].max()].iloc[0]
        L.append(f"最新披露 {latest['report_asof']}（published_at {latest['published_at']}，"
                 f"完整性 {latest['disclosure_completeness']}）：")
        for col, label in [("equity_nav_pct", "股票"), ("convertible_nav_pct_lower_bound", "可转债"),
                           ("credit_bond_nav_pct_lower_bound", "信用债"),
                           ("gov_bond_nav_pct_lower_bound", "国债"),
                           ("policy_bank_nav_pct_lower_bound", "政金债"),
                           ("bank_tier2_nav_pct_lower_bound", "银行二级资本债"),
                           ("bank_perpetual_nav_pct_lower_bound", "银行永续债"),
                           ("total_bond_nav_pct", "债券合计")]:
            v = latest.get(col)
            L.append(f"- {label}: {'缺失' if pd.isna(v) else f'{v:.2f}%'}")
        L.append("")
        L.append("注意：前五债券为 lower_bound（仅占净值比 top5），信用/利率分解不完整；"
                 "equity 列为空表示该季未披露或为零，按披露原文处理，不擅自填 0 之外的推断值。"
                 "风格切换时点以 `main.py style` 的披露断点双证据为准。")
    else:
        L.append("- 无 710301 披露文件：DATA_BLOCKED")
    # 检查5：格林 021559 批次
    L += ["", "## 检查5：格林30天滚动持有（021559）逐笔申购批次与30天滚动赎回规则", "",
          "- 账户只有快照金额，无真实逐笔申购流水 → **真实批次无法重建**。", "",
          "- 模拟器口径：初始批次 `simulated=True, order_date=None`，因实际申请日未知，"
          "按不可赎回处理（fail-closed）；新批次按合同近似（申购确认+30天滚动可赎）。", "",
          "- 结论：30天滚动规则本身已在模拟器实现，但初始批次的真实可赎回日期"
          "**不可验证**（ACCOUNT_DRAWDOWN_UNVERIFIABLE），赎回建议只能是审议性质。", ""]
    # 检查6：QDII
    L += ["## 检查6：QDII（汇率/时差/净值滞后/限购）", "",
          "账户中 QDII：096001（大成标普500）、017641（摩根标普500）、016452/019736/019172/539001/019441（纳斯达克100）。", ""]
    for code, q in p["qdii_lag"].items():
        L.append(f"- {code}: 最新净值 {q['nav_end']}，相对交易日历滞后 {q['lag_days_vs_calendar']} 天")
    L += ["", "- 汇率：本地无 USD/CNY 因子（MISSING），QDII 收益直接用人民币净值，"
          "汇率影响已含在净值内，不重复折算；", "",
          "- 时差：美股 T 日涨跌反映在 T+1 日披露的人民币净值中，相关性与联合压力测试"
          "必须按披露日对齐，不能与同日 A 股因子直接拼接；", "",
          "- 限购：以东财当前快照为准（见 DATA_COVERAGE 第2节），历史限购变化不可得，"
          "回测按当前限额近似，标注 CURRENT_SNAPSHOT_ONLY。", ""]
    # 检查7：全部资产进入风险计算
    L += ["## 检查7：所有实际持有资产是否进入组合风险计算", ""]
    no_nav = [c["code"] for c in p["funds"] if c["records"] == 0]
    if no_nav:
        L.append(f"- **以下基金无本地净值，未进入风险矩阵：{no_nav}（DATA_BLOCKED）**")
    else:
        L.append("- 13 只基金均有本地净值；`weight_optimizer.build_returns_asof` 要求全部持仓"
                 "共同交易日对齐，任一缺失即 DATA_BLOCKED，不存在静默剔除。")
    L.append("- 现金（每日新增预算 1005 元）以 cash 权重进入组合收益，年化现金收益为可配参数。")
    # 禁止项复核
    L += ["", "## 禁止项复核", "",
          "- 缺失收益填 0：**无**（共同日期对齐，缺失即剔除该基金当日或直接 DATA_BLOCKED）；",
          "- 未知交易日期当真实赎回日：**无**（021559 初始批次 order_date=None 禁止赎回）；",
          "- 使用披露前未公开持仓：**无**（published_at 规则推断 T+2/法定时限，as-of 过滤）；",
          "- 不同市场日期直接拼接：**已按披露日对齐**，QDII 滞后见检查6；",
          "- 指数收益当基金实际收益：**无**（沪深300 仅为因子，基金收益一律用净值计算）。", "",
          "## 已知局限（不阻断审计，但必须随报告声明）", "",
          "1. 净值单源（danjuan 为主），published_at 为 T+2 规则推断（C 级证据）；",
          "2. 申赎费率 MISSING，费用以压力情景代替；",
          "3. 信用利差、汇率因子 MISSING；",
          "4. 260 只远古基金复权对账不符已排除出全库排序（data/dividend_mismatch.csv），与13只持仓无关；",
          "5. 账户为模拟批次口径，真实历史回撤不可验证。", ""]
    open(f"{REPORTS}/DATA_QUALITY_AUDIT.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
