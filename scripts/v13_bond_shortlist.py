#!/usr/bin/env python3
"""P1/P2: 2026-10-30 冲击以来债基抗跌分组筛选与候选池尽调落地。

输入: reports/bond_resilience_20261030_latest.csv（同日期精确对比榜单）
     + 本地 NAV 缓存补充 63/126 日指标、波动率
     + data/fund_status.csv 交易状态/限额
输出: reports/v13_bond_shortlist.csv, reports/V13_BOND_RESILIENCE.md
原则: 分组同类比较 + 多指标 Pareto；不用单一综合分；不补造历史。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fundkit import metrics, storage  # noqa: E402

END = "2026-11-12"
BASELINE = "2026-10-29"
RANK_CSV = "reports/bond_resilience_20261030_latest.csv"
OUT_CSV = "reports/v13_bond_shortlist.csv"
OUT_MD = "reports/V13_BOND_RESILIENCE.md"
WATCHLIST = ["021559", "004400", "710301", "014846"]
LOCK_PAT = re.compile(r"定开|封闭|个月持有|天持有|持有期|滚动持有|年持有")
CB_PAT = re.compile(r"转债|可转换")

GROUP_MAP = {
    "债券型-中短债": "短债/中短债",
    "债券型-长债": "中长纯债",
    "债券型-利率债": "利率债",
    "债券型-信用债": "信用债",
    "债券型-混合一级": "一级债基",
    "债券型-混合二级": "二级债基",
}


def group_of(row) -> str:
    if CB_PAT.search(str(row["name"])):
        return "可转债敞口"
    return GROUP_MAP.get(str(row["fund_type"]), "其他/待核验")


def share_class(name: str) -> str:
    m = re.search(r"([A-E])$", str(name))
    return m.group(1) if m else "A"


def base_name(name: str) -> str:
    return re.sub(r"[A-E]$", "", str(name))


def pareto_mask(df: pd.DataFrame, ups: list[str], downs: list[str]) -> np.ndarray:
    """简单非支配前沿：返回不被任何其他行在所有维度上占优的掩码。"""
    vals = df[ups + downs].to_numpy(dtype=float)
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            up_dom = all(vals[j][k] >= vals[i][k] for k in range(len(ups)))
            dn_dom = all(vals[j][len(ups) + k] <= vals[i][len(ups) + k] for k in range(len(downs)))
            strict = any(vals[j][k] > vals[i][k] for k in range(len(ups))) or \
                any(vals[j][len(ups) + k] < vals[i][len(ups) + k] for k in range(len(downs)))
            if up_dom and dn_dom and strict:
                keep[i] = False
                break
    return keep


def enrich_window_metrics(codes: list[str]) -> pd.DataFrame:
    """补 63/126 日收益、63/126/252 日回撤、252 日年化波动（仅用 <=END 已公布净值）。"""
    divs = storage.load_dividends()
    splits = storage.load_splits()
    rows = {}
    end = pd.Timestamp(END)
    for code in codes:
        try:
            hist = storage.load_nav(code)
            hist = metrics.adjust_nav(hist, divs.get(code, ()), splits.get(code, ()))
            hist = hist.sort_values("date")
            hist = hist[hist["date"] <= end]
            r = {"code": code}
            for win in (63, 126, 252):
                w = hist.tail(win + 1)
                if len(w) >= max(20, int(win * 0.75)):
                    r[f"ret_{win}d"] = float(w["nav"].iloc[-1] / w["nav"].iloc[0] - 1.0)
                    r[f"mdd_{win}d"] = float(metrics.compute_metrics(w, min_days=5)["max_drawdown"])
                else:
                    r[f"ret_{win}d"] = None
                    r[f"mdd_{win}d"] = None
            if len(hist) >= 190:
                daily = hist["nav"].pct_change().dropna().tail(252)
                r["vol_252d_ann"] = float(daily.std() * np.sqrt(252))
            else:
                r["vol_252d_ann"] = None
            rows[code] = r
        except Exception:
            rows[code] = {"code": code}
    return pd.DataFrame(list(rows.values()))


def main() -> None:
    df = pd.read_csv(RANK_CSV, dtype={"code": str})
    status = storage.load_trade_status()
    status["code"] = status["code"].astype(str).str.zfill(6)
    df = df.merge(status, on="code", how="left")
    df["group"] = df.apply(group_of, axis=1)
    df["share"] = df["name"].map(share_class)
    df["base_name"] = df["name"].map(base_name)
    df["has_lock"] = df["name"].astype(str).str.contains(LOCK_PAT)
    df["open_sub"] = df["sub_status"].astype(str).str.contains("开放")
    df["daily_limit_ok"] = df["daily_limit"].fillna(0) >= 1000.0  # 低于1000元/日视为实质限购

    # 预警/排除标记（不静默剔除，全部留痕）
    def flags(r):
        f = []
        if not r["open_sub"]:
            f.append("暂停/限制申购")
        if not r["daily_limit_ok"]:
            f.append(f"日限额{int(r['daily_limit']) if pd.notna(r['daily_limit']) else 0}元")
        if r["has_lock"]:
            f.append("持有期/定开(流动性受限)")
        if (r["one_year_observations"] or 0) < 190:
            f.append("近一年样本不足")
        if pd.isna(r.get("shock_day_return")):
            f.append("缺10-30单日数据")
        return ";".join(f) if f else ""
    df["warnings"] = df.apply(flags, axis=1)

    # 候选准入：可申购、限额可接受、无锁定、一年样本足够、近一年非负收益（短期抗跌但长期亏损者不留榜，仅留痕）
    elig = df[(df["open_sub"]) & (df["daily_limit_ok"]) & (~df["has_lock"]) &
              (df["one_year_observations"] >= 190) &
              (df["one_year_total_return"] > 0)].copy()

    # A/C 去重：同一 base_name 保留 A 份额（无 A 则取历史最长者）
    elig = elig.sort_values(["share", "one_year_observations"], ascending=[True, False])
    elig = elig.drop_duplicates("base_name", keep="first")

    extra = enrich_window_metrics(elig["code"].tolist() + WATCHLIST)
    elig = elig.merge(extra, on="code", how="left")
    df = df.merge(extra, on="code", how="left")

    # 分组 Pareto（冲击期收益↑、冲击期回撤↓、近一年收益↑、近一年回撤↓），每组取前沿+次优共5~10只
    picks = []
    for g, sub in elig.groupby("group"):
        sub = sub.dropna(subset=["return_since_shock", "max_drawdown_since_shock",
                                 "one_year_total_return", "one_year_mdd"])
        if sub.empty:
            continue
        mask = pareto_mask(sub, ["return_since_shock", "one_year_total_return"],
                           ["max_drawdown_since_shock", "one_year_mdd"])
        front = sub[mask]
        # 前沿内按冲击期回撤升序取前 10
        take = front.sort_values(["max_drawdown_since_shock", "return_since_shock"],
                                 ascending=[True, False]).head(10)
        picks.append(take)
    short = pd.concat(picks).sort_values(["group", "max_drawdown_since_shock"]) if picks else pd.DataFrame()

    cols = ["code", "name", "group", "fund_type", "shock_day_return", "return_since_shock",
            "max_drawdown_since_shock", "ret_63d", "mdd_63d", "ret_126d", "mdd_126d",
            "one_year_total_return", "one_year_mdd", "vol_252d_ann",
            "sub_status", "red_status", "daily_limit", "warnings",
            "baseline_nav_date", "end_nav_date", "days_since_shock"]
    short[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 观察名单 4 只的全市场名次（无论是否入选）
    # 全市场名次先在全量榜上计算，再取观察名单子集
    df["overall_rank"] = df["return_since_shock"].rank(ascending=False, method="min").astype(int)
    watch = df[df["code"].isin(WATCHLIST)].copy()

    audit = {
        "baseline": BASELINE, "end": END,
        "total_ranked": int(len(df)), "eligible_after_filters": int(len(elig)),
        "shortlist": int(len(short)),
        "groups": {g: int(len(s)) for g, s in short.groupby("group")} if len(short) else {},
    }
    Path(OUT_CSV).with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# V13 债基抗跌筛选（{BASELINE} ~ {END}）", "",
             f"- 同日期精确对比债基：{len(df)} 只（全市场债券型 7448 只，其余缺基准/截止日净值）",
             f"- 通过可交易性/限额/锁定/样本过滤：{len(elig)} 只（A/C 已去重）",
             f"- Pareto 前沿入选：{len(short)} 只，分组：{audit['groups']}", "",
             "## 观察名单（持仓相关 4 只，固定展示）", ""]
    for _, r in watch.sort_values("code").iterrows():
        lines.append(
            f"- {r['code']} {r['name']}（{r['group']}）：总榜第 {r['overall_rank']}/{len(df)}；"
            f"10-30 单日 {r['shock_day_return']*100:.2f}%；冲击期累计 {r['return_since_shock']*100:.2f}%、"
            f"MDD {r['max_drawdown_since_shock']*100:.2f}%；近一年 {r['one_year_total_return']*100:.2f}%、"
            f"1Y-MDD {r['one_year_mdd']*100:.2f}%；预警：{r['warnings'] or '无'}")
    lines += ["", "## 入选名单（按组，组内按冲击期回撤升序）", ""]
    for g, sub in short.groupby("group"):
        lines.append(f"### {g}")
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['code']} {r['name']}：冲击期 {r['return_since_shock']*100:.2f}%/"
                f"MDD {r['max_drawdown_since_shock']*100:.2f}%；63日 {fmt(r.get('ret_63d'))}；"
                f"126日 {fmt(r.get('ret_126d'))}；1Y {r['one_year_total_return']*100:.2f}%/"
                f"MDD {r['one_year_mdd']*100:.2f}%；日限额 {fmt_lim(r['daily_limit'])}")
        lines.append("")
    lines += ["## 方法说明", "",
              "- 排名为分组同类比较 + 多指标 Pareto 前沿（冲击期收益↑、冲击期MDD↓、1Y收益↑、1Y-MDD↓），无单一综合分。",
              "- 短窗口只报告区间收益，不年化。",
              "- 抗跌≠无信用风险：未纳入季报持仓尽调前，所有候选仅为研究观察对象（见 P2）。",
              "- A/C 份额按基础名去重保留 A 类；C 类销售服务费会侵蚀长期收益。",
              f"- 数据截止 {END}（已公布净值），复权在切片前完成。"]
    Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")
    print("shortlist:", len(short), audit["groups"])
    print("OUT:", OUT_CSV, OUT_MD)


def fmt(x):
    return "NA" if x is None or (isinstance(x, float) and (np.isnan(x))) else f"{x*100:.2f}%"


def fmt_lim(x):
    if pd.isna(x):
        return "未知"
    return "不限" if x >= 1e9 else f"{int(x)}元/日"


if __name__ == "__main__":
    main()
