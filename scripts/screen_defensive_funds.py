#!/usr/bin/env python3
"""Build same-category defensive fund rankings without liquidity exclusions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundkit.defensive_screen import (  # noqa: E402
    LOCK, Shock, adjusted_asof, calculate_fund, category, market_category, money_index,
    rank_candidates, share_identity,
)

WATCH = ("025368", "019851", "021559", "004400", "710301", "014846")


def read_csv(path):
    d = pd.read_csv(path, dtype={"code": str})
    if "code" in d:
        d["code"] = d.code.str.zfill(6)
    return d


def optional_csv(path):
    return read_csv(path) if path.is_file() else pd.DataFrame()


def channel_purchase_status(value):
    """Describe one sales channel's purchase label; never infer global investability."""
    if value is None or pd.isna(value) or not str(value).strip():
        return "未查询到该渠道状态"
    label = str(value).strip()
    mapping = {
        "开放申购": "该渠道开放申购",
        "限大额": "该渠道限额申购",
        "暂停申购": "该渠道暂停申购",
        "暂停交易": "该渠道显示暂停交易",
        "封闭期": "该渠道显示封闭期",
        "定期开放": "该渠道显示定期开放",
    }
    return mapping.get(label, f"该渠道状态待核实：{label}")


def screen(data_dir, asof, events, top, min_peers, *, universe="defensive"):
    funds = read_csv(data_dir / "fund_list.csv").drop_duplicates("code")
    status = optional_csv(data_dir / "fund_status.csv")
    status = status.drop_duplicates("code").set_index("code").to_dict("index") if len(status) else {}
    div = optional_csv(data_dir / "dividends.csv")
    div = {c: list(zip(s.ex_date, s["div"])) for c, s in div.groupby("code")} if len(div) else {}
    split_file = data_dir / "splits.json"
    splits = json.loads(split_file.read_text(encoding="utf-8")) if split_file.exists() else {}
    rows = []
    for fund in funds.itertuples(index=False):
        code, name = fund.code, str(fund.name)
        if universe == "all":
            group, family, currency, basis = market_category(name, fund.type)
            selected = [s for s in events if not s.families or family in s.families]
        else:
            group = category(name, fund.type)
            family, currency, basis, selected = "domestic", "CNY", "fund_nav", events
        if group is None:
            continue
        base, share = share_identity(name)
        channel = status.get(code, {})
        raw_sub_status = channel.get("sub_status")
        channel_source = channel.get("status_source")
        if channel_source is None or pd.isna(channel_source) or not str(channel_source).strip():
            channel_source = "来源未记录" if channel else "未查询到渠道记录"
        channel_checked_at = channel.get("status_checked_at")
        if channel_checked_at is None or pd.isna(channel_checked_at) or not str(channel_checked_at).strip():
            channel_checked_at = "未知"
        row = {"code": code, "name": name, "fund_type": fund.type, "group": group,
               "product": base, "share": share, "has_lock": bool(LOCK.search(name)),
               "scenario_family": family, "currency": currency, "pricing_basis": basis,
               "required_shocks": len(selected), "shock_set": ",".join(s.id for s in selected),
               **{k: channel.get(k) for k in ("sub_status", "red_status", "daily_limit")},
               "channel_purchase_status": channel_purchase_status(raw_sub_status),
               "channel_status_source": channel_source,
               "channel_status_checked_at": channel_checked_at,
               "selection_status": "", "data_issue": "", "asof_exact": False,
               "horizon_tier": "", "shock_observations": 0, "nav_observations": 0}
        folder = {"income_per_10000": "money", "adjusted_close": "listed", "fund_nav": "nav"}[basis]
        # Dedicated market-total-return cache is built from the official style
        # vendor NAV plus audited event adjustments. Existing NAV is a fallback.
        adj = data_dir / "total_return" / f"{code}.csv"
        already_adjusted = universe == "all" and basis == "fund_nav" and adj.is_file()
        p = adj if already_adjusted else data_dir / folder / f"{code}.csv"
        if not p.is_file():
            row["selection_status"] = "无净值缓存"
        else:
            try:
                raw = read_csv(p)
                if basis == "income_per_10000":
                    series = money_index(raw, asof)
                elif already_adjusted or basis == "adjusted_close":
                    series = adjusted_asof(raw, asof)
                else:
                    series = adjusted_asof(raw, asof, div.get(code, ()), splits.get(code, ()))
                row.update(calculate_fund(series, asof, selected))
                row["nav_audit"] = ("万份收益复利" if basis == "income_per_10000" else
                                    "供应商后复权收盘价" if basis == "adjusted_close" else
                                    "采集器已复权，方法见缓存" if already_adjusted else
                                    "有累计净值交叉核验" if (
                    "acc_nav" in raw and pd.to_numeric(raw.loc[
                        pd.to_datetime(raw.date).le(asof), "acc_nav"], errors="coerce"
                    ).notna().all()
                ) else "累计净值不全，复权依赖事件缓存")
                if not row["asof_exact"]:
                    row["selection_status"] = "缺截止日净值"
                elif len(selected) < 2:
                    row["selection_status"] = "同类冲击窗口不足"
            except (ValueError, OSError, TypeError, KeyError) as e:
                row["selection_status"] = "净值需核验"
                row["data_issue"] = str(e)
        rows.append(row)
    if not rows:
        raise ValueError("基金清单没有筛选范围内的品种")
    return rank_candidates(pd.DataFrame(rows), shock_count=len(events), top=top, min_peers=min_peers)


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def write_reports(out, full, short, events, end, watch_codes):
    out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "universe.csv", index=False, encoding="utf-8-sig")
    short.to_csv(out / "shortlist.csv", index=False, encoding="utf-8-sig")
    channel_columns = [
        "code", "name", "fund_type", "group", "has_lock", "sub_status", "red_status",
        "daily_limit", "channel_purchase_status", "channel_status_source",
        "channel_status_checked_at", "selection_status",
    ]
    full[[c for c in channel_columns if c in full.columns]].to_csv(
        out / "purchase_channel_status.csv", index=False, encoding="utf-8-sig"
    )
    if "peer_rank" not in full:
        full = full.assign(peer_rank=float("nan"))
    ranked = full.loc[full.peer_rank.notna()].sort_values(["group", "horizon_tier", "peer_rank"])
    ranked.to_csv(out / "ranking.csv", index=False, encoding="utf-8-sig")
    full.loc[full.peer_rank.isna()].to_csv(out / "unranked.csv", index=False, encoding="utf-8-sig")
    watch = (full.set_index("code").reindex(watch_codes).rename_axis("code")
             .reset_index())
    watch["selection_status"] = watch.selection_status.fillna("清单中无该代码")
    watch.to_csv(out / "watchlist.csv", index=False, encoding="utf-8-sig")
    by_type = (full.assign(fund_type=full.fund_type.fillna("类型待核验"))
               .groupby("fund_type").agg(share_codes=("code", "size"),
                                         exact_date=("asof_exact", "sum"),
                                         ranked_products=("peer_rank", "count")))
    by_type.to_csv(out / "type_coverage.csv", encoding="utf-8-sig")
    audit = {
        "end": end.date().isoformat(),
        "universe_share_codes": len(full),
        "funds_with_local_nav_file": int(full.selection_status.ne("无净值缓存").sum()),
        "exact_asof_share_codes": int(full.asof_exact.sum()),
        "fully_scored_representatives": int(full.peer_rank.notna().sum()) if "peer_rank" in full else 0,
        "shortlist": len(short),
        "by_status": dict(Counter(full.selection_status)),
        "by_group": full.groupby("group").selection_status.value_counts().unstack(fill_value=0)
                        .to_dict(orient="index"),
        "by_pricing_basis": full.groupby("pricing_basis").selection_status.value_counts()
                                .unstack(fill_value=0).to_dict(orient="index"),
        "purchase_channel_status_counts": {str(k): int(v) for k, v in full.channel_purchase_status.value_counts(dropna=False).items()},
        "purchase_channel_status_sources": sorted(full.channel_status_source.dropna().astype(str).unique().tolist()),
        "purchase_channel_status_checked_at": sorted(full.channel_status_checked_at.dropna().astype(str).unique().tolist()),
        "purchase_channel_status_is_ranking_filter": False,
        "purchase_channel_status_note": "单一销售渠道快照，仅作参考；不代表全渠道可购买状态，不参与评分或入榜筛选。赎回状态单独记录，不作为申购条件。",
        "shocks": [dict(id=s.id, label=s.label, baseline=s.baseline.date().isoformat(),
                        end=s.end.date().isoformat(), source=s.source, families=s.families) for s in events],
    }
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    channel_summary = "；".join(
        f"{k}: {v}" for k, v in audit["purchase_channel_status_counts"].items()
    )
    lines = [f"# 稳健基金分组筛选（{audit['end']}）", "",
             f"目录内范围 {len(full)} 个份额代码，本地有净值文件 "
             f"{audit['funds_with_local_nav_file']} 个，截止日净值精确匹配 "
             f"{audit['exact_asof_share_codes']} 个，实际评分 "
             f"{audit['fully_scored_representatives']} 只代表份额，入围 {len(short)} 只。",
             "覆盖不足时，名次只代表有数据的同类样本，不代表全市场。",
             "申购状态仅是天天基金渠道快照，不代表其他销售平台；不参与评分、排序或剔除。赎回状态单独列示，不作为能否申购的条件。",
             f"渠道申购状态分布：{channel_summary}",
             "另见 purchase_channel_status.csv，包含渠道来源、核验时间、申购状态和赎回状态。", "",
             "| 代码 | 名称 | 结果 | 组内名次 | 6月收益/回撤 | 1年收益/回撤 | "
             "2年年化/回撤 | 3年年化/回撤 | 最差冲击收益/回撤 | 有持有期 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in watch.to_dict("records"):
        val = lambda key: pct(r.get(key))  # noqa: E731
        locked = r.get("has_lock")
        peer_rank = r.get("peer_rank")
        rank = f"{int(peer_rank)}/{int(r['peer_count'])}" if pd.notna(peer_rank) else "—"
        lines.append(f"| {r['code']} | {r['name'] if pd.notna(r['name']) else '—'} | "
                     f"{r['selection_status']} | {rank} | {val('return_6m')}/{val('mdd_6m')} | "
                     f"{val('return_1y')}/{val('mdd_1y')} | {val('cagr_2y')}/{val('mdd_2y')} | "
                     f"{val('cagr_3y')}/{val('mdd_3y')} | "
                     f"{val('worst_shock_return')}/{val('worst_shock_mdd')} | "
                     f"{'—' if pd.isna(locked) else '是' if locked else '否'} |")
    lines += ["", "## 冲击窗口", ""] + [
        f"- {s.label}：{s.baseline.date()} → {s.end.date()}  [行情来源]({s.source})"
        for s in events
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("by_group", "shocks")},
                     ensure_ascii=False))
    return audit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "defensive_funds")
    p.add_argument("--asof", required=True)
    p.add_argument("--universe", choices=("defensive", "all"), default="defensive")
    p.add_argument("--shocks", type=Path, default=ROOT / "config" / "defensive_fund_shocks.json")
    p.add_argument("--top", type=int, default=10, help="每个类型×历史档的入围上限")
    p.add_argument("--min-peers", type=int, default=3, help="同类同历史档最少可比基金数")
    p.add_argument("--watch", default=",".join(WATCH))
    a = p.parse_args(argv)
    end = pd.Timestamp(a.asof).normalize()
    raw = json.loads(a.shocks.read_text(encoding="utf-8"))
    events = [Shock(r["id"], r["label"], pd.Timestamp(r["baseline"]).normalize(),
                    pd.Timestamp(r["end"]).normalize(), r.get("source", ""),
                    tuple(r.get("families", ()))) for r in raw]
    events = [s for s in events if s.end <= end]
    if len({s.id for s in events}) != len(events) or len(events) < 2:
        p.error("截至指定日期至少需要两个不同的完整冲击窗口")
    full, short = screen(a.data_dir, end, events, a.top, a.min_peers, universe=a.universe)
    codes = list(dict.fromkeys(c.strip().zfill(6) for c in a.watch.split(",") if c.strip()))
    write_reports(a.output, full, short, events, end, codes)


if __name__ == "__main__":
    main()
