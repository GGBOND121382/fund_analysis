#!/usr/bin/env python3
"""实战榜：M0 排序 + 锁定期过滤 + M2/M3 尾部风险否决（离线复算，不访问网络）。

证据基础（reports/all_funds/models_20261120/OOS_REPORT.md）：
- M3（日度下行）在 2026-05 危机折 −3.28% vs M0' −36.72%，尾部保护真实；
- 三模型 ΔMDD 均值方向一致为负（风险估计更保守）；
- 但作为排序替换的整体升级证据不足（OOS_UNDETERMINED）。
因此实战用法 = 保留 M0 排序，仅否决尾部风险最差十分位
（SL_daily<10 或 SD95<10，百分位语义 0-100 越高越好，见 rank_models_m123.py）。
否决 ≠ 改分：被否决基金仍留在渠道榜/研究榜中并带原因。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TRADEABLE_SGSTAT = {"开放", "限制大额"}
LOCK_RE = re.compile(r"滚动|持有|定开|封闭")
VETO_QUANTILE = 10.0  # 尾部风险最差十分位否决线
PERIODS = ("5y", "3y", "1y", "6m")
PERIOD_NAMES = {"5y": "五年", "3y": "三年", "1y": "一年", "6m": "半年"}
ROOT = Path(__file__).resolve().parents[1]

OUT_COLS = ["practical_rank", "code", "name", "fund_type", "group",
            "period_return", "annualized_return", "mu_lower",
            "D", "D95", "L_daily", "SD95", "SL_daily", "tail_flag",
            "SCORE_M0", "sgstat"]


def build_period(inp: Path, models: Path, period: str):
    cand = pd.read_csv(inp / f"top_candidates_channel_tradeable_{period}.csv",
                       dtype={"code": str})
    st = pd.read_csv(inp / f"ths_tradeable_status_{period}.csv",
                     dtype={"code": str, "buy": str, "sgstat": str})
    m1 = pd.read_csv(models / f"full_ranking_m1_{period}.csv",
                     dtype={"code": str}, low_memory=False)
    for d in (cand, st, m1):
        d["code"] = d["code"].str.zfill(6)
    m = cand.merge(st[["code", "sgstat", "buy"]], on="code", how="left")
    m = m[(m["buy"] == "1") & (m["sgstat"].isin(TRADEABLE_SGSTAT))].copy()
    m["locked"] = m["name"].astype(str).str.contains(LOCK_RE)
    m = m.merge(m1[["code", "D95", "L_daily", "SD95", "SL_daily", "mu_lower"]],
                on="code", how="left")
    veto_tail = ((m["SD95"] < VETO_QUANTILE) | (m["SL_daily"] < VETO_QUANTILE))
    veto_missing = m["SD95"].isna() | m["SL_daily"].isna()

    board = m.loc[~m["locked"] & ~veto_tail & ~veto_missing].copy()
    board = board.sort_values("SCORE_M0", ascending=False).reset_index(drop=True)
    board.insert(0, "practical_rank", range(1, len(board) + 1))
    # 榜内相对尾部标记（M2/M3 指标的实际用法）：D95 或日度下行处于榜内最差 10%
    # 只是提示层，不改变 M0 名次；绝对否决线见上（全市场分位 <10）。
    in_d95 = 100.0 - 100.0 * (board["D95"].rank(method="average") - 1) / max(len(board) - 1, 1)
    in_ld = 100.0 - 100.0 * (board["L_daily"].rank(method="average") - 1) / max(len(board) - 1, 1)
    board["tail_flag"] = np.where((in_d95 < 10) | (in_ld < 10),
                                  "榜内尾部最差10%", "")

    audit = pd.concat([
        m.loc[m["locked"]].assign(exclude_reason="锁定期(滚动/持有/定开/封闭)"),
        m.loc[~m["locked"] & veto_tail].assign(exclude_reason="尾部风险最差十分位否决(SD95或SL_daily<10)"),
        m.loc[~m["locked"] & ~veto_tail & veto_missing].assign(exclude_reason="尾部指标缺失(数据不足,不进入实战榜)"),
    ])
    return board, audit, dict(period=period, channel_tradeable=int(len(m)),
                              locked=int(m["locked"].sum()),
                              veto_tail=int((~m["locked"] & veto_tail).sum()),
                              tail_missing=int((~m["locked"] & ~veto_tail
                                                & veto_missing).sum()),
                              tail_flagged=int((board["tail_flag"] != "").sum()),
                              final=int(len(board)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "reports" / "all_funds"
                                           / "comprehensive_20261120"))
    ap.add_argument("--models", default=str(ROOT / "reports" / "all_funds"
                                            / "models_20261120"))
    ap.add_argument("--candidates", type=int, default=200)
    args = ap.parse_args()
    inp, models = Path(args.input), Path(args.models)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    md = ["# 实战榜：M0 排序 + 锁定期过滤 + M1/M2/M3 指标实战化（asof 2026-11-20）",
          "",
          "规则：同花顺可申购（buy=1 且开放/限制大额）→ 剔除滚动持有/持有期/定开/封闭",
          f"→ 否决全市场尾部风险最差十分位（SD95<{VETO_QUANTILE:.0f} 或 SL_daily<{VETO_QUANTILE:.0f}，",
          "分位语义 0-100 越高越好）→ 尾部指标缺失者不进入（不补零、不造数）→ 按 SCORE_M0 排序。",
          "",
          "M1/M2/M3 的实战用法（不改动 M0 名次）：",
          "- **M1** → `M1保守收益` 列：类内 EB 收缩后的下四分位年化估计，比名义年化更保守，做预期管理用；",
          "- **M2** → `D95` 列 + 尾部标记：Bootstrap 情景 95% 尾部回撤；",
          "- **M3** → `日度下行` 列 + 尾部标记：日度下行半标准差；",
          "- `尾部标记` = D95 或日度下行处于**榜内**最差 10%（相对提示层，不改名次）。",
          "",
          "证据基础：P2 样本外验证中 M3 尾部保护真实（2026-05 危机折 −3.3% vs M0' −36.7%），",
          "但排序级整体升级证据不足（OOS_UNDETERMINED），故只作信息层/否决层。",
          "被排除基金及原因见 `practical_excluded_{期限}.csv`。",
          ""]
    summary = {}
    for p in PERIODS:
        board, audit, info = build_period(inp, models, p)
        board[OUT_COLS].to_csv(inp / f"practical_board_{p}.csv", index=False)
        audit.to_csv(inp / f"practical_excluded_{p}.csv", index=False)
        summary[p] = info
        md.append(f"## {PERIOD_NAMES[p]}期（{info['final']} 只；渠道可交易 {info['channel_tradeable']} "
                  f"− 锁定期 {info['locked']} − 尾部否决 {info['veto_tail']} "
                  f"− 尾部指标缺失 {info['tail_missing']}；榜内尾部标记 {info['tail_flagged']}）")
        md.append("")
        md.append("| 名次 | 代码 | 名称 | 年化收益 | M1保守收益 | 最大回撤 | D95 | 日度下行 | 尾部标记 | 总分 |")
        md.append("|--:|---|---|---:|---:|---:|---:|---:|---|---:|")
        for _, r in board.head(50).iterrows():
            flag = r["tail_flag"] if r["tail_flag"] else ""
            md.append(f"| {r['practical_rank']} | {r['code']} | {r['name']} "
                      f"| {r['annualized_return']*100:.2f}% | {r['mu_lower']*100:.2f}% "
                      f"| {r['D']*100:.2f}% | {r['D95']*100:.2f}% "
                      f"| {r['L_daily']*100:.2f}% | {flag} | **{r['SCORE_M0']:.2f}** |")
        md.append("")
    (inp / "practical_boards.md").write_text("\n".join(md), encoding="utf-8")
    run = dict(git_sha=git_sha, veto_quantile=VETO_QUANTILE,
               evidence="reports/all_funds/models_20261120/OOS_REPORT.md",
               periods=summary)
    (inp / "practical_boards_audit.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
