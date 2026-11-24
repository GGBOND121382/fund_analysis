#!/usr/bin/env python3
"""从同花顺核查缓存（ths_tradeable_status_*.csv）+ 渠道可交易候选序列生成
同花顺可交易榜：top10_ths_tradeable_{p}.csv、完整榜 ths_tradeable_board_{p}.csv
和可读汇总 ths_tradeable_boards.md。不访问网络，纯离线复算。

P0-T2：输出同时包含 period_return（窗口累计收益）与 annualized_return（年化收益，
等于主榜 R 列），展示表明确区分两口径。
"""
import argparse
import re
from pathlib import Path

import pandas as pd

TRADEABLE_SGSTAT = {"开放", "限制大额"}  # 与 scripts/check_ths_tradeable.py 保持一致
# 名称层面的锁定识别：滚动持有 / 持有期 / 定开 / 封闭 / "N天(个月)持有(滚动)"
# 此类基金申购后存在锁定期，不能随时赎回；渠道状态为"开放"不等于自由申赎。
LOCK_RE = re.compile(r"滚动|持有|定开|封闭")

PERIODS = ("5y", "3y", "1y", "6m")
PERIOD_NAMES = {"5y": "五年", "3y": "三年", "1y": "一年", "6m": "半年"}
OUT_COLS = ["ths_rank", "code", "name", "fund_type", "period_return", "annualized_return",
            "D", "L", "Aret", "Add", "SR", "SD", "SL", "SA", "SCORE_M0", "sgstat",
            "lock_status"]


def build_board(cand: pd.DataFrame, status: pd.DataFrame, top: int):
    cand = cand.copy()
    status = status.copy()
    cand["code"] = cand["code"].str.zfill(6)
    status["code"] = status["code"].str.zfill(6)
    status["buy"] = status["buy"].astype(str)
    if "annualized_return" not in cand.columns and "R" in cand.columns:
        cand["annualized_return"] = cand["R"]
    m = cand.merge(status[["code", "sgstat", "buy"]], on="code", how="left")
    ok = m[(m["buy"] == "1") & (m["sgstat"].isin(TRADEABLE_SGSTAT))].copy()
    ok["lock_status"] = ["锁定期(滚动/持有/定开/封闭)" if LOCK_RE.search(str(n))
                         else "自由申赎" for n in ok["name"]]
    ok.insert(0, "ths_rank", range(1, len(ok) + 1))
    return ok.head(top), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/all_funds/comprehensive_20261120")
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    inp = Path(args.input)
    md = ["# M0 综合评分 · 同花顺可交易完整榜单（asof 2026-11-20，交易状态核查 2026-11-24）",
          "",
          "口径：M0 综合评分（0.40·SR+0.35·SD+0.15·SA+0.10·SL）排序 → 剔除天天基金渠道不可申购"
          "/限额<500元 → 剔除同花顺爱基金 buy=0（暂不支持交易/未代销）与 buy=2（停售）。",
          "收益双口径：`period_return`=窗口累计收益，`annualized_return`=按实际时间跨度年化（=主榜 R 列）。",
          ""]
    counts = {}
    open_counts = {}
    open_md = ["# M0 综合评分 · 自由申赎榜（asof 2026-11-20，交易状态核查 2026-11-24）",
               "",
               "在 `ths_tradeable_board_*.csv`（同花顺可申购）基础上，进一步剔除名称含"
               "“滚动/持有/定开/封闭”的锁定期基金——它们申购后在锁定期内不可赎回。",
               "完整渠道榜（含锁定期标记）见 ths_tradeable_boards.md；锁定识别为名称层面，"
               "个别产品合同细则未逐一核验，临界情形以基金公司公告为准。",
               ""]
    for p in PERIODS:
        cand = pd.read_csv(inp / f"top_candidates_channel_tradeable_{p}.csv", dtype={"code": str})
        st = pd.read_csv(inp / f"ths_tradeable_status_{p}.csv",
                         dtype={"code": str, "buy": str, "sgstat": str})
        top_df, board = build_board(cand.head(args.candidates), st, args.top)
        top_df.to_csv(inp / f"top10_ths_tradeable_{p}.csv", index=False)
        board[OUT_COLS].to_csv(inp / f"ths_tradeable_board_{p}.csv", index=False)
        free = board.loc[board.lock_status.eq("自由申赎")].copy()
        free["open_rank"] = range(1, len(free) + 1)
        free[["open_rank"] + OUT_COLS].to_csv(inp / f"ths_open_board_{p}.csv", index=False)
        counts[p] = len(board)
        open_counts[p] = len(free)
        md.append(f"## {PERIOD_NAMES[p]}期（{len(board)} 只，其中锁定期 {int((board.lock_status != '自由申赎').sum())} 只；核查范围=渠道可交易前{args.candidates}名）")
        md.append("")
        md.append("| 名次 | 代码 | 名称 | 累计收益 | 年化收益 | 最大回撤 | 冲击收益 | 冲击回撤 | SR | SD | SL | SA | 总分 | 同花顺 | 申赎 |")
        md.append("|--:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for _, r in board.iterrows():
            sg = "限制大额" if r["sgstat"] == "限制大额" else "开放"
            lk = "锁定期" if r["lock_status"] != "自由申赎" else "自由"
            md.append(f"| {r['ths_rank']} | {r['code']} | {r['name']} | {r['period_return']*100:.2f}% "
                      f"| {r['annualized_return']*100:.2f}% | {r['D']*100:.2f}% | {r['Aret']*100:.2f}% "
                      f"| {r['Add']*100:.2f}% | {r['SR']:.1f} | {r['SD']:.1f} | {r['SL']:.1f} "
                      f"| {r['SA']:.1f} | **{r['SCORE_M0']:.2f}** | {sg} | {lk} |")
        md.append("")
        open_md.append(f"## {PERIOD_NAMES[p]}期自由申赎榜（{len(free)} 只）")
        open_md.append("")
        open_md.append("| 名次 | 代码 | 名称 | 累计收益 | 年化收益 | 最大回撤 | 冲击收益 | 冲击回撤 | SR | SD | SL | SA | 总分 |")
        open_md.append("|--:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in free.iterrows():
            open_md.append(f"| {r['open_rank']} | {r['code']} | {r['name']} | {r['period_return']*100:.2f}% "
                           f"| {r['annualized_return']*100:.2f}% | {r['D']*100:.2f}% | {r['Aret']*100:.2f}% "
                           f"| {r['Add']*100:.2f}% | {r['SR']:.1f} | {r['SD']:.1f} | {r['SL']:.1f} "
                           f"| {r['SA']:.1f} | **{r['SCORE_M0']:.2f}** |")
        open_md.append("")
    (inp / "ths_tradeable_boards.md").write_text("\n".join(md))
    (inp / "ths_open_boards.md").write_text("\n".join(open_md))
    print("channel-tradeable:", counts, "open:", open_counts)


if __name__ == "__main__":
    main()
