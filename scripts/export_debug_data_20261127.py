#!/usr/bin/env python3
"""导出 ChatGPT 算法审计调试数据包 -> debug_data/20261127/。

三段内容：
1. sample/    代表性基金（M0-M3 各期限 TOP50 ∪ 类别抽样 ∪ 异常/关注 ∪ 多份额）原始序列
2. features/  四期限全市场特征（含未通过质检基金及原因、月度收益、EB 原料）
3. oos/       7 个形成日的实验数据（形成前特征/月度 与 形成后日度 分文件，无前视）

不修改任何算法与参数；只做数据搬运与记录。所有随机抽样 seed=42。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_industry_fund_ranking import load_catalog, load_paths  # noqa: E402
from run_oos_model_comparison import fold_features, FORMATION_DATES, HOLD_MONTHS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "all_funds"
OUT = ROOT / "debug_data" / "20261127"
ASOF = pd.Timestamp("2026-11-20")
SEED = 42
LOCK_RE = re.compile(r"滚动|持有|定开|封闭")
SHARD_LIMIT = 450 * 1024  # 单分片目标上限 450KB（留余量 <500KB）
ANOMALY = ["003184", "005973", "006964", "016886", "015023", "519648"]
WATCHLIST = ["025368", "019851"]
SAMPLE_CAP = 500

manifest = []


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(p: Path, section: str, desc: str, *, rows=None, date_range=None, notes=""):
    manifest.append(dict(
        path=str(p.relative_to(OUT)), section=section, description=desc,
        rows=rows, date_range=date_range, bytes=p.stat().st_size,
        sha256=sha256(p), notes=notes))


def write_csv(df: pd.DataFrame, p: Path, section: str, desc: str, **kw):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    record(p, section, desc, rows=len(df), **kw)


def write_csv_gz(df: pd.DataFrame, p: Path, section: str, desc: str, **kw):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    record(p, section, desc, rows=len(df), **kw)


# ---------------------------------------------------------------- 公共加载
def load_all():
    catalog = load_catalog(DATA / "fund_list.csv")
    print(f"catalog: {len(catalog)}", flush=True)
    fund_status = pd.read_csv(DATA / "fund_status.csv", dtype={"code": str})
    fund_status["code"] = fund_status["code"].str.zfill(6)
    return catalog, fund_status


def trade_lock_frame(catalog, fund_status):
    d = catalog[["code", "product_key", "share", "group", "fund_type",
                 "pricing_basis", "is_long_periodic", "is_periodic_unverified",
                 "periodic_status", "periodic_months"]].copy()
    d["lock_name_flag"] = d.code.map(
        dict(zip(catalog.code, catalog.name))).map(
        lambda n: bool(LOCK_RE.search(str(n))))
    m = d.merge(fund_status, on="code", how="left")
    return m


# ---------------------------------------------------------------- 第 1 段
def select_sample(catalog):
    comp = ROOT / "reports" / "all_funds" / "comprehensive_20261120"
    models_dir = ROOT / "reports" / "all_funds" / "models_20261120"
    tags = {}

    def add(codes, tag):
        for c in codes:
            tags.setdefault(c, set()).add(tag)

    for p in ("5y", "3y", "1y", "6m"):
        a = pd.read_csv(comp / f"all_scored_{p}.csv", dtype={"code": str},
                        low_memory=False)
        a["code"] = a["code"].str.zfill(6)
        scored = a[a.status.eq("SCORED")]
        add(scored.nlargest(50, "SCORE_M0")["code"], f"M0_TOP50_{p}")
        for m in ("m1", "m2", "m3"):
            r = pd.read_csv(models_dir / f"full_ranking_{m}_{p}.csv",
                            dtype={"code": str}, low_memory=False)
            add(r.head(50)["code"].str.zfill(6), f"{m.upper()}_TOP50_{p}")
    add(ANOMALY, "ANOMALY")
    add(WATCHLIST, "WATCHLIST")

    # 同一 product 的其它份额
    pk = catalog.set_index("code")["product_key"]
    top_pks = {pk.get(c) for c in tags if c in pk.index}
    fam = catalog.loc[catalog.product_key.isin(top_pks), "code"]
    add(fam, "SAME_PRODUCT_SHARE")

    # 恒定净值（货币基金）、负收益、普通低波动样本
    money = catalog.loc[catalog.pricing_basis.eq("income_per_10000"), "code"]
    add(pd.Series(money).head(5), "CONSTANT_NAV_MONEY")
    a1 = pd.read_csv(comp / "all_scored_1y.csv", dtype={"code": str},
                     low_memory=False)
    a1["code"] = a1["code"].str.zfill(6)
    s1 = a1[a1.status.eq("SCORED")]
    add(s1.nsmallest(5, "period_return")["code"], "NEGATIVE_RETURN")
    bond = s1.loc[s1.group.eq("中长纯债")]
    add(bond.iloc[len(bond) // 2: len(bond) // 2 + 5]["code"], "NORMAL_LOWVOL")

    # 类别分层随机抽样补足到 SAMPLE_CAP
    rng = np.random.default_rng(SEED)
    rest = [c for c in catalog.code if c not in tags]
    by_group = catalog.loc[catalog.code.isin(rest)]
    need = max(0, SAMPLE_CAP - len(tags))
    if need:
        frac = by_group.group.value_counts(normalize=True)
        quota = (frac * need).round().astype(int)
        picked = []
        for g, k in quota.items():
            pool = by_group.loc[by_group.group.eq(g), "code"].to_numpy()
            if len(pool) and k:
                picked.extend(rng.choice(pool, size=min(k, len(pool)),
                                         replace=False).tolist())
        add(picked[:need], "GROUP_RANDOM")
    codes = list(tags)[:SAMPLE_CAP]
    meta = catalog.loc[catalog.code.isin(codes),
                       ["code", "name", "group", "fund_type", "product_key",
                        "share", "pricing_basis"]].copy()
    meta["selection_tags"] = meta.code.map(lambda c: "|".join(sorted(tags[c])))
    return meta


def export_sample(catalog, fund_status):
    meta = select_sample(catalog)
    codes = set(meta.code)
    # 首条净值日期（成立时间缓存不可得，用缓存首净值日近似，README 注明）
    first_nav = {}
    rows_buf, shard_idx, shard_size = [], 1, 0
    nav_dir = OUT / "sample"
    bucket = dict(zip(catalog.code, catalog.pricing_basis))
    bmap = {"fund_nav": "total_return", "income_per_10000": "money",
            "adjusted_close": "listed"}
    for r in meta.itertuples(index=False):
        f = DATA / bmap[r.pricing_basis] / f"{r.code}.csv"
        if not f.is_file():
            continue
        d = pd.read_csv(f, dtype={"date": str})
        d.columns = [c.strip("﻿") for c in d.columns]
        first_nav[r.code] = d["date"].iloc[0] if len(d) else None
        out = pd.DataFrame({"code": r.code, "date": d["date"],
                            "nav": d[d.columns[1]],
                            "source": d.get("source", "")})
        b = out.to_csv(index=False).encode()
        if shard_size + len(b) > SHARD_LIMIT and rows_buf:
            part = pd.concat(rows_buf)
            p = nav_dir / f"nav_shard_{shard_idx:02d}.csv"
            write_csv(part, p, "sample", "代表性基金净值序列分片",
                      date_range=[part.date.min(), part.date.max()])
            shard_idx += 1
            rows_buf, shard_size = [], 0
        rows_buf.append(out)
        shard_size += len(b)
    if rows_buf:
        part = pd.concat(rows_buf)
        p = nav_dir / f"nav_shard_{shard_idx:02d}.csv"
        write_csv(part, p, "sample", "代表性基金净值序列分片",
                  date_range=[part.date.min(), part.date.max()])
    meta["first_nav_date_in_cache"] = meta.code.map(first_nav)
    meta["inception_note"] = "成立时间未缓存；first_nav_date 为缓存内首条净值"
    write_csv(meta, OUT / "sample" / "sample_funds.csv", "sample",
              "代表性基金元数据与入选标签")
    # 交易/锁定状态
    tl = trade_lock_frame(catalog, fund_status)
    tl = tl.loc[tl.code.isin(codes)]
    ths_cols = {}
    ths_file = (ROOT / "reports" / "all_funds" / "comprehensive_20261120"
                / "ths_tradeable_status_3y.csv")
    if ths_file.is_file():
        ths = pd.read_csv(ths_file, dtype={"code": str, "buy": str})
        ths["code"] = ths["code"].str.zfill(6)
        ths_cols = ths.set_index("code")
    tl["ths_buy"] = tl.code.map(ths_cols["buy"]) if len(ths_cols) else np.nan
    tl["ths_sgstat"] = tl.code.map(ths_cols["sgstat"]) if len(ths_cols) else np.nan
    tl["ths_checked_at"] = (tl.code.map(ths_cols["checked_at"])
                            if len(ths_cols) else np.nan)
    tl["fee_note"] = "申购/赎回费率未缓存；实验统一假设 roundtrip 0.1%"
    write_csv(tl, OUT / "sample" / "trade_lock_status.csv", "sample",
              "交易状态（天天基金+THS）与锁定期信息")
    # 分红拆分事件
    div = pd.read_csv(ROOT / "data" / "dividends.csv", dtype={"code": str})
    div["code"] = div["code"].str.zfill(6)
    write_csv(div.loc[div.code.isin(codes)],
              OUT / "sample" / "events_dividends.csv", "sample",
              "样本基金分红事件（data/dividends.csv 子集）")
    sp = json.loads((ROOT / "data" / "splits.json").read_text())
    sp_rows = [(c, d, ratio) for c, evs in sp.items() if c in codes
               for d, ratio in evs]
    write_csv(pd.DataFrame(sp_rows, columns=["code", "date", "split_ratio"]),
              OUT / "sample" / "events_splits.csv", "sample",
              "样本基金份额拆分事件（data/splits.json 子集）")
    return codes


# ---------------------------------------------------------------- 第 2 段
def export_features(catalog, fund_status):
    comp = ROOT / "reports" / "all_funds" / "comprehensive_20261120"
    models_dir = ROOT / "reports" / "all_funds" / "models_20261120"
    tl = trade_lock_frame(catalog, fund_status)
    tl = tl[["code", "sub_status", "red_status", "daily_limit",
             "status_checked_at", "lock_name_flag", "periodic_status",
             "periodic_months", "is_long_periodic", "is_periodic_unverified"]]
    for p in ("5y", "3y", "1y", "6m"):
        a = pd.read_csv(comp / f"all_scored_{p}.csv", dtype={"code": str},
                        low_memory=False)
        a["code"] = a["code"].str.zfill(6)
        m1 = pd.read_csv(models_dir / f"full_ranking_m1_{p}.csv",
                         dtype={"code": str}, low_memory=False)
        m1["code"] = m1["code"].str.zfill(6)
        extra = m1[["code", "D95", "L_daily", "annual_se",
                    "observed_log_cagr", "eb_log_cagr", "eb_se",
                    "peer_count", "mu_lower"]]
        feat = a.merge(extra, on="code", how="left").merge(tl, on="code",
                                                           how="left")
        write_csv_gz(feat, OUT / "features" / f"features_{p}.csv.gz",
                     "features", f"{p} 全市场特征（含质检失败及原因、EB原料、交易/锁定）")
    return None


def export_monthly(catalog, paths):
    """四期限全市场完整月度收益（仅 SCORED 基金；质检失败者无窗口序列）。"""
    from fundkit.industry_rank import HORIZONS, window_dates
    comp = ROOT / "reports" / "all_funds" / "comprehensive_20261120"
    for p in ("5y", "3y", "1y", "6m"):
        a = pd.read_csv(comp / f"all_scored_{p}.csv", dtype={"code": str},
                        low_memory=False)
        codes = a.loc[a.status.eq("SCORED"), "code"].str.zfill(6).tolist()
        n_months, _ = HORIZONS[p]
        begin, period_m, complete = window_dates(ASOF, n_months)
        rows = []
        for c in codes:
            nav = paths.get(c)
            if nav is None:
                continue
            m = nav.monthly(ASOF, period_m, complete)
            if m is None:
                continue
            months = pd.period_range(end=period_m, periods=len(m), freq="M")
            rows.append(pd.DataFrame({"code": c,
                                      "month": months.astype(str),
                                      "monthly_return": m}))
        out = pd.concat(rows, ignore_index=True)
        write_csv_gz(out, OUT / "features" / f"monthly_{p}.csv.gz",
                     "features", f"{p} 窗口内完整月度收益序列",
                     date_range=[out.month.min(), out.month.max()])


# ---------------------------------------------------------------- 第 3 段
def export_oos(catalog, paths):
    import re as _re
    low_vol_re = _re.compile("债|固收|绝对收益|货币|FOF-稳健")
    fac = pd.read_csv(ROOT / "data" / "factors" / "daily.csv",
                      parse_dates=["date"])
    write_csv(fac, OUT / "oos" / "factors_daily.csv", "oos",
              "基准指数(沪深300)/国债收益率/无风险利率（全量复制）",
              date_range=[str(fac.date.min().date()),
                          str(fac.date.max().date())])
    td = pd.DatetimeIndex(fac.loc[fac.csi300_price_index.notna(), "date"])
    td = td[(td >= pd.Timestamp("2023-01-01")) & (td <= ASOF)]
    write_csv(pd.DataFrame({"trade_date": td.strftime("%Y-%m-%d")}),
              OUT / "oos" / "market_calendar.csv", "oos",
              "沪深交易所交易日历（沪深300有行情的日期）",
              date_range=[str(td.min().date()), str(td.max().date())])
    for t_str in FORMATION_DATES:
        t = pd.Timestamp(t_str)
        end = t + pd.DateOffset(months=HOLD_MONTHS)
        print("fold", t_str, flush=True)
        df = fold_features(catalog, paths, t, low_vol_re)
        feat = df.drop(columns=["nav_obj", "nav_i", "nav_j"])
        feat.insert(0, "formation_date", t_str)
        feat["trade_status_pit"] = "PIT_UNAVAILABLE_历史交易状态未存档"
        write_csv_gz(feat, OUT / "oos" / f"fold_{t_str}_pre_features.csv.gz",
                     "oos", f"{t_str} 形成日候选基金特征（仅用形成日前数据）")
        # 形成前月度收益（1y 窗，同实验口径）
        from fundkit.industry_rank import window_dates
        begin, period_m, complete = window_dates(t, 12)
        mrows, prows = [], []
        for r in df.itertuples(index=False):
            nav = r.nav_obj
            m = nav.monthly(t, period_m, complete)
            if m is not None:
                months = pd.period_range(end=period_m, periods=len(m), freq="M")
                mrows.append(pd.DataFrame({"code": r.code,
                                           "month": months.astype(str),
                                           "monthly_return": m}))
            # nav_obj 已在形成日截断（无前视）；验证序列必须从完整路径取。
            full = paths.get(r.code)
            if full is None:
                continue
            i, j = full.right(t), full.right(end)
            if i < 0 or j <= i:
                continue
            prows.append(pd.DataFrame({"code": r.code,
                                       "date": full.dates[i: j + 1]
                                       .strftime("%Y-%m-%d"),
                                       "nav": full.value[i: j + 1]}))
        if mrows:
            mm = pd.concat(mrows, ignore_index=True)
            write_csv_gz(mm, OUT / "oos" / f"fold_{t_str}_pre_monthly.csv.gz",
                         "oos", f"{t_str} 形成前月度收益（1y窗）",
                         date_range=[mm.month.min(), mm.month.max()])
        post = pd.concat(prows, ignore_index=True)
        write_csv_gz(post, OUT / "oos" / f"fold_{t_str}_post_daily.csv.gz",
                     "oos", f"{t_str} 形成后6个月日度净值（验证用，分离存储）",
                     date_range=[post.date.min(), post.date.max()],
                     notes="形成日之后的数据，禁止用于打分")


# ---------------------------------------------------------------- 收尾
def write_manifest():
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    m = dict(generated_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
             git_sha=git_sha, asof=str(ASOF.date()), seed=SEED,
             files=manifest)
    p = OUT / "MANIFEST.json"
    p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog, fund_status = load_all()
    print("== section 1: sample ==", flush=True)
    export_sample(catalog, fund_status)
    print("== section 2: features ==", flush=True)
    export_features(catalog, fund_status)
    print("== loading full-market nav paths ==", flush=True)
    paths, _ = load_paths(catalog, DATA, ASOF)
    print(f"paths: {len(paths)}", flush=True)
    export_monthly(catalog, paths)
    print("== section 3: oos ==", flush=True)
    export_oos(catalog, paths)
    write_manifest()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
