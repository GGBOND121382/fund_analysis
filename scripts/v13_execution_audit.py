#!/usr/bin/env python3
"""P4: B2 / A1–A7 动态申购与风险门真实执行审计（真实净值、固定窗口、固定权重）。

窗口: 2026-05-12 ~ 2026-11-09（与 v1.1/v1.2 halfyear 冻结测试窗一致）。
权重: reports/v13_weights_recent252.json 的池A最优（仅执行机制审计用，非OOS结论）。
输出: reports/v13_trade_events.csv, reports/v13_risk_gate_audit.csv,
      reports/v13_execution_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fundkit import policies, simulator, storage  # noqa: E402
from fundkit.adapters import CST, LocalDataAdapter, load_account_snapshot  # noqa: E402
from fundkit.v11_ablation import build_ablation_rules  # noqa: E402

START, END = "2026-05-12", "2026-11-09"
WEIGHTS_JSON = "reports/v13_weights_recent252.json"
FEE = 0.003


def build_initial_lots(adapter, codes, weights, start, capital, fee_rate):
    lots, cash = {}, capital * float(weights.get("cash", 0.0))
    start_ts = pd.Timestamp(start)
    fees = 0.0
    for code in codes:
        w = float(weights.get(code, 0.0))
        if w <= 1e-12:
            continue
        vis = adapter.nav([code], start_ts.tz_localize(CST) + pd.Timedelta(hours=15))
        vis = vis[vis["nav_date"] <= start_ts]
        if vis.empty:
            raise ValueError(f"DATA_BLOCKED: {code} 窗口起点无已披露净值")
        nav0 = float(vis.sort_values("nav_date").iloc[-1]["unit_nav"])
        gross = capital * w
        fee = gross * fee_rate
        fees += fee
        lots[code] = [simulator.Lot(code, (gross - fee) / nav0, start_ts,
                                    simulated=True, order_date=None,
                                    cost_basis_cny=gross, cost_basis_known=True,
                                    basis_source="counterfactual_start")]
    return lots, cash, fees


def run_one(family, rule_params, base_params, snap, adapter, codes, weights,
            capital, equity_codes, daily_limits):
    params = policies.load_params()
    params["target_weights"] = dict(weights)
    params["daily_limits"] = daily_limits
    params["equity_index_codes"] = sorted(equity_codes)
    params["asset_categories"] = base_params["asset_categories"]
    for section, values in (rule_params or {}).items():
        params[section] = {**params.get(section, {}), **values}
    policy = policies.build_policy(family, params)
    lots, cash, fees = build_initial_lots(adapter, codes, weights, START, capital, FEE)
    sim_params = dict(simulator.SIM_DEFAULTS)
    sim_params["subscribe_fee_rate"] = FEE
    sim_params["redeem_fee_rate"] = FEE
    sim_params["initial_investment_fee_cny"] = fees
    config = simulator.SimConfig(
        budget_per_day=float(snap["new_cash_budget_cny_per_execution_day"]),
        allow_new_equity_index=True,
        equity_index_codes=frozenset(equity_codes),
        daily_limits=daily_limits,
        params=sim_params)
    engine = simulator.Simulator(codes, config, adapter)
    return engine.run(START, END, initial_lots=lots, initial_cash=cash, policy=policy)


def main() -> None:
    snap = load_account_snapshot()
    adapter = LocalDataAdapter()
    all_funds = snap["funds"]
    codes = [str(f["code"]) for f in all_funds]
    asset_categories = {str(f["code"]): str(f.get("category", "other")) for f in all_funds}
    equity_codes = sorted(c for c, k in asset_categories.items() if k == "equity_index")
    # 与 main.py:543-550 一致：限额来自 fund_status 快照，仅收录正值；缺省=不限
    daily_limits = {}
    st = storage.load_trade_status()
    st["code"] = st["code"].astype(str).str.zfill(6)
    for _, row in st.iterrows():
        if pd.notna(row["daily_limit"]) and float(row["daily_limit"]) > 0:
            daily_limits[str(row["code"])] = float(row["daily_limit"])
    capital = float(sum(float(f["value_cny"]) for f in all_funds))
    wdoc = json.load(open(WEIGHTS_JSON))
    weights = {k: float(v) for k, v in wdoc["weights"].items()}

    rules = [{"rule_id": "B2_cashflow_only", "family": "B2", "params": {}}]
    rules.extend(build_ablation_rules(include_v13=True))

    all_events, gate_rows, summary = [], [], {}
    trade_sigs = {}
    for rule in rules:
        rid = rule["rule_id"]
        try:
            result = run_one(rule["family"], rule.get("params"),
                             {"asset_categories": asset_categories},
                             snap, adapter, codes, weights, capital,
                             equity_codes, daily_limits)
        except Exception as exc:  # noqa: BLE001
            summary[rid] = {"status": "RUN_FAILED", "reason": str(exc)[:200]}
            continue
        ev = result["events"].copy()
        ev["rule_id"] = rid
        all_events.append(ev)
        daily = result["daily"]
        # 申购/赎回执行序列（用于参数可识别性）
        buys = ev[ev.get("action", ev.get("event", pd.Series(dtype=str))).astype(str)
                  .str.contains("BUY|SUBSCRIBE", na=False)] if len(ev) else ev
        sig = tuple(zip(ev.get("date", []), ev.get("code", []))) if len(ev) else ()
        exec_rows = ev[ev.get("event").notna()] if "event" in ev.columns and len(ev) else ev.iloc[0:0]
        trade_sigs[rid] = [tuple(x) for x in
                           exec_rows[["date", "code", "event"]].itertuples(index=False, name=None)] \
            if len(exec_rows) else []
        # 风险门审计行
        if len(ev) and "action" in ev.columns:
            gate = ev[ev["action"].astype(str).isin(
                ["B6_PRE_RISK", "RISK_DATA_BLOCKED", "DE_RISK_PAUSE_BUYS",
                 "B6_DATA_BLOCKED_MISSING_INFLOW", "B6_DATA_BLOCKED_MISSING_MARK"])]
            for _, g in gate.iterrows():
                gate_rows.append({"rule_id": rid, "date": g.get("date"),
                                  "action": g.get("action"),
                                  "status": g.get("status"),
                                  "mdd_p95": g.get("mdd_p95")})
        m = result["metrics"]
        buys_exec = exec_rows[exec_rows["event"] == "BUY"] if len(exec_rows) else exec_rows
        sells_exec = exec_rows[exec_rows["event"] == "REDEEM"] if len(exec_rows) else exec_rows
        summary[rid] = {
            "status": "OK",
            "cagr": float(m["cagr"]), "mdd": float(m["max_drawdown"]),
            "sharpe": None if m["sharpe"] is None else float(m["sharpe"]),
            "buy_count": int(len(buys_exec)),
            "redeem_count": int(len(sells_exec)),
            "buy_amount_total": float(pd.to_numeric(
                buys_exec.get("amount", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "final_value": float(daily["value"].iloc[-1]) if len(daily) else None,
        }

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_df.to_csv("reports/v13_trade_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gate_rows).to_csv("reports/v13_risk_gate_audit.csv",
                                   index=False, encoding="utf-8-sig")

    # 参数可识别性：A1/A2/A3/A7 交易序列完全一致 -> PARAMETER_NOT_IDENTIFIABLE
    ident_groups = {}
    for rid, sig in trade_sigs.items():
        ident_groups.setdefault(tuple(sig), []).append(rid)
    identifiability = [{"rules": v, "verdict": "PARAMETER_NOT_IDENTIFIABLE"}
                       for v in ident_groups.values() if len(v) > 1]
    for rid, s in summary.items():
        if s.get("status") == "OK" and s["buy_count"] == 0 and s["redeem_count"] == 0:
            s["degenerate"] = "DEGENERATE_ABLATION"

    out = {"window": [START, END], "fee_rate_assumption": FEE,
           "weights_source": WEIGHTS_JSON, "weights": weights,
           "capital_cny": capital, "rules": summary,
           "identifiability": identifiability,
           "note": "执行机制审计；权重为 asof=2026-11-12 训练结果，窗口内表现为参考，非独立OOS"}
    Path("reports/v13_execution_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for rid, s in summary.items():
        print(rid, json.dumps(s, ensure_ascii=False, default=str)[:220])
    print("identifiability:", json.dumps(identifiability, ensure_ascii=False))


if __name__ == "__main__":
    main()
