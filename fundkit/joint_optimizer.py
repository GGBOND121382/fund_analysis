"""目标权重 + 定投/止盈/止损/再入场规则的外层联合搜索。

本模块不抓取数据。训练窗口生成目标权重候选；验证窗口联合选择权重与规则；
测试窗口只评估冻结后的单一候选。所有评估由调用方提供的真实账户模拟器完成。

注意：费用情景是统一费率压力假设，不等于基金真实费率表；结果始终 RESEARCH_ONLY。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import math


@dataclass(frozen=True)
class JointSearchConfig:
    target_annual_return: float = 0.05
    max_drawdown: float = 0.04
    min_validation_days: int = 252
    min_test_days: int = 252
    fee_stress_rates: tuple[float, ...] = (0.003, 0.006)
    seed: int = 42
    evaluate_ablation_on_test: bool = False

    def validate(self):
        if not (-1 < self.target_annual_return < 2):
            raise ValueError("年化收益目标非法")
        if not (0 < self.max_drawdown < 1):
            raise ValueError("最大回撤预算非法")
        if self.min_validation_days < 2 or self.min_test_days < 2:
            raise ValueError("样本窗口长度非法")
        if not self.fee_stress_rates or any(not (0 <= x < 1) for x in self.fee_stress_rates):
            raise ValueError("费用压力情景非法")


def build_rule_grid(asset_categories: Mapping[str, str] | None = None) -> list[dict]:
    """预先声明的止盈/止损候选网格；各阈值是待检验搜索点，不是推荐参数。"""
    categories = dict(asset_categories or {})
    bond_codes = sorted(c for c, k in categories.items() if k == "bond")
    gold_codes = sorted(c for c, k in categories.items() if k == "gold")
    equity_codes = sorted(c for c, k in categories.items() if k == "equity_index")
    other_codes = sorted(c for c, k in categories.items()
                         if k not in ("bond", "gold", "equity_index"))
    rules = [{"family": "B2", "params": {}, "rule_id": "B2_cashflow_only"}]
    for trigger, dest in product((0.01, 0.02), (0.25, 0.50)):
        rules.append({
            "family": "B3",
            "params": {"B3": {"weight_trigger": trigger,
                              "destination_fraction": dest,
                              "allow_redeem": True,
                              "check_frequency": "weekly"}},
            "rule_id": f"B3_trigger{trigger:.3f}_dest{dest:.2f}",
        })
    for pause, cooldown in product((0.01, 0.02), (5, 10)):
        rules.append({
            "family": "B4",
            "params": {"B4": {"pause_cdar95": pause,
                              "reenter_cdar95_low": pause * 0.5,
                              "reenter_cooldown_days": cooldown}},
            "rule_id": f"B4_cdar{pause:.3f}_cool{cooldown}",
        })

    for k in range(18):
        margin = (0.005, 0.010, 0.015)[k % 3]
        bond_stop = (0.010, 0.015, 0.020)[k % 3]
        gold_stop = (0.030, 0.040, 0.050)[(k // 2) % 3]
        equity_stop = (0.080, 0.100, 0.120)[(k // 3) % 3]
        bond_take = (0.020, 0.025, 0.030)[(k // 4) % 3]
        gold_take = (0.060, 0.080, 0.100)[(k // 5) % 3]
        equity_take = (0.100, 0.120, 0.150)[(k // 6) % 3]
        bond_trail = (0.005, 0.0075, 0.010)[(k // 3) % 3]
        gold_trail = (0.020, 0.030, 0.040)[(k // 4) % 3]
        equity_trail = (0.040, 0.050, 0.060)[(k // 5) % 3]
        other_stop = (0.020, 0.025, 0.030)[k % 3]
        other_take = (0.040, 0.050, 0.060)[(k // 4) % 3]
        other_trail = (0.015, 0.020, 0.025)[(k // 5) % 3]
        redeem_fraction = (0.25, 0.50)[k % 2]
        cooldown = (5, 10, 20)[(k // 3) % 3]
        stop_by_code = {}
        stop_by_code.update({c: bond_stop for c in bond_codes})
        stop_by_code.update({c: gold_stop for c in gold_codes})
        stop_by_code.update({c: equity_stop for c in equity_codes})
        stop_by_code.update({c: other_stop for c in other_codes})
        reenter_by_code = {c: v * 0.5 for c, v in stop_by_code.items()}
        stop_redeem_by_code = {c: redeem_fraction for c in stop_by_code}
        take_by_code = {}
        take_by_code.update({c: bond_take for c in bond_codes})
        take_by_code.update({c: gold_take for c in gold_codes})
        take_by_code.update({c: equity_take for c in equity_codes})
        take_by_code.update({c: other_take for c in other_codes})
        trail_by_code = {}
        trail_by_code.update({c: bond_trail for c in bond_codes})
        trail_by_code.update({c: gold_trail for c in gold_codes})
        trail_by_code.update({c: equity_trail for c in equity_codes})
        trail_by_code.update({c: other_trail for c in other_codes})
        take_redeem_by_code = {c: redeem_fraction for c in take_by_code}
        rules.append({
            "family": "B5",
            "params": {"B5": {
                "mdd_budget": 0.04,
                "internal_drawdown_margin": margin,
                "reenter_mdd_fraction": (0.50, 0.60, 0.70)[(k // 2) % 3],
                "reenter_cooldown_days": cooldown,
                "allow_redeem": True,
                "portfolio_redeem_fraction": redeem_fraction,
                "portfolio_risk_coverage": 0.80,
                "portfolio_exit_escalation_step": 0.005,
                "cdar_lookback": 126,
                "cdar_guard_threshold": (0.020, 0.030)[(k // 3) % 2],
                "fund_pause_drawdown": bond_stop,
                "fund_reenter_drawdown": bond_stop * 0.5,
                "fund_redeem_fraction": redeem_fraction,
                "fund_pause_drawdown_by_code": stop_by_code,
                "fund_reenter_drawdown_by_code": reenter_by_code,
                "fund_redeem_fraction_by_code": stop_redeem_by_code,
                "fund_stop_vol_multiplier_by_category": {
                    "bond": 2.0, "gold": 2.5, "equity_index": 2.5, "other": 2.0,
                },
                "fund_stop_drawdown_cap": 0.12,
                "fund_stop_escalation_step": 0.005,
                "fund_peak_lookback_days": 252,
                "take_profit_activation_by_code": take_by_code,
                "take_profit_trailing_drawdown_by_code": trail_by_code,
                "take_profit_redeem_fraction_by_code": take_redeem_by_code,
                "take_profit_cooldown_days": cooldown,
            }},
            "rule_id": (
                f"B5_m{margin:.3f}_bs{bond_stop:.3f}_gs{gold_stop:.3f}_"
                f"es{equity_stop:.3f}_bt{bond_take:.3f}_gt{gold_take:.3f}_"
                f"et{equity_take:.3f}_rf{redeem_fraction:.2f}_cool{cooldown}"
            ),
        })
    from fundkit.v11_ablation import build_ablation_rules
    rules.extend(build_ablation_rules())
    return rules

def build_rolling_folds(dates: Sequence, train_days: int = 504,
                        validation_days: int = 252, test_days: int = 252,
                        step_days: int = 126, folds: int = 1,
                        embargo_days: int = 0) -> list[dict]:
    """从共同交易日期构建严格前向滚动折；不允许 train/val/test 重叠。"""
    import pandas as pd

    idx = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    if min(train_days, validation_days, test_days, step_days, folds) < 1 or embargo_days < 0:
        raise ValueError("滚动折长度和折数必须为正数；信息隔离天数不得为负数")
    if folds > 1 and step_days < test_days:
        raise ValueError("OVERLAPPING_TEST_WINDOWS_BLOCKED: step_days 必须不小于 test_days")
    # Require published-data embargo between training/validation AND
    # validation/test to prevent as-of NAV leakage (T+2 inferred publication).
    required = (train_days + validation_days + test_days +
                2 * embargo_days + (folds - 1) * step_days)
    if len(idx) < required:
        raise ValueError(f"INSUFFICIENT_HISTORY: 需要至少 {required} 个共同交易日，实际 {len(idx)}")
    first = len(idx) - required
    out = []
    for k in range(folds):
        s = first + k * step_days
        tr = idx[s:s + train_days]
        val_start = s + train_days + embargo_days
        test_start = val_start + validation_days + embargo_days
        va = idx[val_start:val_start + validation_days]
        te = idx[test_start:test_start + test_days]
        if not (tr[-1] < va[0] and va[-1] < te[0]):
            raise AssertionError("内部错误：滚动折时间区间重叠")
        out.append({
            "fold": k + 1,
            "train": (str(tr[0].date()), str(tr[-1].date())),
            "validation": (str(va[0].date()), str(va[-1].date())),
            "test": (str(te[0].date()), str(te[-1].date())),
            "n_train": len(tr), "n_validation": len(va), "n_test": len(te),
            "embargo_trading_days": embargo_days,
        })
    return out


def _aggregate_fee_metrics(rows: list[dict]) -> dict:
    """以不利费用情景聚合：收益取最小、回撤取最大。"""
    required = ("cagr", "max_drawdown")
    for row in rows:
        if any(row.get(k) is None or not math.isfinite(float(row[k])) for k in required):
            raise ValueError("DATA_BLOCKED: 模拟器没有返回有效 CAGR/MDD")
    out = {
        "cagr_worst_fee": min(float(x["cagr"]) for x in rows),
        "mdd_worst_fee": max(float(x["max_drawdown"]) for x in rows),
        "cagr_best_fee": max(float(x["cagr"]) for x in rows),
        "mdd_best_fee": min(float(x["max_drawdown"]) for x in rows),
        "fee_scenarios": rows,
    }
    for key in ("sharpe", "sortino", "cdar95", "xirr"):
        vals = [float(x[key]) for x in rows if x.get(key) is not None and
                math.isfinite(float(x[key]))]
        if vals:
            out[key + "_conservative"] = min(vals) if key in ("sharpe", "sortino", "xirr") else max(vals)
    out["days_min"] = min(int(x.get("days", 0)) for x in rows)
    out["redeem_events_max"] = max(int(x.get("redeem_events", 0)) for x in rows)
    out["buy_events_max"] = max(int(x.get("buy_events", 0)) for x in rows)
    for key in ("turnover_ann", "fees_total"):
        vals = [float(x[key]) for x in rows if x.get(key) is not None and
                math.isfinite(float(x[key]))]
        if vals:
            out[key + "_max"] = max(vals)
    return out


def optimize_joint_fold(weight_candidates: list[dict], rule_candidates: list[dict],
                        evaluator: Callable[[Mapping[str, float], dict, tuple[str, str], float], dict],
                        validation_window: tuple[str, str],
                        test_window: tuple[str, str],
                        cfg: JointSearchConfig | None = None) -> dict:
    """在验证集联合搜索权重和退出规则；选中后测试集只跑该候选。

    evaluator(weights, rule, window, uniform_fee_rate) -> simulator metrics。
    不允许 evaluator 在验证/测试期使用未来持仓或未来披露；由调用方保证 as-of。
    """
    cfg = cfg or JointSearchConfig()
    cfg.validate()
    if not weight_candidates or not rule_candidates:
        raise ValueError("候选权重或规则集合为空")
    if validation_window[0] >= validation_window[1] or test_window[0] >= test_window[1]:
        raise ValueError("验证/测试窗口日期非法")
    if validation_window[1] >= test_window[0]:
        raise ValueError("LOOKAHEAD_BLOCKED: 验证窗口必须严格早于测试窗口")

    trials = []
    for wi, wc in enumerate(weight_candidates):
        weights = dict(wc["weights"])
        vals = [float(v) for v in weights.values()]
        if (not vals or any(not math.isfinite(v) or v < -1e-10 for v in vals) or
                abs(sum(vals) - 1.0) > 1e-6):
            raise ValueError("候选权重必须有限、非负且满足权重守恒")
        for rule in rule_candidates:
            fee_rows = []
            for fee in cfg.fee_stress_rates:
                metrics = evaluator(weights, rule, validation_window, float(fee))
                fee_rows.append({"uniform_fee_rate_assumption": float(fee), **metrics})
            agg = _aggregate_fee_metrics(fee_rows)
            if agg["days_min"] < cfg.min_validation_days:
                raise ValueError(
                    f"INSUFFICIENT_VALIDATION_HORIZON: 至少需要{cfg.min_validation_days}个有效日，"
                    f"实际{agg['days_min']}日")
            feasible = (agg["cagr_worst_fee"] > cfg.target_annual_return and
                        agg["mdd_worst_fee"] <= cfg.max_drawdown + 1e-12)
            trials.append({
                "weight_candidate_id": wi,
                "weights": weights,
                "weight_prescreen": {k: wc.get(k) for k in
                                     ("historical_mdd", "scenario_mdd_p95",
                                      "scenario_return_q25", "static_risk_feasible")},
                "rule_id": rule["rule_id"],
                "family": rule["family"],
                "rule_params": rule["params"],
                "validation": agg,
                "validation_feasible": bool(feasible),
            })
    feasible_rows = [x for x in trials if x["validation_feasible"]]
    if feasible_rows:
        selected = max(feasible_rows, key=lambda x: (
            x["validation"].get("sharpe_conservative", -999),
            x["validation"]["cagr_worst_fee"],
            -x["validation"]["mdd_worst_fee"],
            -x["weight_prescreen"].get("turnover_proxy", 0.0)))
        validation_status = "VALIDATION_FEASIBLE_CANDIDATE"
    else:
        # 无可行解时只选风险最小的诊断候选，明确标记失败，不伪装为推荐。
        selected = min(trials, key=lambda x: (
            x["validation"]["mdd_worst_fee"],
            -x["validation"].get("sharpe_conservative", -999),
            -x["validation"]["cagr_worst_fee"]))
        validation_status = "NO_VALIDATION_FEASIBLE_CANDIDATE"

    # 测试集只评估验证集选出的一个冻结候选，不在测试集重选参数。
    test_rows = []
    for fee in cfg.fee_stress_rates:
        metrics = evaluator(selected["weights"],
                            {"family": selected["family"],
                             "params": selected["rule_params"],
                             "rule_id": selected["rule_id"]},
                            test_window, float(fee))
        test_rows.append({"uniform_fee_rate_assumption": float(fee), **metrics})
    test_agg = _aggregate_fee_metrics(test_rows)
    test_horizon_ok = test_agg["days_min"] >= cfg.min_test_days
    test_pass = (test_horizon_ok and
                 test_agg["cagr_worst_fee"] > cfg.target_annual_return and
                 test_agg["mdd_worst_fee"] <= cfg.max_drawdown + 1e-12)
    if not test_horizon_ok:
        status = "INSUFFICIENT_TEST_HORIZON"
    elif validation_status != "VALIDATION_FEASIBLE_CANDIDATE":
        status = "NO_VALIDATION_FEASIBLE_CANDIDATE"
    elif test_pass:
        status = "TEST_WINDOW_CRITERIA_MET_NOT_GENERALIZED"
    else:
        status = "OOS_TARGET_UNMET"
    # Diagnostic ablation only, with EXACT SAME validation-selected weights.
    # Do not select the winner using this test set.
    ablation_test = []
    if cfg.evaluate_ablation_on_test:
        for rule in rule_candidates:
            fee_rows = []
            for fee in cfg.fee_stress_rates:
                metrics = evaluator(selected["weights"], rule, test_window, float(fee))
                fee_rows.append({"uniform_fee_rate_assumption": float(fee), **metrics})
            ablation_test.append({
                "rule_id": rule["rule_id"], "family": rule["family"],
                "params": rule.get("params", {}),
                "test": _aggregate_fee_metrics(fee_rows),
            })
    return {
        "status": status,
        "ablation_test": ablation_test,
        "ablation_test_not_for_model_selection": True,
        "trade_ready": False,
        "research_only": True,
        "objective": {"annualized_net_return_strictly_gt": cfg.target_annual_return,
                      "portfolio_mdd_lte": cfg.max_drawdown,
                      "priority": "return_and_drawdown_constraints_then_worst_fee_sharpe"},
        "validation_window": validation_window,
        "test_window": test_window,
        "validation_status": validation_status,
        "selected": selected,
        "test": test_agg,
        "test_pass": bool(test_pass),
        "trial_count": len(trials),
        "trials": trials,
        "fee_assumptions": list(cfg.fee_stress_rates),
        "limits": [
            "目标权重候选必须只由训练期数据生成；验证期只选参，测试期只评估冻结候选",
            "统一费率仅为敏感性假设，不代替基金真实申赎费率和持有期费率表",
            "单个测试窗口达标不构成长期有效性证明；需多个滚动折和不同市场状态重复验证",
            "当前账户逐笔批次缺失时，模拟器起点是反事实组合，不是用户真实历史账户",
            "4%是历史/情景风险预算，不是未来最大回撤硬保证",
        ],
    }