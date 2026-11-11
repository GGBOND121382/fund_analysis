"""Pre-registered B6 calibration grid for multi-month exploratory research.

Weights, fees, account, period are identical across variants. No setting is a
validated investment recommendation. Cash reserve is zero in all ablations
to match B2 baseline's research target and avoid cash-buffer confounding.
"""

def build_ablation_rules():
    common = {
        "cash_reserve_weight": 0.0, "risk_paths": 64,
        "simulate_redeem": True, "deferred_horizon_days": 10,
        "optimize_exit_fraction": True,
    }
    rules = []
    def add(rule_id, description, **kw):
        rules.append({
            "family": "B6", "rule_id": rule_id,
            "params": {"B6": {**common, **kw}}, "description": description,
        })
    add("A1_no_timing", "100% daily deposit; no exits (risk-gated)",
        base_fraction=1.0, release_fraction=0.0,
        entry_timing_enabled=False, exit_mode="none")
    add("A2_base75", "75% baseline, expired cash released, no trend",
        base_fraction=.75, release_fraction=0.0,
        entry_timing_enabled=False, exit_mode="none")
    add("A2_base50", "50% baseline, expired cash released, no trend",
        base_fraction=.50, release_fraction=0.0,
        entry_timing_enabled=False, exit_mode="none")
    add("A3_timing75_release50", "75% baseline, 50% deferred tactical release",
        base_fraction=.75, release_fraction=.50,
        entry_timing_enabled=True, exit_mode="none")
    add("A3_timing75_release100", "75% baseline, 100% deferred tactical release",
        base_fraction=.75, release_fraction=1.0,
        entry_timing_enabled=True, exit_mode="none")
    add("A4_takeprofit", "100% baseline with lot-aware trailing take-profit",
        base_fraction=1.0, release_fraction=0.0,
        entry_timing_enabled=False, exit_mode="take_profit_only")
    add("A5_stop", "100% baseline with drawdown stop and re-entry",
        base_fraction=1.0, release_fraction=0.0,
        entry_timing_enabled=False, exit_mode="stop_only")
    add("A6_joint75", "75% baseline, tactical release, both exit modes",
        base_fraction=.75, release_fraction=.50,
        entry_timing_enabled=True, exit_mode="both")
    add("A6_joint50", "50% baseline, tactical release, both exit modes",
        base_fraction=.50, release_fraction=.50,
        entry_timing_enabled=True, exit_mode="both")
    return rules
