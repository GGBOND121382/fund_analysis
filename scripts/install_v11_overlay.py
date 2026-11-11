"""Apply the v1.1 B6 overlay to a clean or v1.0.1 repository checkout.

Run: python path/to/scripts/install_v11_overlay.py --repo /path/to/fund_analysis
Does not delete data and never edits B0-B5 logic. Idempotent on repeated runs.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

FILES=['fundkit/v11_cash.py','fundkit/v11_signals.py',
       'fundkit/v11_exit.py','fundkit/v11_policy.py','fundkit/v11_forecast.py','fundkit/v11_ablation.py']


def patch_text(path: Path, before: str, after: str):
    src=path.read_text(encoding='utf-8')
    if after in src:
        return False
    if src.count(before)!=1:
        raise RuntimeError(f'BASE_VERSION_MISMATCH: {path} does not contain exactly one anchor: {before[:80]}')
    path.write_text(src.replace(before,after,1),encoding='utf-8')
    return True


def install(repo,source):
    repo=Path(repo).resolve();source=Path(source).resolve()
    if not all((repo/f).exists() for f in ('fundkit/policies.py','fundkit/simulator.py','config/policy_params.json')):
        raise ValueError('expected v1.0.1 fund_analysis checkout with fundkit modules')
    # Verify all anchors before mutating. Protect against arbitrary branches.
    policies=(repo/'fundkit/policies.py').read_text(encoding='utf8')
    simulator=(repo/'fundkit/simulator.py').read_text(encoding='utf8')
    anchors=[('"B5": lambda: make_b5(params),','"B6": lambda: __import__("fundkit.v11_policy", fromlist=["make_b6"]).make_b6(params),'),
             ('"unit_nav_history": visible_hist,','"external_cash_inflow_today": cfg.budget_per_day,')]
    for a,b in [anchors[0]]:
        if a not in policies and b not in policies:
            raise ValueError('BASE_VERSION_MISMATCH: B5 registry anchor absent')
    if anchors[1][0] not in simulator and anchors[1][1] not in simulator:
        raise ValueError('BASE_VERSION_MISMATCH: cash inflow state anchor absent')
    if '            orders = policy(state) if policy else {}' not in simulator and 'event_start = len(events)' not in simulator:
        raise ValueError('BASE_VERSION_MISMATCH: execution callback anchor absent')
    if '            # 估值与单位净值' not in simulator and 'policy.on_execution' not in simulator:
        raise ValueError('BASE_VERSION_MISMATCH: end-of-trade anchor absent')
    for f in FILES:
        (repo/f).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/f,repo/f)
    pol=repo/'fundkit/policies.py'
    patch_text(pol,'"B5": lambda: make_b5(params),',
               '"B5": lambda: make_b5(params),\n        "B6": lambda: __import__("fundkit.v11_policy", fromlist=["make_b6"]).make_b6(params),')
    sim=repo/'fundkit/simulator.py'
    patch_text(sim,'"unit_nav_history": visible_hist,',
               '"unit_nav_history": visible_hist,\n                "external_cash_inflow_today": cfg.budget_per_day,\n                "lot_states": {code:[{\n                    "lot_id": f"{code}:{idx}", "shares":float(lot.shares),\n                    "order_date": str(lot.order_date) if lot.order_date is not None else None,\n                    "confirm_date":str(lot.confirm_date),\n                    "redeemable": bool(self._redeemable(code,lot,day)),\n                    "cost_basis_cny":lot.cost_basis_cny,\n                    "cost_basis_known":bool(lot.cost_basis_known)\n                    } for idx,lot in enumerate(lot_list) if lot.shares>1e-10]\n                    for code,lot_list in lots.items()},')
    patch_text(sim,'            orders = policy(state) if policy else {}',
               '            event_start = len(events)\n            orders = policy(state) if policy else {}')
    patch_text(sim,'            # 估值与单位净值',
               '            # Reconcile actual fills, not suggested orders, into B6 subledger.\n            if callable(getattr(policy,"on_execution",None)):\n                policy.on_execution(day,events[event_start:])\n\n            # 估值与单位净值')
    joint=repo/'fundkit/joint_optimizer.py'
    if joint.exists():
        jointtext=joint.read_text(encoding='utf8')
        marker='    return rules\n\ndef build_rolling_folds'
        replacement=('    from fundkit.v11_ablation import build_ablation_rules\n'
                     '    rules.extend(build_ablation_rules())\n'
                     '    return rules\n\ndef build_rolling_folds')
        if marker not in jointtext and replacement not in jointtext:
            raise RuntimeError('BASE_VERSION_MISMATCH: joint_optimizer rule-grid anchor absent')
        patch_text(joint,marker,replacement)
    paramsfile=repo/'config/policy_params.json'
    params=json.loads(paramsfile.read_text(encoding='utf8'))
    params.setdefault('B6',{
        'base_fraction':1.0, 'release_fraction':0.5,
        'deferred_horizon_days':10, 'cash_reserve_weight':0.05,
        'risk_seed':42, 'risk_paths':32, 'simulate_redeem':True,
        'exit_mode':'both', 'entry_timing_enabled':True, 'optimize_exit_fraction':True,
        'take_profit_activation_by_category':{'bond':.025,'gold':.08,'equity_index':.12},
        'take_profit_trailing_drawdown_by_category':{'bond':.0075,'gold':.03,'equity_index':.05},
        'take_profit_redeem_fraction':.25,
        'min_profit_after_fee':.002,
        'redeem_fee_proxy':.003})
    paramsfile.write_text(json.dumps(params,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('B6 overlay installed, B0-B5 retained; research only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True)
    args=p.parse_args()
    install(args.repo,Path(__file__).resolve().parent.parent)
