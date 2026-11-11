"""Pre-registered small B6 ablation set, appended to legacy rule grid.

A0 is the original B0 research baseline in existing evaluation.py, not
redefined or re-run within v1.1. Rules A1..A6 all use identical target funds,
initial account and fee scenarios (injected by the original joint optimizer).
"""

def build_ablation_rules():
    common={'cash_reserve_weight':.05,'risk_paths':32,
            'simulate_redeem':True,'deferred_horizon_days':10}
    def add(id_,description,extra):
        return {'family':'B6','rule_id':id_,
                'params':{'B6':{**common,**extra}}, 'description':description}
    return [
        add('A1_fixed_weights','Fixed contributions, no TP or stop',
            {'base_fraction':1,'release_fraction':0,'entry_timing_enabled':False,'exit_mode':'none'}),
        add('A2_dynamic_cash','Deferred contributions without timing alpha',
            {'base_fraction':.75,'release_fraction':0,'entry_timing_enabled':False,'exit_mode':'none'}),
        add('A3_entry_timing','Deferred cash plus published signals and risk gate',
            {'base_fraction':.75,'release_fraction':.5,'entry_timing_enabled':True,'exit_mode':'none'}),
        add('A4_take_profit','Fixed contributions and per-lot armed TP only',
            {'base_fraction':1,'release_fraction':0,'entry_timing_enabled':False,'exit_mode':'take_profit_only'}),
        add('A5_stop_reentry','Fixed contributions and legacy B5 risk stop only',
            {'base_fraction':1,'release_fraction':0,'entry_timing_enabled':False,'exit_mode':'stop_only'}),
        add('A6_joint','Dynamic contributions plus tactical signals and both exits',
            {'base_fraction':.75,'release_fraction':.5,'entry_timing_enabled':True,'exit_mode':'both'}),
    ]
