"""Evidence-locked all-market fund discovery (research vs execution vs legacy).

Examples:
  python scripts/run_fund_discovery.py --asof 2026-11-20 \
    --snapshot reports/all_funds/industry_rank_20261120 \
    --output reports/fund_discovery_20261120

No confirmed forecast/economic-risk/trading evidence => NO certified top ten.
Optional forecast, cash, valuation and transaction sources enable an explicitly
UNVALIDATED model-score experiment, never a manufactured OOS endorsement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from fundkit.fund_master import (product_review_hint, read_timestamped_evidence,
                                 resolve_liquidity)
from fundkit.pricing_audit import audit_valuation
from fundkit.return_models import resolve_forecasts, fit_category_forecasts
from fundkit.opportunity_rank import (OpportunityInput, BUDGETS,
                                      stationary_bootstrap,cash_paths,
                                      optimize_opportunity)
from fundkit.industry_rank import HORIZONS, assemble
from scripts.run_industry_fund_ranking import load_catalog, load_paths


def read_csv(path, required=()):
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=["code",*required])
    df=pd.read_csv(path,dtype={"code":str},low_memory=False)
    if not set(required).issubset(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(set(required)-set(df.columns))}")
    return df


def source_fund_rows(snapshot, horizon):
    """Existing report is a PARTIAL pool (896 contract-unknown funds excluded)."""
    src=Path(snapshot)
    ranked=read_csv(src/f"ranking_{horizon}.csv",
                    ("code","name","group","period_return","max_drawdown"))
    missing=read_csv(src/f"unranked_{horizon}.csv",("code","reason"))
    if ranked.empty:
        raise ValueError("snapshot ranking absent or empty")
    ranked["feature_status"]="HISTORICAL_FEATURE_AVAILABLE"
    seen=set(ranked.code.astype(str))
    excluded=missing.loc[~missing.code.isin(seen),["code","reason"]].copy()
    excluded["feature_status"]="SNAPSHOT_EXCLUDED_NO_FEATURES"
    excluded=excluded.rename(columns={"reason":"feature_reason"})
    result=pd.concat([ranked,excluded],ignore_index=True,sort=False)
    if result.code.duplicated().any():
        raise ValueError("duplicate share codes across snapshot")
    return result


def raw_fund_rows(catalog, paths, asof, horizon):
    """Re-evaluate all share classes, including previously regex-excluded funds."""
    frame,rejected=assemble(catalog,paths,asof,horizon,[])
    frame["feature_status"]="HISTORICAL_FEATURE_AVAILABLE"
    if rejected:
        others=pd.DataFrame(rejected,columns=["code","group","feature_reason"])
        others=others.loc[~others.code.isin(frame.code)]
        others["feature_status"]="HISTORY_INSUFFICIENT_OR_DUPLICATE"
        frame=pd.concat([frame,others],ignore_index=True,sort=False)
    return frame


def validate_cash_assumption(args, asof):
    if args.cash_rate_annual is None:
        return False
    if not args.cash_source or not args.cash_known_at:
        raise ValueError("cash rate requires --cash-source and --cash-known-at")
    t=pd.to_datetime(args.cash_known_at,errors="coerce",utc=True)
    if pd.isna(t) or t.tz_convert(None)>pd.Timestamp(asof):
        raise ValueError("cash rate must be published by formation date")
    if not np.isfinite(args.cash_rate_annual) or args.cash_rate_annual <= -1:
        raise ValueError("cash rate invalid")
    return True


def code_seed(code,seed):
    return int(hashlib.sha256(f"{seed}:{code}".encode()).hexdigest()[:8],16)


def analyze_horizon(frame, liquidity, valuation, forecasts, asof, months, cash_ok,
                    args, paths):
    original=frame.copy()
    original["code"]=original.code.astype(str).str.zfill(6)
    original["contract_review_hint"]=original.get("name",pd.Series("",index=original.index)).fillna("").map(product_review_hint)
    original=original.merge(liquidity,on="code",how="left",validate="one_to_one")
    original=audit_valuation(original,valuation)
    original=original.merge(forecasts,on="code",how="left",validate="one_to_one")
    original["opportunity_status"]="EVIDENCE_INSUFFICIENT"
    original["opportunity_rank"]=np.nan
    original["score_annual"]=np.nan
    original["best_weight"]=np.nan
    original["scenario_risk_upper"]=np.nan
    original["score_class"]="NOT_RANKED"
    budgets={}
    for b in BUDGETS:
        budgets[b]=[]
    count=0
    for idx,row in original.iterrows():
        code=row.code
        reason=None
        if row.feature_status!="HISTORICAL_FEATURE_AVAILABLE":
            reason="NO_COMPARABLE_HISTORICAL_FEATURE"
        elif not cash_ok:
            reason="CASH_FORECAST_MISSING"
        elif row.forecast_status!="EVIDENCE_SUPPLIED_UNVALIDATED":
            reason="FORECAST_MISSING"
        elif not bool(row.economic_risk_verified):
            reason="ECONOMIC_RISK_UNVERIFIED"
        elif not np.isfinite(row.total_fee_rate) or row.total_fee_rate<0:
            reason="TRADING_COST_UNVERIFIED"
        elif paths is None or code not in paths:
            reason="DAILY_NAV_PATH_MISSING"
        if reason:
            original.at[idx,"opportunity_status"]=reason
            continue
        # A forecast not above cash cannot produce positive value for w>0
        # with nonnegative costs. Keep this as a valid zero-score candidate.
        if row.forecast_mu_annual <= args.cash_rate_annual:
            estimates=[dict(budget=float(b),weight=0.,risk_upper=0.,
                           opportunity_score_annual=0.,status="NO_POSITIVE_EXCESS") for b in BUDGETS]
        else:
            nav=paths[code]
            nav_values=nav.value[nav.dates<=pd.Timestamp(asof)]
            try:
                fund=stationary_bootstrap(nav_values,steps=21*months,
                     scenarios=args.scenarios,block_length=args.block_length,
                     seed=code_seed(code,args.seed))
                cash=cash_paths(args.cash_rate_annual,steps=21*months,
                                scenarios=args.scenarios)
                setting=OpportunityInput(
                    forecast_annual=float(row.forecast_mu_annual),
                    cash_annual=float(args.cash_rate_annual),
                    fee_rate_roundtrip=float(row.total_fee_rate),
                    economic_shock_loss=float(row.economic_mdd_upper),
                    decision_months=months,grid_points=args.grid_points)
                estimates=optimize_opportunity(fund,cash,setting)
            except (ValueError,FloatingPointError,OverflowError) as exc:
                original.at[idx,"opportunity_status"]="PATH_RISK_MODEL_FAILED"
                continue
        for e in estimates:
            budgets[e["budget"]].append(dict(code=code,**e))
        score=[x for x in estimates if abs(x["budget"]-0.04)<1e-9][0]
        original.at[idx,"score_annual"]=score["opportunity_score_annual"]
        original.at[idx,"best_weight"]=score["weight"]
        original.at[idx,"scenario_risk_upper"]=score["risk_upper"]
        original.at[idx,"opportunity_status"]=score["status"]
        original.at[idx,"score_class"]="EXPERIMENTAL_NOT_OOS_VERIFIED"
        count+=1

    # 4% is the reference risk-budget scenario, 1/2/8 are published as sensitivity.
    good=original.score_annual.notna()
    ranked=original.loc[good].copy()
    if len(ranked):
        # Ties retain the same rank; zero-value products are not "best opportunities".
        ranked=ranked.sort_values(["score_annual","code"],ascending=[False,True])
        positive=ranked.score_annual>0
        ranked.loc[positive,"opportunity_rank"]=ranked.loc[positive,"score_annual"].rank(
            method="min",ascending=False).astype(int)
        original.loc[ranked.index,"opportunity_rank"]=ranked.opportunity_rank
    investable=original.loc[(original.score_annual>0)
                            &original.executable_horizon.fillna(False).eq(True)].copy()
    if args.order_amount is None:
        investable=investable.iloc[:0]
    else:
        investable=investable.loc[investable.max_subscription_amount >= args.order_amount]
    # Precheck is not an order fill, and model scores have NOT passed OOS.
    investable["execution_label"]="POTENTIAL_PRECHECK_ONLY_NOT_VERIFIED_STRATEGY"
    investable=investable.sort_values(["score_annual","code"],ascending=[False,True])
    stats={"feature_rows":int(len(original)),
           "forecast_supplied":int(original.forecast_status.eq("EVIDENCE_SUPPLIED_UNVALIDATED").sum()),
           "economic_risk_evidence":int(original.economic_risk_verified.sum()),
           "trading_evidence":int(original.liquidity_status.eq("STATUS_SOURCE_OBSERVED").sum()),
           "experimentally_scored":count,
           "opportunity_positive":int((original.score_annual>0).sum()),
           "investable_precheck_only":int(len(investable)),
           "status_counts":dict(Counter(original.opportunity_status))}
    return original,investable,budgets,stats


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asof",required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,help="original NAV cache + full fund_list.csv; reintroduces old excluded funds")
    p.add_argument("--liquidity-evidence",type=Path)
    p.add_argument("--valuation-evidence",type=Path)
    p.add_argument("--forecast-evidence",type=Path)
    p.add_argument("--benchmark-history",type=Path,
                   help="dated group,date,nav,known_at,source_url observations for EB M1")
    p.add_argument("--cash-rate-annual",type=float)
    p.add_argument("--cash-source",help="point-in-time public cash-yield/benchmark source URL")
    p.add_argument("--cash-known-at")
    p.add_argument("--decision-months",type=int,choices=[3,6,12],default=6)
    p.add_argument("--order-amount",type=float,help="required investment amount, same currency as limits")
    p.add_argument("--scenarios",type=int,default=256)
    p.add_argument("--block-length",type=int,default=10)
    p.add_argument("--grid-points",type=int,default=101)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(argv)
    if a.scenarios<32 or a.grid_points<2 or a.block_length<1:
        p.error("invalid simulation settings")
    if a.order_amount is not None and a.order_amount <= 0:
        p.error("order amount must be positive")
    asof=pd.Timestamp(a.asof)
    if a.output.resolve()==a.snapshot.resolve():
        p.error("output must differ from source snapshot")
    old=json.loads((a.snapshot/"audit.json").read_text(encoding="utf-8-sig"))
    if pd.Timestamp(old["asof"])!=asof:
        p.error("snapshot cutoff differs from asof")

    cash_ok=validate_cash_assumption(a,asof)
    liquid_evidence=read_timestamped_evidence(a.liquidity_evidence,asof,
        ("subscription_status","redemption_status"))
    valuation_evidence=read_timestamped_evidence(a.valuation_evidence,asof,
        ("pricing_method","economic_mdd_upper"))
    forecast_evidence=read_timestamped_evidence(a.forecast_evidence,asof,
        ("trained_through","model_id","forecast_mu_annual","decision_months"))
    path_cache=None
    catalog=None
    if a.data_dir is not None:
        catalog=load_catalog(a.data_dir/"fund_list.csv")
        path_cache,path_audit=load_paths(catalog,a.data_dir,asof)
    if a.benchmark_history and path_cache is None:
        p.error("--benchmark-history requires --data-dir and full NAV paths")
    benchmark=(pd.read_csv(a.benchmark_history) if a.benchmark_history else None)
    a.output.mkdir(parents=True,exist_ok=True)
    results={}
    for h in HORIZONS:
        frame=(source_fund_rows(a.snapshot,h) if catalog is None else
               raw_fund_rows(catalog,path_cache,asof,h))
        codes=frame.code.astype(str).str.zfill(6).unique().tolist()
        liq=resolve_liquidity(codes,liquid_evidence,asof,a.decision_months)
        forecasts_for_h=forecast_evidence
        if benchmark is not None:
            # This is an UNVALIDATED candidate forecast; the source records
            # and every training date are checked against the formation date.
            model=fit_category_forecasts(frame.loc[frame.group.notna()],
                path_cache,benchmark,asof,HORIZONS[h][0],a.decision_months)
            model.to_csv(a.output/f"model_diagnostics_{h}.csv",
                         index=False,encoding="utf-8-sig")
            if len(model):
                forecasts_for_h=pd.concat([model,forecast_evidence],ignore_index=True)
        fc=resolve_forecasts(codes,forecasts_for_h,asof,a.decision_months)
        research,investable,sensitivity,stats=analyze_horizon(
            frame,liq,valuation_evidence,fc,asof,a.decision_months,
            cash_ok,a,path_cache)
        research.to_csv(a.output/f"research_{h}.csv",index=False,encoding="utf-8-sig")
        investable.to_csv(a.output/f"investable_{h}.csv",index=False,encoding="utf-8-sig")
        # Output provisional model ranking only if backed by ALL inputs.
        provisional=research.loc[research.score_annual.gt(0)].sort_values(
            ["score_annual","code"],ascending=[False,True]).head(10)
        provisional.to_csv(a.output/f"top10_experimental_{h}.csv",index=False,encoding="utf-8-sig")
        for budget,rows in sensitivity.items():
            pd.DataFrame(rows,columns=["code","budget","opportunity_score_annual",
                  "weight","risk_upper","status"]).to_csv(
                a.output/f"budget_{int(100*budget)}pct_{h}.csv",index=False,encoding="utf-8-sig")
        results[h]=stats
    conditions={"trade": bool(a.liquidity_evidence), "valuation": bool(a.valuation_evidence),
                "forecast": bool(a.forecast_evidence) or bool(a.benchmark_history),
                "benchmark_history":bool(a.benchmark_history),
                "cash":cash_ok,"nav_paths":a.data_dir is not None,
                "asof_historical_catalog":False,
                "independent_oos_validation":False}
    out=dict(asof=a.asof, source_snapshot=str(a.snapshot), data_mode=(
             "full_local_nav_catalog" if catalog is not None else "published_partial_snapshot"),
             experiment_only=True,certified_investable_ranking=False,
             oos_acceptance="EVIDENCE_INSUFFICIENT",
             explanation="No pretrained model / economic-risk verification or independent PIT OOS has been inferred from NAV alone",
             assumption_fields=conditions,decision_months=a.decision_months,
             comparison_budgets=list(BUDGETS),seed=a.seed,block_length=a.block_length,
             scenarios=a.scenarios,grid_points=a.grid_points,cash_rate=a.cash_rate_annual,
             cash_source=a.cash_source or "",cash_known_at=a.cash_known_at or "",
             full_snapshot_exclusions_retained_for_audit=(
                old.get("periodic_unverified_excluded",0) if catalog is None else 0),
             horizons=results)
    (a.output/"audit.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",
                                       encoding="utf-8")
    print(json.dumps({"oos_acceptance":out["oos_acceptance"],"horizons":results},
                     ensure_ascii=False),flush=True)
    return out


if __name__=="__main__":
    main()
