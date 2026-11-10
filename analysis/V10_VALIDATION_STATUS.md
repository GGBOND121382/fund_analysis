# v1.0.1 Validation Status

Date: 2026-11-11  
Objective: annualized net return > 5%; portfolio maximum drawdown <= 4%; risk-first.

## Code state

The v1.0.1 research branch contains:
- robust/shrinkage return estimates and a single-fund 40% governance cap;
- block-bootstrap drawdown and return scenarios;
- historical and scenario Sharpe reporting, with Sharpe-first ranking only after both return and drawdown constraints are met;
- fund-level stop-loss and cost-basis-aware trailing take-profit;
- initial counterfactual subscription fees included in unit NAV, TWR, CAGR, Sharpe, XIRR, fees and turnover;
- missing fund history blocks new purchases rather than assuming zero drawdown;
- non-overlapping OOS guardrails and explicit DATA_BLOCKED reports;
- a one-command local pipeline: `python scripts/run_v10_research.py`.

## What has and has not been verified

- Unit tests and compile checks are run in GitHub Actions; see the workflow for the current commit.
- This repository intentionally does not contain the user's local `data/nav/`, `data/factors/`, `config/account_snapshot.json`, or current trade-status cache. GitHub Actions therefore cannot reproduce the real account's data-dependent backtest.
- `analysis/README.md` and the other committed `analysis/*` reports are the v0.9 historical export. They must not be cited as v1.0.1 results.
- The previous v0.9 export reported 507 common trading days for the 13-fund universe and blocked a 504+252+252 day joint walk-forward. This is a historical data-coverage finding; rerun the local pipeline to confirm the current cache.
- No result currently establishes that any strategy satisfies both the 5% net-return target and 4% drawdown budget across independent full-year OOS windows.
- Account drawdown remains unverifiable without actual account cashflows and acquisition lots. Counterfactual start portfolios are not actual account history.
- Uniform fee stress assumptions are not actual fund-specific fee schedules.

## Required local acceptance

Run:

```bash
python scripts/run_v10_research.py --asof YYYY-MM-DD
```

Review `reports/v10_research_run.json`, `reports/weight_optimization_research.json`,
`reports/joint_strategy_optimization.json`, `reports/oos_summary.json`,
`reports/OOS_BACKTEST_REPORT.md`, and `reports/OPTIMIZATION_RESULT.json`.

A `DATA_BLOCKED`, `INSUFFICIENT_INDEPENDENT_OOS_WINDOWS`, `RETURN_TARGET_UNMET`,
or `RISK_CONSTRAINT_UNMET` result is not a successful strategy acceptance. Do not
promote a research candidate into live target weights or automated orders.
