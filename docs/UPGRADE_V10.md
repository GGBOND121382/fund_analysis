# v1.0 Risk-Budgeted Fund Strategy Upgrade

## Scope and status

This is an algorithmic research upgrade on top of v0.9. It does not place orders and does not establish that the account can achieve the user's objectives.

Fixed objectives:
- Long-run annualized net return strictly greater than 5%.
- Portfolio maximum drawdown no greater than 4%.
- Risk budget has priority over return maximization.
- If evidence is insufficient, report it explicitly; do not relax either target.

This release addresses recent-return chasing, concentrated single-fund candidate weights, incomplete B5 exit behavior, and overlapping OOS aggregation.

## 1. Robust expected-return estimate

For asset i, estimate annualized sample mean and volatility:

\[
\hat\mu_i=252\bar r_i,\qquad \hat\sigma_i=\sqrt{252}s_i.
\]

Use the category median as prior when at least three funds are available in the category; otherwise use the cross-sectional median:

\[
\mu_{0,c}=\operatorname{median}_{j\in c}(\hat\mu_j).
\]

Shrink the sample estimate:

\[
a_n=\frac{n}{n+\tau_\mu},\qquad
\hat\mu_i^{shr}=a_n\hat\mu_i+(1-a_n)\mu_{0,c}.
\]

Subtract an estimation-uncertainty allowance:

\[
\hat\mu_i^{LCB}=
\hat\mu_i^{shr}-z_\mu\hat\sigma_i\sqrt{\frac{252}{n}}.
\]

Defaults are tau_mu = 504 observations and z_mu = 1. These are regularization defaults to validate, not proven optimal constants.

For each candidate portfolio, use the lower of the weighted conservative return estimate and moving-block-bootstrap 25th-percentile annualized return as its conservative return score. Among candidates satisfying the return and drawdown constraints, rank by the 25th-percentile scenario Sharpe, then conservative return, drawdown and turnover. Sharpe uses a configurable annual risk-free assumption (default 2%) and includes the initial turnover fee in the first daily return. The 2% rate is an explicit assumption, not a live yield feed.

## 2. Concentration and transaction-cost guardrails

The default single-fund target-weight cap is 40%. This is an explicit governance guardrail intended to prevent a short-history fund from occupying nearly 60-80% of the candidate portfolio. It is not an industry-derived optimum and remains configurable.

The default one-way turnover-cost proxy is 30bp in target-weight search. The joint strategy search uses 30bp and 60bp uniform fee stress cases. These are stress assumptions, not actual fund-specific fee schedules. Actual subscription fees, holding-period redemption fees, discounts, minimum charges, and fund-specific exceptions are still required before a live recommendation.

## 3. Covariance risk contributions for portfolio de-risking

The B5 portfolio exit no longer redeems every fund by the same fraction. It aligns published fund total-return histories on common dates, estimates a diagonal-shrunk covariance matrix, and calculates volatility risk contributions:

\[
RC_i=\frac{w_i(\Sigma w)_i}{\sqrt{w^\top\Sigma w}}.
\]

Only positive contributions rank potential reductions. The policy selects contributors until the configured coverage ratio is reached, then applies the configured partial-redemption fraction.

This is a covariance/volatility contribution proxy, not a CDaR or expected-shortfall contribution. If common history is insufficient, the policy pauses new purchases but blocks risk-ranked redemption rather than inventing a ranking.

## 4. B5 fund-level stop-loss and trailing take-profit

### Stop-loss

Fund drawdown is calculated from the published total-return NAV high-water mark over a rolling lookback. A category-aware volatility-adjusted threshold is:

\[
\tau_{i,t}^{stop}=
\min(\tau_{cap},\max(\tau_{base,i},k_{c(i)}\hat\sigma_{20,i})).
\]

When fund drawdown reaches the threshold:
- pause new purchases for that fund;
- if redemption is enabled and units are redeemable, redeem the configured fraction;
- if units are locked, record a blocked-redemption event and keep monitoring;
- a deeper drawdown or newly matured redeemable lots can trigger a subsequent tranche;
- resume only after recovery threshold and cooldown conditions are satisfied.

The category multipliers and threshold grid are candidate parameters, not empirically established optimal stop levels.

### Trailing take-profit

A take-profit trigger requires both:
1. Return on redeemable units, calculated from a known cost basis, has reached the activation threshold.
2. Total-return NAV has pulled back from its rolling high-water mark by the configured trailing threshold.

Then the policy redeems the configured fraction of redeemable units and applies a cooldown.

The simulator tracks cost basis for simulated purchases and counterfactual backtest-start lots. Initial counterfactual purchases are charged the same subscription-fee stress rate as subsequent purchases; the fee is included in the starting unit-NAV loss, Sharpe, TWR, XIRR, fee totals and turnover. Partial redemption releases cost basis pro rata. Snapshot-derived synthetic lots deliberately have unknown historical cost basis; live take-profit is therefore blocked for those existing lots until verified lot-level acquisition cost is supplied. A snapshot market value is not treated as the investor's purchase cost.

### Portfolio-level guard

The portfolio soft trigger is the 4% budget minus an internal buffer. It pauses new purchases and, when research redemption is enabled, reduces the highest positive covariance-risk contributors. Recovery requires portfolio drawdown/CDaR conditions and a cooldown period.

The 4% target is not a guaranteed hard loss limit. NAV publication delays, fund dealing cutoffs, lockups, and redemption settlement mean an observed threshold can be crossed before the portfolio can fully exit.

## 5. Fund NAV history and fee accounting

The simulator exposes only total-return NAV observations published by the decision timestamp. It also exposes total/redeemable shares, market value, cost basis, whether the basis is known, and redeemable position return when the basis is known.

BUY and REDEEM events record gross amount, fee, net proceeds/investment, and cost basis released. Turnover counts both purchases and redemptions rather than sells only.

The policy default remains allow_redeem = false. The research joint optimizer explicitly enables redemption in its historical simulator; this does not authorize live automated redemption.

## 6. Equity-index fund scope

Optimization and B2 dynamic allocation include all fund categories in the account snapshot, including CSI 300, S&P 500, and Nasdaq index funds. This treats index funds as the user's selected fund asset classes; it does not introduce direct stock trading.

B1 remains the explicit baseline that stops new equity-index fund purchases. Existing index-fund positions remain in portfolio valuation and risk calculations.

## 7. OOS integrity

build_rolling_folds rejects overlapping test windows when more than one fold is requested: step_days must be at least test_days.

The legacy evaluation report labels short diagnostic windows, one primary 12-month window, and the overlapping 24-month diagnostic separately. Overlapping diagnostics are excluded from primary aggregation. At least two independent full-year test windows are required before the evaluation can claim cross-window OOS validation.

Given the current common-history limitation (previously 507 common trading days), the expected outcome may remain INSUFFICIENT_INDEPENDENT_OOS_WINDOWS. That is the correct result, not a failure to be hidden by shortening the test window.

## 8. What remains unverified / not implemented

This upgrade does not yet provide:
- a complete holdings-based bond risk engine with verified modified duration, spread duration, bank capital-bond exposure, and liquidity measures;
- persistent storage/reconciliation for B5 pause states, recovery counters, redemption cooldowns, and high-water marks across monitor process restarts;
- historical fund-specific subscription/redemption fee schedules and actual account acquisition batches (the backtest uses explicit uniform fee stress assumptions, including initial counterfactual purchases);
- reliable historical dealing cutoffs and settlement calendars for each QDII fund;
- multiple independent full-year OOS windows for the complete current fund universe;
- proof that any strategy achieves net CAGR >5% while keeping MDD <=4%; Sharpe ranking is an additional selection metric, not a substitute for either hard research target.

Do not describe candidate weights or stop/take-profit thresholds as investment recommendations until local data audit, fee/lot integration, and independent OOS validation pass. B5's state is currently held in the policy closure during a simulation; without a persistent state store and order/position reconciliation it is not a reliable live monitoring or execution service.

## 9. Verification

Run:

    python -m compileall -q fundkit tests main.py
    python -m unittest discover -s tests -p "test_*.py" -v

Research commands:

    python main.py optimize-weights --asof YYYY-MM-DD
    python main.py optimize-strategy --asof YYYY-MM-DD
    python main.py validate

All outputs remain research-only. No command in this upgrade sends a real subscription or redemption order.
