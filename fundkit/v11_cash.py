"""v1.1 dynamic investment budget; research only, no broker connection.

The deferred ledger is a SUBLEDGER of available cash. It is never added to
state['cash'] or to NAV. Only actual confirmed BUY fills retire deferred cash.
"""
from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class TimingConfig:
    base_fraction: float = 1.0
    release_fraction: float = .50
    deferred_horizon_days: int = 10  # trading sessions, not calendar days
    cash_reserve_weight: float = .05
    min_signal_score: float = 0.0

    def validate(self):
        if not (0 <= self.base_fraction <= 1 and 0 <= self.release_fraction <= 1 and
                0 <= self.cash_reserve_weight < 1 and self.deferred_horizon_days >= 1 and
                math.isfinite(self.min_signal_score)):
            raise ValueError('B6 timing configuration invalid')


class DeferredCashLedger:
    """A cash-planning subledger with per-day fill acknowledgements.

    Only external inflows are allocated to this pool. Previously held cash and
    unsettled redemption proceeds are not added implicitly. B6 must receive
    an explicit simulator fill callback after *every* policy invocation.
    """
    def __init__(self):
        self._lots = []
        self._session = 0
        self._pending_day = None
        self._last_day = None
        self._inflow_this_day = 0.
        self._previous_amount = 0.
        self._approved_buy = None

    @property
    def total(self):
        return float(sum(x['amount'] for x in self._lots))

    def begin(self, day, inflow):
        day = str(pd.Timestamp(day).date())
        if self._pending_day is not None:
            raise RuntimeError('FILL_ACK_REQUIRED: prior decision lacks execution acknowledgement')
        if self._last_day is not None and day <= self._last_day:
            raise ValueError('OUT_OF_ORDER_SESSION: settlement policy dates not increasing')
        if not math.isfinite(float(inflow)) or float(inflow) < 0:
            raise ValueError('DATA_BLOCKED: invalid external inflow')
        self._session += 1
        self._previous_amount = self.total
        self._inflow_this_day = float(inflow)
        if float(inflow) > 0:
            self._lots.append({'day': day, 'session': self._session, 'amount': float(inflow)})
        self._pending_day = day
        self._last_day = day

    def oldest_age(self):
        return max((self._session - x['session'] for x in self._lots if x['amount'] > 1e-10), default=0)

    def plan(self, cfg: TimingConfig, *, available_cash: float, total_value: float,
             risk_pass: bool, eligible_signal: bool):
        cfg.validate()
        if self._pending_day is None:
            raise RuntimeError('begin(day, inflow) must precede plan')
        cash = float(available_cash)
        value = float(total_value)
        if (not math.isfinite(cash) or not math.isfinite(value) or
                cash < 0 or value < 0):
            raise ValueError('DATA_BLOCKED: invalid current cash/NAV')
        floor = cfg.cash_reserve_weight * value
        spendable = max(0., cash - floor)
        base = cfg.base_fraction * self._inflow_this_day
        # Old funds can be accelerated only by AS-OF signals and a verified risk gate.
        tactical = (cfg.release_fraction * self._previous_amount
                    if (risk_pass and eligible_signal) else 0.0)
        amount = min(spendable, self.total, base + tactical)
        if amount < 0 or not math.isfinite(amount):
            raise AssertionError('negative or nonfinite investment budget')
        self._approved_buy = amount
        return {
            'approved_buy_cny': amount,
            'base_budget_cny': min(base, amount),
            'tactical_release_cny': min(tactical, max(0.0, amount - min(base, amount))),
            'deferred_cash_before_cny': self.total,
            'oldest_deferred_trading_days': self.oldest_age(),
            'expired_review_required': self.oldest_age() >= cfg.deferred_horizon_days,
            'tactical_blocked': not (risk_pass and eligible_signal),
            'cash_reserve_floor_cny': floor,
        }

    def ack_execution(self, day, events):
        day = str(pd.Timestamp(day).date())
        if day != self._pending_day:
            raise ValueError('FILL_ACK_MISMATCH: acknowledgement for wrong day')
        actual_buys = 0.
        for x in events:
            if (x.get('event') == 'BUY' and str(x.get('date')) == day):
                amt = float(x['amount'])
                if not math.isfinite(amt) or amt < 0:
                    raise ValueError('FILL_ACK_INVALID_BUY')
                actual_buys += amt
        if self._approved_buy is not None and actual_buys > self._approved_buy + .02:
            raise ValueError('FILL_ACK_OVER_BUDGET: actual buy exceeds approved amount')
        if actual_buys > self.total + 1.01:  # BUY log rounds to cents
            raise ValueError('CASH_LEDGER_OVERSPEND: BUY > deferred subledger')
        remaining = actual_buys
        for lot in self._lots:
            take = min(lot['amount'], remaining)
            lot['amount'] -= take
            remaining -= take
        self._lots = [lot for lot in self._lots if lot['amount'] > 1e-7]
        self._pending_day = None
        self._approved_buy = None
        return {'actual_buys_cny': actual_buys, 'deferred_after_cny': self.total}

    def snapshot(self):
        return {'session': self._session, 'last_day': self._last_day,
                'pending_day': self._pending_day, 'lots': [dict(x) for x in self._lots]}
