
"""
Reconstructs a user's financial state and forecasts their balance forward
for a 90-day window, per the "90-Day Safety Check" rules in the problem
statement.
"""
from datetime import timedelta
from collections import defaultdict
from statistics import median

FORECAST_DAYS = 90

# Categories that represent one-off / non-recurring activity and should
# never be auto-projected forward from history, even if they happen to
# repeat a few times by chance.
NON_RECURRING_CATEGORIES = {
    "windfall", "investment", "work_expense",
}
NON_RECURRING_EVENT_TYPES = {
    "investment_purchase", "investment_valuation", "investment_sale", "refund",
}


def build_recurring_templates(events, as_of):
    """For a user's settled events strictly before `as_of`, detect recurring
    (category, direction) patterns and return a list of templates:
    {category, direction, currency, avg_amount, interval_days, last_date,
     flexibility, minimum_allowed_amount, event_type}
    """
    groups = defaultdict(list)
    for e in events:
        if e["status"] != "settled":
            continue
        if e["event_type"] in NON_RECURRING_EVENT_TYPES:
            continue
        if e["category"] in NON_RECURRING_CATEGORIES:
            continue
        d = e["_settlement_date"] or e["_event_date"]
        if d is None or d >= as_of:
            continue
        groups[(e["category"], e["direction"])].append(e)

    templates = []
    for (category, direction), rows in groups.items():
        rows.sort(key=lambda e: e["_settlement_date"] or e["_event_date"])
        if len(rows) < 3:
            continue
        dates = [r["_settlement_date"] or r["_event_date"] for r in rows]
        diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        diffs = [d for d in diffs if d > 0]
        if not diffs:
            continue
        interval = median(diffs)
        if interval < 3 or interval > 100:
            # Not a plausible recurring cadence (too rapid or too sparse).
            continue
        recent = rows[-3:]
        amounts = [r["_amount_home"] for r in recent if r["_amount_home"] is not None]
        if not amounts:
            continue
        avg_amount = sum(amounts) / len(amounts)
        last = recent[-1]
        templates.append({
            "category": category,
            "direction": direction,
            "event_type": last["event_type"],
            "avg_amount": avg_amount,
            "interval_days": interval,
            "last_date": dates[-1],
            "flexibility": last.get("flexibility") or "fixed",
            "minimum_allowed_amount": last.get("minimum_allowed_amount"),
        })
    return templates


def confirmed_future_events(events, window_start, window_end):
    """Explicit dataset rows that should be counted as confirmed future cash
    movements within [window_start, window_end]:
      - status == 'scheduled' or 'pending' debits (expense/debt_payment)
      - status == 'scheduled' credits ONLY for salary (confirmed income)
      - status == 'settled' rows whose settlement date happens to fall in
        the future window (defensive, should be rare)
    Excludes cancelled, failed, unrealized, and any pending/scheduled
    credit that is not salary (bonuses, refunds, investment gains, windfalls
    must not be counted until settled).
    """
    out = []
    for e in events:
        d = e["_settlement_date"] or e["_event_date"]
        if d is None or d < window_start or d > window_end:
            continue
        status = e["status"]
        if status in ("cancelled", "failed", "unrealized"):
            continue
        if status == "settled":
            out.append(e)
            continue
        if status not in ("scheduled", "pending"):
            continue
        if e["direction"] == "credit":
            if e["category"] == "salary":
                out.append(e)
            # other future credits (refunds, bonuses, investment gains,
            # windfalls) are deliberately excluded until settled.
            continue
        # debit / non_cash handled normally
        if e["direction"] == "debit":
            out.append(e)
    return out


class Forecast:
    """Builds a day-indexed running-balance array for [request_date,
    request_date + FORECAST_DAYS] and answers safety questions against it."""

    def __init__(self, starting_balance, window_start, deltas):
        """deltas: list of (date, signed_amount_in_home_currency, event_ref)"""
        self.window_start = window_start
        self.n_days = FORECAST_DAYS
        self.daily_delta = [0.0] * (self.n_days + 1)
        self.events_by_day = defaultdict(list)
        for d, amount, ref in deltas:
            offset = (d - window_start).days
            if offset < 0:
                offset = 0  # already-due amounts on/near the request date
            if offset > self.n_days:
                continue
            self.daily_delta[offset] += amount
            self.events_by_day[offset].append((amount, ref))
        self.balance = [0.0] * (self.n_days + 1)
        running = starting_balance
        for day in range(self.n_days + 1):
            running += self.daily_delta[day]
            self.balance[day] = running
        # suffix minimum: min balance from day d to the end of the window
        self.suffix_min = [0.0] * (self.n_days + 1)
        m = float("inf")
        for day in range(self.n_days, -1, -1):
            m = min(m, self.balance[day])
            self.suffix_min[day] = m

    def min_balance_from(self, day_offset):
        day_offset = max(0, min(day_offset, self.n_days))
        return self.suffix_min[day_offset]

    def amount_safe_to_pay(self, requested_amount, minimum_balance_to_keep):
        headroom = self.min_balance_from(0) - minimum_balance_to_keep
        return max(0.0, min(requested_amount, headroom))

    def earliest_full_payment_day(self, requested_amount, minimum_balance_to_keep):
        for day in range(0, self.n_days + 1):
            if self.min_balance_from(day) - requested_amount >= minimum_balance_to_keep:
                return day
        return None

    def safe_after_payments(self, payments, minimum_balance_to_keep):
        """payments: list of (day_offset, amount). Returns True if applying
        all payments (on top of the existing forecast) keeps every point in
        the window at/above minimum_balance_to_keep."""
        extra = [0.0] * (self.n_days + 1)
        for day, amt in payments:
            day = max(0, min(day, self.n_days))
            extra[day] += amt
        running_extra = 0.0
        for day in range(self.n_days + 1):
            running_extra += extra[day]
            if self.balance[day] - running_extra < minimum_balance_to_keep - 1e-6:
                return False
        return True
          
