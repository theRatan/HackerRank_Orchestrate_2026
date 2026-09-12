
"""
Turns a reconstructed financial forecast into the required output row for
one request, following the allowed-value and ranking rules in
problem_statement.md.
"""
from datetime import timedelta
from .forecast import (
    Forecast, build_recurring_templates, confirmed_future_events, FORECAST_DAYS,
)
from .data import parse_date, to_float

SIGN = {"debit": -1, "credit": 1, "non_cash": 0}


def prepare_user_events(ds, user_id, fx):
    """Annotate each of a user's events with parsed dates and home-currency
    amount, ready for forecasting."""
    profile = ds.profiles[user_id]
    home_ccy = profile["home_currency"]
    out = []
    for e in ds.events_by_user.get(user_id, []):
        e = dict(e)
        e["_event_date"] = parse_date(e.get("event_date"))
        e["_settlement_date"] = parse_date(e.get("settlement_date")) or e["_event_date"]
        amt = ds.event_amount(e)
        d = e["_settlement_date"] or e["_event_date"]
        e["_amount_home"] = fx.convert(amt, e["currency"], home_ccy, d) if amt is not None else None
        out.append(e)
    return out


def build_forecast(ds, fx, user_id, request_date):
    profile = ds.profiles[user_id]
    home_ccy = profile["home_currency"]
    starting_balance = to_float(profile["current_available_balance"])
    minimum_balance = to_float(profile["minimum_balance_to_keep"])

    events = prepare_user_events(ds, user_id, fx)
    window_start = request_date
    window_end = request_date + timedelta(days=FORECAST_DAYS)

    deltas = []

    # 1) Confirmed explicit future events (scheduled/pending debits,
    #    scheduled salary credits, and any settled row that happens to be
    #    dated within the window).
    confirmed = confirmed_future_events(events, window_start, window_end)
    confirmed_keys = set()
    for e in confirmed:
        if e["_amount_home"] is None:
            continue
        d = e["_settlement_date"] or e["_event_date"]
        signed = SIGN.get(e["direction"], 0) * e["_amount_home"]
        if signed == 0:
            continue
        deltas.append((d, signed, {"event_id": e["event_id"], "category": e["category"]}))
        confirmed_keys.add((e["category"], e["direction"]))

    # 2) Recurring background cash flow, projected forward from settled
    #    history. Skip a (category, direction) pair already covered by an
    #    explicit confirmed event landing in the window, to avoid double
    #    counting the same real-world commitment.
    #    Salary is deliberately NOT auto-projected: only the explicit
    #    confirmed 'scheduled' salary event counts as future income, per
    #    the "count confirmed salary only on its settlement date" rule.
    templates = build_recurring_templates(events, window_start)
    for t in templates:
        if t["category"] == "salary" and t["direction"] == "credit":
            continue
        if (t["category"], t["direction"]) in confirmed_keys:
            # still project further-out occurrences beyond the explicit one
            pass
        next_date = t["last_date"]
        interval = max(3, int(round(t["interval_days"])))
        # advance to first occurrence >= window_start
        while next_date < window_start:
            next_date = next_date + timedelta(days=interval)
        signed_amt = SIGN.get(t["direction"], 0) * t["avg_amount"]
        d = next_date
        while d <= window_end:
            # avoid landing within 10 days of an explicit confirmed event of
            # the same category (that explicit row already represents this
            # occurrence).
            clash = any(
                ref["category"] == t["category"] and abs((dd - d).days) <= 10
                for dd, amt, ref in deltas if "category" in ref
            )
            if not clash and signed_amt != 0:
                deltas.append((d, signed_amt, {"event_id": None, "category": t["category"]}))
            d = d + timedelta(days=interval)

    fc = Forecast(starting_balance, window_start, deltas)
    return fc, minimum_balance, home_ccy


def fmt_amount(x):
    # keep integers clean, otherwise 2 decimal places
    r = round(x, 2)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.2f}"


def eligible_installment_options(ds, request, fc, minimum_balance, home_ccy, fx, profile):
    accepted = set((profile.get("payment_methods_user_will_consider") or "").split("|"))
    if "installments" not in accepted:
        return []
    max_months = to_float(profile.get("max_installment_months"))
    request_date = request["_request_date"]
    options = []
    for opt in ds.payment_options_by_request.get(request["request_id"], []):
        if opt["payment_method"] != "installments":
            continue
        n = int(to_float(opt["number_of_payments"]) or 0)
        freq = to_float(opt["payment_frequency_days"]) or 30
        first_date = parse_date(opt["first_payment_date"])
        pay_amt = to_float(opt["payment_amount"])
        total = to_float(opt["total_payable_amount"])
        if n <= 0 or first_date is None or pay_amt is None:
            continue
        span_months = (n - 1) * freq / 30.0
        if max_months is not None and span_months > max_months + 1e-6:
            continue
        schedule = [first_date + timedelta(days=int(round(freq * i))) for i in range(n)]
        last_date = schedule[-1]
        if last_date > request_date + timedelta(days=FORECAST_DAYS):
            continue
        payments_offsets = [((d - request_date).days, pay_amt) for d in schedule]
        if not fc.safe_after_payments(payments_offsets, minimum_balance):
            continue
        options.append({
            "payment_option_id": opt["payment_option_id"],
            "schedule": schedule,
            "pay_amt": pay_amt,
            "n": n,
            "total": total,
            "last_date": last_date,
        })
    options.sort(key=lambda o: (o["total"], o["schedule"][0], o["n"], o["payment_option_id"]))
    return options


def decide(ds, fx, request):
    user_id = request["user_id"]
    profile = ds.profiles[user_id]
    home_ccy = profile["home_currency"]
    request_date = parse_date(request["request_date"])
    request["_request_date"] = request_date
    desired_completion = parse_date(request["desired_completion_date"])
    requested_amount = to_float(request["requested_amount"])
    allows_partial = str(request.get("allows_partial_payment", "")).strip().lower() == "true"
    accepted_methods = set((profile.get("payment_methods_user_will_consider") or "").split("|"))

    fc, minimum_balance, _ = build_forecast(ds, fx, user_id, request_date)

    amount_safe_to_pay = fc.amount_safe_to_pay(requested_amount, minimum_balance)
    earliest_day = fc.earliest_full_payment_day(requested_amount, minimum_balance)
    earliest_date = request_date + timedelta(days=earliest_day) if earliest_day is not None else None

    full_now_ok = amount_safe_to_pay >= requested_amount - 1e-6 and "full_payment" in accepted_methods

    partial_ok = (
        allows_partial
        and "partial_payment" in accepted_methods
        and 0 < amount_safe_to_pay < requested_amount - 1e-9
        and earliest_date is not None
        and earliest_date <= desired_completion
    )

    installment_opts = eligible_installment_options(ds, request, fc, minimum_balance, home_ccy, fx, profile)
    installments_ok = bool(installment_opts) and installment_opts[0]["last_date"] <= desired_completion

    wait_ok = (
        not full_now_ok
        and earliest_date is not None
        and "full_payment" in accepted_methods
    )

    explanation_balance = fmt_amount(fc.min_balance_from(0))

    if full_now_ok:
        status = "affordable_now"
        method = "full_payment"
        plan = f"{request_date.isoformat()}:{fmt_amount(requested_amount)}"
        earliest_out = request_date.isoformat()
        explanation = (
            f"Pay {home_ccy} {fmt_amount(requested_amount)} on {request_date.isoformat()}. "
            f"The 90-day forecast keeps the balance at or above the {home_ccy} "
            f"{fmt_amount(minimum_balance)} minimum throughout."
        )
    elif partial_ok:
        status = "affordable_with_plan"
        method = "partial_payment"
        remainder = requested_amount - amount_safe_to_pay
        plan = (
            f"{request_date.isoformat()}:{fmt_amount(amount_safe_to_pay)}"
            f"|{earliest_date.isoformat()}:{fmt_amount(remainder)}"
        )
        earliest_out = earliest_date.isoformat()
        explanation = (
            f"Pay {home_ccy} {fmt_amount(amount_safe_to_pay)} on {request_date.isoformat()}, "
            f"then the remaining {home_ccy} {fmt_amount(remainder)} on {earliest_date.isoformat()} "
            f"once the balance recovers above the {home_ccy} {fmt_amount(minimum_balance)} minimum."
        )
    elif installments_ok:
        status = "affordable_with_plan"
        method = "installments"
        opt = installment_opts[0]
        plan = "|".join(f"{d.isoformat()}:{fmt_amount(opt['pay_amt'])}" for d in opt["schedule"])
        earliest_out = earliest_date.isoformat() if earliest_date else ""
        explanation = (
            f"Use {opt['n']} installments of {home_ccy} {fmt_amount(opt['pay_amt'])} starting "
            f"{opt['schedule'][0].isoformat()} (option {opt['payment_option_id']}). This keeps the "
            f"balance at or above the {home_ccy} {fmt_amount(minimum_balance)} minimum throughout."
        )
    elif wait_ok:
        status = "affordable_later"
        method = "wait"
        plan = f"{earliest_date.isoformat()}:{fmt_amount(requested_amount)}"
        earliest_out = earliest_date.isoformat()
        late_note = "" if earliest_date <= desired_completion else " This is after the requested completion date."
        explanation = (
            f"Wait until {earliest_date.isoformat()}, then pay {home_ccy} {fmt_amount(requested_amount)} "
            f"in full. Paying sooner would take the balance below the {home_ccy} "
            f"{fmt_amount(minimum_balance)} minimum.{late_note}"
        )
    else:
        status = "not_affordable"
        method = "not_recommended"
        plan = "none"
        earliest_out = ""
        explanation = (
            f"No accepted payment method safely completes this {home_ccy} {fmt_amount(requested_amount)} "
            f"request within the 90-day forecast while keeping the balance at or above the {home_ccy} "
            f"{fmt_amount(minimum_balance)} minimum."
        )

    amount_safe_to_pay = max(0.0, min(requested_amount, amount_safe_to_pay))

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": fmt_amount(amount_safe_to_pay),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan,
        "earliest_date_for_full_payment": earliest_out,
        "spending_changes_needed": "none",
        "decision_explanation": explanation,
              }
      
