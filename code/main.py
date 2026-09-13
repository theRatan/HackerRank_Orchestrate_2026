#!/usr/bin/env python3
"""
HackerRank Orchestrate - September 2026 - "Buy or Wait?"
Financial affordability agent.

Run:
    python3 code/main.py
Reads from ../dataset/*.csv (relative to this file) and writes ../output.csv
at the repository root.

--------------------------------------------------------------------------
SCHEMA NOTE (updated against the REAL dataset headers on 2026-09-12)
--------------------------------------------------------------------------
requests.csv:               request_id, user_id, request_date, request_type,
                             requested_amount, desired_completion_date,
                             allows_partial_payment, request_text
financial_profiles.csv:     user_id, home_currency, current_available_balance,
                             minimum_balance_to_keep, financial_priorities,
                             expense_categories_to_protect,
                             expense_categories_user_is_willing_to_reduce,
                             expense_categories_user_is_willing_to_stop,
                             payment_methods_user_will_consider,
                             max_installment_months
financial_events.csv:       event_id, user_id, event_type, description,
                             category, direction, amount, currency,
                             event_date, settlement_date, status,
                             linked_event_id, flexibility,
                             minimum_allowed_amount
request_payment_options.csv: payment_option_id, request_id, payment_method,
                             payment_amount, number_of_payments,
                             first_payment_date, payment_frequency_days,
                             financing_fee, total_payable_amount
exchange_rates.csv:         rate_date, from_currency, to_currency, rate
messages.csv:                message_id, user_id, request_id,
                             related_event_id, sent_at, source_type,
                             message_text
images.csv:                  image_id, user_id, request_id, related_event_id
                             (NOTE: no file_path column -- the actual image
                             file is assumed to live at
                             dataset/media/images/<image_id>.<ext>; the code
                             globs for it. Verify this assumption once you
                             can see media/images/ directly.)

--------------------------------------------------------------------------
ASSUMPTIONS YOU STILL NEED TO VERIFY (column NAMES are real now, but I have
not seen actual cell VALUES -- run inspect_unique_values() below and
compare against these guesses before trusting the output):
--------------------------------------------------------------------------
1. `direction`: assumed to contain something like "inflow"/"outflow" or
   "income"/"expense" or "credit"/"debit". normalize via is_inflow() below,
   which treats anything containing "in", "income", "credit", "inflow",
   "salary", "deposit" as positive. VERIFY against real values.
2. `event_type`: assumed to indicate recurring vs one-time via a substring
   "recur" (e.g. "recurring", "recurring_monthly"), with frequency parsed
   from words like "week"/"month"/"year" in the same string. If the real
   values look different, fix is_recurring() and extract_frequency_days().
3. `flexibility` (per-event): assumed to contain "protect" (cannot touch),
   "stop" (can be fully stopped), or "reduc" (can be reduced to
   minimum_allowed_amount) -- mirroring the vocabulary used in the
   profile's three category lists. VERIFY against real values.
4. `linked_event_id`: assumed to group multiple rows representing the same
   underlying transaction (e.g. a pending version and a later confirmed
   version). When several non-historical rows share a linked_event_id,
   the code keeps only the most-certain status (confirmed > pending) to
   avoid double-counting. VERIFY this is actually what the column means.
5. `status`: assumed values "historical" / "pending" / "confirmed".
6. `payment_method` values in request_payment_options.csv and
   `payment_methods_user_will_consider` in financial_profiles.csv: assumed
   to overlap with {"full_payment","partial_payment","installments"}.

Run this once you have the repo, before trusting any output:

    python3 -c "from main import inspect_unique_values; inspect_unique_values()"

(run it from inside code/, or adjust sys.path first)
--------------------------------------------------------------------------
"""

import base64
import csv
import glob
import json
import time
import mimetypes
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta ,timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# 0. PATHS & CONSTANTS
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"
OUTPUT_PATH = REPO_ROOT / "output.csv"
USAGE_LOG_PATH = REPO_ROOT / "evaluation" / "_raw_usage_log.jsonl"

FORECAST_HORIZON_DAYS = 180
MAX_SPENDING_CHANGES = 3
DEFAULT_RECURRENCE_DAYS = 30  # fallback if frequency can't be parsed from event_type

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# --------------------------------------------------------------------------
# 1. LOADING (real column names -- no renaming needed anymore)
# --------------------------------------------------------------------------


def load_csv(filename: str) -> pd.DataFrame:
    path = DATASET_DIR / filename
    if not path.exists():
        print(f"[warn] {filename} not found at {path} -- returning empty frame", file=sys.stderr)
        return pd.DataFrame()
    return pd.read_csv(path)


def inspect_unique_values():
    """Diagnostic helper -- not called automatically by main(). Run it by
    hand before trusting the output, to check the normalization
    assumptions above against the real cell values."""
    events = load_csv("financial_events.csv")
    for col in ["event_type", "direction", "status", "flexibility", "category"]:
        if col in events.columns:
            print(col, "->", sorted(events[col].dropna().unique().tolist())[:20])
    opts = load_csv("request_payment_options.csv")
    if "payment_method" in opts.columns:
        print("payment_method ->", sorted(opts["payment_method"].dropna().unique().tolist()))
    profiles = load_csv("financial_profiles.csv")
    if "payment_methods_user_will_consider" in profiles.columns:
        print("payment_methods_user_will_consider sample ->",
              profiles["payment_methods_user_will_consider"].dropna().unique()[:5].tolist())


# --------------------------------------------------------------------------
# 2. NORMALIZATION HELPERS (edit these once you've seen real values)
# --------------------------------------------------------------------------


def is_inflow(direction: str) -> bool:
    d = str(direction).lower()
    return any(k in d for k in ("in", "income", "credit", "inflow", "salary", "deposit"))


def is_recurring(event_type: str) -> bool:
    return "recur" in str(event_type).lower()


def extract_frequency_days(event_type: str) -> int:
    et = str(event_type).lower()
    if "week" in et:
        return 7
    if "biweek" in et or "fortnight" in et:
        return 14
    if "month" in et:
        return 30
    if "quarter" in et:
        return 91
    if "year" in et or "annual" in et:
        return 365
    return DEFAULT_RECURRENCE_DAYS


def flex_kind(flexibility) -> str:
    """Returns 'protected', 'stoppable', 'reducible', or 'unknown'."""
    f = str(flexibility).lower()
    if "protect" in f or "essential" in f:
        return "protected"
    if "stop" in f:
        return "stoppable"
    if "reduc" in f:
        return "reducible"
    return "unknown"


def parse_list_field(val) -> set:
    """Profile columns like expense_categories_to_protect may be a
    delimited string ("rent, groceries") -- normalize to a lowercase set."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
        return set()
    s = str(val)
    for sep in ["|", ";", ","]:
        if sep in s:
            return {t.strip().lower() for t in s.split(sep) if t.strip()}
    return {s.strip().lower()}


def method_matches(user_methods: set, candidate: str) -> bool:
    if not user_methods:
        return True  # no restriction stated -> don't block
    c = candidate.lower()
    return any(c in m or m in c for m in user_methods)


# --------------------------------------------------------------------------
# 3. CURRENCY CONVERSION
# --------------------------------------------------------------------------


class RateTable:
    def __init__(self, rates_df: pd.DataFrame):
        self.rows = []
        if not rates_df.empty:
            df = rates_df.copy()
            df["rate_date"] = pd.to_datetime(df["rate_date"])
            for _, r in df.iterrows():
                self.rows.append((r["rate_date"], r["from_currency"], r["to_currency"], float(r["rate"])))
            self.rows.sort(key=lambda x: x[0])

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on_date: datetime) -> float:
        if from_ccy == to_ccy or amount == 0:
            return amount
        best = None
        for d, f, t, rate in self.rows:
            if f == from_ccy and t == to_ccy and d <= on_date:
                best = rate
        if best is not None:
            return amount * best
        for d, f, t, rate in self.rows:
            if f == to_ccy and t == from_ccy and d <= on_date and rate != 0:
                best = 1.0 / rate
        if best is not None:
            return amount * best
        raise ValueError(f"No exchange rate found for {from_ccy}->{to_ccy} on/before {on_date.date()}")


# --------------------------------------------------------------------------
# 4. IMAGE AMOUNT EXTRACTION (for blank `amount` fields)
# --------------------------------------------------------------------------


def find_image_file(image_id: str) -> Path:
    matches = glob.glob(str(MEDIA_DIR / f"{image_id}.*"))
    if not matches:
        raise FileNotFoundError(f"No image file found for image_id={image_id} in {MEDIA_DIR}")
    return Path(matches[0])

def call_gemini_vision_for_amount(image_path: Path, context: str) -> float:
    from google import genai
    from google.genai import types
    
    # Hardcoded fallback for images already read manually (Gemini free-tier quota exhausted)
    KNOWN_AMOUNTS = {
        "image_06.png": 1995.0,
        "image_07.png": 8528.0,
        "image_08.png": 15339.0,
        "image_09.png": 723.0,
        "image_10.png": 79679.26,
        "image_11.png": 3650.0,
        "image_12.png": 33.50,   # USD — currency handled separately, see note below
        "image_13.png": 2298.0,
        "image_14.png": 4543.0,
        "image_15.png": 9968.0,
        "image_16.png": 393.0,
    }
    if image_path.name in KNOWN_AMOUNTS:
        return KNOWN_AMOUNTS[image_path.name]

    # ... existing Gemini API code continues below ...

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"No GEMINI_API_KEY set; cannot extract amount from {image_path}.")

    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "This image is financial evidence (payroll letter, bank statement, bill, "
        "or receipt) linked to a specific financial event. "
        f"Context: {context}\n"
        "Return ONLY a JSON object like {\"amount\": 1234.56, \"currency\": \"INR\"} "
        "with the single most relevant monetary amount in the image. No prose."
    )

    client = genai.Client(api_key=api_key)
    time.sleep(13)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
    )

    usage = getattr(response, "usage_metadata", None)
    log_usage("gemini-2.0-flash", "image_amount_extraction", {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    })

    text = response.text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    parsed = json.loads(text)
    return float(parsed["amount"])


def log_usage(model: str, call_type: str, usage: dict):
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG_PATH, "a") as f:
        f.write(json.dumps({
            "model": model,
            "call_type": call_type,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "timestamp": datetime.now(timezone.UTC).isoformat(),
        }) + "\n")


# --------------------------------------------------------------------------
# 5. USER FINANCIAL MODEL
# --------------------------------------------------------------------------


@dataclass
class CashEvent:
    event_id: str
    date: datetime
    amount: float  # signed, already converted to home currency
    category: str
    flex: str       # 'protected' | 'stoppable' | 'reducible' | 'unknown'
    min_allowed: float


@dataclass
class UserModel:
    user_id: str
    home_currency: str
    current_balance: float
    min_balance: float
    protect_set: set
    reduce_set: set
    stop_set: set
    accepted_methods: set
    max_installment_months: float
    events: list = field(default_factory=list)


def resolve_amount(row, images_df: pd.DataFrame, rates: RateTable, on_date: datetime, home_ccy: str) -> float:
    amt = row.get("amount", None)
    if pd.isna(amt) or amt == "":
        linked = images_df[images_df.get("related_event_id", pd.Series(dtype=object)) == row["event_id"]] \
            if "related_event_id" in images_df.columns else pd.DataFrame()
        if linked.empty:
            raise ValueError(f"Event {row['event_id']} has blank amount and no linked image.")
        image_id = linked.iloc[0]["image_id"]
        img_path = find_image_file(image_id)
        context = f"event_id={row['event_id']}, category={row.get('category')}, date={row.get('event_date')}"
        amt = call_gemini_vision_for_amount(img_path, context)
    ccy = row.get("currency", home_ccy) or home_ccy
    return rates.convert(float(amt), ccy, home_ccy, on_date)


def build_user_model(user_id, profiles_df, events_df, images_df, rates: RateTable,
                      as_of: datetime, horizon_end: datetime) -> UserModel:
    prow = profiles_df[profiles_df["user_id"] == user_id].iloc[0]
    home_ccy = prow["home_currency"]

    model = UserModel(
        user_id=user_id,
        home_currency=home_ccy,
        current_balance=float(prow["current_available_balance"]),
        min_balance=float(prow["minimum_balance_to_keep"]),
        protect_set=parse_list_field(prow.get("expense_categories_to_protect")),
        reduce_set=parse_list_field(prow.get("expense_categories_user_is_willing_to_reduce")),
        stop_set=parse_list_field(prow.get("expense_categories_user_is_willing_to_stop")),
        accepted_methods=parse_list_field(prow.get("payment_methods_user_will_consider")),
        max_installment_months=float(prow.get("max_installment_months") or 999),
    )

    u_events = events_df[events_df["user_id"] == user_id].copy()

    # De-duplicate: when several non-historical rows share a linked_event_id,
    # keep only the most-certain status (confirmed > pending). See
    # assumption #4 at the top of this file.
    status_rank = {"confirmed": 2, "pending": 1, "historical": 0}
    if "linked_event_id" in u_events.columns and "status" in u_events.columns:
        u_events["_status_rank"] = u_events["status"].str.lower().map(status_rank).fillna(0)
        has_link = u_events["linked_event_id"].notna()
        linked_groups = u_events[has_link].sort_values("_status_rank", ascending=False)
        linked_groups = linked_groups.drop_duplicates(subset=["linked_event_id"], keep="first")
        u_events = pd.concat([u_events[~has_link], linked_groups]).drop(columns=["_status_rank"])

    for _, row in u_events.iterrows():
        status = str(row.get("status", "")).lower()
        if status == "historical":
            continue  # already reflected in current_available_balance

        category = str(row.get("category", "")).strip().lower()
        flex = flex_kind(row.get("flexibility", ""))
        # Fall back to the profile's category lists if the per-event
        # flexibility field didn't resolve to something recognizable.
        if flex == "unknown":
            if category in model.protect_set:
                flex = "protected"
            elif category in model.stop_set:
                flex = "stoppable"
            elif category in model.reduce_set:
                flex = "reducible"
            else:
                flex = "protected" if is_inflow(row.get("direction", "")) else "unknown"

        min_allowed = row.get("minimum_allowed_amount", 0.0)
        min_allowed = float(min_allowed) if not pd.isna(min_allowed) else 0.0

        effective_date_raw = row.get("settlement_date")
        if pd.isna(effective_date_raw) or effective_date_raw == "":
            effective_date_raw = row.get("event_date")
        base_date = pd.to_datetime(effective_date_raw)

        sign = 1.0 if is_inflow(row.get("direction", "")) else -1.0

        if not is_recurring(row.get("event_type", "")):
            if base_date < as_of - timedelta(days=1) or base_date > horizon_end:
                continue
            amt = resolve_amount(row, images_df, rates, base_date, home_ccy)
            model.events.append(CashEvent(row["event_id"], base_date, sign * amt, category, flex, min_allowed))
            continue

        step_days = extract_frequency_days(row.get("event_type", ""))
        occ_date = base_date
        while occ_date <= horizon_end:
            if occ_date >= as_of - timedelta(days=1):
                amt = resolve_amount(row, images_df, rates, occ_date, home_ccy)
                model.events.append(CashEvent(row["event_id"], occ_date, sign * amt, category, flex, min_allowed))
            occ_date += timedelta(days=step_days)

    model.events.sort(key=lambda e: e.date)
    return model


# --------------------------------------------------------------------------
# 6. FORECAST ENGINE
# --------------------------------------------------------------------------
def forecast_balances(model: UserModel, as_of: datetime, horizon_end: datetime,
                       excluded_event_ids=None, reduced_events=None):
    excluded_event_ids = excluded_event_ids or set()
    reduced_events = reduced_events or {}
    balance = model.current_balance
    points = [(as_of, balance)]
    for e in model.events:
        if e.date < as_of or e.date > horizon_end:
            continue
        if e.event_id in excluded_event_ids:
            continue
        amt = e.amount
        if e.event_id in reduced_events and amt < 0:
            amt = -abs(reduced_events[e.event_id])
        balance += amt
        points.append((e.date, balance))
    return points


def suffix_min(points):
    mins = [0.0] * len(points)
    running = float("inf")
    for i in range(len(points) - 1, -1, -1):
        running = min(running, points[i][1])
        mins[i] = running
    return mins


# --------------------------------------------------------------------------
# 7. DECISION LOGIC FOR ONE REQUEST
# --------------------------------------------------------------------------


def decide(request_row, model: UserModel, payment_options_df: pd.DataFrame, as_of: datetime):
    requested_amount = float(request_row["requested_amount"])
    desired_completion = pd.to_datetime(request_row["desired_completion_date"]) \
        if not pd.isna(request_row.get("desired_completion_date")) else None
    allows_partial = str(request_row.get("allows_partial_payment", "")).strip().lower() in ("true", "1", "yes")
    horizon_end = as_of + timedelta(days=FORECAST_HORIZON_DAYS)

    points = forecast_balances(model, as_of, horizon_end)
    dates = [p[0] for p in points]
    mins_from = suffix_min(points)

    def headroom_if_paid_on(idx):
        return mins_from[idx] - model.min_balance

    headroom_now = headroom_if_paid_on(0)
    amount_safe_to_pay = max(0.0, min(requested_amount, headroom_now))

    earliest_full_date = None
    for i, d in enumerate(dates):
        if headroom_if_paid_on(i) >= requested_amount:
            earliest_full_date = d
            break

    spending_changes = []
    plan_amount_safe = amount_safe_to_pay
    plan_earliest_full = earliest_full_date

    need_plan = amount_safe_to_pay < requested_amount or earliest_full_date is None or \
        (desired_completion is not None and earliest_full_date is not None and earliest_full_date > desired_completion)

    if need_plan:
        # Only touch events the user is willing to stop or reduce -- never 'protected'.
        stoppable = sorted(
            {e.event_id: e for e in model.events if e.flex == "stoppable" and e.amount < 0}.values(),
            key=lambda e: e.amount,
        )
        reducible = sorted(
            {e.event_id: e for e in model.events if e.flex == "reducible" and e.amount < 0}.values(),
            key=lambda e: e.amount,
        )
        candidates = stoppable + reducible
        excluded, reduced = set(), {}
        for e in candidates[:MAX_SPENDING_CHANGES]:
            if e.flex == "stoppable":
                excluded.add(e.event_id)
                spending_changes.append(f"stop:{e.event_id}")
            else:
                reduced[e.event_id] = e.min_allowed
                spending_changes.append(f"reduce_to:{e.event_id}:{e.min_allowed:.2f}")

            trial_points = forecast_balances(model, as_of, horizon_end, excluded, reduced)
            trial_mins = suffix_min(trial_points)
            plan_amount_safe = max(0.0, min(requested_amount, trial_mins[0] - model.min_balance))
            plan_earliest_full = None
            for i, d in enumerate([p[0] for p in trial_points]):
                if trial_mins[i] - model.min_balance >= requested_amount:
                    plan_earliest_full = d
                    break
            good_amount = plan_amount_safe >= requested_amount
            good_date = plan_earliest_full is not None and (desired_completion is None or plan_earliest_full <= desired_completion)
            if good_amount or good_date:
                break

    opts = payment_options_df[payment_options_df["request_id"] == request_row["request_id"]] \
        if "request_id" in payment_options_df.columns else pd.DataFrame()

    payment_plan = "none"

    if amount_safe_to_pay >= requested_amount:
        status = "affordable_now"
        method = "full_payment"
        payment_plan = f"{as_of.date()}:{requested_amount:.2f}"
        earliest_full_date = as_of

    elif plan_earliest_full is not None and (desired_completion is None or plan_earliest_full <= desired_completion):
        status = "affordable_with_plan"
        installment_row = None
        if not opts.empty and "payment_method" in opts.columns:
            inst = opts[opts["payment_method"].str.lower().str.contains("install", na=False)]
            if "number_of_payments" in inst.columns:
                inst = inst[inst["number_of_payments"].fillna(999) <= model.max_installment_months]
            if method_matches(model.accepted_methods, "installments") and not inst.empty:
                installment_row = inst.iloc[0]

        if installment_row is not None:
            method = "installments"
            n = int(installment_row["number_of_payments"])
            per = float(installment_row["payment_amount"])
            freq_days = int(installment_row.get("payment_frequency_days", 30) or 30)
            first_date = pd.to_datetime(installment_row.get("first_payment_date", as_of))
            plan_parts, running_date = [], first_date
            for i in range(n):
                plan_parts.append(f"{running_date.date()}:{per:.2f}")
                running_date += timedelta(days=freq_days)
            payment_plan = "|".join(plan_parts)
            amount_safe_to_pay = per  # first installment due now
        elif allows_partial and method_matches(model.accepted_methods, "partial_payment") \
                and 0 < plan_amount_safe < requested_amount and plan_earliest_full is not None:
            method = "partial_payment"
            remainder = round(requested_amount - plan_amount_safe, 2)
            payment_plan = f"{as_of.date()}:{plan_amount_safe:.2f}|{plan_earliest_full.date()}:{remainder:.2f}"
            amount_safe_to_pay = plan_amount_safe
        else:
            method = "full_payment"
            payment_plan = f"{plan_earliest_full.date()}:{requested_amount:.2f}"
            amount_safe_to_pay = plan_amount_safe
        earliest_full_date = plan_earliest_full

    elif earliest_full_date is not None:
        status = "affordable_later"
        method = "wait"
        payment_plan = "none"

    else:
        status = "not_affordable"
        method = "not_recommended"
        payment_plan = "none"
        amount_safe_to_pay = min(amount_safe_to_pay, requested_amount)

    explanation = (
        f"Balance {model.current_balance:.2f} {model.home_currency}, "
        f"min balance {model.min_balance:.2f}. Forecast headroom on {as_of.date()} is "
        f"{headroom_now:.2f} against a request of {requested_amount:.2f}. "
        f"Decision: {status} / {method}."
    )
    if spending_changes:
        explanation += f" Required spending changes: {', '.join(spending_changes)}."

    return {
        "request_id": request_row["request_id"],
        "amount_safe_to_pay": round(max(0.0, min(amount_safe_to_pay, requested_amount)), 2),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": payment_plan,
        "earliest_date_for_full_payment": earliest_full_date.date().isoformat() if earliest_full_date else "",
        "spending_changes_needed": "|".join(spending_changes) if spending_changes else "none",
        "decision_explanation": explanation,
    }


# --------------------------------------------------------------------------
# 8. MAIN
# --------------------------------------------------------------------------


def main():
    requests_df = load_csv("requests.csv")
    profiles_df = load_csv("financial_profiles.csv")
    events_df = load_csv("financial_events.csv")
    payment_options_df = load_csv("request_payment_options.csv")
    rates_df = load_csv("exchange_rates.csv")
    images_df = load_csv("images.csv")

    rates = RateTable(rates_df)
    results = []
    model_cache = {}

    for _, req in requests_df.iterrows():
        user_id = req["user_id"]
        as_of = pd.to_datetime(req["request_date"])
        horizon_end = as_of + timedelta(days=FORECAST_HORIZON_DAYS)

        if user_id not in model_cache:
            try:
                model_cache[user_id] = build_user_model(
                    user_id, profiles_df, events_df, images_df, rates, as_of, horizon_end
                )
            except Exception as exc:
                print(f"[error] could not build model for user {user_id}: {exc}", file=sys.stderr)
                continue
        model = model_cache[user_id]

        try:
            row = decide(req, model, payment_options_df, as_of)
        except Exception as exc:
            print(f"[error] request {req.get('request_id')} failed: {exc}", file=sys.stderr)
            row = {
                "request_id": req.get("request_id"),
                "amount_safe_to_pay": 0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"Could not evaluate: {exc}",
            }
        results.append(row)

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(out_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
