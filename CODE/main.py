
#!/usr/bin/env python3
"""
HackerRank Orchestrate (September 2026) — Buy or Wait?

Reads dataset/requests.csv and every supporting file, reconstructs each
user's financial state, runs a 90-day safety-checked forecast, and writes
output.csv (in the repository root) with one prediction per request.

Usage:
    python3 code/main.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.data import Dataset, parse_date
from lib.decision import decide

REQUIRED_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    dataset_dir = os.path.join(repo_root, "dataset")

    print(f"Loading dataset from {dataset_dir} ...")
    ds = Dataset(dataset_dir)
    fx = ds._fx

    # Optional LLM-based message/image enrichment. Only runs if a Gemini
    # API key is configured; otherwise the deterministic pipeline below
    # (financial_profiles.csv + financial_events.csv + payment options,
    # with manually-verified image amounts already baked into
    # lib/data.py::IMAGE_DERIVED_AMOUNTS) is used as-is.
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "model": None, "provider": None}
    try:
        from lib.enrich import maybe_enrich_with_gemini
        maybe_enrich_with_gemini(ds, usage)
    except Exception as exc:  # pragma: no cover - enrichment is best-effort
        print(f"(message/image enrichment skipped: {exc})")

    rows = []
    t0 = time.time()
    for request in ds.requests:
        rows.append(decide(ds, fx, request))
    elapsed = time.time() - t0
    print(f"Generated {len(rows)} predictions in {elapsed:.2f}s")

    out_path = os.path.join(repo_root, "output.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {out_path}")

    _write_usage_report(repo_root, usage, len(rows))


def _write_usage_report(repo_root, usage, n_requests):
    eval_dir = os.path.join(repo_root, "code", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, "usage_report.md")

    if usage["calls"] == 0:
        content = f"""# Token Usage & Cost Report

This run of `code/main.py` used **no LLM API calls**. The full pipeline is
deterministic: it reconstructs each user's financial state from
`financial_profiles.csv` and `financial_events.csv`, resolves the 16
blank-amount events via manually-verified values read directly from their
linked receipt/invoice images (see `lib/data.py::IMAGE_DERIVED_AMOUNTS`),
and runs a rule-based 90-day cash-flow forecast to produce every field in
`output.csv`.

- Requests scored: {n_requests}
- Model calls: 0
- Input tokens: 0
- Output tokens: 0
- Estimated cost: $0.00

To enable optional LLM-based enrichment of `messages.csv` (salary changes,
cancellations, delays, confirmations expressed in free text), set
`GEMINI_API_KEY` in `.env` before running `python3 code/main.py`. When a key
is present, `lib/enrich.py` calls the Gemini API and this file is
regenerated with the real per-run token counts and cost estimate for that
run.
"""
    else:
        model = usage["model"]
        provider = usage["provider"]
        calls = usage["calls"]
        in_tok = usage["input_tokens"]
        out_tok = usage["output_tokens"]
        total_tok = in_tok + out_tok
        # Gemini 2.0 Flash public pricing at time of writing: $0.10 / 1M
        # input tokens, $0.40 / 1M output tokens. Update if pricing changes.
        price_in_per_m = 0.10
        price_out_per_m = 0.40
        cost = (in_tok / 1_000_000) * price_in_per_m + (out_tok / 1_000_000) * price_out_per_m
        content = f"""# Token Usage & Cost Report

Summary of the final full-dataset run that produced `output.csv`.

| Metric | Value |
|---|---|
| Provider | {provider} |
| Model | {model} |
| Requests scored | {n_requests} |
| Model calls | {calls} |
| Total input tokens | {in_tok} |
| Total output tokens | {out_tok} |
| Total tokens | {total_tok} |
| Avg tokens / request | {total_tok / max(1, n_requests):.1f} |
| Avg tokens / call | {total_tok / max(1, calls):.1f} |
| Estimated total cost (USD) | ${cost:.4f} |
| Estimated cost / request (USD) | ${cost / max(1, n_requests):.6f} |

Pricing assumed: ${price_in_per_m:.2f} / 1M input tokens, ${price_out_per_m:.2f} / 1M
output tokens (Gemini Flash public pricing at time of writing — update if
your account's pricing differs).

Gemini calls were used only to enrich signal extraction from
`messages.csv` (confirm / amend / cancel / delay salary and expense
facts). The core forecast, currency conversion, payment-option matching,
and 90-day safety check remain fully deterministic and rule-based.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
  
