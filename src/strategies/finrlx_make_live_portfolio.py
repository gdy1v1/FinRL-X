#!/usr/bin/env python3
"""Create an auditable diversified candidate portfolio from today's FinRL-X ranking."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

BUCKETS = ["growth_tech", "cyclical", "real_assets", "defensive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--per-bucket", type=int, default=3)
    ap.add_argument("--output", default="data/live_portfolio_today.csv")
    ap.add_argument("--summary", default="data/live_portfolio_today.json")
    ap.add_argument("--as-of", default="2026-10-02")
    args = ap.parse_args()

    pred_path = args.predictions
    if not pred_path:
        files = sorted(glob.glob("data/sp500_ml_mixed_alpha_*.csv"))
        if not files:
            raise FileNotFoundError("No mixed-alpha prediction CSV found")
        pred_path = files[-1]

    df = pd.read_csv(pred_path)
    if "datadate" in df.columns and (df["datadate"] == "mixed").any():
        df = df[df["datadate"] == "mixed"].copy()

    rank_col = "rank_mixed" if "rank_mixed" in df.columns else "rank_best"

    # Execution gate: a stock cannot enter a "buy today" portfolio without a
    # verified positive reference price from the latest completed session.
    df["trade_price"] = pd.to_numeric(df.get("trade_price"), errors="coerce")
    eligible = df[df["trade_price"].notna() & (df["trade_price"] > 0)].copy()

    picks = []
    short_buckets = {}
    for bucket in BUCKETS:
        b = eligible[eligible["bucket"] == bucket].sort_values(rank_col).head(args.per_bucket).copy()
        if len(b) < args.per_bucket:
            short_buckets[bucket] = len(b)
        picks.append(b)
    out = pd.concat(picks, ignore_index=True)
    if out.empty:
        raise RuntimeError("No executable portfolio candidates generated")
    if short_buckets:
        raise RuntimeError(f"Insufficient priced candidates by bucket: {short_buckets}")

    out["weight"] = 1.0 / len(out)
    out["weight_pct"] = out["weight"] * 100
    out["as_of"] = args.as_of
    out["selection_method"] = f"top_{args.per_bucket}_per_bucket_mixed_alpha"

    keep = [
        "tic", "bucket", "weight", "weight_pct", "as_of", "trade_price",
        "original_datadate", "data_vintage", "filing_date", "accepted_date",
        "bucket_pred", "unified_pred", "mixed_score", rank_col, "best_model",
        "selection_method",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].sort_values(["bucket", rank_col] if rank_col in keep else ["bucket"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    summary = {
        "as_of": args.as_of,
        "prediction_file": pred_path,
        "portfolio_file": args.output,
        "holdings": len(out),
        "per_bucket": args.per_bucket,
        "weighting": "equal",
        "eligibility_rule": "trade_price must be non-null and > 0",
        "eligible_ranked_stocks": int(len(eligible)),
        "tickers": out["tic"].tolist(),
        "buckets": out.groupby("bucket")["tic"].apply(list).to_dict(),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
