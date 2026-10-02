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
    picks = []
    for bucket in BUCKETS:
        b = df[df["bucket"] == bucket].sort_values(rank_col).head(args.per_bucket).copy()
        picks.append(b)
    out = pd.concat(picks, ignore_index=True)
    if out.empty:
        raise RuntimeError("No portfolio candidates generated")

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
        "tickers": out["tic"].tolist(),
        "buckets": out.groupby("bucket")["tic"].apply(list).to_dict(),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
