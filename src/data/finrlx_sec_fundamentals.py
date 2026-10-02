#!/usr/bin/env python3
"""Fetch recent SEC Company Facts for a reproducible live FinRL-X snapshot.

Only facts filed on or before --as-of are eligible.  The output is raw,
traceable source data; downstream code derives only ratios that can be
computed defensibly and carries forward other trained features.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

CONCEPTS = {
    "Revenues": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"],
    "NetIncomeLoss": ["NetIncomeLoss"],
    "Assets": ["Assets"],
    "CurrentAssets": ["AssetsCurrent"],
    "Liabilities": ["Liabilities"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "StockholdersEquity": ["StockholdersEquity"],
    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    ],
    "AccountsReceivable": ["AccountsReceivableNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent"],
    "AccountsPayable": ["AccountsPayableCurrent"],
    "LongTermDebtCurrent": ["LongTermDebtCurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent"],
    "LongTermDebtNoncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"],
    "InterestExpense": ["InterestExpenseNonOperating", "InterestExpense"],
    "DepreciationAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"
    ],
    "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
    "SharesOutstanding": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
    "WeightedAverageSharesDiluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "DividendsCashPaid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
        "PaymentsOfOrdinaryDividends"
    ],
}


def latest_fact(companyfacts: dict, candidates: list[str], as_of: pd.Timestamp):
    """Return the most recently filed usable fact, constrained by filing date."""
    usgaap = companyfacts.get("facts", {}).get("us-gaap", {})
    best = None
    for concept in candidates:
        item = usgaap.get(concept)
        if not item:
            continue
        for unit, unit_rows in item.get("units", {}).items():
            for r in unit_rows:
                if r.get("form") not in {"10-Q", "10-K"}:
                    continue
                if r.get("val") is None or not r.get("end") or not r.get("filed"):
                    continue
                filed = pd.to_datetime(r.get("filed"), errors="coerce")
                end = pd.to_datetime(r.get("end"), errors="coerce")
                if pd.isna(filed) or pd.isna(end) or filed > as_of:
                    continue
                key = (filed, end)
                if best is None or key > best[0]:
                    best = (key, r.get("val"), r.get("end"), r.get("filed"), concept, unit)
    if best is None:
        return None, None, None, None, None
    _, val, end, filed, concept, unit = best
    return val, end, filed, concept, unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constituents", default="data/sp500_historical_constituents.csv")
    ap.add_argument("--output", default="data/sec_companyfacts_latest.csv")
    ap.add_argument("--as-of", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--sleep", type=float, default=0.12)
    args = ap.parse_args()
    as_of = pd.Timestamp(args.as_of)

    hist = pd.read_csv(args.constituents)
    hist["date"] = pd.to_datetime(hist["date"])
    valid = hist[hist["date"] <= as_of]
    if valid.empty:
        raise ValueError(f"No constituent snapshot on/before {as_of.date()}")
    tickers = sorted({x.strip() for x in str(valid.iloc[-1]["tickers"]).split(",") if x.strip()})

    s = requests.Session()
    s.headers.update({
        "User-Agent": os.getenv(
            "SEC_USER_AGENT",
            "FinRL-X research https://github.com/gdy1v1/FinRL-X"
        ),
        "Accept-Encoding": "gzip, deflate",
    })

    # Prefer the persisted ticker→CIK mapping from the previous successful
    # snapshot. GitHub-hosted runners are sometimes blocked by www.sec.gov/files
    # even when data.sec.gov APIs remain available.
    ticker_map = {}
    out_path = Path(args.output)
    if out_path.exists():
        try:
            prev = pd.read_csv(out_path, usecols=lambda x: x in {"ticker", "cik"})
            for r in prev.dropna(subset=["ticker", "cik"]).itertuples(index=False):
                cik = str(r.cik).split(".")[0].zfill(10)
                ticker_map[str(r.ticker).upper()] = cik
            print(f"Loaded {len(ticker_map)} cached ticker→CIK mappings from {out_path}")
        except Exception as e:
            print(f"WARNING: could not read cached CIK mapping: {e}")

    try:
        mapping = s.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
        mapping.raise_for_status()
        fresh_map = {
            str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
            for v in mapping.json().values()
        }
        ticker_map.update(fresh_map)
        print(f"Refreshed ticker→CIK mapping from SEC ({len(fresh_map)} entries)")
    except Exception as e:
        print(f"WARNING: SEC ticker mapping unavailable; using cache only: {e}")

    rows = []
    for i, ticker in enumerate(tickers, 1):
        cik = ticker_map.get(ticker.replace(".", "-").upper()) or ticker_map.get(ticker.upper())
        if not cik:
            rows.append({"ticker": ticker, "error": "CIK not found"})
            continue
        try:
            r = s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", timeout=30)
            r.raise_for_status()
            facts = r.json()
            row = {"ticker": ticker, "cik": cik, "entity_name": facts.get("entityName")}
            ends, filed_dates = [], []
            for out_name, concepts in CONCEPTS.items():
                val, end, filed, source_concept, unit = latest_fact(facts, concepts, as_of)
                row[out_name] = val
                row[f"{out_name}_end"] = end
                row[f"{out_name}_filed"] = filed
                row[f"{out_name}_concept"] = source_concept
                row[f"{out_name}_unit"] = unit
                if end:
                    ends.append(end)
                if filed:
                    filed_dates.append(filed)
            row["latest_fact_end"] = max(ends) if ends else None
            row["latest_filed"] = max(filed_dates) if filed_dates else None
            row["as_of"] = args.as_of
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": ticker, "cik": cik, "error": str(e), "as_of": args.as_of})
        time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"SEC Company Facts: {i}/{len(tickers)}")

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Never destroy a previously valid snapshot just because SEC blocks a CI
    # runner.  Require broad successful coverage before replacing the cache.
    if "error" in out.columns:
        success_count = int(out["error"].isna().sum())
    else:
        success_count = len(out)
    min_success = min(400, max(1, int(len(tickers) * 0.75)))

    if success_count < min_success and out_path.exists():
        print(
            f"WARNING: only {success_count}/{len(tickers)} SEC rows succeeded; "
            f"preserving existing valid snapshot at {out_path}"
        )
        return

    out.to_csv(out_path, index=False)
    print(
        f"Wrote {len(out)} rows to {out_path} (as-of {args.as_of}); "
        f"successful rows={success_count}"
    )


if __name__ == "__main__":
    main()
