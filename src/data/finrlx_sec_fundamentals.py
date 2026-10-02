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
    mapping = s.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
    mapping.raise_for_status()
    ticker_map = {
        str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
        for v in mapping.json().values()
    }

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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output} (as-of {args.as_of})")


if __name__ == "__main__":
    main()
