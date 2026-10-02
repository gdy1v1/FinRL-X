#!/usr/bin/env python3
"""Fetch recent SEC Company Facts as a no-key fundamental-data supplement.

This does not invent FinRL factors that SEC does not publish.  It writes a
traceable raw/normalized supplement that can be used to fill or validate
recent fundamentals when FMP is unavailable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

SEC_HEADERS = {
    "User-Agent": "FinRL-X data refresh research-contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

CONCEPTS = {
    "Revenues": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "NetIncomeLoss": ["NetIncomeLoss"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "StockholdersEquity": ["StockholdersEquity"],
    "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def latest_quarter_fact(companyfacts: dict, candidates: list[str]):
    usgaap = companyfacts.get("facts", {}).get("us-gaap", {})
    for concept in candidates:
        item = usgaap.get(concept)
        if not item:
            continue
        units = item.get("units", {})
        for unit_rows in units.values():
            rows = [
                r for r in unit_rows
                if r.get("form") in {"10-Q", "10-K"} and r.get("end") and r.get("val") is not None
            ]
            if rows:
                rows.sort(key=lambda r: (r.get("end", ""), r.get("filed", "")))
                r = rows[-1]
                return r.get("val"), r.get("end"), r.get("filed"), concept
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constituents", default="data/sp500_historical_constituents.csv")
    ap.add_argument("--output", default="data/sec_companyfacts_latest.csv")
    ap.add_argument("--sleep", type=float, default=0.12)
    args = ap.parse_args()

    hist = pd.read_csv(args.constituents)
    tickers = sorted({x.strip() for x in str(hist.iloc[-1]["tickers"]).split(",") if x.strip()})

    s = requests.Session()
    s.headers.update(SEC_HEADERS)
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
            continue
        try:
            r = s.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", timeout=30)
            r.raise_for_status()
            facts = r.json()
            row = {"ticker": ticker, "cik": cik, "entity_name": facts.get("entityName")}
            ends = []
            filed = []
            for out_name, concepts in CONCEPTS.items():
                val, end, fdate, source_concept = latest_quarter_fact(facts, concepts)
                row[out_name] = val
                row[f"{out_name}_concept"] = source_concept
                if end:
                    ends.append(end)
                if fdate:
                    filed.append(fdate)
            row["latest_fact_end"] = max(ends) if ends else None
            row["latest_filed"] = max(filed) if filed else None
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": ticker, "cik": cik, "error": str(e)})
        time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"SEC Company Facts: {i}/{len(tickers)}")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
