#!/usr/bin/env python3
"""Extend FinRL-X S&P 500 point-in-time constituent snapshots through the requested as-of date.

The official dataset currently ends at 2026-04-17.  This script applies
S&P Dow Jones Indices announced changes and emits NYSE-session snapshots.
It is deterministic and safe to re-run.

Source announcements:
- 2026-04-30 VEEV replaces CTRA effective 2026-05-07
- 2026-05-27 FDXF added 2026-06-01; EPAM removed 2026-06-02
- 2026-06-05 MRVL/FLEX replace POOL/CPB effective 2026-06-22
- 2026-06-23 HONA added 2026-06-29; CAG removed 2026-06-30
- 2026-07-31 FERG replaces EA effective 2026-08-05
- 2026-08-13 RDDT replaces AVB effective 2026-08-18; EQR renamed VMRK
- 2026-09-04 BE/P/ILMN replace TAP/TTD/BLDR effective 2026-09-21
- 2026-10-01 VYLR added after the Corteva spin-off; CTVA remains until 2026-10-06
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

TARGET_DATE = pd.Timestamp("2026-10-02")

# (effective date, additions, deletions, renames)
CHANGES = [
    ("2026-05-07", ["VEEV"], ["CTRA"], {}),
    # Spin-off timing intentionally preserves the one-session transitional membership.
    ("2026-06-01", ["FDXF"], [], {}),
    ("2026-06-02", [], ["EPAM"], {}),
    ("2026-06-22", ["MRVL", "FLEX"], ["POOL", "CPB"], {}),
    ("2026-06-29", ["HONA"], [], {}),
    ("2026-06-30", [], ["CAG"], {}),
    ("2026-08-05", ["FERG"], ["EA"], {}),
    ("2026-08-18", ["RDDT"], ["AVB"], {"EQR": "VMRK"}),
    ("2026-09-21", ["BE", "P", "ILMN"], ["TAP", "TTD", "BLDR"], {}),
    # S&P Global's Oct. 1 table lists VYLR as an addition on Oct. 1 and
    # CTVA's deletion on Oct. 6, so Oct. 1-5 legitimately has one extra constituent.
    ("2026-10-01", ["VYLR"], [], {}),
    ("2026-10-06", ["TWLO"], ["CTVA", "WBD"], {}),
]


def parse_members(value: str) -> set[str]:
    return {x.strip() for x in str(value).split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sp500_historical_constituents.csv")
    ap.add_argument("--target-date", default=TARGET_DATE.strftime("%Y-%m-%d"))
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    if "date" not in df.columns or "tickers" not in df.columns:
        raise ValueError("Expected columns: date,tickers")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    last_date = df["date"].iloc[-1]
    target = pd.Timestamp(args.target_date)

    if last_date >= target:
        print(f"Already covered through {last_date.date()}; target={target.date()}")
        return

    members = parse_members(df.iloc[-1]["tickers"])
    changes = {
        pd.Timestamp(d): (set(adds), set(dels), renames)
        for d, adds, dels, renames in CHANGES
    }

    nyse = mcal.get_calendar("NYSE")
    sessions = nyse.schedule(
        start_date=(last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        end_date=target.strftime("%Y-%m-%d"),
    ).index.tz_localize(None)

    rows = []
    for session in sessions:
        if session in changes:
            adds, dels, renames = changes[session]
            for old, new in renames.items():
                if old in members:
                    members.remove(old)
                    members.add(new)
            members.difference_update(dels)
            members.update(adds)
        rows.append({"date": session, "tickers": ",".join(sorted(members))})

    if rows:
        out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        print(
            f"Extended {path} from {last_date.date()} through "
            f"{pd.to_datetime(out['date']).max().date()} ({len(rows)} new sessions)"
        )
        print(f"Final membership count: {len(members)}")


if __name__ == "__main__":
    main()
