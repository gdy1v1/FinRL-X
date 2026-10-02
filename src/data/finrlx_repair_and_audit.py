#!/usr/bin/env python3
"""Repair recoverable FinRL-X trading labels and generate integrity audits.

Rules:
- Never fabricate prices.
- Never replace unavailable/delisted prices with 0.
- Future/unrealized y_return remains NULL.
- y_return is always log(next_trade_price / trade_price).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import py7zr
import yfinance as yf

TRADEDATE_MAP = {
    "03-31": ("06-01", 0),
    "06-30": ("09-01", 0),
    "09-30": ("12-01", 0),
    "12-31": ("03-01", 1),
}


def compute_tradedate(datadate: str) -> str | None:
    s = str(datadate)[:10]
    if len(s) != 10:
        return None
    spec = TRADEDATE_MAP.get(s[5:])
    if not spec:
        return None
    mmdd, add_year = spec
    return f"{int(s[:4]) + add_year}-{mmdd}"


def extract_archive(archive: Path, workdir: Path) -> Path:
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(workdir)
    dbs = list(workdir.rglob("*.db"))
    if not dbs:
        raise FileNotFoundError("No SQLite .db found inside archive")
    if len(dbs) > 1:
        dbs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return dbs[0]


def repack_archive(workdir: Path, archive: Path) -> None:
    tmp = archive.with_suffix(".tmp.7z")
    if tmp.exists():
        tmp.unlink()
    with py7zr.SevenZipFile(tmp, "w") as z:
        for p in sorted(workdir.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(workdir)))
    tmp.replace(archive)


def ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fundamental_data)")}
    for name, typ in [
        ("tradedate", "TEXT"),
        ("actual_tradedate", "TEXT"),
        ("trade_price", "REAL"),
        ("y_return", "REAL"),
    ]:
        if name not in cols:
            conn.execute(f"ALTER TABLE fundamental_data ADD COLUMN {name} {typ}")
    conn.commit()


def fill_dates(conn: sqlite3.Connection) -> int:
    df = pd.read_sql(
        "SELECT ticker, datadate, tradedate, actual_tradedate FROM fundamental_data",
        conn,
    )
    nyse = mcal.get_calendar("NYSE")
    updates = []
    for r in df.itertuples(index=False):
        td = r.tradedate or compute_tradedate(r.datadate)
        if not td:
            continue
        actual = r.actual_tradedate
        if not actual:
            ts = pd.Timestamp(td)
            sched = nyse.schedule(
                start_date=ts.strftime("%Y-%m-%d"),
                end_date=(ts + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            )
            if not sched.empty:
                actual = sched.index[0].strftime("%Y-%m-%d")
        if td != r.tradedate or actual != r.actual_tradedate:
            updates.append((td, actual, r.ticker, str(r.datadate)[:10]))
    conn.executemany(
        """UPDATE fundamental_data SET tradedate=?, actual_tradedate=?
           WHERE ticker=? AND datadate=?""",
        updates,
    )
    conn.commit()
    return len(updates)


def download_missing_prices(conn: sqlite3.Connection, as_of: pd.Timestamp) -> tuple[int, list[str]]:
    df = pd.read_sql(
        """SELECT ticker, datadate, actual_tradedate
           FROM fundamental_data
           WHERE (trade_price IS NULL OR trade_price <= 0)
             AND actual_tradedate IS NOT NULL""",
        conn,
    )
    df["actual_tradedate"] = pd.to_datetime(df["actual_tradedate"], errors="coerce")
    df = df[df["actual_tradedate"].notna() & (df["actual_tradedate"] <= as_of)].copy()
    if df.empty:
        return 0, []

    updated = 0
    unresolved: set[str] = set()
    cursor = conn.cursor()

    for ticker, g in df.groupby("ticker"):
        yft = str(ticker).replace(".", "-")
        start = (g["actual_tradedate"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        end = (g["actual_tradedate"].max() + pd.Timedelta(days=8)).strftime("%Y-%m-%d")
        try:
            px = yf.download(yft, start=start, end=end, auto_adjust=True, progress=False)
            if px.empty:
                unresolved.add(str(ticker))
                continue
            close = px["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.dropna()

            for r in g.itertuples(index=False):
                dt = pd.Timestamp(r.actual_tradedate)
                candidates = close[close.index >= dt]
                candidates = candidates[candidates.index <= dt + pd.Timedelta(days=5)]
                if candidates.empty:
                    unresolved.add(str(ticker))
                    continue
                price = float(candidates.iloc[0])
                if not math.isfinite(price) or price <= 0:
                    unresolved.add(str(ticker))
                    continue
                cursor.execute(
                    """UPDATE fundamental_data SET trade_price=?
                       WHERE ticker=? AND datadate=?""",
                    (round(price, 6), r.ticker, str(r.datadate)[:10]),
                )
                updated += 1
        except Exception:
            unresolved.add(str(ticker))

    conn.commit()
    return updated, sorted(unresolved)


def recompute_returns(conn: sqlite3.Connection, as_of: pd.Timestamp) -> int:
    df = pd.read_sql(
        """SELECT ticker, datadate, actual_tradedate, trade_price
           FROM fundamental_data ORDER BY ticker, datadate""",
        conn,
    )
    df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")
    df["actual_tradedate"] = pd.to_datetime(df["actual_tradedate"], errors="coerce")
    df["next_trade_price"] = df.groupby("ticker")["trade_price"].shift(-1)
    df["next_actual_tradedate"] = df.groupby("ticker")["actual_tradedate"].shift(-1)

    realized = (
        (df["trade_price"] > 0)
        & (df["next_trade_price"] > 0)
        & df["next_actual_tradedate"].notna()
        & (df["next_actual_tradedate"] <= as_of)
    )
    df["new_y"] = np.nan
    df.loc[realized, "new_y"] = np.log(
        df.loc[realized, "next_trade_price"] / df.loc[realized, "trade_price"]
    )

    vals = []
    for r in df.itertuples(index=False):
        val = None if pd.isna(r.new_y) else round(float(r.new_y), 10)
        vals.append((val, r.ticker, str(r.datadate)[:10]))
    conn.executemany(
        "UPDATE fundamental_data SET y_return=? WHERE ticker=? AND datadate=?",
        vals,
    )
    conn.commit()
    return len(vals)


def audit(conn: sqlite3.Connection, as_of: pd.Timestamp, unresolved_price_tickers: list[str]) -> dict:
    df = pd.read_sql("SELECT * FROM fundamental_data", conn)
    required = [c for c in ["ticker", "datadate", "tradedate", "actual_tradedate", "trade_price", "y_return"] if c in df.columns]
    counts = {c: int(df[c].isna().sum()) for c in required}

    dupes = int(df.duplicated(["ticker", "datadate"]).sum()) if {"ticker", "datadate"} <= set(df.columns) else None
    zero_y = int((pd.to_numeric(df.get("y_return"), errors="coerce") == 0).sum()) if "y_return" in df else None

    mismatch = 0
    if {"ticker", "datadate", "trade_price", "y_return", "actual_tradedate"} <= set(df.columns):
        x = df[["ticker", "datadate", "trade_price", "y_return", "actual_tradedate"]].copy()
        x = x.sort_values(["ticker", "datadate"])
        x["trade_price"] = pd.to_numeric(x["trade_price"], errors="coerce")
        x["y_return"] = pd.to_numeric(x["y_return"], errors="coerce")
        x["actual_tradedate"] = pd.to_datetime(x["actual_tradedate"], errors="coerce")
        x["next_p"] = x.groupby("ticker")["trade_price"].shift(-1)
        x["next_d"] = x.groupby("ticker")["actual_tradedate"].shift(-1)
        valid = (x.trade_price > 0) & (x.next_p > 0) & x.next_d.notna() & (x.next_d <= as_of)
        expected = np.log(x.loc[valid, "next_p"] / x.loc[valid, "trade_price"])
        mismatch = int(((x.loc[valid, "y_return"] - expected).abs() > 1e-6).sum())

    return {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()) if "ticker" in df else None,
        "date_min": str(df["datadate"].min()) if "datadate" in df else None,
        "date_max": str(df["datadate"].max()) if "datadate" in df else None,
        "null_counts": counts,
        "duplicate_ticker_datadate": dupes,
        "zero_y_return_count": zero_y,
        "y_return_mismatch_count": mismatch,
        "unresolved_price_tickers": unresolved_price_tickers,
    }


def write_report(report: dict, md_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# FinRL-X Data Integrity Audit",
        "",
        f"- As of: **{report['as_of']}**",
        f"- Rows: **{report['rows']}**",
        f"- Unique tickers: **{report['unique_tickers']}**",
        f"- Datadate range: **{report['date_min']} → {report['date_max']}**",
        f"- Duplicate (ticker, datadate): **{report['duplicate_ticker_datadate']}**",
        f"- y_return == 0: **{report['zero_y_return_count']}**",
        f"- y_return formula mismatches: **{report['y_return_mismatch_count']}**",
        "",
        "## Null counts",
        "",
    ]
    for k, v in report["null_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Unresolved price tickers",
        "",
        ", ".join(report["unresolved_price_tickers"]) or "None",
        "",
        "> Missing delisted/unavailable prices and future returns are intentionally left NULL; they are never coerced to zero.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/finrl_trading.7z")
    ap.add_argument("--csv-out", default="data/fundamental_data_full.csv")
    ap.add_argument("--audit-md", default="data/FINRLX_DATA_AUDIT.md")
    ap.add_argument("--audit-json", default="data/finrlx_data_audit.json")
    ap.add_argument("--as-of", default="2026-09-27")
    args = ap.parse_args()

    archive = Path(args.archive)
    as_of = pd.Timestamp(args.as_of)

    with tempfile.TemporaryDirectory(prefix="finrlx_") as td:
        workdir = Path(td)
        db = extract_archive(archive, workdir)
        conn = sqlite3.connect(db)
        ensure_columns(conn)
        date_updates = fill_dates(conn)
        price_updates, unresolved = download_missing_prices(conn, as_of)
        recomputed = recompute_returns(conn, as_of)
        report = audit(conn, as_of, unresolved)
        report["date_updates"] = date_updates
        report["price_updates"] = price_updates
        report["y_return_rows_recomputed"] = recomputed

        full = pd.read_sql("SELECT * FROM fundamental_data ORDER BY ticker, datadate", conn)
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(args.csv_out, index=False)
        conn.close()

        write_report(report, Path(args.audit_md), Path(args.audit_json))
        repack_archive(workdir, archive)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
