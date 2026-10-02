#!/usr/bin/env python3
"""Merge latest SEC Company Facts into the persistent FinRL-X main database.

Purpose
-------
The official historical database currently ends at 2026-03-31.  This updater
appends each current constituent's latest publicly filed fiscal period when it
is newer than that ticker's latest stored datadate.

Data-quality rules
------------------
- Never invent a fiscal period.
- Never overwrite historical rows with future information.
- Use SEC latest_fact_end as datadate.
- Use SEC latest_filed as information-availability date.
- Use first available market close on/after filing date as trade_price.
- New latest rows keep y_return NULL until a later observation exists.
- Features that cannot be reconstructed reliably from SEC are carried forward
  from the ticker's prior row and explicitly counted in the audit.
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
import py7zr
import yfinance as yf


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def find_main_db(workdir: Path) -> Path:
    dbs = list(workdir.rglob("*.db"))
    if not dbs:
        raise FileNotFoundError("No SQLite DB found in archive")
    good = []
    for db in dbs:
        try:
            conn = sqlite3.connect(db)
            hit = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fundamental_data'"
            ).fetchone()
            conn.close()
            if hit:
                good.append(db)
        except sqlite3.DatabaseError:
            pass
    if not good:
        raise RuntimeError("No DB containing fundamental_data found")
    good.sort(key=lambda p: p.stat().st_size, reverse=True)
    return good[0]


def get_price_on_or_after(ticker: str, filed: pd.Timestamp) -> tuple[str | None, float | None]:
    if pd.isna(filed):
        return None, None
    yf_ticker = ticker.replace(".", "-")
    start = filed.strftime("%Y-%m-%d")
    end = (filed + pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    try:
        px = yf.download(
            yf_ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if px.empty:
            return None, None
        close = px["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return None, None
        dt = pd.Timestamp(close.index[0]).tz_localize(None)
        price = float(close.iloc[0])
        if not math.isfinite(price) or price <= 0:
            return None, None
        return dt.strftime("%Y-%m-%d"), price
    except Exception:
        return None, None


def derive_overlay(secrow: pd.Series, price: float | None, row_columns: set[str]) -> dict:
    def s(name):
        return num(secrow.get(name))

    rev = s("Revenues")
    ni = s("NetIncomeLoss")
    assets = s("Assets")
    liab = s("Liabilities")
    eq = s("StockholdersEquity")
    eps = s("EarningsPerShareDiluted")
    ocf = s("OperatingCashFlow")
    capex = s("Capex")

    # Older valid SEC snapshot contains these core fields.  Newer richer
    # snapshots may contain additional fields; use them when present.
    gp = s("GrossProfit")
    opinc = s("OperatingIncomeLoss")
    ca = s("CurrentAssets")
    cl = s("CurrentLiabilities")
    cash = s("Cash")
    ar = s("AccountsReceivable")
    apay = s("AccountsPayable")
    cogs = s("CostOfRevenue")
    dcur = s("LongTermDebtCurrent")
    dlt = s("LongTermDebtNoncurrent")
    interest = s("InterestExpense")
    da = s("DepreciationAmortization")
    shares = s("SharesOutstanding")
    if pd.isna(shares) or shares <= 0:
        shares = s("WeightedAverageSharesDiluted")

    d = {}
    if pd.notna(eps):
        d["EPS"] = eps
    if pd.notna(ni) and pd.notna(eq) and eq != 0:
        d["roe"] = ni / eq
    if pd.notna(rev) and pd.notna(assets) and assets != 0:
        d["asset_turnover"] = rev / assets
    if pd.notna(liab) and pd.notna(assets) and assets != 0:
        d["debt_ratio"] = liab / assets
        d["debt_to_assets"] = liab / assets
    if pd.notna(liab) and pd.notna(eq) and eq != 0:
        d["debt_to_equity"] = liab / eq
    if pd.notna(gp) and pd.notna(rev) and rev != 0:
        d["gross_margin"] = gp / rev
    if pd.notna(opinc) and pd.notna(rev) and rev != 0:
        d["operating_margin"] = opinc / rev
    if pd.notna(ca) and pd.notna(cl) and cl != 0:
        d["cur_ratio"] = ca / cl
    if pd.notna(rev) and pd.notna(ar) and ar != 0:
        d["acc_rec_turnover"] = rev / ar
    if pd.isna(cogs) and pd.notna(rev) and pd.notna(gp):
        cogs = rev - gp
    if pd.notna(cogs) and pd.notna(apay) and apay != 0:
        d["payables_turnover"] = cogs / apay
    if pd.notna(opinc) and pd.notna(interest) and interest != 0:
        d["interest_coverage"] = opinc / abs(interest)
    if pd.notna(eq) and pd.notna(shares) and shares > 0:
        d["BPS"] = eq / shares
    if pd.notna(ocf) and pd.notna(shares) and shares > 0:
        d["ocf_per_share"] = ocf / shares
    if pd.notna(ocf) and pd.notna(capex) and pd.notna(shares) and shares > 0:
        fcf = ocf - abs(capex)
        d["fcf_per_share"] = fcf / shares
        d["capex_per_share"] = abs(capex) / shares
        if ocf != 0:
            d["fcf_to_ocf"] = fcf / ocf
    if pd.notna(cash) and pd.notna(shares) and shares > 0:
        d["cash_per_share"] = cash / shares
    if pd.notna(ocf) and pd.notna(cl) and cl != 0:
        d["ocf_ratio"] = ocf / cl
    if pd.notna(ni) and pd.notna(da) and pd.notna(liab) and liab != 0:
        d["solvency_ratio"] = (ni + da) / liab

    if price is not None and pd.notna(shares) and shares > 0:
        mcap = price * shares
        if pd.notna(eps) and eps != 0:
            d["pe"] = price / eps
        if pd.notna(rev) and rev != 0:
            d["ps"] = price / (rev / shares)
        if pd.notna(eq) and eq != 0:
            d["pb"] = mcap / eq
        debt = sum(x for x in [dcur, dlt] if pd.notna(x))
        if debt > 0 and mcap > 0:
            d["debt_to_mktcap"] = debt / mcap
        ebitda = opinc + da if pd.notna(opinc) and pd.notna(da) else np.nan
        if pd.notna(ebitda) and ebitda != 0:
            d["ev_multiple"] = (mcap + debt - (cash if pd.notna(cash) else 0)) / ebitda

    return {
        k: float(v)
        for k, v in d.items()
        if k in row_columns and pd.notna(v) and math.isfinite(float(v))
    }


def export_csv(conn: sqlite3.Connection, out_path: Path) -> None:
    df = pd.read_sql("SELECT * FROM fundamental_data ORDER BY ticker, datadate", conn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def repack(workdir: Path, archive: Path) -> None:
    tmp = archive.with_suffix(".tmp.7z")
    if tmp.exists():
        tmp.unlink()
    with py7zr.SevenZipFile(tmp, "w") as z:
        for p in sorted(workdir.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(workdir)))
    tmp.replace(archive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/finrl_trading.7z")
    ap.add_argument("--sec", default="data/sec_companyfacts_latest.csv")
    ap.add_argument("--csv-out", default="data/fundamental_data_full.csv")
    ap.add_argument("--audit", default="data/main_db_update_audit.json")
    ap.add_argument("--as-of", default="2026-10-02")
    args = ap.parse_args()

    archive = Path(args.archive)
    sec_path = Path(args.sec)
    as_of = pd.Timestamp(args.as_of)

    sec = pd.read_csv(sec_path)
    if "latest_fact_end" not in sec.columns or "latest_filed" not in sec.columns:
        raise ValueError("SEC snapshot lacks latest_fact_end/latest_filed")
    sec["latest_fact_end"] = pd.to_datetime(sec["latest_fact_end"], errors="coerce")
    sec["latest_filed"] = pd.to_datetime(sec["latest_filed"], errors="coerce")
    sec = sec[
        sec["latest_fact_end"].notna()
        & sec["latest_filed"].notna()
        & (sec["latest_filed"] <= as_of)
    ].copy()

    with tempfile.TemporaryDirectory(prefix="finrlx_main_update_") as td:
        workdir = Path(td)
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(workdir)
        db = find_main_db(workdir)
        conn = sqlite3.connect(db)

        info = conn.execute("PRAGMA table_info(fundamental_data)").fetchall()
        cols = [r[1] for r in info]
        row_cols = set(cols)
        insert_cols = [c for c in cols if c not in {"id", "created_at"}]
        qmarks = ",".join("?" for _ in insert_cols)
        insert_sql = (
            f"INSERT OR REPLACE INTO fundamental_data ({','.join(insert_cols)}) "
            f"VALUES ({qmarks})"
        )

        before = conn.execute(
            "SELECT COUNT(*), MIN(datadate), MAX(datadate) FROM fundamental_data"
        ).fetchone()

        added = 0
        skipped_not_newer = 0
        skipped_no_history = []
        skipped_no_price = []
        overlay_counts = {}
        new_dates = []

        for secrow in sec.itertuples(index=False):
            ticker = str(secrow.ticker)
            sec_series = pd.Series(secrow._asdict())
            fact_end = pd.Timestamp(sec_series["latest_fact_end"])
            filed = pd.Timestamp(sec_series["latest_filed"])

            base = pd.read_sql(
                "SELECT * FROM fundamental_data WHERE ticker=? ORDER BY datadate DESC LIMIT 1",
                conn,
                params=[ticker],
            )
            if base.empty:
                skipped_no_history.append(ticker)
                continue

            base_date = pd.to_datetime(base.iloc[0]["datadate"], errors="coerce")
            if pd.notna(base_date) and fact_end <= base_date:
                skipped_not_newer += 1
                continue

            actual_trade_date, trade_price = get_price_on_or_after(ticker, filed)
            if trade_price is None:
                skipped_no_price.append(ticker)

            row = base.iloc[0].to_dict()
            row["ticker"] = ticker
            row["datadate"] = fact_end.strftime("%Y-%m-%d")
            row["y_return"] = None
            if "filing_date" in row_cols:
                row["filing_date"] = filed.strftime("%Y-%m-%d")
            if "accepted_date" in row_cols:
                row["accepted_date"] = filed.strftime("%Y-%m-%d")
            if "tradedate" in row_cols:
                row["tradedate"] = filed.strftime("%Y-%m-%d")
            if "actual_tradedate" in row_cols:
                row["actual_tradedate"] = actual_trade_date
            if trade_price is not None:
                if "trade_price" in row_cols:
                    row["trade_price"] = trade_price
                if "adj_close_q" in row_cols:
                    row["adj_close_q"] = trade_price
            else:
                if "trade_price" in row_cols:
                    row["trade_price"] = None
                if "adj_close_q" in row_cols:
                    row["adj_close_q"] = None

            overlay = derive_overlay(sec_series, trade_price, row_cols)
            for k, v in overlay.items():
                row[k] = v
                overlay_counts[k] = overlay_counts.get(k, 0) + 1

            values = []
            for col in insert_cols:
                v = row.get(col)
                if not isinstance(v, (dict, list)) and pd.isna(v):
                    v = None
                values.append(v)
            conn.execute(insert_sql, values)
            added += 1
            new_dates.append(fact_end.strftime("%Y-%m-%d"))

        conn.commit()

        # Recompute y_return in strict ticker sequence.  Only a pair of known
        # positive prices creates a label.  The newest observation remains NULL.
        df = pd.read_sql(
            "SELECT ticker, datadate, trade_price FROM fundamental_data ORDER BY ticker, datadate",
            conn,
        )
        df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")
        df["next_price"] = df.groupby("ticker")["trade_price"].shift(-1)
        good = (df["trade_price"] > 0) & (df["next_price"] > 0)
        df["new_y"] = np.nan
        df.loc[good, "new_y"] = np.log(
            df.loc[good, "next_price"] / df.loc[good, "trade_price"]
        )
        df.loc[df["new_y"].abs() < 1e-12, "new_y"] = np.nan
        conn.executemany(
            "UPDATE fundamental_data SET y_return=? WHERE ticker=? AND datadate=?",
            [
                (
                    None if pd.isna(r.new_y) else float(r.new_y),
                    r.ticker,
                    str(r.datadate)[:10],
                )
                for r in df.itertuples(index=False)
            ],
        )
        conn.commit()

        after = conn.execute(
            "SELECT COUNT(*), MIN(datadate), MAX(datadate) FROM fundamental_data"
        ).fetchone()
        latest_count = conn.execute(
            "SELECT COUNT(*) FROM fundamental_data WHERE datadate > '2026-03-31'"
        ).fetchone()[0]
        q2_count = conn.execute(
            "SELECT COUNT(*) FROM fundamental_data WHERE datadate >= '2026-04-01' AND datadate <= '2026-08-31'"
        ).fetchone()[0]

        export_csv(conn, Path(args.csv_out))
        conn.close()
        repack(workdir, archive)

    audit = {
        "as_of": args.as_of,
        "before_rows": before[0],
        "before_min_datadate": before[1],
        "before_max_datadate": before[2],
        "after_rows": after[0],
        "after_min_datadate": after[1],
        "after_max_datadate": after[2],
        "rows_added": added,
        "rows_after_2026_03_31": latest_count,
        "rows_2026_04_01_through_2026_08_31": q2_count,
        "skipped_not_newer": skipped_not_newer,
        "skipped_no_history": sorted(set(skipped_no_history)),
        "rows_without_trade_price": sorted(set(skipped_no_price)),
        "overlay_counts": dict(sorted(overlay_counts.items())),
        "new_datadates": sorted(set(new_dates)),
    }
    Path(args.audit).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
