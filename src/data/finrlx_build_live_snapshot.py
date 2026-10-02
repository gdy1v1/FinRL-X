#!/usr/bin/env python3
"""Build a live inference DB for today's FinRL-X stock selection.

Design:
- Keep the historical training DB unchanged.
- Start each current S&P 500 member from its latest historical factor row.
- Overlay only SEC facts that were public by --as-of and can be mapped
  defensibly to trained FinRL-X features.
- Carry forward trained factors that cannot be reconstructed reliably.
- Use the latest completed-session adjusted close as the live trade_price.
- Never create y_return for a live inference row.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def ratio(a, b):
    a, b = num(a), num(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def latest_prices(tickers: list[str], cutoff: pd.Timestamp) -> tuple[dict[str, float], str | None]:
    if not tickers:
        return {}, None
    yf_tickers = [t.replace(".", "-") for t in tickers]
    start = (cutoff - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    # yfinance end is exclusive; +1 includes the cutoff session if present.
    end = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if px.empty:
        return {}, None
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
    prices = {}
    used_dates = []
    for tic in tickers:
        key = tic.replace(".", "-")
        if key not in close.columns:
            continue
        s = close[key].dropna()
        s = s[s.index <= cutoff]
        if s.empty:
            continue
        prices[tic] = float(s.iloc[-1])
        used_dates.append(pd.Timestamp(s.index[-1]).strftime("%Y-%m-%d"))
    return prices, max(used_dates) if used_dates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finrl_trading.db")
    ap.add_argument("--output-db", default="data/finrl_live.db")
    ap.add_argument("--sec", default="data/sec_companyfacts_latest.csv")
    ap.add_argument("--constituents", default="data/sp500_historical_constituents.csv")
    ap.add_argument("--as-of", default="2026-10-02")
    ap.add_argument("--price-cutoff", default="2026-10-01")
    ap.add_argument("--audit", default="data/live_snapshot_audit.json")
    args = ap.parse_args()

    as_of = pd.Timestamp(args.as_of)
    price_cutoff = pd.Timestamp(args.price_cutoff)

    shutil.copy2(args.db, args.output_db)
    conn = sqlite3.connect(args.output_db)
    hist = pd.read_csv(args.constituents)
    hist["date"] = pd.to_datetime(hist["date"])
    valid = hist[hist["date"] <= as_of]
    if valid.empty:
        raise RuntimeError("No current constituent snapshot")
    members = sorted({x.strip() for x in valid.iloc[-1]["tickers"].split(",") if x.strip()})

    sec = pd.read_csv(args.sec)
    sec = sec[sec["ticker"].isin(members)].copy()
    if "latest_filed" in sec:
        sec["latest_filed"] = pd.to_datetime(sec["latest_filed"], errors="coerce")
        sec = sec[(sec["latest_filed"].isna()) | (sec["latest_filed"] <= as_of)]
    sec_by = sec.set_index("ticker", drop=False)

    prices, price_date = latest_prices(members, price_cutoff)

    info = conn.execute("PRAGMA table_info(fundamental_data)").fetchall()
    cols = [r[1] for r in info]
    insert_cols = [c for c in cols if c not in {"id", "created_at"}]
    qmarks = ",".join("?" for _ in insert_cols)
    sql = f"INSERT OR REPLACE INTO fundamental_data ({','.join(insert_cols)}) VALUES ({qmarks})"

    created = 0
    sec_overlay = 0
    carried_only = 0
    no_history = []
    no_price = []
    feature_updates = {}

    for ticker in members:
        base = pd.read_sql(
            "SELECT * FROM fundamental_data WHERE ticker=? ORDER BY datadate DESC LIMIT 1",
            conn,
            params=[ticker],
        )
        if base.empty:
            no_history.append(ticker)
            continue
        row = base.iloc[0].to_dict()
        base_date = pd.to_datetime(str(row.get("datadate"))[:10], errors="coerce")

        secrow = sec_by.loc[ticker] if ticker in sec_by.index else None
        sec_end = pd.NaT
        if secrow is not None:
            sec_end = pd.to_datetime(secrow.get("latest_fact_end"), errors="coerce")

        # Make every scoreable current member a live row.  If SEC has a newer
        # period, use that period end; otherwise use as-of as a synthetic sort key.
        live_date = sec_end if pd.notna(sec_end) and (pd.isna(base_date) or sec_end > base_date) else as_of
        row["datadate"] = live_date.strftime("%Y-%m-%d")
        row["ticker"] = ticker
        row["y_return"] = None
        if "tradedate" in row:
            row["tradedate"] = args.as_of
        if "actual_tradedate" in row:
            row["actual_tradedate"] = args.as_of

        price = prices.get(ticker)
        if price is None:
            no_price.append(ticker)
        else:
            row["trade_price"] = price
            row["adj_close_q"] = price

        updated = []
        if secrow is not None:
            def s(name):
                return num(secrow.get(name))

            rev = s("Revenues")
            cogs = s("CostOfRevenue")
            gp = s("GrossProfit")
            opinc = s("OperatingIncomeLoss")
            ni = s("NetIncomeLoss")
            assets = s("Assets")
            ca = s("CurrentAssets")
            liab = s("Liabilities")
            cl = s("CurrentLiabilities")
            eq = s("StockholdersEquity")
            cash = s("Cash")
            ar = s("AccountsReceivable")
            apay = s("AccountsPayable")
            dcur = s("LongTermDebtCurrent")
            dlt = s("LongTermDebtNoncurrent")
            interest = s("InterestExpense")
            da = s("DepreciationAmortization")
            eps = s("EarningsPerShareDiluted")
            shares = s("SharesOutstanding")
            if pd.isna(shares) or shares <= 0:
                shares = s("WeightedAverageSharesDiluted")
            ocf = s("OperatingCashFlow")
            capex = s("Capex")

            derived = {}
            if pd.notna(eps):
                derived["EPS"] = eps
            if pd.notna(eq) and pd.notna(shares) and shares > 0:
                derived["BPS"] = eq / shares
            if pd.notna(ni) and pd.notna(eq) and eq != 0:
                derived["roe"] = ni / eq
            if pd.notna(gp) and pd.notna(rev) and rev != 0:
                derived["gross_margin"] = gp / rev
            if pd.notna(opinc) and pd.notna(rev) and rev != 0:
                derived["operating_margin"] = opinc / rev
            if pd.notna(liab) and pd.notna(assets) and assets != 0:
                derived["debt_ratio"] = liab / assets
                derived["debt_to_assets"] = liab / assets
            if pd.notna(liab) and pd.notna(eq) and eq != 0:
                derived["debt_to_equity"] = liab / eq
            if pd.notna(ca) and pd.notna(cl) and cl != 0:
                derived["cur_ratio"] = ca / cl
            if pd.notna(rev) and pd.notna(ar) and ar != 0:
                derived["acc_rec_turnover"] = rev / ar
            if pd.notna(rev) and pd.notna(assets) and assets != 0:
                derived["asset_turnover"] = rev / assets
            if pd.isna(cogs) and pd.notna(rev) and pd.notna(gp):
                cogs = rev - gp
            if pd.notna(cogs) and pd.notna(apay) and apay != 0:
                derived["payables_turnover"] = cogs / apay
            if pd.notna(opinc) and pd.notna(interest) and interest != 0:
                derived["interest_coverage"] = opinc / abs(interest)
            if pd.notna(ocf) and pd.notna(capex) and pd.notna(shares) and shares > 0:
                fcf = ocf - abs(capex)
                derived["fcf_per_share"] = fcf / shares
                derived["capex_per_share"] = abs(capex) / shares
                if ocf != 0:
                    derived["fcf_to_ocf"] = fcf / ocf
            if pd.notna(ocf) and pd.notna(shares) and shares > 0:
                derived["ocf_per_share"] = ocf / shares
            if pd.notna(cash) and pd.notna(shares) and shares > 0:
                derived["cash_per_share"] = cash / shares
            if pd.notna(ocf) and pd.notna(cl) and cl != 0:
                derived["ocf_ratio"] = ocf / cl
            if pd.notna(ni) and pd.notna(da) and pd.notna(liab) and liab != 0:
                derived["solvency_ratio"] = (ni + da) / liab

            mcap = np.nan
            if price is not None and pd.notna(shares) and shares > 0:
                mcap = price * shares
                if pd.notna(eps) and eps != 0:
                    derived["pe"] = price / eps
                if pd.notna(rev) and rev != 0:
                    derived["ps"] = price / (rev / shares)
                if pd.notna(eq) and eq != 0:
                    derived["pb"] = mcap / eq
                debt = sum(x for x in [dcur, dlt] if pd.notna(x))
                if debt > 0 and mcap > 0:
                    derived["debt_to_mktcap"] = debt / mcap
                ebitda = opinc + da if pd.notna(opinc) and pd.notna(da) else np.nan
                if pd.notna(ebitda) and ebitda != 0:
                    ev = mcap + debt - (cash if pd.notna(cash) else 0)
                    derived["ev_multiple"] = ev / ebitda

            for k, v in derived.items():
                if k in row and pd.notna(v) and math.isfinite(float(v)):
                    row[k] = float(v)
                    updated.append(k)
                    feature_updates[k] = feature_updates.get(k, 0) + 1

            filed = secrow.get("latest_filed")
            if pd.notna(filed):
                fd = pd.Timestamp(filed).strftime("%Y-%m-%d")
                if "filing_date" in row:
                    row["filing_date"] = fd
                if "accepted_date" in row:
                    row["accepted_date"] = fd
            if updated:
                sec_overlay += 1
            else:
                carried_only += 1
        else:
            carried_only += 1

        values = []
        for col in insert_cols:
            v = row.get(col)
            if not isinstance(v, (list, dict)) and pd.isna(v):
                v = None
            values.append(v)
        conn.execute(sql, values)
        created += 1

    conn.commit()
    conn.close()

    audit = {
        "as_of": args.as_of,
        "price_cutoff": args.price_cutoff,
        "price_date_used": price_date,
        "current_members": len(members),
        "live_rows_created": created,
        "rows_with_sec_overlay": sec_overlay,
        "rows_carry_forward_only": carried_only,
        "unscored_no_history": no_history,
        "missing_reference_price": no_price,
        "feature_update_counts": dict(sorted(feature_updates.items())),
    }
    Path(args.audit).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
