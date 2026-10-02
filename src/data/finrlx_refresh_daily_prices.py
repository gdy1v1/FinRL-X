#!/usr/bin/env python3
"""Build a persistent FinRL-X daily OHLCV cache.

The official adaptive-rotation pipeline consumes files named
    data/fmp_daily/{SYMBOL}_daily.csv
with columns:
    date,open,high,low,close,volume

This builder expands the universe beyond the small adaptive-rotation asset set:
- every symbol appearing in the point-in-time S&P 500 constituent history;
- the latest/current S&P 500 members;
- all symbols referenced by the adaptive-rotation config;
- core benchmarks (^GSPC, ^VIX, SPY, QQQ).

Yahoo Finance is used as the free default, matching deploy.sh.  Historical
symbols that Yahoo no longer serves are recorded in the audit instead of being
silently fabricated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
import yaml


CORE_SYMBOLS = {"^GSPC", "^VIX", "SPY", "QQQ"}


def parse_members(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {x.strip() for x in str(value).split(",") if x.strip()}


def yahoo_symbol(symbol: str) -> str:
    # Yahoo encodes US share classes with a dash (BRK.B -> BRK-B, BF.B -> BF-B).
    if "." in symbol and not symbol.startswith("^"):
        return symbol.replace(".", "-")
    return symbol


def load_universe(constituents_path: Path, config_path: Path | None) -> tuple[list[str], set[str]]:
    df = pd.read_csv(constituents_path)
    if "date" not in df.columns or "tickers" not in df.columns:
        raise ValueError(f"{constituents_path} must contain date,tickers")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        raise ValueError(f"{constituents_path} has no valid rows")

    historical: set[str] = set()
    for value in df["tickers"]:
        historical.update(parse_members(value))

    current = parse_members(df.iloc[-1]["tickers"])
    symbols = set(historical) | CORE_SYMBOLS

    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        for group in (cfg.get("asset_groups") or {}).values():
            symbols.update(group.get("symbols") or [])

        fallback = (cfg.get("portfolio") or {}).get("fallback") or {}
        symbols.update(fallback.get("symbols") or [])

        benchmark = cfg.get("benchmark") or {}
        if benchmark.get("excess_return_benchmark"):
            symbols.add(benchmark["excess_return_benchmark"])

    return sorted(s for s in symbols if s), current


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Individual downloads can still return a one-symbol MultiIndex.
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}; got {list(df.columns)}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df = df.drop_duplicates("date", keep="last")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def download_symbol(symbol: str, start: str, end_exclusive: str, retries: int = 3) -> pd.DataFrame:
    alias = yahoo_symbol(symbol)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                alias,
                start=start,
                end=end_exclusive,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            df = normalize_frame(raw)
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if attempt < retries:
            time.sleep(1.5 * attempt)

    if last_error:
        raise last_error
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def merge_existing(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return fresh

    try:
        old = pd.read_csv(path)
        old = normalize_frame(old)
    except Exception:
        old = pd.DataFrame(columns=fresh.columns)

    out = pd.concat([old, fresh], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates("date", keep="last")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def existing_refresh_start(path: Path, requested_start: pd.Timestamp) -> str:
    if not path.exists():
        return requested_start.strftime("%Y-%m-%d")

    try:
        old = pd.read_csv(path, usecols=["date"])
        last = pd.to_datetime(old["date"], errors="coerce").max()
        if pd.isna(last):
            return requested_start.strftime("%Y-%m-%d")
        # Re-fetch a short overlap so late adjustments/repairs are not missed.
        refresh = max(requested_start, last - pd.Timedelta(days=10))
        return refresh.strftime("%Y-%m-%d")
    except Exception:
        return requested_start.strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constituents", default="data/sp500_historical_constituents.csv")
    ap.add_argument("--config", default="src/strategies/AdaptiveRotationConf_v1.2.1.yaml")
    ap.add_argument("--output-dir", default="data/fmp_daily")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", required=True, help="Inclusive last completed trading date")
    ap.add_argument("--audit", default="data/daily_price_audit.json")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    constituents_path = Path(args.constituents)
    config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir)
    audit_path = Path(args.audit)

    requested_start = pd.Timestamp(args.start)
    cutoff = pd.Timestamp(args.end)
    end_exclusive = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    symbols, current_members = load_universe(constituents_path, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded: list[str] = []
    failed: dict[str, str] = {}
    ranges: dict[str, dict[str, object]] = {}

    print(f"Universe: {len(symbols)} symbols")
    print(f"Current S&P 500 members: {len(current_members)}")
    print(f"Requested range: {requested_start.date()} -> {cutoff.date()}")

    for idx, symbol in enumerate(symbols, 1):
        path = output_dir / f"{symbol}_daily.csv"
        fetch_start = existing_refresh_start(path, requested_start)

        try:
            fresh = download_symbol(symbol, fetch_start, end_exclusive, retries=args.retries)
            if fresh.empty:
                # Keep an existing historical file even if a delisted symbol has no new rows.
                if path.exists():
                    existing = normalize_frame(pd.read_csv(path))
                    if not existing.empty:
                        combined = existing
                    else:
                        raise RuntimeError("Yahoo returned no rows")
                else:
                    raise RuntimeError("Yahoo returned no rows")
            else:
                combined = merge_existing(path, fresh)

            combined.to_csv(path, index=False)
            succeeded.append(symbol)
            ranges[symbol] = {
                "rows": int(len(combined)),
                "min_date": str(combined["date"].min()),
                "max_date": str(combined["date"].max()),
            }
            print(f"[{idx:03d}/{len(symbols)}] OK   {symbol:8s} {len(combined):5d} rows through {combined['date'].max()}")
        except Exception as exc:  # noqa: BLE001
            failed[symbol] = str(exc)
            print(f"[{idx:03d}/{len(symbols)}] FAIL {symbol:8s} {exc}")

    fresh_current = []
    stale_current = []
    missing_current = []

    for symbol in sorted(current_members):
        info = ranges.get(symbol)
        if not info:
            missing_current.append(symbol)
            continue
        max_date = pd.Timestamp(str(info["max_date"]))
        # One-session tolerance covers occasional symbol-specific missing bars.
        if max_date >= cutoff - pd.Timedelta(days=3):
            fresh_current.append(symbol)
        else:
            stale_current.append({"symbol": symbol, "max_date": str(info["max_date"])})

    all_max_dates = [pd.Timestamp(str(v["max_date"])) for v in ranges.values() if v.get("max_date")]
    all_min_dates = [pd.Timestamp(str(v["min_date"])) for v in ranges.values() if v.get("min_date")]

    audit = {
        "price_cutoff": cutoff.strftime("%Y-%m-%d"),
        "requested_start": requested_start.strftime("%Y-%m-%d"),
        "universe_size": len(symbols),
        "current_member_count": len(current_members),
        "files_written": len(succeeded),
        "failed_count": len(failed),
        "failed": failed,
        "current_fresh_count": len(fresh_current),
        "current_missing_count": len(missing_current),
        "current_missing": missing_current,
        "current_stale_count": len(stale_current),
        "current_stale": stale_current,
        "global_min_date": min(all_min_dates).strftime("%Y-%m-%d") if all_min_dates else None,
        "global_max_date": max(all_max_dates).strftime("%Y-%m-%d") if all_max_dates else None,
        "format": "per-symbol CSV: date,open,high,low,close,volume",
        "source": "Yahoo Finance via yfinance (free default, matching official deploy.sh)",
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(audit, indent=2, ensure_ascii=False))

    # Current-universe coverage is the hard requirement for today's selection.
    # Historical delisted symbols may legitimately be unavailable from Yahoo.
    min_required = min(480, max(1, math.floor(len(current_members) * 0.95)))
    if len(fresh_current) < min_required:
        raise SystemExit(
            f"Insufficient current-member daily coverage: {len(fresh_current)} < {min_required}"
        )


if __name__ == "__main__":
    main()
