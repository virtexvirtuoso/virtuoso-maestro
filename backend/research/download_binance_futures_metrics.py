#!/usr/bin/env python3
"""
Download Binance Futures 5-minute metrics from data.binance.vision.

Data includes: OI, Top Trader L/S, Global L/S, Taker Buy/Sell ratio
Resolution: 5 minutes (288 bars/day)
Available: 2021-12-01 → present
Source: https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{DATE}.zip

Usage:
    python download_binance_futures_metrics.py                    # Download all assets
    python download_binance_futures_metrics.py --symbols BTC ETH  # Specific assets
    python download_binance_futures_metrics.py --start 2024-01-01 # Custom start date
    python download_binance_futures_metrics.py --compile           # Compile daily zips into per-asset CSVs
"""

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# === CONFIG ===
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/derivatives_5m"))
RAW_DIR = OUTPUT_DIR / "raw"  # daily CSVs extracted from zips
COMPILED_DIR = OUTPUT_DIR / "compiled"  # merged per-asset CSVs

# 24 assets matching our spot universe
SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "DOGE": "DOGEUSDT",
    "DOT": "DOTUSDT",
    "LINK": "LINKUSDT",
    "MATIC": "MATICUSDT",
    "UNI": "UNIUSDT",
    "XRP": "XRPUSDT",
    "AAVE": "AAVEUSDT",
    "ATOM": "ATOMUSDT",
    "CRV": "CRVUSDT",
    "DYDX": "DYDXUSDT",
    "FTM": "FTMUSDT",
    "INJ": "INJUSDT",
    "OP": "OPUSDT",
    "ARB": "ARBUSDT",
    "SUI": "SUIUSDT",
    "TIA": "TIAUSDT",
    "FET": "FETUSDT",
    "RENDER": "RENDERUSDT",
    "TAO": "TAOUSDT",
    "SEI": "SEIUSDT",
}

# Known listing dates (approximate) — skip dates before listing
LISTING_DATES = {
    "ARBUSDT": "2023-03-23",
    "SUIUSDT": "2023-05-03",
    "TIAUSDT": "2023-10-31",
    "SEIUSDT": "2023-08-15",
    "TAOUSDT": "2024-04-10",
    "RENDERUSDT": "2024-07-20",  # was RNDR before
    "DYDXUSDT": "2023-02-01",
}

DEFAULT_START = "2022-01-01"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2
CONCURRENT_SKIP_404 = True  # skip silently on 404

# Columns in the metrics CSV
EXPECTED_COLS = [
    "create_time", "symbol",
    "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]


def download_day(symbol: str, date_str: str, session: requests.Session) -> pd.DataFrame | None:
    """Download and extract one day's metrics zip. Returns DataFrame or None."""
    url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    raw_csv = RAW_DIR / symbol / f"{symbol}-metrics-{date_str}.csv"

    # Skip if already downloaded
    if raw_csv.exists():
        try:
            df = pd.read_csv(raw_csv)
            if len(df) > 0:
                return df
        except Exception:
            pass  # re-download corrupt file

    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 451:
                print(f"  ⚠ Geo-blocked for {symbol} — try VPS", file=sys.stderr)
                return None
            r.raise_for_status()

            # Extract CSV from zip
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith('.csv') and 'metrics' in n]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)

            if len(df) == 0:
                return None

            # Save raw CSV
            raw_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_csv, index=False)
            return df

        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  ✗ Failed {symbol} {date_str}: {e}", file=sys.stderr)
                return None

    return None


def download_symbol(symbol: str, start_date: str, end_date: str) -> dict:
    """Download all daily metrics for one symbol. Returns stats dict."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Respect listing dates
    listing = LISTING_DATES.get(symbol)
    if listing:
        listing_dt = datetime.strptime(listing, "%Y-%m-%d")
        if start < listing_dt:
            start = listing_dt

    total_days = (end - start).days + 1
    downloaded = 0
    skipped = 0
    cached = 0
    failed = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "Maestro/1.0"})

    # Build list of dates to download (not cached)
    dates_to_download = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        raw_csv = RAW_DIR / symbol / f"{symbol}-metrics-{date_str}.csv"
        if raw_csv.exists():
            cached += 1
        else:
            dates_to_download.append(date_str)
        current += timedelta(days=1)

    # Download with thread pool
    batch_start = time.time()

    def _dl(date_str):
        s = requests.Session()
        s.headers.update({"User-Agent": "Maestro/1.0"})
        return date_str, download_day(symbol, date_str, s)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_dl, d): d for d in dates_to_download}
        for i, future in enumerate(as_completed(futures), 1):
            date_str, df = future.result()
            if df is not None:
                downloaded += 1
            else:
                skipped += 1

            if i % 100 == 0:
                elapsed = time.time() - batch_start
                rate = i / max(elapsed, 0.1)
                pct = (i + cached) / total_days * 100
                print(f"    {symbol}: {downloaded} new, {cached} cached, {skipped} skipped "
                      f"({pct:.0f}%, {rate:.1f} req/s)")

    return {
        "symbol": symbol,
        "total_days": total_days,
        "downloaded": downloaded,
        "cached": cached,
        "skipped": skipped,
        "failed": failed,
    }


def compile_symbol(symbol: str) -> dict:
    """Merge all raw daily CSVs into one compiled CSV per symbol. Returns stats."""
    raw_dir = RAW_DIR / symbol
    if not raw_dir.exists():
        return {"symbol": symbol, "rows": 0, "days": 0}

    csv_files = sorted(raw_dir.glob(f"{symbol}-metrics-*.csv"))
    if not csv_files:
        return {"symbol": symbol, "rows": 0, "days": 0}

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return {"symbol": symbol, "rows": 0, "days": 0}

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["create_time"]).sort_values("create_time").reset_index(drop=True)

    # Rename columns for clarity
    merged = merged.rename(columns={
        "create_time": "timestamp",
        "sum_open_interest": "oi_contracts",
        "sum_open_interest_value": "oi_usd",
        "count_toptrader_long_short_ratio": "top_trader_accounts_lsr",
        "sum_toptrader_long_short_ratio": "top_trader_positions_lsr",
        "count_long_short_ratio": "global_accounts_lsr",
        "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
    })

    # Save 5m compiled
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    out_5m = COMPILED_DIR / f"{symbol}_5m.csv"
    merged.to_csv(out_5m, index=False)

    # Also create 15m resampled
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged.set_index("timestamp")
    numeric_cols = ["oi_contracts", "oi_usd", "top_trader_accounts_lsr",
                    "top_trader_positions_lsr", "global_accounts_lsr", "taker_buy_sell_ratio"]

    resampled_15m = merged[numeric_cols].resample("15min").agg({
        "oi_contracts": "last",
        "oi_usd": "last",
        "top_trader_accounts_lsr": "mean",
        "top_trader_positions_lsr": "mean",
        "global_accounts_lsr": "mean",
        "taker_buy_sell_ratio": "mean",
    }).dropna()
    resampled_15m["symbol"] = symbol
    out_15m = COMPILED_DIR / f"{symbol}_15m.csv"
    resampled_15m.to_csv(out_15m)

    # And 1h
    resampled_1h = merged[numeric_cols].resample("1h").agg({
        "oi_contracts": "last",
        "oi_usd": "last",
        "top_trader_accounts_lsr": "mean",
        "top_trader_positions_lsr": "mean",
        "global_accounts_lsr": "mean",
        "taker_buy_sell_ratio": "mean",
    }).dropna()
    resampled_1h["symbol"] = symbol
    out_1h = COMPILED_DIR / f"{symbol}_1h.csv"
    resampled_1h.to_csv(out_1h)

    rows_5m = len(merged)
    first = merged.index[0] if len(merged) > 0 else None
    last = merged.index[-1] if len(merged) > 0 else None

    return {
        "symbol": symbol,
        "rows_5m": rows_5m,
        "rows_15m": len(resampled_15m),
        "rows_1h": len(resampled_1h),
        "days": len(csv_files),
        "first": str(first),
        "last": str(last),
    }


def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures 5m metrics")
    parser.add_argument("--symbols", nargs="+", help="Asset tickers (e.g. BTC ETH SOL)")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Start date (default: {DEFAULT_START})")
    parser.add_argument("--end", default=None, help="End date (default: yesterday)")
    parser.add_argument("--compile", action="store_true", help="Compile raw CSVs into merged per-asset files")
    parser.add_argument("--compile-only", action="store_true", help="Only compile, don't download")
    args = parser.parse_args()

    # Determine end date (yesterday — today's file may not be ready)
    if args.end:
        end_date = args.end
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Filter symbols
    if args.symbols:
        symbols = {k: v for k, v in SYMBOLS.items() if k in [s.upper() for s in args.symbols]}
    else:
        symbols = SYMBOLS

    if not symbols:
        print("No valid symbols specified")
        sys.exit(1)

    # Create dirs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.compile_only:
        print(f"📥 Downloading {len(symbols)} symbols: {args.start} → {end_date}")
        print(f"   Output: {RAW_DIR}")
        print()

        total_stats = {"downloaded": 0, "cached": 0, "skipped": 0}
        for ticker, symbol in sorted(symbols.items()):
            print(f"  ▸ {ticker} ({symbol})")
            t0 = time.time()
            stats = download_symbol(symbol, args.start, end_date)
            elapsed = time.time() - t0
            print(f"    ✓ {stats['downloaded']} new + {stats['cached']} cached + "
                  f"{stats['skipped']} skipped ({elapsed:.1f}s)")
            total_stats["downloaded"] += stats["downloaded"]
            total_stats["cached"] += stats["cached"]
            total_stats["skipped"] += stats["skipped"]

        print(f"\n📊 Total: {total_stats['downloaded']} downloaded, "
              f"{total_stats['cached']} cached, {total_stats['skipped']} skipped")

    if args.compile or args.compile_only:
        print(f"\n📦 Compiling raw CSVs → merged files")
        for ticker, symbol in sorted(symbols.items()):
            stats = compile_symbol(symbol)
            if stats.get("rows_5m", 0) > 0:
                print(f"  ✓ {ticker}: {stats['rows_5m']:,} rows (5m), "
                      f"{stats['rows_15m']:,} (15m), {stats['rows_1h']:,} (1h) | "
                      f"{stats['first']} → {stats['last']}")
            else:
                print(f"  ✗ {ticker}: no data")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
