"""
Download Binance Futures aggTrades and build 1-minute orderflow CSVs.
Outputs canonical {symbol}_1m.csv files in maestro-data/bars/1m/v1/.

Usage:
  python download_orderflow.py                    # All symbols, full date range
  python download_orderflow.py --symbols SOLUSDT  # Single symbol
  python download_orderflow.py --start 2024-06-01 # Custom start date

Author: Maestro 🎼
Date: 2026-03-07
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from backend.config.data_paths import RAW_AGGTRADES, BARS_1M_V1, raw_aggtrades_dir

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
# Raw aggTrades zips: maestro-data/raw/binance_aggtrades/{SYMBOL}/
# Aggregated 1m bars output: maestro-data/bars/1m/v1/
RAW_DIR = RAW_AGGTRADES        # split per symbol via raw_aggtrades_dir(sym)
OUT_DIR = BARS_1M_V1
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to download (confirmed available on Binance futures)
DEFAULT_SYMBOLS = [
    "SOLUSDT",
    "SUIUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "INJUSDT",
    "OPUSDT",
    "ARBUSDT",
    "TIAUSDT",
    "FETUSDT",
    "TAOUSDT",
    "RNDRUSDT",  # RENDER on Binance futures
]

# Default date range matching BTC/ETH data
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-02-24"


def download_day(symbol: str, date_str: str) -> Path | None:
    """Download a single day's aggTrades ZIP. Returns path or None."""
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    sym_dir = raw_aggtrades_dir(symbol)
    sym_dir.mkdir(parents=True, exist_ok=True)
    raw_path = sym_dir / filename

    # Skip if already downloaded
    if raw_path.exists() and raw_path.stat().st_size > 100:
        return raw_path

    url = f"{BASE_URL}/{symbol}/{filename}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            raw_path.write_bytes(resp.content)
            return raw_path
        elif resp.status_code == 404:
            return None  # Data doesn't exist for this date
        else:
            print(f"    HTTP {resp.status_code} for {filename}")
            return None
    except Exception as e:
        print(f"    Error downloading {filename}: {e}")
        return None


def process_zip_to_1m(zip_path: Path) -> pd.DataFrame | None:
    """Extract aggTrades ZIP and aggregate to 1-minute orderflow bars."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
    except Exception as e:
        print(f"    Error reading {zip_path.name}: {e}")
        return None

    if len(df) == 0:
        return None

    # Binance aggTrades columns:
    # agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["dollar"] = df["price"] * df["quantity"]

    # is_buyer_maker=True means the buyer placed a limit order, so the SELLER was aggressive
    # is_buyer_maker=False means the buyer was aggressive (market buy)
    df["is_buy"] = ~df["is_buyer_maker"]

    # Resample to 1-minute bars
    df = df.set_index("timestamp")

    bars = df.resample("1min").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        dollar_volume=("dollar", "sum"),
        trade_count=("agg_trade_id", "count"),
    )

    # Buy/sell volume split
    buys = df[df["is_buy"]].resample("1min").agg(
        buy_vol=("quantity", "sum"),
        buy_dollar=("dollar", "sum"),
    )
    sells = df[~df["is_buy"]].resample("1min").agg(
        sell_vol=("quantity", "sum"),
        sell_dollar=("dollar", "sum"),
    )

    bars = bars.join(buys).join(sells)
    bars = bars.fillna({"buy_vol": 0, "sell_vol": 0, "buy_dollar": 0, "sell_dollar": 0})

    # Derived fields
    bars["delta"] = bars["buy_vol"] - bars["sell_vol"]
    bars["delta_dollar"] = bars["buy_dollar"] - bars["sell_dollar"]
    bars["buy_pct"] = bars["buy_vol"] / bars["volume"].clip(lower=1e-10)

    # Drop empty bars
    bars = bars.dropna(subset=["open"])

    return bars


def download_and_process_symbol(symbol: str, start_date: str, end_date: str, max_workers: int = 8):
    """Download all days for a symbol and build the 1-minute orderflow CSV."""
    # New canonical name: {sym}_1m.csv (in BARS_1M_V1)
    out_file = OUT_DIR / f"{symbol.lower()}_1m.csv"

    # Check if already exists
    if out_file.exists():
        existing = pd.read_csv(out_file, nrows=1)
        print(f"  ⚠️  {out_file.name} already exists ({os.path.getsize(out_file) / 1e6:.1f}MB). Skipping.")
        print(f"      Delete it first if you want to re-download.")
        return out_file

    print(f"\n{'='*60}")
    print(f"  {symbol} | {start_date} → {end_date}")
    print(f"{'='*60}")

    # Generate date list
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"  Downloading {len(dates)} days...")

    # Download in parallel
    downloaded = []
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_day, symbol, d): d for d in dates}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                downloaded.append((futures[future], result))
            else:
                d = futures[future]
                # Check if it's a 404 (data doesn't exist) vs error
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(dates)} ({len(downloaded)} downloaded, {failed} missing)")

    print(f"  Downloaded: {len(downloaded)}/{len(dates)} days ({failed} missing)")

    if len(downloaded) == 0:
        print(f"  ❌ No data found for {symbol}")
        return None

    # Sort by date
    downloaded.sort(key=lambda x: x[0])

    # Process each day
    print(f"  Processing to 1-minute bars...")
    all_bars = []
    for i, (date_str, zip_path) in enumerate(downloaded):
        bars = process_zip_to_1m(zip_path)
        if bars is not None and len(bars) > 0:
            all_bars.append(bars)

        if (i + 1) % 100 == 0:
            print(f"    Processed: {i+1}/{len(downloaded)}")

    if len(all_bars) == 0:
        print(f"  ❌ No valid data for {symbol}")
        return None

    # Concatenate and save
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep="first")]  # Dedupe

    # Match exact column order of existing files
    cols = ["open", "high", "low", "close", "volume", "dollar_volume",
            "buy_vol", "sell_vol", "buy_dollar", "sell_dollar",
            "trade_count", "delta", "delta_dollar", "buy_pct"]
    result = result[cols]

    result.to_csv(out_file, index=True, index_label="timestamp")

    size_mb = os.path.getsize(out_file) / 1e6
    print(f"  ✅ {symbol}: {len(result):,} bars | {result.index[0]} → {result.index[-1]} | {size_mb:.1f}MB")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download Binance aggTrades → 1m orderflow CSVs")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to download")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=8, help="Download concurrency")
    args = parser.parse_args()

    print("=" * 60)
    print("BINANCE AGGTRADES → ORDERFLOW CSV PIPELINE")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Range: {args.start} → {args.end}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    results = {}
    for symbol in args.symbols:
        out = download_and_process_symbol(symbol, args.start, args.end, args.workers)
        results[symbol] = str(out) if out else "FAILED"

    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for sym, path in results.items():
        status = "✅" if path != "FAILED" else "❌"
        print(f"  {status} {sym}: {path}")

    # List all available 1m bar files
    print(f"\nAll 1m bar CSVs in {OUT_DIR}:")
    for f in sorted(OUT_DIR.glob("*_1m.csv")):
        size = f.stat().st_size / 1e6
        print(f"  {f.name} ({size:.1f}MB)")


if __name__ == "__main__":
    main()
