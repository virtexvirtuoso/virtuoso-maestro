"""
Download Binance aggTrades → aggregate into 1m orderflow bars → delete raw.

Output: 1m bars with OHLCV + buy_vol + sell_vol + trade_count + cvd
Process one day at a time to minimize disk usage (~50MB raw per day compressed).

Usage:
  python download_orderflow.py --asset BTCUSDT --start 2024-01-01 --end 2026-02-24
  python download_orderflow.py --asset ETHUSDT --start 2024-01-01 --end 2026-02-24
"""
import os
import sys
import gzip
import shutil
import argparse
import urllib.request
import csv
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from backend.config.data_paths import BARS_1M_V1, raw_aggtrades_dir

DATA_DIR = BARS_1M_V1

def download_day(symbol: str, date: datetime) -> Path | None:
    """Download one day of aggTrades. Returns path to zip or None."""
    date_str = date.strftime("%Y-%m-%d")
    fname = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{fname}"

    sym_dir = raw_aggtrades_dir(symbol)
    sym_dir.mkdir(parents=True, exist_ok=True)
    out_path = sym_dir / fname

    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path
    except Exception as e:
        # Try .csv.gz format
        fname_gz = f"{symbol}-aggTrades-{date_str}.csv.gz"
        url_gz = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{fname_gz}"
        try:
            out_gz = sym_dir / fname_gz
            urllib.request.urlretrieve(url_gz, out_gz)
            return out_gz
        except:
            print(f"  ⚠ No data for {date_str}: {e}")
            return None

def extract_and_aggregate(zip_path: Path, symbol: str) -> pd.DataFrame | None:
    """Extract aggTrades and aggregate into 1m orderflow bars."""
    import zipfile
    
    try:
        col_names = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "is_buyer_maker"]
        if str(zip_path).endswith('.zip'):
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Peek first line to check for header
                    first_line = f.readline().decode().strip()
                    f.seek(0)
                    has_header = not first_line[0].isdigit()
                    df = pd.read_csv(f, header=0 if has_header else None,
                                     names=None if has_header else col_names,
                                     dtype={"price": float, "quantity": float} if has_header else {"price": float, "qty": float})
                    if has_header:
                        df.columns = col_names
        elif str(zip_path).endswith('.gz'):
            with gzip.open(zip_path, 'rt') as f:
                first_line = f.readline().strip()
                f.seek(0)
                has_header = not first_line[0].isdigit()
                df = pd.read_csv(f, header=0 if has_header else None,
                                 names=None if has_header else col_names)
                if has_header:
                    df.columns = col_names
        else:
            return None
    except Exception as e:
        print(f"  ⚠ Failed to read {zip_path.name}: {e}")
        return None
    
    if len(df) == 0:
        return None
    
    # Fix is_buyer_maker (may be string "true"/"false")
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].str.strip().str.lower() == "true"
    
    # Determine timestamp unit (ms vs us)
    ts_sample = df["timestamp"].iloc[0]
    if ts_sample > 1e15:  # microseconds
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
    else:  # milliseconds
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    df["dollar_vol"] = df["price"] * df["qty"]
    
    # Buyer is taker (aggressor) when is_buyer_maker = False
    # Seller is taker (aggressor) when is_buyer_maker = True
    df["buy_vol"] = df["qty"] * (~df["is_buyer_maker"])  # taker buy
    df["sell_vol"] = df["qty"] * df["is_buyer_maker"]     # taker sell
    df["buy_dollar"] = df["dollar_vol"] * (~df["is_buyer_maker"])
    df["sell_dollar"] = df["dollar_vol"] * df["is_buyer_maker"]
    
    # Aggregate to 1-minute bars
    df.set_index("timestamp", inplace=True)
    
    bars = df.resample("1min").agg({
        "price": ["first", "max", "min", "last"],
        "qty": "sum",
        "dollar_vol": "sum",
        "buy_vol": "sum",
        "sell_vol": "sum",
        "buy_dollar": "sum",
        "sell_dollar": "sum",
        "agg_id": "count",  # trade count
    })
    
    bars.columns = ["open", "high", "low", "close", "volume", "dollar_volume",
                     "buy_vol", "sell_vol", "buy_dollar", "sell_dollar", "trade_count"]
    
    # Drop empty bars (no trades)
    bars = bars.dropna(subset=["open"])
    
    # Add derived features
    bars["delta"] = bars["buy_vol"] - bars["sell_vol"]  # volume delta
    bars["delta_dollar"] = bars["buy_dollar"] - bars["sell_dollar"]
    bars["buy_pct"] = bars["buy_vol"] / bars["volume"].clip(lower=1e-10)
    
    return bars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTCUSDT", help="Symbol (e.g., BTCUSDT)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-02-24", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    out_file = DATA_DIR / f"{args.asset.lower()}_1m.csv"
    
    # Check existing data to resume
    existing_dates = set()
    all_bars = []
    if out_file.exists():
        existing = pd.read_csv(out_file, parse_dates=["timestamp"], index_col="timestamp")
        existing_dates = set(existing.index.date)
        all_bars.append(existing)
        print(f"Resuming: {len(existing_dates)} days already downloaded")
    
    total_days = (end - start).days + 1
    downloaded = 0
    errors = 0
    
    current = start
    while current <= end:
        if current.date() in existing_dates:
            current += timedelta(days=1)
            continue
        
        day_num = (current - start).days + 1
        print(f"[{day_num}/{total_days}] {current.strftime('%Y-%m-%d')} ", end="", flush=True)
        
        # Download
        zip_path = download_day(args.asset, current)
        if zip_path is None:
            errors += 1
            current += timedelta(days=1)
            continue
        
        fsize = zip_path.stat().st_size / 1e6
        print(f"({fsize:.1f} MB) ", end="", flush=True)
        
        # Aggregate
        bars = extract_and_aggregate(zip_path, args.asset)
        
        # Delete raw
        zip_path.unlink()
        
        if bars is not None and len(bars) > 0:
            all_bars.append(bars)
            downloaded += 1
            print(f"→ {len(bars)} bars, Δ={bars['delta'].sum():.0f}")
        else:
            print("→ empty")
            errors += 1
        
        # Save checkpoint every 30 days
        if downloaded > 0 and downloaded % 30 == 0:
            combined = pd.concat(all_bars).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.to_csv(out_file, index_label="timestamp")
            print(f"  💾 Checkpoint: {len(combined)} bars saved")
        
        current += timedelta(days=1)
    
    # Final save
    if all_bars:
        combined = pd.concat(all_bars).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.to_csv(out_file, index_label="timestamp")
        
        days_covered = len(set(combined.index.date))
        print(f"\n{'='*50}")
        print(f"✅ {args.asset}: {len(combined):,} 1m bars across {days_covered} days")
        print(f"   Downloaded: {downloaded}, Errors: {errors}")
        print(f"   File: {out_file} ({out_file.stat().st_size/1e6:.1f} MB)")
        print(f"   Range: {combined.index[0]} → {combined.index[-1]}")
    else:
        print("No data downloaded")

if __name__ == "__main__":
    main()
