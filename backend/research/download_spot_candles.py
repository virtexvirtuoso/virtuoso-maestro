"""
Download historical spot candles from Binance for multiple timeframes.
Supports 1h, 4h, 1d. Saves to ~/Desktop/maestro/data/spot/{tf}/

Usage:
    python download_spot_candles.py                  # All assets, 4h + 1h
    python download_spot_candles.py --assets BTC ETH  # Specific assets
    python download_spot_candles.py --timeframes 4h   # Specific TF only
    python download_spot_candles.py --start 2022-01-01 # Custom start date
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Binance klines API (no key needed for spot)
BASE_URL = "https://data-api.binance.vision/api/v3/klines"

# All 21 assets from our spot daily set + 3 from derivatives
ASSETS = [
    "ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT",
    "ETH", "FET", "FIL", "FTM", "INJ", "LINK", "NEAR", "OP",
    "RENDER", "SOL", "SUI", "TAO", "TIA", "UNI", "XRP"
]

# Binance symbol mapping (most are just +USDT)
SYMBOL_MAP = {
    "RENDER": "RENDERUSDT",
    "FTM": "FTMUSDT",  # Note: may have migrated to S token
}

TIMEFRAME_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 3600 * 1000,
    "4h": 4 * 3600 * 1000,
    "1d": 24 * 3600 * 1000,
}

DATA_DIR = Path.home() / "Desktop" / "maestro" / "data" / "spot"


def get_symbol(asset: str) -> str:
    return SYMBOL_MAP.get(asset, f"{asset}USDT")


def download_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Download klines from Binance with pagination (1000 candles per request)."""
    all_klines = []
    current_start = start_ms
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        
        for attempt in range(5):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 400:
                    # Symbol might not exist
                    error = resp.json().get("msg", "")
                    if "Invalid symbol" in error:
                        return None
                    print(f"  400 error: {error}")
                    return None
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  Failed after 5 attempts: {e}")
                return all_klines
        
        data = resp.json()
        if not data:
            break
            
        all_klines.extend(data)
        
        # Move start to after last candle
        last_ts = data[-1][0]
        if last_ts == current_start:
            break  # No progress
        current_start = last_ts + 1
        
        # Rate limiting: ~1200 weight/min, klines = 1-2 weight
        time.sleep(0.25)
    
    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """Convert Binance klines to clean DataFrame."""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    # Convert types
    for col in ["Open", "High", "Low", "Close", "Volume", "quote_volume", 
                 "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["trades"] = df["trades"].astype(int)
    
    # Timestamp to datetime
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    
    # Keep useful columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", 
             "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    return df


def check_data_quality(df: pd.DataFrame, asset: str, tf: str) -> dict:
    """Basic data quality checks."""
    if df.empty:
        return {"status": "EMPTY"}
    
    n = len(df)
    first = df["Date"].iloc[0]
    last = df["Date"].iloc[-1]
    
    # Check for gaps
    delta_map = {"5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
                  "1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}
    expected_delta = delta_map.get(tf, timedelta(days=1))
    
    diffs = df["Date"].diff().dropna()
    gaps = diffs[diffs > expected_delta * 1.5]
    
    # Check for zero/null prices
    zero_close = (df["Close"] <= 0).sum()
    null_close = df["Close"].isna().sum()
    zero_volume = (df["Volume"] <= 0).sum()
    
    return {
        "status": "OK" if zero_close == 0 and null_close == 0 else "ISSUES",
        "rows": n,
        "first": str(first),
        "last": str(last),
        "gaps": len(gaps),
        "max_gap": str(gaps.max()) if len(gaps) > 0 else "none",
        "zero_close": zero_close,
        "null_close": null_close,
        "zero_volume_bars": zero_volume,
    }


def main():
    parser = argparse.ArgumentParser(description="Download Binance spot candles")
    parser.add_argument("--assets", nargs="+", default=ASSETS, help="Assets to download")
    parser.add_argument("--timeframes", nargs="+", default=["4h", "1h"], 
                        help="Timeframes (1h, 4h, 1d)")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (default: now)")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be downloaded")
    args = parser.parse_args()
    
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.utcnow()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    print(f"=== Binance Spot Candle Downloader ===")
    print(f"Assets: {len(args.assets)} ({', '.join(args.assets[:5])}{'...' if len(args.assets) > 5 else ''})")
    print(f"Timeframes: {args.timeframes}")
    print(f"Range: {args.start} → {args.end or 'now'}")
    print(f"Output: {DATA_DIR}/{{tf}}/")
    print()
    
    if args.dry_run:
        total = len(args.assets) * len(args.timeframes)
        print(f"Would download {total} files. Exiting (dry-run).")
        return
    
    results = []
    
    for tf in args.timeframes:
        tf_dir = DATA_DIR / tf
        tf_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Timeframe: {tf}")
        print(f"{'='*60}")
        
        for i, asset in enumerate(args.assets):
            symbol = get_symbol(asset)
            outfile = tf_dir / f"{asset}_spot_{tf}.csv"
            
            # Skip if already downloaded and recent
            if outfile.exists():
                existing = pd.read_csv(outfile)
                if len(existing) > 0:
                    last_date = pd.to_datetime(existing["Date"]).max()
                    if (datetime.utcnow() - last_date).days < 2:
                        print(f"  [{i+1}/{len(args.assets)}] {asset} — already up to date ({len(existing)} rows)")
                        results.append({"asset": asset, "tf": tf, "status": "CACHED", "rows": len(existing)})
                        continue
            
            print(f"  [{i+1}/{len(args.assets)}] {asset} ({symbol})...", end=" ", flush=True)
            
            klines = download_klines(symbol, tf, start_ms, end_ms)
            
            if klines is None:
                print(f"FAILED (invalid symbol?)")
                results.append({"asset": asset, "tf": tf, "status": "FAILED", "rows": 0})
                continue
            
            df = klines_to_dataframe(klines)
            
            if df.empty:
                print(f"EMPTY")
                results.append({"asset": asset, "tf": tf, "status": "EMPTY", "rows": 0})
                continue
            
            # Quality check
            quality = check_data_quality(df, asset, tf)
            
            # Save
            df.to_csv(outfile, index=False)
            
            print(f"{quality['rows']} rows, {quality['first'][:10]} → {quality['last'][:10]}"
                  f" | gaps:{quality['gaps']} zero_vol:{quality['zero_volume_bars']}"
                  f" {'⚠️' if quality['status'] != 'OK' else '✅'}")
            
            results.append({"asset": asset, "tf": tf, **quality})
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    ok = sum(1 for r in results if r.get("status") in ("OK", "CACHED"))
    failed = sum(1 for r in results if r.get("status") == "FAILED")
    issues = sum(1 for r in results if r.get("status") == "ISSUES")
    
    print(f"Total: {len(results)} | OK: {ok} | Issues: {issues} | Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed assets:")
        for r in results:
            if r.get("status") == "FAILED":
                print(f"  - {r['asset']} ({r['tf']})")
    
    # Save results manifest
    manifest = pd.DataFrame(results)
    manifest_path = DATA_DIR / "download_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
