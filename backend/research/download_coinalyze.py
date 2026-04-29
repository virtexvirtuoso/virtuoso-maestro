"""
Download historical derivatives data from Coinalyze API.
Endpoints: open-interest, funding-rate, liquidation, long-short-ratio, taker buy/sell
Intervals: 1hour, 4hour, daily

Usage:
    python3 download_coinalyze.py                          # All assets, 1h+4h+daily
    python3 download_coinalyze.py --assets BTC ETH SOL     # Specific assets
    python3 download_coinalyze.py --intervals 1hour 4hour  # Specific intervals
"""

import os
import time
import json
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_KEY = os.environ.get("COINALYZE_API_KEY", "***COINGLASS_KEY_REDACTED***")
BASE_URL = "https://api.coinalyze.net/v1"

# Match our spot data assets + derivatives-only assets
ASSETS = [
    "ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT",
    "ETH", "FET", "FIL", "FTM", "INJ", "LINK", "NEAR", "OP",
    "RENDER", "SOL", "SUI", "TAO", "TIA", "UNI", "XRP"
]

# Coinalyze symbol format: {ASSET}USDT.6 for Binance Futures
# .6 = Binance Futures (USDT-margined)
SYMBOL_SUFFIX = ".6"

ENDPOINTS = {
    "oi": {
        "path": "open-interest-history",
        "columns": ["t", "o", "h", "l", "c"],
        "rename": {"t": "timestamp", "o": "oi_open", "h": "oi_high", "l": "oi_low", "c": "oi_close"},
    },
    "funding": {
        "path": "funding-rate-history",
        "columns": ["t", "o", "h", "l", "c"],
        "rename": {"t": "timestamp", "o": "fr_open", "h": "fr_high", "l": "fr_low", "c": "fr_close"},
    },
    "liquidations": {
        "path": "liquidation-history",
        "columns": ["t", "l", "s"],
        "rename": {"t": "timestamp", "l": "liq_long", "s": "liq_short"},
    },
    "lsr": {
        "path": "long-short-ratio-history",
        "columns": ["t", "r", "l", "s"],
        "rename": {"t": "timestamp", "r": "lsr_ratio", "l": "lsr_long_pct", "s": "lsr_short_pct"},
    },
}

# Max data per request: Coinalyze limits vary, use pagination
MAX_POINTS = 5000  # safe limit per request

DATA_DIR = Path.home() / "Desktop" / "maestro" / "data" / "derivatives"

INTERVAL_SECONDS = {
    "1hour": 3600,
    "4hour": 4 * 3600,
    "daily": 86400,
}


def fetch_data(endpoint_key: str, symbol: str, interval: str, from_ts: int, to_ts: int) -> list:
    """Fetch data from Coinalyze with pagination."""
    ep = ENDPOINTS[endpoint_key]
    all_data = []
    current_from = from_ts
    
    while current_from < to_ts:
        params = {
            "api_key": API_KEY,
            "symbols": symbol,
            "interval": interval,
            "from": current_from,
            "to": to_ts,
        }
        
        for attempt in range(3):
            try:
                resp = requests.get(f"{BASE_URL}/{ep['path']}", params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 60
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 400:
                    msg = resp.json().get("message", "")
                    if "not found" in msg.lower() or "invalid" in msg.lower():
                        return None  # Symbol doesn't exist
                    print(f"    400: {msg}")
                    return all_data
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"    Failed: {e}")
                return all_data
        
        result = resp.json()
        
        # Coinalyze returns list of objects, one per symbol
        if isinstance(result, list) and result:
            history = result[0].get("history", []) if isinstance(result[0], dict) else []
        elif isinstance(result, dict):
            history = result.get("history", [])
        else:
            break
        
        if not history:
            break
        
        all_data.extend(history)
        
        last_ts = history[-1]["t"]
        if last_ts <= current_from:
            break
        current_from = last_ts + INTERVAL_SECONDS.get(interval, 3600)
        
        # Rate limiting: ~10 req/min for free tier
        time.sleep(6)
    
    return all_data


def to_dataframe(data: list, endpoint_key: str) -> pd.DataFrame:
    """Convert API response to DataFrame."""
    if not data:
        return pd.DataFrame()
    
    ep = ENDPOINTS[endpoint_key]
    df = pd.DataFrame(data)
    
    # Only keep expected columns
    available = [c for c in ep["columns"] if c in df.columns]
    df = df[available]
    
    # Rename
    rename_map = {k: v for k, v in ep["rename"].items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # Convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Coinalyze derivatives data")
    parser.add_argument("--assets", nargs="+", default=ASSETS)
    parser.add_argument("--intervals", nargs="+", default=["1hour", "4hour", "daily"])
    parser.add_argument("--endpoints", nargs="+", default=["oi", "funding", "liquidations", "lsr"])
    parser.add_argument("--months-back", type=int, default=24, help="Months of history")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.months_back * 30)
    from_ts = int(start.timestamp())
    to_ts = int(now.timestamp())
    
    print(f"=== Coinalyze Derivatives Downloader ===")
    print(f"Assets: {len(args.assets)}")
    print(f"Intervals: {args.intervals}")
    print(f"Endpoints: {args.endpoints}")
    print(f"Range: {start.strftime('%Y-%m-%d')} → {now.strftime('%Y-%m-%d')}")
    print(f"Output: {DATA_DIR}/{{interval}}/")
    
    if args.dry_run:
        total = len(args.assets) * len(args.intervals) * len(args.endpoints)
        print(f"\nWould download {total} datasets. Exiting.")
        return
    
    results = []
    
    for interval in args.intervals:
        interval_dir = DATA_DIR / interval
        interval_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Interval: {interval}")
        print(f"{'='*60}")
        
        for i, asset in enumerate(args.assets):
            symbol = f"{asset}USDT{SYMBOL_SUFFIX}"
            print(f"\n  [{i+1}/{len(args.assets)}] {asset} ({symbol})")
            
            for ep_key in args.endpoints:
                outfile = interval_dir / f"{asset}_{ep_key}_{interval}.csv"
                
                # Skip if recent
                if outfile.exists():
                    existing = pd.read_csv(outfile)
                    if len(existing) > 100:
                        print(f"    {ep_key}: cached ({len(existing)} rows)")
                        results.append({"asset": asset, "interval": interval, 
                                       "endpoint": ep_key, "status": "CACHED", "rows": len(existing)})
                        continue
                
                print(f"    {ep_key}...", end=" ", flush=True)
                
                data = fetch_data(ep_key, symbol, interval, from_ts, to_ts)
                
                if data is None:
                    print("NOT FOUND")
                    results.append({"asset": asset, "interval": interval,
                                   "endpoint": ep_key, "status": "NOT_FOUND", "rows": 0})
                    continue
                
                df = to_dataframe(data, ep_key)
                
                if df.empty:
                    print("EMPTY")
                    results.append({"asset": asset, "interval": interval,
                                   "endpoint": ep_key, "status": "EMPTY", "rows": 0})
                    continue
                
                df.to_csv(outfile, index=False)
                
                first = df["timestamp"].iloc[0]
                last = df["timestamp"].iloc[-1]
                print(f"{len(df)} rows ({str(first)[:10]} → {str(last)[:10]})")
                
                results.append({"asset": asset, "interval": interval,
                               "endpoint": ep_key, "status": "OK", "rows": len(df)})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    ok = sum(1 for r in results if r["status"] in ("OK", "CACHED"))
    failed = sum(1 for r in results if r["status"] in ("NOT_FOUND", "EMPTY"))
    print(f"Total: {len(results)} | OK: {ok} | Failed: {failed}")
    
    # Save manifest
    manifest = pd.DataFrame(results)
    manifest.to_csv(DATA_DIR / "coinalyze_manifest.csv", index=False)
    print(f"Manifest: {DATA_DIR}/coinalyze_manifest.csv")


if __name__ == "__main__":
    main()
