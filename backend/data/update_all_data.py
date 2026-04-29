#!/usr/bin/env python3
"""
Update all local data to current date.
- OHLCV: Binance USDT perps via CCXT (4h, 1d for 13 research assets + extras)
- Derivatives: Coinalyze (OI, funding, LSR, liquidations)
"""
import ccxt
import pandas as pd
import numpy as np
import os
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
OHLCV_DIR = BASE / "data" / "ohlcv"
DERIV_DIR = BASE / "data" / "derivatives"

# Research assets
ASSETS = ["btc","eth","sol","link","avax","sui","inj","arb","op","render","tia","tao","fet",
          "cgpt","cocos","sent"]
TIMEFRAMES = ["4h", "1d", "1h", "15m"]
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# Coinalyze
COINALYZE_KEY = os.environ.get("COINALYZE_API_KEY", "")
DERIV_ASSETS = ["btc","eth","sol","link","avax","sui","inj","arb","op","render","tia","tao","fet"]
DERIV_TYPES = ["open_interest", "funding_rate", "long_short_ratio", "liquidations"]


def update_ohlcv():
    """Incrementally update OHLCV CSVs from Binance."""
    print("=" * 60)
    print("OHLCV UPDATE (Binance USDT Perps)")
    print("=" * 60)
    
    exchange = ccxt.binance()
    exchange.options['defaultType'] = 'future'
    
    updated = 0
    for asset in ASSETS:
        symbol = f"{asset.upper()}/USDT"
        # Handle naming
        if asset == "render":
            symbol = "RENDER/USDT"
        
        for tf in TIMEFRAMES:
            fname = f"binance_{asset}_usdt_{tf}.csv"
            fpath = OHLCV_DIR / fname
            
            # Determine start time
            if fpath.exists():
                df_existing = pd.read_csv(fpath)
                if len(df_existing) == 0:
                    continue
                last_ts = pd.Timestamp(df_existing["timestamp"].iloc[-1])
                since_ms = int(last_ts.timestamp() * 1000) + TF_MS[tf]
            else:
                # Start from 2021-01-01
                since_ms = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                df_existing = None
            
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            if now_ms - since_ms < TF_MS[tf]:
                continue  # Already up to date
            
            try:
                all_candles = []
                fetch_since = since_ms
                while fetch_since < now_ms:
                    candles = exchange.fetch_ohlcv(symbol, tf, since=fetch_since, limit=1000)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    fetch_since = candles[-1][0] + TF_MS[tf]
                    time.sleep(0.1)
                
                if not all_candles:
                    continue
                
                new_df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
                
                if df_existing is not None:
                    df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
                    combined = pd.concat([df_existing, new_df]).drop_duplicates(subset="timestamp").sort_values("timestamp")
                else:
                    combined = new_df.sort_values("timestamp")
                
                combined.to_csv(fpath, index=False)
                new_count = len(combined) - (len(df_existing) if df_existing is not None else 0)
                if new_count > 0:
                    print(f"  {asset.upper()} {tf}: +{new_count} candles → {len(combined)} total (to {combined['timestamp'].iloc[-1]})")
                    updated += 1
                    
            except Exception as e:
                if "does not have market symbol" not in str(e):
                    print(f"  ⚠ {asset.upper()} {tf}: {e}")
    
    print(f"\nOHLCV: {updated} files updated")
    return updated


def update_derivatives():
    """Update derivatives data from Coinalyze."""
    print("\n" + "=" * 60)
    print("DERIVATIVES UPDATE (Coinalyze)")
    print("=" * 60)
    
    if not COINALYZE_KEY:
        # Try to read from env file
        env_path = Path.home() / ".coinalyze_key"
        if env_path.exists():
            COINALYZE_KEY_local = env_path.read_text().strip()
        else:
            print("  ⚠ No COINALYZE_API_KEY set. Skipping derivatives.")
            print("    Set via: echo 'YOUR_KEY' > ~/.coinalyze_key")
            return 0
    else:
        COINALYZE_KEY_local = COINALYZE_KEY
    
    BASE_URL = "https://api.coinalyze.net/v1"
    headers = {"api_key": COINALYZE_KEY_local}
    
    # Symbol mapping for Coinalyze (Binance futures)
    def coinalyze_symbol(asset):
        return f"{asset.upper()}USDT_PERP.A"
    
    ENDPOINT_MAP = {
        "open_interest": "open-interest-history",
        "funding_rate": "funding-rate-history", 
        "long_short_ratio": "long-short-ratio-history",
        "liquidations": "liquidation-history",
    }
    
    updated = 0
    for asset in DERIV_ASSETS:
        sym = coinalyze_symbol(asset)
        for dtype in DERIV_TYPES:
            fname = f"{asset}_{dtype.replace('_rate', '')}.csv"
            # Try common naming patterns
            for fn in [fname, f"{asset}_{dtype}.csv"]:
                fpath = DERIV_DIR / fn
                if fpath.exists():
                    break
            
            endpoint = ENDPOINT_MAP.get(dtype)
            if not endpoint:
                continue
            
            # Get last date
            if fpath.exists():
                try:
                    df_existing = pd.read_csv(fpath)
                    last_col = df_existing.columns[0]
                    last_date = pd.Timestamp(df_existing[last_col].iloc[-1])
                    from_ts = int(last_date.timestamp()) + 86400
                except:
                    from_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
                    df_existing = None
            else:
                from_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
                df_existing = None
            
            to_ts = int(datetime.now(timezone.utc).timestamp())
            if to_ts - from_ts < 86400:
                continue
            
            try:
                r = requests.get(f"{BASE_URL}/{endpoint}", 
                               params={"symbols": sym, "interval": "daily", 
                                       "from": from_ts, "to": to_ts},
                               headers=headers)
                if r.status_code != 200:
                    print(f"  ⚠ {asset.upper()} {dtype}: HTTP {r.status_code}")
                    continue
                
                data = r.json()
                if not data or not isinstance(data, list) or len(data) == 0:
                    continue
                
                records = data[0].get("history", [])
                if not records:
                    continue
                
                new_df = pd.DataFrame(records)
                if "t" in new_df.columns:
                    new_df["timestamp"] = pd.to_datetime(new_df["t"], unit="s")
                    new_df = new_df.drop(columns=["t"])
                
                if df_existing is not None:
                    combined = pd.concat([df_existing, new_df]).drop_duplicates().sort_values(
                        df_existing.columns[0]).reset_index(drop=True)
                else:
                    combined = new_df
                
                combined.to_csv(fpath, index=False)
                print(f"  {asset.upper()} {dtype}: +{len(records)} rows")
                updated += 1
                sleep(1.6)  # 40 calls/min rate limit
                
            except Exception as e:
                print(f"  ⚠ {asset.upper()} {dtype}: {e}")
    
    print(f"\nDerivatives: {updated} files updated")
    return updated


if __name__ == "__main__":
    print(f"Data update started: {datetime.now()}")
    print(f"OHLCV dir: {OHLCV_DIR}")
    print(f"Derivatives dir: {DERIV_DIR}")
    
    n1 = update_ohlcv()
    n2 = update_derivatives()
    
    print(f"\n{'='*60}")
    print(f"DONE: {n1} OHLCV + {n2} derivatives files updated")
    print(f"Orderflow V2: check separate download_orderflow_v2.py processes")
