#!/usr/bin/env python3
"""Download daily OHLCV for 30 tokens from Binance USDT perpetuals since 2021-01-01."""

import ccxt
import pandas as pd
import time
import os

TOKENS = [
    'btc','eth','sol','bnb','xrp','ada','doge','avax','link','dot',
    'matic','uni','atom','near','ftm','algo','sand','mana','gala','axs',
    'ape','ldo','arb','op','sui','inj','fet','fil','ren','luna'
]

OUT_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
os.makedirs(OUT_DIR, exist_ok=True)

SINCE = '2021-01-01T00:00:00Z'
# We want at least ~1500 rows (2021-01-01 to 2026-02-14 ≈ 1871 days)
MIN_ROWS = 1400

exchange = ccxt.binance()
exchange.options['defaultType'] = 'future'
since_ms = exchange.parse8601(SINCE)

failed = []
skipped = []
downloaded = []

for token in TOKENS:
    symbol = f"{token.upper()}/USDT"
    # Handle LUNA -> LUNA2 or 1000LUNA etc
    if token == 'luna':
        candidates = ['LUNA2/USDT', 'LUNC/USDT', 'LUNA/USDT']
    elif token == 'matic':
        candidates = ['MATIC/USDT', 'POL/USDT']
    else:
        candidates = [symbol]
    
    out_path = os.path.join(OUT_DIR, f'binance_{token}_usdt_1d.csv')
    
    # Check existing
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if len(existing) >= MIN_ROWS:
            print(f"SKIP {token}: already {len(existing)} rows")
            skipped.append(token)
            continue
    
    success = False
    for sym in candidates:
        try:
            print(f"Downloading {sym}...")
            all_candles = []
            current_since = since_ms
            
            while True:
                candles = exchange.fetch_ohlcv(sym, '1d', since=current_since, limit=1000)
                if not candles:
                    break
                all_candles.extend(candles)
                last_ts = candles[-1][0]
                if len(candles) < 1000:
                    break
                current_since = last_ts + 86400000  # next day
                time.sleep(0.1)
            
            if not all_candles:
                print(f"  No data for {sym}")
                continue
            
            df = pd.DataFrame(all_candles, columns=['timestamp_ms','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.strftime('%Y-%m-%d')
            df = df[['timestamp','open','high','low','close','volume']].drop_duplicates('timestamp')
            df.to_csv(out_path, index=False)
            print(f"  OK: {len(df)} rows, {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            downloaded.append(token)
            success = True
            break
        except Exception as e:
            print(f"  Failed {sym}: {e}")
            time.sleep(0.5)
    
    if not success:
        failed.append(token)
    time.sleep(0.2)

print(f"\n=== SUMMARY ===")
print(f"Downloaded: {downloaded}")
print(f"Skipped (enough data): {skipped}")
print(f"Failed: {failed}")
