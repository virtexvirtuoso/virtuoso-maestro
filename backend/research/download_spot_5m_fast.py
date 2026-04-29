#!/usr/bin/env python3
"""Fast parallel 5m spot candle downloader using threading."""

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/spot/5m"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "ADA": "ADAUSDT", "APT": "APTUSDT", "ARB": "ARBUSDT", "ATOM": "ATOMUSDT",
    "AVAX": "AVAXUSDT", "BNB": "BNBUSDT", "BTC": "BTCUSDT", "CRV": "CRVUSDT",
    "DOGE": "DOGEUSDT", "DOT": "DOTUSDT", "ETH": "ETHUSDT", "FET": "FETUSDT",
    "FIL": "FILUSDT", "FTM": "FTMUSDT", "INJ": "INJUSDT", "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT", "OP": "OPUSDT", "RENDER": "RENDERUSDT", "SOL": "SOLUSDT",
    "SUI": "SUIUSDT", "TAO": "TAOUSDT", "TIA": "TIAUSDT", "UNI": "UNIUSDT",
    "XRP": "XRPUSDT",
}

START = "2024-01-01"
INTERVAL_MS = 5 * 60 * 1000  # 5 min in ms
BATCH_SIZE = 1000  # max klines per request
BATCH_MS = BATCH_SIZE * INTERVAL_MS  # time covered per request


def fetch_batch(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch one batch of klines."""
    for attempt in range(3):
        try:
            r = requests.get(BASE_URL, params={
                "symbol": symbol, "interval": "5m",
                "startTime": start_ms, "endTime": end_ms, "limit": BATCH_SIZE
            }, timeout=15)
            if r.status_code == 429:
                time.sleep(30)
                continue
            if r.status_code != 200:
                return []
            return r.json()
        except Exception:
            time.sleep(2 ** attempt)
    return []


def download_asset(asset: str, symbol: str) -> dict:
    """Download all 5m candles for one asset."""
    out_file = OUTPUT_DIR / f"{asset}_spot_5m.csv"
    
    # Skip if already done
    if out_file.exists():
        df = pd.read_csv(out_file)
        last = pd.to_datetime(df.iloc[-1, 0])
        if (datetime.now() - last).days < 2:
            return {"asset": asset, "status": "cached", "rows": len(df)}

    start_ms = int(datetime.strptime(START, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.now().timestamp() * 1000)
    
    # Build list of batch start times
    batches = []
    t = start_ms
    while t < end_ms:
        batches.append(t)
        t += BATCH_MS
    
    # Download batches in parallel (5 threads per asset)
    all_klines = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_batch, symbol, b, min(b + BATCH_MS - 1, end_ms)): b for b in batches}
        for future in as_completed(futures):
            data = future.result()
            if data:
                all_klines.extend(data)
    
    if not all_klines:
        return {"asset": asset, "status": "failed", "rows": 0}
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "Date", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = df["trades"].astype(int)
    df = df.drop(columns=["close_time", "ignore"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    df.to_csv(out_file, index=False)
    return {"asset": asset, "status": "ok", "rows": len(df)}


def main():
    print(f"📥 Downloading 5m spot candles for {len(ASSETS)} assets")
    print(f"   From: {START} → now")
    print(f"   Output: {OUTPUT_DIR}\n")
    
    for i, (asset, symbol) in enumerate(sorted(ASSETS.items()), 1):
        t0 = time.time()
        result = download_asset(asset, symbol)
        elapsed = time.time() - t0
        status = result["status"]
        rows = result["rows"]
        icon = "✅" if status == "ok" else "📦" if status == "cached" else "❌"
        print(f"  [{i:2d}/{len(ASSETS)}] {asset:6s} {icon} {rows:,} rows ({elapsed:.1f}s) [{status}]")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
