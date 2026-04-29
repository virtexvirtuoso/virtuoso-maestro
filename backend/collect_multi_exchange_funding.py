"""
Collect multi-exchange funding rate data from Coinalyze API.
Exchange codes: .A=Binance, .4=OKX, .0=BitMEX, .3=Bybit, .F=Bitget (USDT)
"""
import os
import requests
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

API_KEY = os.environ.get('COINALYZE_API_KEY')
if not API_KEY:
    raise RuntimeError('COINALYZE_API_KEY environment variable is required')
HEADERS = {'api_key': API_KEY}
BASE = 'https://api.coinalyze.net/v1/funding-rate-history'
OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'derivatives' / 'multi_exchange'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exchange code mapping (best guesses from Coinalyze conventions)
EXCHANGE_CODES = {
    'binance': 'A',
    'okx': '4',
    'bitmex': '0',
    'bybit': '3',
    'bitget': 'F',  # only USDT pairs
}

# Tokens to collect - USD and USDT perps
TOKENS = {
    'BTC': {
        'usd': ['A', '4', '0', '3'],
        'usdt': ['A', '4', '0', '3', 'F'],
    },
    'ETH': {
        'usd': ['A', '4', '0', '3'],
        'usdt': ['A', '4', '0', '3', 'F'],
    },
    'SOL': {
        'usd': ['A', '4'],
        'usdt': ['A', '4', '3', 'F'],
    },
}

# Time range: ~2 years back
FROM_TS = int(datetime(2024, 1, 1).timestamp())
TO_TS = int(datetime(2026, 2, 14).timestamp())


def fetch_funding(symbol: str, from_ts: int, to_ts: int) -> list:
    """Fetch funding rate history for a symbol, paginating if needed."""
    all_data = []
    current_from = from_ts
    
    while current_from < to_ts:
        current_to = min(current_from + 365 * 86400, to_ts)  # max 1 year chunks
        r = requests.get(BASE, headers=HEADERS, params={
            'symbols': symbol,
            'interval': 'daily',
            'from': str(current_from),
            'to': str(current_to),
        })
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data and data[0].get('history'):
                all_data.extend(data[0]['history'])
        else:
            print(f"  Error {r.status_code} for {symbol}: {r.text[:200]}")
        
        current_from = current_to
        time.sleep(0.5)
    
    return all_data


def collect_token(token: str, pairs: dict):
    """Collect all exchange funding data for a token."""
    results = {}
    
    for pair_type, exchanges in pairs.items():
        suffix = 'USD_PERP' if pair_type == 'usd' else 'USDT_PERP'
        for ex_code in exchanges:
            symbol = f'{token}{suffix}.{ex_code}'
            ex_name = next((k for k, v in EXCHANGE_CODES.items() if v == ex_code), ex_code)
            print(f"  Fetching {symbol} ({ex_name})...", end=' ')
            
            history = fetch_funding(symbol, FROM_TS, TO_TS)
            if history:
                print(f"{len(history)} points")
                key = f"{ex_name}_{pair_type}"
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                df = df.rename(columns={'c': 'funding_close', 'o': 'funding_open', 'h': 'funding_high', 'l': 'funding_low'})
                df['exchange'] = ex_name
                df['pair_type'] = pair_type
                df = df.drop(columns=['t'])
                results[key] = df
            else:
                print("no data")
            time.sleep(0.3)
    
    if results:
        combined = pd.concat(results.values(), ignore_index=True)
        out_path = OUT_DIR / f'{token.lower()}_multi_exchange_funding.csv'
        combined.to_csv(out_path, index=False)
        print(f"  Saved {len(combined)} rows to {out_path}")
        return combined
    return None


def main():
    all_data = {}
    for token, pairs in TOKENS.items():
        print(f"\n=== {token} ===")
        df = collect_token(token, pairs)
        if df is not None:
            all_data[token] = df
            # Summary
            print(f"  Exchanges with data: {df['exchange'].unique().tolist()}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\n=== Collection Complete ===")
    for token, df in all_data.items():
        print(f"{token}: {len(df)} rows, {df['exchange'].nunique()} exchanges, {df['timestamp'].nunique()} dates")


if __name__ == '__main__':
    main()
