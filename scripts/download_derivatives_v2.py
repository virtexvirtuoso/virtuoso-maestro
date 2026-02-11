#!/usr/bin/env python3
"""
Download Derivatives Data v2 - Direct Binance Futures API

Uses direct HTTP requests for OI, LSR, and other derivatives data.
"""
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time

OUTPUT_DIR = os.path.expanduser('~/derivatives_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = 'https://fapi.binance.com'

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT',
           'TIAUSDT', 'INJUSDT', 'LINKUSDT', 'AVAXUSDT', 'FETUSDT', 'TAOUSDT', 'RENDERUSDT']


def get_open_interest_hist(symbol, period='1h', limit=500):
    """Get Open Interest history."""
    url = f'{BASE_URL}/futures/data/openInterestHist'
    params = {'symbol': symbol, 'period': period, 'limit': limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'])
                df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'])
                return df
    except Exception as e:
        print(f"    OI Error: {e}")
    return None


def get_long_short_ratio(symbol, period='1h', limit=500):
    """Get Top Traders Long/Short Ratio (Accounts)."""
    url = f'{BASE_URL}/futures/data/topLongShortAccountRatio'
    params = {'symbol': symbol, 'period': period, 'limit': limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
    except Exception as e:
        print(f"    LSR Error: {e}")
    return None


def get_long_short_position_ratio(symbol, period='1h', limit=500):
    """Get Top Traders Long/Short Ratio (Positions)."""
    url = f'{BASE_URL}/futures/data/topLongShortPositionRatio'
    params = {'symbol': symbol, 'period': period, 'limit': limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
    except Exception as e:
        print(f"    Position Ratio Error: {e}")
    return None


def get_global_long_short_ratio(symbol, period='1h', limit=500):
    """Get Global Long/Short Ratio (all accounts)."""
    url = f'{BASE_URL}/futures/data/globalLongShortAccountRatio'
    params = {'symbol': symbol, 'period': period, 'limit': limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
    except Exception as e:
        print(f"    Global LSR Error: {e}")
    return None


def get_taker_buy_sell_ratio(symbol, period='1h', limit=500):
    """Get Taker Buy/Sell Volume Ratio."""
    url = f'{BASE_URL}/futures/data/takerlongshortRatio'
    params = {'symbol': symbol, 'period': period, 'limit': limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
    except Exception as e:
        print(f"    Taker Ratio Error: {e}")
    return None


def main():
    print("=" * 60)
    print("DERIVATIVES DATA DOWNLOADER V2")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Symbols: {len(SYMBOLS)}")
    print()

    for symbol in SYMBOLS:
        base = symbol.replace('USDT', '').lower()
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print('='*60)

        # Open Interest (multiple timeframes)
        for period in ['1h', '4h', '1d']:
            print(f"  OI ({period})...", end=' ')
            df = get_open_interest_hist(symbol, period)
            if df is not None:
                path = f"{OUTPUT_DIR}/{base}_oi_{period}.csv"
                df.to_csv(path, index=False)
                print(f"{len(df)} rows")
            else:
                print("no data")
            time.sleep(0.2)

        # Long/Short Ratio (Top Traders by Account)
        print(f"  LSR (accounts)...", end=' ')
        df = get_long_short_ratio(symbol, '1h')
        if df is not None:
            path = f"{OUTPUT_DIR}/{base}_lsr_account.csv"
            df.to_csv(path, index=False)
            print(f"{len(df)} rows")
        else:
            print("no data")

        # Long/Short Ratio (Top Traders by Position)
        print(f"  LSR (positions)...", end=' ')
        df = get_long_short_position_ratio(symbol, '1h')
        if df is not None:
            path = f"{OUTPUT_DIR}/{base}_lsr_position.csv"
            df.to_csv(path, index=False)
            print(f"{len(df)} rows")
        else:
            print("no data")

        # Global Long/Short Ratio
        print(f"  Global LSR...", end=' ')
        df = get_global_long_short_ratio(symbol, '1h')
        if df is not None:
            path = f"{OUTPUT_DIR}/{base}_lsr_global.csv"
            df.to_csv(path, index=False)
            print(f"{len(df)} rows")
        else:
            print("no data")

        # Taker Buy/Sell Ratio
        print(f"  Taker ratio...", end=' ')
        df = get_taker_buy_sell_ratio(symbol, '1h')
        if df is not None:
            path = f"{OUTPUT_DIR}/{base}_taker.csv"
            df.to_csv(path, index=False)
            print(f"{len(df)} rows")
        else:
            print("no data")

        time.sleep(1)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    # Summary
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])
    print(f"\nTotal files: {len(files)}")
    for f in files:
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024
        print(f"  {f}: {size:.1f} KB")


if __name__ == '__main__':
    main()
