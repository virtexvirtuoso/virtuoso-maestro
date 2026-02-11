#!/usr/bin/env python3
"""
Download Derivatives Data from Binance Futures

Downloads:
- Funding Rate History
- Open Interest History
- Long/Short Ratio (Top Traders)

Note: Run from VPS for unrestricted Binance access.
"""
import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# Output directory
OUTPUT_DIR = os.path.expanduser('~/derivatives_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Symbols to download (perpetual futures)
SYMBOLS = [
    'BTC/USDT:USDT',  # BTC perpetual
    'ETH/USDT:USDT',
    'SOL/USDT:USDT',
    'ARB/USDT:USDT',
    'OP/USDT:USDT',
    'SUI/USDT:USDT',
    'TIA/USDT:USDT',
    'INJ/USDT:USDT',
    'LINK/USDT:USDT',
    'AVAX/USDT:USDT',
    'FET/USDT:USDT',
    'TAO/USDT:USDT',
    'RENDER/USDT:USDT',
]

def download_funding_rates(exchange, symbol, days=1000):
    """Download funding rate history."""
    print(f"  Funding rates for {symbol}...")

    all_data = []
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    try:
        while True:
            data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not data:
                break

            all_data.extend(data)
            since = data[-1]['timestamp'] + 1

            if len(data) < 1000:
                break

            time.sleep(0.1)  # Rate limit

        if all_data:
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[['timestamp', 'symbol', 'fundingRate']]

    except Exception as e:
        print(f"    Error: {e}")

    return None


def download_open_interest(exchange, symbol, timeframe='1h', days=1000):
    """Download open interest history."""
    print(f"  Open Interest for {symbol} ({timeframe})...")

    # Binance uses different symbol format for OI
    base_symbol = symbol.replace('/USDT:USDT', 'USDT')

    all_data = []
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    end = int(datetime.now().timestamp() * 1000)

    try:
        # Use fetch_open_interest_history if available
        if hasattr(exchange, 'fetch_open_interest_history'):
            while since < end:
                data = exchange.fetch_open_interest_history(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=500
                )
                if not data:
                    break

                all_data.extend(data)
                since = data[-1]['timestamp'] + 1

                if len(data) < 500:
                    break

                time.sleep(0.1)

        if all_data:
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

    except Exception as e:
        print(f"    Error: {e}")

        # Fallback: try direct API
        try:
            print(f"    Trying direct API...")
            params = {
                'symbol': base_symbol,
                'period': '1h',
                'limit': 500,
            }
            response = exchange.fapiPublicGetOpenInterestHist(params)
            if response:
                df = pd.DataFrame(response)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['openInterest'] = pd.to_numeric(df['sumOpenInterest'])
                return df[['timestamp', 'openInterest']]
        except Exception as e2:
            print(f"    Fallback error: {e2}")

    return None


def download_long_short_ratio(exchange, symbol, period='1h', days=30):
    """Download long/short ratio for top traders."""
    print(f"  Long/Short Ratio for {symbol}...")

    base_symbol = symbol.replace('/USDT:USDT', 'USDT')

    try:
        # Top traders long/short ratio
        params = {
            'symbol': base_symbol,
            'period': period,
            'limit': 500,
        }

        # Account ratio
        response = exchange.fapiPublicGetTopLongShortAccountRatio(params)

        if response:
            df = pd.DataFrame(response)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['longRatio'] = pd.to_numeric(df['longAccount'])
            df['shortRatio'] = pd.to_numeric(df['shortAccount'])
            df['longShortRatio'] = pd.to_numeric(df['longShortRatio'])
            return df[['timestamp', 'longRatio', 'shortRatio', 'longShortRatio']]

    except Exception as e:
        print(f"    Error: {e}")

    return None


def download_taker_buy_sell(exchange, symbol, period='1h', days=30):
    """Download taker buy/sell volume ratio."""
    print(f"  Taker Buy/Sell for {symbol}...")

    base_symbol = symbol.replace('/USDT:USDT', 'USDT')

    try:
        params = {
            'symbol': base_symbol,
            'period': period,
            'limit': 500,
        }

        response = exchange.fapiPublicGetTakerlongshortRatio(params)

        if response:
            df = pd.DataFrame(response)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['buyRatio'] = pd.to_numeric(df['buyVol']) / (
                pd.to_numeric(df['buyVol']) + pd.to_numeric(df['sellVol'])
            )
            return df[['timestamp', 'buyRatio']]

    except Exception as e:
        print(f"    Error: {e}")

    return None


def main():
    print("=" * 60)
    print("DERIVATIVES DATA DOWNLOADER")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Symbols: {len(SYMBOLS)}")
    print()

    # Initialize Binance Futures
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })

    # Load markets
    print("Loading markets...")
    exchange.load_markets()

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print('='*60)

        base = symbol.split('/')[0].lower()

        # 1. Funding Rates
        df_funding = download_funding_rates(exchange, symbol)
        if df_funding is not None:
            path = f"{OUTPUT_DIR}/{base}_funding.csv"
            df_funding.to_csv(path, index=False)
            print(f"    Saved: {len(df_funding)} rows")

        # 2. Open Interest (multiple timeframes)
        for tf in ['1h', '4h', '1d']:
            df_oi = download_open_interest(exchange, symbol, tf)
            if df_oi is not None:
                path = f"{OUTPUT_DIR}/{base}_oi_{tf}.csv"
                df_oi.to_csv(path, index=False)
                print(f"    Saved OI {tf}: {len(df_oi)} rows")

        # 3. Long/Short Ratio
        df_lsr = download_long_short_ratio(exchange, symbol)
        if df_lsr is not None:
            path = f"{OUTPUT_DIR}/{base}_lsr.csv"
            df_lsr.to_csv(path, index=False)
            print(f"    Saved LSR: {len(df_lsr)} rows")

        # 4. Taker Buy/Sell
        df_taker = download_taker_buy_sell(exchange, symbol)
        if df_taker is not None:
            path = f"{OUTPUT_DIR}/{base}_taker.csv"
            df_taker.to_csv(path, index=False)
            print(f"    Saved Taker: {len(df_taker)} rows")

        time.sleep(1)  # Between symbols

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    # Summary
    files = os.listdir(OUTPUT_DIR)
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"Total files: {len(csv_files)}")

    for f in sorted(csv_files):
        path = f"{OUTPUT_DIR}/{f}"
        size = os.path.getsize(path) / 1024
        print(f"  {f}: {size:.1f} KB")


if __name__ == '__main__':
    main()
