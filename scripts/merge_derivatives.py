#!/usr/bin/env python3
"""
Merge OHLCV data with Derivatives data

Creates enriched CSV files with:
- OHLCV data
- Funding rate
- Open Interest
- Long/Short Ratio
- Taker Buy/Sell Ratio
"""
import os
import pandas as pd
from glob import glob
from datetime import datetime

OHLCV_DIR = '/Users/ffv_macmini/Desktop/maestro/data/ohlcv'
DERIV_DIR = '/Users/ffv_macmini/Desktop/maestro/data/derivatives'
OUTPUT_DIR = '/Users/ffv_macmini/Desktop/maestro/data/merged'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map OHLCV symbols to derivatives symbols
SYMBOL_MAP = {
    'btc_usdt': 'btc',
    'eth_usdt': 'eth',
    'sol_usdt': 'sol',
    'arb_usdt': 'arb',
    'op_usdt': 'op',
    'sui_usdt': 'sui',
    'tia_usdt': 'tia',
    'inj_usdt': 'inj',
    'link_usdt': 'link',
    'avax_usdt': 'avax',
    'fet_usdt': 'fet',
    'tao_usdt': 'tao',
    'render_usdt': 'render',
}


def load_derivatives(symbol, deriv_dir):
    """Load all derivatives data for a symbol."""
    result = {}

    # Funding rate
    funding_path = os.path.join(deriv_dir, f'{symbol}_funding.csv')
    if os.path.exists(funding_path):
        df = pd.read_csv(funding_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        result['funding_rate'] = df['fundingRate']

    # Open Interest (prefer 1h)
    for tf in ['1h', '4h', '1d']:
        oi_path = os.path.join(deriv_dir, f'{symbol}_oi_{tf}.csv')
        if os.path.exists(oi_path):
            df = pd.read_csv(oi_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            result['oi'] = df['sumOpenInterest']
            result['oi_value'] = df['sumOpenInterestValue']
            break

    # Long/Short Ratio (global)
    lsr_path = os.path.join(deriv_dir, f'{symbol}_lsr_global.csv')
    if os.path.exists(lsr_path):
        df = pd.read_csv(lsr_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        result['long_ratio'] = df['longAccount'].astype(float)
        result['short_ratio'] = df['shortAccount'].astype(float)
        result['long_short_ratio'] = df['longShortRatio'].astype(float)

    # Taker ratio
    taker_path = os.path.join(deriv_dir, f'{symbol}_taker.csv')
    if os.path.exists(taker_path):
        df = pd.read_csv(taker_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        result['taker_buy_ratio'] = df['buyVol'].astype(float) / (
            df['buyVol'].astype(float) + df['sellVol'].astype(float)
        )

    return result


def merge_ohlcv_with_derivatives(ohlcv_path, derivatives, output_path):
    """Merge OHLCV data with derivatives data."""
    # Load OHLCV
    df = pd.read_csv(ohlcv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Add each derivatives column
    for col_name, series in derivatives.items():
        # Resample to match OHLCV frequency and forward-fill
        df[col_name] = series.reindex(df.index, method='ffill')

    # Reset index and save
    df = df.reset_index()
    df.to_csv(output_path, index=False)

    return len(df), df[list(derivatives.keys())].notna().sum()


def main():
    print("=" * 60)
    print("MERGE OHLCV WITH DERIVATIVES")
    print("=" * 60)
    print(f"OHLCV: {OHLCV_DIR}")
    print(f"Derivatives: {DERIV_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Process each OHLCV file
    ohlcv_files = glob(os.path.join(OHLCV_DIR, '*.csv'))
    processed = 0
    skipped = 0

    for ohlcv_path in sorted(ohlcv_files):
        filename = os.path.basename(ohlcv_path)
        # Parse: binance_btc_usdt_1h.csv
        parts = filename.replace('.csv', '').split('_')
        exchange = parts[0]
        symbol = '_'.join(parts[1:-1])  # btc_usdt
        timeframe = parts[-1]

        # Check if we have derivatives for this symbol
        if symbol not in SYMBOL_MAP:
            skipped += 1
            continue

        deriv_symbol = SYMBOL_MAP[symbol]

        # Load derivatives
        derivatives = load_derivatives(deriv_symbol, DERIV_DIR)
        if not derivatives:
            skipped += 1
            continue

        # Merge
        output_path = os.path.join(OUTPUT_DIR, filename)
        rows, coverage = merge_ohlcv_with_derivatives(ohlcv_path, derivatives, output_path)

        print(f"  {filename}: {rows} rows, coverage: {dict(coverage)}")
        processed += 1

    print()
    print("=" * 60)
    print(f"COMPLETE: {processed} files merged, {skipped} skipped")
    print("=" * 60)

    # Summary of output files
    output_files = sorted(glob(os.path.join(OUTPUT_DIR, '*.csv')))
    total_size = sum(os.path.getsize(f) for f in output_files) / 1024 / 1024
    print(f"\nOutput: {len(output_files)} files, {total_size:.1f} MB")


if __name__ == '__main__':
    main()
