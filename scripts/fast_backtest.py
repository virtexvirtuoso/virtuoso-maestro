#!/usr/bin/env python3
"""
Fast Backtest - Direct CSV Reading

Reads from CSV files (no RethinkDB overhead) for maximum speed.
Skips 1m data (too large).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from glob import glob
import warnings
warnings.filterwarnings('ignore')

from strategies import list_strategies, load_strategy
from engine.vectorized_backtester import backtest_strategy

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data/merged'  # Use merged data with derivatives
SKIP_PATTERNS = ['_1m.csv', '_5m.csv']  # Skip 1m and 5m (too slow)


def get_csv_files():
    """Get all CSV files, prioritized by size (smallest first)."""
    files = glob(os.path.join(DATA_DIR, '*.csv'))

    # Filter out 1m files
    files = [f for f in files if not any(p in f for p in SKIP_PATTERNS)]

    # Sort by file size (smallest first for faster initial progress)
    files.sort(key=os.path.getsize)

    return files


def parse_filename(filepath):
    """Parse CSV filename into components."""
    # binance_btc_usdt_1d.csv -> exchange, symbol, timeframe
    name = os.path.basename(filepath).replace('.csv', '')
    parts = name.split('_')
    exchange = parts[0].upper()
    timeframe = parts[-1]
    symbol = '_'.join(parts[1:-1])
    return exchange, symbol, timeframe


def run_one(args):
    """Run one strategy on one file."""
    filepath, strategy_name = args

    try:
        df = pd.read_csv(filepath)

        if len(df) < 100:
            return None

        # Ensure correct columns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        module = load_strategy(strategy_name)
        if module is None:
            return None

        # Parse timeframe from filename BEFORE backtesting
        exchange, symbol, timeframe = parse_filename(filepath)

        # Pass timeframe explicitly for correct Sharpe annualization
        result = backtest_strategy(df, module.generate_signals, strategy_name, timeframe=timeframe)
        result['exchange'] = exchange
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        result['rows'] = len(df)

        return result

    except Exception as e:
        return {'strategy': strategy_name, 'file': os.path.basename(filepath), 'error': str(e)}


def main():
    print("=" * 70)
    print("MAESTRO FAST BACKTESTER (CSV-direct)")
    print("=" * 70)
    start_time = datetime.now()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get files and strategies
    files = get_csv_files()
    strategies = list_strategies()

    # Summary
    total_size = sum(os.path.getsize(f) for f in files) / 1024 / 1024
    print(f"\nFiles: {len(files)} ({total_size:.1f} MB)")
    print(f"Strategies: {len(strategies)}")
    print(f"Total tests: {len(files) * len(strategies):,}")

    # Build task list
    tasks = [(f, s) for f in files for s in strategies]

    # Process
    results = []
    workers = min(cpu_count(), 8)
    batch_size = 200  # Smaller batches for more frequent updates

    print(f"\nRunning with {workers} workers...")
    print("-" * 70)

    with Pool(workers) as pool:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = pool.map(run_one, batch)

            for r in batch_results:
                if r:
                    results.append(r)

            done = min(i + batch_size, len(tasks))
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - done) / rate if rate > 0 else 0

            print(f"  [{done:5d}/{len(tasks)}] {done/len(tasks)*100:5.1f}% | "
                  f"{len(results)} results | {rate:.0f}/sec | ETA {eta:.0f}s")

    elapsed = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 70)
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print("=" * 70)

    # Results
    df = pd.DataFrame(results)
    df_valid = df[~df.get('error', pd.Series(dtype=str)).notna()] if 'error' in df.columns else df

    # Filter valid rows that have sharpe_ratio
    if 'sharpe_ratio' in df.columns:
        df_valid = df[df['sharpe_ratio'].notna() & ~df.get('error', pd.Series(dtype=str)).notna()]

    print(f"\nTotal: {len(df)}, Valid: {len(df_valid)}")

    if len(df_valid) > 0:
        print("\n" + "=" * 70)
        print("TOP 25 BY SHARPE RATIO")
        print("=" * 70)
        top = df_valid.nlargest(25, 'sharpe_ratio')[
            ['strategy', 'symbol', 'timeframe', 'sharpe_ratio', 'total_return', 'win_rate', 'total_trades']
        ]
        print(top.to_string(index=False))

        print("\n" + "=" * 70)
        print("TOP 25 BY TOTAL RETURN")
        print("=" * 70)
        top = df_valid.nlargest(25, 'total_return')[
            ['strategy', 'symbol', 'timeframe', 'total_return', 'sharpe_ratio', 'win_rate', 'total_trades']
        ]
        print(top.to_string(index=False))

        print("\n" + "=" * 70)
        print("STRATEGY RANKINGS (by avg Sharpe)")
        print("=" * 70)
        strat_avg = df_valid.groupby('strategy').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2).sort_values('sharpe_ratio', ascending=False)
        print(strat_avg.head(35).to_string())

        print("\n" + "=" * 70)
        print("SYMBOL RANKINGS (by avg Sharpe)")
        print("=" * 70)
        sym_avg = df_valid.groupby('symbol').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
        }).round(2).sort_values('sharpe_ratio', ascending=False)
        print(sym_avg.to_string())

    # Save
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'{out_dir}/fast_results_{ts}.csv'
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
