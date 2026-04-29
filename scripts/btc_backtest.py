#!/usr/bin/env python3
"""
Bitcoin Strategy Screener - Test ALL strategies on BTC across ALL timeframes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob
import warnings
warnings.filterwarnings('ignore')

from strategies import list_strategies, load_strategy
from engine.vectorized_backtester import backtest_strategy

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data/merged'

def main():
    print("=" * 80)
    print("BITCOIN STRATEGY SCREENER - All Strategies × All Timeframes")
    print("=" * 80)
    start_time = datetime.now()

    # Get BTC files (skip 1m - too large)
    btc_files = sorted(glob(os.path.join(DATA_DIR, 'binance_btc_usdt_*.csv')))
    btc_files = [f for f in btc_files if '_1m.csv' not in f]  # Skip 1m

    strategies = list_strategies()

    print(f"\nTimeframes: {[os.path.basename(f).split('_')[-1].replace('.csv','') for f in btc_files]}")
    print(f"Strategies: {len(strategies)}")
    print(f"Total tests: {len(btc_files) * len(strategies)}")
    print("-" * 80)

    results = []

    for filepath in btc_files:
        timeframe = os.path.basename(filepath).split('_')[-1].replace('.csv', '')
        print(f"\n>>> Testing {timeframe} timeframe...")

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"    Data: {len(df):,} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

        for strategy_name in strategies:
            try:
                module = load_strategy(strategy_name)
                if module is None:
                    continue

                result = backtest_strategy(
                    df, module.generate_signals, strategy_name,
                    timeframe=timeframe,
                    position_mode='fixed',
                    position_size=0.1
                )
                result['timeframe'] = timeframe
                result['rows'] = len(df)
                results.append(result)

            except Exception as e:
                results.append({
                    'strategy': strategy_name,
                    'timeframe': timeframe,
                    'error': str(e)
                })

    elapsed = (datetime.now() - start_time).total_seconds()

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results['sharpe_ratio'].notna() & ~df_results.get('error', pd.Series(dtype=str)).notna()]

    print("\n" + "=" * 80)
    print(f"COMPLETE in {elapsed:.1f}s")
    print("=" * 80)
    print(f"Total: {len(df_results)}, Valid: {len(df_valid)}")

    if len(df_valid) > 0:
        # Top by Sharpe
        print("\n" + "=" * 80)
        print("TOP 30 BY SHARPE RATIO (BTC/USDT)")
        print("=" * 80)
        top_sharpe = df_valid.nlargest(30, 'sharpe_ratio')[
            ['strategy', 'timeframe', 'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'total_trades']
        ]
        print(top_sharpe.to_string(index=False))

        # Top by Return
        print("\n" + "=" * 80)
        print("TOP 30 BY TOTAL RETURN (BTC/USDT)")
        print("=" * 80)
        top_return = df_valid.nlargest(30, 'total_return')[
            ['strategy', 'timeframe', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
        ]
        print(top_return.to_string(index=False))

        # Best by timeframe
        print("\n" + "=" * 80)
        print("BEST STRATEGY PER TIMEFRAME")
        print("=" * 80)
        for tf in sorted(df_valid['timeframe'].unique()):
            tf_data = df_valid[df_valid['timeframe'] == tf]
            if len(tf_data) > 0:
                best = tf_data.nlargest(1, 'sharpe_ratio').iloc[0]
                print(f"  {tf:>4}: {best['strategy']:<30} Sharpe={best['sharpe_ratio']:.2f}  Return={best['total_return']*100:.1f}%  MaxDD={best['max_drawdown']*100:.1f}%")

        # Strategy rankings
        print("\n" + "=" * 80)
        print("STRATEGY RANKINGS (avg across all timeframes)")
        print("=" * 80)
        strat_avg = df_valid.groupby('strategy').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(3).sort_values('sharpe_ratio', ascending=False)
        print(strat_avg.head(40).to_string())

        # Timeframe rankings
        print("\n" + "=" * 80)
        print("TIMEFRAME RANKINGS (avg across all strategies)")
        print("=" * 80)
        tf_avg = df_valid.groupby('timeframe').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
        }).round(3).sort_values('sharpe_ratio', ascending=False)
        print(tf_avg.to_string())

    # Save
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'{out_dir}/btc_all_strategies_{ts}.csv'
    df_results.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
