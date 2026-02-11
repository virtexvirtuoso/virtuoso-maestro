#!/usr/bin/env python3
"""
Smart Batch Backtester

Runs strategies in parallel batches, prioritized by timeframe.
Skips 1m data (too large for screening).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
from rethinkdb import RethinkDB
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from strategies import list_strategies, load_strategy
from engine.vectorized_backtester import backtest_strategy

# Timeframe priority (fastest first)
TIMEFRAME_PRIORITY = ['1d', '4h', '1h', '15m', '5m']
SKIP_TIMEFRAMES = ['1m']  # Too large for screening


def get_tables_by_timeframe(conn, db):
    """Get tables grouped by timeframe, sorted by priority."""
    r = RethinkDB()
    tables = list(r.db(db).table_list().run(conn))
    tables = [t for t in tables if t.startswith('trade_') and t != 'trade_metadata']

    # Group by timeframe
    by_tf = {}
    for t in tables:
        parts = t.split('_')
        tf = parts[-1]
        if tf in SKIP_TIMEFRAMES:
            continue
        by_tf.setdefault(tf, []).append(t)

    # Sort by priority
    result = []
    for tf in TIMEFRAME_PRIORITY:
        if tf in by_tf:
            result.extend([(t, tf) for t in sorted(by_tf[tf])])

    # Add any remaining
    for tf, tables in by_tf.items():
        if tf not in TIMEFRAME_PRIORITY:
            result.extend([(t, tf) for t in sorted(tables)])

    return result


def load_data_cached(table_name, db_name='filos-dev'):
    """Load OHLCV data from RethinkDB."""
    r = RethinkDB()
    conn = r.connect(host='localhost', port=28015)
    cursor = r.db(db_name).table(table_name).order_by('timestamp').run(conn)
    data = list(cursor)
    conn.close()

    if not data:
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def run_strategy_batch(args):
    """Run one strategy on one table."""
    table_name, strategy_name = args

    try:
        df = load_data_cached(table_name)
        if df is None or len(df) < 100:
            return None

        module = load_strategy(strategy_name)
        if module is None:
            return None

        result = backtest_strategy(df, module.generate_signals, strategy_name)

        # Parse table name
        parts = table_name.split('_')
        result['exchange'] = parts[1]
        result['symbol'] = '_'.join(parts[2:-1])
        result['timeframe'] = parts[-1]
        result['rows'] = len(df)

        return result
    except Exception as e:
        return {'strategy': strategy_name, 'table': table_name, 'error': str(e)}


def main():
    print("=" * 70)
    print("MAESTRO SMART BATCH BACKTESTER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU cores: {cpu_count()}")
    print()

    r = RethinkDB()
    conn = r.connect(host='localhost', port=28015)

    # Get tables (excluding 1m)
    tables = get_tables_by_timeframe(conn, 'filos-dev')
    strategies = list_strategies()
    conn.close()

    print(f"Tables (excl 1m): {len(tables)}")
    print(f"Strategies: {len(strategies)}")

    # Show breakdown
    tf_counts = {}
    for t, tf in tables:
        tf_counts[tf] = tf_counts.get(tf, 0) + 1
    print(f"By timeframe: {tf_counts}")

    total = len(tables) * len(strategies)
    print(f"Total backtests: {total:,}")
    print()

    # Build all tasks
    tasks = []
    for table, tf in tables:
        for strat in strategies:
            tasks.append((table, strat))

    # Process in parallel batches
    results = []
    batch_size = 500
    workers = min(cpu_count(), 8)

    print(f"Running with {workers} workers, batch size {batch_size}...")
    print("-" * 70)

    start = datetime.now()

    with Pool(workers) as pool:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = pool.map(run_strategy_batch, batch)

            for r in batch_results:
                if r:
                    results.append(r)

            done = min(i + batch_size, len(tasks))
            elapsed = (datetime.now() - start).total_seconds()
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - done) / rate if rate > 0 else 0

            print(f"  [{done:5d}/{len(tasks)}] {done/len(tasks)*100:5.1f}% | "
                  f"{len(results)} results | {rate:.0f}/sec | ETA {eta:.0f}s")

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    df = pd.DataFrame(results)
    df_valid = df[df.get('error').isna()] if 'error' in df.columns else df

    print(f"Total: {len(df)}, Valid: {len(df_valid)}")

    if len(df_valid) > 0:
        # Top by Sharpe
        print("\n" + "=" * 70)
        print("TOP 25 BY SHARPE RATIO")
        print("=" * 70)
        top = df_valid.nlargest(25, 'sharpe_ratio')[
            ['strategy', 'symbol', 'timeframe', 'sharpe_ratio', 'total_return', 'win_rate', 'total_trades']
        ]
        print(top.to_string(index=False))

        # Top by return
        print("\n" + "=" * 70)
        print("TOP 25 BY TOTAL RETURN")
        print("=" * 70)
        top = df_valid.nlargest(25, 'total_return')[
            ['strategy', 'symbol', 'timeframe', 'total_return', 'sharpe_ratio', 'win_rate', 'total_trades']
        ]
        print(top.to_string(index=False))

        # Strategy rankings
        print("\n" + "=" * 70)
        print("STRATEGY RANKINGS (avg Sharpe)")
        print("=" * 70)
        strat_avg = df_valid.groupby('strategy').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2).sort_values('sharpe_ratio', ascending=False)
        print(strat_avg.head(30).to_string())

    # Save
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'{out_dir}/batch_results_{ts}.csv'
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
