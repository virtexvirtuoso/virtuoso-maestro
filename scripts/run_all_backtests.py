#!/usr/bin/env python3
"""
Comprehensive Strategy Backtester

Runs all 65 strategies on all symbols and timeframes.
Results saved to RethinkDB and CSV for analysis.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
from rethinkdb import RethinkDB
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from strategies import list_strategies, load_strategy
from engine.vectorized_backtester import backtest_strategy


def get_all_tables(conn, db):
    """Get all OHLCV tables from RethinkDB."""
    r = RethinkDB()
    tables = list(r.db(db).table_list().run(conn))
    # Filter for trade tables only
    return [t for t in tables if t.startswith('trade_') and t != 'trade_metadata']


def load_ohlcv_data(conn, db, table_name):
    """Load OHLCV data from RethinkDB table."""
    r = RethinkDB()
    cursor = r.db(db).table(table_name).order_by('timestamp').run(conn)
    data = list(cursor)

    if not data:
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def parse_table_name(table_name):
    """Parse table name into components."""
    # Format: trade_EXCHANGE_symbol_timeframe
    parts = table_name.split('_')
    if len(parts) >= 4:
        exchange = parts[1]
        timeframe = parts[-1]
        symbol = '_'.join(parts[2:-1])
        return exchange, symbol, timeframe
    return None, None, None


def run_single_backtest(args):
    """Run a single strategy on a single table. Used for parallel execution."""
    table_name, strategy_name, db_name = args

    try:
        r = RethinkDB()
        conn = r.connect(host='localhost', port=28015)

        # Load data
        df = load_ohlcv_data(conn, db_name, table_name)
        conn.close()

        if df is None or len(df) < 100:
            return None

        # Load strategy
        module = load_strategy(strategy_name)
        if module is None:
            return None

        # Run backtest
        result = backtest_strategy(df, module.generate_signals, strategy_name)

        # Add metadata
        exchange, symbol, timeframe = parse_table_name(table_name)
        result['exchange'] = exchange
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        result['table'] = table_name
        result['rows'] = len(df)
        result['start_date'] = df['timestamp'].min().isoformat()
        result['end_date'] = df['timestamp'].max().isoformat()

        return result

    except Exception as e:
        return {
            'strategy': strategy_name,
            'table': table_name,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("MAESTRO COMPREHENSIVE STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Connect to RethinkDB
    r = RethinkDB()
    conn = r.connect(host='localhost', port=28015)
    db_name = 'filos-dev'

    # Get all tables
    tables = get_all_tables(conn, db_name)
    print(f"Found {len(tables)} OHLCV tables")

    # Get all strategies
    strategies = list_strategies()
    print(f"Found {len(strategies)} strategies")

    # Calculate total tests
    total_tests = len(tables) * len(strategies)
    print(f"Total backtests to run: {total_tests:,}")
    print()

    conn.close()

    # Build task list
    tasks = []
    for table in tables:
        for strategy in strategies:
            tasks.append((table, strategy, db_name))

    # Run backtests (sequential for stability)
    results = []
    completed = 0
    errors = 0

    print("Running backtests...")
    print("-" * 70)

    for i, task in enumerate(tasks, 1):
        result = run_single_backtest(task)
        if result:
            results.append(result)
            if result.get('error'):
                errors += 1
        completed += 1

        # Progress update every 100 tests
        if i % 100 == 0 or i == len(tasks):
            pct = i / len(tasks) * 100
            print(f"  [{i:5d}/{len(tasks)}] {pct:5.1f}% complete | {len(results)} results | {errors} errors")

    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Filter out errors for summary
    df_valid = df_results[df_results['error'].isna()].copy()

    print(f"\nTotal results: {len(df_results)}")
    print(f"Valid results: {len(df_valid)}")
    print(f"Errors: {errors}")

    # Summary statistics
    if len(df_valid) > 0:
        print("\n" + "=" * 70)
        print("TOP 20 STRATEGIES BY SHARPE RATIO")
        print("=" * 70)

        top_sharpe = (df_valid
                      .sort_values('sharpe_ratio', ascending=False)
                      .head(20)[['strategy', 'symbol', 'timeframe', 'sharpe_ratio',
                                 'total_return', 'win_rate', 'total_trades']])
        print(top_sharpe.to_string(index=False))

        print("\n" + "=" * 70)
        print("TOP 20 STRATEGIES BY TOTAL RETURN")
        print("=" * 70)

        top_return = (df_valid
                      .sort_values('total_return', ascending=False)
                      .head(20)[['strategy', 'symbol', 'timeframe', 'total_return',
                                 'sharpe_ratio', 'win_rate', 'total_trades']])
        print(top_return.to_string(index=False))

        print("\n" + "=" * 70)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("=" * 70)

        strategy_summary = (df_valid
                           .groupby('strategy')
                           .agg({
                               'sharpe_ratio': ['mean', 'std', 'max'],
                               'total_return': ['mean', 'max'],
                               'win_rate': 'mean',
                               'total_trades': 'sum'
                           })
                           .round(2))
        strategy_summary.columns = ['sharpe_mean', 'sharpe_std', 'sharpe_max',
                                    'return_mean', 'return_max', 'win_rate_avg', 'total_trades']
        strategy_summary = strategy_summary.sort_values('sharpe_mean', ascending=False)
        print(strategy_summary.head(30).to_string())

    # Save results
    output_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'{output_dir}/all_strategies_{timestamp}.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save to RethinkDB
    try:
        conn = r.connect(host='localhost', port=28015)
        db = r.db(db_name)

        # Create results table if needed
        if 'backtest_results' not in list(db.table_list().run(conn)):
            db.table_create('backtest_results').run(conn)

        # Insert results
        records = df_results.to_dict('records')
        for rec in records:
            rec['run_timestamp'] = datetime.now().isoformat()

        db.table('backtest_results').insert(records).run(conn)
        print(f"Results saved to RethinkDB: backtest_results table")
        conn.close()
    except Exception as e:
        print(f"Warning: Could not save to RethinkDB: {e}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
