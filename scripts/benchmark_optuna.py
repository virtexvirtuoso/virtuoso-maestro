#!/usr/bin/env python3
"""
Phase 0: Optuna Walk-Forward Benchmark

Measures baseline performance of the current WF engine so we can verify
improvements after each optimization phase.

Usage:
    cd ~/Desktop/maestro && PYTHONPATH=backend python scripts/benchmark_optuna.py
    cd ~/Desktop/maestro && PYTHONPATH=backend python scripts/benchmark_optuna.py --parallel
    cd ~/Desktop/maestro && PYTHONPATH=backend python scripts/benchmark_optuna.py --compare baseline.json

Metrics captured:
    - Wall time (total + per fold)
    - Trials completed vs requested
    - Pruning rate
    - Peak memory (RSS)
    - Aggregate Sharpe / return
"""
import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Import stdlib resource by absolute path — backend/resource/ shadows the stdlib module
_resource = importlib.import_module('resource')
if not hasattr(_resource, 'getrusage'):
    # Shadowed by backend/resource package — force reload from stdlib
    import importlib.util
    _spec = importlib.util.find_spec('resource')
    if _spec and _spec.origin and 'lib-dynload' in _spec.origin:
        _resource = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_resource)
    else:
        _resource = None

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from engine_v2.strategy_adapter import EMACrossStrategy
from engine_v2.walk_forward_optuna import WalkForwardConfig, WalkForwardOptuna
from engine_v2.vectorbt_engine import BacktestConfig

# Try importing parallel engine
try:
    from engine_v2.parallel_walk_forward import ParallelWalkForward, ParallelConfig
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False

DATA_PATH = Path(__file__).parent.parent / 'data' / 'spot' / 'BTC_spot_daily.csv'
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'benchmark'


def load_data() -> pd.DataFrame:
    """Load BTC daily data."""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['Date'])
    df = df.set_index('timestamp')
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume',
    })
    df = df[['open', 'high', 'low', 'close', 'volume']]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df


def get_peak_memory_mb() -> float:
    """Get peak RSS in MB (macOS/Linux)."""
    if _resource is None or not hasattr(_resource, 'getrusage'):
        return 0.0
    ru = _resource.getrusage(_resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KB
    if sys.platform == 'darwin':
        return ru.ru_maxrss / 1e6
    return ru.ru_maxrss / 1e3


def run_benchmark(parallel: bool = False, n_splits: int = 5, n_trials: int = 50) -> dict:
    """Run a single benchmark pass."""
    print(f"{'='*60}")
    print(f"OPTUNA WALK-FORWARD BENCHMARK")
    print(f"{'='*60}")
    print(f"Mode:     {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    print(f"Splits:   {n_splits}")
    print(f"Trials:   {n_trials}")
    print(f"Strategy: EMACrossStrategy (4 params)")
    print(f"Data:     BTC spot daily")

    data = load_data()
    print(f"Bars:     {len(data)} ({data.index[0].date()} to {data.index[-1].date()})")
    print(f"{'='*60}")

    strategy = EMACrossStrategy()
    wf_config = WalkForwardConfig(
        num_splits=n_splits,
        train_splits=2,
        test_splits=1,
        n_trials=n_trials,
        optimization_metric='sharpe_ratio',
        use_vwr_ranking=True,
        pruning_enabled=True,
        n_startup_trials=10,
        use_dashboard_storage=False,  # Don't pollute the real DB
    )
    backtest_config = BacktestConfig(
        cash=100000.0,
        commission=0.001,
        slippage=0.0005,
    )

    mem_before = get_peak_memory_mb()
    t_start = time.time()

    if parallel and HAS_PARALLEL:
        engine = ParallelWalkForward(
            data=data,
            strategy=strategy,
            config=wf_config,
            backtest_config=backtest_config,
            parallel_config=ParallelConfig(enabled=True),
        )
    else:
        engine = WalkForwardOptuna(
            data=data,
            strategy=strategy,
            config=wf_config,
            backtest_config=backtest_config,
        )

    result = engine.run()
    t_end = time.time()
    mem_after = get_peak_memory_mb()

    wall_time = t_end - t_start

    # Collect per-fold metrics
    fold_metrics = []
    failed_folds = 0
    for i, fold in enumerate(result.fold_results):
        is_failed = (fold.sharpe_ratio == 0 and fold.num_trades == 0)
        if is_failed:
            failed_folds += 1
        fold_metrics.append({
            'fold': i,
            'sharpe': fold.sharpe_ratio,
            'return': fold.total_return,
            'trades': fold.num_trades,
            'max_dd': fold.max_drawdown,
            'failed': is_failed,
        })

    # Build result
    benchmark = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'parallel' if (parallel and HAS_PARALLEL) else 'sequential',
        'config': {
            'n_splits': n_splits,
            'n_trials': n_trials,
            'strategy': 'EMACrossStrategy',
            'n_params': len(strategy.get_param_space()),
            'data_bars': len(data),
        },
        'timing': {
            'wall_time_s': round(wall_time, 2),
            'per_fold_avg_s': round(wall_time / max(len(result.fold_results), 1), 2),
        },
        'memory': {
            'peak_rss_mb': round(mem_after, 1),
            'delta_mb': round(mem_after - mem_before, 1),
        },
        'results': {
            'total_folds': len(result.fold_results),
            'failed_folds': failed_folds,
            'avg_sharpe': round(result.aggregate_metrics.get('avg_sharpe', 0), 4),
            'total_return': round(result.aggregate_metrics.get('total_return', 0), 4),
            'total_trades': result.aggregate_metrics.get('total_trades', 0),
            'max_drawdown': round(result.aggregate_metrics.get('max_drawdown', 0), 4),
        },
        'folds': fold_metrics,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Wall time:     {wall_time:.2f}s")
    print(f"Per fold avg:  {wall_time / max(len(result.fold_results), 1):.2f}s")
    print(f"Peak memory:   {mem_after:.1f} MB")
    print(f"Folds:         {len(result.fold_results)} ({failed_folds} failed)")
    print(f"Avg Sharpe:    {result.aggregate_metrics.get('avg_sharpe', 0):.4f}")
    print(f"Total Return:  {result.aggregate_metrics.get('total_return', 0):.2%}")
    print(f"Total Trades:  {result.aggregate_metrics.get('total_trades', 0)}")
    print(f"\nPer-fold breakdown:")
    for fm in fold_metrics:
        status = "FAIL" if fm['failed'] else "OK"
        print(f"  Fold {fm['fold']}: Sharpe={fm['sharpe']:.3f} "
              f"Return={fm['return']:.2%} Trades={fm['trades']} [{status}]")

    return benchmark


def compare(current: dict, baseline_path: str):
    """Compare current benchmark against a saved baseline."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\n{'='*60}")
    print(f"COMPARISON vs {baseline_path}")
    print(f"{'='*60}")

    b_time = baseline['timing']['wall_time_s']
    c_time = current['timing']['wall_time_s']
    speedup = b_time / c_time if c_time > 0 else float('inf')

    print(f"Wall time:  {b_time:.2f}s → {c_time:.2f}s ({speedup:.2f}x)")
    print(f"Memory:     {baseline['memory']['peak_rss_mb']:.1f} MB → {current['memory']['peak_rss_mb']:.1f} MB")
    print(f"Avg Sharpe: {baseline['results']['avg_sharpe']:.4f} → {current['results']['avg_sharpe']:.4f}")
    print(f"Failed:     {baseline['results']['failed_folds']} → {current['results']['failed_folds']}")

    if speedup > 1.1:
        print(f"\n✓ {speedup:.1f}x speedup achieved")
    elif speedup < 0.9:
        print(f"\n✗ {1/speedup:.1f}x REGRESSION")
    else:
        print(f"\n~ No significant change")


def main():
    parser = argparse.ArgumentParser(description='Optuna WF Benchmark')
    parser.add_argument('--parallel', action='store_true', help='Use ParallelWalkForward')
    parser.add_argument('--splits', type=int, default=5, help='WF splits (default: 5)')
    parser.add_argument('--trials', type=int, default=50, help='Trials per fold (default: 50)')
    parser.add_argument('--compare', type=str, default=None, help='Compare against baseline JSON')
    parser.add_argument('--save', type=str, default=None, help='Save results to file (default: auto)')
    args = parser.parse_args()

    result = run_benchmark(
        parallel=args.parallel,
        n_splits=args.splits,
        n_trials=args.trials,
    )

    if args.compare:
        compare(result, args.compare)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.save:
        out_path = args.save
    else:
        mode = result['mode']
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = str(RESULTS_DIR / f'benchmark_{mode}_{ts}.json')

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
