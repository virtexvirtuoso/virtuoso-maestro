#!/usr/bin/env python3
"""
BTC High-Performance Strategy Optimizer

Targets the TOP performing strategies on their OPTIMAL timeframes:
- CapitulationReversal+VolumeFilter @ 15m: Sharpe 1.06, +122%
- ADXSmas @ 4h: Sharpe 0.88, +1098%
- Momentum+TrendFilter @ 1d: Sharpe 0.95, +1170%
- Momentum @ 1d: Sharpe 0.90, +1218%
- VWAP @ 1d: Sharpe 0.83, +371%
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from datetime import datetime
import json
import logging
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from engine.vectorized_backtester import backtest_strategy

# Import strategies directly
from strategies.hybrids.capitulation_reversal_volume import generate_signals as cap_rev_signals
from strategies.technical.adx_smas import generate_signals as adx_smas_signals
from strategies.hybrids.momentum_trend import generate_signals as mom_trend_signals
from strategies.technical.momentum import generate_signals as momentum_signals
from strategies.scalping.vwap import generate_signals as vwap_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data/merged'

# Strategy configs: name, signal_func, timeframe, param_space
STRATEGY_CONFIGS = [
    {
        'name': 'CapitulationReversal+VolumeFilter',
        'signal_func': cap_rev_signals,
        'timeframe': '15m',
        'params': {
            'vol_mult': ('float', 2.0, 8.0),
            'period': ('int', 10, 40),
        },
        'defaults': {'vol_mult': 4.0, 'period': 20}
    },
    {
        'name': 'ADXSmas',
        'signal_func': adx_smas_signals,
        'timeframe': '4h',
        'params': {
            'adx_period': ('int', 10, 25),
            'sma_fast': ('int', 5, 20),
            'sma_slow': ('int', 25, 80),
            'adx_threshold': ('float', 15.0, 40.0),
        },
        'defaults': {'adx_period': 14, 'sma_fast': 10, 'sma_slow': 30, 'adx_threshold': 20}
    },
    {
        'name': 'Momentum+TrendFilter',
        'signal_func': mom_trend_signals,
        'timeframe': '1d',
        'params': {
            'mom_period': ('int', 5, 30),
            'trend_period': ('int', 20, 100),
        },
        'defaults': {'mom_period': 14, 'trend_period': 50}
    },
    {
        'name': 'Momentum',
        'signal_func': momentum_signals,
        'timeframe': '1d',
        'params': {
            'period': ('int', 5, 40),
        },
        'defaults': {'period': 14}
    },
    {
        'name': 'VWAP',
        'signal_func': vwap_signals,
        'timeframe': '1d',
        'params': {
            'std_mult': ('float', 1.0, 4.0),
            'session_hours': ('int', 12, 48),
        },
        'defaults': {'std_mult': 2.0, 'session_hours': 24}
    },
]


def load_btc_data(timeframe: str) -> pd.DataFrame:
    """Load BTC data for timeframe."""
    filepath = os.path.join(DATA_DIR, f'binance_btc_usdt_{timeframe}.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def time_series_split(df: pd.DataFrame, n_splits: int = 5, train_ratio: float = 0.7):
    """Generate walk-forward train/test splits."""
    n = len(df)
    fold_size = n // n_splits

    for i in range(n_splits):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)  # Overlapping folds

        fold_data = df.iloc[start:end].copy()
        train_size = int(len(fold_data) * train_ratio)

        train = fold_data.iloc[:train_size]
        test = fold_data.iloc[train_size:]

        if len(train) > 50 and len(test) > 20:
            yield i, train, test


def optimize_strategy(config: dict, n_trials: int = 100, n_splits: int = 5) -> dict:
    """Run walk-forward optimization for a strategy."""
    name = config['name']
    signal_func = config['signal_func']
    timeframe = config['timeframe']
    param_space = config['params']
    defaults = config['defaults']

    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing {name} on {timeframe}")
    logger.info(f"{'='*60}")

    # Load data
    df = load_btc_data(timeframe)
    logger.info(f"Data: {len(df):,} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")

    fold_results = []
    all_params = []

    for fold_idx, train_df, test_df in time_series_split(df, n_splits):
        logger.info(f"\n--- Fold {fold_idx}: Train={len(train_df)}, Test={len(test_df)} ---")

        # Create objective for this fold
        def objective(trial):
            params = {}
            for param_name, (ptype, low, high) in param_space.items():
                if ptype == 'int':
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)

            # Constraint: sma_fast < sma_slow for ADXSmas
            if 'sma_fast' in params and 'sma_slow' in params:
                if params['sma_fast'] >= params['sma_slow']:
                    return float('-inf')

            try:
                result = backtest_strategy(
                    train_df,
                    lambda df: signal_func(df, **params),
                    name,
                    timeframe=timeframe,
                    position_mode='fixed',
                    position_size=0.1
                )
                sharpe = result.get('sharpe_ratio', 0)
                if pd.isna(sharpe) or sharpe == float('-inf'):
                    return float('-inf')
                return sharpe
            except Exception as e:
                return float('-inf')

        # Optimize on training data
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(n_startup_trials=10, multivariate=True),
            pruner=HyperbandPruner()
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

        best_params = study.best_params
        all_params.append(best_params)

        # Test on out-of-sample data
        try:
            test_result = backtest_strategy(
                test_df,
                lambda df: signal_func(df, **best_params),
                name,
                timeframe=timeframe,
                position_mode='fixed',
                position_size=0.1
            )

            fold_result = {
                'fold': fold_idx,
                'train_sharpe': study.best_value,
                'test_sharpe': test_result.get('sharpe_ratio', 0),
                'test_return': test_result.get('total_return', 0),
                'test_max_dd': test_result.get('max_drawdown', 0),
                'test_trades': test_result.get('total_trades', 0),
                'test_win_rate': test_result.get('win_rate', 0),
                'params': best_params
            }

            logger.info(f"  Train Sharpe: {study.best_value:.3f}")
            logger.info(f"  Test Sharpe: {fold_result['test_sharpe']:.3f}, Return: {fold_result['test_return']*100:.1f}%")
            logger.info(f"  Params: {best_params}")

        except Exception as e:
            logger.error(f"  Test failed: {e}")
            fold_result = {
                'fold': fold_idx,
                'train_sharpe': study.best_value,
                'test_sharpe': 0,
                'test_return': 0,
                'test_max_dd': 0,
                'test_trades': 0,
                'test_win_rate': 0,
                'params': best_params
            }

        fold_results.append(fold_result)

    # Aggregate results
    valid_folds = [f for f in fold_results if f['test_trades'] > 0]

    if valid_folds:
        avg_test_sharpe = np.mean([f['test_sharpe'] for f in valid_folds])
        avg_test_return = np.mean([f['test_return'] for f in valid_folds])
        total_test_return = np.prod([1 + f['test_return'] for f in valid_folds]) - 1
        avg_test_dd = np.mean([f['test_max_dd'] for f in valid_folds])
        total_trades = sum([f['test_trades'] for f in valid_folds])
        avg_win_rate = np.mean([f['test_win_rate'] for f in valid_folds])
    else:
        avg_test_sharpe = avg_test_return = total_test_return = avg_test_dd = total_trades = avg_win_rate = 0

    return {
        'strategy': name,
        'timeframe': timeframe,
        'aggregate': {
            'avg_test_sharpe': avg_test_sharpe,
            'avg_test_return': avg_test_return,
            'compounded_return': total_test_return,
            'avg_max_dd': avg_test_dd,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'valid_folds': len(valid_folds),
        },
        'folds': fold_results,
        'optimal_params': all_params,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100, help='Optuna trials per fold')
    parser.add_argument('--splits', type=int, default=5, help='Walk-forward splits')
    parser.add_argument('--strategy', type=str, help='Single strategy to optimize (optional)')
    args = parser.parse_args()

    print("=" * 80)
    print("BTC HIGH-PERFORMANCE STRATEGY OPTIMIZER")
    print("=" * 80)
    print(f"Trials/fold: {args.trials}, Splits: {args.splits}")

    start_time = datetime.now()
    results = []

    configs = STRATEGY_CONFIGS
    if args.strategy:
        configs = [c for c in configs if c['name'].lower() == args.strategy.lower()]
        if not configs:
            print(f"Strategy '{args.strategy}' not found")
            return

    for config in configs:
        try:
            result = optimize_strategy(config, n_trials=args.trials, n_splits=args.splits)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to optimize {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (datetime.now() - start_time).total_seconds()

    # Print summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 80)

    # Sort by avg test Sharpe
    results.sort(key=lambda x: x['aggregate']['avg_test_sharpe'], reverse=True)

    print(f"\n{'Strategy':<35} {'TF':>4} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'WR':>6}")
    print("-" * 80)

    for r in results:
        agg = r['aggregate']
        print(f"{r['strategy']:<35} {r['timeframe']:>4} "
              f"{agg['avg_test_sharpe']:>8.3f} "
              f"{agg['compounded_return']*100:>9.1f}% "
              f"{agg['avg_max_dd']*100:>7.1f}% "
              f"{agg['total_trades']:>7} "
              f"{agg['avg_win_rate']*100:>5.1f}%")

    # Save results
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Detailed JSON
    json_path = f'{out_dir}/btc_optuna_highperf_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary CSV
    csv_data = []
    for r in results:
        agg = r['aggregate']
        csv_data.append({
            'strategy': r['strategy'],
            'timeframe': r['timeframe'],
            'avg_sharpe': agg['avg_test_sharpe'],
            'compounded_return': agg['compounded_return'],
            'avg_max_dd': agg['avg_max_dd'],
            'total_trades': agg['total_trades'],
            'avg_win_rate': agg['avg_win_rate'],
            'valid_folds': agg['valid_folds'],
        })

    csv_path = f'{out_dir}/btc_optuna_highperf_summary_{ts}.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")

    # Print best parameters for top strategy
    if results:
        best = results[0]
        print(f"\n{'='*80}")
        print(f"BEST STRATEGY: {best['strategy']} @ {best['timeframe']}")
        print(f"{'='*80}")
        print(f"Avg Test Sharpe: {best['aggregate']['avg_test_sharpe']:.3f}")
        print(f"Compounded Return: {best['aggregate']['compounded_return']*100:.1f}%")
        print("\nOptimal parameters per fold:")
        for i, params in enumerate(best['optimal_params']):
            print(f"  Fold {i}: {params}")


if __name__ == '__main__':
    main()
