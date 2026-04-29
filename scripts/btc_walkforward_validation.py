#!/usr/bin/env python3
"""
BTC Walk-Forward Validation

Proper time-series cross-validation for the top 3 BTC strategies:
1. MaestroVWAP_BTC (1d) - Best risk-adjusted
2. MaestroMomentumTrend_BTC (1d) - Best Sharpe
3. MaestroADXSmas_BTC (4h) - Highest returns

Uses rolling walk-forward with:
- 70% train / 30% test per fold
- 5 rolling folds
- Out-of-sample performance tracking
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from engine.vectorized_backtester import backtest_strategy

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data/merged'


def load_data(timeframe: str) -> pd.DataFrame:
    """Load BTC data."""
    filepath = os.path.join(DATA_DIR, f'binance_btc_usdt_{timeframe}.csv')
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Strategy signal generators with optimized params
def vwap_signals(df: pd.DataFrame) -> pd.Series:
    """VWAP mean reversion signals."""
    signals = pd.Series(0, index=df.index)
    std_mult = 1.88

    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap_product = typical * df['volume']

    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        session_start = timestamps.dt.date != timestamps.dt.date.shift(1)
    else:
        session_start = pd.Series(False, index=df.index)
        session_start.iloc[::24] = True

    session_id = session_start.cumsum()
    cum_vol = df.groupby(session_id)['volume'].cumsum()
    cum_vwap_product = vwap_product.groupby(session_id).cumsum()

    vwap = cum_vwap_product / cum_vol.replace(0, 1e-10)
    vwap_std = (df['close'] - vwap).rolling(20, min_periods=5).std().fillna(0)
    upper = vwap + std_mult * vwap_std
    lower = vwap - std_mult * vwap_std

    signals[df['close'] < lower] = 1
    signals[df['close'] > upper] = -1
    return signals


def momentum_trend_signals(df: pd.DataFrame) -> pd.Series:
    """Momentum + Trend Filter signals."""
    signals = pd.Series(0, index=df.index)
    mom_period = 13
    trend_period = 87

    mom = df['close'].pct_change(mom_period)
    sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > sma
    downtrend = df['close'] < sma

    signals[(mom > 0) & uptrend] = 1
    signals[(mom < 0) & downtrend] = -1
    return signals


def adx_smas_signals(df: pd.DataFrame) -> pd.Series:
    """ADX + SMAs signals."""
    signals = pd.Series(0, index=df.index)
    adx_period = 12
    sma_fast = 11
    sma_slow = 79
    adx_threshold = 39.0

    # ADX calculation
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean()

    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    plus_di = 100 * plus_dm.rolling(adx_period).mean() / atr
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(adx_period).mean()

    # SMAs
    sma_fast_line = close.rolling(sma_fast).mean()
    sma_slow_line = close.rolling(sma_slow).mean()

    trending = adx > adx_threshold
    signals[trending & (sma_fast_line > sma_slow_line)] = 1
    signals[trending & (sma_fast_line < sma_slow_line)] = -1

    return signals


def walk_forward_split(df: pd.DataFrame, n_splits: int = 5, train_ratio: float = 0.7):
    """Generate walk-forward splits."""
    n = len(df)
    fold_size = n // n_splits

    for i in range(n_splits):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)

        fold_data = df.iloc[start:end].copy()
        train_size = int(len(fold_data) * train_ratio)

        train = fold_data.iloc[:train_size].reset_index(drop=True)
        test = fold_data.iloc[train_size:].reset_index(drop=True)

        if len(train) > 50 and len(test) > 20:
            yield i, train, test


def run_walkforward(name: str, signal_func, timeframe: str, n_splits: int = 5):
    """Run walk-forward validation for a strategy."""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD: {name} @ {timeframe}")
    print(f"{'='*70}")

    df = load_data(timeframe)
    print(f"Data: {len(df):,} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")

    results = []

    for fold_idx, train_df, test_df in walk_forward_split(df, n_splits):
        print(f"\n--- Fold {fold_idx} ---")
        print(f"  Train: {len(train_df):,} bars ({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]})")
        print(f"  Test:  {len(test_df):,} bars ({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})")

        # Train performance
        try:
            train_result = backtest_strategy(
                train_df, signal_func, name,
                timeframe=timeframe, position_mode='fixed', position_size=0.1
            )
            train_sharpe = train_result.get('sharpe_ratio', 0)
            train_return = train_result.get('total_return', 0)
        except Exception as e:
            train_sharpe = train_return = 0

        # Test performance (out-of-sample)
        try:
            test_result = backtest_strategy(
                test_df, signal_func, name,
                timeframe=timeframe, position_mode='fixed', position_size=0.1
            )
            test_sharpe = test_result.get('sharpe_ratio', 0) or 0
            test_return = test_result.get('total_return', 0) or 0
            test_dd = test_result.get('max_drawdown', 0) or 0
            test_trades = test_result.get('total_trades', 0) or 0
            test_wr = test_result.get('win_rate', 0) or 0
        except Exception as e:
            test_sharpe = test_return = test_dd = test_trades = test_wr = 0

        print(f"  Train: Sharpe={train_sharpe:.3f}, Return={train_return*100:.1f}%")
        print(f"  Test:  Sharpe={test_sharpe:.3f}, Return={test_return*100:.1f}%, MaxDD={test_dd*100:.1f}%, Trades={test_trades}")

        results.append({
            'fold': fold_idx,
            'train_sharpe': train_sharpe,
            'train_return': train_return,
            'test_sharpe': test_sharpe,
            'test_return': test_return,
            'test_max_dd': test_dd,
            'test_trades': test_trades,
            'test_win_rate': test_wr,
        })

    # Aggregate results
    valid = [r for r in results if r['test_trades'] > 0]

    if valid:
        avg_sharpe = np.mean([r['test_sharpe'] for r in valid])
        avg_return = np.mean([r['test_return'] for r in valid])
        compounded = np.prod([1 + r['test_return'] for r in valid]) - 1
        avg_dd = np.mean([r['test_max_dd'] for r in valid])
        total_trades = sum(r['test_trades'] for r in valid)
        avg_wr = np.mean([r['test_win_rate'] for r in valid])
        consistency = sum(1 for r in valid if r['test_sharpe'] > 0) / len(valid)
    else:
        avg_sharpe = avg_return = compounded = avg_dd = total_trades = avg_wr = consistency = 0

    summary = {
        'strategy': name,
        'timeframe': timeframe,
        'avg_test_sharpe': avg_sharpe,
        'avg_test_return': avg_return,
        'compounded_return': compounded,
        'avg_max_dd': avg_dd,
        'total_trades': total_trades,
        'avg_win_rate': avg_wr,
        'consistency': consistency,
        'valid_folds': len(valid),
        'folds': results
    }

    print(f"\n{'='*50}")
    print(f"SUMMARY: {name}")
    print(f"{'='*50}")
    print(f"  Avg OOS Sharpe:      {avg_sharpe:.3f}")
    print(f"  Avg OOS Return:      {avg_return*100:.1f}%")
    print(f"  Compounded Return:   {compounded*100:.1f}%")
    print(f"  Avg Max Drawdown:    {avg_dd*100:.1f}%")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Avg Win Rate:        {avg_wr*100:.1f}%")
    print(f"  Consistency:         {consistency*100:.0f}% folds profitable")

    return summary


def main():
    print("=" * 80)
    print("BTC WALK-FORWARD VALIDATION")
    print("Top 3 Optimized Strategies")
    print("=" * 80)

    start_time = datetime.now()

    strategies = [
        ('MaestroVWAP_BTC', vwap_signals, '1d'),
        ('MaestroMomentumTrend_BTC', momentum_trend_signals, '1d'),
        ('MaestroADXSmas_BTC', adx_smas_signals, '4h'),
    ]

    all_results = []

    for name, signal_func, timeframe in strategies:
        result = run_walkforward(name, signal_func, timeframe, n_splits=5)
        all_results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 80)

    # Sort by consistency * sharpe (risk-adjusted ranking)
    all_results.sort(key=lambda x: x['consistency'] * x['avg_test_sharpe'], reverse=True)

    print(f"\n{'Strategy':<30} {'TF':>4} {'Sharpe':>8} {'Return':>10} {'DD':>8} {'WR':>6} {'Consist':>8}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['strategy']:<30} {r['timeframe']:>4} "
              f"{r['avg_test_sharpe']:>8.3f} "
              f"{r['compounded_return']*100:>9.1f}% "
              f"{r['avg_max_dd']*100:>7.1f}% "
              f"{r['avg_win_rate']*100:>5.1f}% "
              f"{r['consistency']*100:>7.0f}%")

    # Save results
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = f'{out_dir}/btc_walkforward_validation_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV summary
    csv_data = [{
        'strategy': r['strategy'],
        'timeframe': r['timeframe'],
        'avg_sharpe': r['avg_test_sharpe'],
        'compounded_return': r['compounded_return'],
        'avg_max_dd': r['avg_max_dd'],
        'total_trades': r['total_trades'],
        'avg_win_rate': r['avg_win_rate'],
        'consistency': r['consistency'],
    } for r in all_results]

    csv_path = f'{out_dir}/btc_walkforward_summary_{ts}.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")

    # Recommendation
    best = all_results[0]
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print(f"Best Strategy: {best['strategy']} @ {best['timeframe']}")
    print(f"  - OOS Sharpe: {best['avg_test_sharpe']:.3f}")
    print(f"  - Compounded Return: {best['compounded_return']*100:.1f}%")
    print(f"  - Consistency: {best['consistency']*100:.0f}% of folds profitable")


if __name__ == '__main__':
    main()
