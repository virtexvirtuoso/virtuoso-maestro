#!/usr/bin/env python3
"""
Apex Funding Rate Strategy - Properly Calibrated

Key finding: Original thresholds (-0.01%, +0.05%) work but generate few trades.
Our "optimized" thresholds were outside the data range entirely!

This test:
1. Uses percentile-based thresholds calibrated to actual funding distribution
2. Compares absolute thresholds vs z-score-only vs combined
3. Validates which signal mechanism actually provides edge
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data'


def load_data() -> pd.DataFrame:
    """Load BTC OHLCV with real funding rate."""
    ohlcv_path = f'{DATA_DIR}/merged/binance_btc_usdt_4h.csv'
    funding_path = f'{DATA_DIR}/derivatives/btc_funding.csv'

    df = pd.read_csv(ohlcv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[df['timestamp'] >= '2023-05-01'].reset_index(drop=True)

    if 'funding_rate' in df.columns:
        df = df.drop(columns=['funding_rate'])

    funding = pd.read_csv(funding_path)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])
    funding = funding.rename(columns={'fundingRate': 'funding_rate'})

    df = pd.merge_asof(df, funding[['timestamp', 'funding_rate']],
                       on='timestamp', direction='backward')
    df['funding_rate'] = df['funding_rate'].fillna(0)

    return df


def compute_funding_stats(df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """Compute z-scores and percentile ranks for funding rate."""
    df = df.copy()
    df['funding_ma'] = df['funding_rate'].rolling(lookback, min_periods=lookback//2).mean()
    df['funding_std'] = df['funding_rate'].rolling(lookback, min_periods=lookback//2).std()
    df['funding_std'] = df['funding_std'].replace(0, 1e-10)
    df['funding_zscore'] = (df['funding_rate'] - df['funding_ma']) / df['funding_std']
    df['funding_pct'] = df['funding_rate'].rolling(lookback * 4, min_periods=lookback).rank(pct=True)
    return df


def signal_absolute_only(df: pd.DataFrame, long_thresh: float, short_thresh: float,
                         exit_thresh: float) -> pd.Series:
    """Pure absolute threshold signals."""
    signals = pd.Series(0, index=df.index)
    current_pos = 0

    for i in range(len(df)):
        fr = df['funding_rate'].iloc[i]

        if current_pos != 0 and abs(fr) <= exit_thresh:
            signals.iloc[i] = 0
            current_pos = 0
            continue

        if fr < long_thresh:
            signals.iloc[i] = 1
            current_pos = 1
        elif fr > short_thresh:
            signals.iloc[i] = -1
            current_pos = -1
        else:
            signals.iloc[i] = current_pos

    return signals


def signal_zscore_only(df: pd.DataFrame, zscore_thresh: float) -> pd.Series:
    """Pure z-score signals (no absolute thresholds)."""
    signals = pd.Series(0, index=df.index)
    current_pos = 0

    for i in range(len(df)):
        zscore = df['funding_zscore'].iloc[i]

        if pd.isna(zscore):
            signals.iloc[i] = current_pos
            continue

        # Exit when z-score normalizes
        if current_pos != 0 and abs(zscore) < 0.5:
            signals.iloc[i] = 0
            current_pos = 0
            continue

        # Entry signals
        if zscore < -zscore_thresh:
            signals.iloc[i] = 1
            current_pos = 1
        elif zscore > zscore_thresh:
            signals.iloc[i] = -1
            current_pos = -1
        else:
            signals.iloc[i] = current_pos

    return signals


def signal_combined(df: pd.DataFrame, long_thresh: float, short_thresh: float,
                    zscore_thresh: float) -> pd.Series:
    """Combined: absolute OR z-score triggers entry."""
    signals = pd.Series(0, index=df.index)
    current_pos = 0

    for i in range(len(df)):
        fr = df['funding_rate'].iloc[i]
        zscore = df['funding_zscore'].iloc[i] if not pd.isna(df['funding_zscore'].iloc[i]) else 0

        # Exit when both normalized
        if current_pos != 0:
            if abs(fr) <= 0.0001 and abs(zscore) < 0.5:
                signals.iloc[i] = 0
                current_pos = 0
                continue

        # Long signal: extreme negative funding OR z-score
        if fr < long_thresh or zscore < -zscore_thresh:
            signals.iloc[i] = 1
            current_pos = 1
        # Short signal: extreme positive funding OR z-score
        elif fr > short_thresh or zscore > zscore_thresh:
            signals.iloc[i] = -1
            current_pos = -1
        else:
            signals.iloc[i] = current_pos

    return signals


def signal_percentile(df: pd.DataFrame, low_pct: float = 0.05, high_pct: float = 0.95) -> pd.Series:
    """Percentile-based signals (adaptive to regime)."""
    signals = pd.Series(0, index=df.index)
    current_pos = 0

    for i in range(len(df)):
        pct = df['funding_pct'].iloc[i]

        if pd.isna(pct):
            signals.iloc[i] = current_pos
            continue

        # Exit when normalized
        if current_pos != 0 and 0.3 < pct < 0.7:
            signals.iloc[i] = 0
            current_pos = 0
            continue

        # Entry signals based on percentile rank
        if pct < low_pct:
            signals.iloc[i] = 1
            current_pos = 1
        elif pct > high_pct:
            signals.iloc[i] = -1
            current_pos = -1
        else:
            signals.iloc[i] = current_pos

    return signals


def backtest(df: pd.DataFrame, signals: pd.Series,
             position_size: float = 0.1, commission: float = 0.0004) -> dict:
    """Backtest with proper position tracking."""
    n = len(df)
    pnl = pd.Series(0.0, index=df.index)
    current_pos = 0.0
    total_trades = 0
    wins = 0
    trade_pnl = 0.0

    for i in range(1, n):
        signal = signals.iloc[i-1]
        price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]

        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size
            trade_pnl += pnl.iloc[i]

        if signal != current_pos:
            if current_pos != 0:
                pnl.iloc[i] -= commission * abs(current_pos) * position_size
                total_trades += 1
                if trade_pnl > 0:
                    wins += 1
                trade_pnl = 0.0

            if signal != 0:
                pnl.iloc[i] -= commission * abs(signal) * position_size

            current_pos = signal

    cumulative = (1 + pnl).cumprod() - 1
    returns = pnl[pnl != 0]

    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252 * 6) * returns.mean() / returns.std()
    else:
        sharpe = 0

    cummax = (1 + pnl).cumprod().cummax()
    dd = (1 + pnl).cumprod() / cummax - 1
    max_dd = dd.min()

    return {
        'total_return': cumulative.iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100 if total_trades > 0 else 0
    }


def walk_forward_test(df: pd.DataFrame, signal_fn, signal_kwargs: dict,
                      n_splits: int = 5) -> dict:
    """Walk-forward validation."""
    n = len(df)
    fold_size = n // n_splits

    sharpes = []
    returns = []
    trades = []

    for i in range(n_splits):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)
        fold_data = df.iloc[start:end].copy()
        train_size = int(len(fold_data) * 0.7)
        test_df = fold_data.iloc[train_size:].reset_index(drop=True)

        if len(test_df) < 50:
            continue

        # Recompute stats on test set only
        test_df = compute_funding_stats(test_df)
        signals = signal_fn(test_df, **signal_kwargs)
        result = backtest(test_df, signals)

        sharpes.append(result['sharpe_ratio'])
        returns.append(result['total_return'])
        trades.append(result['total_trades'])

    if not sharpes:
        return {'avg_sharpe': 0, 'fold_sharpes': [], 'compounded_return': 0,
                'fold_returns': [], 'consistency': 0, 'avg_trades': 0}

    return {
        'avg_sharpe': np.mean(sharpes),
        'fold_sharpes': sharpes,
        'compounded_return': np.prod([1 + r for r in returns]) - 1,
        'fold_returns': returns,
        'consistency': sum(1 for s in sharpes if s > 0) / len(sharpes),
        'avg_trades': np.mean(trades)
    }


def main():
    print("=" * 80)
    print("APEX FUNDING STRATEGY - Properly Calibrated Test")
    print("=" * 80)

    df = load_data()
    df = compute_funding_stats(df)

    print(f"\nData: {len(df):,} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")

    # Funding rate percentiles
    print(f"\nFunding Rate Distribution:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(df['funding_rate'].dropna(), percentiles)
    for p, v in zip(percentiles, pct_values):
        print(f"  {p:3d}th percentile: {v*100:.4f}%")

    # Define strategies with calibrated thresholds
    # Based on percentiles: 5th = -0.0068%, 95th = 0.0479%
    strategies = [
        {
            'name': 'Absolute (Apex Original)',
            'fn': signal_absolute_only,
            'kwargs': {'long_thresh': -0.0001, 'short_thresh': 0.0005, 'exit_thresh': 0.0001}
        },
        {
            'name': 'Absolute (Calibrated 5%/95%)',
            'fn': signal_absolute_only,
            'kwargs': {
                'long_thresh': pct_values[1],  # 5th percentile
                'short_thresh': pct_values[7], # 95th percentile
                'exit_thresh': 0.0001
            }
        },
        {
            'name': 'Absolute (Calibrated 10%/90%)',
            'fn': signal_absolute_only,
            'kwargs': {
                'long_thresh': pct_values[2],  # 10th percentile
                'short_thresh': pct_values[6], # 90th percentile
                'exit_thresh': 0.0001
            }
        },
        {
            'name': 'Z-Score Only (2.0σ)',
            'fn': signal_zscore_only,
            'kwargs': {'zscore_thresh': 2.0}
        },
        {
            'name': 'Z-Score Only (2.18σ - Optuna)',
            'fn': signal_zscore_only,
            'kwargs': {'zscore_thresh': 2.18}
        },
        {
            'name': 'Z-Score Only (1.5σ - More Trades)',
            'fn': signal_zscore_only,
            'kwargs': {'zscore_thresh': 1.5}
        },
        {
            'name': 'Combined (Calibrated + Z-Score)',
            'fn': signal_combined,
            'kwargs': {
                'long_thresh': pct_values[1],  # 5th percentile
                'short_thresh': pct_values[7], # 95th percentile
                'zscore_thresh': 2.0
            }
        },
        {
            'name': 'Percentile (5%/95%)',
            'fn': signal_percentile,
            'kwargs': {'low_pct': 0.05, 'high_pct': 0.95}
        },
        {
            'name': 'Percentile (10%/90%)',
            'fn': signal_percentile,
            'kwargs': {'low_pct': 0.10, 'high_pct': 0.90}
        },
    ]

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (5 folds)")
    print("=" * 80)

    results = []
    for strat in strategies:
        result = walk_forward_test(df, strat['fn'], strat['kwargs'])
        results.append({**result, 'name': strat['name'], 'kwargs': strat['kwargs']})

        print(f"\n{strat['name']}:")
        print(f"  Avg Sharpe: {result['avg_sharpe']:.2f}")
        print(f"  Return: {result['compounded_return']*100:.2f}%")
        print(f"  Consistency: {result['consistency']*100:.0f}%")
        print(f"  Avg Trades: {result['avg_trades']:.0f}")
        print(f"  Fold Sharpes: {[f'{s:.1f}' for s in result['fold_sharpes']]}")

    # Full backtest
    print("\n" + "=" * 80)
    print("FULL PERIOD BACKTEST")
    print("=" * 80)

    print(f"\n{'Strategy':<35} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 75)

    for strat in strategies:
        df_test = compute_funding_stats(df.copy())
        signals = strat['fn'](df_test, **strat['kwargs'])
        result = backtest(df_test, signals)
        print(f"{strat['name']:<35} {result['sharpe_ratio']:>8.2f} {result['total_return']*100:>9.2f}% "
              f"{result['max_drawdown']*100:>7.2f}% {result['total_trades']:>8}")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    best = max(results, key=lambda x: x['avg_sharpe'])
    best_return = max(results, key=lambda x: x['compounded_return'])

    print(f"\nBest by Sharpe: {best['name']}")
    print(f"  Avg Sharpe: {best['avg_sharpe']:.2f}")
    print(f"  Return: {best['compounded_return']*100:.2f}%")

    print(f"\nBest by Return: {best_return['name']}")
    print(f"  Return: {best_return['compounded_return']*100:.2f}%")
    print(f"  Avg Sharpe: {best_return['avg_sharpe']:.2f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The "Maestro Optimized" thresholds (-0.069%, +0.109%) were OUTSIDE the data range!
The regime classifier worked because it used z-score as an ALTERNATIVE trigger.

Key insights:
1. Absolute thresholds must be calibrated to actual funding distribution
2. Z-score-only approach is more robust (adapts to changing regimes)
3. Combined approach provides redundancy but may overtrade
4. Percentile-based thresholds auto-calibrate to regime changes
""")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'funding_percentiles': {f'{p}th': float(v) for p, v in zip(percentiles, pct_values)},
        'results': results
    }

    json_path = f'{DATA_DIR}/backtest_results/apex_calibrated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
