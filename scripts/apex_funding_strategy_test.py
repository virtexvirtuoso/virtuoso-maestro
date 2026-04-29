#!/usr/bin/env python3
"""
Apex Funding Rate Arbitrage Strategy Test

Original thresholds from STRATEGIES_ROADMAP.md:
- LONG when funding < -0.01% (-0.0001)
- SHORT when funding > +0.05% (+0.0005)
- EXIT when funding normalizes to ±0.01%

Testing with REAL funding rate data to validate.
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


def apex_strategy_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Apex Funding Rate Arbitrage signals.

    Original logic:
    - LONG when funding < long_threshold
    - SHORT when funding > short_threshold
    - EXIT when funding normalizes to ±exit_threshold
    """
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']

    long_thresh = params['long_threshold']
    short_thresh = params['short_threshold']
    exit_thresh = params['exit_threshold']

    current_pos = 0

    for i in range(len(df)):
        fr = funding.iloc[i]

        # Check exit first
        if current_pos != 0:
            if abs(fr) <= exit_thresh:
                signals.iloc[i] = 0
                current_pos = 0
                continue

        # Entry signals
        if fr < long_thresh:
            signals.iloc[i] = 1
            current_pos = 1
        elif fr > short_thresh:
            signals.iloc[i] = -1
            current_pos = -1
        else:
            signals.iloc[i] = current_pos  # Hold position

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

        # P&L from holding position
        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size
            trade_pnl += pnl.iloc[i]

        # Position change
        if signal != current_pos:
            if current_pos != 0:
                # Close previous position
                pnl.iloc[i] -= commission * abs(current_pos) * position_size
                total_trades += 1
                if trade_pnl > 0:
                    wins += 1
                trade_pnl = 0.0

            if signal != 0:
                # Open new position
                pnl.iloc[i] -= commission * abs(signal) * position_size

            current_pos = signal

    # Metrics
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


def walk_forward_test(df: pd.DataFrame, params: dict, n_splits: int = 5) -> dict:
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

        signals = apex_strategy_signals(test_df, params)
        result = backtest(test_df, signals)

        sharpes.append(result['sharpe_ratio'])
        returns.append(result['total_return'])
        trades.append(result['total_trades'])

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
    print("APEX FUNDING RATE ARBITRAGE - Real Data Test")
    print("=" * 80)

    df = load_data()
    print(f"\nData: {len(df):,} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")

    # Funding rate statistics
    print(f"\nFunding Rate Statistics:")
    print(f"  Mean: {df['funding_rate'].mean()*100:.4f}%")
    print(f"  Std:  {df['funding_rate'].std()*100:.4f}%")
    print(f"  Min:  {df['funding_rate'].min()*100:.4f}%")
    print(f"  Max:  {df['funding_rate'].max()*100:.4f}%")

    # Count signals at Apex thresholds
    apex_long = (df['funding_rate'] < -0.0001).sum()
    apex_short = (df['funding_rate'] > 0.0005).sum()
    print(f"\n  Bars with funding < -0.01%: {apex_long} ({apex_long/len(df)*100:.1f}%)")
    print(f"  Bars with funding > +0.05%: {apex_short} ({apex_short/len(df)*100:.1f}%)")

    # Define strategies to test
    strategies = {
        'Apex Original': {
            'long_threshold': -0.0001,   # -0.01%
            'short_threshold': 0.0005,   # +0.05%
            'exit_threshold': 0.0001,    # ±0.01%
        },
        'Apex Tighter': {
            'long_threshold': -0.0003,   # -0.03%
            'short_threshold': 0.001,    # +0.10%
            'exit_threshold': 0.0001,    # ±0.01%
        },
        'Maestro Optimized': {
            'long_threshold': -0.000691,  # Our optimized
            'short_threshold': 0.001086,
            'exit_threshold': 0.0001,
        },
        'Conservative': {
            'long_threshold': -0.001,     # -0.10%
            'short_threshold': 0.002,     # +0.20%
            'exit_threshold': 0.0002,
        },
    }

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (5 folds)")
    print("=" * 80)

    results = []

    for name, params in strategies.items():
        result = walk_forward_test(df, params)
        results.append({'strategy': name, **result, 'params': params})

        print(f"\n{name}:")
        print(f"  Thresholds: L<{params['long_threshold']*100:.3f}%, S>{params['short_threshold']*100:.3f}%")
        print(f"  Avg Sharpe: {result['avg_sharpe']:.2f}")
        print(f"  Return: {result['compounded_return']*100:.2f}%")
        print(f"  Consistency: {result['consistency']*100:.0f}%")
        print(f"  Avg Trades: {result['avg_trades']:.0f}")
        print(f"  Fold Sharpes: {[f'{s:.1f}' for s in result['fold_sharpes']]}")

    # Full backtest comparison
    print("\n" + "=" * 80)
    print("FULL PERIOD BACKTEST")
    print("=" * 80)

    print(f"\n{'Strategy':<25} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Trades':>10}")
    print("-" * 70)

    for name, params in strategies.items():
        signals = apex_strategy_signals(df, params)
        result = backtest(df, signals)
        print(f"{name:<25} {result['sharpe_ratio']:>10.2f} {result['total_return']*100:>11.2f}% "
              f"{result['max_drawdown']*100:>9.2f}% {result['total_trades']:>10}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best = max(results, key=lambda x: x['avg_sharpe'])
    print(f"\nBest Strategy: {best['strategy']}")
    print(f"  Avg Sharpe: {best['avg_sharpe']:.2f}")
    print(f"  Return: {best['compounded_return']*100:.2f}%")
    print(f"  Consistency: {best['consistency']*100:.0f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_bars': len(df),
        'funding_stats': {
            'mean': float(df['funding_rate'].mean()),
            'std': float(df['funding_rate'].std()),
            'min': float(df['funding_rate'].min()),
            'max': float(df['funding_rate'].max()),
        },
        'results': results
    }

    json_path = f'{DATA_DIR}/backtest_results/apex_strategy_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
