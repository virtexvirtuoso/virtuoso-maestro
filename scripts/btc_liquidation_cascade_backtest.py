#!/usr/bin/env python3
"""
BTC Liquidation Cascade Strategy Backtest

Liquidation cascades occur when:
1. Price moves sharply against leveraged positions
2. Forced liquidations trigger more price movement
3. Chain reaction accelerates the move

Detection signals:
1. Volume spike + directional price move
2. ATR expansion (volatility spike)
3. OI drop (positions being liquidated)
4. Extreme taker buy/sell ratio

Strategy:
- ENTER: When cascade detected, trade WITH the cascade
- EXIT: When cascade exhausts (volatility normalizes)
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

DATA_DIR = '/Users/ffv_macmini/Desktop/maestro/data'
OHLCV_PATH = f'{DATA_DIR}/merged/binance_btc_usdt_4h.csv'
FUNDING_PATH = f'{DATA_DIR}/derivatives/btc_funding.csv'


def load_data() -> pd.DataFrame:
    """Load OHLCV data with synthetic liquidation proxy."""
    df = pd.read_csv(OHLCV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Filter to period with reasonable data
    df = df[df['timestamp'] >= '2023-05-01'].reset_index(drop=True)

    # Calculate indicators for liquidation detection
    # 1. Volume metrics
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)

    # 2. Volatility (ATR)
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(20).mean()
    df['atr_ratio'] = df['atr'] / df['atr_ma'].replace(0, 1)

    # 3. Price move magnitude
    df['return'] = df['close'].pct_change()
    df['return_abs'] = df['return'].abs()
    df['return_ma'] = df['return_abs'].rolling(20).mean()
    df['return_ratio'] = df['return_abs'] / df['return_ma'].replace(0, 1e-10)

    # 4. Synthetic liquidation proxy
    # High volume + high volatility + directional move = likely liquidations
    df['liq_pressure'] = (
        df['volume_ratio'] * 0.3 +
        df['atr_ratio'] * 0.3 +
        df['return_ratio'] * 0.4
    )

    # Direction of liquidations
    df['liq_direction'] = np.sign(df['return'])

    # Cascade detection: pressure > threshold + sustained direction
    df['liq_pressure_ma'] = df['liq_pressure'].rolling(3).mean()

    print(f"Data: {len(df):,} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


# =============================================================================
# STRATEGY 1: Cascade Momentum (Trade WITH the cascade)
# =============================================================================
def cascade_momentum_signals(
    df: pd.DataFrame,
    pressure_threshold: float = 2.0,   # Volume/vol spike threshold
    min_move: float = 0.02,            # Minimum 2% move to trigger
    cooldown: int = 6                   # Bars to wait after cascade
) -> pd.Series:
    """
    Trade WITH liquidation cascades.

    Logic: When cascade starts, momentum continues until exhaustion.
    """
    signals = pd.Series(0, index=df.index)
    n = len(df)

    # Cascade detection
    cascade_up = (
        (df['liq_pressure'] > pressure_threshold) &
        (df['return'] > min_move) &
        (df['volume_ratio'] > 1.5)
    )
    cascade_down = (
        (df['liq_pressure'] > pressure_threshold) &
        (df['return'] < -min_move) &
        (df['volume_ratio'] > 1.5)
    )

    # Follow the cascade
    signals[cascade_up] = 1   # Long: short squeeze
    signals[cascade_down] = -1  # Short: long liquidation

    return signals


# =============================================================================
# STRATEGY 2: Cascade Exhaustion (Fade the cascade)
# =============================================================================
def cascade_exhaustion_signals(
    df: pd.DataFrame,
    pressure_threshold: float = 2.5,   # High threshold for exhaustion
    exhaustion_bars: int = 3,          # Bars of high pressure before fade
    min_move: float = 0.03             # Min 3% move for exhaustion
) -> pd.Series:
    """
    Fade liquidation cascades after exhaustion.

    Logic: After extreme cascade, positions exhausted → mean reversion.
    """
    signals = pd.Series(0, index=df.index)

    # Rolling max pressure in last N bars
    rolling_pressure = df['liq_pressure'].rolling(exhaustion_bars).max()

    # Recent directional move
    recent_return = df['close'].pct_change(exhaustion_bars)

    # Exhaustion conditions
    exhaustion_up = (
        (rolling_pressure > pressure_threshold) &
        (recent_return > min_move) &
        (df['liq_pressure'] < df['liq_pressure'].shift(1))  # Pressure declining
    )

    exhaustion_down = (
        (rolling_pressure > pressure_threshold) &
        (recent_return < -min_move) &
        (df['liq_pressure'] < df['liq_pressure'].shift(1))
    )

    # Fade the move (contrarian)
    signals[exhaustion_up] = -1   # Short after long squeeze
    signals[exhaustion_down] = 1  # Long after short squeeze

    return signals


# =============================================================================
# STRATEGY 3: Funding + Liquidation Combined
# =============================================================================
def funding_liquidation_combined_signals(
    df: pd.DataFrame,
    pressure_threshold: float = 1.8,
    funding_extreme: float = 0.0005
) -> pd.Series:
    """
    Combine funding rate + liquidation signals.

    Logic:
    - Extreme funding = crowded position
    - High liquidation pressure = forced unwinding
    - Both together = stronger signal
    """
    signals = pd.Series(0, index=df.index)

    # Load funding if available
    try:
        funding = pd.read_csv(FUNDING_PATH)
        funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])
        funding = funding.rename(columns={'fundingRate': 'funding_rate'})

        # Merge
        df_merged = pd.merge_asof(
            df.reset_index(),
            funding[['timestamp', 'funding_rate']],
            on='timestamp',
            direction='backward'
        ).set_index('index')

        fr = df_merged['funding_rate'].fillna(0)
    except:
        # Fallback to synthetic
        fr = pd.Series(0, index=df.index)

    # Combined signals
    # High liq pressure + extreme negative funding = short squeeze → LONG
    squeeze = (
        (df['liq_pressure'] > pressure_threshold) &
        (fr < -funding_extreme) &
        (df['return'] > 0)
    )

    # High liq pressure + extreme positive funding = long liquidation → SHORT
    crash = (
        (df['liq_pressure'] > pressure_threshold) &
        (fr > funding_extreme) &
        (df['return'] < 0)
    )

    signals[squeeze] = 1
    signals[crash] = -1

    return signals


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    position_size: float = 0.1,
    commission: float = 0.0004
) -> dict:
    """Simple backtest."""
    n = len(df)
    pnl = pd.Series(0.0, index=df.index)

    current_pos = 0.0
    total_trades = 0
    wins = 0

    for i in range(1, n):
        signal = signals.iloc[i-1]
        price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]

        # P&L
        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size

        # Signal processing
        if signal != 0 and signal != current_pos:
            if current_pos != 0:
                pnl.iloc[i] -= commission * abs(current_pos) * position_size
                if pnl.iloc[i] > 0:
                    wins += 1
            current_pos = signal
            total_trades += 1
            pnl.iloc[i] -= commission * abs(signal) * position_size

        elif signal == 0 and current_pos != 0:
            pnl.iloc[i] -= commission * abs(current_pos) * position_size
            if pnl.iloc[i] > 0:
                wins += 1
            current_pos = 0.0
            total_trades += 1

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

    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'total_return': cumulative.iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate
    }


def walk_forward_split(df: pd.DataFrame, n_splits: int = 5, train_ratio: float = 0.7):
    """Walk-forward splits."""
    n = len(df)
    fold_size = n // n_splits

    for i in range(n_splits):
        start = i * fold_size
        end = min((i + 2) * fold_size, n)

        fold_data = df.iloc[start:end].copy()
        train_size = int(len(fold_data) * train_ratio)

        train = fold_data.iloc[:train_size].reset_index(drop=True)
        test = fold_data.iloc[train_size:].reset_index(drop=True)

        if len(train) > 100 and len(test) > 50:
            yield i, train, test


def run_walkforward(name: str, signal_func, df: pd.DataFrame, **kwargs):
    """Run walk-forward validation."""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD: {name}")
    print(f"{'='*70}")

    results = []

    for fold_idx, train_df, test_df in walk_forward_split(df, n_splits=5):
        train_signals = signal_func(train_df, **kwargs)
        test_signals = signal_func(test_df, **kwargs)

        train_result = backtest(train_df, train_signals)
        test_result = backtest(test_df, test_signals)

        print(f"Fold {fold_idx}: Train Sharpe={train_result['sharpe_ratio']:.3f}, "
              f"Test Sharpe={test_result['sharpe_ratio']:.3f}, "
              f"Test Return={test_result['total_return']*100:.1f}%")

        results.append({
            'fold': fold_idx,
            'train_sharpe': train_result['sharpe_ratio'],
            'train_return': train_result['total_return'],
            'test_sharpe': test_result['sharpe_ratio'],
            'test_return': test_result['total_return'],
            'test_max_dd': test_result['max_drawdown'],
            'test_trades': test_result['total_trades'],
            'test_win_rate': test_result['win_rate'],
        })

    valid = [r for r in results if r['test_trades'] > 0]

    if valid:
        avg_sharpe = np.mean([r['test_sharpe'] for r in valid])
        compounded = np.prod([1 + r['test_return'] for r in valid]) - 1
        avg_dd = np.mean([r['test_max_dd'] for r in valid])
        total_trades = sum(r['test_trades'] for r in valid)
        avg_wr = np.mean([r['test_win_rate'] for r in valid])
        consistency = sum(1 for r in valid if r['test_sharpe'] > 0) / len(valid)
    else:
        avg_sharpe = compounded = avg_dd = total_trades = avg_wr = consistency = 0

    print(f"\nSUMMARY: Sharpe={avg_sharpe:.3f}, Return={compounded*100:.1f}%, "
          f"Consistency={consistency*100:.0f}%")

    return {
        'strategy': name,
        'avg_test_sharpe': avg_sharpe,
        'compounded_return': compounded,
        'avg_max_dd': avg_dd,
        'total_trades': total_trades,
        'avg_win_rate': avg_wr,
        'consistency': consistency,
        'folds': results
    }


def main():
    print("=" * 80)
    print("BTC LIQUIDATION CASCADE STRATEGY BACKTEST")
    print("=" * 80)

    df = load_data()

    strategies = [
        ('CascadeMomentum', cascade_momentum_signals,
         {'pressure_threshold': 2.0, 'min_move': 0.02}),
        ('CascadeExhaustion', cascade_exhaustion_signals,
         {'pressure_threshold': 2.5, 'min_move': 0.03}),
        ('FundingLiqCombined', funding_liquidation_combined_signals,
         {'pressure_threshold': 1.8, 'funding_extreme': 0.0005}),
    ]

    all_results = []
    for name, func, kwargs in strategies:
        result = run_walkforward(name, func, df, **kwargs)
        all_results.append(result)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    all_results.sort(key=lambda x: x['avg_test_sharpe'], reverse=True)

    print(f"\n{'Strategy':<25} {'Sharpe':>8} {'Return':>10} {'DD':>8} {'Consist':>8}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['avg_test_sharpe']:>8.3f} "
              f"{r['compounded_return']*100:>9.1f}% {r['avg_max_dd']*100:>7.1f}% "
              f"{r['consistency']*100:>7.0f}%")

    # Save
    out_dir = f'{DATA_DIR}/backtest_results'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{out_dir}/btc_liquidation_backtest_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
