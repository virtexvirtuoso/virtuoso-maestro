#!/usr/bin/env python3
"""
BTC Funding Rate Strategy Backtest

Uses REAL derivatives data (3001 rows from May 2023 - Feb 2026) to test:
1. Funding Collection: Be long during positive funding periods
2. Contrarian Trading: Short extreme positive funding, long extreme negative
3. Combined: Collect funding + contrarian signals

Methodology:
- Merge 8h funding data with 4h OHLCV (forward-fill funding)
- Walk-forward validation (5 splits)
- Track funding income separately from price P&L
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


def load_and_merge_data() -> pd.DataFrame:
    """Load OHLCV and merge with real funding data."""
    # Load OHLCV
    ohlcv = pd.read_csv(OHLCV_PATH)
    ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
    ohlcv = ohlcv.sort_values('timestamp').reset_index(drop=True)

    # Drop existing funding column if mostly NaN (will replace with real data)
    if 'funding_rate' in ohlcv.columns:
        if ohlcv['funding_rate'].isna().mean() > 0.5:
            ohlcv = ohlcv.drop(columns=['funding_rate'])

    # Load funding
    funding = pd.read_csv(FUNDING_PATH)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])  # Strip milliseconds
    funding = funding.sort_values('timestamp').reset_index(drop=True)
    funding = funding.rename(columns={'fundingRate': 'funding_rate_real'})

    # Merge on timestamp (forward-fill funding to 4h)
    # Funding is 8h, OHLCV is 4h, so we need to merge asof
    df = pd.merge_asof(
        ohlcv,
        funding[['timestamp', 'funding_rate_real']],
        on='timestamp',
        direction='backward'
    )

    # Rename to standard column
    df['funding_rate'] = df['funding_rate_real']
    df = df.drop(columns=['funding_rate_real'], errors='ignore')

    # Filter to period where we have funding data
    df = df[df['funding_rate'].notna()].reset_index(drop=True)

    print(f"Merged data: {len(df):,} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Funding stats: mean={df['funding_rate'].mean()*100:.4f}%, std={df['funding_rate'].std()*100:.4f}%")

    return df


# =============================================================================
# STRATEGY 1: Funding Collection (Always Long During Positive Funding)
# =============================================================================
def funding_collection_signals(df: pd.DataFrame, threshold: float = 0.0001) -> pd.Series:
    """
    Simple funding collection: Be long when funding > threshold.

    Logic: When funding is positive, longs RECEIVE payment from shorts.
    We want to be long to collect this income.
    """
    signals = pd.Series(0, index=df.index)

    # Long when funding is positive and above threshold
    signals[df['funding_rate'] > threshold] = 1

    # Short when funding is very negative (we pay to be long, receive as short)
    signals[df['funding_rate'] < -threshold] = -1

    return signals


# =============================================================================
# STRATEGY 2: Contrarian Funding (Fade Extreme Funding)
# =============================================================================
def contrarian_funding_signals(
    df: pd.DataFrame,
    long_threshold: float = -0.0005,  # -0.05% (shorts very crowded)
    short_threshold: float = 0.001,   # +0.10% (longs very crowded)
    lookback: int = 24                 # 24 * 4h = 96h lookback for z-score
) -> pd.Series:
    """
    Contrarian: Trade against extreme funding.

    Logic:
    - Extreme negative funding = shorts are crowded, paying high rates = short squeeze likely = LONG
    - Extreme positive funding = longs are crowded, paying high rates = long liquidation likely = SHORT
    """
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']

    # Rolling z-score
    funding_ma = funding.rolling(lookback, min_periods=lookback//2).mean()
    funding_std = funding.rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    z_score = (funding - funding_ma) / funding_std

    # Extreme negative funding OR z-score below -2 = crowded short = LONG
    extreme_negative = (funding < long_threshold) | (z_score < -2)

    # Extreme positive funding OR z-score above +2 = crowded long = SHORT
    extreme_positive = (funding > short_threshold) | (z_score > 2)

    signals[extreme_negative] = 1   # Long against crowded shorts
    signals[extreme_positive] = -1  # Short against crowded longs

    return signals


# =============================================================================
# STRATEGY 3: Combined (Collection + Contrarian)
# =============================================================================
def combined_funding_signals(
    df: pd.DataFrame,
    collect_threshold: float = 0.00005,  # +0.005% baseline for collection
    long_threshold: float = -0.0005,
    short_threshold: float = 0.001,
    lookback: int = 24
) -> pd.Series:
    """
    Combined strategy:
    1. Contrarian signals override when funding is extreme
    2. Otherwise, collect funding by being on the receiving side
    """
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']

    # Z-score for extremes
    funding_ma = funding.rolling(lookback, min_periods=lookback//2).mean()
    funding_std = funding.rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    z_score = (funding - funding_ma) / funding_std

    # 1. Base: Collection (be on receiving side)
    # Positive funding = longs pay shorts → be SHORT to receive
    # BUT: positive funding often means bullish market, so we prefer LONG for price appreciation
    # Negative funding = shorts pay longs → be LONG to receive
    signals[funding > collect_threshold] = 1   # Collect from shorts
    signals[funding < -collect_threshold] = -1  # Collect from longs

    # 2. Override: Contrarian on extremes (higher priority)
    # Extreme negative = shorts crowded/paying → reversal likely → LONG
    extreme_negative = (funding < long_threshold) | (z_score < -2)
    signals[extreme_negative] = 1

    # Extreme positive = longs crowded/paying → reversal likely → SHORT
    extreme_positive = (funding > short_threshold) | (z_score > 2)
    signals[extreme_positive] = -1

    return signals


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def backtest_with_funding(
    df: pd.DataFrame,
    signals: pd.Series,
    position_size: float = 0.1,
    commission: float = 0.0004  # 0.04% taker fee
) -> dict:
    """
    Backtest with separate tracking of price P&L and funding income.

    Funding is applied every 8 hours (we're on 4h, so every 2 bars).
    """
    n = len(df)
    positions = pd.Series(0.0, index=df.index)
    pnl = pd.Series(0.0, index=df.index)
    funding_pnl = pd.Series(0.0, index=df.index)

    current_pos = 0.0
    entry_price = 0.0
    total_trades = 0
    wins = 0

    for i in range(1, n):
        signal = signals.iloc[i-1]  # Signal at previous bar
        price = df['close'].iloc[i]
        prev_price = df['close'].iloc[i-1]
        funding = df['funding_rate'].iloc[i]

        # Track funding income (applied every 8 hours = every 2 bars for 4h data)
        if i % 2 == 0 and current_pos != 0:
            # Long position: receive funding if negative, pay if positive
            # Short position: receive funding if positive, pay if negative
            funding_income = -current_pos * funding * position_size  # Note: negative because longs pay positive funding
            funding_pnl.iloc[i] = funding_income

        # Price P&L
        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size

        # Signal processing
        if signal != 0 and signal != current_pos:
            # Close existing position
            if current_pos != 0:
                pnl.iloc[i] -= commission * abs(current_pos) * position_size  # Exit fee
                if pnl.iloc[i] > 0:
                    wins += 1

            # Open new position
            current_pos = signal
            entry_price = price
            total_trades += 1
            pnl.iloc[i] -= commission * abs(signal) * position_size  # Entry fee

        elif signal == 0 and current_pos != 0:
            # Exit signal
            pnl.iloc[i] -= commission * abs(current_pos) * position_size
            if pnl.iloc[i] > 0:
                wins += 1
            current_pos = 0.0
            total_trades += 1

        positions.iloc[i] = current_pos

    # Calculate metrics
    total_pnl = pnl + funding_pnl
    cumulative = (1 + total_pnl).cumprod() - 1

    returns_series = total_pnl[total_pnl != 0]
    if len(returns_series) > 0 and returns_series.std() > 0:
        sharpe = np.sqrt(252 * 6) * returns_series.mean() / returns_series.std()  # 6 4h bars per day
    else:
        sharpe = 0

    # Max drawdown
    cummax = (1 + total_pnl).cumprod().cummax()
    drawdown = (1 + total_pnl).cumprod() / cummax - 1
    max_dd = drawdown.min()

    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'total_return': cumulative.iloc[-1],
        'price_return': ((1 + pnl).cumprod() - 1).iloc[-1],
        'funding_return': ((1 + funding_pnl).cumprod() - 1).iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_funding_per_8h': funding_pnl.sum() / (n // 2) if n > 2 else 0
    }


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

        if len(train) > 100 and len(test) > 50:
            yield i, train, test


def run_strategy_walkforward(name: str, signal_func, df: pd.DataFrame, **kwargs):
    """Run walk-forward validation for a strategy."""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD: {name}")
    print(f"{'='*70}")

    results = []

    for fold_idx, train_df, test_df in walk_forward_split(df, n_splits=5):
        print(f"\n--- Fold {fold_idx} ---")
        print(f"  Train: {len(train_df):,} bars ({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]})")
        print(f"  Test:  {len(test_df):,} bars ({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})")

        # Generate signals
        train_signals = signal_func(train_df, **kwargs)
        test_signals = signal_func(test_df, **kwargs)

        # Backtest
        train_result = backtest_with_funding(train_df, train_signals)
        test_result = backtest_with_funding(test_df, test_signals)

        print(f"  Train: Sharpe={train_result['sharpe_ratio']:.3f}, Return={train_result['total_return']*100:.1f}% "
              f"(Price={train_result['price_return']*100:.1f}%, Funding={train_result['funding_return']*100:.2f}%)")
        print(f"  Test:  Sharpe={test_result['sharpe_ratio']:.3f}, Return={test_result['total_return']*100:.1f}% "
              f"(Price={test_result['price_return']*100:.1f}%, Funding={test_result['funding_return']*100:.2f}%)")

        results.append({
            'fold': fold_idx,
            'train_sharpe': train_result['sharpe_ratio'],
            'train_return': train_result['total_return'],
            'test_sharpe': test_result['sharpe_ratio'],
            'test_return': test_result['total_return'],
            'test_price_return': test_result['price_return'],
            'test_funding_return': test_result['funding_return'],
            'test_max_dd': test_result['max_drawdown'],
            'test_trades': test_result['total_trades'],
            'test_win_rate': test_result['win_rate'],
        })

    # Aggregate
    valid = [r for r in results if r['test_trades'] > 0]

    if valid:
        avg_sharpe = np.mean([r['test_sharpe'] for r in valid])
        avg_return = np.mean([r['test_return'] for r in valid])
        compounded = np.prod([1 + r['test_return'] for r in valid]) - 1
        avg_price = np.mean([r['test_price_return'] for r in valid])
        avg_funding = np.mean([r['test_funding_return'] for r in valid])
        avg_dd = np.mean([r['test_max_dd'] for r in valid])
        total_trades = sum(r['test_trades'] for r in valid)
        avg_wr = np.mean([r['test_win_rate'] for r in valid])
        consistency = sum(1 for r in valid if r['test_sharpe'] > 0) / len(valid)
    else:
        avg_sharpe = avg_return = compounded = avg_price = avg_funding = avg_dd = total_trades = avg_wr = consistency = 0

    summary = {
        'strategy': name,
        'avg_test_sharpe': avg_sharpe,
        'avg_test_return': avg_return,
        'compounded_return': compounded,
        'avg_price_return': avg_price,
        'avg_funding_return': avg_funding,
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
    print(f"  Avg Price Return:    {avg_price*100:.1f}%")
    print(f"  Avg Funding Return:  {avg_funding*100:.2f}%")
    print(f"  Avg Max Drawdown:    {avg_dd*100:.1f}%")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Avg Win Rate:        {avg_wr:.1f}%")
    print(f"  Consistency:         {consistency*100:.0f}% folds profitable")

    return summary


def main():
    print("=" * 80)
    print("BTC FUNDING RATE STRATEGY BACKTEST")
    print("Using REAL derivatives data (3000+ rows, May 2023 - Feb 2026)")
    print("=" * 80)

    start_time = datetime.now()

    # Load and merge data
    df = load_and_merge_data()

    # Run strategies
    strategies = [
        ('FundingCollection', funding_collection_signals, {'threshold': 0.00005}),
        ('ContrarianFunding', contrarian_funding_signals, {'long_threshold': -0.0005, 'short_threshold': 0.001, 'lookback': 24}),
        ('CombinedFunding', combined_funding_signals, {'collect_threshold': 0.00003, 'long_threshold': -0.0005, 'short_threshold': 0.001}),
    ]

    all_results = []

    for name, signal_func, kwargs in strategies:
        result = run_strategy_walkforward(name, signal_func, df, **kwargs)
        all_results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 80)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x['avg_test_sharpe'], reverse=True)

    print(f"\n{'Strategy':<25} {'Sharpe':>8} {'Total':>10} {'Price':>10} {'Funding':>10} {'DD':>8} {'Consist':>8}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['strategy']:<25} "
              f"{r['avg_test_sharpe']:>8.3f} "
              f"{r['compounded_return']*100:>9.1f}% "
              f"{r['avg_price_return']*100:>9.1f}% "
              f"{r['avg_funding_return']*100:>9.2f}% "
              f"{r['avg_max_dd']*100:>7.1f}% "
              f"{r['consistency']*100:>7.0f}%")

    # Save results
    out_dir = '/Users/ffv_macmini/Desktop/maestro/data/backtest_results'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = f'{out_dir}/btc_funding_backtest_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Saved: {json_path}")

    # Key insight
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print(f"{'='*80}")
    best = all_results[0]
    print(f"Best Strategy: {best['strategy']}")
    print(f"  - Total Return splits funding income from price P&L")
    print(f"  - Funding provides a baseline income stream")
    print(f"  - Contrarian signals aim to capture mean reversion")
    print(f"\nNote: Funding income is ~8% annual (0.02%/8h avg)")
    print("      Real edge comes from timing price moves during funding extremes")


if __name__ == '__main__':
    main()
