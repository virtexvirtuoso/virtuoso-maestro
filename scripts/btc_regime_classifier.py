#!/usr/bin/env python3
"""
BTC Market Regime Classifier

Detects market regimes:
1. TRENDING_UP: Strong uptrend, momentum strategies work
2. TRENDING_DOWN: Strong downtrend, short momentum works
3. RANGING: Sideways, mean-reversion strategies work (contrarian funding!)
4. VOLATILE: High volatility, reduce exposure

Uses:
- ADX for trend strength
- ATR percentile for volatility regime
- Price position relative to SMAs
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
    """Load and prepare data."""
    df = pd.read_csv(OHLCV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[df['timestamp'] >= '2023-05-01'].reset_index(drop=True)

    # Load funding
    if 'funding_rate' in df.columns:
        df = df.drop(columns=['funding_rate'])

    funding = pd.read_csv(FUNDING_PATH)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])
    funding = funding.rename(columns={'fundingRate': 'funding_rate_real'})

    df = pd.merge_asof(df, funding[['timestamp', 'funding_rate_real']],
                       on='timestamp', direction='backward')
    df['funding_rate'] = df['funding_rate_real'].fillna(0)

    return df


def calculate_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate regime detection indicators."""
    # ADX for trend strength
    period = 14
    high, low, close = df['high'], df['low'], df['close']

    tr = pd.concat([high - low,
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(period).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # Volatility percentile
    df['atr'] = atr
    df['atr_pct'] = df['atr'].rolling(168).rank(pct=True)  # 7 days in 4h

    # Trend direction (multiple SMAs)
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['sma_200'] = close.rolling(200).mean()

    # Trend alignment
    df['trend_alignment'] = (
        (close > df['sma_20']).astype(int) +
        (close > df['sma_50']).astype(int) +
        (close > df['sma_200']).astype(int) +
        (df['sma_20'] > df['sma_50']).astype(int) +
        (df['sma_50'] > df['sma_200']).astype(int)
    )

    return df


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market regime.

    TRENDING_UP: ADX > 25, trend_alignment >= 4
    TRENDING_DOWN: ADX > 25, trend_alignment <= 1
    RANGING: ADX <= 25
    VOLATILE: ATR percentile > 80%
    """
    df['regime'] = 'RANGING'

    # Trending up
    trending_up = (df['adx'] > 25) & (df['trend_alignment'] >= 4)
    df.loc[trending_up, 'regime'] = 'TRENDING_UP'

    # Trending down
    trending_down = (df['adx'] > 25) & (df['trend_alignment'] <= 1)
    df.loc[trending_down, 'regime'] = 'TRENDING_DOWN'

    # Volatile (override)
    volatile = df['atr_pct'] > 0.8
    df.loc[volatile, 'regime'] = 'VOLATILE'

    return df


# =============================================================================
# REGIME-ADAPTIVE CONTRARIAN FUNDING STRATEGY
# =============================================================================
def adaptive_contrarian_signals(
    df: pd.DataFrame,
    long_threshold: float = -0.0005,
    short_threshold: float = 0.001,
    lookback: int = 24
) -> pd.Series:
    """
    Contrarian funding signals with regime filter.

    - In RANGING: Full contrarian (best regime for mean-reversion)
    - In VOLATILE: Contrarian with tighter thresholds
    - In TRENDING_UP: Only take long contrarian signals
    - In TRENDING_DOWN: Only take short contrarian signals
    """
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']

    # Z-score
    funding_ma = funding.rolling(lookback, min_periods=lookback//2).mean()
    funding_std = funding.rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    z_score = (funding - funding_ma) / funding_std

    # Base contrarian signals
    extreme_negative = (funding < long_threshold) | (z_score < -2)
    extreme_positive = (funding > short_threshold) | (z_score > 2)

    # Apply regime filters
    for i in range(len(df)):
        regime = df['regime'].iloc[i]

        if regime == 'RANGING':
            # Full contrarian
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1
            elif extreme_positive.iloc[i]:
                signals.iloc[i] = -1

        elif regime == 'VOLATILE':
            # Tighter thresholds
            if funding.iloc[i] < long_threshold * 2 or z_score.iloc[i] < -2.5:
                signals.iloc[i] = 1
            elif funding.iloc[i] > short_threshold * 2 or z_score.iloc[i] > 2.5:
                signals.iloc[i] = -1

        elif regime == 'TRENDING_UP':
            # Only long signals (with trend)
            if extreme_negative.iloc[i]:
                signals.iloc[i] = 1
            # Exit shorts
            elif signals.iloc[i-1] if i > 0 else 0 == -1:
                signals.iloc[i] = 0

        elif regime == 'TRENDING_DOWN':
            # Only short signals (with trend)
            if extreme_positive.iloc[i]:
                signals.iloc[i] = -1
            # Exit longs
            elif signals.iloc[i-1] if i > 0 else 0 == 1:
                signals.iloc[i] = 0

    return signals


def base_contrarian_signals(
    df: pd.DataFrame,
    long_threshold: float = -0.0005,
    short_threshold: float = 0.001,
    lookback: int = 24
) -> pd.Series:
    """Base contrarian without regime filter (for comparison)."""
    signals = pd.Series(0, index=df.index)
    funding = df['funding_rate']

    funding_ma = funding.rolling(lookback, min_periods=lookback//2).mean()
    funding_std = funding.rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    z_score = (funding - funding_ma) / funding_std

    extreme_negative = (funding < long_threshold) | (z_score < -2)
    extreme_positive = (funding > short_threshold) | (z_score > 2)

    signals[extreme_negative] = 1
    signals[extreme_positive] = -1

    return signals


def backtest(df: pd.DataFrame, signals: pd.Series, position_size: float = 0.1,
             commission: float = 0.0004) -> dict:
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

        if current_pos != 0:
            price_return = (price / prev_price - 1) * current_pos
            pnl.iloc[i] = price_return * position_size

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
            'test_sharpe': test_result['sharpe_ratio'],
            'test_return': test_result['total_return'],
            'test_max_dd': test_result['max_drawdown'],
            'test_trades': test_result['total_trades'],
        })

    valid = [r for r in results if r['test_trades'] > 0]
    if valid:
        avg_sharpe = np.mean([r['test_sharpe'] for r in valid])
        compounded = np.prod([1 + r['test_return'] for r in valid]) - 1
        consistency = sum(1 for r in valid if r['test_sharpe'] > 0) / len(valid)
    else:
        avg_sharpe = compounded = consistency = 0

    print(f"\nSUMMARY: Sharpe={avg_sharpe:.3f}, Return={compounded*100:.1f}%, "
          f"Consistency={consistency*100:.0f}%")

    return {
        'strategy': name,
        'avg_test_sharpe': avg_sharpe,
        'compounded_return': compounded,
        'consistency': consistency,
        'folds': results
    }


def main():
    print("=" * 80)
    print("BTC REGIME CLASSIFIER + ADAPTIVE CONTRARIAN FUNDING")
    print("=" * 80)

    df = load_data()
    df = calculate_regime_indicators(df)
    df = classify_regime(df)

    # Regime distribution
    print(f"\nData: {len(df):,} bars")
    print("\nRegime Distribution:")
    print(df['regime'].value_counts())

    # Compare base vs adaptive
    strategies = [
        ('BaseContrarian', base_contrarian_signals, {}),
        ('AdaptiveContrarian', adaptive_contrarian_signals, {}),
    ]

    all_results = []
    for name, func, kwargs in strategies:
        result = run_walkforward(name, func, df, **kwargs)
        all_results.append(result)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    all_results.sort(key=lambda x: x['avg_test_sharpe'], reverse=True)

    print(f"\n{'Strategy':<25} {'Sharpe':>8} {'Return':>10} {'Consist':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['avg_test_sharpe']:>8.3f} "
              f"{r['compounded_return']*100:>9.1f}% "
              f"{r['consistency']*100:>7.0f}%")

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{DATA_DIR}/backtest_results/btc_regime_adaptive_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
