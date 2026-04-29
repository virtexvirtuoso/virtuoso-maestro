#!/usr/bin/env python3
"""
BTC Leads Alts: Cross-Asset Signal Investigation

Hypothesis: BTC funding/regime signals can predict alt moves.

Strategies to test:
1. BTC_SIGNAL_ALT_TRADE: Use BTC contrarian signals to trade alts (higher beta)
2. BTC_REGIME_ALT_SELECTION: Long alts in BTC uptrend, short in downtrend
3. ALT_FUNDING_AMPLIFIED: Alt's own funding + BTC confirmation
4. BTC_CONSOLIDATION_ALT_ROTATION: Trade alts when BTC is ranging
5. BETA_WEIGHTED: Size alt positions by their beta to BTC

Data required:
- BTC 4h OHLCV + funding
- Alt 4h OHLCV + funding (ETH, SOL, etc.)
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


def load_btc_data() -> pd.DataFrame:
    """Load BTC with funding and regime indicators."""
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

    # Rename columns with btc_ prefix
    df = df.rename(columns={
        'open': 'btc_open', 'high': 'btc_high', 'low': 'btc_low',
        'close': 'btc_close', 'volume': 'btc_volume',
        'funding_rate': 'btc_funding'
    })

    return df


def load_alt_data(symbol: str) -> pd.DataFrame:
    """Load alt coin data."""
    # Try different file patterns
    patterns = [
        f'{DATA_DIR}/merged/binance_{symbol}_usdt_4h.csv',
        f'{DATA_DIR}/merged/binance_{symbol.lower()}_usdt_4h.csv',
        f'{DATA_DIR}/binance_{symbol.lower()}_usdt_4h.csv',
    ]

    df = None
    for path in patterns:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Load alt funding if available
    funding_path = f'{DATA_DIR}/derivatives/{symbol.lower()}_funding.csv'
    if os.path.exists(funding_path):
        funding = pd.read_csv(funding_path)
        funding['timestamp'] = pd.to_datetime(funding['timestamp'].str[:19])
        if 'fundingRate' in funding.columns:
            funding = funding.rename(columns={'fundingRate': 'alt_funding'})
        elif 'funding_rate' in funding.columns:
            funding = funding.rename(columns={'funding_rate': 'alt_funding'})
        df = pd.merge_asof(df, funding[['timestamp', 'alt_funding']],
                           on='timestamp', direction='backward')
    else:
        df['alt_funding'] = 0

    df['alt_funding'] = df['alt_funding'].fillna(0)

    # Rename columns
    df = df.rename(columns={
        'open': 'alt_open', 'high': 'alt_high', 'low': 'alt_low',
        'close': 'alt_close', 'volume': 'alt_volume'
    })

    return df


def calculate_btc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BTC regime and signal indicators."""
    close = df['btc_close']
    high = df['btc_high']
    low = df['btc_low']

    # ADX
    period = 14
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['btc_adx'] = dx.rolling(period).mean()

    df['btc_atr'] = atr
    df['btc_atr_pct'] = df['btc_atr'].rolling(168).rank(pct=True)

    # SMAs
    df['btc_sma_20'] = close.rolling(20).mean()
    df['btc_sma_50'] = close.rolling(50).mean()
    df['btc_sma_200'] = close.rolling(200).mean()

    # Trend alignment
    df['btc_trend_alignment'] = (
        (close > df['btc_sma_20']).astype(int) +
        (close > df['btc_sma_50']).astype(int) +
        (close > df['btc_sma_200']).astype(int) +
        (df['btc_sma_20'] > df['btc_sma_50']).astype(int) +
        (df['btc_sma_50'] > df['btc_sma_200']).astype(int)
    )

    # Regime
    df['btc_regime'] = 'RANGING'
    trending_up = (df['btc_adx'] > 25) & (df['btc_trend_alignment'] >= 4)
    trending_down = (df['btc_adx'] > 25) & (df['btc_trend_alignment'] <= 1)
    volatile = df['btc_atr_pct'] > 0.8

    df.loc[trending_up, 'btc_regime'] = 'TRENDING_UP'
    df.loc[trending_down, 'btc_regime'] = 'TRENDING_DOWN'
    df.loc[volatile, 'btc_regime'] = 'VOLATILE'

    # Funding z-score
    lookback = 22
    df['btc_funding_ma'] = df['btc_funding'].rolling(lookback, min_periods=lookback//2).mean()
    df['btc_funding_std'] = df['btc_funding'].rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
    df['btc_funding_zscore'] = (df['btc_funding'] - df['btc_funding_ma']) / df['btc_funding_std']

    # BTC contrarian signals (optimized thresholds)
    df['btc_extreme_negative'] = (df['btc_funding'] < -0.000691) | (df['btc_funding_zscore'] < -2.18)
    df['btc_extreme_positive'] = (df['btc_funding'] > 0.001086) | (df['btc_funding_zscore'] > 2.18)

    # BTC returns for beta calculation
    df['btc_return'] = df['btc_close'].pct_change()

    return df


def calculate_alt_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate alt-specific indicators."""
    if 'alt_close' not in df.columns:
        return df

    # Alt returns
    df['alt_return'] = df['alt_close'].pct_change()

    # Rolling beta (60-bar lookback)
    lookback = 60
    cov = df['alt_return'].rolling(lookback).cov(df['btc_return'])
    var = df['btc_return'].rolling(lookback).var()
    df['alt_beta'] = cov / (var + 1e-10)
    df['alt_beta'] = df['alt_beta'].clip(-5, 5)  # Limit extreme values

    # Alt funding z-score (if available)
    if df['alt_funding'].abs().sum() > 0:
        lookback = 22
        df['alt_funding_ma'] = df['alt_funding'].rolling(lookback, min_periods=lookback//2).mean()
        df['alt_funding_std'] = df['alt_funding'].rolling(lookback, min_periods=lookback//2).std().replace(0, 1e-10)
        df['alt_funding_zscore'] = (df['alt_funding'] - df['alt_funding_ma']) / df['alt_funding_std']

        df['alt_extreme_negative'] = (df['alt_funding'] < -0.0005) | (df['alt_funding_zscore'] < -2)
        df['alt_extreme_positive'] = (df['alt_funding'] > 0.001) | (df['alt_funding_zscore'] > 2)
    else:
        df['alt_funding_zscore'] = 0
        df['alt_extreme_negative'] = False
        df['alt_extreme_positive'] = False

    return df


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_btc_signal_alt_trade(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 1: Use BTC contrarian signals to trade alts.

    Hypothesis: When BTC funding is extreme, BTC will revert.
    Alts follow BTC with higher beta → bigger moves.
    """
    signals = pd.Series(0, index=df.index)

    for i in range(len(df)):
        regime = df['btc_regime'].iloc[i]

        if regime == 'RANGING':
            if df['btc_extreme_negative'].iloc[i]:
                signals.iloc[i] = 1  # Long alt
            elif df['btc_extreme_positive'].iloc[i]:
                signals.iloc[i] = -1  # Short alt

        elif regime == 'TRENDING_UP':
            if df['btc_extreme_negative'].iloc[i]:
                signals.iloc[i] = 1  # Long alt (with trend)

        elif regime == 'TRENDING_DOWN':
            if df['btc_extreme_positive'].iloc[i]:
                signals.iloc[i] = -1  # Short alt (with trend)

    return signals


def strategy_btc_regime_alt_momentum(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 2: Trade alts based on BTC regime.

    - BTC TRENDING_UP → Long alts (momentum)
    - BTC TRENDING_DOWN → Short alts
    - BTC RANGING → No trade (wait for direction)
    - BTC VOLATILE → No trade (too risky)
    """
    signals = pd.Series(0, index=df.index)

    signals[df['btc_regime'] == 'TRENDING_UP'] = 1
    signals[df['btc_regime'] == 'TRENDING_DOWN'] = -1

    return signals


def strategy_alt_funding_btc_confirm(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 3: Alt's own funding signal, confirmed by BTC direction.

    Only trade alt when:
    - Alt has extreme funding (crowded positioning)
    - BTC regime confirms the direction
    """
    signals = pd.Series(0, index=df.index)

    if 'alt_extreme_negative' not in df.columns:
        return signals

    # Long alt: Alt funding extreme negative + BTC not trending down
    long_cond = (
        df['alt_extreme_negative'] &
        (df['btc_regime'] != 'TRENDING_DOWN')
    )

    # Short alt: Alt funding extreme positive + BTC not trending up
    short_cond = (
        df['alt_extreme_positive'] &
        (df['btc_regime'] != 'TRENDING_UP')
    )

    signals[long_cond] = 1
    signals[short_cond] = -1

    return signals


def strategy_btc_ranging_alt_rotation(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 4: Trade alts when BTC is consolidating.

    Hypothesis: When BTC ranges, capital rotates to alts.
    Look for alt-specific signals during BTC consolidation.
    """
    signals = pd.Series(0, index=df.index)

    btc_ranging = df['btc_regime'] == 'RANGING'

    # During BTC ranging, use alt's own funding signals
    if 'alt_extreme_negative' in df.columns:
        signals[btc_ranging & df['alt_extreme_negative']] = 1
        signals[btc_ranging & df['alt_extreme_positive']] = -1
    else:
        # If no alt funding, use BTC signals but only in ranging
        signals[btc_ranging & df['btc_extreme_negative']] = 1
        signals[btc_ranging & df['btc_extreme_positive']] = -1

    return signals


def strategy_beta_weighted(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 5: BTC signal + beta-weighted sizing.

    Returns position size (not just direction) based on alt's beta.
    Higher beta = smaller position, lower beta = larger position.
    """
    signals = pd.Series(0.0, index=df.index)

    # Base signal from BTC contrarian
    base_signal = strategy_btc_signal_alt_trade(df)

    # Weight by inverse beta (higher beta = smaller position)
    # Normalize beta to 0.5-1.5 range for position sizing
    beta = df['alt_beta'].fillna(1.0)
    weight = 1.0 / beta.clip(0.5, 3.0)  # Higher beta → smaller weight
    weight = weight / weight.mean()  # Normalize

    signals = base_signal * weight.clip(0.5, 1.5)

    return signals


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def backtest(df: pd.DataFrame, signals: pd.Series, position_size: float = 0.15,
             commission: float = 0.0004) -> dict:
    """Backtest strategy on alt."""
    if 'alt_close' not in df.columns:
        return None

    n = len(df)
    pnl = pd.Series(0.0, index=df.index)
    current_pos = 0.0
    total_trades = 0
    wins = 0

    for i in range(1, n):
        signal = signals.iloc[i-1]
        price = df['alt_close'].iloc[i]
        prev_price = df['alt_close'].iloc[i-1]

        # Handle float signals (beta-weighted)
        signal_direction = np.sign(signal)
        signal_size = abs(signal) if abs(signal) > 0.01 else 0

        if current_pos != 0:
            price_return = (price / prev_price - 1) * np.sign(current_pos)
            pnl.iloc[i] = price_return * abs(current_pos) * position_size

        if signal_size > 0.01 and signal_direction != np.sign(current_pos):
            if current_pos != 0:
                pnl.iloc[i] -= commission * abs(current_pos) * position_size
                if pnl.iloc[i] > 0:
                    wins += 1
            current_pos = signal
            total_trades += 1
            pnl.iloc[i] -= commission * abs(signal) * position_size
        elif signal_size < 0.01 and current_pos != 0:
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

    return {
        'total_return': cumulative.iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100 if total_trades > 0 else 0
    }


def walk_forward_test(df: pd.DataFrame, strategy_func, n_splits: int = 5) -> dict:
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

        signals = strategy_func(test_df)
        result = backtest(test_df, signals)

        if result is None:
            continue

        sharpes.append(result['sharpe_ratio'])
        returns.append(result['total_return'])
        trades.append(result['total_trades'])

    if not sharpes:
        return None

    return {
        'avg_sharpe': np.mean(sharpes),
        'total_return': np.prod([1 + r for r in returns]) - 1,
        'consistency': sum(1 for s in sharpes if s > 0) / len(sharpes),
        'avg_trades': np.mean(trades),
        'fold_sharpes': sharpes,
        'fold_returns': returns
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BTC LEADS ALTS: Cross-Asset Signal Investigation")
    print("=" * 80)

    # Load BTC data
    print("\nLoading BTC data...")
    btc_df = load_btc_data()
    btc_df = calculate_btc_indicators(btc_df)
    print(f"BTC data: {len(btc_df):,} bars")

    # Regime distribution
    print(f"\nBTC Regime Distribution:")
    print(btc_df['btc_regime'].value_counts())

    # Define alts to test (all available with funding data)
    alts = ['eth', 'sol', 'avax', 'link', 'arb', 'op', 'inj', 'sui', 'tia', 'fet', 'render', 'tao']

    # Define strategies
    strategies = {
        'BTC_Signal_Alt_Trade': strategy_btc_signal_alt_trade,
        'BTC_Regime_Momentum': strategy_btc_regime_alt_momentum,
        'Alt_Funding_BTC_Confirm': strategy_alt_funding_btc_confirm,
        'BTC_Ranging_Alt_Rotation': strategy_btc_ranging_alt_rotation,
        'Beta_Weighted': strategy_beta_weighted,
    }

    results = []

    for alt in alts:
        print(f"\n{'='*60}")
        print(f"Testing: {alt.upper()}")
        print("=" * 60)

        alt_df = load_alt_data(alt)
        if alt_df is None:
            print(f"  No data found for {alt}")
            continue

        # Merge with BTC
        merged = pd.merge(btc_df, alt_df, on='timestamp', how='inner')
        merged = calculate_alt_indicators(merged)

        if len(merged) < 500:
            print(f"  Insufficient data for {alt}: {len(merged)} bars")
            continue

        print(f"  Data: {len(merged):,} bars")
        print(f"  Avg Beta: {merged['alt_beta'].mean():.2f}")

        for strat_name, strat_func in strategies.items():
            result = walk_forward_test(merged, strat_func)

            if result is None:
                continue

            results.append({
                'alt': alt.upper(),
                'strategy': strat_name,
                **result
            })

            print(f"  {strat_name:<30} Sharpe: {result['avg_sharpe']:>6.2f}  "
                  f"Return: {result['total_return']*100:>6.1f}%  "
                  f"Consist: {result['consistency']*100:>3.0f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Best Strategies by Alt")
    print("=" * 80)

    if not results:
        print("No valid results found.")
        return

    results_df = pd.DataFrame(results)

    # Best by Sharpe for each alt
    print(f"\n{'Alt':<8} {'Best Strategy':<35} {'Sharpe':>8} {'Return':>10} {'Consist':>8}")
    print("-" * 75)

    for alt in results_df['alt'].unique():
        alt_results = results_df[results_df['alt'] == alt]
        best = alt_results.loc[alt_results['avg_sharpe'].idxmax()]
        print(f"{best['alt']:<8} {best['strategy']:<35} {best['avg_sharpe']:>8.2f} "
              f"{best['total_return']*100:>9.1f}% {best['consistency']*100:>7.0f}%")

    # Best strategies overall
    print("\n" + "=" * 80)
    print("BEST STRATEGIES OVERALL (by avg Sharpe across alts)")
    print("=" * 80)

    strat_summary = results_df.groupby('strategy').agg({
        'avg_sharpe': 'mean',
        'total_return': 'mean',
        'consistency': 'mean'
    }).sort_values('avg_sharpe', ascending=False)

    print(f"\n{'Strategy':<35} {'Avg Sharpe':>12} {'Avg Return':>12} {'Avg Consist':>12}")
    print("-" * 75)
    for strat, row in strat_summary.iterrows():
        print(f"{strat:<35} {row['avg_sharpe']:>12.2f} {row['total_return']*100:>11.1f}% "
              f"{row['consistency']*100:>11.0f}%")

    # Best alt/strategy combinations
    print("\n" + "=" * 80)
    print("TOP 10 ALT/STRATEGY COMBINATIONS")
    print("=" * 80)

    top10 = results_df.nlargest(10, 'avg_sharpe')
    print(f"\n{'Rank':<6} {'Alt':<8} {'Strategy':<30} {'Sharpe':>8} {'Return':>10}")
    print("-" * 70)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<6} {row['alt']:<8} {row['strategy']:<30} {row['avg_sharpe']:>8.2f} "
              f"{row['total_return']*100:>9.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'all_results': results,
        'strategy_summary': strat_summary.to_dict(),
        'top_10': top10.to_dict('records')
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f'{DATA_DIR}/backtest_results/btc_leads_alts_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
