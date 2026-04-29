"""
FRAPS - Funding Rate Arbitrage Pressure Signal

Compares funding rates ACROSS exchanges to detect arbitrage pressure.
When funding diverges, arbitrageurs create directional pressure:
- Arbers SHORT on high-funding exchanges → downward pressure
- Arbers LONG on low-funding exchanges → upward pressure

Signal: when dispersion is HIGH and consensus is strong,
fade the consensus direction (arb pressure normalizes).
"""
import numpy as np
import pandas as pd
from typing import Optional


def compute_fraps_features(funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FRAPS features from multi-exchange funding data.
    
    Input: DataFrame with columns [timestamp, funding_close, exchange, token]
    Output: DataFrame indexed by timestamp with FRAPS features
    """
    # Normalize exchange names - combine USD and USDT variants
    df = funding_df.copy()
    df['exchange_base'] = df['exchange'].str.replace('_usdt', '')
    
    # Pivot: rows=timestamp, cols=exchange, values=funding_close
    pivot = df.pivot_table(
        index='timestamp', columns='exchange_base',
        values='funding_close', aggfunc='mean'
    )
    pivot = pivot.sort_index()
    
    features = pd.DataFrame(index=pivot.index)
    
    # 1. Mean funding across exchanges
    features['funding_mean'] = pivot.mean(axis=1)
    
    # 2. Funding dispersion (std across exchanges)
    features['funding_std'] = pivot.std(axis=1)
    
    # 3. Funding range (max - min)
    features['funding_range'] = pivot.max(axis=1) - pivot.min(axis=1)
    
    # 4. Direction consensus: fraction of exchanges with positive funding
    features['consensus'] = (pivot > 0).sum(axis=1) / pivot.notna().sum(axis=1)
    
    # 5. Number of exchanges reporting
    features['n_exchanges'] = pivot.notna().sum(axis=1)
    
    # Store individual exchange funding for reference
    for col in pivot.columns:
        features[f'funding_{col}'] = pivot[col]
    
    return features


def generate_fraps_signal(
    funding_df: pd.DataFrame,
    price_df: pd.DataFrame,
    dispersion_zscore_threshold: float = 1.5,
    consensus_threshold: float = 0.8,
    lookback: int = 60,
    hold_period: int = 3,
) -> pd.DataFrame:
    """
    Generate FRAPS trading signals.
    
    Parameters:
    - dispersion_zscore_threshold: min z-score of dispersion to trigger signal
    - consensus_threshold: min fraction of exchanges agreeing on direction
    - lookback: days for rolling z-score calculation
    - hold_period: days to hold position after signal
    
    Returns DataFrame with 'signal' column: 1=long, -1=short, 0=neutral
    """
    features = compute_fraps_features(funding_df)
    
    # Ensure price index is datetime
    price = price_df.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
    features.index = pd.to_datetime(features.index)
    
    # Align
    combined = price[['close']].join(features, how='inner')
    
    # Rolling z-score of dispersion
    roll_mean = combined['funding_std'].rolling(lookback, min_periods=max(lookback // 2, 10)).mean()
    roll_std = combined['funding_std'].rolling(lookback, min_periods=max(lookback // 2, 10)).std()
    combined['dispersion_zscore'] = (combined['funding_std'] - roll_mean) / roll_std.replace(0, np.nan)
    
    # Generate raw signals
    combined['raw_signal'] = 0
    
    # High dispersion + strong positive consensus → fade (go short, arbs will push down)
    long_consensus = (combined['consensus'] >= consensus_threshold)
    short_consensus = (combined['consensus'] <= (1 - consensus_threshold))
    high_dispersion = (combined['dispersion_zscore'] >= dispersion_zscore_threshold)
    
    combined.loc[high_dispersion & long_consensus, 'raw_signal'] = -1  # fade positive consensus
    combined.loc[high_dispersion & short_consensus, 'raw_signal'] = 1   # fade negative consensus
    
    # Apply hold period
    combined['signal'] = 0
    signal_dates = combined.index[combined['raw_signal'] != 0]
    for date in signal_dates:
        idx = combined.index.get_loc(date)
        end_idx = min(idx + hold_period, len(combined))
        combined.iloc[idx:end_idx, combined.columns.get_loc('signal')] = combined.loc[date, 'raw_signal']
    
    return combined


def backtest_fraps(
    funding_df: pd.DataFrame,
    price_df: pd.DataFrame,
    dispersion_zscore_threshold: float = 1.5,
    consensus_threshold: float = 0.8,
    lookback: int = 60,
    hold_period: int = 3,
    commission_bps: float = 20,
) -> dict:
    """
    Backtest FRAPS strategy and return performance metrics.
    """
    result = generate_fraps_signal(
        funding_df, price_df,
        dispersion_zscore_threshold, consensus_threshold,
        lookback, hold_period
    )
    
    # Calculate returns
    result['price_return'] = result['close'].pct_change()
    result['strategy_return'] = result['signal'].shift(1) * result['price_return']
    
    # Commission on signal changes
    result['signal_change'] = result['signal'].diff().abs()
    commission = commission_bps / 10000
    result['strategy_return'] -= result['signal_change'] * commission
    
    # Drop NaN
    result = result.dropna(subset=['strategy_return'])
    
    if len(result) == 0:
        return {'sharpe': 0, 'total_return': 0, 'n_trades': 0, 'win_rate': 0}
    
    # Metrics
    strat_returns = result['strategy_return']
    n_trades = int(result['signal_change'].gt(0).sum())
    
    # Only count returns when we have a position
    active = strat_returns[result['signal'].shift(1) != 0]
    win_rate = float((active > 0).sum() / max(len(active), 1))
    
    total_return = float((1 + strat_returns).prod() - 1)
    sharpe = float(strat_returns.mean() / strat_returns.std() * np.sqrt(365)) if strat_returns.std() > 0 else 0
    max_dd = float((strat_returns.cumsum() - strat_returns.cumsum().cummax()).min())
    
    return {
        'sharpe': round(sharpe, 4),
        'total_return': round(total_return * 100, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate * 100, 2),
        'n_days': len(result),
        'avg_daily_return': round(float(strat_returns.mean()) * 100, 4),
        'result_df': result,
    }
