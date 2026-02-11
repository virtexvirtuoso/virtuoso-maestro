"""Full Derivatives Combo Hybrid"""
import pandas as pd
import numpy as np

NAME = "DerivativesCombo"
CATEGORY = "hybrid"
DESCRIPTION = "OI + CVD + Funding (2 of 3 must agree)"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, lookback: int = 14, short_threshold: float = -0.0001, long_threshold: float = 0.0005) -> pd.Series:
    """Full derivatives combo - 2 of 3 must agree."""
    
    # OI Divergence
    oi_signals = pd.Series(0, index=df.index)
    if 'open_interest' in df.columns and not df['open_interest'].isna().all():
        price = df['close']
        oi = df['open_interest'].ffill()
        price_low = price.rolling(lookback).min()
        oi_low = oi.rolling(lookback).min()
        oi_high = oi.rolling(lookback).max()
        oi_range = oi_high - oi_low
        oi_position = (oi - oi_low) / oi_range.replace(0, np.nan)
        price_at_low = price <= price_low * 1.02
        oi_signals[price_at_low & (oi_position > 0.4)] = 1
    
    # CVD Divergence
    cvd_signals = pd.Series(0, index=df.index)
    if 'open_interest' in df.columns and not df['open_interest'].isna().all():
        oi = df['open_interest'].ffill()
        oi_change = oi.pct_change(lookback)
        cvd = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        cvd_change = cvd.diff(lookback)
        oi_rising = oi_change > 0.05
        cvd_signals[oi_rising & (cvd_change < 0)] = 1
    
    # Funding
    funding_signals = pd.Series(0, index=df.index)
    if 'funding_rate' in df.columns and not df['funding_rate'].isna().all():
        funding = df['funding_rate'].ffill()
        funding_signals[funding < short_threshold] = 1
        funding_signals[funding > long_threshold] = -1
    
    # 2 of 3 must agree
    signals = pd.Series(0, index=df.index)
    long_votes = (oi_signals == 1).astype(int) + (cvd_signals == 1).astype(int) + (funding_signals == 1).astype(int)
    short_votes = (oi_signals == -1).astype(int) + (cvd_signals == -1).astype(int) + (funding_signals == -1).astype(int)
    
    signals[long_votes >= 2] = 1
    signals[short_votes >= 2] = -1
    return signals
