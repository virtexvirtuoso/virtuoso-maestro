"""OI Divergence + Funding Rate Hybrid"""
import pandas as pd
import numpy as np

NAME = "OIDivergence+FundingRate"
CATEGORY = "hybrid"
DESCRIPTION = "OI divergence + extreme funding confluence"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, lookback: int = 14, short_threshold: float = -0.0001, long_threshold: float = 0.0005) -> pd.Series:
    """OI divergence + funding rate confluence."""
    signals = pd.Series(0, index=df.index)
    
    # OI Divergence
    oi_signals = pd.Series(0, index=df.index)
    if 'open_interest' in df.columns and not df['open_interest'].isna().all():
        price = df['close']
        oi = df['open_interest'].ffill()
        
        price_low = price.rolling(lookback).min()
        price_high = price.rolling(lookback).max()
        oi_low = oi.rolling(lookback).min()
        oi_high = oi.rolling(lookback).max()
        
        oi_range = oi_high - oi_low
        oi_position = (oi - oi_low) / oi_range.replace(0, np.nan)
        
        price_at_low = price <= price_low * 1.02
        price_at_high = price >= price_high * 0.98
        
        oi_signals[price_at_low & (oi_position > 0.4)] = 1
        oi_signals[price_at_high & (oi_position < 0.6)] = -1
    
    # Funding Rate
    funding_signals = pd.Series(0, index=df.index)
    if 'funding_rate' in df.columns and not df['funding_rate'].isna().all():
        funding = df['funding_rate'].ffill()
        funding_signals[funding < short_threshold] = 1
        funding_signals[funding > long_threshold] = -1
    
    # Both must agree
    signals[(oi_signals == 1) & (funding_signals == 1)] = 1
    signals[(oi_signals == -1) & (funding_signals == -1)] = -1
    return signals
