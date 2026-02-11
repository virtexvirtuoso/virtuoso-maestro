"""Funding Rate Strategy"""
import pandas as pd

NAME = "FundingRate"
CATEGORY = "funding"
DESCRIPTION = "Extreme funding rate reversal"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, short_threshold: float = -0.0001, long_threshold: float = 0.0005) -> pd.Series:
    """Extreme funding rate reversal signals."""
    signals = pd.Series(0, index=df.index)
    
    if 'funding_rate' not in df.columns or df['funding_rate'].isna().all():
        # Proxy from price action
        returns = df['close'].pct_change(8)
        vol_ratio = df['volume'] / df['volume'].rolling(24).mean()
        funding_proxy = returns * vol_ratio * 0.1
        
        signals[funding_proxy < short_threshold * 10] = 1
        signals[funding_proxy > long_threshold * 10] = -1
        return signals
    
    funding = df['funding_rate'].ffill()
    signals[funding < short_threshold] = 1
    signals[funding > long_threshold] = -1
    return signals
