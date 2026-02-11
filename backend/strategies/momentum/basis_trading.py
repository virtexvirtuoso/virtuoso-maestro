"""Basis Trading Strategy"""
import pandas as pd

NAME = "BasisTrading"
CATEGORY = "momentum"
DESCRIPTION = "Spot-futures basis trading"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, lookback: int = 24) -> pd.Series:
    """Basis trading signals."""
    signals = pd.Series(0, index=df.index)
    
    # Proxy basis from funding rate if available
    if 'funding_rate' in df.columns:
        funding = df['funding_rate'].fillna(0)
        funding_ma = funding.rolling(lookback).mean()
        
        # Fade extreme funding
        signals[funding > funding_ma + funding.rolling(lookback).std() * 2] = -1
        signals[funding < funding_ma - funding.rolling(lookback).std() * 2] = 1
    else:
        # Use price momentum as proxy
        returns = df['close'].pct_change(lookback)
        signals[returns > 0.1] = -1  # Overextended
        signals[returns < -0.1] = 1  # Oversold
    
    return signals
