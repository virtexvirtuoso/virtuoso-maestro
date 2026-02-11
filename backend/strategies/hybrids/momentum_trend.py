"""Momentum + Trend Filter Hybrid"""
import pandas as pd

NAME = "Momentum+TrendFilter"
CATEGORY = "hybrid"
DESCRIPTION = "Momentum aligned with 50 SMA trend"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, mom_period: int = 14, trend_period: int = 50) -> pd.Series:
    """Momentum with trend filter."""
    signals = pd.Series(0, index=df.index)
    
    # Momentum
    mom = df['close'].pct_change(mom_period)
    
    # Trend filter
    sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > sma
    downtrend = df['close'] < sma
    
    signals[(mom > 0) & uptrend] = 1
    signals[(mom < 0) & downtrend] = -1
    return signals
