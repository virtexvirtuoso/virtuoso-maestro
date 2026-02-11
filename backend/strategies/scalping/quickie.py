"""Quickie Strategy"""
import pandas as pd

NAME = "Quickie"
CATEGORY = "scalping"
DESCRIPTION = "Quick mean reversion scalp"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 10, threshold: float = 1.5) -> pd.Series:
    """Quickie signals."""
    signals = pd.Series(0, index=df.index)
    
    # Fast z-score
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    zscore = (df['close'] - sma) / std.replace(0, 1e-10)
    
    # Quick reversion
    signals[zscore < -threshold] = 1
    signals[zscore > threshold] = -1
    
    return signals
