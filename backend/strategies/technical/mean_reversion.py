"""Mean Reversion Strategy"""
import pandas as pd

NAME = "MeanReversion"
CATEGORY = "price_based"
DESCRIPTION = "Z-score mean reversion (2 std)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 20, threshold: float = 2.0) -> pd.Series:
    """Mean reversion signals."""
    signals = pd.Series(0, index=df.index)
    
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    zscore = (df['close'] - sma) / std.replace(0, 1e-10)
    
    signals[zscore < -threshold] = 1   # Oversold
    signals[zscore > threshold] = -1   # Overbought
    return signals
