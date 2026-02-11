"""SMA Crossover Strategy"""
import pandas as pd

NAME = "SMA_Cross"
CATEGORY = "price_based"
DESCRIPTION = "SMA 10/30 crossover"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """SMA crossover signals."""
    signals = pd.Series(0, index=df.index)
    
    sma_fast = df['close'].rolling(fast).mean()
    sma_slow = df['close'].rolling(slow).mean()
    
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1
    return signals
