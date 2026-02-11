"""EMA Crossover Strategy"""
import pandas as pd

NAME = "EMA_Cross"
CATEGORY = "price_based"
DESCRIPTION = "EMA 12/26 crossover"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """EMA crossover signals."""
    signals = pd.Series(0, index=df.index)
    
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    
    signals[ema_fast > ema_slow] = 1
    signals[ema_fast < ema_slow] = -1
    return signals
