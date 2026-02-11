"""MACD Crossover Strategy"""
import pandas as pd

NAME = "MACD"
CATEGORY = "price_based"
DESCRIPTION = "MACD line crosses signal line"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD crossover signals."""
    signals = pd.Series(0, index=df.index)
    
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    
    signals[macd_line > signal_line] = 1
    signals[macd_line < signal_line] = -1
    return signals
