"""Awesome MACD Strategy"""
import pandas as pd

NAME = "AwesomeMACD"
CATEGORY = "technical"
DESCRIPTION = "MACD + Awesome Oscillator confluence"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, ao_fast: int = 5, ao_slow: int = 34) -> pd.Series:
    """Awesome MACD signals."""
    signals = pd.Series(0, index=df.index)
    
    # MACD
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    
    # Awesome Oscillator (median price based)
    median = (df['high'] + df['low']) / 2
    ao = median.rolling(ao_fast).mean() - median.rolling(ao_slow).mean()
    
    # Confluence
    signals[(macd_hist > 0) & (ao > 0)] = 1
    signals[(macd_hist < 0) & (ao < 0)] = -1
    
    return signals
