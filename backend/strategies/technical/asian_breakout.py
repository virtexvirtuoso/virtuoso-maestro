"""Asian Breakout Strategy"""
import pandas as pd

NAME = "AsianBreakout"
CATEGORY = "session"
DESCRIPTION = "Asian range breakout"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Asian session range breakout signals."""
    signals = pd.Series(0, index=df.index)
    
    if not hasattr(df.index, 'hour'):
        return signals
    
    hour = df.index.hour
    asian = (hour >= 0) & (hour < 8)
    
    asian_high = df['high'].where(asian).ffill()
    asian_low = df['low'].where(asian).ffill()
    
    non_asian = ~asian
    price = df['close']
    
    signals[non_asian & (price > asian_high)] = 1
    signals[non_asian & (price < asian_low)] = -1
    return signals
