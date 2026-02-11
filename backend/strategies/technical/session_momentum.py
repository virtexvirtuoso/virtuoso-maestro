"""Session Momentum Strategy"""
import pandas as pd

NAME = "SessionMomentum"
CATEGORY = "session"
DESCRIPTION = "London/NY session momentum"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Session open momentum signals."""
    signals = pd.Series(0, index=df.index)
    
    if not hasattr(df.index, 'hour'):
        return signals
    
    hour = df.index.hour
    london_open = (hour >= 8) & (hour < 10)
    ny_open = (hour >= 13) & (hour < 15)
    
    momentum = df['close'].pct_change(3)
    
    signals[(london_open | ny_open) & (momentum > 0.005)] = 1
    signals[(london_open | ny_open) & (momentum < -0.005)] = -1
    return signals
