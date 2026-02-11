"""Price Momentum Strategy"""
import pandas as pd

NAME = "Momentum"
CATEGORY = "price_based"
DESCRIPTION = "Price momentum over 14 periods"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Momentum signals."""
    signals = pd.Series(0, index=df.index)
    
    mom = df['close'].pct_change(period)
    
    signals[mom > 0] = 1
    signals[mom < 0] = -1
    return signals
