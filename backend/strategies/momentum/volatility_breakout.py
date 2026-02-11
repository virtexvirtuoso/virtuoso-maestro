"""Volatility Breakout Strategy"""
import pandas as pd

NAME = "VolatilityBreakout"
CATEGORY = "momentum"
DESCRIPTION = "Volatility expansion breakout"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, atr_period: int = 14, threshold: float = 1.5) -> pd.Series:
    """Volatility breakout signals."""
    signals = pd.Series(0, index=df.index)
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    atr_avg = atr.rolling(50).mean()
    
    # Volatility expansion
    vol_expanding = atr > atr_avg * threshold
    
    # Direction
    mom = df['close'].pct_change(5)
    
    signals[vol_expanding & (mom > 0)] = 1
    signals[vol_expanding & (mom < 0)] = -1
    
    return signals
