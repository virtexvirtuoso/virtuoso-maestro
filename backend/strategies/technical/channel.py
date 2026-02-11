"""Donchian Channel Strategy"""
import pandas as pd

NAME = "Channel"
CATEGORY = "technical"
DESCRIPTION = "Donchian Channel breakout"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Donchian Channel signals."""
    signals = pd.Series(0, index=df.index)
    
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    
    # Breakout signals
    signals[df['close'] > upper.shift(1)] = 1
    signals[df['close'] < lower.shift(1)] = -1
    
    return signals
