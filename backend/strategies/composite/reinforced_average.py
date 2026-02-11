"""Reinforced Average Strategy"""
import pandas as pd

NAME = "ReinforcedAverage"
CATEGORY = "composite"
DESCRIPTION = "Multiple MA confluence"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Reinforced Average signals."""
    signals = pd.Series(0, index=df.index)
    
    # Multiple SMAs
    sma_5 = df['close'].rolling(5).mean()
    sma_10 = df['close'].rolling(10).mean()
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    
    # All aligned bullish
    bullish = (sma_5 > sma_10) & (sma_10 > sma_20) & (sma_20 > sma_50)
    bearish = (sma_5 < sma_10) & (sma_10 < sma_20) & (sma_20 < sma_50)
    
    # Price above/below all
    above_all = (df['close'] > sma_5) & (df['close'] > sma_50)
    below_all = (df['close'] < sma_5) & (df['close'] < sma_50)
    
    signals[bullish & above_all] = 1
    signals[bearish & below_all] = -1
    
    return signals
