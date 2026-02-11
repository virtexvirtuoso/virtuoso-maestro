"""Scalp RSI Strategy"""
import pandas as pd

NAME = "ScalpRSI"
CATEGORY = "scalping"
DESCRIPTION = "Fast RSI scalping"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 7, oversold: int = 25, overbought: int = 75) -> pd.Series:
    """Scalp RSI signals."""
    signals = pd.Series(0, index=df.index)
    
    delta = df['close'].diff()
    # Use Wilder's smoothing (alpha=1/period) for proper RSI
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Quick reversals
    signals[rsi < oversold] = 1
    signals[rsi > overbought] = -1
    
    return signals
