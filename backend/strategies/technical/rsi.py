"""RSI Overbought/Oversold Strategy"""
import pandas as pd

NAME = "RSI"
CATEGORY = "price_based"
DESCRIPTION = "RSI oversold (<30) / overbought (>70)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """RSI reversal signals."""
    signals = pd.Series(0, index=df.index)
    
    delta = df['close'].diff()
    # Use Wilder's smoothing (alpha=1/period) for proper RSI
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals[rsi < oversold] = 1
    signals[rsi > overbought] = -1
    return signals
