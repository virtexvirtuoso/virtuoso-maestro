"""Smooth Scalp Strategy"""
import pandas as pd

NAME = "SmoothScalp"
CATEGORY = "scalping"
DESCRIPTION = "Smoothed indicators for scalping"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 5, slow: int = 13) -> pd.Series:
    """Smooth scalp signals."""
    signals = pd.Series(0, index=df.index)
    
    # Smoothed averages
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    
    # Smoothed RSI with Wilder's smoothing
    rsi_period = 7
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Combined
    signals[(ema_fast > ema_slow) & (rsi < 60)] = 1
    signals[(ema_fast < ema_slow) & (rsi > 40)] = -1
    
    return signals
