"""Smooth Operator Strategy"""
import pandas as pd

NAME = "SmoothOperator"
CATEGORY = "composite"
DESCRIPTION = "Smoothed multi-indicator"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Smooth Operator signals."""
    signals = pd.Series(0, index=df.index)
    
    # Smoothed EMAs
    ema_8 = df['close'].ewm(span=8).mean()
    ema_21 = df['close'].ewm(span=21).mean()
    
    # Smoothed RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    macd_signal = macd.ewm(span=9).mean()
    
    # Confluence
    signals[(ema_8 > ema_21) & (rsi > 50) & (macd > macd_signal)] = 1
    signals[(ema_8 < ema_21) & (rsi < 50) & (macd < macd_signal)] = -1
    
    return signals
