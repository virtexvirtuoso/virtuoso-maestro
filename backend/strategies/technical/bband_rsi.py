"""Bollinger Bands + RSI Strategy"""
import pandas as pd

NAME = "BBandRSI"
CATEGORY = "technical"
DESCRIPTION = "Bollinger Bands with RSI confirmation"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """BBand + RSI signals."""
    signals = pd.Series(0, index=df.index)
    
    # Bollinger Bands
    sma = df['close'].rolling(bb_period).mean()
    std = df['close'].rolling(bb_period).std()
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    
    # RSI with Wilder's smoothing
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Long: price at lower band + RSI oversold
    signals[(df['close'] <= lower) & (rsi < oversold)] = 1
    # Short: price at upper band + RSI overbought
    signals[(df['close'] >= upper) & (rsi > overbought)] = -1
    
    return signals
