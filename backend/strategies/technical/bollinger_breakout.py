"""Bollinger Bands Breakout Strategy"""
import pandas as pd

NAME = "BollingerBreakout"
CATEGORY = "price_based"
DESCRIPTION = "Breakout from Bollinger Bands (2 std)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    """Long on upper band break, short on lower."""
    signals = pd.Series(0, index=df.index)
    
    sma = df['close'].rolling(period).mean()
    std_dev = df['close'].rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    
    signals[df['close'] > upper] = 1
    signals[df['close'] < lower] = -1
    return signals
