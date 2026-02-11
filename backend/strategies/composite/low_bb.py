"""Low BB Strategy"""
import pandas as pd

NAME = "LowBB"
CATEGORY = "composite"
DESCRIPTION = "Buy at lower Bollinger Band"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    """Low BB signals."""
    signals = pd.Series(0, index=df.index)
    
    sma = df['close'].rolling(period).mean()
    std_dev = df['close'].rolling(period).std()
    lower = sma - std * std_dev
    upper = sma + std * std_dev
    
    # Buy at lower band, sell at upper
    signals[df['close'] <= lower] = 1
    signals[df['close'] >= upper] = -1
    
    return signals
