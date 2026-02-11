"""MACD + Volume Filter Hybrid"""
import pandas as pd

NAME = "MACD+VolumeFilter"
CATEGORY = "hybrid"
DESCRIPTION = "MACD crossover with high volume confirmation"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, vol_mult: float = 2.0, period: int = 20) -> pd.Series:
    """MACD with volume confirmation."""
    signals = pd.Series(0, index=df.index)
    
    # MACD
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    
    signals[macd_line > signal_line] = 1
    signals[macd_line < signal_line] = -1
    
    # Volume filter
    vol_avg = df['volume'].rolling(period).mean()
    vol_filter = df['volume'] > vol_avg * vol_mult
    signals[~vol_filter] = 0
    
    return signals
