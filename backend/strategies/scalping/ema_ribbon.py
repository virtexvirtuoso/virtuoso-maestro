"""EMA Ribbon Strategy"""
import pandas as pd

NAME = "EMARibbon"
CATEGORY = "scalping"
DESCRIPTION = "EMA Ribbon trend following"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """EMA Ribbon signals."""
    signals = pd.Series(0, index=df.index)
    
    # EMA Ribbon
    ema_8 = df['close'].ewm(span=8).mean()
    ema_13 = df['close'].ewm(span=13).mean()
    ema_21 = df['close'].ewm(span=21).mean()
    ema_34 = df['close'].ewm(span=34).mean()
    ema_55 = df['close'].ewm(span=55).mean()
    
    # All EMAs aligned
    bullish = (ema_8 > ema_13) & (ema_13 > ema_21) & (ema_21 > ema_34) & (ema_34 > ema_55)
    bearish = (ema_8 < ema_13) & (ema_13 < ema_21) & (ema_21 < ema_34) & (ema_34 < ema_55)
    
    signals[bullish] = 1
    signals[bearish] = -1
    
    return signals
