"""Volume Breakout Strategy"""
import pandas as pd

NAME = "VolumeBreakout"
CATEGORY = "price_based"
DESCRIPTION = "Price breakout with 2x volume spike"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, vol_mult: float = 2.0, period: int = 20) -> pd.Series:
    """Breakout with volume confirmation."""
    signals = pd.Series(0, index=df.index)
    
    vol_avg = df['volume'].rolling(period).mean()
    high_vol = df['volume'] > vol_avg * vol_mult
    
    price_up = df['close'] > df['close'].shift(1)
    price_down = df['close'] < df['close'].shift(1)
    
    signals[high_vol & price_up] = 1
    signals[high_vol & price_down] = -1
    return signals
