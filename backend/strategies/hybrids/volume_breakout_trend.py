"""Volume Breakout + Trend Filter Hybrid"""
import pandas as pd

NAME = "VolumeBreakout+TrendFilter"
CATEGORY = "hybrid"
DESCRIPTION = "Volume breakout in trend direction"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, vol_mult: float = 2.0, vol_period: int = 20, trend_period: int = 50) -> pd.Series:
    """Volume breakout with trend filter."""
    signals = pd.Series(0, index=df.index)
    
    # Volume breakout
    vol_avg = df['volume'].rolling(vol_period).mean()
    high_vol = df['volume'] > vol_avg * vol_mult
    
    # Trend filter
    sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > sma
    downtrend = df['close'] < sma
    
    price_up = df['close'] > df['close'].shift(1)
    price_down = df['close'] < df['close'].shift(1)
    
    signals[high_vol & price_up & uptrend] = 1
    signals[high_vol & price_down & downtrend] = -1
    return signals
