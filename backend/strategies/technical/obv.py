"""On-Balance Volume Strategy"""
import pandas as pd
import numpy as np

NAME = "OBV"
CATEGORY = "price_based"
DESCRIPTION = "On-Balance Volume trend"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """OBV trend signals."""
    signals = pd.Series(0, index=df.index)
    
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(period).mean()
    
    signals[obv > obv_sma] = 1
    signals[obv < obv_sma] = -1
    return signals
