"""OBV + Trend Filter Hybrid"""
import pandas as pd
import numpy as np

NAME = "OBV+TrendFilter"
CATEGORY = "hybrid"
DESCRIPTION = "OBV divergence aligned with trend"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, obv_period: int = 20, trend_period: int = 50) -> pd.Series:
    """OBV with trend filter."""
    signals = pd.Series(0, index=df.index)
    
    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(obv_period).mean()
    
    # Trend filter
    sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > sma
    downtrend = df['close'] < sma
    
    # Only take signals in trend direction
    signals[(obv > obv_sma) & uptrend] = 1
    signals[(obv < obv_sma) & downtrend] = -1
    return signals
