"""Bollinger + OBV Hybrid"""
import pandas as pd
import numpy as np

NAME = "BollingerBreakout+OBV"
CATEGORY = "hybrid"
DESCRIPTION = "Bollinger breakout with OBV confirmation"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, obv_period: int = 20) -> pd.Series:
    """Bollinger breakout with OBV confirmation."""
    signals = pd.Series(0, index=df.index)
    
    # Bollinger
    sma = df['close'].rolling(bb_period).mean()
    std_dev = df['close'].rolling(bb_period).std()
    upper = sma + bb_std * std_dev
    lower = sma - bb_std * std_dev
    
    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(obv_period).mean()
    
    signals[(df['close'] > upper) & (obv > obv_sma)] = 1
    signals[(df['close'] < lower) & (obv < obv_sma)] = -1
    return signals
