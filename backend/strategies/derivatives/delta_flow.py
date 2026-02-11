"""Delta Flow Strategy"""
import pandas as pd
import numpy as np

NAME = "DeltaFlow"
CATEGORY = "derivatives"
DESCRIPTION = "Order flow delta analysis"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Delta flow signals."""
    signals = pd.Series(0, index=df.index)
    
    # Use buy/sell volume if available
    if 'buy_vol' in df.columns and 'sell_vol' in df.columns:
        delta = df['buy_vol'] - df['sell_vol']
    else:
        # Proxy from price action
        delta = np.sign(df['close'].diff()) * df['volume']
    
    delta_ma = delta.rolling(lookback).mean()
    delta_std = delta.rolling(lookback).std()
    
    # Strong buying/selling pressure
    signals[delta > delta_ma + delta_std] = 1
    signals[delta < delta_ma - delta_std] = -1
    
    return signals
