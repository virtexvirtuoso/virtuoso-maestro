"""CVD Scalp Strategy"""
import pandas as pd
import numpy as np

NAME = "CVDScalp"
CATEGORY = "derivatives"
DESCRIPTION = "CVD divergence scalping"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """CVD scalp signals."""
    signals = pd.Series(0, index=df.index)
    
    # Calculate CVD if not present
    if 'cvd' in df.columns:
        cvd = df['cvd']
    else:
        cvd = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # CVD momentum
    cvd_mom = cvd.diff(lookback)
    price_mom = df['close'].pct_change(lookback)
    
    # Divergence
    cvd_up = cvd_mom > 0
    cvd_down = cvd_mom < 0
    price_up = price_mom > 0
    price_down = price_mom < 0
    
    # Bullish div: price down, CVD up
    signals[price_down & cvd_up] = 1
    # Bearish div: price up, CVD down
    signals[price_up & cvd_down] = -1
    
    return signals
