"""Fernando Strategy (Glucksmann Thesis)"""
import pandas as pd

NAME = "Fernando"
CATEGORY = "composite"
DESCRIPTION = "BBW + VLI from ETH Zurich thesis"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, bb_period: int = 20, vli_fast: int = 20, vli_slow: int = 100) -> pd.Series:
    """Fernando strategy signals."""
    signals = pd.Series(0, index=df.index)
    
    # Bollinger Bands
    sma = df['close'].rolling(bb_period).mean()
    std = df['close'].rolling(bb_period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # Bollinger Band Width (BBW)
    bbw = (upper - lower) / sma
    
    # Volatility Level Index (VLI)
    vli_fast_line = bbw.rolling(vli_fast).mean()
    vli_slow_line = bbw.rolling(vli_slow).mean()
    
    # Low volatility environment
    low_vol = vli_fast_line < vli_slow_line
    
    # Volume condition
    vol_sma_10 = df['volume'].rolling(10).mean()
    vol_sma_50 = df['volume'].rolling(50).mean()
    vol_condition = vol_sma_10 > vol_sma_50
    
    # Entry: Low vol + volume surge + price at bands
    signals[low_vol & vol_condition & (df['close'] < lower)] = 1
    signals[low_vol & vol_condition & (df['close'] > upper)] = -1
    
    return signals
