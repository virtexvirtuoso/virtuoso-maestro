"""ADX with SMAs Strategy"""
import pandas as pd
import numpy as np

NAME = "ADXSmas"
CATEGORY = "technical"
DESCRIPTION = "ADX trend filter with SMA crossover"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, adx_period: int = 14, sma_fast: int = 10, sma_slow: int = 30, adx_threshold: float = 20) -> pd.Series:
    """ADX + SMA signals."""
    signals = pd.Series(0, index=df.index)
    
    # ADX calculation
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean()
    
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
    
    plus_di = 100 * plus_dm.rolling(adx_period).mean() / atr
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(adx_period).mean()
    
    # SMAs
    sma_fast_line = close.rolling(sma_fast).mean()
    sma_slow_line = close.rolling(sma_slow).mean()
    
    # Trend confirmed by ADX
    trending = adx > adx_threshold
    signals[trending & (sma_fast_line > sma_slow_line)] = 1
    signals[trending & (sma_fast_line < sma_slow_line)] = -1
    
    return signals
