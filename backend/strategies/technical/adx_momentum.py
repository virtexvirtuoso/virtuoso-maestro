"""ADX Momentum Strategy"""
import pandas as pd
import numpy as np

NAME = "ADXMomentum"
CATEGORY = "technical"
DESCRIPTION = "ADX trend strength with momentum confirmation"
REQUIRES_DERIVATIVES = False

def _calc_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """Calculate ADX, +DI, -DI."""
    high, low, close = df['high'], df['low'], df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

def generate_signals(df: pd.DataFrame, period: int = 14, threshold: float = 25) -> pd.Series:
    """ADX momentum signals."""
    signals = pd.Series(0, index=df.index)
    
    adx, plus_di, minus_di = _calc_adx(df, period)
    
    # Strong trend + directional
    strong_trend = adx > threshold
    signals[strong_trend & (plus_di > minus_di)] = 1
    signals[strong_trend & (minus_di > plus_di)] = -1
    
    return signals
