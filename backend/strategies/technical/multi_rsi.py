"""Multi-Timeframe RSI Strategy"""
import pandas as pd

NAME = "MultiRSI"
CATEGORY = "technical"
DESCRIPTION = "Multiple RSI periods for confluence"
REQUIRES_DERIVATIVES = False

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    # Use Wilder's smoothing (alpha=1/period) for proper RSI
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def generate_signals(df: pd.DataFrame, fast: int = 7, mid: int = 14, slow: int = 21) -> pd.Series:
    """Multi RSI signals."""
    signals = pd.Series(0, index=df.index)
    
    rsi_fast = _rsi(df['close'], fast)
    rsi_mid = _rsi(df['close'], mid)
    rsi_slow = _rsi(df['close'], slow)
    
    # All RSIs agree
    all_oversold = (rsi_fast < 30) & (rsi_mid < 35) & (rsi_slow < 40)
    all_overbought = (rsi_fast > 70) & (rsi_mid > 65) & (rsi_slow > 60)
    
    signals[all_oversold] = 1
    signals[all_overbought] = -1
    
    return signals
