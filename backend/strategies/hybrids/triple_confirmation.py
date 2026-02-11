"""Triple Confirmation Hybrid"""
import pandas as pd

NAME = "TripleConfirmation"
CATEGORY = "hybrid"
DESCRIPTION = "MACD + RSI + Volume all agree"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, rsi_period: int = 14, oversold: int = 30, overbought: int = 70, vol_mult: float = 2.0, vol_period: int = 20, rsi_lookback: int = 5) -> pd.Series:
    """MACD + RSI + Volume triple confirmation - uses RSI recovery pattern."""
    signals = pd.Series(0, index=df.index)

    # MACD calculation
    ema_fast = df['close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['close'].ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    macd_long = macd_line > signal_line
    macd_short = macd_line < signal_line

    # RSI with Wilder's smoothing
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # RSI recovery pattern (avoids indicator conflict)
    was_oversold = rsi.rolling(rsi_lookback).min() < oversold
    rsi_recovering = (rsi > oversold) & was_oversold

    was_overbought = rsi.rolling(rsi_lookback).max() > overbought
    rsi_declining = (rsi < overbought) & was_overbought

    # Volume filter
    vol_avg = df['volume'].rolling(vol_period, min_periods=5).mean()
    vol_filter = df['volume'] > vol_avg * vol_mult

    signals[macd_long & rsi_recovering & vol_filter] = 1
    signals[macd_short & rsi_declining & vol_filter] = -1
    return signals
