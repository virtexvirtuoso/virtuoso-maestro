"""Trend Following ATR Strategy"""
import pandas as pd

NAME = "TrendFollowingATR"
CATEGORY = "momentum"
DESCRIPTION = "ATR-based trend following"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, atr_period: int = 14, atr_mult: float = 2.0, trend_period: int = 50) -> pd.Series:
    """Trend Following ATR signals with proper breakout detection."""
    signals = pd.Series(0, index=df.index)

    # ATR calculation
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=5).mean()

    # Trend filter
    sma = df['close'].rolling(trend_period, min_periods=10).mean()
    uptrend = df['close'] > sma
    downtrend = df['close'] < sma

    # ATR channel breakout (Keltner-style)
    # Use rolling high/low for channel, not current close
    channel_high = df['high'].rolling(atr_period).max()
    channel_low = df['low'].rolling(atr_period).min()

    # Breakout signals: price breaks above/below channel + ATR buffer
    breakout_up = df['close'] > (channel_high.shift(1) + atr_mult * atr * 0.5)
    breakout_down = df['close'] < (channel_low.shift(1) - atr_mult * atr * 0.5)

    # Long: uptrend + breakout above recent highs
    signals[uptrend & breakout_up] = 1
    # Short: downtrend + breakout below recent lows
    signals[downtrend & breakout_down] = -1

    return signals
