"""EMA Cross + Volume Filter Hybrid"""
import pandas as pd

NAME = "EMA_Cross+VolumeFilter"
CATEGORY = "hybrid"
DESCRIPTION = "EMA crossover with high volume confirmation"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, vol_mult: float = 2.0, period: int = 20) -> pd.Series:
    """EMA crossover with volume confirmation - detects actual cross events."""
    signals = pd.Series(0, index=df.index)

    # EMA calculations
    ema_fast = df['close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['close'].ewm(span=slow, min_periods=slow).mean()

    # Detect actual crossover events (not just current relationship)
    fast_above_slow = ema_fast > ema_slow
    fast_above_slow_prev = fast_above_slow.shift(1).astype(bool).fillna(False)

    bullish_cross = fast_above_slow & ~fast_above_slow_prev  # Just crossed above
    bearish_cross = ~fast_above_slow & fast_above_slow_prev  # Just crossed below

    # Volume filter - only signal on high volume crossovers
    vol_avg = df['volume'].rolling(period, min_periods=5).mean()
    vol_filter = df['volume'] > vol_avg * vol_mult

    signals[bullish_cross & vol_filter] = 1
    signals[bearish_cross & vol_filter] = -1

    return signals
