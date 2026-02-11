"""EMA Skip Pump Strategy"""
import pandas as pd

NAME = "EMASkipPump"
CATEGORY = "composite"
DESCRIPTION = "EMA cross with pump filter"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, fast: int = 12, slow: int = 26, pump_threshold: float = 0.05) -> pd.Series:
    """EMA Skip Pump signals - detects actual crossover events."""
    signals = pd.Series(0, index=df.index)

    ema_fast = df['close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['close'].ewm(span=slow, min_periods=slow).mean()

    # Detect pumps to skip
    returns = df['close'].pct_change()
    is_pump = returns > pump_threshold
    is_dump = returns < -pump_threshold

    # Detect actual crossover events (relationship changed from previous bar)
    fast_above_slow = ema_fast > ema_slow
    fast_above_slow_prev = fast_above_slow.shift(1).astype(bool).fillna(False)

    bullish_cross = fast_above_slow & ~fast_above_slow_prev  # Just crossed above
    bearish_cross = ~fast_above_slow & fast_above_slow_prev  # Just crossed below

    # Signal on crossovers, but skip extreme moves
    signals[bullish_cross & ~is_pump] = 1
    signals[bearish_cross & ~is_dump] = -1

    return signals
