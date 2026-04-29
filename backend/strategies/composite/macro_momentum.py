"""
MacroMomentum Strategy (S5 Full System)
Macro-aware crypto momentum with trailing stop.

Entry: trend (close > SMA_slow) AND momentum (ROC > threshold) AND golden cross (SMA_fast > SMA_slow)
Exit: trailing stop from peak
Position sizing: scaled by macro score (0-6)
"""
import numpy as np
import pandas as pd

NAME = "MacroMomentum"
CATEGORY = "composite"
DESCRIPTION = "Macro-aware crypto momentum with trailing stop and position sizing"
REQUIRES_DERIVATIVES = False

DEFAULT_PARAMS = dict(
    sma_slow=200,
    sma_fast=50,
    momentum_period=30,
    trailing_stop_pct=0.15,
    roc_threshold=0.0,
)


def generate_signals(
    df: pd.DataFrame,
    sma_slow: int = 200,
    sma_fast: int = 50,
    momentum_period: int = 30,
    trailing_stop_pct: float = 0.15,
    roc_threshold: float = 0.0,
    **kwargs,
) -> pd.Series:
    """
    Generate {1, 0, -1} signals. 1 = long, -1 = exit/flat, 0 = hold previous.
    Shifted by 1 to avoid lookahead bias (signal on bar N trades on bar N+1).
    """
    close = df["close"].copy()
    n = len(close)

    sma_s = close.rolling(sma_slow).mean()
    sma_f = close.rolling(sma_fast).mean()
    roc = close.pct_change(momentum_period)

    # Entry conditions (all must be true)
    trend_up = close > sma_s
    momentum_pos = roc > roc_threshold
    golden_cross = sma_f > sma_s

    entry_cond = trend_up & momentum_pos & golden_cross

    # Build signals with trailing stop logic
    signals = pd.Series(0, index=df.index, dtype=int)
    in_position = False
    peak = 0.0

    for i in range(sma_slow, n):
        if not in_position:
            if entry_cond.iloc[i]:
                signals.iloc[i] = 1
                in_position = True
                peak = close.iloc[i]
        else:
            peak = max(peak, close.iloc[i])
            stop_price = peak * (1 - trailing_stop_pct)
            if close.iloc[i] < stop_price:
                signals.iloc[i] = -1
                in_position = False
            else:
                signals.iloc[i] = 1

    # Shift by 1 to avoid lookahead bias
    signals = signals.shift(1).fillna(0).astype(int)
    return signals


def position_size(
    signal: pd.Series,
    macro_score: pd.Series,
    threshold_high: int = 5,
    threshold_mid: int = 3,
    threshold_low: int = 1,
    size_high: float = 1.0,
    size_mid: float = 0.6,
    size_low: float = 0.3,
) -> pd.Series:
    """
    Scale position size by macro score.
    Returns pd.Series of position sizes [0.0, 1.0].
    """
    # Align indices
    score = macro_score.reindex(signal.index, method="ffill").fillna(0)

    size = pd.Series(0.0, index=signal.index)
    size[score >= threshold_high] = size_high
    size[(score >= threshold_mid) & (score < threshold_high)] = size_mid
    size[(score >= threshold_low) & (score < threshold_mid)] = size_low

    # Only apply when signal is long
    size[signal != 1] = 0.0
    return size
