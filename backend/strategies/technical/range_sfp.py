"""Range Quarter Scoring + Swing Failure Pattern (SFP) + MSB (HSAKA/ICT)

Identifies consolidation ranges, scores price position within quarters,
and detects Swing Failure Patterns where price sweeps a range boundary
but fails to hold, indicating a liquidity grab and potential reversal.

ICT Concept: Smart money hunts stop losses beyond range boundaries. When
price sweeps above/below a range and closes back inside, it signals a
failed breakout (SFP) — a high-probability reversal entry.
"""
import numpy as np
import pandas as pd

NAME = "RangeSFP"
CATEGORY = "price_structure"
DESCRIPTION = "Range quarter scoring with Swing Failure Pattern detection"
REQUIRES_DERIVATIVES = False


def generate_signals(df: pd.DataFrame, lookback: int = 50, sfp_threshold: float = 0.005,
                     atr_period: int = 14) -> pd.Series:
    """Generate signals based on SFP at range boundaries."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < max(lookback, atr_period + 1):
        return signals

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    op = df['open'].values

    # ATR for range validation
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = pd.Series(tr).rolling(atr_period).mean().values

    for i in range(lookback, n):
        window_high = np.max(high[i - lookback:i])
        window_low = np.min(low[i - lookback:i])
        range_width = window_high - window_low

        # Validate range width >= 2x ATR
        if np.isnan(atr[i]) or range_width < 2 * atr[i]:
            continue

        threshold_abs = window_high * sfp_threshold

        # Bearish SFP: high sweeps above range, close back inside
        if high[i] > window_high + threshold_abs and close[i] <= window_high:
            signals.iloc[i] = -1
        # Bullish SFP: low sweeps below range, close back inside
        elif low[i] < window_low - threshold_abs and close[i] >= window_low:
            signals.iloc[i] = 1
        else:
            # Quarter scoring fallback
            q1 = window_low + range_width * 0.25
            q4 = window_low + range_width * 0.75
            if close[i] <= q1:
                signals.iloc[i] = 1
            elif close[i] >= q4:
                signals.iloc[i] = -1

    return signals
