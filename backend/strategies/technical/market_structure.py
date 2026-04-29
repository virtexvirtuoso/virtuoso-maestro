"""Market Structure — HH/HL/LH/LL + BOS/CHoCH (HSAKA/ICT)

Detects swing highs and lows, classifies market structure as uptrend (HH+HL)
or downtrend (LH+LL), then generates signals on Break of Structure (BOS) for
continuation or Change of Character (CHoCH) for reversal.

ICT Concept: Market structure is the backbone of price action. A series of
higher highs and higher lows defines bullish structure; lower highs and lower
lows defines bearish structure. BOS confirms trend continuation while CHoCH
signals potential reversal.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

NAME = "MarketStructure"
CATEGORY = "price_structure"
DESCRIPTION = "HH/HL/LH/LL structure with BOS and CHoCH detection"
REQUIRES_DERIVATIVES = False


def generate_signals(df: pd.DataFrame, swing_window: int = 5, min_swings: int = 3) -> pd.Series:
    """Generate +1 (long), -1 (short), 0 (flat) signals based on market structure."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < swing_window * 2 + 1:
        return signals

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Detect swing highs and lows
    swing_high_idx = argrelextrema(high, np.greater_equal, order=swing_window)[0]
    swing_low_idx = argrelextrema(low, np.less_equal, order=swing_window)[0]

    if len(swing_high_idx) < min_swings or len(swing_low_idx) < min_swings:
        return signals

    # Build combined swing list: (index, price, type)
    swings = []
    for i in swing_high_idx:
        swings.append((i, high[i], 'H'))
    for i in swing_low_idx:
        swings.append((i, low[i], 'L'))
    swings.sort(key=lambda x: x[0])

    # Track structure
    prev_swing_high = None
    prev_swing_low = None
    trend = 0  # 1=up, -1=down, 0=undefined
    prev_trend = 0

    for idx, price, stype in swings:
        if stype == 'H':
            if prev_swing_high is not None:
                if price > prev_swing_high:  # Higher High
                    new_trend = 1
                else:  # Lower High
                    new_trend = -1

                # Detect CHoCH (trend reversal) or BOS (continuation)
                if prev_trend != 0 and new_trend != prev_trend:
                    # CHoCH — first opposing break
                    signals.iloc[idx] = new_trend
                elif new_trend == trend:
                    # BOS — continuation
                    signals.iloc[idx] = new_trend

                prev_trend = trend
                trend = new_trend
            prev_swing_high = price

        elif stype == 'L':
            if prev_swing_low is not None:
                if price > prev_swing_low:  # Higher Low
                    new_trend = 1
                else:  # Lower Low
                    new_trend = -1

                if prev_trend != 0 and new_trend != prev_trend:
                    signals.iloc[idx] = new_trend
                elif new_trend == trend:
                    signals.iloc[idx] = new_trend

                prev_trend = trend
                trend = new_trend
            prev_swing_low = price

    # Forward-fill signals to maintain position
    signals = signals.replace(0, np.nan).ffill().fillna(0).astype(int)
    return signals
