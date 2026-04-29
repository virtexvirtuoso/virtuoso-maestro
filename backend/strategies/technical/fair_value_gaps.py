"""Fair Value Gap (FVG) Detection + Fill Tracking (HSAKA/ICT)

Detects imbalances in price delivery where a 3-candle sequence leaves a gap
(candle i high vs candle i+2 low). Tracks unfilled FVGs and generates signals
when price returns to fill them — a mean-reversion play on institutional
order flow imbalances.

ICT Concept: Fair Value Gaps represent inefficient price delivery. Institutional
algorithms tend to revisit these gaps to rebalance order flow. Trading the fill
of these gaps provides high-probability entries.
"""
import numpy as np
import pandas as pd

NAME = "FairValueGaps"
CATEGORY = "price_structure"
DESCRIPTION = "FVG detection with fill-trade signals on gap revisitation"
REQUIRES_DERIVATIVES = False


def generate_signals(df: pd.DataFrame, lookback: int = 50, proximity_pct: float = 0.01,
                     max_gap_age: int = 20) -> pd.Series:
    """Generate signals when price approaches unfilled FVGs."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < 3:
        return signals

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Detect all FVGs (need i, i+1, i+2 so iterate to n-2)
    # bullish_fvgs: gap_top=low[i+2], gap_bottom=high[i] (gap up)
    # bearish_fvgs: gap_top=low[i], gap_bottom=high[i+2] (gap down)
    active_bull_gaps = []  # (gap_bottom, gap_top, birth_bar)
    active_bear_gaps = []

    for i in range(n - 2):
        # Bullish FVG: candle i+2 low > candle i high
        if low[i + 2] > high[i]:
            active_bull_gaps.append((high[i], low[i + 2], i + 2))
        # Bearish FVG: candle i+2 high < candle i low
        if high[i + 2] < low[i]:
            active_bear_gaps.append((high[i + 2], low[i], i + 2))

    # Now scan each bar for proximity to active unfilled gaps
    for i in range(2, n):
        best_signal = 0
        price = close[i]
        threshold = price * proximity_pct

        # Check bullish FVGs (price approaching from above = fill trade = long)
        for gap_bot, gap_top, birth in active_bull_gaps:
            age = i - birth
            if age < 0 or age > max_gap_age:
                continue
            # Check if already filled (price went through gap)
            # Simplified: if low ever touched gap_top area, it's filled
            if i > birth and np.any(low[birth + 1:i + 1] <= gap_bot):
                continue
            # Price approaching gap from above
            gap_mid = (gap_bot + gap_top) / 2
            if abs(price - gap_top) <= threshold or (gap_bot <= price <= gap_top):
                best_signal = 1
                break

        if best_signal == 0:
            # Check bearish FVGs (price approaching from below = fill trade = short)
            for gap_bot, gap_top, birth in active_bear_gaps:
                age = i - birth
                if age < 0 or age > max_gap_age:
                    continue
                if i > birth and np.any(high[birth + 1:i + 1] >= gap_top):
                    continue
                gap_mid = (gap_bot + gap_top) / 2
                if abs(price - gap_bot) <= threshold or (gap_bot <= price <= gap_top):
                    best_signal = -1
                    break

        signals.iloc[i] = best_signal

    return signals
