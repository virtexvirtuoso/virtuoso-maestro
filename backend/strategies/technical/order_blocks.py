"""Order Blocks — Institutional Zone Detection (HSAKA/ICT)

Identifies Order Blocks: the last opposing candle before a strong impulsive
move, marking zones where institutional orders were placed. Signals when
price returns to these zones for potential continuation.

ICT Concept: Order Blocks are the footprint of institutional accumulation or
distribution. The zone before a strong displacement move contains unfilled
orders that act as support/resistance on revisitation.
"""
import numpy as np
import pandas as pd

NAME = "OrderBlocks"
CATEGORY = "price_structure"
DESCRIPTION = "Institutional order block zone detection and retest signals"
REQUIRES_DERIVATIVES = False


def generate_signals(df: pd.DataFrame, body_threshold: float = 0.005,
                     vol_threshold: float = 1.5, expansion_factor: float = 1.5,
                     max_blocks: int = 5) -> pd.Series:
    """Generate signals when price returns to order block zones."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < 10:
        return signals

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    op = df['open'].values
    volume = df['volume'].values

    candle_range = high - low
    avg_vol = pd.Series(volume).rolling(20, min_periods=5).mean().values
    avg_range = pd.Series(candle_range).rolling(3, min_periods=1).mean().values

    # Active OB zones: (zone_low, zone_high, type, birth_bar)
    bull_obs = []
    bear_obs = []

    for i in range(4, n):
        # Check for strong bullish expansion candle
        if (close[i] > op[i] * (1 + body_threshold) and
                volume[i] >= vol_threshold * avg_vol[i - 1] if not np.isnan(avg_vol[i - 1]) else False and
                candle_range[i] >= expansion_factor * avg_range[i - 1] if not np.isnan(avg_range[i - 1]) else False):
            # Zone = low to high of 3 candles before
            zone_low = np.min(low[max(0, i - 3):i])
            zone_high = np.max(high[max(0, i - 3):i])
            bull_obs.append((zone_low, zone_high, i))
            if len(bull_obs) > max_blocks:
                bull_obs.pop(0)

        # Check for strong bearish expansion candle
        if (close[i] < op[i] * (1 - body_threshold) and
                volume[i] >= vol_threshold * avg_vol[i - 1] if not np.isnan(avg_vol[i - 1]) else False and
                candle_range[i] >= expansion_factor * avg_range[i - 1] if not np.isnan(avg_range[i - 1]) else False):
            zone_low = np.min(low[max(0, i - 3):i])
            zone_high = np.max(high[max(0, i - 3):i])
            bear_obs.append((zone_low, zone_high, i))
            if len(bear_obs) > max_blocks:
                bear_obs.pop(0)

        # Check if price returns to any active OB zone
        for zl, zh, birth in bull_obs:
            if low[i] <= zh and close[i] >= zl and i > birth:
                signals.iloc[i] = 1
                break

        if signals.iloc[i] == 0:
            for zl, zh, birth in bear_obs:
                if high[i] >= zl and close[i] <= zh and i > birth:
                    signals.iloc[i] = -1
                    break

    return signals
