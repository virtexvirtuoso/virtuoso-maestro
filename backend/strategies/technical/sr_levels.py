"""Support/Resistance Levels — Proximity-Clustered S/R (HSAKA/ICT)

Finds swing highs and lows, clusters nearby levels using proximity grouping,
scores them by touch count, recency, and volume, then generates signals
when price approaches strong levels with candle rejection confirmation.

ICT Concept: Key levels are where institutional orders cluster. More touches
and higher volume at a level indicate stronger institutional interest. Candle
rejection (long wicks) at these levels confirms the defense.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

NAME = "SRLevels"
CATEGORY = "price_structure"
DESCRIPTION = "DBSCAN-clustered support/resistance with bounce confirmation"
REQUIRES_DERIVATIVES = False


def generate_signals(df: pd.DataFrame, swing_window: int = 10, group_threshold: float = 0.005,
                     proximity_pct: float = 0.01, min_touches: int = 2) -> pd.Series:
    """Generate signals at strong S/R levels with rejection confirmation."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < swing_window * 2 + 1:
        return signals

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    op = df['open'].values

    # Find swing points
    swing_high_idx = argrelextrema(high, np.greater_equal, order=swing_window)[0]
    swing_low_idx = argrelextrema(low, np.less_equal, order=swing_window)[0]

    if len(swing_high_idx) + len(swing_low_idx) < min_touches:
        return signals

    # Collect all swing levels with metadata
    levels = []  # (price, bar_index, volume, type)
    for i in swing_high_idx:
        levels.append((high[i], i, volume[i], 'R'))
    for i in swing_low_idx:
        levels.append((low[i], i, volume[i], 'S'))

    levels.sort(key=lambda x: x[0])

    # Cluster by proximity
    clusters = []  # each: {'price': avg, 'touches': count, 'vol': total, 'last_bar': max_idx, 'type': S/R}
    used = [False] * len(levels)
    for i in range(len(levels)):
        if used[i]:
            continue
        cluster_prices = [levels[i][0]]
        cluster_bars = [levels[i][1]]
        cluster_vols = [levels[i][2]]
        cluster_types = [levels[i][3]]
        used[i] = True
        for j in range(i + 1, len(levels)):
            if used[j]:
                continue
            if abs(levels[j][0] - np.mean(cluster_prices)) / np.mean(cluster_prices) <= group_threshold:
                cluster_prices.append(levels[j][0])
                cluster_bars.append(levels[j][1])
                cluster_vols.append(levels[j][2])
                cluster_types.append(levels[j][3])
                used[j] = True

        # Determine type by majority
        s_count = cluster_types.count('S')
        r_count = cluster_types.count('R')
        level_type = 'S' if s_count >= r_count else 'R'

        clusters.append({
            'price': np.mean(cluster_prices),
            'touches': len(cluster_prices),
            'vol': np.sum(cluster_vols),
            'last_bar': max(cluster_bars),
            'type': level_type
        })

    # Filter by min touches
    clusters = [c for c in clusters if c['touches'] >= min_touches]
    if not clusters:
        return signals

    # Score levels: touches * volume * recency
    for i in range(n):
        price = close[i]
        candle_range = high[i] - low[i]
        if candle_range == 0:
            continue

        # Wick ratio for rejection confirmation
        upper_wick = high[i] - max(close[i], op[i])
        lower_wick = min(close[i], op[i]) - low[i]

        best_signal = 0
        best_score = 0

        for c in clusters:
            if c['last_bar'] >= i:
                continue  # Only use levels formed before current bar

            dist_pct = abs(price - c['price']) / c['price']
            if dist_pct > proximity_pct:
                continue

            # Recency decay
            age = i - c['last_bar']
            recency = np.exp(-age / n * 3)
            score = c['touches'] * c['vol'] * recency

            if score <= best_score:
                continue

            if c['type'] == 'S' and price >= c['price'] * (1 - proximity_pct):
                # Near support — need lower wick rejection
                if lower_wick > 0.5 * candle_range:
                    best_signal = 1
                    best_score = score
            elif c['type'] == 'R' and price <= c['price'] * (1 + proximity_pct):
                # Near resistance — need upper wick rejection
                if upper_wick > 0.5 * candle_range:
                    best_signal = -1
                    best_score = score

        signals.iloc[i] = best_signal

    return signals
