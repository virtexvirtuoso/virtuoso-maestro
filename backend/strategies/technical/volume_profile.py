"""Volume Profile — POC/Value Area/HVN/LVN (HSAKA/ICT)

Builds a volume profile over a rolling lookback window, identifying the
Point of Control (highest volume price), Value Area (70% of volume), and
generates mean-reversion signals at value area boundaries.

ICT Concept: Volume Profile reveals where institutional activity concentrates.
Price below the Value Area Low is undervalued (buy), above Value Area High is
overvalued (sell). The POC acts as a fair price magnet.
"""
import numpy as np
import pandas as pd

NAME = "VolumeProfile"
CATEGORY = "price_structure"
DESCRIPTION = "Volume Profile with POC and Value Area boundary signals"
REQUIRES_DERIVATIVES = False


def generate_signals(df: pd.DataFrame, bins: int = 50, value_area_pct: float = 0.70,
                     lookback: int = 30) -> pd.Series:
    """Generate signals based on price position relative to Value Area."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < lookback:
        return signals

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values

    for i in range(lookback, n):
        start = i - lookback
        w_high = high[start:i]
        w_low = low[start:i]
        w_close = close[start:i]
        w_vol = volume[start:i]

        price_min = np.min(w_low)
        price_max = np.max(w_high)
        if price_max <= price_min:
            continue

        # Build volume profile: distribute each bar's volume into bins
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        vol_profile = np.zeros(bins)

        for j in range(len(w_close)):
            # Distribute volume to bins that the candle spans
            bar_low = w_low[j]
            bar_high = w_high[j]
            for b in range(bins):
                if bin_edges[b + 1] >= bar_low and bin_edges[b] <= bar_high:
                    vol_profile[b] += w_vol[j] / max(1, int((bar_high - bar_low) / ((price_max - price_min) / bins) + 1))

        total_vol = np.sum(vol_profile)
        if total_vol == 0:
            continue

        # POC = bin with max volume
        poc_bin = np.argmax(vol_profile)
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2

        # Value Area: expand from POC until value_area_pct of volume captured
        va_vol = vol_profile[poc_bin]
        lo_idx = poc_bin - 1
        hi_idx = poc_bin + 1
        target_vol = total_vol * value_area_pct

        while va_vol < target_vol and (lo_idx >= 0 or hi_idx < bins):
            add_lo = vol_profile[lo_idx] if lo_idx >= 0 else 0
            add_hi = vol_profile[hi_idx] if hi_idx < bins else 0
            if add_lo >= add_hi and lo_idx >= 0:
                va_vol += add_lo
                lo_idx -= 1
            elif hi_idx < bins:
                va_vol += add_hi
                hi_idx += 1
            else:
                lo_idx -= 1

        val = bin_edges[max(0, lo_idx + 1)]
        vah = bin_edges[min(bins, hi_idx)]

        # Signal generation
        if close[i] <= val:
            signals.iloc[i] = 1   # Below value area = undervalued
        elif close[i] >= vah:
            signals.iloc[i] = -1  # Above value area = overvalued

    return signals
