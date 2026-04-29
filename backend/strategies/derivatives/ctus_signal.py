"""
CTUS - Crowded Trade Unwinding Signal

Fades crowded positioning when all derivatives metrics align at extremes.

Logic:
- OI z-score > threshold (lots of new positions)
- Funding z-score > threshold (one side paying heavily)
- LSR z-score > threshold (retail piled in one direction)
- When all three align: fade the crowd

Direction:
- All extreme positive (everyone long) → SHORT
- All extreme negative (everyone short) → LONG

Expected frequency: 3-8 trades per month per asset

Parameters:
- oi_z_thresh: z-score threshold for OI level
- funding_z_thresh: z-score threshold for funding rate
- lsr_z_thresh: z-score threshold for LSR
- lookback: rolling window for z-score calculation
- hold_bars: how long to hold the position
"""
import pandas as pd
import numpy as np

NAME = "CTUS"
CATEGORY = "derivatives"
DESCRIPTION = "Crowded Trade Unwinding Signal - fade extreme positioning"
REQUIRES_DERIVATIVES = True


def generate_signals(
    df: pd.DataFrame,
    oi_z_thresh: float = 1.5,
    funding_z_thresh: float = 1.5,
    lsr_z_thresh: float = 1.0,
    lookback: int = 30,
    hold_bars: int = 3,
) -> pd.Series:
    """
    Generate CTUS signals.

    Required columns: close, oi_close, funding_rate, long_ratio
    Returns: Series of {-1, 0, 1}
    """
    signals = pd.Series(0, index=df.index, dtype=int)
    n = len(df)
    if n < lookback + 5:
        return signals

    # --- Z-scores ---
    # OI level z-score (high OI = lots of positions open = crowded)
    oi = df['oi_close'].astype(float)
    oi_mean = oi.rolling(lookback).mean()
    oi_std = oi.rolling(lookback).std().replace(0, np.nan)
    oi_z = (oi - oi_mean) / oi_std

    # Funding z-score
    funding = df['funding_rate'].astype(float)
    f_mean = funding.rolling(lookback).mean()
    f_std = funding.rolling(lookback).std().replace(0, np.nan)
    funding_z = (funding - f_mean) / f_std

    # LSR z-score (long_ratio - 50 = deviation from neutral)
    lsr = df['long_ratio'].astype(float) - 50  # center around 0
    lsr_mean = lsr.rolling(lookback).mean()
    lsr_std = lsr.rolling(lookback).std().replace(0, np.nan)
    lsr_z = (lsr - lsr_mean) / lsr_std

    # --- Signal generation ---
    # Everyone long: OI high + funding positive extreme + LSR long-skewed
    crowd_long = (oi_z > oi_z_thresh) & (funding_z > funding_z_thresh) & (lsr_z > lsr_z_thresh)
    # Everyone short: OI high + funding negative extreme + LSR short-skewed
    crowd_short = (oi_z > oi_z_thresh) & (funding_z < -funding_z_thresh) & (lsr_z < -lsr_z_thresh)

    signals[crowd_long] = -1  # fade longs → go short
    signals[crowd_short] = 1  # fade shorts → go long

    # Hold positions
    signals = _hold_signal(signals, hold_bars)
    return signals


def _hold_signal(signals: pd.Series, hold_bars: int) -> pd.Series:
    """Extend signals for hold_bars periods."""
    result = signals.copy()
    current_signal = 0
    hold_remaining = 0
    for i in range(len(result)):
        if result.iloc[i] != 0:
            current_signal = result.iloc[i]
            hold_remaining = hold_bars
        elif hold_remaining > 0:
            result.iloc[i] = current_signal
            hold_remaining -= 1
    return result
