"""
CAS - Cascade Absorption Signal

Detects liquidation cascades being absorbed and fades them.

Logic:
1. OI drops sharply (z-score below threshold) = cascade happening
2. Funding rate at extreme = crowded positioning that triggered cascade
3. LSR shows retail heavily one-sided
4. Signal fires when OI stops dropping (absorption) + price stabilizes

Entry: Fade the cascade direction
- If funding was extreme positive (longs crowded) → cascade is long liquidations → buy the dip
- If funding was extreme negative (shorts crowded) → cascade is short liquidations → sell the rip

Parameters:
- oi_drop_z: z-score threshold for OI drop (negative, e.g., -1.5)
- funding_extreme_z: z-score threshold for funding extremes
- lsr_extreme_z: z-score for LSR deviation from neutral
- lookback: rolling window for z-score calculation
- absorption_bars: how many bars OI must stabilize to confirm absorption
"""
import pandas as pd
import numpy as np

NAME = "CAS"
CATEGORY = "derivatives"
DESCRIPTION = "Cascade Absorption Signal - fade liquidation cascades"
REQUIRES_DERIVATIVES = True


def generate_signals(
    df: pd.DataFrame,
    oi_drop_z: float = -1.5,
    funding_extreme_z: float = 1.5,
    lsr_extreme_z: float = 1.5,
    lookback: int = 30,
    absorption_bars: int = 2,
    hold_bars: int = 5,
) -> pd.Series:
    """
    Generate CAS signals.

    Required columns: close, oi_close, funding_rate, long_ratio
    Returns: Series of {-1, 0, 1}
    """
    signals = pd.Series(0, index=df.index, dtype=int)
    n = len(df)
    if n < lookback + 10:
        return signals

    # --- Compute z-scores ---
    oi = df['oi_close'].astype(float)
    oi_pct = oi.pct_change()
    oi_pct_mean = oi_pct.rolling(lookback).mean()
    oi_pct_std = oi_pct.rolling(lookback).std().replace(0, np.nan)
    oi_z = (oi_pct - oi_pct_mean) / oi_pct_std

    funding = df['funding_rate'].astype(float)
    f_mean = funding.rolling(lookback).mean()
    f_std = funding.rolling(lookback).std().replace(0, np.nan)
    funding_z = (funding - f_mean) / f_std

    lsr = df['long_ratio'].astype(float)
    lsr_mean = lsr.rolling(lookback).mean()
    lsr_std = lsr.rolling(lookback).std().replace(0, np.nan)
    lsr_z = (lsr - lsr_mean) / lsr_std

    # Price stabilization: absolute return < median absolute return
    price_ret = df['close'].pct_change().abs()
    median_ret = price_ret.rolling(lookback).median()

    for i in range(lookback + absorption_bars, n):
        # Check for cascade in recent bars
        cascade_detected = False
        cascade_direction = 0  # +1 = longs liquidated, -1 = shorts liquidated

        for j in range(1, absorption_bars + 2):
            idx = i - j
            if idx < 0:
                break
            if oi_z.iloc[idx] <= oi_drop_z:
                # OI dropped sharply = cascade
                # Determine direction from funding
                if funding_z.iloc[idx] >= funding_extreme_z:
                    # Funding was extreme positive → longs were crowded → longs liquidated
                    cascade_direction = 1  # buy signal (fade the cascade)
                elif funding_z.iloc[idx] <= -funding_extreme_z:
                    # Funding was extreme negative → shorts crowded → shorts liquidated
                    cascade_direction = -1  # sell signal
                if cascade_direction != 0:
                    cascade_detected = True
                    break

        if not cascade_detected:
            continue

        # Check LSR confirms crowding
        if cascade_direction == 1 and lsr_z.iloc[i - 1] < lsr_extreme_z * 0.5:
            continue  # Longs weren't that crowded per LSR
        if cascade_direction == -1 and lsr_z.iloc[i - 1] > -lsr_extreme_z * 0.5:
            continue

        # Check absorption: OI stopped dropping (recent OI change > oi_drop_z)
        if oi_z.iloc[i] <= oi_drop_z:
            continue  # Still cascading

        # Check price stabilization
        if price_ret.iloc[i] > median_ret.iloc[i] * 2:
            continue  # Price still volatile

        signals.iloc[i] = cascade_direction

    # Hold positions for hold_bars
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
