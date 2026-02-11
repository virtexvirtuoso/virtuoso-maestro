"""Micro Basis Strategy"""
import pandas as pd

NAME = "MicroBasis"
CATEGORY = "derivatives"
DESCRIPTION = "Micro timeframe basis scalping"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, lookback: int = 8, z_threshold: float = 2.0) -> pd.Series:
    """Micro basis signals using z-score for spike detection."""
    signals = pd.Series(0, index=df.index)

    if 'funding_rate' in df.columns:
        funding = df['funding_rate'].fillna(0)

        # Z-score based spike detection (handles negative funding rates correctly)
        funding_ma = funding.rolling(lookback, min_periods=3).mean()
        funding_std = funding.rolling(lookback, min_periods=3).std().replace(0, 1e-10)
        z_score = (funding - funding_ma) / funding_std

        # High positive z-score = funding spiked up (shorts pay longs) = short
        # High negative z-score = funding spiked down (longs pay shorts) = long
        signals[z_score < -z_threshold] = 1   # Funding dropped = long opportunity
        signals[z_score > z_threshold] = -1   # Funding spiked = short opportunity

    return signals
