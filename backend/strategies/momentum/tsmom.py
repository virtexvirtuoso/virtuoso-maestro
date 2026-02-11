"""Time-Series Momentum Strategy"""
import pandas as pd
import numpy as np

NAME = "TSMOM"
CATEGORY = "momentum"
DESCRIPTION = "Time-series momentum (trend following)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, lookback: int = 20, vol_lookback: int = 20, t_threshold: float = 2.0) -> pd.Series:
    """TSMOM signals with proper volatility scaling (t-statistic)."""
    signals = pd.Series(0, index=df.index)

    # Returns
    returns = df['close'].pct_change()

    # Momentum signal (cumulative return over lookback)
    mom = df['close'].pct_change(lookback)

    # Volatility scaling - proper t-statistic: mom / (vol * sqrt(lookback))
    # This normalizes the momentum signal by its standard error
    vol = returns.rolling(vol_lookback, min_periods=5).std()
    scaled_mom = mom / (vol.replace(0, 1e-10) * np.sqrt(lookback))

    # Long positive momentum, short negative (using t-stat threshold)
    signals[scaled_mom > t_threshold] = 1
    signals[scaled_mom < -t_threshold] = -1

    return signals
