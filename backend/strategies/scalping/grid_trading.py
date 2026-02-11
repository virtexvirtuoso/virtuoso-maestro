"""Grid Trading Strategy"""
import pandas as pd

NAME = "GridTrading"
CATEGORY = "scalping"
DESCRIPTION = "Grid-based range trading"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, lookback: int = 50, num_grids: int = 5) -> pd.Series:
    """Grid trading signals."""
    signals = pd.Series(0, index=df.index)

    # Range detection
    high = df['high'].rolling(lookback, min_periods=10).max()
    low = df['low'].rolling(lookback, min_periods=10).min()
    range_size = high - low

    # Handle zero range (flat market) - no signals when range is too small
    min_range = df['close'].rolling(lookback, min_periods=10).std() * 0.1  # 10% of std as minimum
    valid_range = range_size > min_range

    # Current position in range (safe division)
    position = (df['close'] - low) / range_size.replace(0, 1e-10)

    # Buy low grids, sell high grids (only when range is valid)
    signals[(position < 0.2) & valid_range] = 1
    signals[(position > 0.8) & valid_range] = -1

    return signals
