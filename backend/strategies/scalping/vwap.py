"""VWAP Strategy"""
import pandas as pd

NAME = "VWAP"
CATEGORY = "scalping"
DESCRIPTION = "Volume-Weighted Average Price reversion"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, std_mult: float = 2.0, session_hours: int = 24) -> pd.Series:
    """VWAP signals with daily session reset."""
    signals = pd.Series(0, index=df.index)

    # Calculate typical price and volume-weighted values
    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap_product = typical * df['volume']

    # Detect session boundaries (new day) if timestamp available
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        session_start = timestamps.dt.date != timestamps.dt.date.shift(1)
    else:
        # Fallback: reset every session_hours bars (approximate daily for hourly data)
        session_start = pd.Series(False, index=df.index)
        session_start.iloc[::session_hours] = True

    # Calculate session-based cumulative values
    session_id = session_start.cumsum()
    cum_vol = df.groupby(session_id)['volume'].cumsum()
    cum_vwap_product = vwap_product.groupby(session_id).cumsum()

    # VWAP with session reset (avoid division by zero on first bar)
    vwap = cum_vwap_product / cum_vol.replace(0, 1e-10)

    # VWAP bands (rolling std of deviation from VWAP)
    vwap_std = (df['close'] - vwap).rolling(20, min_periods=5).std().fillna(0)
    upper = vwap + std_mult * vwap_std
    lower = vwap - std_mult * vwap_std

    # Mean reversion signals
    signals[df['close'] < lower] = 1
    signals[df['close'] > upper] = -1

    return signals
