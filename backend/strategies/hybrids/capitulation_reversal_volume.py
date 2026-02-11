"""Capitulation Reversal + Volume Filter Hybrid"""
import pandas as pd

NAME = "CapitulationReversal+VolumeFilter"
CATEGORY = "hybrid"
DESCRIPTION = "Capitulation reversal with 4x volume spike"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, vol_mult: float = 4.0, period: int = 20) -> pd.Series:
    """Capitulation reversal with extreme volume."""
    signals = pd.Series(0, index=df.index)
    
    vol_avg = df['volume'].rolling(period).mean()
    vol_spike = df['volume'] > vol_avg * vol_mult
    
    price_drop = df['close'].pct_change() < -0.03
    price_pump = df['close'].pct_change() > 0.03
    
    signals[vol_spike & price_drop] = 1
    signals[vol_spike & price_pump] = -1
    return signals
