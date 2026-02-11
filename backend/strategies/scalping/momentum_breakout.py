"""Momentum Breakout Strategy"""
import pandas as pd

NAME = "MomentumBreakout"
CATEGORY = "scalping"
DESCRIPTION = "Momentum with volume breakout"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, lookback: int = 10, vol_mult: float = 1.5) -> pd.Series:
    """Momentum breakout signals."""
    signals = pd.Series(0, index=df.index)
    
    # Momentum
    mom = df['close'].pct_change(lookback)
    
    # Volume confirmation
    vol_avg = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_avg * vol_mult
    
    # Strong momentum + volume
    signals[(mom > 0.02) & high_vol] = 1
    signals[(mom < -0.02) & high_vol] = -1
    
    return signals
