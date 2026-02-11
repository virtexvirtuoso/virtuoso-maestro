"""Liquidation Scalp Strategy"""
import pandas as pd

NAME = "LiquidationScalp"
CATEGORY = "derivatives"
DESCRIPTION = "Quick liquidation reversals"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, liq_mult: float = 3.0) -> pd.Series:
    """Liquidation scalp signals."""
    signals = pd.Series(0, index=df.index)
    
    if 'liq_long' in df.columns and 'liq_short' in df.columns:
        liq_long = df['liq_long'].fillna(0)
        liq_short = df['liq_short'].fillna(0)
        
        liq_long_avg = liq_long.rolling(20).mean()
        liq_short_avg = liq_short.rolling(20).mean()
        
        # Spike in liquidations
        long_liq_spike = liq_long > liq_long_avg * liq_mult
        short_liq_spike = liq_short > liq_short_avg * liq_mult
        
        # Fade the liquidations
        signals[long_liq_spike] = 1  # Buy after long liquidations
        signals[short_liq_spike] = -1  # Short after short liquidations
    else:
        # Fallback
        vol_spike = df['volume'] > df['volume'].rolling(20).mean() * 3
        price_drop = df['close'].pct_change() < -0.02
        price_pump = df['close'].pct_change() > 0.02
        
        signals[vol_spike & price_drop] = 1
        signals[vol_spike & price_pump] = -1
    
    return signals
