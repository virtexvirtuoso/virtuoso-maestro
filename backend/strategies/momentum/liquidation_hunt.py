"""Liquidation Hunt Strategy"""
import pandas as pd

NAME = "LiquidationHunt"
CATEGORY = "momentum"
DESCRIPTION = "Trade liquidation cascades"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, vol_mult: float = 3.0, price_move: float = 0.02) -> pd.Series:
    """Liquidation hunt signals."""
    signals = pd.Series(0, index=df.index)
    
    # Volume spike
    vol_avg = df['volume'].rolling(20).mean()
    vol_spike = df['volume'] > vol_avg * vol_mult
    
    # Sharp price move
    price_change = df['close'].pct_change()
    sharp_drop = price_change < -price_move
    sharp_pump = price_change > price_move
    
    # Use liquidation data if available
    if 'liq_long' in df.columns and 'liq_short' in df.columns:
        liq_long = df['liq_long'].fillna(0)
        liq_short = df['liq_short'].fillna(0)
        liq_avg = (liq_long + liq_short).rolling(20).mean()
        
        long_cascade = liq_long > liq_avg * 2
        short_cascade = liq_short > liq_avg * 2
        
        signals[long_cascade & sharp_drop] = 1  # Buy after long liquidations
        signals[short_cascade & sharp_pump] = -1  # Short after short liquidations
    else:
        # Fallback to volume + price
        signals[vol_spike & sharp_drop] = 1
        signals[vol_spike & sharp_pump] = -1
    
    return signals
