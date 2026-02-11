"""Mean Reversion Bands Strategy"""
import pandas as pd

NAME = "MeanReversionBands"
CATEGORY = "momentum"
DESCRIPTION = "Keltner Channel mean reversion"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0) -> pd.Series:
    """Mean Reversion Bands signals."""
    signals = pd.Series(0, index=df.index)
    
    # EMA center
    ema = df['close'].ewm(span=ema_period).mean()
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    # Keltner Channels
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    # Mean reversion
    signals[df['close'] < lower] = 1
    signals[df['close'] > upper] = -1
    
    return signals
