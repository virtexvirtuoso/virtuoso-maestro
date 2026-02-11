"""Combined Bin Cluc Strategy"""
import pandas as pd

NAME = "CombinedBinCluc"
CATEGORY = "composite"
DESCRIPTION = "Binary + Cluckie hybrid"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Combined Bin Cluc signals."""
    signals = pd.Series(0, index=df.index)
    
    # Bollinger
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    lower_bb = sma_20 - 2 * std_20
    
    # RSI with Wilder's smoothing
    delta = df['close'].diff()
    rsi_period = 14
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA
    ema_50 = df['close'].ewm(span=50).mean()
    ema_200 = df['close'].ewm(span=200).mean()
    
    # Volume
    vol_ma = df['volume'].rolling(20).mean()
    
    # Cluckie: BB + RSI oversold
    cluc_buy = (df['close'] < lower_bb) & (rsi < 30)
    
    # Binary: Trend + volume
    bin_buy = (ema_50 > ema_200) & (df['volume'] > vol_ma)
    
    signals[cluc_buy | bin_buy] = 1
    signals[(rsi > 70) & (df['close'] > sma_20 + 2 * std_20)] = -1
    
    return signals
