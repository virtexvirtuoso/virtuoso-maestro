"""Stochastic RSI Strategy"""
import pandas as pd

NAME = "StochRSI"
CATEGORY = "scalping"
DESCRIPTION = "Stochastic RSI overbought/oversold"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> pd.Series:
    """Stochastic RSI signals."""
    signals = pd.Series(0, index=df.index)
    
    # RSI with Wilder's smoothing
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    stoch_k = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)
    stoch_d = stoch_k.rolling(d_period).mean()
    
    # Signals
    signals[(stoch_k < 20) & (stoch_k > stoch_d)] = 1
    signals[(stoch_k > 80) & (stoch_k < stoch_d)] = -1
    
    return signals
