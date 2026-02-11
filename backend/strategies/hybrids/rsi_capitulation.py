"""RSI + Capitulation/Euphoria Filter Hybrid with Regime Filter"""
import pandas as pd

NAME = "RSI+CapitulationFilter"
CATEGORY = "hybrid"
DESCRIPTION = "RSI extremes after capitulation or euphoria (regime-filtered)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, rsi_period: int = 14, oversold: int = 30, overbought: int = 70, vol_mult: float = 4.0, trend_period: int = 50) -> pd.Series:
    """RSI oversold after capitulation, overbought after euphoria. Regime-filtered."""
    signals = pd.Series(0, index=df.index)

    # Regime detection: SMA trend filter
    trend_sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > trend_sma    # Price above SMA = uptrend
    downtrend = df['close'] < trend_sma  # Price below SMA = downtrend

    # RSI with Wilder's smoothing
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike detection
    vol_avg = df['volume'].rolling(20).mean()
    vol_spike = df['volume'] > vol_avg * vol_mult

    # Capitulation filter (panic selling)
    price_drop = df['close'].pct_change() < -0.03
    capitulation = vol_spike & price_drop

    # Euphoria filter (FOMO buying)
    price_pump = df['close'].pct_change() > 0.03
    euphoria = vol_spike & price_pump

    # Long: capitulation + oversold (reversal signal - allowed in downtrends)
    signals[capitulation & (rsi < oversold)] = 1

    # Short: euphoria + overbought + NOT in uptrend (regime-filtered)
    # Reason: Fading euphoria in strong uptrends loses money
    signals[euphoria & (rsi > overbought) & ~uptrend] = -1

    return signals
