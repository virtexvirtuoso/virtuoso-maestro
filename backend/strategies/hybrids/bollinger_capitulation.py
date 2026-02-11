"""Bollinger + Capitulation/Euphoria Filter Hybrid with Regime Filter"""
import pandas as pd

NAME = "BollingerBreakout+CapitulationFilter"
CATEGORY = "hybrid"
DESCRIPTION = "Bollinger bounce after capitulation or fade euphoria (regime-filtered)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, vol_mult: float = 4.0, trend_period: int = 50) -> pd.Series:
    """Bollinger bounce after capitulation, fade at euphoria. Regime-filtered."""
    signals = pd.Series(0, index=df.index)

    # Regime detection: SMA trend filter
    trend_sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > trend_sma    # Price above SMA = uptrend
    downtrend = df['close'] < trend_sma  # Price below SMA = downtrend

    # Bollinger bands
    sma = df['close'].rolling(bb_period).mean()
    std_dev = df['close'].rolling(bb_period).std()
    lower = sma - bb_std * std_dev
    upper = sma + bb_std * std_dev

    # Volume spike detection
    vol_avg = df['volume'].rolling(20).mean()
    vol_spike = df['volume'] > vol_avg * vol_mult

    # Capitulation filter (panic selling)
    price_drop = df['close'].pct_change() < -0.03
    capitulation = vol_spike & price_drop

    # Euphoria filter (FOMO buying)
    price_pump = df['close'].pct_change() > 0.03
    euphoria = vol_spike & price_pump

    # Long: capitulation + lower band (reversal signal - allowed in downtrends)
    signals[capitulation & (df['close'] < lower)] = 1

    # Short: euphoria + upper band + NOT in uptrend (regime-filtered)
    # Reason: Fading euphoria in strong uptrends loses money
    signals[euphoria & (df['close'] > upper) & ~uptrend] = -1

    return signals
