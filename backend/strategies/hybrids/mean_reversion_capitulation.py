"""Mean Reversion + Capitulation/Euphoria Filter Hybrid with Regime Filter"""
import pandas as pd

NAME = "MeanReversion+CapitulationFilter"
CATEGORY = "hybrid"
DESCRIPTION = "Mean reversion after capitulation or euphoria (regime-filtered)"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, mr_period: int = 20, threshold: float = 2.0, vol_mult: float = 4.0, trend_period: int = 50) -> pd.Series:
    """Mean reversion after capitulation or euphoria. Regime-filtered."""
    signals = pd.Series(0, index=df.index)

    # Regime detection: SMA trend filter
    trend_sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > trend_sma    # Price above SMA = uptrend
    downtrend = df['close'] < trend_sma  # Price below SMA = downtrend

    # Mean reversion z-score
    sma = df['close'].rolling(mr_period).mean()
    std = df['close'].rolling(mr_period).std()
    zscore = (df['close'] - sma) / std.replace(0, 1e-10)

    # Volume spike detection
    vol_avg = df['volume'].rolling(20).mean()
    vol_spike = df['volume'] > vol_avg * vol_mult

    # Capitulation filter (panic selling)
    price_drop = df['close'].pct_change() < -0.03
    capitulation = vol_spike & price_drop

    # Euphoria filter (FOMO buying)
    price_pump = df['close'].pct_change() > 0.03
    euphoria = vol_spike & price_pump

    # Long: oversold + capitulation (reversal signal - allowed in downtrends)
    signals[capitulation & (zscore < -threshold)] = 1

    # Short: overbought + euphoria + NOT in uptrend (regime-filtered)
    # Reason: Fading euphoria in strong uptrends loses money
    signals[euphoria & (zscore > threshold) & ~uptrend] = -1

    return signals
