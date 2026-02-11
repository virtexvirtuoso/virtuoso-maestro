"""Capitulation/Euphoria Cascade Hybrid with Regime Filter"""
import pandas as pd

NAME = "CapitulationCascade"
CATEGORY = "hybrid"
DESCRIPTION = "Multiple capitulation or euphoria signals aligned (regime-filtered)"
REQUIRES_DERIVATIVES = True

def generate_signals(df: pd.DataFrame, vol_mult: float = 4.0, period: int = 20, rsi_period: int = 14, oversold: int = 30, overbought: int = 70, oi_drop: float = 0.1, trend_period: int = 50) -> pd.Series:
    """Capitulation or euphoria cascade - multiple signals aligned. Regime-filtered."""
    signals = pd.Series(0, index=df.index)

    # Regime detection: SMA trend filter
    trend_sma = df['close'].rolling(trend_period).mean()
    uptrend = df['close'] > trend_sma    # Price above SMA = uptrend
    downtrend = df['close'] < trend_sma  # Price below SMA = downtrend

    # Volume spike detection
    vol_avg = df['volume'].rolling(period).mean()
    vol_spike = df['volume'] > vol_avg * vol_mult

    # Capitulation (panic selling)
    price_drop = df['close'].pct_change() < -0.03
    capitulation = vol_spike & price_drop

    # Euphoria (FOMO buying)
    price_pump = df['close'].pct_change() > 0.03
    euphoria = vol_spike & price_pump

    # OI analysis
    oi_cap = pd.Series(False, index=df.index)
    oi_euph = pd.Series(False, index=df.index)
    if 'open_interest' in df.columns and not df['open_interest'].isna().all():
        oi = df['open_interest'].ffill()
        oi_change = oi.pct_change()
        oi_cap = (oi_change < -oi_drop) & price_drop   # OI dropping with price = longs liquidated
        oi_euph = (oi_change > oi_drop) & price_pump   # OI expanding with price = overleveraged longs

    # RSI with Wilder's smoothing
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < oversold
    rsi_overbought = rsi > overbought

    # Long: Capitulation + (OI capitulation OR RSI oversold) - reversal signal
    signals[capitulation & (oi_cap | rsi_oversold)] = 1

    # Short: Euphoria + (OI expansion OR RSI overbought) + NOT in uptrend (regime-filtered)
    # Reason: Fading euphoria in strong uptrends loses money
    signals[euphoria & (oi_euph | rsi_overbought) & ~uptrend] = -1

    return signals
