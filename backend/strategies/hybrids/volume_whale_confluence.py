"""
Volume Whale Confluence Strategy

Thesis: Real edge comes from targeting high-volume markets in mispriced categories
using volume and whales as confirmation.

Combines the three strongest signal types from backtesting:
1. Volume spike gate (whale/institutional interest)
2. Funding rate extremes (crowded positioning = mispricing)
3. Capitulation/euphoria detection (panic selling or FOMO buying)

Uses confluence scoring: each signal component adds +1 to a directional score.
Entry only fires when score >= min_confluence, ensuring multiple confirmations.

Best on: 1D timeframe, high-beta altcoins (RENDER, SUI, FET, TAO, ARB).
"""

import numpy as np
import pandas as pd

NAME = "VolumeWhaleConfluence"
CATEGORY = "hybrid"
DESCRIPTION = "Volume + funding extremes + capitulation confluence (whale confirmation)"
REQUIRES_DERIVATIVES = True


def generate_signals(
    df: pd.DataFrame,
    vol_period: int = 20,
    vol_mult: float = 2.5,
    funding_lookback: int = 20,
    funding_zscore_thresh: float = 1.5,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    price_move_pct: float = 0.03,
    min_confluence: int = 3,
    trend_period: int = 50,
) -> pd.Series:
    """
    Generate signals using confluence of volume, funding, and capitulation.

    Scoring system (per bar, per direction):
      +1  Volume spike (volume > vol_mult * rolling average)
      +1  Funding rate Z-score extreme (contrarian: negative Z = long, positive Z = short)
      +1  Capitulation/euphoria (large price move + volume spike)
      +1  RSI extreme (oversold = long, overbought = short)

    Long fires when bull_score >= min_confluence.
    Short fires when bear_score >= min_confluence AND price below trend SMA
    (regime filter: avoid shorting strong uptrends, lesson from CapitulationCascade).

    Args:
        df: OHLCV DataFrame with optional funding_rate column
        vol_period: Rolling window for volume average
        vol_mult: Multiplier for volume spike detection
        funding_lookback: Rolling window for funding Z-score
        funding_zscore_thresh: Z-score threshold for funding extreme
        rsi_period: RSI calculation period
        rsi_oversold: RSI oversold threshold (long signal)
        rsi_overbought: RSI overbought threshold (short signal)
        price_move_pct: Minimum price move for capitulation/euphoria (e.g., 0.03 = 3%)
        min_confluence: Minimum score required for signal (2-4)
        trend_period: SMA period for regime filter

    Returns:
        Signal series: 1=long, -1=short, 0=flat
    """
    signals = pd.Series(0, index=df.index)

    # =========================================================================
    # PILLAR 1: Volume spike detection (whale activity gate)
    # =========================================================================
    vol_avg = df['volume'].rolling(vol_period, min_periods=1).mean()
    volume_spike = df['volume'] > (vol_avg * vol_mult)

    # =========================================================================
    # PILLAR 2: Funding rate mispricing (contrarian signal)
    # =========================================================================
    # Use real funding_rate column from merged data; fall back to zero
    if 'funding_rate' in df.columns:
        funding = df['funding_rate'].fillna(0.0)
    else:
        funding = pd.Series(0.0, index=df.index)

    funding_has_signal = funding.std() > 1e-10

    if funding_has_signal:
        funding_ma = funding.rolling(funding_lookback, min_periods=1).mean()
        funding_std = funding.rolling(funding_lookback, min_periods=1).std()
        funding_std = funding_std.replace(0, np.nan).fillna(1e-10)
        funding_z = (funding - funding_ma) / funding_std

        # Extreme negative funding = shorts paying = crowded short = contrarian LONG
        funding_bullish = funding_z < -funding_zscore_thresh
        # Extreme positive funding = longs paying = crowded long = contrarian SHORT
        funding_bearish = funding_z > funding_zscore_thresh
    else:
        funding_bullish = pd.Series(False, index=df.index)
        funding_bearish = pd.Series(False, index=df.index)

    # =========================================================================
    # PILLAR 3: Capitulation / euphoria detection
    # =========================================================================
    price_change = df['close'].pct_change()

    # Capitulation: large drop + volume spike = panic selling (contrarian buy)
    capitulation = (price_change < -price_move_pct) & volume_spike
    # Euphoria: large pump + volume spike = FOMO buying (contrarian sell)
    euphoria = (price_change > price_move_pct) & volume_spike

    # =========================================================================
    # RSI confirmation
    # =========================================================================
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan).fillna(1e-10)
    rsi = 100 - (100 / (1 + rs))

    rsi_bull = rsi < rsi_oversold
    rsi_bear = rsi > rsi_overbought

    # =========================================================================
    # CONFLUENCE SCORING
    # =========================================================================
    # Each component contributes +1 to the directional score (0 or 1, cast to int)
    bull_score = (
        volume_spike.astype(int)
        + funding_bullish.astype(int)
        + capitulation.astype(int)
        + rsi_bull.astype(int)
    )

    bear_score = (
        volume_spike.astype(int)
        + funding_bearish.astype(int)
        + euphoria.astype(int)
        + rsi_bear.astype(int)
    )

    # =========================================================================
    # REGIME FILTER (short side only)
    # =========================================================================
    # Avoid shorting strong uptrends -- fading euphoria in bull markets loses money
    trend_sma = df['close'].rolling(trend_period, min_periods=1).mean()
    not_strong_uptrend = df['close'] <= trend_sma

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    signals[bull_score >= min_confluence] = 1
    signals[(bear_score >= min_confluence) & not_strong_uptrend] = -1

    # When both fire on the same bar (rare), bull wins (mean-reversion bias)
    both = (bull_score >= min_confluence) & (bear_score >= min_confluence)
    signals[both] = 1

    return signals
