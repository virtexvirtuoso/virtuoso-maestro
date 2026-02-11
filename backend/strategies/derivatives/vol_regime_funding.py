"""
Volatility Regime Funding Overlay Strategy

Combines volatility regime detection with funding rate signals.

Logic:
- High vol + extreme funding = contrarian fade
- Low vol + funding drift = follow direction
- IV skew > 0.15 + negative funding = strong long

Data Sources (priority order):
1. Real funding + options IV from pre-computed signals
2. Funding proxy + ATR-based vol regime
"""

import pandas as pd
import numpy as np

try:
    from .mixins import DerivativesDataMixin
except ImportError:
    from strategies.derivatives.mixins import DerivativesDataMixin

NAME = "VolRegimeFunding"
CATEGORY = "derivatives"
DESCRIPTION = "Vol regime + funding overlay with multi-source data"
REQUIRES_DERIVATIVES = True


class VolRegimeFundingOverlayStrategy(DerivativesDataMixin):
    """Volatility regime with funding overlay strategy."""

    def __init__(
        self,
        low_vol_threshold: float = 0.25,
        high_vol_threshold: float = 0.75,
        funding_extreme_positive: float = 0.0003,
        funding_extreme_negative: float = -0.0001,
        iv_skew_threshold: float = 0.15,
        atr_period: int = 14,
        lookback: int = 168
    ):
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.funding_extreme_positive = funding_extreme_positive
        self.funding_extreme_negative = funding_extreme_negative
        self.iv_skew_threshold = iv_skew_threshold
        self.atr_period = atr_period
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame, symbol: str = "BTC") -> pd.Series:
        """Generate trading signals based on vol regime and funding."""
        signals = pd.Series(0, index=df.index)

        # Get volatility regime
        vol_percentile = self.get_vol_percentile(df, self.atr_period, self.lookback)
        high_vol = vol_percentile > self.high_vol_threshold
        low_vol = vol_percentile < self.low_vol_threshold

        # Get funding data
        funding = self.get_funding_rate(df, symbol, use_proxy=True, calibrated=True)

        # Check for options IV data
        iv_skew = None
        if 'iv_skew' in df.columns:
            iv_skew = df['iv_skew']
        else:
            # Try to get from pre-computed signals
            fusion = self.get_fusion_signal(symbol)
            if 'component_signals' in fusion:
                options_sig = fusion['component_signals'].get('options_iv', {})
                if 'iv_skew' in options_sig:
                    iv_skew = pd.Series(options_sig['iv_skew'], index=df.index)

        # ===== LOW VOLATILITY REGIME =====
        # Mean reversion: fade extreme funding
        # Positive funding = longs crowded = short
        signals[low_vol & (funding > self.funding_extreme_positive)] = -1
        # Negative funding = shorts crowded = long
        signals[low_vol & (funding < self.funding_extreme_negative)] = 1

        # ===== HIGH VOLATILITY REGIME =====
        # Contrarian: extreme funding = reversion signal
        # High vol + positive funding = overleveraged longs = fade
        signals[high_vol & (funding > self.funding_extreme_positive)] = -1
        # High vol + negative funding = panic shorts = reversal long
        signals[high_vol & (funding < self.funding_extreme_negative)] = 1

        # ===== OPTIONS IV OVERLAY =====
        if iv_skew is not None:
            # IV skew > threshold = fear (puts expensive) + negative funding = strong long
            fear_regime = iv_skew > self.iv_skew_threshold
            signals[fear_regime & (funding < 0)] = 1

            # IV skew < -threshold = complacency = potential short
            complacency_regime = iv_skew < -self.iv_skew_threshold
            signals[complacency_regime & (funding > 0)] = -1

        return signals


def generate_signals(
    df: pd.DataFrame,
    low_vol_threshold: float = 0.25,
    high_vol_threshold: float = 0.75,
    funding_extreme_positive: float = 0.0003,
    funding_extreme_negative: float = -0.0001,
    iv_skew_threshold: float = 0.15,
    atr_period: int = 14,
    symbol: str = "BTC"
) -> pd.Series:
    """
    Generate vol regime + funding signals.

    Args:
        df: OHLCV DataFrame (may include 'funding_rate', 'iv_skew' columns)
        low_vol_threshold: Vol percentile below this = low vol
        high_vol_threshold: Vol percentile above this = high vol
        funding_extreme_positive: Positive funding threshold
        funding_extreme_negative: Negative funding threshold
        iv_skew_threshold: IV skew threshold for fear detection
        atr_period: ATR calculation period
        symbol: Base symbol for data lookup

    Returns:
        Signal series: 1=long, -1=short, 0=neutral
    """
    strategy = VolRegimeFundingOverlayStrategy(
        low_vol_threshold=low_vol_threshold,
        high_vol_threshold=high_vol_threshold,
        funding_extreme_positive=funding_extreme_positive,
        funding_extreme_negative=funding_extreme_negative,
        iv_skew_threshold=iv_skew_threshold,
        atr_period=atr_period
    )
    return strategy.generate_signals(df, symbol)
