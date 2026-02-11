"""
Real OI Momentum Strategy

Trades OI-price confirmation and divergence patterns.

Logic:
- Long: OI rising + price rising (trend confirmation, new money entering)
- Short: OI rising + price falling (distribution, smart money selling)
- Exit: OI divergence (price continues but OI declines = exhaustion)

Data Sources (priority order):
1. Real OI from Coinalyze (unlimited daily history)
2. Real OI from DataFrame column
3. Synthetic OI proxy (65% correlation)
"""

import pandas as pd
import numpy as np

try:
    from .mixins import DerivativesDataMixin
except ImportError:
    from strategies.derivatives.mixins import DerivativesDataMixin

NAME = "RealOIMomentum"
CATEGORY = "derivatives"
DESCRIPTION = "OI-price confirmation/divergence with real data + proxy fallbacks"
REQUIRES_DERIVATIVES = True


class RealOIMomentumStrategy(DerivativesDataMixin):
    """OI momentum strategy using real data with proxy fallback."""

    def __init__(
        self,
        oi_change_threshold: float = 5.0,
        price_change_threshold: float = 2.0,
        divergence_threshold: float = -3.0,
        lookback: int = 24
    ):
        self.oi_change_threshold = oi_change_threshold
        self.price_change_threshold = price_change_threshold
        self.divergence_threshold = divergence_threshold
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame, symbol: str = "BTC") -> pd.Series:
        """Generate trading signals based on OI-price relationship."""
        signals = pd.Series(0, index=df.index)

        # Get OI change (real or proxy)
        oi_change = self.get_oi_change(df, symbol, period=self.lookback)
        price_change = df['close'].pct_change(self.lookback) * 100

        if oi_change.std() < 1e-10:
            return signals  # No valid data

        # OI rising significantly
        oi_rising = oi_change > self.oi_change_threshold
        oi_falling = oi_change < self.divergence_threshold

        # Price direction
        price_rising = price_change > self.price_change_threshold
        price_falling = price_change < -self.price_change_threshold

        # Confirmation patterns
        # Long: OI up + price up = new money entering long
        bullish_confirmation = oi_rising & price_rising

        # Short: OI up + price down = distribution (smart money selling to latecomers)
        bearish_distribution = oi_rising & price_falling

        signals[bullish_confirmation] = 1
        signals[bearish_distribution] = -1

        # Divergence exits
        # Price rising but OI falling = exhaustion (exit longs)
        # Price falling but OI falling = capitulation ending (exit shorts)
        exhaustion = oi_falling & (price_rising | price_falling)
        signals[exhaustion] = 0

        return signals


def generate_signals(
    df: pd.DataFrame,
    oi_change_threshold: float = 5.0,
    price_change_threshold: float = 2.0,
    divergence_threshold: float = -3.0,
    lookback: int = 24,
    symbol: str = "BTC"
) -> pd.Series:
    """
    Generate OI momentum signals.

    Args:
        df: OHLCV DataFrame (may include 'oi' or 'open_interest' column)
        oi_change_threshold: OI % change threshold for rising
        price_change_threshold: Price % change threshold
        divergence_threshold: OI % change threshold for divergence
        lookback: Period for change calculation
        symbol: Base symbol for data lookup

    Returns:
        Signal series: 1=long, -1=short, 0=neutral
    """
    strategy = RealOIMomentumStrategy(
        oi_change_threshold=oi_change_threshold,
        price_change_threshold=price_change_threshold,
        divergence_threshold=divergence_threshold,
        lookback=lookback
    )
    return strategy.generate_signals(df, symbol)
