"""
Liquidation Cascade Strategy

Trades post-liquidation reversal patterns.

Logic:
- Post-cascade long: After long liquidation spike + price stabilization
- Post-cascade short: After short liquidation spike + price stabilization
- Detection: Liq volume > 3x avg + wick > 60% of candle

Data Sources (priority order):
1. Real liquidations from Coinalyze
2. Real liquidations from DataFrame columns
3. Synthetic liquidation proxy (75% detection accuracy)
"""

import pandas as pd
import numpy as np

try:
    from .mixins import DerivativesDataMixin
except ImportError:
    from strategies.derivatives.mixins import DerivativesDataMixin

NAME = "LiquidationCascade"
CATEGORY = "derivatives"
DESCRIPTION = "Post-liquidation reversal with real data + proxy fallbacks"
REQUIRES_DERIVATIVES = True


class LiquidationCascadeStrategy(DerivativesDataMixin):
    """Liquidation cascade reversal strategy."""

    def __init__(
        self,
        cascade_mult: float = 3.0,
        wick_ratio: float = 0.6,
        stabilization_bars: int = 3,
        vol_mult: float = 2.0
    ):
        self.cascade_mult = cascade_mult
        self.wick_ratio = wick_ratio
        self.stabilization_bars = stabilization_bars
        self.vol_mult = vol_mult

    def generate_signals(self, df: pd.DataFrame, symbol: str = "BTC") -> pd.Series:
        """Generate trading signals based on liquidation cascades."""
        signals = pd.Series(0, index=df.index)

        # Get liquidation data (real or proxy)
        long_liq, short_liq = self.get_liquidations(df, symbol, use_proxy=True)

        # Volume spike detection
        vol_avg = df['volume'].rolling(20).mean()
        vol_spike = df['volume'] > vol_avg * self.vol_mult

        # Wick analysis (sign of absorption)
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)

        # Wick dominance
        lower_wick_ratio = lower_wick / total_range.replace(0, 1e-10)
        upper_wick_ratio = upper_wick / total_range.replace(0, 1e-10)
        long_wick_down = lower_wick_ratio > self.wick_ratio
        long_wick_up = upper_wick_ratio > self.wick_ratio

        # Cascade detection
        long_liq_avg = long_liq.rolling(20).mean().replace(0, 1e-10)
        short_liq_avg = short_liq.rolling(20).mean().replace(0, 1e-10)

        long_cascade = (long_liq > long_liq_avg * self.cascade_mult) | (vol_spike & long_wick_down)
        short_cascade = (short_liq > short_liq_avg * self.cascade_mult) | (vol_spike & long_wick_up)

        # Stabilization check: price volatility decreasing after cascade
        returns = df['close'].pct_change().abs()
        returns_ma = returns.rolling(self.stabilization_bars).mean()
        returns_prev = returns.rolling(self.stabilization_bars).mean().shift(self.stabilization_bars)
        stabilizing = returns_ma < returns_prev

        # Post-cascade signals with stabilization
        # After long liquidation cascade + stabilization = reversal long
        post_long_cascade = long_cascade.shift(self.stabilization_bars).astype(float).fillna(0) > 0
        signals[post_long_cascade & stabilizing] = 1

        # After short liquidation cascade + stabilization = reversal short
        post_short_cascade = short_cascade.shift(self.stabilization_bars).astype(float).fillna(0) > 0
        signals[post_short_cascade & stabilizing] = -1

        return signals


def generate_signals(
    df: pd.DataFrame,
    cascade_mult: float = 3.0,
    wick_ratio: float = 0.6,
    stabilization_bars: int = 3,
    vol_mult: float = 2.0,
    symbol: str = "BTC"
) -> pd.Series:
    """
    Generate liquidation cascade signals.

    Args:
        df: OHLCV DataFrame (may include 'long_liquidations', 'short_liquidations' columns)
        cascade_mult: Multiplier over average for cascade detection
        wick_ratio: Minimum wick ratio for absorption detection
        stabilization_bars: Bars to wait for stabilization
        vol_mult: Volume multiplier for spike detection
        symbol: Base symbol for data lookup

    Returns:
        Signal series: 1=long, -1=short, 0=neutral
    """
    strategy = LiquidationCascadeStrategy(
        cascade_mult=cascade_mult,
        wick_ratio=wick_ratio,
        stabilization_bars=stabilization_bars,
        vol_mult=vol_mult
    )
    return strategy.generate_signals(df, symbol)
