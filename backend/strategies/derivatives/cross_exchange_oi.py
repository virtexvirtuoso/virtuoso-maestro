"""
Cross-Exchange OI Divergence Strategy

Trades position migration patterns between exchanges.

Logic:
- OI rising on one exchange while falling on another = position migration
- Follow the destination exchange (larger, smarter money flows)
- Binance OI surge + others falling = bullish (retail following smart money)

Data Sources (priority order):
1. Multi-exchange OI from Coinalyze (Binance, Bybit, OKX)
2. Single OI with price divergence as proxy
"""

import pandas as pd
import numpy as np

try:
    from .mixins import DerivativesDataMixin
except ImportError:
    from strategies.derivatives.mixins import DerivativesDataMixin

NAME = "CrossExchangeOI"
CATEGORY = "derivatives"
DESCRIPTION = "Cross-exchange OI divergence with multi-source data"
REQUIRES_DERIVATIVES = True


class CrossExchangeOIDivergenceStrategy(DerivativesDataMixin):
    """Cross-exchange OI divergence strategy."""

    def __init__(
        self,
        oi_change_threshold: float = 5.0,
        price_threshold: float = 2.0,
        lookback: int = 24,
        divergence_threshold: float = 3.0
    ):
        self.oi_change_threshold = oi_change_threshold
        self.price_threshold = price_threshold
        self.lookback = lookback
        self.divergence_threshold = divergence_threshold

    def generate_signals(self, df: pd.DataFrame, symbol: str = "BTC") -> pd.Series:
        """Generate trading signals based on cross-exchange OI divergence."""
        signals = pd.Series(0, index=df.index)

        # Check for multi-exchange OI data
        binance_col = next((c for c in df.columns if 'binance' in c.lower() and 'oi' in c.lower()), None)
        bybit_col = next((c for c in df.columns if 'bybit' in c.lower() and 'oi' in c.lower()), None)
        okx_col = next((c for c in df.columns if 'okx' in c.lower() and 'oi' in c.lower()), None)

        if binance_col and (bybit_col or okx_col):
            # Real multi-exchange data available
            binance_oi = df[binance_col].pct_change(self.lookback) * 100
            other_oi = df[bybit_col or okx_col].pct_change(self.lookback) * 100

            # Binance increasing while others decreasing = money flowing to Binance
            binance_up = binance_oi > self.oi_change_threshold
            other_down = other_oi < -self.divergence_threshold

            # Bullish divergence: smart money accumulating on Binance
            signals[binance_up & other_down] = 1

            # Bearish divergence: money leaving Binance for others
            binance_down = binance_oi < -self.oi_change_threshold
            other_up = other_oi > self.divergence_threshold
            signals[binance_down & other_up] = -1

        else:
            # Fallback: Use OI-price divergence as proxy
            oi_change = self.get_oi_change(df, symbol, period=self.lookback)
            price_change = df['close'].pct_change(self.lookback) * 100

            # OI-price divergence (smart money positioning)
            # OI up + price down = accumulation (bullish)
            oi_up = oi_change > self.oi_change_threshold
            price_down = price_change < -self.price_threshold
            signals[oi_up & price_down] = 1

            # OI up + price up strongly = distribution coming (bearish)
            price_up_strong = price_change > self.price_threshold * 2
            signals[oi_up & price_up_strong] = -1

        return signals


def generate_signals(
    df: pd.DataFrame,
    oi_change_threshold: float = 5.0,
    price_threshold: float = 2.0,
    lookback: int = 24,
    divergence_threshold: float = 3.0,
    symbol: str = "BTC"
) -> pd.Series:
    """
    Generate cross-exchange OI divergence signals.

    Args:
        df: OHLCV DataFrame (may include exchange-specific OI columns)
        oi_change_threshold: OI % change threshold
        price_threshold: Price % change threshold
        lookback: Period for change calculation
        divergence_threshold: Cross-exchange divergence threshold
        symbol: Base symbol for data lookup

    Returns:
        Signal series: 1=long, -1=short, 0=neutral
    """
    strategy = CrossExchangeOIDivergenceStrategy(
        oi_change_threshold=oi_change_threshold,
        price_threshold=price_threshold,
        lookback=lookback,
        divergence_threshold=divergence_threshold
    )
    return strategy.generate_signals(df, symbol)
