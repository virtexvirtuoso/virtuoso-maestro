"""
Spot-Perp Basis Strategy

Trades the premium/discount between spot and perpetual prices.

Logic:
- Long: Basis < -0.5% (backwardation, perp discount)
- Short: Basis > +1.0% (contango, perp premium)
- Exit: Basis mean-reverts to 0

Data Sources (priority order):
1. Real spot/perp prices from DataFrame columns
2. Pre-computed basis from JSON signals
3. Synthetic basis proxy from OHLCV
"""

import pandas as pd
import numpy as np

try:
    from .mixins import DerivativesDataMixin
except ImportError:
    from strategies.derivatives.mixins import DerivativesDataMixin

NAME = "SpotPerpBasis"
CATEGORY = "derivatives"
DESCRIPTION = "Spot-perp basis mean reversion with real data + proxy fallbacks"
REQUIRES_DERIVATIVES = True


class SpotPerpBasisStrategy(DerivativesDataMixin):
    """Basis trading strategy using spot-perp spread."""

    def __init__(
        self,
        long_threshold: float = -0.5,
        short_threshold: float = 1.0,
        exit_threshold: float = 0.1,
        lookback: int = 24,
        use_zscore: bool = True,
        zscore_extreme: float = 2.0
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback
        self.use_zscore = use_zscore
        self.zscore_extreme = zscore_extreme

    def generate_signals(self, df: pd.DataFrame, symbol: str = "BTC") -> pd.Series:
        """Generate trading signals based on basis."""
        signals = pd.Series(0, index=df.index)

        # Get basis data (real or proxy)
        basis = self.get_basis(df, symbol, use_proxy=True)

        if basis.std() < 1e-10:
            return signals  # No valid data

        # Z-score based detection
        if self.use_zscore:
            basis_ma = basis.rolling(self.lookback, min_periods=1).mean()
            basis_std = basis.rolling(self.lookback, min_periods=1).std().replace(0, 1e-10)
            z_score = (basis - basis_ma) / basis_std

            # Extreme backwardation = mean reversion long
            extreme_backwardation_z = z_score < -self.zscore_extreme

            # Extreme contango = mean reversion short
            extreme_contango_z = z_score > self.zscore_extreme
        else:
            extreme_backwardation_z = pd.Series(False, index=df.index)
            extreme_contango_z = pd.Series(False, index=df.index)

        # Absolute threshold based signals
        extreme_backwardation_abs = basis < self.long_threshold
        extreme_contango_abs = basis > self.short_threshold

        # Combine signals
        signals[(extreme_backwardation_abs | extreme_backwardation_z)] = 1   # Long
        signals[(extreme_contango_abs | extreme_contango_z)] = -1  # Short

        # Exit zones (basis near zero)
        neutral_basis = (basis > -self.exit_threshold) & (basis < self.exit_threshold)
        signals[neutral_basis] = 0

        return signals


def generate_signals(
    df: pd.DataFrame,
    long_threshold: float = -0.5,
    short_threshold: float = 1.0,
    exit_threshold: float = 0.1,
    lookback: int = 24,
    use_zscore: bool = True,
    symbol: str = "BTC"
) -> pd.Series:
    """
    Generate spot-perp basis signals.

    Args:
        df: OHLCV DataFrame (may include 'basis', 'spot_price', 'perp_price' columns)
        long_threshold: Basis % below this = long (backwardation)
        short_threshold: Basis % above this = short (contango)
        exit_threshold: Basis % within +/- this = exit
        lookback: Rolling period for z-score
        use_zscore: Use z-score based detection
        symbol: Base symbol for data lookup

    Returns:
        Signal series: 1=long, -1=short, 0=neutral
    """
    strategy = SpotPerpBasisStrategy(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        exit_threshold=exit_threshold,
        lookback=lookback,
        use_zscore=use_zscore
    )
    return strategy.generate_signals(df, symbol)
