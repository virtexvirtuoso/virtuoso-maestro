"""
Derivatives Strategies - 6 crypto-native strategies using real derivatives data

Strategies leverage Maestro's derivatives data infrastructure:
- Real data from Coinalyze (OI, funding, liquidations, L/S ratio)
- Pre-computed fusion signals (data/signals/*.json)
- Calibrated synthetic proxies (87% direction accuracy)

Strategies:
- RealFundingRate: Contrarian funding rate signals
- SpotPerpBasis: Spot-perp basis mean reversion
- RealOIMomentum: OI-price confirmation/divergence
- LiquidationCascade: Post-liquidation reversal
- CrossExchangeOI: Cross-exchange OI divergence
- VolRegimeFunding: Vol regime + funding overlay

Usage:
    from strategies.derivatives import RealFundingRateStrategy
    from strategies.derivatives.mixins import DerivativesDataMixin

    strategy = RealFundingRateStrategy()
    signals = strategy.generate_signals(df, symbol='BTC')
"""

from .mixins import DerivativesDataMixin
from .real_funding_rate import RealFundingRateStrategy
from .spot_perp_basis import SpotPerpBasisStrategy
from .real_oi_momentum import RealOIMomentumStrategy
from .liquidation_cascade import LiquidationCascadeStrategy
from .cross_exchange_oi import CrossExchangeOIDivergenceStrategy
from .vol_regime_funding import VolRegimeFundingOverlayStrategy

__all__ = [
    'DerivativesDataMixin',
    'RealFundingRateStrategy',
    'SpotPerpBasisStrategy',
    'RealOIMomentumStrategy',
    'LiquidationCascadeStrategy',
    'CrossExchangeOIDivergenceStrategy',
    'VolRegimeFundingOverlayStrategy',
]

# Strategy name to class mapping
STRATEGY_CLASSES = {
    'RealFundingRate': RealFundingRateStrategy,
    'SpotPerpBasis': SpotPerpBasisStrategy,
    'RealOIMomentum': RealOIMomentumStrategy,
    'LiquidationCascade': LiquidationCascadeStrategy,
    'CrossExchangeOI': CrossExchangeOIDivergenceStrategy,
    'VolRegimeFunding': VolRegimeFundingOverlayStrategy,
}
