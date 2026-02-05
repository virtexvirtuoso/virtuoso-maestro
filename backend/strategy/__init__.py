from strategy.bollinger_bands_strategy import BollingerBandsStrategy
from strategy.channel_strategy import ChannelStrategy
from strategy.ema_cross_strategy import EmaCrossStrategy
from strategy.fernando_strategy import FernandoStrategy
from strategy.ichimoku_strategy import IchimokuStrategy
from strategy.ma_cross_strategy import MaCrossStrategy
from strategy.macd_strategy import MACDStrategy
from strategy.rsi_strategy import RSIStrategy
from strategy.macd_strategyx import MACDStrategyX
from strategy.scalping_strategies import (
    VWAPStrategy, ScalpRSIStrategy, MomentumBreakoutStrategy,
    StochRSIStrategy, EMARibbonStrategy
)
from strategy.advanced_strategies import (
    FundingRateArbitrage, BasisTradingStrategy, OpenInterestDivergence,
    LiquidationHuntStrategy, GridTradingStrategy, VolatilityBreakoutStrategy,
    TrendFollowingATR, MeanReversionBands
)

def register(cls):
    return {str(cls.__name__): cls}


__STRATEGY_CATALOG__ = {
    # Classic
    **register(BollingerBandsStrategy),
    **register(ChannelStrategy),
    **register(EmaCrossStrategy),
    **register(IchimokuStrategy),
    **register(MaCrossStrategy),
    **register(MACDStrategy),
    **register(RSIStrategy),
    **register(FernandoStrategy),
    **register(MACDStrategyX),
    # Scalping
    **register(VWAPStrategy),
    **register(ScalpRSIStrategy),
    **register(MomentumBreakoutStrategy),
    **register(StochRSIStrategy),
    **register(EMARibbonStrategy),
    # Advanced Derivatives
    **register(FundingRateArbitrage),
    **register(BasisTradingStrategy),
    **register(OpenInterestDivergence),
    **register(LiquidationHuntStrategy),
    **register(GridTradingStrategy),
    **register(VolatilityBreakoutStrategy),
    **register(TrendFollowingATR),
    **register(MeanReversionBands),
}
