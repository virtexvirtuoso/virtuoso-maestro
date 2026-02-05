from strategy.bollinger_bands_strategy import BollingerBandsStrategy
from strategy.channel_strategy import ChannelStrategy
from strategy.ema_cross_strategy import EmaCrossStrategy
from strategy.fernando_strategy import FernandoStrategy
from strategy.ichimoku_strategy import IchimokuStrategy
from strategy.ma_cross_strategy import MaCrossStrategy
from strategy.macd_strategy import MACDStrategy
from strategy.rsi_strategy import RSIStrategy
from strategy.macd_strategyx import MACDStrategyX

def register(cls):
    return {str(cls.__name__): cls}


__STRATEGY_CATALOG__ = {
    **register(BollingerBandsStrategy),
    **register(ChannelStrategy),
    **register(EmaCrossStrategy),
    **register(IchimokuStrategy),
    **register(MaCrossStrategy),
    **register(MACDStrategy),
    **register(RSIStrategy),
    **register(FernandoStrategy),
    **register(MACDStrategyX),
}
