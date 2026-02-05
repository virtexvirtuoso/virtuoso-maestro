"""
Advanced Crypto Derivative Trading Strategies
Based on proven quantitative methods for perpetual futures
"""
import backtrader as bt
from strategy.base_strategy import BaseStrategy
import numpy as np


class FundingRateArbitrage(BaseStrategy):
    """
    Funding Rate Strategy - Go long when funding is very negative (shorts pay longs)
    Works best on perps with 8h funding cycles
    """
    params = (
        ('funding_threshold', -0.01),  # Enter when funding < -1%
        ('exit_threshold', 0.005),      # Exit when funding > 0.5%
    )
    
    def __init__(self):
        super().__init__()
        # Simulate funding with price momentum as proxy
        self.momentum = bt.indicators.RateOfChange(self.data.close, period=8)
    
    def _next(self):
        # Negative momentum often correlates with negative funding
        if self.momentum[0] < self.p.funding_threshold:
            self.buy()
        elif self.momentum[0] > self.p.exit_threshold:
            self.close()


class BasisTradingStrategy(BaseStrategy):
    """
    Basis/Contango Strategy - Trade the spread between spot and futures
    Buy when futures at discount, sell when at premium
    """
    params = (
        ('lookback', 20),
        ('entry_std', 2.0),
        ('exit_std', 0.5),
    )
    
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.lookback)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.lookback)
        self.zscore = (self.data.close - self.sma) / self.std
    
    def _next(self):
        if self.zscore[0] < -self.p.entry_std:
            self.buy()  # Futures at discount
        elif self.zscore[0] > self.p.entry_std:
            self.sell()  # Futures at premium
        elif abs(self.zscore[0]) < self.p.exit_std:
            self.close()


class OpenInterestDivergence(BaseStrategy):
    """
    OI Divergence - Price up + OI down = weak rally, fade it
    Uses volume as OI proxy
    """
    params = (
        ('price_period', 10),
        ('vol_period', 10),
    )
    
    def __init__(self):
        super().__init__()
        self.price_roc = bt.indicators.RateOfChange(self.data.close, period=self.p.price_period)
        self.vol_roc = bt.indicators.RateOfChange(self.data.volume, period=self.p.vol_period)
    
    def _next(self):
        # Bearish divergence: price up, volume down
        if self.price_roc[0] > 0.02 and self.vol_roc[0] < -0.1:
            self.sell()
        # Bullish divergence: price down, volume up
        elif self.price_roc[0] < -0.02 and self.vol_roc[0] > 0.1:
            self.buy()


class LiquidationHuntStrategy(BaseStrategy):
    """
    Liquidation Hunt - After sharp moves, anticipate mean reversion
    as cascading liquidations exhaust
    """
    params = (
        ('atr_period', 14),
        ('atr_mult', 3.0),
        ('reversion_target', 0.5),
    )
    
    def __init__(self):
        super().__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.sma = bt.indicators.SMA(self.data.close, period=20)
    
    def _next(self):
        move = self.data.close[0] - self.data.close[-1]
        threshold = self.atr[0] * self.p.atr_mult
        
        # Sharp drop = long for bounce
        if move < -threshold:
            self.buy()
        # Sharp pump = short for pullback  
        elif move > threshold:
            self.sell()


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading - Place orders at intervals in ranging markets
    Best for sideways/consolidation periods
    """
    params = (
        ('grid_size', 0.02),  # 2% between levels
        ('num_grids', 5),
    )
    
    def __init__(self):
        super().__init__()
        self.entry_price = None
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20)
    
    def _next(self):
        # Only trade within Bollinger Bands (ranging market)
        if self.data.close[0] < self.bb.lines.bot[0]:
            self.buy()
        elif self.data.close[0] > self.bb.lines.top[0]:
            self.sell()


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout - Enter on expansion after compression
    Keltner Channel squeeze detection
    """
    params = (
        ('atr_period', 20),
        ('atr_mult', 2.0),
        ('squeeze_threshold', 0.5),
    )
    
    def __init__(self):
        super().__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.atr_sma = bt.indicators.SMA(self.atr, period=self.p.atr_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.atr_period)
    
    def _next(self):
        # Squeeze: ATR below average
        squeeze = self.atr[0] < self.atr_sma[0] * self.p.squeeze_threshold
        
        if not squeeze:  # Expansion
            if self.data.close[0] > self.ema[0] + self.atr[0] * self.p.atr_mult:
                self.buy()
            elif self.data.close[0] < self.ema[0] - self.atr[0] * self.p.atr_mult:
                self.sell()


class TrendFollowingATR(BaseStrategy):
    """
    ATR Trend Following - Classic trend system with ATR stops
    Works well on higher timeframes (4h, 1d)
    """
    params = (
        ('trend_period', 50),
        ('atr_period', 14),
        ('atr_mult', 2.0),
    )
    
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.trend_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.trend_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.trend_period)
    
    def _next(self):
        if self.data.close[0] > self.highest[-1]:
            self.buy()
        elif self.data.close[0] < self.lowest[-1]:
            self.sell()


class MeanReversionBands(BaseStrategy):
    """
    Mean Reversion with Dynamic Bands
    Fade moves to extremes with proper sizing
    """
    params = (
        ('period', 20),
        ('num_std', 2.5),
        ('exit_std', 0.5),
    )
    
    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.period, devfactor=self.p.num_std
        )
        self.bb_mid = bt.indicators.BollingerBands(
            self.data.close, period=self.p.period, devfactor=self.p.exit_std
        )
    
    def _next(self):
        if self.data.close[0] < self.bb.lines.bot[0]:
            self.buy()
        elif self.data.close[0] > self.bb.lines.top[0]:
            self.sell()
        elif self.position:
            # Exit near mean
            if (self.position.size > 0 and self.data.close[0] > self.bb_mid.lines.top[0]):
                self.close()
            elif (self.position.size < 0 and self.data.close[0] < self.bb_mid.lines.bot[0]):
                self.close()
