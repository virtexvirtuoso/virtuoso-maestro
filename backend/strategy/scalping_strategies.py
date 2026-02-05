"""
Scalping Strategies for Lower Timeframes
"""
import backtrader as bt
from strategy.base_strategy import BaseStrategy


class VWAPStrategy(BaseStrategy):
    """VWAP Mean Reversion - buy below VWAP, sell above"""
    params = (
        ('period', 20),
        ('dev_factor', 1.5),
    )
    
    def __init__(self):
        super().__init__()
        self.vwap = bt.indicators.WeightedMovingAverage(
            self.data.close, period=self.p.period,
            weights=self.data.volume
        )
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.period)
    
    def _next(self):
        if self.data.close[0] < self.vwap[0] - self.p.dev_factor * self.std[0]:
            self.buy()
        elif self.data.close[0] > self.vwap[0] + self.p.dev_factor * self.std[0]:
            self.sell()


class ScalpRSIStrategy(BaseStrategy):
    """Quick RSI scalping - tight entries/exits"""
    params = (
        ('rsi_period', 7),
        ('oversold', 25),
        ('overbought', 75),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
    
    def _next(self):
        if self.rsi[0] < self.p.oversold:
            self.buy()
        elif self.rsi[0] > self.p.overbought:
            self.sell()


class MomentumBreakoutStrategy(BaseStrategy):
    """Breakout with volume confirmation"""
    params = (
        ('lookback', 20),
        ('vol_mult', 1.5),
    )
    
    def __init__(self):
        super().__init__()
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.lookback)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.lookback)
        self.vol_sma = bt.indicators.SMA(self.data.volume, period=self.p.lookback)
    
    def _next(self):
        vol_spike = self.data.volume[0] > self.vol_sma[0] * self.p.vol_mult
        if self.data.close[0] > self.highest[-1] and vol_spike:
            self.buy()
        elif self.data.close[0] < self.lowest[-1] and vol_spike:
            self.sell()


class StochRSIStrategy(BaseStrategy):
    """Stochastic RSI - popular for scalping"""
    params = (
        ('rsi_period', 14),
        ('stoch_period', 14),
        ('k_period', 3),
        ('d_period', 3),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.stoch = bt.indicators.Stochastic(
            self.rsi, period=self.p.stoch_period,
            period_dfast=self.p.k_period, period_dslow=self.p.d_period
        )
    
    def _next(self):
        if self.stoch.percK[0] < 20 and self.stoch.percK[0] > self.stoch.percK[-1]:
            self.buy()
        elif self.stoch.percK[0] > 80 and self.stoch.percK[0] < self.stoch.percK[-1]:
            self.sell()


class EMARibbonStrategy(BaseStrategy):
    """EMA Ribbon - trend strength"""
    params = (
        ('fast', 8),
        ('medium', 13),
        ('slow', 21),
    )
    
    def __init__(self):
        super().__init__()
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.fast)
        self.ema_med = bt.indicators.EMA(self.data.close, period=self.p.medium)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.slow)
    
    def _next(self):
        if self.ema_fast[0] > self.ema_med[0] > self.ema_slow[0]:
            if self.ema_fast[-1] <= self.ema_med[-1]:  # Just crossed
                self.buy()
        elif self.ema_fast[0] < self.ema_med[0] < self.ema_slow[0]:
            if self.ema_fast[-1] >= self.ema_med[-1]:
                self.sell()
