import backtrader as bt
from strategy.base_strategy import BaseStrategy
from backtrader import indicators as btind


class BollingerBandsW(bt.Indicator):
    alias = ('BBW',)
    lines = ('bbw',)
    params = (('period', 20), ('devfactor', 2.0), ('movav', btind.MovingAverageSimple),)

    plotinfo = dict(subplot=True,
                    bbw=dict(_name='bbw', color='green', ls='--', _skipnan=True), )

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(self.datas[0],
                                                 period=self.params.period,
                                                 devfactor=self.params.devfactor)
        self.lines.bbw = (self.boll.top - self.boll.bot) / self.boll.mid


class VolatilityLevel(bt.Indicator):
    alias = ('VolatilityLevelIndex',)
    lines = ('bbw', 'VLI_fast', 'VLI_top', 'VLI_slow',)
    params = (
        ('period', 20),
        ('devfactor', 2.0),
        ('movav', btind.MovingAverageSimple),
        ('fast_period', 20),
        ('slow_period', 100),
    )
    plotinfo = dict(subplot=True,
                    bbw=dict(_name='bbw', color='green', ls='--', _skipnan=True),
                    VLI_fast=dict(_name='VLI_fast', color='red', ls='--', _skipnan=True),
                    VLI_top=dict(_name='VLI_top', color='lightred', ls='--', _skipnan=True),
                    VLI_slow=dict(_name='VLI_slow', color='blue', ls='--', _skipnan=True),
                    )

    def __init__(self):
        self.lines.bbw = BollingerBandsW(self.data, period=self.params.period, devfactor=self.params.devfactor)
        self.lines.VLI_fast = self.p.movav(self.lines.bbw, period=self.p.fast_period)
        self.lines.VLI_slow = self.p.movav(self.lines.bbw, period=self.p.slow_period)
        std = bt.ind.StdDev(self.lines.bbw, period=self.p.slow_period)
        self.lines.VLI_top = self.lines.VLI_slow + 2 * std


class FernandoStrategy(BaseStrategy):

    ## Strategy is based on Alain Glucksmann's Backtesting of Trading Strategies for Bitcoin thesis. This uses custom indicator BollingerBandsWidth and VolatilityLevelIndex
    ## https://ethz.ch/content/dam/ethz/special-interest/mtec/chair-of-entrepreneurial-risks-dam/documents/dissertation/master%20thesis/Master_Thesis_Gl%C3%BCcksmann_13June2019.pdf

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datavolume = self.datas[0].volume
        self.datalow = self.datas[0].low
        self.sma_veryfast = btind.MovingAverageSimple(self.dataclose, period=10)
        self.sma_fast = btind.MovingAverageSimple(self.dataclose, period=20)
        self.sma_mid = btind.MovingAverageSimple(self.dataclose, period=50)
        self.sma_slow = btind.MovingAverageSimple(self.dataclose, period=100)
        self.sma_veryslow = btind.MovingAverageSimple(self.dataclose, period=200)

        # ### BollingerBandsWidth = upperband - lowerband/middleband.
        # When price action becomes more volatile, the difference between the top- and bottom-band widens
        self.bbw = BollingerBandsW()
        self.boll = btind.BollingerBands(self.dataclose)
        self.std = btind.StdDev(self.bbw.l.bbw, period=100)
        self.lines_bbw = (self.boll.l.top - self.boll.l.bot) / self.boll.l.mid
        ## VolatilityLevelIndex contains four different lines, where one of them is the Bollinger Band Width (BBW)
        self.volatility_level = VolatilityLevel()
        self.low_volatility_level = self.volatility_level.l.VLI_fast < self.volatility_level.l.VLI_slow
        self.high_volatility_level = self.volatility_level.l.VLI_fast > self.volatility_level.l.VLI_slow
        self.extreme_volatility_level = self.bbw.l.bbw > self.volatility_level.l.VLI_top
        ## Volume Conditions
        self.vol_condition = (btind.MovingAverageSimple(self.datavolume, period=10) >
                              btind.MovingAverageSimple(self.datavolume, period=50))

        ## CrossUp & CrossDown
        self.crossdown_boll_top = bt.ind.CrossDown(self.dataclose, self.boll.top)
        self.crossup_boll_bot = bt.ind.CrossUp(self.dataclose, self.boll.bot)

        ##Pricing
        self.highest_high = btind.Highest(self.dataclose, period=20)
        self.low_of_last_candle = self.datalow[0]
        self.close_of_price = self.dataclose[0]
        # self.trade_profit = ?
        # self.stopwin = ?
        self.stop_win = None
        self.stop_loss = None
        self.order = None

    @staticmethod
    def get_params():
        return dict()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.stop_loss = self.stop_loss if self.stop_loss else 0.05 * self.dataclose[-1]

                take_profit = order.executed.price * (1.0 + self.stop_loss)
                sl_ord = self.sell(exectype=bt.Order.Stop,
                                   price=order.executed.price * (1.0 - self.stop_loss))
                sl_ord.addinfo(name="Stop")
                tkp_ord = self.buy(exectype=bt.Order.Limit,
                                   price=take_profit)
                tkp_ord.addinfo(name="Prof")

                if self.stop_win:
                    self.sell(price=(order.executed.price * (1 + self.stop_win)),
                              exectype=bt.Order.Limit)

        # Write down: no pending order
        self.order = None

    def _next(self):
        if self.order:
            return

        ## For long the cross-down of the candle-close with the upperband are used as entry-signals
        if self.crossdown_boll_top and self.vol_condition:
            if self.close_of_price > self.sma_fast:
                if self.bbw.bbw < self.volatility_level.l.VLI_top:
                    if self.low_volatility_level:
                        if self.sma_mid > self.sma_veryslow:
                            self.buy()
                    else:
                        self.buy()
            elif self.sma_slow > self.sma_veryslow:
                self.buy()
                self.stop_loss = self.low_of_last_candle  # stoploss at: : low of last candle

        ## Close cross-down of a candle-close with the lower band
        if self.crossup_boll_bot and self.vol_condition:
            self.close()
            self.stop_loss = None
            self.stop_win = None
            # if trade_profit > 3% add stopwin at 1%
            portfolio_value = self.broker.get_value()
            trade_profit = self.broker.get_value([self.data]) / portfolio_value

            if trade_profit > 0.03:
                self.stop_win = 0.01
            elif trade_profit > 0.20:
                self.stop_win = 0.15
            elif trade_profit > 0.25:
                self.stop_win = 0.20
            elif trade_profit > 0.30:
                self.stop_win = 0.25
            elif trade_profit > 0.35:
                self.stop_win = 0.30
            elif trade_profit > 0.40:
                self.stop_win = 0.35

        ##Short Inverse of buy stratedgy
        if self.crossup_boll_bot and self.vol_condition:
            self.sell()
            self.stop_loss = self.highest_high
