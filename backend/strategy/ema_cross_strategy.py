import backtrader as bt

from strategy.base_strategy import BaseStrategy


class EmaCrossStrategy(BaseStrategy):
    params = dict(ema50=50,
                  ema51=51,
                  ema100=100,
                  ema101=101,
                  )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.ema50_value = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.p.ema50)
        self.ema100_value = bt.indicators.ExponentialMovingAverage(self.dataclose, period=self.p.ema100)
        self.ema51_value = bt.indicators.ExponentialMovingAverage(self.dataclose(-1), period=self.p.ema51)
        self.ema101_value = bt.indicators.ExponentialMovingAverage(self.dataclose(-1), period=self.p.ema101)
        self.even_crossover = bt.indicators.CrossOver(self.ema50_value, self.ema100_value)
        self.odd_crossover = bt.indicators.CrossOver(self.ema101_value, self.ema51_value)
        # To keep track of pending orders and buy price/commission
        self.order = None

    @staticmethod
    def get_params():
        return dict(ema50=50,
                    ema51=51,
                    ema100=100,
                    ema101=101,
                    )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Write down: no pending order
        self.order = None

    def _next(self):

        if self.order:
            return

        if not self.position:
            if self.even_crossover[0] > 0 and self.odd_crossover[0] > 0:
                self.order = self.order_target_percent(target=0.05)
        else:
            if self.even_crossover[0] < 0 and self.odd_crossover[0] < 0:
                self.close()
                self.order = self.order_target_percent(target=-0.05)
