import backtrader as bt

from strategy.base_strategy import BaseStrategy


class MaCrossStrategy(BaseStrategy):
    params = dict(sma_1=50,
                  sma_2=200,
                  prev_sma_1=50,
                  prev_sma_2=200,
                  )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.sma_1 = bt.indicators.SMA(self.dataclose, period=self.p.sma_1)
        self.sma_2 = bt.indicators.SMA(self.dataclose, period=self.p.sma_2)
        self.prev_sma_1 = bt.indicators.SMA(self.dataclose(-1), period=self.p.prev_sma_1)
        self.prev_sma_2 = bt.indicators.SMA(self.dataclose(-1), period=self.p.prev_sma_2)
        self.sma_crossover = bt.indicators.CrossOver(self.sma_1, self.sma_2)
        self.prev_sma_crossover = bt.indicators.CrossOver(self.prev_sma_2, self.prev_sma_1)
        # To keep track of pending orders and buy price/commission
        self.order = None

    @staticmethod
    def get_params():
        return dict(sma_1=60,
                    sma_2=120,
                    prev_sma_1=60,
                    prev_sma_2=120,
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
            if (self.sma_crossover[0] > 0 and self.prev_sma_crossover[0] > 0) or (self.dataclose[0] > self.sma_1[0]):
                self.order = self.order_target_percent(target=0.05)
        else:
            if self.sma_crossover[0] < 0 and self.prev_sma_crossover[0] < 0:
                self.close()
                self.order = self.order_target_percent(target=-0.05)
