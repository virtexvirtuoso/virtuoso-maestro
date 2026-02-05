import backtrader as bt

from strategy.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    params = dict(period=20)

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.rsi = bt.indicators.RSI(self.dataclose, period=self.p.period)
        self.rsi_prev = bt.indicators.RSI(self.dataclose(-1), period=self.p.period)
        # To keep track of pending orders and buy price/commission
        self.order = None

    @staticmethod
    def get_params():
        return dict(period=20)

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
            if self.rsi[0] > 60 >= self.rsi[0]:
                self.order = self.order_target_percent(target=0.05)
        else:
            if not (self.rsi[0] > 60 >= self.rsi[0]):
                self.close()
                self.order = self.order_target_percent(target=-0.05)
