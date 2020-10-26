import backtrader as bt

from strategy.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    params = dict()

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.macd = bt.indicators.MACD(self.dataclose)
        self.macd_prev = bt.indicators.MACD(self.dataclose(-1))
        self.mcross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.mcross_prev = bt.indicators.CrossOver(self.macd_prev.macd, self.macd_prev.signal)
        # To keep track of pending orders and buy price/commission
        self.order = None

    @staticmethod
    def get_params():
        return dict()

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
            if self.mcross[0] > 0.0 > self.mcross_prev[0]:
                self.order = self.order_target_percent(target=0.05)
        else:
            if self.mcross[0] < 0.0 or self.mcross_prev[0] > 0.0:
                self.close()
                self.order = self.order_target_percent(target=-0.05)
