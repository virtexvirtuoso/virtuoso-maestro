import backtrader as bt

from strategy.base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    params = dict(period=20)

    def __init__(self):
        super().__init__()
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.bollinger_bands = bt.indicators.BollingerBands(period=self.p.period)
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

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.bollinger_bands.l.bot[0]:
                # Keep track of the created order to avoid a 2nd order
                self.order = self.order_target_percent(target=0.05)
        else:
            if self.dataclose[0] > self.bollinger_bands.l.top[0]:
                self.close()
                # Keep track of the created order to avoid a 2nd order
                self.order = self.order_target_percent(target=-0.05)
