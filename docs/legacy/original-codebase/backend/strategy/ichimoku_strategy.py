import backtrader as bt

from strategy.base_strategy import BaseStrategy


class IchimokuStrategy(BaseStrategy):
    params = dict()

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.dataclose = self.datas[0].close
        self.datalow = self.datas[0].low
        self.ichimoku = bt.indicators.Ichimoku()
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

        senkou_span_a = self.ichimoku.senkou_span_a[0]
        senkou_span_b = self.ichimoku.senkou_span_b[0]

        if not self.position:
            if (self.dataclose[0] > senkou_span_a and self.dataclose[0] > senkou_span_b and
                    (self.datalow[0] < senkou_span_a or self.datalow[0] < senkou_span_b)):
                self.order = self.order_target_percent(target=0.05)
        else:
            if not (self.dataclose[0] > senkou_span_a and self.dataclose[0] > senkou_span_b and
                    (self.datalow[0] < senkou_span_a or self.datalow[0] < senkou_span_b)):
                self.close()
                self.order = self.order_target_percent(target=-0.05)
