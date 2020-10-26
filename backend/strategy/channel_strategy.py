import backtrader as bt

from strategy.base_strategy import BaseStrategy


class ChannelStrategy(BaseStrategy):
    params = dict(period_long_term=50,
                  period_short_term=20
                  )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high(-1)
        self.channel_long_term = bt.indicators.Highest(self.datahigh, period=self.p.period_long_term)
        self.channel_short_term = bt.indicators.Highest(self.datahigh, period=self.p.period_short_term)

        # To keep track of pending orders and buy price/commission
        self.order = None

    @staticmethod
    def get_params():
        return dict(period_long_term=50,
                    period_short_term=20)

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
            if self.dataclose[0] < self.channel_short_term[0] < self.channel_long_term[0]:
                self.order = self.order_target_percent(target=0.05)
        else:
            if self.dataclose[0] > self.channel_long_term[0] > self.channel_short_term[0]:
                self.close()
                self.order = self.order_target_percent(target=-0.05)
