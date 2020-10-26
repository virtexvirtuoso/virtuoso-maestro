import backtrader as bt

from strategy.base_strategy import BaseStrategy

class MACDStrategyX(BaseStrategy):
  params = dict(macd1=12,
                macd2= 26,
                macdsig=9,
                trailpercent=0.40,
                smaperiod=30,
                dirperiod=10,
                )

def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.macd = bt.indicators.MACD(self.dataclose)
        self.macd_prev = bt.indicators.MACD(self.dataclose(-1))
        # Cross of macd.macd and macd.signal
        self.mcross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.mcross_prev = bt.indicators.CrossOver(self.macd_prev.macd, self.macd_prev.signal)
        # Control market trend
        self.sma = bt.indicators.SMA(self.data, period=self.p.smaperiod)
        self.smadir = self.sma - self.sma(-self.p.dirperiod)
        # To keep track of pending orders and buy price/commission
        self.order = None

@staticmethod
def get_params():
        return dict(macd1=12,
                    macd2= 26,
                    macdsig=9,
                    trailpercent=0.40,
                    smaperiod=30,
                    dirperiod=10,
                    )

def notify_order(self, order):
    if order.status == order.Completed:
        pass

    if not order.alive():
        self.order = None  # No pending orders

def start(self):
        self.order = None  # Avoid operrations on pending order

def next(self):
        if self.order:
            return  # pending order execution

        if not self.position:  # not in the market
            if self.mcross[0] > 0.0 and self.smadir < 0.0:
                self.order = self.buy()
                self.order = 'none'

        elif self.order is None: # Position in Market
            self.order = self.sell(exectype=bt.Order.StopTrail,trailpercent=self.p.trailpercent)
            tcheck = self.data.close * (1.0 - self.p.trailpercent)
