from abc import abstractmethod
from logging import Logger
import backtrader as bt

from logger.logger_builder import LoggerBuilder


class BaseStrategy(bt.Strategy):
    params = dict(tid=None, optim_engine=None)

    def __init__(self, logger: Logger = None):
        super().__init__()
        self.logger = logger if logger else LoggerBuilder(self.__class__.__name__).build()

    def log(self, txt):
        """ Logging function fot this strategy"""
        dt = self.datas[0].datetime.date(0)
        self.logger.info(f'feed date: {dt.isoformat()}, {txt}')

    def next(self):
        if self.p.optim_engine:
            self.p.optim_engine.update_progress()
        self._next()

    @abstractmethod
    def _next(self):
        raise NotImplementedError()

    def stop(self):
        # self.log(f'Ending Value {self.broker.getvalue()} (params: {self.params.__dict__})')
        pass

    @staticmethod
    @abstractmethod
    def get_params():
        raise NotImplementedError()
