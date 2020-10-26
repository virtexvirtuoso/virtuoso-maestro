from abc import abstractmethod
from datetime import datetime, timedelta
from functools import singledispatch, partial
from logging import Logger
from threading import Thread
from typing import Optional, Dict
import math
import backtrader as bt
import backtrader.analyzers as btanalyzers
import numpy as np
import pytz
import base64
import traceback
# Import mock pyfolio instead of real one
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyfolio_mock import create_full_tear_sheet, create_returns_tear_sheet, create_position_tear_sheet, create_txn_tear_sheet
# Mock utils module
class MockUtils:
    @staticmethod
    def check_intraday(*args, **kwargs):
        return None
utils = MockUtils()
from rethinkdb import RethinkDB
from io import BytesIO
from config.config_reader import RethinkDbConfig, OptimizationOutputConfig, WalkForwardConfig
from datasource.providers import DataSourceProviders
from engine.optimizer import OptimizationType
from logger.logger_builder import LoggerBuilder
from strategy.base_strategy import BaseStrategy
from utils.fractional_commission import CommInfoFractional


class OptimizationEngine(Thread):

    def __init__(self, tid: str, test_name: str, test_timestamp: datetime, provider: DataSourceProviders,
                 symbol: str, bin_size: str, strategy: BaseStrategy, strategy_params: Dict[str, float],
                 cash: float, commissions: float,
                 start_date: datetime,
                 end_date: datetime,
                 rethinkdb_config: RethinkDbConfig,
                 optimization_output: OptimizationOutputConfig, walkforward_config: WalkForwardConfig,
                 logger: Logger = None):
        super().__init__(name=f'optim_{test_name}')
        self.tid = tid
        self.test_name = test_name
        self.test_timestamp = test_timestamp
        self.provider = provider
        self.symbol = symbol
        self.bin_size = bin_size
        self.strategy = strategy
        self.strategy_params = strategy_params
        self.cash = cash
        self.commissions = commissions
        self.start_date = start_date
        self.end_date = end_date
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.walkforward_config = walkforward_config
        self.logger = logger if logger else LoggerBuilder(name=self.name).build()
        self.cur_bar = 0
        self.total_bars = None
        self.rconn = None

    def get_cerebro(self):
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=self.commissions)
        cerebro.broker.addcommissioninfo(CommInfoFractional())
        return cerebro

    @abstractmethod
    def update_progress(self):
        raise NotImplementedError()

    def add_strategy(self, cerebro: bt.Cerebro, opt_params=None):

        params = self.strategy.get_params()

        for k, v in self.strategy_params.items():
            if k in params:
                params[k] = type(params[k])(v)

        if opt_params:
            for k, v in opt_params.items():
                params[k] = v

        cerebro.addstrategy(
            self.strategy,
            tid=self.tid,
            optim_engine=self,
            **params
        )

    def add_opt_strategy(self, cerebro: bt.Cerebro, strategy: BaseStrategy, data_size: int):
        params_grid = {}
        params = self.strategy.get_params()

        for k, v in self.strategy_params.items():
            if k in params:
                params[k] = type(params[k])(v)

        for k, v in params.items():
            params_grid[k] = list(set(filter(lambda x: x > 0 and data_size - x - 1 > 0,
                                             map(lambda x: type(params[k])(x),
                                                 np.random.normal(v, v * 2, 6)))))

        self.logger.warning(f'Grid Search: {params_grid}')

        cerebro.optstrategy(
            strategy,
            tid=self.tid,
            **params_grid
        )

    def run_and_save(self, cerebro: bt.Cerebro, num_split: Optional[int]):
        for analyzer, kwargs in self.analyzers:
            cerebro.addanalyzer(analyzer, _name=analyzer.__name__, **kwargs)

        cerebro.addobserver(bt.observers.BuySell, barplot=False, plot=False)
        cerebro.addobserver(bt.observers.Trades, plot=False)

        start = datetime.utcnow()
        thestrats = cerebro.run()
        processing_time = datetime.utcnow() - start
        rdb = RethinkDB()
        rcon = self.get_rdb_connection()
        
        for strategy in thestrats:
            analyzers = {}
            parameters = dict(filter(lambda x: x[0] not in ['tid', 'optim_engine'], strategy.params.__dict__.items()))
            indicators = {}
            for indicator in [indicator for indicator in strategy.getindicators() if isinstance(indicator, bt.Indicator)]:
                res = {}
                for line_name in indicator.plotlines._getkeys():
                    line = indicator._getline(line_name)
                    res[line_name] = {
                        'values': list(map(lambda x: None if np.isnan(x) else x
                                            ,line.get(size=len(line), ago=0))),
                        'color': 'Test',
                    }
                indicators[f'{indicator.__class__.__name__} {indicator._plotlabel()}'] = keys_to_strings(res)

            for analyzer, _ in self.analyzers:
                info = strategy.analyzers.getbyname(analyzer.__name__)

                if analyzer.__name__ == btanalyzers.PyFolio.__name__:
                    returns, positions, transactions, gross_lev = info.get_pf_items()
                    positions = utils.check_intraday(estimate='infer',
                                                     returns=returns,
                                                     positions=positions,
                                                     transactions=transactions)

                    # Mock performance stats
                    res = {
                        'total_return': 0.0,
                        'annual_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'volatility': 0.0
                    }

                    analyzers['pyfolio_sheets'] = self.get_pyfolio_sheets(positions, returns, transactions)
                else:
                    res = info.get_analysis()

                analysis = keys_to_strings(res)

                analyzers[analyzer.__name__] = analysis

            observers = {}
            for observer in strategy.observers:
                if observer.__class__.__name__ == bt.observers.BuySell.__name__:
                    timestamps = list(map(lambda x: bt.num2date(x).astimezone(pytz.UTC).timestamp() * 1000,
                                          strategy.data.datetime.array))

                    res = {
                        'buy': list(filter(lambda x: ~np.isnan(x[1]),
                                           zip(timestamps, observer.buy.get(size=len(strategy.data))))),
                        'sell': list(filter(lambda x: ~np.isnan(x[1]),
                                            zip(timestamps, observer.sell.get(size=len(strategy.data))))),
                    }
                    observers[observer.__class__.__name__] = keys_to_strings(res)

                if observer.__class__.__name__ == bt.observers.Trades.__name__:
                    pnl = list(filter(lambda x: ~np.isnan(x),
                                      map(lambda x: x[0] if ~np.isnan(x[0]) else x[1],
                                          zip(observer.pnlplus.get(size=len(strategy.data)),
                                              observer.pnlminus.get(size=len(strategy.data))))))
                    res = {'pnl': pnl}
                    observers[observer.__class__.__name__] = keys_to_strings(res)

            record = rdb.table(self.optimization_output.results).get(self.tid).run(rcon)

            start_date = bt.num2date(strategy.data.datetime.array[0]).astimezone(pytz.UTC).timestamp() * 1000
            end_date = bt.num2date(strategy.data.datetime.array[-1]).astimezone(pytz.UTC).timestamp() * 1000

            if not record:
                res = keys_to_strings(self.get_record(num_split=num_split,
                                                      processing_time=processing_time,
                                                      analyzers=analyzers,
                                                      observers=observers,
                                                      parameters=parameters,
                                                      indicators=indicators,
                                                      start_date=start_date,
                                                      end_date=end_date
                                                      ))

                db_res = (rdb.table(self.optimization_output.results)
                          .insert(res)
                          .run(rcon))

            else:

                res = keys_to_strings({
                    "optimizations": {
                        self.kind.value: rdb.row["optimizations"][self.kind.value].append(
                            self.get_test_record(num_split=num_split,
                                                 processing_time=processing_time,
                                                 analyzers=analyzers,
                                                 observers=observers,
                                                 parameters=parameters,
                                                 indicators=indicators,
                                                 start_date=start_date,
                                                 end_date=end_date
                                                 ))
                    }
                })
                db_res = (rdb.table(self.optimization_output.results)
                          .get(self.tid)
                          .update(res).run(rcon))
            self.logger.info(f"DB Results: {db_res}")

    def get_pyfolio_sheets(self, positions, returns, transactions):

        sheets_list = [
            ('returns_tear_sheet', partial(pf.create_returns_tear_sheet,
                                           returns=returns,
                                           positions=positions,
                                           transactions=transactions,
                                           return_fig=True)),
            ('interesting_times_tear_sheet', partial(pf.create_interesting_times_tear_sheet,
                                                     returns=returns)),
            ('position_tear_sheet', partial(pf.create_position_tear_sheet,
                                            returns=returns,
                                            positions=positions,
                                            transactions=transactions,
                                            return_fig=True)),
            ('txn_tear_sheet', partial(pf.create_txn_tear_sheet,
                                       returns=returns,
                                       positions=positions,
                                       transactions=transactions,
                                       estimate_intraday=False,
                                       return_fig=True))
        ]

        sheets = {}

        for name, fig_fn in sheets_list:

            fig = None

            try:
                fig = fig_fn()
            except Exception:
                self.logger.error(f'Caught exception on pyfolio plot generation ({name}).\n{traceback.format_exc()}')

            if fig:
                with BytesIO() as buff:
                    fig.savefig(buff)
                    sheets[name] = base64.b64encode(buff.getvalue()).decode('utf-8')

        return sheets

    @property
    def analyzers(self):
        # Analyzer
        analyzers_list = [
            (btanalyzers.PyFolio, {}),
            (btanalyzers.TradeAnalyzer, {}),
        ]
        return analyzers_list

    @property
    @abstractmethod
    def kind(self):
        raise NotImplementedError()

    def get_rdb_connection(self):
        if not self.rconn:
            self.rconn = RethinkDB().connect(**self.rethinkdb_config.__dict__)
        return self.rconn

    def save_progress_to_db(self, cur, total):
        batch = (10 ** (self._order_of_magnitude(total) - 1))
        if cur % batch == 0 or cur == total or cur <= 1:
            self.logger.info(f'Progress {self.cur_bar}/{self.total_bars}')
            rcon = self.get_rdb_connection()
            rdb = RethinkDB()

            record = rdb.table(self.optimization_output.progress).get(self.tid).run(rcon)

            if not record:
                (rdb.table(self.optimization_output.progress)
                 .insert(self.get_progress_record(cur=cur, total=total))
                 .run(rcon))
            else:
                (rdb.table(self.optimization_output.progress)
                 .get(self.tid)
                 .update({
                    "optimizations": {
                        self.kind.value: self.get_progress_inner_record(cur=cur, total=total)
                    }
                }).run(rcon))

    @staticmethod
    def _order_of_magnitude(number):
        return math.floor(math.log(number, 10))

    def get_record(self, num_split: int, processing_time: timedelta, analyzers: Dict[str, Dict],
                   observers, parameters, indicators, start_date, end_date):
        r = {
            'tid': self.tid,
            'test_name': self.test_name,
            'creation_time': datetime.utcnow().astimezone(pytz.UTC).timestamp() * 1000,
            'strategy': self.strategy.__name__,
            'provider': self.provider.value.upper(),
            'symbol': self.symbol,
            'timeframe': self.bin_size,
            'cash': self.cash,
            'commissions': self.commissions,
            'start_date': self.start_date.timestamp() * 1000,
            'end_date': self.end_date.timestamp() * 1000,
            'parameters': parameters,
            'indicators': indicators,
            'optimizations': {
                OptimizationType.BACKTESTING.value: [],
                OptimizationType.WALKFORWARD.value: []
            }
        }

        r['optimizations'][self.kind.value].append(
            self.get_test_record(analyzers, observers, parameters, indicators, num_split,
                                 processing_time, start_date, end_date)
        )

        return keys_to_strings(r)

    def get_test_record(self, analyzers, observers, parameters, indicators, num_split,
                        processing_time, start_date, end_date):
        return {
            'test_timestamp': self.test_timestamp.timestamp() * 1000,
            'num_split': num_split,
            'start_date': start_date,
            'end_date': end_date,
            'processing_time': processing_time.total_seconds(),
            'kind': self.kind.value,
            'analyzers': analyzers,
            'observers': observers,
            'parameters': parameters,
            'indicators': indicators,
            'cash': self.cash,
            'commissions': self.commissions
        }

    def get_progress_record(self, cur, total):
        r = {
            'tid': self.tid,
            'test_name': self.test_name,
            'optimizations': {}
        }

        r['optimizations'][self.kind.value] = self.get_progress_inner_record(cur, total)

        return r

    def get_progress_inner_record(self, cur, total):
        return {
            'kind': self.kind.value,
            'strategy': self.strategy.__name__,
            'symbol': self.symbol,
            'timeframe': self.bin_size,
            'current': cur,
            'total': total
        }


@singledispatch
def keys_to_strings(ob):
    if isinstance(ob, float) and (np.isnan(ob) or np.isinf(ob)):
        ob = None
    return ob


@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}


@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]
