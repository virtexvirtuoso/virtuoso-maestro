from datetime import datetime
from enum import Enum
from logging import Logger
import hashlib
from typing import Dict

import pytz
from rethinkdb import RethinkDB
import strategy
from config.config_reader import RethinkDbConfig, OptimizationOutputConfig, WalkForwardConfig
from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder


class OptimizationType(Enum):
    BACKTESTING = 'BACKTESTING'
    WALKFORWARD = 'WALKFORWARD'
    BOTH = 'BOTH'


class Optimizer:

    def __init__(self,
                 rethinkdb_config: RethinkDbConfig,
                 optimization_output: OptimizationOutputConfig,
                 walkforward_config: WalkForwardConfig,
                 logger: Logger = None):
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        self.walkforward_config = walkforward_config
        self.connection = RethinkDB().connect(**self.rethinkdb_config.__dict__)
        self._create_optim_tables()

    def run(self, tid: str, test_name: str, provider: DataSourceProviders, symbol: str, bin_size: str, strategy_name: str,
            strategy_params: Dict[str, float], cash: float, commissions: float,
            start_date, end_date, engine):
        engine = engine(tid=tid,
                        test_name=test_name,
                        test_timestamp=datetime.now().astimezone(pytz.UTC),
                        provider=provider,
                        symbol=symbol,
                        bin_size=bin_size,
                        strategy=strategy.__STRATEGY_CATALOG__[strategy_name],
                        strategy_params=strategy_params,
                        cash=cash,
                        commissions=commissions,
                        start_date=start_date,
                        end_date=end_date,
                        rethinkdb_config=self.rethinkdb_config,
                        optimization_output=self.optimization_output,
                        walkforward_config=self.walkforward_config)
        engine.start()
        return tid

    def _create_optim_tables(self):
        r = RethinkDB()
        table_results = self.optimization_output.results
        progress = self.optimization_output.progress

        current_tables = set(r.table_list().run(self.connection))

        for t in [table_results, progress]:
            if t in current_tables:
                self.logger.warning(f'Optimization table {t} already exists, no need to create it')
            else:
                self.logger.warning(f'Optimization table {t} does not exist, creating it')
                r.table_create(t, primary_key='tid').run(self.connection)
