from logging import Logger

import pandas as pd
import backtrader as bt
from rethinkdb import RethinkDB
import backtrader.feeds as btfeeds
from datetime import datetime

from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder


class RethinkDBDataFeedBuilder:

    def __init__(self, host: str, port: int, db: str, logger: Logger=None):
        self.connection = RethinkDB().connect(host=host, port=port, db=db)
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()

    def read_data_frame(self, provider: DataSourceProviders, symbol: str,
                        bin_size: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        r = RethinkDB()
        records = (r.table(f'trade_{provider.value.upper()}_{symbol.lower()}_{bin_size}')
                   .order_by(index='timestamp')
                   .pluck('timestamp', 'open', 'high', 'low', 'close', 'volume')
                   .run(self.connection))

        dataframe = pd.DataFrame(records).set_index('timestamp')
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)
        full_size = len(dataframe)
        dataframe = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)]
        self.logger.info(f'Filtering data from {start_date} to {end_date}. '
                         f'Before Filtering: {full_size}, after {len(dataframe)}')
        return dataframe

    def to_datafeed(self, provider: DataSourceProviders, symbol: str, bin_size: str, df: pd.DataFrame) -> btfeeds.PandasData:
        return bt.feeds.PandasData(name=f'{provider.value}_{symbol}_{bin_size}', dataname=df, datetime=None)

    def build(self, provider: DataSourceProviders, symbol: str, bin_size: str,
              start_date: datetime, end_date: datetime) -> btfeeds.PandasData:
        data_frame = self.read_data_frame(provider=provider,
                                          symbol=symbol, bin_size=bin_size,
                                          start_date=start_date, end_date=end_date)
        return bt.feeds.PandasData(name=f'{provider.value}_{symbol}_{bin_size}',
                                   dataname=data_frame,
                                   datetime=None)
