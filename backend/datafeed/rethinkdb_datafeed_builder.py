"""RethinkDB Data Feed Builder - OHLCV data loading for Backtrader."""

from logging import Logger
from typing import Any

import pandas as pd
import backtrader as bt
from rethinkdb import RethinkDB
import backtrader.feeds as btfeeds
from datetime import datetime

from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder


class RethinkDBDataFeedBuilder:
    """
    Builds Backtrader-compatible data feeds from RethinkDB.

    Supports two initialization modes:
    1. Direct parameters: host, port, db
    2. Config object: rethinkdb_config with host, port, db attributes

    Provides methods for:
    - read_data_frame: Load raw DataFrame
    - build_dataframe: Alias for read_data_frame (V2 compatibility)
    - to_datafeed: Convert DataFrame to Backtrader feed
    - build: Load and convert in one step
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: str = None,
        rethinkdb_config: Any = None,
        logger: Logger = None
    ):
        # Support both direct params and config object
        if rethinkdb_config is not None:
            self._host = rethinkdb_config.host
            self._port = rethinkdb_config.port
            self._db = rethinkdb_config.db
        else:
            self._host = host or '127.0.0.1'
            self._port = port or 28015
            self._db = db or 'filos-dev'

        self._connection = None
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()

    @property
    def connection(self):
        """Lazy connection initialization."""
        if self._connection is None:
            self._connection = RethinkDB().connect(
                host=self._host,
                port=self._port,
                db=self._db
            )
        return self._connection

    def read_data_frame(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load OHLCV data from RethinkDB as a pandas DataFrame.

        Args:
            provider: Data source provider (e.g., BINANCE, BITMEX)
            symbol: Trading pair symbol (e.g., 'ethbtc')
            bin_size: Candle size (e.g., '1d', '1h')
            start_date: Start of data range (inclusive)
            end_date: End of data range (inclusive)

        Returns:
            DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        """
        r = RethinkDB()
        table_name = f'trade_{provider.value.upper()}_{symbol.lower()}_{bin_size}'

        records = (r.table(table_name)
                   .order_by(index='timestamp')
                   .pluck('timestamp', 'open', 'high', 'low', 'close', 'volume')
                   .run(self.connection))

        dataframe = pd.DataFrame(records).set_index('timestamp')
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)

        full_size = len(dataframe)
        dataframe = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)]

        self.logger.info(
            f'Filtering data from {start_date} to {end_date}. '
            f'Before Filtering: {full_size}, after {len(dataframe)}'
        )

        return dataframe

    def build_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Alias for read_data_frame for V2 API compatibility.

        This method provides a cleaner name for the V2 engine integration.
        """
        return self.read_data_frame(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date
        )

    def to_datafeed(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        df: pd.DataFrame
    ) -> btfeeds.PandasData:
        """Convert a DataFrame to a Backtrader-compatible data feed."""
        return bt.feeds.PandasData(
            name=f'{provider.value}_{symbol}_{bin_size}',
            dataname=df,
            datetime=None
        )

    def build(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> btfeeds.PandasData:
        """
        Load data and build Backtrader feed in one step.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size
            start_date: Start of data range
            end_date: End of data range

        Returns:
            Backtrader PandasData feed ready for Cerebro
        """
        data_frame = self.read_data_frame(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date
        )
        return bt.feeds.PandasData(
            name=f'{provider.value}_{symbol}_{bin_size}',
            dataname=data_frame,
            datetime=None
        )

    def close(self):
        """Close the database connection."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
