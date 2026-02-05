"""
Data Adapter Layer - Abstraction for switching between data sources.

Enables switching between RethinkDB (V1), Parquet (cache), and QuestDB (V2)
without code changes in the consuming engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import pandas as pd

from datasource.providers import DataSourceProviders


@dataclass
class DataRequest:
    """Parameters for loading data from any adapter."""
    provider: DataSourceProviders
    symbol: str
    bin_size: str
    start_date: datetime
    end_date: datetime


class DataAdapter(ABC):
    """
    Abstract base class for data source adapters.

    All data loading in Maestro should go through this interface,
    enabling transparent switching between data sources.
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load OHLCV data as a pandas DataFrame.

        Args:
            provider: Data source provider (e.g., BINANCE, BITMEX)
            symbol: Trading pair symbol (e.g., 'ethbtc', 'xbtusd')
            bin_size: Candle size (e.g., '1m', '5m', '1h', '1d')
            start_date: Start of data range (inclusive, UTC)
            end_date: End of data range (inclusive, UTC)

        Returns:
            DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and a DatetimeIndex named 'timestamp'.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this data source is available and accessible.

        Returns:
            True if data source can be connected to, False otherwise.
        """
        pass

    def has_data(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        Check if data exists for the given parameters.

        Default implementation tries to load data and checks if non-empty.
        Subclasses may override for more efficient checks.
        """
        try:
            df = self.load_dataframe(provider, symbol, bin_size, start_date, end_date)
            return df is not None and len(df) > 0
        except Exception:
            return False


class RethinkDBAdapter(DataAdapter):
    """
    RethinkDB data adapter wrapping the existing RethinkDBDataFeedBuilder.

    This is the primary data source for V1 and serves as fallback for V2.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 28015,
        db: str = 'filos-dev',
        rethinkdb_config: Any = None,
        logger: logging.Logger = None
    ):
        super().__init__(logger)

        # Support both direct params and config object
        if rethinkdb_config is not None:
            self.host = rethinkdb_config.host
            self.port = rethinkdb_config.port
            self.db = rethinkdb_config.db
        else:
            self.host = host
            self.port = port
            self.db = db

        self._connection = None

    def _get_connection(self):
        """Lazy connection initialization."""
        if self._connection is None:
            from rethinkdb import RethinkDB
            r = RethinkDB()
            self._connection = r.connect(host=self.host, port=self.port, db=self.db)
        return self._connection

    def load_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load OHLCV data from RethinkDB."""
        from rethinkdb import RethinkDB
        r = RethinkDB()

        conn = self._get_connection()
        table_name = f'trade_{provider.value.upper()}_{symbol.lower()}_{bin_size}'

        self.logger.debug(f'Loading data from {table_name}')

        records = (r.table(table_name)
                   .order_by(index='timestamp')
                   .pluck('timestamp', 'open', 'high', 'low', 'close', 'volume')
                   .run(conn))

        dataframe = pd.DataFrame(records).set_index('timestamp')
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)

        full_size = len(dataframe)
        dataframe = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)]

        self.logger.info(
            f'Loaded {len(dataframe)} records from {table_name} '
            f'({start_date} to {end_date}, filtered from {full_size})'
        )

        return dataframe

    def is_available(self) -> bool:
        """Check if RethinkDB is accessible."""
        try:
            self._get_connection()
            return True
        except Exception as e:
            self.logger.warning(f'RethinkDB not available: {e}')
            return False

    def close(self):
        """Close the database connection."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None


class ParquetAdapter(DataAdapter):
    """
    Parquet file adapter for local data caching.

    Stores data in efficient columnar format for fast reads.
    Useful for:
    - Caching frequently accessed data
    - Offline development without database
    - Data archival and backup
    """

    def __init__(
        self,
        data_dir: str = './data/parquet',
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> Path:
        """Generate file path for given data parameters."""
        filename = f'{provider.value.upper()}_{symbol.lower()}_{bin_size}.parquet'
        return self.data_dir / filename

    def load_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load OHLCV data from Parquet file."""
        file_path = self._get_file_path(provider, symbol, bin_size)

        if not file_path.exists():
            raise FileNotFoundError(
                f'Parquet file not found: {file_path}. '
                'Use save_dataframe() to cache data first.'
            )

        self.logger.debug(f'Loading data from {file_path}')

        # Read parquet with optional date filtering via PyArrow
        try:
            import pyarrow.parquet as pq

            # Read with predicate pushdown if available
            df = pq.read_table(
                file_path,
                filters=[
                    ('timestamp', '>=', start_date),
                    ('timestamp', '<=', end_date)
                ] if hasattr(pq, 'read_table') else None
            ).to_pandas()
        except ImportError:
            # Fallback to pandas parquet reader
            df = pd.read_parquet(file_path)

        # Ensure proper index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        df.index = pd.to_datetime(df.index, utc=True)

        # Filter by date range
        full_size = len(df)
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        self.logger.info(
            f'Loaded {len(df)} records from {file_path} '
            f'({start_date} to {end_date}, filtered from {full_size})'
        )

        return df

    def save_dataframe(
        self,
        df: pd.DataFrame,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> Path:
        """
        Save DataFrame to Parquet file.

        Args:
            df: OHLCV DataFrame to save
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size

        Returns:
            Path to saved file
        """
        file_path = self._get_file_path(provider, symbol, bin_size)

        # Ensure timestamp is a column for better parquet support
        df_to_save = df.copy()
        if df_to_save.index.name == 'timestamp':
            df_to_save = df_to_save.reset_index()

        df_to_save.to_parquet(file_path, index=False)
        self.logger.info(f'Saved {len(df_to_save)} records to {file_path}')

        return file_path

    def is_available(self) -> bool:
        """Check if Parquet data directory is accessible."""
        try:
            return self.data_dir.exists() and self.data_dir.is_dir()
        except Exception as e:
            self.logger.warning(f'Parquet data directory not available: {e}')
            return False

    def has_data(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """Check if Parquet file exists for given parameters."""
        file_path = self._get_file_path(provider, symbol, bin_size)
        return file_path.exists()


class QuestDBAdapter(DataAdapter):
    """
    QuestDB adapter stub for Phase 2 implementation.

    QuestDB is a high-performance time-series database that will be used
    in Phase 2 for faster data storage and retrieval.

    TODO: Implement in Phase 2
    - Connect to QuestDB via PostgreSQL wire protocol
    - Use ILP for fast writes
    - Leverage time-series specific queries
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 8812,  # QuestDB PostgreSQL port
        db: str = 'qdb',
        logger: logging.Logger = None
    ):
        super().__init__(logger)
        self.host = host
        self.port = port
        self.db = db
        self._connection = None

        self.logger.warning(
            'QuestDBAdapter is a Phase 2 stub. '
            'Full implementation will be added in Phase 2.'
        )

    def load_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load OHLCV data from QuestDB (stub)."""
        raise NotImplementedError(
            'QuestDBAdapter.load_dataframe() is a Phase 2 stub. '
            'Use RethinkDBAdapter or ParquetAdapter for now.'
        )

    def is_available(self) -> bool:
        """Check if QuestDB is accessible (stub)."""
        # Always return False until Phase 2 implementation
        return False


def get_adapter(
    config: Any = None,
    adapter_type: str = 'rethinkdb',
    **kwargs
) -> DataAdapter:
    """
    Factory function to create the appropriate data adapter.

    Args:
        config: Configuration object (e.g., RethinkDbConfig)
        adapter_type: Type of adapter ('rethinkdb', 'parquet', 'questdb')
        **kwargs: Additional arguments passed to adapter constructor

    Returns:
        Configured DataAdapter instance

    Examples:
        # Using config object
        adapter = get_adapter(config=rethinkdb_config, adapter_type='rethinkdb')

        # Using direct parameters
        adapter = get_adapter(adapter_type='parquet', data_dir='./cache')

        # Auto-select based on availability (with fallback)
        adapter = get_adapter(adapter_type='auto', config=rethinkdb_config)
    """
    adapter_type = adapter_type.lower()

    if adapter_type == 'rethinkdb':
        if config is not None:
            return RethinkDBAdapter(rethinkdb_config=config, **kwargs)
        return RethinkDBAdapter(**kwargs)

    elif adapter_type == 'parquet':
        return ParquetAdapter(**kwargs)

    elif adapter_type == 'questdb':
        return QuestDBAdapter(**kwargs)

    elif adapter_type == 'auto':
        # Try adapters in order of preference
        # Phase 2: QuestDB > Parquet > RethinkDB
        # Currently: RethinkDB > Parquet

        adapters_to_try = [
            ('rethinkdb', lambda: RethinkDBAdapter(rethinkdb_config=config, **kwargs)),
            ('parquet', lambda: ParquetAdapter(**kwargs)),
        ]

        for name, factory in adapters_to_try:
            try:
                adapter = factory()
                if adapter.is_available():
                    logging.getLogger(__name__).info(f'Auto-selected {name} adapter')
                    return adapter
            except Exception:
                continue

        # Fallback to RethinkDB even if not available (will fail on use)
        return RethinkDBAdapter(rethinkdb_config=config, **kwargs)

    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Valid types: 'rethinkdb', 'parquet', 'questdb', 'auto'"
        )


# Convenience alias for backward compatibility with existing code
def create_data_adapter(rethinkdb_config: Any = None, **kwargs) -> DataAdapter:
    """Create a data adapter using the default (RethinkDB) source."""
    return get_adapter(config=rethinkdb_config, adapter_type='rethinkdb', **kwargs)
