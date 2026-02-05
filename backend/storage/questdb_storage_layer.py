"""QuestDB storage layer with connection pooling for high-performance OHLCV storage."""

from __future__ import annotations
import logging
import socket
from contextlib import contextmanager
from typing import List, Dict, Optional, Generator
from datetime import datetime

import pandas as pd
import psycopg
from psycopg_pool import ConnectionPool

from datasource.providers import DataSourceProviders
from storage.storage_layer import StorageLayer
from schema.bucketed_trade_data import BucketedTradeData
from logger.logger_builder import LoggerBuilder


class QuestDBStorageLayer(StorageLayer):
    """
    QuestDB storage layer using PostgreSQL wire protocol for queries
    and InfluxDB Line Protocol (ILP) for high-speed ingestion.

    Connection pooling via psycopg_pool.ConnectionPool.
    """

    __METADATA_TABLE__ = 'trade_metadata'

    def __init__(
        self,
        host: str = 'localhost',
        pg_port: int = 8812,
        ilp_port: int = 9009,
        user: str = 'admin',
        password: str = 'quest',
        database: str = 'qdb',
        pool_min: int = 2,
        pool_max: int = 10,
        logger: logging.Logger = None
    ):
        super().__init__(logger=logger)
        self.host = host
        self.pg_port = pg_port
        self.ilp_port = ilp_port
        self.user = user
        self.password = password
        self.database = database
        self.pool_min = pool_min
        self.pool_max = pool_max

        # Build connection string
        self._conninfo = f"host={host} port={pg_port} user={user} password={password} dbname={database}"

        # Initialize connection pool
        self._pool: Optional[ConnectionPool] = None
        self._init_pool()

        # Ensure metadata table exists
        self._create_metadata_table()

    def _init_pool(self) -> None:
        """Initialize the threaded connection pool."""
        try:
            self._pool = ConnectionPool(
                conninfo=self._conninfo,
                min_size=self.pool_min,
                max_size=self.pool_max,
                open=True
            )
            self.logger.info(f'QuestDB connection pool initialized (min={self.pool_min}, max={self.pool_max})')
        except psycopg.Error as e:
            self.logger.error(f'Failed to initialize QuestDB connection pool: {e}')
            raise

    @contextmanager
    def get_connection(self) -> Generator:
        """
        Context manager for getting a connection from the pool.
        Automatically returns connection to pool when done.

        Usage:
            with storage.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM table")
        """
        with self._pool.connection() as conn:
            yield conn

    def spawn(self) -> QuestDBStorageLayer:
        """Create a new instance with same configuration."""
        return QuestDBStorageLayer(
            host=self.host,
            pg_port=self.pg_port,
            ilp_port=self.ilp_port,
            user=self.user,
            password=self.password,
            database=self.database,
            pool_min=self.pool_min,
            pool_max=self.pool_max
        )

    def _create_metadata_table(self) -> None:
        """Create metadata table if it doesn't exist."""
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.__METADATA_TABLE__} (
                table_name SYMBOL,
                provider SYMBOL,
                symbol SYMBOL,
                bin_size SYMBOL,
                start LONG,
                updated_at TIMESTAMP
            ) TIMESTAMP(updated_at);
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                self.logger.info(f'Metadata table {self.__METADATA_TABLE__} ready')

    def create_trade_table(self, table_name: str) -> None:
        """
        Create a trade table with TIMESTAMP partitioned by DAY.

        Schema matches standard OHLCV format:
        - timestamp: Primary key, partitioned by day
        - open, high, low, close: Price data
        - volume: Trade volume
        """
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                self.logger.info(f'Trade table {table_name} created with DAY partition')

    def _get_table_name(self, provider: DataSourceProviders, symbol: str, bin_size: str) -> str:
        """Generate table name from provider, symbol, and bin size."""
        return f'trade_{provider.value}_{symbol.lower()}_{bin_size}'.replace('-', '_')

    def add(self, provider: DataSourceProviders, symbol: str, bin_size: str) -> None:
        """Add a new trade data table and register in metadata."""
        table_name = self._get_table_name(provider=provider, symbol=symbol, bin_size=bin_size)
        self.create_trade_table(table_name)
        self._update_metadata(
            table_name=table_name,
            provider=provider,
            symbol=symbol,
            bin_size=bin_size
        )

    def _update_metadata(
        self,
        table_name: str,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> None:
        """Insert or update metadata entry for a table."""
        now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        query = f"""
            INSERT INTO {self.__METADATA_TABLE__}
            (table_name, provider, symbol, bin_size, start, updated_at)
            VALUES ('{table_name}', '{provider.value}', '{symbol}', '{bin_size}', 0, '{now}');
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                self.logger.info(f'Metadata updated for {table_name}')

    def bucketed_trade_data_list(self, provider: DataSourceProviders) -> List[BucketedTradeData]:
        """Get all bucketed trade data for a provider."""
        query = f"""
            SELECT symbol, bin_size, start
            FROM {self.__METADATA_TABLE__}
            WHERE provider = '{provider.value}'
            LATEST ON updated_at PARTITION BY table_name;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                for row in rows:
                    yield BucketedTradeData(
                        symbol=row[0],
                        bin_size=row[1],
                        start=row[2]
                    )

    def save(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        offset: int,
        results: List[Dict]
    ) -> BucketedTradeData:
        """Save OHLCV records using ILP for speed."""
        table_name = self._get_table_name(provider=provider, symbol=symbol, bin_size=bin_size)

        # Convert to DataFrame and ingest via ILP
        if results:
            df = pd.DataFrame(results)
            self.ingest_dataframe(table_name, df)

        # Update metadata with new offset
        now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        query = f"""
            INSERT INTO {self.__METADATA_TABLE__}
            (table_name, provider, symbol, bin_size, start, updated_at)
            VALUES ('{table_name}', '{provider.value}', '{symbol}', '{bin_size}', {offset}, '{now}');
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)

        self.logger.info(f'Saved {len(results)} records to {table_name}')
        return BucketedTradeData(symbol=symbol, bin_size=bin_size, start=offset)

    def ingest_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        """
        Ingest a DataFrame using InfluxDB Line Protocol (ILP) for maximum speed.

        ILP format: table_name field1=value1,field2=value2 timestamp_ns

        Args:
            table_name: Target table name
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            Number of rows ingested
        """
        if df.empty:
            return 0

        # Build ILP messages
        lines = []
        for _, row in df.iterrows():
            # Convert timestamp to nanoseconds
            ts = row.get('timestamp')
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            elif isinstance(ts, (int, float)):
                # Assume milliseconds
                ts = pd.to_datetime(ts, unit='ms')

            ts_ns = int(ts.timestamp() * 1e9)

            # Build line: table_name open=1.0,high=2.0,low=0.5,close=1.5,volume=100.0 timestamp_ns
            fields = []
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in row:
                    fields.append(f'{col}={float(row[col])}')

            if fields:
                line = f'{table_name} {",".join(fields)} {ts_ns}'
                lines.append(line)

        # Send via TCP socket to ILP port
        if lines:
            message = '\n'.join(lines) + '\n'
            self._send_ilp(message)

        self.logger.debug(f'Ingested {len(lines)} rows via ILP')
        return len(lines)

    def _send_ilp(self, message: str) -> None:
        """Send data to QuestDB via InfluxDB Line Protocol over TCP."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.ilp_port))
                sock.sendall(message.encode('utf-8'))
        except socket.error as e:
            self.logger.error(f'ILP send failed: {e}')
            raise

    def query_ohlcv(
        self,
        table_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query OHLCV data with optional time range filtering.

        Args:
            table_name: Table to query
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            limit: Maximum number of rows to return

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        conditions = []
        if start_time:
            ts = start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            conditions.append(f"timestamp >= '{ts}'")
        if end_time:
            ts = end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            conditions.append(f"timestamp <= '{ts}'")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {table_name}
            {where_clause}
            ORDER BY timestamp
            {limit_clause};
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=columns)

        self.logger.debug(f'Queried {len(df)} rows from {table_name}')
        return df

    def size(self) -> int:
        """Return number of registered tables in metadata."""
        query = f"""
            SELECT COUNT(DISTINCT table_name) FROM {self.__METADATA_TABLE__};
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()
                return result[0] if result else 0

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in QuestDB."""
        query = "SELECT table_name FROM tables() WHERE table_name = %s;"
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (table_name,))
                return cur.fetchone() is not None

    def drop_table(self, table_name: str) -> None:
        """Drop a table if it exists."""
        query = f"DROP TABLE IF EXISTS {table_name};"
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                self.logger.info(f'Dropped table {table_name}')

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.close()
            self.logger.info('QuestDB connection pool closed')

    def __enter__(self) -> QuestDBStorageLayer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
