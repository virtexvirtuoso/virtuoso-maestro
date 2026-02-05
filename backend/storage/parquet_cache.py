"""
Parquet Cache Layer - High-performance local file cache for OHLCV data.

Uses Polars for fast read/write with zstd compression.
Provides 10-100x faster repeated access than database queries.

Architecture:
- ParquetCache: Low-level cache operations (get/put/invalidate)
- CachedDataAdapter: High-level adapter with cache-through pattern
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import pandas as pd

from datasource.providers import DataSourceProviders


class ParquetCache:
    """
    Local Parquet file cache for OHLCV data with expiration.

    Uses Polars for efficient I/O with zstd compression.
    Files are named: {PROVIDER}_{symbol}_{bin_size}.parquet

    Features:
    - Time-based expiration (default: 1 day)
    - zstd compression for smaller files
    - Lazy loading with predicate pushdown for efficient date filtering
    """

    DEFAULT_MAX_AGE_SECONDS = 86400  # 1 day

    def __init__(
        self,
        cache_dir: str = './data/cache',
        max_age_seconds: int = None,
        logger: logging.Logger = None
    ):
        """
        Initialize Parquet cache.

        Args:
            cache_dir: Directory for cache files
            max_age_seconds: Cache expiration time in seconds (default: 1 day)
            logger: Optional logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_seconds or self.DEFAULT_MAX_AGE_SECONDS
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.logger.debug(
            f'ParquetCache initialized: dir={self.cache_dir}, '
            f'max_age={self.max_age_seconds}s'
        )

    def _get_cache_path(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> Path:
        """Generate cache file path for given parameters."""
        filename = f'{provider.value.upper()}_{symbol.lower()}_{bin_size}.parquet'
        return self.cache_dir / filename

    def is_cached(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> bool:
        """
        Check if valid (non-expired) cache entry exists.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size

        Returns:
            True if cache file exists and is not expired
        """
        cache_path = self._get_cache_path(provider, symbol, bin_size)

        if not cache_path.exists():
            return False

        # Check file age
        mtime = cache_path.stat().st_mtime
        age_seconds = time.time() - mtime

        if age_seconds > self.max_age_seconds:
            self.logger.debug(
                f'Cache expired: {cache_path.name} '
                f'(age={age_seconds:.0f}s > max={self.max_age_seconds}s)'
            )
            return False

        return True

    def get(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get data from cache with efficient date filtering.

        Uses Polars lazy scan with filter for predicate pushdown,
        avoiding loading entire file into memory.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            DataFrame with OHLCV data, or None if not cached
        """
        if not self.is_cached(provider, symbol, bin_size):
            return None

        cache_path = self._get_cache_path(provider, symbol, bin_size)
        load_start = time.perf_counter()

        try:
            # Use lazy scan for efficient filtering (predicate pushdown)
            lazy_df = pl.scan_parquet(cache_path)

            # Convert datetime to UTC timestamp for filtering
            # Polars uses microseconds for datetime
            start_ts = start_date.timestamp() * 1_000_000
            end_ts = end_date.timestamp() * 1_000_000

            # Apply filter and collect
            df = (
                lazy_df
                .filter(
                    (pl.col('timestamp') >= start_ts) &
                    (pl.col('timestamp') <= end_ts)
                )
                .collect()
            )

            # Convert to pandas with proper index
            pandas_df = df.to_pandas()

            # Convert timestamp from microseconds to datetime
            pandas_df['timestamp'] = pd.to_datetime(
                pandas_df['timestamp'], unit='us', utc=True
            )
            pandas_df = pandas_df.set_index('timestamp')

            load_time_ms = (time.perf_counter() - load_start) * 1000

            self.logger.info(
                f'Cache HIT: {cache_path.name} - '
                f'{len(pandas_df)} rows in {load_time_ms:.1f}ms'
            )

            return pandas_df

        except Exception as e:
            self.logger.warning(f'Cache read error: {cache_path.name} - {e}')
            return None

    def put(
        self,
        df: pd.DataFrame,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> Tuple[Path, int]:
        """
        Store data in cache with zstd compression.

        Args:
            df: OHLCV DataFrame (with timestamp index)
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size

        Returns:
            Tuple of (cache file path, compressed file size in bytes)
        """
        cache_path = self._get_cache_path(provider, symbol, bin_size)
        write_start = time.perf_counter()

        # Prepare DataFrame for Polars
        df_to_save = df.copy()

        # Ensure timestamp is a column (not index) for Parquet
        if df_to_save.index.name == 'timestamp':
            df_to_save = df_to_save.reset_index()

        # Convert timestamp to microseconds for efficient storage
        if 'timestamp' in df_to_save.columns:
            # Convert to microseconds since epoch
            df_to_save['timestamp'] = (
                pd.to_datetime(df_to_save['timestamp'], utc=True)
                .astype('int64') // 1000  # ns to us
            )

        # Convert to Polars and write with zstd compression
        pl_df = pl.from_pandas(df_to_save)
        pl_df.write_parquet(
            cache_path,
            compression='zstd',
            compression_level=3  # Balance between speed and ratio
        )

        file_size = cache_path.stat().st_size
        write_time_ms = (time.perf_counter() - write_start) * 1000

        self.logger.info(
            f'Cache PUT: {cache_path.name} - '
            f'{len(df)} rows, {file_size / 1024:.1f}KB in {write_time_ms:.1f}ms'
        )

        return cache_path, file_size

    def invalidate(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> bool:
        """
        Delete a specific cache entry.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size

        Returns:
            True if entry was deleted, False if it didn't exist
        """
        cache_path = self._get_cache_path(provider, symbol, bin_size)

        if cache_path.exists():
            cache_path.unlink()
            self.logger.info(f'Cache invalidated: {cache_path.name}')
            return True

        return False

    def invalidate_all(self) -> int:
        """
        Delete all cache entries.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob('*.parquet'):
            cache_file.unlink()
            count += 1

        if count > 0:
            self.logger.info(f'Cache cleared: {count} files deleted')

        return count

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        cache_files = list(self.cache_dir.glob('*.parquet'))
        total_size = sum(f.stat().st_size for f in cache_files)

        now = time.time()
        expired_count = sum(
            1 for f in cache_files
            if (now - f.stat().st_mtime) > self.max_age_seconds
        )

        return {
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'expired_count': expired_count,
            'max_age_seconds': self.max_age_seconds,
        }


class CachedDataAdapter:
    """
    Cache-through wrapper for any DataAdapter.

    Checks cache first, falls back to primary adapter on miss,
    and populates cache from primary adapter results.

    Example:
        primary = RethinkDBAdapter(config=rethinkdb_config)
        cache = ParquetCache(cache_dir='./data/cache')
        adapter = CachedDataAdapter(primary, cache)

        # First call: cache miss, loads from DB, caches result
        df = adapter.load_dataframe(...)

        # Second call: cache hit, loads from Parquet (10-100x faster)
        df = adapter.load_dataframe(...)
    """

    def __init__(
        self,
        primary_adapter,  # DataAdapter
        cache: ParquetCache,
        logger: logging.Logger = None
    ):
        """
        Initialize cached adapter.

        Args:
            primary_adapter: Primary data source adapter
            cache: ParquetCache instance for caching
            logger: Optional logger instance
        """
        self.primary = primary_adapter
        self.cache = cache
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Stats tracking
        self._hits = 0
        self._misses = 0

    def load_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load OHLCV data with cache-through pattern.

        1. Check cache for valid entry
        2. On hit: return cached data
        3. On miss: load from primary, cache result, return data

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first
        cached_df = self.cache.get(provider, symbol, bin_size, start_date, end_date)

        if cached_df is not None:
            self._hits += 1
            self.logger.debug(f'Cache hit for {provider.value}/{symbol}/{bin_size}')
            return cached_df

        # Cache miss - load from primary adapter
        self._misses += 1
        self.logger.debug(f'Cache miss for {provider.value}/{symbol}/{bin_size}')

        load_start = time.perf_counter()
        df = self.primary.load_dataframe(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date
        )
        load_time_ms = (time.perf_counter() - load_start) * 1000

        self.logger.info(
            f'Primary adapter load: {len(df)} rows in {load_time_ms:.1f}ms'
        )

        # Cache the result (full dataset for future queries)
        # Note: We cache the full result, not just the filtered range
        # This allows future queries with different ranges to use the cache
        if len(df) > 0:
            self.cache.put(df, provider, symbol, bin_size)

        return df

    def is_available(self) -> bool:
        """Check if primary adapter is available."""
        return self.primary.is_available()

    def has_data(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """
        Check if data exists (in cache or primary).
        """
        # Check cache first
        if self.cache.is_cached(provider, symbol, bin_size):
            return True

        # Fall back to primary
        return self.primary.has_data(
            provider, symbol, bin_size, start_date, end_date
        )

    def invalidate_cache(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str
    ) -> bool:
        """Invalidate cache entry for given parameters."""
        return self.cache.invalidate(provider, symbol, bin_size)

    def get_stats(self) -> dict:
        """Get cache and adapter statistics."""
        cache_stats = self.cache.get_stats()
        cache_stats['cache_hits'] = self._hits
        cache_stats['cache_misses'] = self._misses
        cache_stats['hit_rate'] = (
            self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0 else 0.0
        )
        return cache_stats
