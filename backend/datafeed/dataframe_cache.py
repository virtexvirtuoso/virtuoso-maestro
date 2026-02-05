"""
DataFrame Cache - In-memory cache for walk-forward optimization.

Loads data once and slices in-memory for each split, eliminating repeated DB queries.
Provides O(1) slicing operations via iloc (index) and loc (datetime) access.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .data_adapter import DataAdapter
from datasource.providers import DataSourceProviders


@dataclass
class CacheKey:
    """Unique key for cached DataFrames."""
    provider: DataSourceProviders
    symbol: str
    bin_size: str
    start_date: datetime
    end_date: datetime

    def __hash__(self):
        return hash((
            self.provider.value,
            self.symbol.lower(),
            self.bin_size,
            self.start_date.isoformat() if self.start_date else None,
            self.end_date.isoformat() if self.end_date else None,
        ))

    def __eq__(self, other):
        if not isinstance(other, CacheKey):
            return False
        return (
            self.provider == other.provider
            and self.symbol.lower() == other.symbol.lower()
            and self.bin_size == other.bin_size
            and self.start_date == other.start_date
            and self.end_date == other.end_date
        )


class DataFrameCache:
    """
    In-memory cache for OHLCV DataFrames.

    Loads data once from the data adapter and provides fast in-memory slicing
    for walk-forward optimization splits. Eliminates repeated database queries.

    Example usage:
        adapter = RethinkDBAdapter(rethinkdb_config=config)
        cache = DataFrameCache(data_adapter=adapter)

        # First call loads from DB
        df = cache.get_dataframe(provider, symbol, bin_size, start, end)

        # Subsequent calls return cached data
        df2 = cache.get_dataframe(provider, symbol, bin_size, start, end)

        # Fast slicing by index (for TimeSeriesSplit indices)
        train_df = cache.slice_by_index(df, train_indices)
        test_df = cache.slice_by_index(df, test_indices)

        # Or slice by datetime range
        subset = cache.slice_by_date(df, start_date, end_date)

        # Check memory usage
        print(f"Cache using {cache.memory_usage() / 1e6:.2f} MB")

        # Clear cache to free memory
        cache.clear()
    """

    def __init__(
        self,
        data_adapter: DataAdapter,
        logger: logging.Logger = None,
    ):
        """
        Initialize the DataFrame cache.

        Args:
            data_adapter: DataAdapter instance for loading data from source
            logger: Logger instance
        """
        self.data_adapter = data_adapter
        self.logger = logger or logging.getLogger(__name__)

        # Internal cache: Dict[CacheKey, pd.DataFrame]
        self._cache: Dict[CacheKey, pd.DataFrame] = {}

        # Statistics for debugging/monitoring
        self._stats = {
            'hits': 0,
            'misses': 0,
            'loads': 0,
        }

    def get_dataframe(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get a DataFrame, loading from adapter if not cached.

        First call loads data from the data adapter (DB query).
        Subsequent calls with same parameters return the cached DataFrame.

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
        key = CacheKey(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date,
        )

        if key in self._cache:
            self._stats['hits'] += 1
            self.logger.debug(
                f"Cache HIT for {provider.value}:{symbol}:{bin_size} "
                f"(hits={self._stats['hits']}, misses={self._stats['misses']})"
            )
            return self._cache[key]

        # Cache miss - load from adapter
        self._stats['misses'] += 1
        self._stats['loads'] += 1

        self.logger.info(
            f"Cache MISS for {provider.value}:{symbol}:{bin_size} - loading from adapter"
        )

        df = self.data_adapter.load_dataframe(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date,
        )

        # Store in cache
        self._cache[key] = df

        self.logger.info(
            f"Cached {len(df)} rows for {provider.value}:{symbol}:{bin_size} "
            f"({df.memory_usage(deep=True).sum() / 1e6:.2f} MB)"
        )

        return df

    def slice_by_index(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Slice DataFrame by integer indices using iloc for O(1) access.

        This is the preferred method for walk-forward splits where
        TimeSeriesSplit provides integer indices.

        Args:
            df: Source DataFrame to slice
            indices: Array of integer indices (from TimeSeriesSplit)

        Returns:
            Sliced DataFrame (view into original data when possible)
        """
        # Use iloc for O(1) index-based access
        # Note: .iloc with array returns a copy, but the slicing itself is O(1)
        return df.iloc[indices].copy()

    def slice_by_date(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Slice DataFrame by datetime range using loc.

        Useful for custom date-based slicing when indices aren't available.

        Args:
            df: Source DataFrame with DatetimeIndex
            start_date: Start of slice range (inclusive)
            end_date: End of slice range (inclusive)

        Returns:
            Sliced DataFrame
        """
        # Ensure datetime is timezone-aware if the index is
        if df.index.tz is not None:
            if start_date.tzinfo is None:
                import pytz
                start_date = pytz.UTC.localize(start_date)
            if end_date.tzinfo is None:
                import pytz
                end_date = pytz.UTC.localize(end_date)

        # Use boolean indexing with loc for datetime slicing
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask].copy()

    def clear(self) -> int:
        """
        Clear all cached DataFrames to free memory.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()

        self.logger.info(f"Cleared {count} cached DataFrames")

        # Reset stats
        self._stats = {'hits': 0, 'misses': 0, 'loads': 0}

        return count

    def memory_usage(self) -> int:
        """
        Calculate total memory usage of cached DataFrames.

        Returns:
            Total bytes used by all cached DataFrames
        """
        total_bytes = 0
        for df in self._cache.values():
            total_bytes += df.memory_usage(deep=True).sum()
        return int(total_bytes)

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with 'hits', 'misses', 'loads', 'entries', 'memory_bytes'
        """
        return {
            **self._stats,
            'entries': len(self._cache),
            'memory_bytes': self.memory_usage(),
        }

    def has_cached(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime,
    ) -> bool:
        """
        Check if data is already cached.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size
            start_date: Start of data range
            end_date: End of data range

        Returns:
            True if data is in cache, False otherwise
        """
        key = CacheKey(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date,
        )
        return key in self._cache

    def invalidate(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> int:
        """
        Invalidate (remove) cached data matching the given parameters.

        If start_date and end_date are None, removes all entries for the
        given provider/symbol/bin_size combination.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []

        for key in self._cache.keys():
            if (key.provider == provider
                    and key.symbol.lower() == symbol.lower()
                    and key.bin_size == bin_size):
                # If dates specified, only remove exact match
                if start_date is not None and end_date is not None:
                    if key.start_date == start_date and key.end_date == end_date:
                        keys_to_remove.append(key)
                else:
                    # Remove all matching provider/symbol/bin_size
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        if keys_to_remove:
            self.logger.info(
                f"Invalidated {len(keys_to_remove)} cache entries for "
                f"{provider.value}:{symbol}:{bin_size}"
            )

        return len(keys_to_remove)

    def preload(
        self,
        provider: DataSourceProviders,
        symbol: str,
        bin_size: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Explicitly preload data into cache.

        Same as get_dataframe() but with clearer intent for preloading.

        Args:
            provider: Data source provider
            symbol: Trading pair symbol
            bin_size: Candle size
            start_date: Start of data range
            end_date: End of data range

        Returns:
            Loaded DataFrame
        """
        return self.get_dataframe(provider, symbol, bin_size, start_date, end_date)

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DataFrameCache("
            f"entries={stats['entries']}, "
            f"hits={stats['hits']}, "
            f"misses={stats['misses']}, "
            f"memory={stats['memory_bytes'] / 1e6:.2f}MB)"
        )


def create_dataframe_cache(
    rethinkdb_config: Any = None,
    adapter_type: str = 'rethinkdb',
    **kwargs
) -> DataFrameCache:
    """
    Factory function to create a DataFrameCache with appropriate adapter.

    Args:
        rethinkdb_config: RethinkDB configuration object
        adapter_type: Type of adapter ('rethinkdb', 'parquet', 'auto')
        **kwargs: Additional arguments passed to adapter

    Returns:
        Configured DataFrameCache instance
    """
    from .data_adapter import get_adapter

    adapter = get_adapter(
        config=rethinkdb_config,
        adapter_type=adapter_type,
        **kwargs
    )

    return DataFrameCache(data_adapter=adapter)
