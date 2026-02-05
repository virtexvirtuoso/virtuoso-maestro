"""
Parquet Cache Tests - Unit tests for the ParquetCache and CachedDataAdapter.

Tests cover:
- ParquetCache: is_cached, get, put, invalidate, expiration
- CachedDataAdapter: cache miss/hit flow, stats tracking
- Performance: timing comparison (cache vs primary adapter)
- Compression: zstd compression ratio validation
"""

import sys
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.parquet_cache import ParquetCache, CachedDataAdapter
from datasource.providers import DataSourceProviders


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing (1000 rows)."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='h', tz='UTC')
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    high = close + np.abs(np.random.randn(1000)) * 0.5
    low = close - np.abs(np.random.randn(1000)) * 0.5
    open_ = close + np.random.randn(1000) * 0.25
    volume = np.random.randint(1000, 100000, 1000).astype(float)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)
    df.index.name = 'timestamp'

    return df


@pytest.fixture
def large_ohlcv_data():
    """Generate larger sample data for performance testing (10000 rows)."""
    dates = pd.date_range(start='2023-01-01', periods=10000, freq='h', tz='UTC')
    np.random.seed(123)

    close = 50000 + np.cumsum(np.random.randn(10000) * 100)
    high = close + np.abs(np.random.randn(10000)) * 50
    low = close - np.abs(np.random.randn(10000)) * 50
    open_ = close + np.random.randn(10000) * 25
    volume = np.random.randint(10000, 1000000, 10000).astype(float)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)
    df.index.name = 'timestamp'

    return df


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache(temp_cache_dir):
    """Create a ParquetCache instance."""
    return ParquetCache(cache_dir=temp_cache_dir, max_age_seconds=3600)


@pytest.fixture
def mock_primary_adapter(sample_ohlcv_data):
    """Create a mock primary data adapter."""
    adapter = Mock()
    adapter.load_dataframe.return_value = sample_ohlcv_data
    adapter.is_available.return_value = True
    adapter.has_data.return_value = True
    return adapter


# =============================================================================
# PARQUET CACHE TESTS
# =============================================================================

class TestParquetCache:
    """Tests for ParquetCache."""

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that initialization creates cache directory."""
        cache_dir = Path(temp_cache_dir) / 'new_subdir'
        cache = ParquetCache(cache_dir=str(cache_dir))
        assert cache_dir.exists()

    def test_is_cached_returns_false_for_missing(self, cache):
        """Test is_cached returns False for non-existent file."""
        assert cache.is_cached(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        ) is False

    def test_is_cached_returns_true_after_put(self, cache, sample_ohlcv_data):
        """Test is_cached returns True after caching data."""
        cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        assert cache.is_cached(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        ) is True

    def test_is_cached_returns_false_for_expired(self, temp_cache_dir, sample_ohlcv_data):
        """Test is_cached returns False for expired cache."""
        # Create cache with very short expiration
        cache = ParquetCache(cache_dir=temp_cache_dir, max_age_seconds=1)

        cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        # Wait for expiration
        time.sleep(1.5)

        assert cache.is_cached(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        ) is False

    def test_put_creates_parquet_file(self, cache, sample_ohlcv_data):
        """Test put creates a Parquet file."""
        path, size = cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        assert path.exists()
        assert path.suffix == '.parquet'
        assert size > 0

    def test_get_returns_none_for_missing(self, cache):
        """Test get returns None for missing cache entry."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = cache.get(
            provider=DataSourceProviders.BINANCE,
            symbol='missing',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        assert result is None

    def test_get_returns_data_after_put(self, cache, sample_ohlcv_data):
        """Test get returns data after successful put."""
        cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = cache.get(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        assert result is not None
        assert len(result) == len(sample_ohlcv_data)
        assert set(result.columns) == {'open', 'high', 'low', 'close', 'volume'}

    def test_get_filters_by_date_range(self, cache, sample_ohlcv_data):
        """Test get filters data by date range."""
        cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        # Request subset of data
        start = datetime(2024, 1, 10, tzinfo=timezone.utc)
        end = datetime(2024, 1, 20, tzinfo=timezone.utc)

        result = cache.get(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        assert result is not None
        assert len(result) < len(sample_ohlcv_data)
        assert result.index.min() >= start
        assert result.index.max() <= end

    def test_invalidate_removes_cache(self, cache, sample_ohlcv_data):
        """Test invalidate removes cache entry."""
        cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        assert cache.is_cached(DataSourceProviders.BINANCE, 'ethbtc', '1h') is True

        result = cache.invalidate(DataSourceProviders.BINANCE, 'ethbtc', '1h')

        assert result is True
        assert cache.is_cached(DataSourceProviders.BINANCE, 'ethbtc', '1h') is False

    def test_invalidate_returns_false_for_missing(self, cache):
        """Test invalidate returns False for non-existent entry."""
        result = cache.invalidate(DataSourceProviders.BINANCE, 'missing', '1h')
        assert result is False

    def test_invalidate_all(self, cache, sample_ohlcv_data):
        """Test invalidate_all clears all cache entries."""
        # Add multiple cache entries
        cache.put(sample_ohlcv_data, DataSourceProviders.BINANCE, 'ethbtc', '1h')
        cache.put(sample_ohlcv_data, DataSourceProviders.BINANCE, 'btcusdt', '1d')

        count = cache.invalidate_all()

        assert count == 2
        assert not cache.is_cached(DataSourceProviders.BINANCE, 'ethbtc', '1h')
        assert not cache.is_cached(DataSourceProviders.BINANCE, 'btcusdt', '1d')

    def test_data_integrity(self, cache, sample_ohlcv_data):
        """Test that cached data matches original data exactly."""
        cache.put(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h'
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = cache.get(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        # Verify numerical values match exactly
        np.testing.assert_array_almost_equal(
            sample_ohlcv_data['close'].values,
            result['close'].values,
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            sample_ohlcv_data['volume'].values,
            result['volume'].values,
            decimal=10
        )

    def test_get_stats(self, cache, sample_ohlcv_data):
        """Test get_stats returns correct statistics."""
        cache.put(sample_ohlcv_data, DataSourceProviders.BINANCE, 'ethbtc', '1h')
        cache.put(sample_ohlcv_data, DataSourceProviders.BINANCE, 'btcusdt', '1d')

        stats = cache.get_stats()

        assert stats['total_files'] == 2
        assert stats['total_size_mb'] > 0
        assert stats['expired_count'] == 0
        assert stats['max_age_seconds'] == 3600


# =============================================================================
# CACHED DATA ADAPTER TESTS
# =============================================================================

class TestCachedDataAdapter:
    """Tests for CachedDataAdapter cache-through pattern."""

    def test_cache_miss_calls_primary(
        self, temp_cache_dir, mock_primary_adapter, sample_ohlcv_data
    ):
        """Test that cache miss loads from primary adapter."""
        cache = ParquetCache(cache_dir=temp_cache_dir)
        adapter = CachedDataAdapter(mock_primary_adapter, cache)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        # Primary adapter should have been called
        mock_primary_adapter.load_dataframe.assert_called_once()
        assert len(result) == len(sample_ohlcv_data)

    def test_cache_hit_skips_primary(
        self, temp_cache_dir, mock_primary_adapter, sample_ohlcv_data
    ):
        """Test that cache hit does NOT call primary adapter."""
        cache = ParquetCache(cache_dir=temp_cache_dir)
        adapter = CachedDataAdapter(mock_primary_adapter, cache)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        # First call - cache miss
        adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        # Reset mock to verify second call doesn't hit primary
        mock_primary_adapter.load_dataframe.reset_mock()

        # Second call - should be cache hit
        result = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1h',
            start_date=start,
            end_date=end
        )

        # Primary adapter should NOT have been called
        mock_primary_adapter.load_dataframe.assert_not_called()
        assert len(result) == len(sample_ohlcv_data)

    def test_stats_tracking(
        self, temp_cache_dir, mock_primary_adapter, sample_ohlcv_data
    ):
        """Test that hit/miss stats are tracked correctly."""
        cache = ParquetCache(cache_dir=temp_cache_dir)
        adapter = CachedDataAdapter(mock_primary_adapter, cache)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        # First call - miss
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )

        # Second call - hit
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )

        # Third call - hit
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )

        stats = adapter.get_stats()

        assert stats['cache_hits'] == 2
        assert stats['cache_misses'] == 1
        assert stats['hit_rate'] == pytest.approx(2/3)

    def test_invalidate_cache(
        self, temp_cache_dir, mock_primary_adapter, sample_ohlcv_data
    ):
        """Test cache invalidation through adapter."""
        cache = ParquetCache(cache_dir=temp_cache_dir)
        adapter = CachedDataAdapter(mock_primary_adapter, cache)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        # Populate cache
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )

        # Invalidate
        adapter.invalidate_cache(DataSourceProviders.BINANCE, 'ethbtc', '1h')

        # Reset mock
        mock_primary_adapter.load_dataframe.reset_mock()

        # Next load should miss cache
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )

        # Primary should have been called again
        mock_primary_adapter.load_dataframe.assert_called_once()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestCachePerformance:
    """Performance tests comparing cache vs primary adapter."""

    def test_cache_faster_than_mock_primary(
        self, temp_cache_dir, sample_ohlcv_data
    ):
        """Test that cache read is faster than simulated slow primary."""
        # Create a slow primary adapter (simulates DB latency)
        slow_adapter = Mock()

        def slow_load(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms DB query
            return sample_ohlcv_data

        slow_adapter.load_dataframe.side_effect = slow_load
        slow_adapter.is_available.return_value = True

        cache = ParquetCache(cache_dir=temp_cache_dir)
        adapter = CachedDataAdapter(slow_adapter, cache)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        # First call - cache miss (slow)
        miss_start = time.perf_counter()
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )
        miss_time = time.perf_counter() - miss_start

        # Second call - cache hit (fast)
        hit_start = time.perf_counter()
        adapter.load_dataframe(
            DataSourceProviders.BINANCE, 'ethbtc', '1h', start, end
        )
        hit_time = time.perf_counter() - hit_start

        # Cache hit should be significantly faster
        assert hit_time < miss_time
        # At least 5x faster (100ms primary vs ~10ms cache)
        speedup = miss_time / hit_time
        assert speedup > 5, f'Expected >5x speedup, got {speedup:.1f}x'

    def test_large_data_performance(self, temp_cache_dir, large_ohlcv_data):
        """Test cache performance with larger dataset."""
        cache = ParquetCache(cache_dir=temp_cache_dir)

        # Measure write time
        write_start = time.perf_counter()
        path, size = cache.put(
            large_ohlcv_data,
            DataSourceProviders.BINANCE,
            'btcusdt',
            '1h'
        )
        write_time = time.perf_counter() - write_start

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        # Measure read time
        read_start = time.perf_counter()
        result = cache.get(
            DataSourceProviders.BINANCE,
            'btcusdt',
            '1h',
            start_date=start,
            end_date=end
        )
        read_time = time.perf_counter() - read_start

        # Both should complete reasonably fast (< 1 second for 10K rows)
        assert write_time < 1.0, f'Write took {write_time:.2f}s'
        assert read_time < 0.5, f'Read took {read_time:.2f}s'
        assert len(result) == 10000


# =============================================================================
# COMPRESSION TESTS
# =============================================================================

class TestCompression:
    """Tests for zstd compression effectiveness."""

    def test_compressed_smaller_than_raw(self, temp_cache_dir, large_ohlcv_data):
        """Test that compressed Parquet is smaller than raw CSV equivalent."""
        cache = ParquetCache(cache_dir=temp_cache_dir)

        # Cache the data
        path, compressed_size = cache.put(
            large_ohlcv_data,
            DataSourceProviders.BINANCE,
            'btcusdt',
            '1h'
        )

        # Calculate raw size (CSV approximation)
        csv_path = Path(temp_cache_dir) / 'temp.csv'
        large_ohlcv_data.to_csv(csv_path)
        raw_size = csv_path.stat().st_size

        # Parquet with zstd should be significantly smaller
        compression_ratio = raw_size / compressed_size
        assert compression_ratio > 2.0, (
            f'Expected >2x compression, got {compression_ratio:.1f}x '
            f'(raw={raw_size/1024:.1f}KB, compressed={compressed_size/1024:.1f}KB)'
        )

        # Log the actual sizes for reference
        print(f'\nCompression stats:')
        print(f'  Raw CSV: {raw_size / 1024:.1f} KB')
        print(f'  Parquet (zstd): {compressed_size / 1024:.1f} KB')
        print(f'  Compression ratio: {compression_ratio:.1f}x')

    def test_uncompressed_parquet_baseline(self, temp_cache_dir, large_ohlcv_data):
        """Test that zstd is smaller than uncompressed Parquet."""
        import polars as pl

        # Write with zstd (default)
        cache = ParquetCache(cache_dir=temp_cache_dir)
        _, zstd_size = cache.put(
            large_ohlcv_data,
            DataSourceProviders.BINANCE,
            'btcusdt',
            '1h'
        )

        # Write uncompressed for comparison
        uncompressed_path = Path(temp_cache_dir) / 'uncompressed.parquet'
        df_to_save = large_ohlcv_data.reset_index()
        df_to_save['timestamp'] = (
            pd.to_datetime(df_to_save['timestamp'], utc=True)
            .astype('int64') // 1000
        )
        pl_df = pl.from_pandas(df_to_save)
        pl_df.write_parquet(uncompressed_path, compression='uncompressed')
        uncompressed_size = uncompressed_path.stat().st_size

        # zstd should be smaller than uncompressed
        assert zstd_size < uncompressed_size, (
            f'zstd ({zstd_size}) should be smaller than '
            f'uncompressed ({uncompressed_size})'
        )


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_dataframe(self, cache):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume'],
            index=pd.DatetimeIndex([], name='timestamp', tz='UTC')
        )

        path, size = cache.put(
            empty_df,
            DataSourceProviders.BINANCE,
            'empty',
            '1h'
        )

        assert path.exists()

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        result = cache.get(
            DataSourceProviders.BINANCE,
            'empty',
            '1h',
            start_date=start,
            end_date=end
        )

        assert result is not None
        assert len(result) == 0

    def test_special_characters_in_symbol(self, cache, sample_ohlcv_data):
        """Test handling of symbols with various formats."""
        symbols = ['ethbtc', 'ETH-BTC', 'eth_btc']

        for symbol in symbols:
            cache.put(
                sample_ohlcv_data,
                DataSourceProviders.BINANCE,
                symbol,
                '1h'
            )

            assert cache.is_cached(DataSourceProviders.BINANCE, symbol, '1h')

    def test_concurrent_access_safety(self, cache, sample_ohlcv_data):
        """Test that cache handles concurrent access gracefully."""
        import threading

        errors = []

        def write_cache(symbol):
            try:
                cache.put(sample_ohlcv_data, DataSourceProviders.BINANCE, symbol, '1h')
            except Exception as e:
                errors.append(e)

        def read_cache(symbol):
            try:
                start = datetime(2024, 1, 1, tzinfo=timezone.utc)
                end = datetime(2024, 12, 31, tzinfo=timezone.utc)
                cache.get(DataSourceProviders.BINANCE, symbol, '1h', start, end)
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent access
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=write_cache, args=(f'sym{i}',))
            t2 = threading.Thread(target=read_cache, args=(f'sym{i}',))
            threads.extend([t1, t2])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f'Concurrent access errors: {errors}'


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
