"""
Data Adapter Tests - Unit tests for the DataAdapter abstraction layer.

Tests cover:
- RethinkDBAdapter: Connection, data loading, availability checks
- ParquetAdapter: File I/O, caching, availability
- QuestDBAdapter: Stub behavior, NotImplementedError
- Factory function: get_adapter() with different adapter types
- Data validation: Ensure adapter output matches direct RethinkDB query
"""

import sys
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datafeed.data_adapter import (
    DataAdapter,
    DataRequest,
    RethinkDBAdapter,
    ParquetAdapter,
    QuestDBAdapter,
    get_adapter,
    create_data_adapter,
)
from datasource.providers import DataSourceProviders


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D', tz='UTC')
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(100))
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_ = close + np.random.randn(100) * 0.5
    volume = np.random.randint(1000, 10000, 100)

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
def temp_parquet_dir():
    """Create a temporary directory for Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_rethinkdb_config():
    """Create a mock RethinkDB config object."""
    config = Mock()
    config.host = '127.0.0.1'
    config.port = 28015
    config.db = 'filos-test'
    return config


# =============================================================================
# RETHINKDB ADAPTER TESTS
# =============================================================================

class TestRethinkDBAdapter:
    """Tests for RethinkDBAdapter."""

    def test_init_with_direct_params(self):
        """Test initialization with direct host/port/db parameters."""
        adapter = RethinkDBAdapter(
            host='localhost',
            port=28015,
            db='test-db'
        )
        assert adapter.host == 'localhost'
        assert adapter.port == 28015
        assert adapter.db == 'test-db'

    def test_init_with_config_object(self, mock_rethinkdb_config):
        """Test initialization with config object."""
        adapter = RethinkDBAdapter(rethinkdb_config=mock_rethinkdb_config)
        assert adapter.host == '127.0.0.1'
        assert adapter.port == 28015
        assert adapter.db == 'filos-test'

    def test_config_takes_precedence(self, mock_rethinkdb_config):
        """Test that config object takes precedence over direct params."""
        adapter = RethinkDBAdapter(
            host='should-be-ignored',
            port=99999,
            db='ignored-db',
            rethinkdb_config=mock_rethinkdb_config
        )
        assert adapter.host == mock_rethinkdb_config.host
        assert adapter.port == mock_rethinkdb_config.port
        assert adapter.db == mock_rethinkdb_config.db

    @patch('rethinkdb.RethinkDB')
    def test_load_dataframe_returns_correct_columns(self, mock_r, sample_ohlcv_data):
        """Test that load_dataframe returns expected columns."""
        # Mock RethinkDB connection and query
        mock_conn = MagicMock()
        mock_r.return_value.connect.return_value = mock_conn

        # Convert sample data to records format (what RethinkDB returns)
        records = sample_ohlcv_data.reset_index().to_dict('records')
        mock_r.return_value.table.return_value.order_by.return_value.pluck.return_value.run.return_value = records

        adapter = RethinkDBAdapter(host='localhost', port=28015, db='test')

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 4, 10, tzinfo=timezone.utc)

        df = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        )

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {'open', 'high', 'low', 'close', 'volume'}
        assert isinstance(df.index, pd.DatetimeIndex)

    @patch('rethinkdb.RethinkDB')
    def test_is_available_success(self, mock_r):
        """Test is_available returns True when connection succeeds."""
        mock_r.return_value.connect.return_value = MagicMock()

        adapter = RethinkDBAdapter(host='localhost', port=28015, db='test')
        assert adapter.is_available() is True

    @patch('rethinkdb.RethinkDB')
    def test_is_available_failure(self, mock_r):
        """Test is_available returns False when connection fails."""
        mock_r.return_value.connect.side_effect = Exception('Connection refused')

        adapter = RethinkDBAdapter(host='localhost', port=28015, db='test')
        assert adapter.is_available() is False


# =============================================================================
# PARQUET ADAPTER TESTS
# =============================================================================

class TestParquetAdapter:
    """Tests for ParquetAdapter."""

    def test_init_creates_directory(self, temp_parquet_dir):
        """Test that initialization creates data directory."""
        data_dir = Path(temp_parquet_dir) / 'new_subdir'
        adapter = ParquetAdapter(data_dir=str(data_dir))
        assert data_dir.exists()

    def test_save_and_load_roundtrip(self, temp_parquet_dir, sample_ohlcv_data):
        """Test saving and loading data preserves content."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)

        # Save data
        adapter.save_dataframe(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d'
        )

        # Load data
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        loaded = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        )

        # Verify data integrity
        assert len(loaded) == len(sample_ohlcv_data)
        assert set(loaded.columns) == set(sample_ohlcv_data.columns)
        np.testing.assert_array_almost_equal(
            loaded['close'].values,
            sample_ohlcv_data['close'].values
        )

    def test_load_nonexistent_file_raises(self, temp_parquet_dir):
        """Test loading non-existent file raises FileNotFoundError."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        with pytest.raises(FileNotFoundError):
            adapter.load_dataframe(
                provider=DataSourceProviders.BINANCE,
                symbol='nonexistent',
                bin_size='1d',
                start_date=start,
                end_date=end
            )

    def test_has_data_returns_false_for_missing(self, temp_parquet_dir):
        """Test has_data returns False for missing file."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        assert adapter.has_data(
            provider=DataSourceProviders.BINANCE,
            symbol='missing',
            bin_size='1d',
            start_date=start,
            end_date=end
        ) is False

    def test_has_data_returns_true_for_existing(self, temp_parquet_dir, sample_ohlcv_data):
        """Test has_data returns True for existing file."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)

        adapter.save_dataframe(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d'
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        assert adapter.has_data(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        ) is True

    def test_is_available(self, temp_parquet_dir):
        """Test is_available returns True for valid directory."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)
        assert adapter.is_available() is True

    def test_date_filtering(self, temp_parquet_dir, sample_ohlcv_data):
        """Test that date range filtering works correctly."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)

        adapter.save_dataframe(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d'
        )

        # Request subset of data
        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 2, 15, tzinfo=timezone.utc)

        loaded = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        )

        # Should have ~32 days of data
        assert len(loaded) < len(sample_ohlcv_data)
        assert loaded.index.min() >= start
        assert loaded.index.max() <= end


# =============================================================================
# QUESTDB ADAPTER TESTS
# =============================================================================

class TestQuestDBAdapter:
    """Tests for QuestDBAdapter (Phase 2 stub)."""

    def test_init(self):
        """Test initialization stores parameters."""
        adapter = QuestDBAdapter(host='localhost', port=8812, db='qdb')
        assert adapter.host == 'localhost'
        assert adapter.port == 8812
        assert adapter.db == 'qdb'

    def test_is_available_returns_false(self):
        """Test is_available always returns False (stub)."""
        adapter = QuestDBAdapter()
        assert adapter.is_available() is False

    def test_load_dataframe_raises_not_implemented(self):
        """Test load_dataframe raises NotImplementedError."""
        adapter = QuestDBAdapter()

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        with pytest.raises(NotImplementedError):
            adapter.load_dataframe(
                provider=DataSourceProviders.BINANCE,
                symbol='ethbtc',
                bin_size='1d',
                start_date=start,
                end_date=end
            )


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestGetAdapterFactory:
    """Tests for get_adapter factory function."""

    def test_get_rethinkdb_adapter(self, mock_rethinkdb_config):
        """Test creating RethinkDB adapter via factory."""
        adapter = get_adapter(config=mock_rethinkdb_config, adapter_type='rethinkdb')
        assert isinstance(adapter, RethinkDBAdapter)

    def test_get_parquet_adapter(self, temp_parquet_dir):
        """Test creating Parquet adapter via factory."""
        adapter = get_adapter(adapter_type='parquet', data_dir=temp_parquet_dir)
        assert isinstance(adapter, ParquetAdapter)

    def test_get_questdb_adapter(self):
        """Test creating QuestDB adapter via factory."""
        adapter = get_adapter(adapter_type='questdb')
        assert isinstance(adapter, QuestDBAdapter)

    def test_case_insensitive_adapter_type(self):
        """Test adapter_type is case-insensitive."""
        adapter1 = get_adapter(adapter_type='RETHINKDB')
        adapter2 = get_adapter(adapter_type='RethinkDB')
        adapter3 = get_adapter(adapter_type='rethinkdb')

        assert type(adapter1) == type(adapter2) == type(adapter3) == RethinkDBAdapter

    def test_invalid_adapter_type_raises(self):
        """Test invalid adapter type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_adapter(adapter_type='invalid')

        assert 'Unknown adapter type' in str(exc_info.value)

    def test_create_data_adapter_convenience(self, mock_rethinkdb_config):
        """Test create_data_adapter convenience function."""
        adapter = create_data_adapter(rethinkdb_config=mock_rethinkdb_config)
        assert isinstance(adapter, RethinkDBAdapter)


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Tests to validate adapter data matches direct query results."""

    @patch('rethinkdb.RethinkDB')
    def test_rethinkdb_adapter_matches_direct_query(self, mock_r, sample_ohlcv_data):
        """Validate that RethinkDBAdapter output matches direct RethinkDB query format."""
        # Setup mock
        records = sample_ohlcv_data.reset_index().to_dict('records')
        mock_conn = MagicMock()
        mock_r.return_value.connect.return_value = mock_conn
        mock_r.return_value.table.return_value.order_by.return_value.pluck.return_value.run.return_value = records

        adapter = RethinkDBAdapter(host='localhost', port=28015, db='test')

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        df = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        )

        # Validate structure
        assert df.index.name == 'timestamp'
        assert df.index.tz is not None  # Should be timezone-aware
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        # Validate data types
        assert df['open'].dtype in [np.float64, np.float32]
        assert df['volume'].dtype in [np.int64, np.int32, np.float64]

    def test_parquet_preserves_data_integrity(self, temp_parquet_dir, sample_ohlcv_data):
        """Validate Parquet save/load preserves exact data."""
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)

        adapter.save_dataframe(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d'
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        loaded = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        )

        # Exact numerical match
        np.testing.assert_array_almost_equal(
            sample_ohlcv_data['open'].values,
            loaded['open'].values,
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            sample_ohlcv_data['close'].values,
            loaded['close'].values,
            decimal=10
        )


# =============================================================================
# V2 ENGINE INTEGRATION TESTS
# =============================================================================

class TestV2EngineIntegration:
    """Test that V2 engine correctly uses DataAdapter."""

    def test_v2_engine_accepts_adapter_data(self, temp_parquet_dir, sample_ohlcv_data):
        """Test V2 engine can use data from Parquet adapter."""
        from engine_v2.vectorbt_engine import VectorBTEngine, BacktestConfig

        # Save test data
        adapter = ParquetAdapter(data_dir=temp_parquet_dir)
        adapter.save_dataframe(
            df=sample_ohlcv_data,
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d'
        )

        # Load via adapter
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        df = adapter.load_dataframe(
            provider=DataSourceProviders.BINANCE,
            symbol='ethbtc',
            bin_size='1d',
            start_date=start,
            end_date=end
        )

        # Create simple signals
        close = df['close']
        entries = (close > close.shift(1)) & (close.shift(1) < close.shift(2))
        exits = (close < close.shift(1)) & (close.shift(1) > close.shift(2))

        # Run V2 engine
        engine = VectorBTEngine(config=BacktestConfig(cash=100000))
        result = engine.run(
            data=df,
            entries=entries,
            exits=exits,
            parameters={'test': True}
        )

        # Verify result
        assert result is not None
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'total_return')


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
