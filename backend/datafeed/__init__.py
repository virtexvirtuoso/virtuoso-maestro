"""
Maestro Datafeed Module - Data adapters and caching for OHLCV data.
"""

from .data_adapter import (
    DataAdapter,
    DataRequest,
    ParquetAdapter,
    QuestDBAdapter,
    RethinkDBAdapter,
    create_data_adapter,
    get_adapter,
)
from .dataframe_cache import (
    CacheKey,
    DataFrameCache,
    create_dataframe_cache,
)

__all__ = [
    # Data Adapters
    'DataAdapter',
    'DataRequest',
    'RethinkDBAdapter',
    'ParquetAdapter',
    'QuestDBAdapter',
    'get_adapter',
    'create_data_adapter',
    # DataFrame Cache
    'CacheKey',
    'DataFrameCache',
    'create_dataframe_cache',
]
