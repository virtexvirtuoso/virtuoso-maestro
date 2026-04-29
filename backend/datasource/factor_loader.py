"""
Fama-French Factor Loader for Maestro Backtesting System.
Downloads, caches, and serves FF factor data via pandas-datareader.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/factors"))
CACHE_TTL = 86400 * 7  # 7 days


class FactorDataLoader:
    """Downloads and caches Fama-French factor data."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def _is_stale(self, path: Path) -> bool:
        if not path.exists():
            return True
        return (time.time() - path.stat().st_mtime) > CACHE_TTL

    def _fetch_ff(self, dataset: str, cache_name: str) -> pd.DataFrame:
        """Generic Fama-French dataset fetcher with caching."""
        cache = self._cache_path(cache_name)

        if not self._is_stale(cache):
            logger.info(f"Loading {cache_name} from cache")
            return pd.read_parquet(cache)

        logger.info(f"Downloading {dataset} from Ken French library")
        import pandas_datareader.data as web
        data = web.DataReader(dataset, 'famafrench', start='1926')[0]

        # Convert PeriodIndex to DatetimeIndex
        if hasattr(data.index, 'to_timestamp'):
            data.index = data.index.to_timestamp()
        data.index = pd.DatetimeIndex(data.index).tz_localize(None)
        data.index.name = "timestamp"

        # Values come as percentages, convert to decimal
        data = data.astype(float) / 100.0
        data = data.dropna(how="all")
        data.to_parquet(cache)
        return data

    def get_ff3(self) -> pd.DataFrame:
        """Fama-French 3 factors: Mkt-RF, SMB, HML + RF."""
        return self._fetch_ff('F-F_Research_Data_Factors', 'ff3')

    def get_ff5(self) -> pd.DataFrame:
        """Fama-French 5 factors: Mkt-RF, SMB, HML, RMW, CMA + RF."""
        return self._fetch_ff('F-F_Research_Data_5_Factors_2x3', 'ff5')

    def get_momentum(self) -> pd.DataFrame:
        """Momentum factor (UMD/Mom)."""
        return self._fetch_ff('F-F_Momentum_Factor', 'momentum')

    def get_all_factors(self) -> pd.DataFrame:
        """Combined DataFrame of FF5 + Momentum."""
        ff5 = self.get_ff5()
        mom = self.get_momentum()
        # Rename momentum column
        if mom.columns.size == 1:
            mom.columns = ['Mom']
        elif 'Mom   ' in mom.columns:
            mom = mom.rename(columns={'Mom   ': 'Mom'})
        combined = ff5.join(mom, how='outer')
        combined = combined.ffill().dropna(how="all")
        return combined
