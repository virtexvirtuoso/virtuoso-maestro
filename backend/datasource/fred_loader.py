"""
FRED Macro Data Loader for Maestro Backtesting System.
Downloads, caches, and serves macroeconomic time series from FRED.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# --- Pre-defined series groups ---
RATES = {
    'fed_funds': 'FEDFUNDS', 'us10y': 'DGS10', 'us2y': 'DGS2',
    'us3m': 'DGS3MO', 'yield_spread_10y2y': 'T10Y2Y',
}
INFLATION = {
    'cpi_yoy': 'CPIAUCSL', 'core_cpi': 'CPILFESL', 'pce': 'PCEPI',
    'breakeven_5y': 'T5YIE', 'breakeven_10y': 'T10YIE',
}
LIQUIDITY = {'m2': 'M2SL', 'm2_velocity': 'M2V', 'excess_reserves': 'EXCSRESNS'}
LABOR = {'unemployment': 'UNRATE', 'nonfarm_payrolls': 'PAYEMS', 'initial_claims': 'ICSA'}
GROWTH = {
    'gdp': 'GDP', 'real_gdp': 'GDPC1', 'industrial_prod': 'INDPRO',
    'retail_sales': 'RSAFS',
}
FINANCIAL = {
    'sp500': 'SP500', 'vix': 'VIXCLS', 'dxy': 'DTWEXBGS',
    'gold': 'GOLDAMGBD228NLBM', 'oil_wti': 'DCOILWTICO',
    'hyg_spread': 'BAMLH0A0HYM2',
}

ALL_GROUPS = {
    'rates': RATES, 'inflation': INFLATION, 'liquidity': LIQUIDITY,
    'labor': LABOR, 'growth': GROWTH, 'financial': FINANCIAL,
}

# Monthly/quarterly FRED series (cache 7 days); all others cache 1 day
MONTHLY_QUARTERLY = {
    'FEDFUNDS', 'CPIAUCSL', 'CPILFESL', 'PCEPI', 'M2SL', 'M2V',
    'EXCSRESNS', 'UNRATE', 'PAYEMS', 'GDP', 'GDPC1', 'INDPRO', 'RSAFS',
}

DATA_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/macro"))

# Try to load FRED API key from btc_wiz .env or environment
def _load_fred_key() -> Optional[str]:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    env_path = Path(os.path.expanduser("~/Desktop/btc_wiz/.env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class MacroDataLoader:
    """Downloads and caches FRED macro time series."""

    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[Path] = None):
        self.api_key = api_key or _load_fred_key()
        if not self.api_key:
            raise ValueError("FRED_API_KEY not found. Set env var or add to ~/Desktop/btc_wiz/.env")
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        from fredapi import Fred
        self.fred = Fred(api_key=self.api_key)

    def _cache_path(self, series_id: str) -> Path:
        return self.data_dir / f"{series_id}.parquet"

    def _cache_ttl(self, series_id: str) -> int:
        """Return cache TTL in seconds."""
        return 86400 * 7 if series_id in MONTHLY_QUARTERLY else 86400

    def _is_stale(self, path: Path, series_id: str) -> bool:
        if not path.exists():
            return True
        age = time.time() - path.stat().st_mtime
        return age > self._cache_ttl(series_id)

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch a single FRED series. Returns pandas Series with DatetimeIndex."""
        cache = self._cache_path(series_id)

        if not self._is_stale(cache, series_id):
            logger.info(f"Loading {series_id} from cache")
            df = pd.read_parquet(cache)
            s = df.iloc[:, 0]
        else:
            logger.info(f"Downloading {series_id} from FRED")
            s = self.fred.get_series(series_id)
            if s is None or s.empty:
                logger.warning(f"No data for {series_id}")
                return pd.Series(dtype=float)
            s = s.astype(float)
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            s.index.name = "timestamp"
            s.name = series_id
            s = s.dropna()
            s.to_frame().to_parquet(cache)

        if start_date:
            s = s[s.index >= pd.Timestamp(start_date)]
        if end_date:
            s = s[s.index <= pd.Timestamp(end_date)]
        return s

    def get_multiple(
        self,
        series_ids: Union[List[str], Dict[str, str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch multiple series, aligned on date. Accepts list of FRED IDs
        or dict of {friendly_name: FRED_ID}."""
        if isinstance(series_ids, dict):
            mapping = series_ids
        else:
            mapping = {sid: sid for sid in series_ids}

        frames = {}
        for name, sid in mapping.items():
            try:
                s = self.get_series(sid, start_date, end_date)
                s.name = name
                frames[name] = s
            except Exception as e:
                logger.error(f"Failed to load {sid}: {e}")

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df.ffill().dropna(how="all")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz else df.index
        df.index.name = "timestamp"
        return df

    def get_group(self, group_name: str, **kwargs) -> pd.DataFrame:
        """Load a pre-defined series group (rates, inflation, etc.)."""
        group = ALL_GROUPS.get(group_name.lower())
        if not group:
            raise ValueError(f"Unknown group: {group_name}. Available: {list(ALL_GROUPS)}")
        return self.get_multiple(group, **kwargs)

    def get_macro_regime(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Return a combined DataFrame of key macro indicators for regime analysis."""
        key_series = {
            'yield_spread': 'T10Y2Y',
            'fed_funds': 'FEDFUNDS',
            'cpi': 'CPIAUCSL',
            'unemployment': 'UNRATE',
            'm2': 'M2SL',
            'vix': 'VIXCLS',
            'hyg_spread': 'BAMLH0A0HYM2',
        }
        df = self.get_multiple(key_series, start_date=start_date)
        # Add derived columns
        if 'cpi' in df.columns:
            df['cpi_yoy_pct'] = df['cpi'].pct_change(12) * 100  # 12-month pct change
        if 'm2' in df.columns:
            df['m2_yoy_pct'] = df['m2'].pct_change(12) * 100
        return df
