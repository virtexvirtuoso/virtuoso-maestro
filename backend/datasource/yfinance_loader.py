"""
YFinance Stock Data Loader for Maestro Backtesting System.
Downloads, caches, and serves OHLCV data for US stocks/ETFs.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Pre-defined symbol groups
BTC_PROXIES = ["MSTR", "IBIT", "FBTC", "GBTC", "MARA", "RIOT", "COIN"]
SECTOR_ETFS = ["SPY", "QQQ", "XLF", "XLE", "XLK", "XLV", "GLD", "TLT", "IWM", "DIA"]
MACRO_PROXIES = ["TLT", "GLD", "UUP", "HYG", "LQD"]

SYMBOL_GROUPS = {
    "BTC_PROXIES": BTC_PROXIES,
    "SECTOR_ETFS": SECTOR_ETFS,
    "MACRO_PROXIES": MACRO_PROXIES,
}

TIMEFRAME_MAP = {
    "1d": "1d",
    "daily": "1d",
    "1w": "1wk",
    "weekly": "1wk",
    "1mo": "1mo",
    "monthly": "1mo",
}

# Cache staleness thresholds in seconds
STALE_THRESHOLDS = {
    "1d": 86400,      # 1 day
    "1wk": 86400 * 3, # 3 days
    "1mo": 86400 * 7,  # 7 days
}

DATA_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/stocks"))


class StockDataLoader:
    """Downloads and caches stock/ETF OHLCV data via yfinance."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        yf_tf = TIMEFRAME_MAP.get(timeframe, timeframe)
        return self.data_dir / f"{symbol.upper()}_{yf_tf}.parquet"

    def _is_stale(self, path: Path, timeframe: str) -> bool:
        if not path.exists():
            return True
        yf_tf = TIMEFRAME_MAP.get(timeframe, timeframe)
        threshold = STALE_THRESHOLDS.get(yf_tf, 86400)
        age = time.time() - path.stat().st_mtime
        return age > threshold

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol. Returns DataFrame with DatetimeIndex
        and float columns: open, high, low, close, volume.
        """
        symbol = symbol.upper()
        yf_tf = TIMEFRAME_MAP.get(timeframe, timeframe)
        cache = self._cache_path(symbol, timeframe)

        if not self._is_stale(cache, timeframe):
            logger.info(f"Loading {symbol} {yf_tf} from cache")
            df = pd.read_parquet(cache)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            logger.info(f"Downloading {symbol} {yf_tf} from yfinance")
            ticker = yf.Ticker(symbol)
            kw = {"interval": yf_tf}
            if start_date:
                kw["start"] = start_date
            else:
                kw["period"] = "max"
            if end_date:
                kw["end"] = end_date
            df = ticker.history(**kw)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            df = self._normalize(df)
            df.to_parquet(cache)

        # Apply date filters on cached data
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output to Maestro-compatible format."""
        df = df.copy()
        # yfinance returns columns like Open, High, Low, Close, Volume
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("open", "high", "low", "close", "volume"):
                col_map[c] = cl
        df = df.rename(columns=col_map)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "timestamp"
        df = df.dropna()
        return df

    def get_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV for multiple symbols."""
        results = {}
        for sym in symbols:
            try:
                results[sym.upper()] = self.get_ohlcv(sym, timeframe, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to load {sym}: {e}")
        return results

    def get_fundamentals(self, symbol: str) -> dict:
        """Get fundamental data for a symbol."""
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info or {}
        keys = [
            "marketCap", "trailingPE", "forwardPE", "dividendYield",
            "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "sector",
            "industry", "shortName", "revenue", "profitMargins",
            "returnOnEquity", "debtToEquity", "freeCashflow",
        ]
        return {k: info.get(k) for k in keys if info.get(k) is not None}

    def get_group(
        self, group_name: str, timeframe: str = "1d", **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Load a pre-defined symbol group."""
        symbols = SYMBOL_GROUPS.get(group_name, [])
        if not symbols:
            raise ValueError(f"Unknown group: {group_name}. Available: {list(SYMBOL_GROUPS)}")
        return self.get_multiple(symbols, timeframe, **kwargs)
