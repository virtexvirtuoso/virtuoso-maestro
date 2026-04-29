"""
Alpha Vantage Data Loader for Maestro Backtesting System.
Downloads, caches, and serves OHLCV, forex, technical indicator,
treasury yield, commodity, economic indicator, and fundamental data.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/alphavantage"))
CACHE_TTL_DAILY = 86400      # 1 day
CACHE_TTL_MONTHLY = 604800   # 7 days

BASE_URL = "https://www.alphavantage.co/query"

COMMODITIES = {
    'brent': 'BRENT',
    'copper': 'COPPER',
    'wheat': 'WHEAT',
    'corn': 'CORN',
    'cotton': 'COTTON',
    'coffee': 'COFFEE',
}

TREASURY_MATURITIES = ['2year', '5year', '7year', '10year', '30year']


def _get_av_key() -> Optional[str]:
    return os.environ.get("ALPHA_VANTAGE_API_KEY") or None


class AlphaVantageLoader:
    """Downloads and caches data from Alpha Vantage API."""

    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[Path] = None):
        self.api_key = api_key or _get_av_key() or 'L35OFXAYA4PZWT6D'
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._call_count = 0

    def _cache_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def _is_stale(self, path: Path, ttl: int = CACHE_TTL_DAILY) -> bool:
        if not path.exists():
            return True
        return (time.time() - path.stat().st_mtime) > ttl

    def _track_call(self):
        self._call_count += 1
        if self._call_count >= 20:
            logger.warning(f"AV call count: {self._call_count}/25 daily limit!")

    def _av_request(self, params: dict) -> dict:
        """Make a request to Alpha Vantage API."""
        self._track_call()
        params['apikey'] = self.api_key
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if 'Error Message' in data:
            raise ValueError(f"AV error: {data['Error Message']}")
        if 'Note' in data:
            logger.warning(f"AV rate limit note: {data['Note']}")
        return data

    def _parse_av_timeseries(self, data: dict, key: str = 'data',
                              value_col: str = 'value', col_name: str = 'value') -> pd.DataFrame:
        """Parse AV time series response into DataFrame."""
        if key not in data:
            logger.warning(f"Key '{key}' not in response: {list(data.keys())}")
            return pd.DataFrame()
        df = pd.DataFrame(data[key])
        if df.empty:
            return df
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df[[value_col]].rename(columns={value_col: col_name})
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = 'timestamp'
        return df.dropna()

    # ── Treasury Yields ──────────────────────────────────────────────

    def get_treasury_yield(self, maturity: str = '10year', interval: str = 'daily') -> pd.DataFrame:
        """Get treasury yield data. Maturities: 2year, 5year, 7year, 10year, 30year."""
        assert maturity in TREASURY_MATURITIES, f"Maturity must be in {TREASURY_MATURITIES}"
        cache_name = f"treasury_{maturity}_{interval}"
        cache = self._cache_path(cache_name)
        ttl = CACHE_TTL_DAILY if interval == 'daily' else CACHE_TTL_MONTHLY

        if not self._is_stale(cache, ttl):
            return pd.read_parquet(cache)

        logger.info(f"Downloading treasury yield {maturity} ({interval})")
        data = self._av_request({'function': 'TREASURY_YIELD', 'interval': interval, 'maturity': maturity})
        df = self._parse_av_timeseries(data, col_name=maturity)
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_yield_curve(self, interval: str = 'daily') -> pd.DataFrame:
        """Get all treasury maturities aligned into one DataFrame."""
        frames = []
        for m in TREASURY_MATURITIES:
            df = self.get_treasury_yield(m, interval)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index().ffill()

    def get_yield_spread(self, long: str = '10year', short: str = '2year',
                          interval: str = 'daily') -> pd.DataFrame:
        """Compute yield spread between two maturities."""
        long_df = self.get_treasury_yield(long, interval)
        short_df = self.get_treasury_yield(short, interval)
        if long_df.empty or short_df.empty:
            return pd.DataFrame()
        combined = pd.concat([long_df, short_df], axis=1).ffill().dropna()
        spread = combined[long] - combined[short]
        return spread.to_frame('spread')

    # ── Commodities ──────────────────────────────────────────────────

    def get_commodity(self, name: str, interval: str = 'monthly') -> pd.DataFrame:
        """Get commodity data. Supports: BRENT, COPPER, WHEAT, CORN, COTTON, COFFEE."""
        name_upper = name.upper()
        if name.lower() in COMMODITIES:
            name_upper = COMMODITIES[name.lower()]
        cache_name = f"commodity_{name_upper}_{interval}"
        cache = self._cache_path(cache_name)
        ttl = CACHE_TTL_MONTHLY if interval == 'monthly' else CACHE_TTL_DAILY

        if not self._is_stale(cache, ttl):
            return pd.read_parquet(cache)

        logger.info(f"Downloading commodity {name_upper} ({interval})")
        data = self._av_request({'function': name_upper, 'interval': interval})
        df = self._parse_av_timeseries(data, col_name=name_upper.lower())
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_all_commodities(self, interval: str = 'monthly') -> pd.DataFrame:
        """Get all commodities aligned into one DataFrame."""
        frames = []
        for name in COMMODITIES.values():
            df = self.get_commodity(name, interval)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index().ffill()

    # ── Economic Indicators ──────────────────────────────────────────

    def _get_economic(self, function: str, interval: str, col_name: str) -> pd.DataFrame:
        ttl = CACHE_TTL_DAILY if interval == 'daily' else CACHE_TTL_MONTHLY
        cache_name = f"econ_{function}_{interval}"
        cache = self._cache_path(cache_name)

        if not self._is_stale(cache, ttl):
            return pd.read_parquet(cache)

        logger.info(f"Downloading economic indicator {function}")
        params = {'function': function}
        if interval:
            params['interval'] = interval
        data = self._av_request(params)
        df = self._parse_av_timeseries(data, col_name=col_name)
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_fed_funds_rate(self) -> pd.DataFrame:
        return self._get_economic('FEDERAL_FUNDS_RATE', 'daily', 'fed_funds_rate')

    def get_cpi(self) -> pd.DataFrame:
        return self._get_economic('CPI', 'monthly', 'cpi')

    def get_inflation(self) -> pd.DataFrame:
        return self._get_economic('INFLATION', 'annual', 'inflation')

    def get_unemployment(self) -> pd.DataFrame:
        return self._get_economic('UNEMPLOYMENT', 'monthly', 'unemployment')

    def get_nonfarm_payroll(self) -> pd.DataFrame:
        return self._get_economic('NONFARM_PAYROLL', 'monthly', 'nonfarm_payroll')

    def get_retail_sales(self) -> pd.DataFrame:
        return self._get_economic('RETAIL_SALES', 'monthly', 'retail_sales')

    # ── Fundamentals ─────────────────────────────────────────────────

    def get_balance_sheet(self, symbol: str) -> dict:
        """Get balance sheet (annual reports). Returns raw JSON."""
        cache_name = f"fundamentals_bs_{symbol.upper()}"
        cache = self._cache_path(cache_name)

        if not self._is_stale(cache, CACHE_TTL_MONTHLY):
            return pd.read_parquet(cache).to_dict()

        logger.info(f"Downloading balance sheet for {symbol}")
        data = self._av_request({'function': 'BALANCE_SHEET', 'symbol': symbol.upper()})
        reports = data.get('annualReports', [])
        if reports:
            df = pd.DataFrame(reports)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.set_index('fiscalDateEnding').sort_index()
            df.index.name = 'timestamp'
            df.to_parquet(cache)
        return data

    def get_earnings(self, symbol: str) -> dict:
        """Get earnings (annual + quarterly). Returns raw JSON."""
        cache_name = f"fundamentals_earn_{symbol.upper()}"
        cache = self._cache_path(cache_name)

        if not self._is_stale(cache, CACHE_TTL_MONTHLY):
            return pd.read_parquet(cache).to_dict()

        logger.info(f"Downloading earnings for {symbol}")
        data = self._av_request({'function': 'EARNINGS', 'symbol': symbol.upper()})
        annual = data.get('annualEarnings', [])
        if annual:
            df = pd.DataFrame(annual)
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.set_index('fiscalDateEnding').sort_index()
            df.index.name = 'timestamp'
            df.to_parquet(cache)
        return data

    # ── Technical Indicators ─────────────────────────────────────────

    def get_indicator(self, symbol: str, indicator: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Generic technical indicator fetcher via AV API."""
        params = params or {}
        period = params.get('time_period', 20)
        series_type = params.get('series_type', 'close')

        cache_name = f"tech_{symbol.upper()}_{indicator.upper()}_{period}"
        cache = self._cache_path(cache_name)

        if not self._is_stale(cache):
            return pd.read_parquet(cache)

        logger.info(f"Downloading {indicator} for {symbol}")
        req_params = {
            'function': indicator.upper(),
            'symbol': symbol.upper(),
            'interval': params.get('interval', 'daily'),
            'series_type': series_type,
        }
        if indicator.upper() not in ('MACD',):
            req_params['time_period'] = str(period)

        data = self._av_request(req_params)
        # Find the technical analysis key
        ta_key = None
        for k in data:
            if 'Technical Analysis' in k:
                ta_key = k
                break
        if not ta_key:
            logger.warning(f"No TA data for {indicator} {symbol}: {list(data.keys())}")
            return pd.DataFrame()

        df = pd.DataFrame(data[ta_key]).T
        df.index = pd.to_datetime(df.index)
        df.index.name = 'timestamp'
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.sort_index().dropna(how='all')
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_sma(self, symbol: str, period: int = 20) -> pd.DataFrame:
        return self.get_indicator(symbol, 'SMA', {'time_period': period})

    def get_rsi(self, symbol: str, period: int = 14) -> pd.DataFrame:
        return self.get_indicator(symbol, 'RSI', {'time_period': period})

    def get_macd(self, symbol: str) -> pd.DataFrame:
        return self.get_indicator(symbol, 'MACD', {})

    def get_bbands(self, symbol: str, period: int = 20) -> pd.DataFrame:
        return self.get_indicator(symbol, 'BBANDS', {'time_period': period})

    def get_adx(self, symbol: str, period: int = 14) -> pd.DataFrame:
        return self.get_indicator(symbol, 'ADX', {'time_period': period})

    # ── Legacy methods (kept for compatibility) ──────────────────────

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        col_map = {}
        for c in df.columns:
            cl = str(c).lower().replace('.', '_')
            for target in ('open', 'high', 'low', 'close', 'volume'):
                if target in cl:
                    col_map[c] = target
                    break
        if col_map:
            df = df.rename(columns=col_map)
        keep = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        if keep:
            df = df[keep]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = 'timestamp'
        return df.sort_index().dropna(how='all')

    def get_daily(self, symbol: str, full: bool = True) -> pd.DataFrame:
        cache_name = f"{symbol.upper()}_daily_{'full' if full else 'compact'}"
        cache = self._cache_path(cache_name)
        if not self._is_stale(cache):
            return pd.read_parquet(cache)
        self._track_call()
        logger.info(f"Downloading {symbol} daily from Alpha Vantage")
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        df, _ = ts.get_daily(symbol=symbol, outputsize='full' if full else 'compact')
        df = self._normalize(df)
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_intraday(self, symbol: str, interval: str = '60min') -> pd.DataFrame:
        cache_name = f"{symbol.upper()}_intraday_{interval}"
        cache = self._cache_path(cache_name)
        if not self._is_stale(cache):
            return pd.read_parquet(cache)
        self._track_call()
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        df, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
        df = self._normalize(df)
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_forex(self, from_currency: str, to_currency: str) -> pd.DataFrame:
        pair = f"{from_currency}_{to_currency}".upper()
        cache = self._cache_path(f"forex_{pair}")
        if not self._is_stale(cache):
            return pd.read_parquet(cache)
        self._track_call()
        from alpha_vantage.foreignexchange import ForeignExchange
        fx = ForeignExchange(key=self.api_key, output_format='pandas')
        df, _ = fx.get_currency_exchange_daily(from_symbol=from_currency,
                                                to_symbol=to_currency, outputsize='full')
        df = self._normalize(df)
        if not df.empty:
            df.to_parquet(cache)
        return df

    def get_commodities(self, commodity: str) -> pd.DataFrame:
        """Legacy method - use get_commodity() instead."""
        return self.get_commodity(commodity, interval='daily')

    def get_technical(self, symbol: str, indicator: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Legacy method - use get_indicator() instead."""
        return self.get_indicator(symbol, indicator, params)
