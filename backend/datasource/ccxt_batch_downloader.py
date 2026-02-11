"""
Generic CCXT-based batch downloader for any supported exchange.
Supports: Binance, Bybit, OKX, Gate, KuCoin, MEXC, and more.
"""
import logging
import time
from threading import Thread
from typing import List, Optional
from datetime import datetime, timedelta

import ccxt
import pytz

from datasource.providers import DataSourceProviders
from logger.logger_builder import LoggerBuilder
from schema.bucketed_trade_data import BucketedTradeData
from storage.storage_layer import StorageLayer


# Map provider enum to CCXT exchange class
EXCHANGE_MAP = {
    DataSourceProviders.BINANCE: 'binance',
    DataSourceProviders.BYBIT: 'bybit',
    DataSourceProviders.OKX: 'okx',
    DataSourceProviders.GATE: 'gate',
    DataSourceProviders.KUCOIN: 'kucoin',
    DataSourceProviders.MEXC: 'mexc',
    DataSourceProviders.BITMEX: 'bitmex',
}

# Timeframe mapping (Maestro format -> CCXT format)
TIMEFRAME_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '12h': '12h',
    '1d': '1d',
    '1w': '1w',
}


class CCXTBatchDownloader:
    """
    Generic batch downloader using CCXT.
    Can download OHLCV data from any CCXT-supported exchange.
    """

    def __init__(self, provider: DataSourceProviders, storage: StorageLayer, 
                 logger: logging.Logger = None, rate_limit: float = 1.0):
        self.provider = provider
        self.storage = storage
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()
        self.rate_limit = rate_limit
        
        # Initialize CCXT exchange
        exchange_id = EXCHANGE_MAP.get(provider)
        if not exchange_id:
            raise ValueError(f"Unsupported provider: {provider}")
        
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        self.logger.info(f"Initialized {exchange_id} exchange")

    def register(self, symbol: str, bin_size: str):
        """Register a symbol/timeframe pair for downloading."""
        self.logger.info(f'Registering {symbol} with bin size {bin_size}')
        self.storage.add(symbol=symbol, bin_size=bin_size, provider=self.provider)
        return self

    def start(self):
        """Start downloading all registered symbol/timeframe pairs."""
        threads: List[Thread] = []
        
        for bucketed_trade_data in self.storage.bucketed_trade_data_list(provider=self.provider):
            t_name = f'ccxt-{self.provider.value}-{bucketed_trade_data.symbol}-{bucketed_trade_data.bin_size}'
            t = CCXTBatchDownloaderWorker(
                name=t_name,
                exchange=self.exchange,
                provider=self.provider,
                bucketed_trade_data=bucketed_trade_data,
                storage=self.storage.spawn(),
                logger=LoggerBuilder(name=t_name).build(),
                rate_limit=self.rate_limit,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()


class CCXTBatchDownloaderWorker(Thread):
    """Worker thread for downloading OHLCV data from a single symbol/timeframe."""

    def __init__(self, name: str, exchange: ccxt.Exchange, provider: DataSourceProviders,
                 bucketed_trade_data: BucketedTradeData, storage: StorageLayer,
                 logger: logging.Logger = None, rate_limit: float = 1.0):
        super().__init__(name=name)
        self.exchange = exchange
        self.provider = provider
        self.bucketed_trade_data = bucketed_trade_data
        self.storage = storage
        self.logger = logger if logger else LoggerBuilder(name=self.__class__.__name__).build()
        self.rate_limit = rate_limit

    def run(self):
        self.logger.info(f'Downloading data for {self.bucketed_trade_data}')
        
        # Convert symbol format (btcusdt -> BTC/USDT)
        symbol = self._normalize_symbol(self.bucketed_trade_data.symbol)
        timeframe = TIMEFRAME_MAP.get(self.bucketed_trade_data.bin_size, self.bucketed_trade_data.bin_size)
        
        sleep_time = self.rate_limit
        consecutive_success = 0
        
        while True:
            try:
                time.sleep(sleep_time)
                
                # Fetch OHLCV data
                since = self.bucketed_trade_data.start
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                
                if not ohlcv:
                    self.logger.info(f'No new data, sleeping for 60s')
                    time.sleep(60)
                    continue
                
                # Convert to standard format
                results = self._convert_ohlcv(ohlcv)
                
                self.logger.info(f'Downloaded {len(results)} candles')
                
                # Save to storage
                self.bucketed_trade_data = self.storage.save(
                    provider=self.provider,
                    symbol=self.bucketed_trade_data.symbol,
                    bin_size=self.bucketed_trade_data.bin_size,
                    results=results,
                    offset=results[-1]['open_time']
                )
                self.bucketed_trade_data.start = results[-1]['open_time']
                
                # Adjust rate limiting
                consecutive_success += 1
                if consecutive_success >= 3:
                    sleep_time = max(0.5, sleep_time * 0.8)
                    consecutive_success = 0
                
                # Check if we're caught up
                if len(ohlcv) < 1000:
                    self.logger.info(f'Caught up to current time, sleeping for 60s')
                    time.sleep(60)
                    
            except ccxt.RateLimitExceeded:
                sleep_time = min(30, sleep_time * 2)
                consecutive_success = 0
                self.logger.warning(f'Rate limited, slowing to {sleep_time}s')
                
            except ccxt.NetworkError as e:
                self.logger.warning(f'Network error: {e}, retrying in 10s')
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f'Fatal error: {e}')
                import traceback
                self.logger.error(traceback.format_exc())
                return -1

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol format: btcusdt -> BTC/USDT"""
        symbol = symbol.upper()
        
        # Common quote currencies
        quotes = ['USDT', 'USDC', 'USD', 'BTC', 'ETH', 'BNB', 'BUSD']
        
        for quote in quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f'{base}/{quote}'
        
        # Fallback: assume USDT
        return f'{symbol}/USDT'

    def _convert_ohlcv(self, ohlcv: list) -> list:
        """Convert CCXT OHLCV format to Maestro format."""
        results = []
        for candle in ohlcv:
            timestamp_ms = candle[0]
            results.append({
                'timestamp': datetime.fromtimestamp(timestamp_ms / 1000).astimezone(pytz.UTC),
                'open_time': timestamp_ms,
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
            })
        return results


def download_historical(provider: DataSourceProviders, symbol: str, timeframe: str,
                        days: int = 365, output_file: Optional[str] = None) -> list:
    """
    Standalone function to download historical OHLCV data.
    
    Args:
        provider: Exchange provider
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        days: Number of days of history to fetch
        output_file: Optional CSV file to save data
        
    Returns:
        List of OHLCV candles
    """
    exchange_id = EXCHANGE_MAP.get(provider)
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
            
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        
        if len(ohlcv) < 1000:
            break
            
        time.sleep(exchange.rateLimit / 1000)
    
    if output_file:
        import pandas as pd
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.to_csv(output_file, index=False)
        
    return all_data
