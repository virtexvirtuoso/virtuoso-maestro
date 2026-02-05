"""
Funding Rate Data Downloader

Downloads historical funding rates from CEX and DEX exchanges.
Required for Funding Rate Arbitrage strategy.

Supported exchanges:
- Binance Futures
- BitMEX  
- Bybit
- OKX
- Drift (DEX)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FundingRateDownloader:
    """
    Download funding rate history from multiple exchanges.
    """
    
    ENDPOINTS = {
        'binance': {
            'base_url': 'https://fapi.binance.com',
            'funding_endpoint': '/fapi/v1/fundingRate',
            'interval_hours': 8,
        },
        'bitmex': {
            'base_url': 'https://www.bitmex.com',
            'funding_endpoint': '/api/v1/funding',
            'interval_hours': 8,
        },
        'bybit': {
            'base_url': 'https://api.bybit.com',
            'funding_endpoint': '/v5/market/funding/history',
            'interval_hours': 8,
        },
        'okx': {
            'base_url': 'https://www.okx.com',
            'funding_endpoint': '/api/v5/public/funding-rate-history',
            'interval_hours': 8,
        }
    }
    
    # Standard symbol mappings
    SYMBOL_MAP = {
        'BTC': {
            'binance': 'BTCUSDT',
            'bitmex': 'XBTUSD',
            'bybit': 'BTCUSDT',
            'okx': 'BTC-USDT-SWAP',
        },
        'ETH': {
            'binance': 'ETHUSDT',
            'bitmex': 'ETHUSD',
            'bybit': 'ETHUSDT',
            'okx': 'ETH-USDT-SWAP',
        },
        'SOL': {
            'binance': 'SOLUSDT',
            'bybit': 'SOLUSDT',
            'okx': 'SOL-USDT-SWAP',
        }
    }
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._own_session = False
    
    async def __aenter__(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._own_session = True
        return self
    
    async def __aexit__(self, *args):
        if self._own_session and self._session:
            await self._session.close()
    
    async def fetch_binance(self, symbol: str, 
                           start_time: datetime, 
                           end_time: datetime) -> pd.DataFrame:
        """
        Fetch Binance funding rate history.
        
        API: GET /fapi/v1/fundingRate
        Params: symbol, startTime, endTime, limit (max 1000)
        """
        config = self.ENDPOINTS['binance']
        url = f"{config['base_url']}{config['funding_endpoint']}"
        
        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        while current_start < end_ts:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': 1000
            }
            
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data:
                            break
                        all_data.extend(data)
                        # Move to next batch
                        current_start = data[-1]['fundingTime'] + 1
                    else:
                        logger.error(f"Binance API error: {resp.status}")
                        break
            except Exception as e:
                logger.error(f"Binance fetch error: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['funding_rate'] = df['fundingRate'].astype(float)
        df['exchange'] = 'binance'
        df['symbol'] = symbol
        
        return df[['timestamp', 'symbol', 'funding_rate', 'exchange']].set_index('timestamp')
    
    async def fetch_bitmex(self, symbol: str,
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """
        Fetch BitMEX funding rate history.
        
        API: GET /api/v1/funding
        Params: symbol, count, start, startTime, endTime
        """
        config = self.ENDPOINTS['bitmex']
        url = f"{config['base_url']}{config['funding_endpoint']}"
        
        all_data = []
        start = 0
        
        while True:
            params = {
                'symbol': symbol,
                'count': 500,
                'start': start,
                'startTime': start_time.isoformat(),
                'endTime': end_time.isoformat()
            }
            
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not data:
                            break
                        all_data.extend(data)
                        start += len(data)
                    else:
                        logger.error(f"BitMEX API error: {resp.status}")
                        break
            except Exception as e:
                logger.error(f"BitMEX fetch error: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['funding_rate'] = df['fundingRate'].astype(float)
        df['exchange'] = 'bitmex'
        df['symbol'] = symbol
        
        return df[['timestamp', 'symbol', 'funding_rate', 'exchange']].set_index('timestamp')
    
    async def fetch_bybit(self, symbol: str,
                         start_time: datetime,
                         end_time: datetime) -> pd.DataFrame:
        """
        Fetch Bybit funding rate history.
        
        API: GET /v5/market/funding/history
        """
        config = self.ENDPOINTS['bybit']
        url = f"{config['base_url']}{config['funding_endpoint']}"
        
        all_data = []
        current_end = int(end_time.timestamp() * 1000)
        start_ts = int(start_time.timestamp() * 1000)
        
        while current_end > start_ts:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'endTime': current_end,
                'limit': 200
            }
            
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        data = result.get('result', {}).get('list', [])
                        if not data:
                            break
                        all_data.extend(data)
                        # Move backwards
                        current_end = int(data[-1]['fundingRateTimestamp']) - 1
                    else:
                        logger.error(f"Bybit API error: {resp.status}")
                        break
            except Exception as e:
                logger.error(f"Bybit fetch error: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(int), unit='ms')
        df['funding_rate'] = df['fundingRate'].astype(float)
        df['exchange'] = 'bybit'
        df['symbol'] = symbol
        
        return df[['timestamp', 'symbol', 'funding_rate', 'exchange']].set_index('timestamp')
    
    async def fetch_funding_rates(self, 
                                  base_symbol: str,
                                  exchanges: List[str],
                                  start_time: datetime,
                                  end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch funding rates from multiple exchanges.
        
        Args:
            base_symbol: Base symbol (e.g., 'BTC', 'ETH')
            exchanges: List of exchanges to fetch from
            start_time: Start datetime
            end_time: End datetime
        
        Returns:
            Dictionary mapping exchange to funding rate DataFrame
        """
        results = {}
        
        fetch_methods = {
            'binance': self.fetch_binance,
            'bitmex': self.fetch_bitmex,
            'bybit': self.fetch_bybit,
        }
        
        tasks = []
        exchange_names = []
        
        for exchange in exchanges:
            if exchange not in fetch_methods:
                logger.warning(f"Unsupported exchange: {exchange}")
                continue
            
            symbol = self.SYMBOL_MAP.get(base_symbol, {}).get(exchange)
            if not symbol:
                logger.warning(f"No symbol mapping for {base_symbol} on {exchange}")
                continue
            
            tasks.append(fetch_methods[exchange](symbol, start_time, end_time))
            exchange_names.append(exchange)
        
        if tasks:
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for exchange, data in zip(exchange_names, fetched):
                if isinstance(data, Exception):
                    logger.error(f"Error fetching {exchange}: {data}")
                elif not data.empty:
                    results[exchange] = data
        
        return results
    
    def calculate_funding_differential(self, 
                                       rates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate funding rate differential between exchanges.
        
        Args:
            rates: Dictionary of funding rate DataFrames
        
        Returns:
            DataFrame with funding differentials for arbitrage
        """
        if len(rates) < 2:
            return pd.DataFrame()
        
        # Combine all funding rates
        combined = pd.DataFrame()
        for exchange, df in rates.items():
            combined[f'{exchange}_rate'] = df['funding_rate']
        
        # Resample to common frequency and forward fill
        combined = combined.resample('8h').last().ffill()
        
        # Calculate all pairwise differentials
        exchanges = list(rates.keys())
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                col1 = f'{ex1}_rate'
                col2 = f'{ex2}_rate'
                diff_col = f'{ex1}_{ex2}_diff'
                combined[diff_col] = combined[col1] - combined[col2]
        
        return combined


# Synchronous wrapper for non-async contexts
def download_funding_rates(base_symbol: str,
                          exchanges: List[str],
                          start_time: datetime,
                          end_time: datetime) -> Dict[str, pd.DataFrame]:
    """
    Synchronous wrapper for funding rate download.
    """
    async def _download():
        async with FundingRateDownloader() as downloader:
            return await downloader.fetch_funding_rates(
                base_symbol, exchanges, start_time, end_time
            )
    
    return asyncio.run(_download())


if __name__ == "__main__":
    # Test download
    from datetime import datetime, timedelta
    
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    
    print("Downloading BTC funding rates...")
    rates = download_funding_rates('BTC', ['binance', 'bybit'], start, end)
    
    for exchange, df in rates.items():
        print(f"\n{exchange}:")
        print(df.head())
        print(f"Total records: {len(df)}")
