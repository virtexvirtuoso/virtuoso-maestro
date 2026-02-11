"""
Data Collectors for Derivatives Metrics

Collects historical data that's missing for backtesting:
- Funding rates
- Open interest
- Liquidations (when available)

Stores to CSV for backtesting, optionally to InfluxDB for real-time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import time
import json

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("Warning: ccxt not installed")


class FundingRateCollector:
    """
    Collect historical funding rates from exchanges.
    
    Binance funding: every 8 hours (00:00, 08:00, 16:00 UTC)
    Bybit funding: every 8 hours
    """
    
    def __init__(self, exchange: str = 'binance', output_dir: str = './funding_data'):
        self.exchange_name = exchange
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None
        
    @property
    def exchange(self):
        if self._exchange is None:
            if not HAS_CCXT:
                raise ImportError("ccxt required")
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        return self._exchange
    
    def fetch_funding_history(self, symbol: str, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              limit: int = 1000) -> pd.DataFrame:
        """
        Fetch funding rate history for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date (YYYY-MM-DD), default 1 year ago
            end_date: End date (YYYY-MM-DD), default now
            limit: Max records per request
            
        Returns:
            DataFrame with columns: timestamp, symbol, funding_rate, mark_price
        """
        # Parse dates
        if end_date is None:
            end_dt = datetime.utcnow()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date is None:
            start_dt = end_dt - timedelta(days=365)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Convert symbol format
        binance_symbol = symbol.replace('/', '')
        
        all_records = []
        current_end = int(end_dt.timestamp() * 1000)
        start_ts = int(start_dt.timestamp() * 1000)
        
        print(f"Fetching funding rates for {symbol} from {start_dt.date()} to {end_dt.date()}")
        
        while current_end > start_ts:
            try:
                if self.exchange_name == 'binance':
                    # Binance futures funding rate history
                    params = {
                        'symbol': binance_symbol,
                        'limit': limit,
                        'endTime': current_end
                    }
                    if start_ts:
                        params['startTime'] = start_ts
                    
                    rates = self.exchange.fapiPublicGetFundingRate(params)
                    
                    if not rates:
                        break
                    
                    for r in rates:
                        all_records.append({
                            'timestamp': pd.to_datetime(int(r['fundingTime']), unit='ms'),
                            'symbol': symbol,
                            'funding_rate': float(r['fundingRate']),
                            'mark_price': float(r.get('markPrice', 0))
                        })
                    
                    # Move window back
                    oldest = min(int(r['fundingTime']) for r in rates)
                    if oldest >= current_end:
                        break
                    current_end = oldest - 1
                    
                elif self.exchange_name == 'bybit':
                    # Bybit funding rate history
                    rates = self.exchange.fetch_funding_rate_history(symbol, start_ts, limit)
                    
                    if not rates:
                        break
                        
                    for r in rates:
                        all_records.append({
                            'timestamp': pd.to_datetime(r['timestamp'], unit='ms'),
                            'symbol': symbol,
                            'funding_rate': float(r['fundingRate']),
                            'mark_price': float(r.get('markPrice', 0))
                        })
                    
                    oldest = min(r['timestamp'] for r in rates)
                    current_end = oldest - 1
                
                print(f"  Fetched {len(all_records)} records so far...")
                time.sleep(0.1)  # Rate limit
                
            except Exception as e:
                print(f"Error fetching funding rates: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=['timestamp', 'symbol'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Total: {len(df)} funding rate records for {symbol}")
        return df
    
    def fetch_multiple_symbols(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Fetch funding rates for multiple symbols."""
        all_data = []
        
        for symbol in symbols:
            df = self.fetch_funding_history(symbol, **kwargs)
            if not df.empty:
                all_data.append(df)
            time.sleep(0.5)  # Be nice to API
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def save(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save funding data to CSV."""
        if df.empty:
            print("No data to save")
            return
        
        if filename is None:
            symbols = df['symbol'].unique()
            symbol_str = '_'.join(s.replace('/', '') for s in symbols[:3])
            filename = f"funding_rates_{symbol_str}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return path
    
    def load(self, filename: str) -> pd.DataFrame:
        """Load funding data from CSV."""
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return df


class OpenInterestCollector:
    """
    Collect historical open interest from exchanges.
    
    Note: Historical OI is limited on most exchanges.
    Binance provides current OI and some history via /futures/data/openInterestHist
    """
    
    def __init__(self, exchange: str = 'binance', output_dir: str = './oi_data'):
        self.exchange_name = exchange
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None
        
    @property
    def exchange(self):
        if self._exchange is None:
            if not HAS_CCXT:
                raise ImportError("ccxt required")
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        return self._exchange
    
    def fetch_oi_history(self, symbol: str,
                         period: str = '1h',
                         limit: int = 500) -> pd.DataFrame:
        """
        Fetch open interest history.
        
        Args:
            symbol: Trading pair
            period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
            limit: Max records (Binance max 500)
        """
        binance_symbol = symbol.replace('/', '')
        
        try:
            if self.exchange_name == 'binance':
                # Binance open interest history
                params = {
                    'symbol': binance_symbol,
                    'period': period,
                    'limit': limit
                }
                
                data = self.exchange.fapiDataGetOpenInterestHist(params)
                
                if not data:
                    return pd.DataFrame()
                
                records = []
                for r in data:
                    records.append({
                        'timestamp': pd.to_datetime(int(r['timestamp']), unit='ms'),
                        'symbol': symbol,
                        'open_interest': float(r['sumOpenInterest']),
                        'open_interest_value': float(r['sumOpenInterestValue'])
                    })
                
                df = pd.DataFrame(records)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                print(f"Fetched {len(df)} OI records for {symbol}")
                return df
                
        except Exception as e:
            print(f"Error fetching OI: {e}")
            return pd.DataFrame()
    
    def fetch_current_oi(self, symbol: str) -> dict:
        """Fetch current open interest (real-time)."""
        try:
            binance_symbol = symbol.replace('/', '')
            oi = self.exchange.fapiPublicGetOpenInterest({'symbol': binance_symbol})
            return {
                'symbol': symbol,
                'open_interest': float(oi['openInterest']),
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def save(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save OI data to CSV."""
        if df.empty:
            return
        
        if filename is None:
            symbol = df['symbol'].iloc[0].replace('/', '')
            filename = f"oi_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return path


class SpotPriceCollector:
    """
    Collect spot prices to enable basis proxy calculation.
    """
    
    def __init__(self, exchange: str = 'binance', output_dir: str = './spot_data'):
        self.exchange_name = exchange
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None
        
    @property
    def exchange(self):
        if self._exchange is None:
            if not HAS_CCXT:
                raise ImportError("ccxt required")
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}  # Spot market
            })
        return self._exchange
    
    def fetch_ohlcv(self, symbol: str, 
                    timeframe: str = '1h',
                    start_date: Optional[str] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """
        Fetch spot OHLCV data.
        """
        if start_date:
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            since = None
        
        all_data = []
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Move forward
                last_ts = ohlcv[-1][0]
                if since and last_ts <= since:
                    break
                since = last_ts + 1
                
                if len(ohlcv) < limit:
                    break
                
                print(f"  Fetched {len(all_data)} spot candles...")
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        print(f"Total: {len(df)} spot candles for {symbol}")
        return df
    
    def save(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save spot data to CSV."""
        if df.empty:
            return
        
        if filename is None:
            symbol = df['symbol'].iloc[0].replace('/', '')
            filename = f"spot_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return path


class LongShortRatioCollector:
    """
    Collect long/short ratio data from exchanges.
    """
    
    def __init__(self, exchange: str = 'binance', output_dir: str = './lsr_data'):
        self.exchange_name = exchange
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None
        
    @property
    def exchange(self):
        if self._exchange is None:
            if not HAS_CCXT:
                raise ImportError("ccxt required")
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        return self._exchange
    
    def fetch_global_lsr(self, symbol: str, period: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Fetch global long/short account ratio.
        """
        binance_symbol = symbol.replace('/', '')
        
        try:
            params = {
                'symbol': binance_symbol,
                'period': period,
                'limit': limit
            }
            
            data = self.exchange.fapiDataGetGlobalLongShortAccountRatio(params)
            
            if not data:
                return pd.DataFrame()
            
            records = []
            for r in data:
                records.append({
                    'timestamp': pd.to_datetime(int(r['timestamp']), unit='ms'),
                    'symbol': symbol,
                    'long_account': float(r['longAccount']),
                    'short_account': float(r['shortAccount']),
                    'long_short_ratio': float(r['longShortRatio'])
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Fetched {len(df)} L/S ratio records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def fetch_top_trader_lsr(self, symbol: str, period: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Fetch top trader long/short ratio (positions).
        """
        binance_symbol = symbol.replace('/', '')
        
        try:
            params = {
                'symbol': binance_symbol,
                'period': period,
                'limit': limit
            }
            
            data = self.exchange.fapiDataGetTopLongShortPositionRatio(params)
            
            if not data:
                return pd.DataFrame()
            
            records = []
            for r in data:
                records.append({
                    'timestamp': pd.to_datetime(int(r['timestamp']), unit='ms'),
                    'symbol': symbol,
                    'long_position': float(r['longAccount']),
                    'short_position': float(r['shortAccount']),
                    'long_short_ratio': float(r['longShortRatio'])
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Fetched {len(df)} top trader L/S records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def save(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save L/S ratio data to CSV."""
        if df.empty:
            return
        
        if filename is None:
            symbol = df['symbol'].iloc[0].replace('/', '')
            filename = f"lsr_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return path


# =============================================================================
# CLI Interface
# =============================================================================

def collect_all_for_symbol(symbol: str, output_dir: str = './collected_data'):
    """
    Collect all available data for a symbol.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Collecting data for {symbol}")
    print(f"{'='*60}\n")
    
    # Funding rates
    print("1. Funding Rates...")
    fc = FundingRateCollector(output_dir=str(output_path / 'funding'))
    funding_df = fc.fetch_funding_history(symbol)
    if not funding_df.empty:
        fc.save(funding_df)
    
    # Open Interest
    print("\n2. Open Interest...")
    oc = OpenInterestCollector(output_dir=str(output_path / 'oi'))
    oi_df = oc.fetch_oi_history(symbol, period='1h')
    if not oi_df.empty:
        oc.save(oi_df)
    
    # Spot prices
    print("\n3. Spot Prices...")
    sc = SpotPriceCollector(output_dir=str(output_path / 'spot'))
    spot_df = sc.fetch_ohlcv(symbol, timeframe='1h')
    if not spot_df.empty:
        sc.save(spot_df)
    
    # Long/Short Ratio
    print("\n4. Long/Short Ratio...")
    lc = LongShortRatioCollector(output_dir=str(output_path / 'lsr'))
    lsr_df = lc.fetch_global_lsr(symbol, period='1h')
    if not lsr_df.empty:
        lc.save(lsr_df)
    
    print(f"\n{'='*60}")
    print(f"Collection complete for {symbol}")
    print(f"Data saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return {
        'funding': funding_df,
        'oi': oi_df,
        'spot': spot_df,
        'lsr': lsr_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect derivatives data")
    parser.add_argument('--symbol', default='BTC/USDT', help='Symbol to collect')
    parser.add_argument('--symbols', nargs='+', help='Multiple symbols')
    parser.add_argument('--output', default='./collected_data', help='Output directory')
    parser.add_argument('--funding-only', action='store_true', help='Only collect funding rates')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    symbols = args.symbols or [args.symbol]
    
    for symbol in symbols:
        if args.funding_only:
            fc = FundingRateCollector(output_dir=args.output)
            df = fc.fetch_funding_history(symbol, start_date=args.start, end_date=args.end)
            if not df.empty:
                fc.save(df)
        else:
            collect_all_for_symbol(symbol, args.output)
