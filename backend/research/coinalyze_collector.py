"""
Coinalyze Data Collector

Free API for multi-year derivatives data:
- Open Interest (daily: unlimited history)
- Funding Rates
- Liquidations
- Long/Short Ratio

Get free API key at: https://coinalyze.net
"""

import os
import requests
import pandas as pd
from pathlib import Path
from time import sleep
from typing import Optional, List, Dict
from datetime import datetime


class CoinalyzeClient:
    """
    Free Coinalyze API client for derivatives data.
    
    Get free API key at: https://coinalyze.net
    Rate limit: 40 calls/minute
    """
    
    BASE_URL = "https://api.coinalyze.net/v1"
    
    # Exchange codes for symbol formatting
    EXCHANGE_CODES = {
        'binance': 'A',
        'bybit': '6', 
        'okx': '5',
        'bitmex': '2',
        'dydx': '7',
        'bitfinex': 'B',
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("COINALYZE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Coinalyze API key required. "
                "Get free key at https://coinalyze.net and set COINALYZE_API_KEY"
            )
        self._call_count = 0
    
    def _format_symbol(self, base: str, quote: str = "USDT", 
                       exchange: str = "binance") -> str:
        """Format symbol for Coinalyze API."""
        code = self.EXCHANGE_CODES.get(exchange, 'A')
        return f"{base}{quote}_PERP.{code}"
    
    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make rate-limited API request."""
        params = params or {}
        params['api_key'] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            
            if resp.status_code == 429:
                retry_after = int(resp.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after}s...")
                sleep(retry_after)
                return self._request(endpoint, params)
            
            resp.raise_for_status()
            return resp.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            return []
    
    def get_open_interest_history(
        self,
        symbols: str,
        interval: str = "daily",
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get open interest history.
        
        Parameters:
        -----------
        symbols : str
            Comma-separated symbols (e.g., "BTCUSDT_PERP.A")
        interval : str
            1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 12hour, daily
        from_ts : int
            Start timestamp (seconds)
        to_ts : int
            End timestamp (seconds)
            
        Returns:
        --------
        DataFrame with OI OHLC data
        
        Note: Daily data has unlimited history (multi-year!)
        """
        params = {
            "symbols": symbols,
            "interval": interval
        }
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        data = self._request("open-interest-history", params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            symbol = item.get('symbol', symbols)
            timestamps = item.get('t', [])
            opens = item.get('o', [])
            highs = item.get('h', [])
            lows = item.get('l', [])
            closes = item.get('c', [])
            
            for i, ts in enumerate(timestamps):
                records.append({
                    'timestamp': pd.to_datetime(ts, unit='s'),
                    'symbol': symbol,
                    'oi_open': opens[i] if i < len(opens) else None,
                    'oi_high': highs[i] if i < len(highs) else None,
                    'oi_low': lows[i] if i < len(lows) else None,
                    'oi_close': closes[i] if i < len(closes) else None,
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_funding_rate_history(
        self,
        symbols: str,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> pd.DataFrame:
        """Get funding rate history."""
        params = {"symbols": symbols}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        data = self._request("funding-rate-history", params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            symbol = item.get('symbol', symbols)
            for i, ts in enumerate(item.get('t', [])):
                records.append({
                    'timestamp': pd.to_datetime(ts, unit='s'),
                    'symbol': symbol,
                    'funding_rate': item['r'][i] if i < len(item.get('r', [])) else None,
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_liquidation_history(
        self,
        symbols: str,
        interval: str = "daily",
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> pd.DataFrame:
        """Get liquidation history."""
        params = {
            "symbols": symbols,
            "interval": interval
        }
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        data = self._request("liquidation-history", params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            symbol = item.get('symbol', symbols)
            for i, ts in enumerate(item.get('t', [])):
                records.append({
                    'timestamp': pd.to_datetime(ts, unit='s'),
                    'symbol': symbol,
                    'long_liquidations': item['l'][i] if i < len(item.get('l', [])) else None,
                    'short_liquidations': item['s'][i] if i < len(item.get('s', [])) else None,
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_long_short_ratio_history(
        self,
        symbols: str,
        interval: str = "daily",
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> pd.DataFrame:
        """Get long/short ratio history."""
        params = {
            "symbols": symbols,
            "interval": interval
        }
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        data = self._request("long-short-ratio-history", params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for item in data:
            symbol = item.get('symbol', symbols)
            for i, ts in enumerate(item.get('t', [])):
                records.append({
                    'timestamp': pd.to_datetime(ts, unit='s'),
                    'symbol': symbol,
                    'long_ratio': item['l'][i] if i < len(item.get('l', [])) else None,
                    'short_ratio': item['s'][i] if i < len(item.get('s', [])) else None,
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df


def collect_all_derivatives_data(
    api_key: str,
    symbols: List[str] = None,
    output_dir: str = "./derivatives_data"
) -> Dict[str, pd.DataFrame]:
    """
    Collect all derivatives data from Coinalyze.
    
    Parameters:
    -----------
    api_key : str
        Coinalyze API key
    symbols : list
        Base symbols (BTC, ETH, SOL, etc.)
    output_dir : str
        Directory to save CSV files
        
    Returns:
    --------
    Dict of DataFrames
    """
    symbols = symbols or ["BTC", "ETH", "SOL"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    client = CoinalyzeClient(api_key)
    all_data = {}
    
    print("="*60)
    print("COINALYZE DATA COLLECTION")
    print("="*60)
    
    for symbol in symbols:
        coinalyze_symbol = f"{symbol}USDT_PERP.A"
        print(f"\n📊 Collecting {symbol}...")
        
        # Open Interest (daily - unlimited history)
        print("  → Open Interest...")
        try:
            oi_df = client.get_open_interest_history(coinalyze_symbol, "daily")
            if not oi_df.empty:
                all_data[f"{symbol}_oi"] = oi_df
                oi_df.to_csv(output_path / f"{symbol}_oi_daily.csv", index=False)
                print(f"    ✓ {len(oi_df)} records: {oi_df['timestamp'].min().date()} → {oi_df['timestamp'].max().date()}")
            sleep(1.5)  # Rate limit
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Funding Rates
        print("  → Funding Rates...")
        try:
            funding_df = client.get_funding_rate_history(coinalyze_symbol)
            if not funding_df.empty:
                all_data[f"{symbol}_funding"] = funding_df
                funding_df.to_csv(output_path / f"{symbol}_funding.csv", index=False)
                print(f"    ✓ {len(funding_df)} records: {funding_df['timestamp'].min().date()} → {funding_df['timestamp'].max().date()}")
            sleep(1.5)
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Liquidations
        print("  → Liquidations...")
        try:
            liq_df = client.get_liquidation_history(coinalyze_symbol, "daily")
            if not liq_df.empty:
                all_data[f"{symbol}_liquidations"] = liq_df
                liq_df.to_csv(output_path / f"{symbol}_liquidations_daily.csv", index=False)
                print(f"    ✓ {len(liq_df)} records: {liq_df['timestamp'].min().date()} → {liq_df['timestamp'].max().date()}")
            sleep(1.5)
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Long/Short Ratio
        print("  → Long/Short Ratio...")
        try:
            lsr_df = client.get_long_short_ratio_history(coinalyze_symbol, "daily")
            if not lsr_df.empty:
                all_data[f"{symbol}_lsr"] = lsr_df
                lsr_df.to_csv(output_path / f"{symbol}_lsr_daily.csv", index=False)
                print(f"    ✓ {len(lsr_df)} records: {lsr_df['timestamp'].min().date()} → {lsr_df['timestamp'].max().date()}")
            sleep(1.5)
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    
    total_records = 0
    for name, df in all_data.items():
        if df is not None and not df.empty:
            total_records += len(df)
            print(f"  {name}: {len(df)} records")
    
    print(f"\nTotal: {total_records} records")
    print(f"Saved to: {output_path}/")
    
    return all_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect derivatives data from Coinalyze")
    parser.add_argument("--api-key", help="Coinalyze API key (or set COINALYZE_API_KEY)")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--output", default="./derivatives_data")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("COINALYZE_API_KEY")
    
    if not api_key:
        print("❌ No API key provided!")
        print("")
        print("Get your FREE API key:")
        print("  1. Go to https://coinalyze.net")
        print("  2. Sign up (free)")
        print("  3. Go to Account → API")
        print("  4. Copy your API key")
        print("")
        print("Then run:")
        print(f"  export COINALYZE_API_KEY=your_key_here")
        print(f"  python {__file__} --symbols BTC ETH SOL")
        exit(1)
    
    collect_all_derivatives_data(api_key, args.symbols, args.output)
