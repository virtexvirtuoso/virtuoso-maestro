"""
Test data loading for validation battery
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from fredapi import Fred

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
FRED_API_KEY = os.getenv('FRED_API_KEY')

def test_crypto_loading():
    print("Testing crypto data loading...")
    
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            print(f"Downloading {name} ({ticker})...")
            data = yf.download(ticker, start="2023-01-01", end="2023-03-01", progress=False)
            print(f"Raw data shape: {data.shape}")
            print(f"Raw columns: {data.columns}")
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                # Get first level of MultiIndex (price type)
                data.columns = data.columns.get_level_values(0)
                print(f"Fixed columns: {data.columns}")
            
            # Create clean dataframe
            df = pd.DataFrame({
                'open': data['Open'],
                'high': data['High'], 
                'low': data['Low'],
                'close': data['Close'],
                'volume': data['Volume']
            }, index=data.index)
            
            crypto_data[name] = df
            print(f"  {name}: SUCCESS - {len(df)} days")
            print(f"  Sample close prices: {df['close'].head(3).values}")
            
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    return crypto_data

def test_fred_loading():
    print("\nTesting FRED data loading...")
    
    fred = Fred(api_key=FRED_API_KEY)
    try:
        m2_data = fred.get_series("M2SL", start="2023-01-01", end="2023-03-01")
        print(f"M2 data loaded: {len(m2_data)} points")
        print(f"Sample M2 values: {m2_data.head(3)}")
        return True
    except Exception as e:
        print(f"FRED loading failed: {e}")
        return False

if __name__ == "__main__":
    crypto_data = test_crypto_loading()
    fred_success = test_fred_loading()
    
    print(f"\nSummary:")
    print(f"Crypto assets loaded: {len(crypto_data)}")
    print(f"FRED working: {fred_success}")