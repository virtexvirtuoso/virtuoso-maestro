"""
MEGA STRATEGY V3.1-H2 — VALIDATION PART 2 (DEBUG VERSION)
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import traceback

print("🐛 DEBUG VERSION - Starting imports...")

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

try:
    from fredapi import Fred
    print("✓ fredapi imported")
except ImportError as e:
    print(f"✗ fredapi import error: {e}")
    sys.exit(1)

try:
    import yfinance as yf
    print("✓ yfinance imported")
except ImportError as e:
    print(f"✗ yfinance import error: {e}")
    sys.exit(1)

try:
    from strategies.composite.mega_strategy_v31 import run_full_strategy
    print("✓ mega_strategy_v31 imported")
except ImportError as e:
    print(f"✗ mega_strategy_v31 import error: {e}")
    sys.exit(1)

try:
    from strategies.composite.mega_strategy_v3 import compute_confluence, ASSET_CONFIGS
    print("✓ mega_strategy_v3 imported")
except ImportError as e:
    print(f"✗ mega_strategy_v3 import error: {e}")
    sys.exit(1)

print("🚀 All imports successful!")

# Configuration
RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD"}  # Start with just BTC for debugging
CROSS_ASSET_MAP = {"gold": "GLD", "bonds": "TLT"}  # Minimal set
FRED_SERIES = {"m2": "M2SL", "yield_curve": "T10Y2Y"}  # Minimal set
WEIGHTS = {"BTC": 1.0}  # Just BTC for now

LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}
TX_COST = 0.001
FRED_API_KEY = os.getenv('FRED_API_KEY')

def load_data():
    """Load minimal data for debugging."""
    print("\n📊 Loading data (2020-2024)...")
    
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            print(f"  Loading {name} ({ticker})...")
            data = yf.download(ticker, start="2020-01-01", end="2024-01-01", progress=False)
            
            if len(data) > 100:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                df = pd.DataFrame({
                    'open': data['Open'], 'high': data['High'], 'low': data['Low'],
                    'close': data['Close'], 'volume': data['Volume']
                }, index=data.index)
                crypto_data[name] = df
                print(f"    ✓ {name}: {len(df)} days")
        except Exception as e:
            print(f"    ✗ {name}: ERROR - {e}")
    
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            print(f"  Loading {col} ({ticker})...")
            data = yf.download(ticker, start="2020-01-01", end="2024-01-01", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            cross_asset_data[col] = data["Close"]
            print(f"    ✓ {col}: {len(data)} days")
        except Exception as e:
            print(f"    ✗ {col}: ERROR - {e}")
    
    fred = Fred(api_key=FRED_API_KEY)
    macro_data = pd.DataFrame()
    for col, series in FRED_SERIES.items():
        try:
            print(f"  Loading {col} ({series})...")
            data = fred.get_series(series, start="2018-01-01", end="2024-01-01")
            macro_data[col] = data
            print(f"    ✓ {col}: {len(data)} points")
        except Exception as e:
            print(f"    ✗ {col}: ERROR - {e}")
    
    if len(macro_data) > 0:
        macro_data = macro_data.resample('D').ffill()
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Crypto: {len(crypto_data)} assets")
    print(f"   Cross-asset: {len(cross_asset_data.columns)} assets")  
    print(f"   Macro: {len(macro_data.columns)} series")
    
    return crypto_data, cross_asset_data, macro_data

def test_simple_strategy_run(crypto_data, macro_data, cross_asset_data):
    """Test if we can run the strategy at all."""
    print("\n🧪 Testing basic strategy execution...")
    
    try:
        portfolio_df, _ = run_full_strategy(
            crypto_data, macro_data, cross_asset_data,
            leverage_map=LEVERAGE_MAP, weights=WEIGHTS, tx_cost=TX_COST
        )
        
        returns = portfolio_df["daily_pnl"]
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        print(f"  ✅ Strategy executed successfully!")
        print(f"     Return periods: {len(returns)}")
        print(f"     Sharpe ratio: {sharpe:.3f}")
        print(f"     Total return: {(1 + returns).prod() - 1:.1%}")
        
        return True, returns
        
    except Exception as e:
        print(f"  ❌ Strategy execution failed!")
        print(f"     Error: {e}")
        traceback.print_exc()
        return False, None

def main():
    print("🔬 VALIDATION PART 2 - DEBUG VERSION")
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Load data
        crypto_data, cross_asset_data, macro_data = load_data()
        
        if not crypto_data:
            print("❌ No crypto data loaded!")
            return
        
        if len(macro_data) == 0:
            print("❌ No macro data loaded!")
            return
            
        # Test basic strategy
        success, returns = test_simple_strategy_run(crypto_data, macro_data, cross_asset_data)
        
        if not success:
            print("❌ Basic strategy test failed!")
            return
            
        print("\n🎉 DEBUG TEST PASSED!")
        print("✅ Ready to run full validation battery")
        
        # Save a simple result
        results = {
            "timestamp": datetime.now().isoformat(),
            "debug_test": "PASSED",
            "data_loaded": True,
            "strategy_executed": True,
            "return_periods": len(returns),
            "basic_sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        }
        
        results_file = RESULTS_DIR / "debug_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Debug results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n💥 DEBUG TEST FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Save error info
        error_results = {
            "timestamp": datetime.now().isoformat(),
            "debug_test": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        results_file = RESULTS_DIR / "debug_error_results.json"
        with open(results_file, "w") as f:
            json.dump(error_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()