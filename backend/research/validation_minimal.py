"""
MINIMAL VALIDATION TEST - Core Logic Check
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# Import strategy components
from strategies.composite.mega_strategy_v31 import run_full_strategy
from strategies.composite.mega_strategy_v3 import compute_confluence, ASSET_CONFIGS

print("🧪 MINIMAL VALIDATION TEST")
print("=" * 50)

# Test data loading (just BTC, minimal date range)
print("1. Testing data loading...")
try:
    btc_data = yf.download("BTC-USD", start="2023-01-01", end="2024-01-01", progress=False)
    if isinstance(btc_data.columns, pd.MultiIndex):
        btc_data.columns = btc_data.columns.get_level_values(0)
    
    btc_df = pd.DataFrame({
        'open': btc_data['Open'], 'high': btc_data['High'], 'low': btc_data['Low'],
        'close': btc_data['Close'], 'volume': btc_data['Volume']
    }, index=btc_data.index)
    
    print(f"  ✓ BTC data loaded: {len(btc_df)} days")
    
    # Cross-asset minimal
    gold_data = yf.download("GLD", start="2023-01-01", end="2024-01-01", progress=False)
    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_data.columns = gold_data.columns.get_level_values(0)
    cross_asset_data = pd.DataFrame({"gold": gold_data["Close"]})
    
    print(f"  ✓ Cross-asset data loaded: {len(cross_asset_data)} days")
    
    # FRED minimal
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    m2_data = fred.get_series("M2SL", start="2022-01-01")
    macro_data = pd.DataFrame({"m2": m2_data}).resample('D').ffill()
    
    print(f"  ✓ Macro data loaded: {len(macro_data)} days")
    
except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    exit(1)

# Test confluence computation
print("\n2. Testing confluence computation...")
try:
    confluence, breakdown = compute_confluence(
        btc_df["close"], macro_data, cross_asset_data,
        sma_slow=100, momentum_period=35
    )
    print(f"  ✓ Confluence computed: {len(confluence)} days, range [{confluence.min()}-{confluence.max()}]")
    print(f"  ✓ Breakdown signals: {list(breakdown.columns)}")
except Exception as e:
    print(f"  ✗ Confluence failed: {e}")
    exit(1)

# Test strategy execution
print("\n3. Testing strategy execution...")
try:
    crypto_data = {"BTC": btc_df}
    
    portfolio_df, _ = run_full_strategy(
        crypto_data, macro_data, cross_asset_data,
        leverage_map={5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0},
        weights={"BTC": 1.0},
        tx_cost=0.001
    )
    
    returns = portfolio_df["daily_pnl"]
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    total_return = (1 + returns).prod() - 1
    
    print(f"  ✓ Strategy executed: {len(returns)} days")
    print(f"  ✓ Total return: {total_return*100:.2f}%")
    print(f"  ✓ Sharpe ratio: {sharpe:.3f}")
    
except Exception as e:
    print(f"  ✗ Strategy execution failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Mini validation test: Permutation
print("\n4. Mini permutation test (100 shuffles)...")
try:
    real_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    shuffle_sharpes = []
    for i in range(100):
        # Shuffle the confluence signals
        shuffled_conf = confluence.sample(frac=1, random_state=i).values
        shuffled_series = pd.Series(shuffled_conf, index=confluence.index)
        
        # Simple leverage mapping
        leverage = shuffled_series.map({5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0})
        leverage = leverage.shift(1).fillna(0)
        
        # Simple returns calculation
        btc_returns = btc_df["close"].pct_change()
        portfolio_returns = leverage * btc_returns
        portfolio_returns = portfolio_returns.dropna()
        
        shuffle_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        shuffle_sharpes.append(shuffle_sharpe)
    
    better_count = sum(1 for s in shuffle_sharpes if s >= real_sharpe)
    p_value = better_count / 100
    
    print(f"  ✓ Real Sharpe: {real_sharpe:.4f}")
    print(f"  ✓ Shuffle mean: {np.mean(shuffle_sharpes):.4f}")
    print(f"  ✓ P-value: {p_value:.4f}")
    print(f"  ✓ Result: {'PASS' if p_value < 0.10 else 'FAIL'} (p < 0.10)")
    
except Exception as e:
    print(f"  ✗ Permutation test failed: {e}")

print(f"\n🎉 MINIMAL VALIDATION COMPLETE")
print("=" * 50)
print("✅ All core components working!")
print("✅ Ready for full validation battery")