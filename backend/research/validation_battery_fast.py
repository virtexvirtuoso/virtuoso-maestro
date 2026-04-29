"""
MEGA STRATEGY V3.1-H2 — COMPREHENSIVE VALIDATION BATTERY (FAST VERSION)

Streamlined version for faster execution with shorter data ranges.
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats
from fredapi import Fred
import yfinance as yf

# Import strategy components
from strategies.composite.mega_strategy_v31 import run_full_strategy
from strategies.composite.mega_strategy_v3 import compute_confluence, ASSET_CONFIGS

# Configuration
RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}
WEIGHTS = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15}

# V3.1-H2 Configuration
LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}
TX_COST = 0.001
FRED_API_KEY = os.getenv('FRED_API_KEY')

print(f"🔬 FAST VALIDATION BATTERY STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

def load_data():
    """Load data with shorter date range for faster execution."""
    print("Loading data (2020-2024 for speed)...")
    
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            data = yf.download(ticker, start="2020-01-01", progress=False)
            if len(data) > 100:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                df = pd.DataFrame({
                    'open': data['Open'], 'high': data['High'], 'low': data['Low'],
                    'close': data['Close'], 'volume': data['Volume']
                }, index=data.index)
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            data = yf.download(ticker, start="2020-01-01", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            cross_asset_data[col] = data["Close"]
            print(f"  {col} ({ticker}): {len(data)} days")
        except Exception as e:
            print(f"  {col}: FAILED - {e}")
    
    fred = Fred(api_key=FRED_API_KEY)
    macro_data = pd.DataFrame()
    for col, series in FRED_SERIES.items():
        try:
            data = fred.get_series(series, start="2018-01-01")
            macro_data[col] = data
            print(f"  {col} ({series}): {len(data)} points")
        except Exception as e:
            print(f"  {col}: FAILED - {e}")
    
    if len(macro_data) > 0:
        macro_data = macro_data.resample('D').ffill()
    
    print(f"Data loaded: Crypto: {len(crypto_data)}, Cross-asset: {len(cross_asset_data.columns)}, Macro: {len(macro_data.columns)}")
    return crypto_data, cross_asset_data, macro_data

def compute_sharpe(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))

def run_backtest_with_cost(crypto_data, macro_data, cross_asset_data, tx_cost=TX_COST):
    portfolio_df, _ = run_full_strategy(
        crypto_data, macro_data, cross_asset_data,
        leverage_map=LEVERAGE_MAP, weights=WEIGHTS, tx_cost=tx_cost
    )
    return portfolio_df["daily_pnl"]

# ═══════════════════════════════════════════════════════════════════════════════
# FAST TESTS (SIMPLIFIED VERSIONS)
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_regime_stationarity(crypto_data, macro_data, cross_asset_data):
    """Fast version: 2 windows instead of 3."""
    print("\n" + "="*60)
    print("TEST 1: REGIME STATIONARITY (FAST)")
    print("="*60)
    
    btc_close = crypto_data["BTC"]["close"]
    
    if "m2" not in macro_data.columns:
        return {"status": "ERROR", "reason": "No M2 data"}
    
    m2 = macro_data["m2"].reindex(btc_close.index, method="ffill").dropna()
    btc_aligned = btc_close.reindex(m2.index).dropna()
    m2_yoy = m2.pct_change(252).dropna()
    
    # Split into 2 windows
    n = len(m2_yoy)
    mid = n // 2
    
    results = {}
    directions = []
    
    for i, (start, end) in enumerate([(0, mid), (mid, n)]):
        window_m2 = m2_yoy.iloc[start:end]
        window_btc = btc_aligned.iloc[start:end]
        
        m2_accel = window_m2 > window_m2.rolling(90).mean()
        btc_fwd_ret = window_btc.pct_change(30).shift(-30)
        
        common_idx = m2_accel.index.intersection(btc_fwd_ret.index)
        m2_final = m2_accel.reindex(common_idx)
        btc_final = btc_fwd_ret.reindex(common_idx)
        
        valid_mask = ~(m2_final.isna() | btc_final.isna())
        if valid_mask.sum() < 20:
            continue
            
        m2_clean = m2_final[valid_mask]
        btc_clean = btc_final[valid_mask]
        
        ret_when_accel = float(btc_clean[m2_clean].mean()) if m2_clean.sum() > 0 else 0
        ret_when_not = float(btc_clean[~m2_clean].mean()) if (~m2_clean).sum() > 0 else 0
        difference = ret_when_accel - ret_when_not
        
        directions.append(difference > 0)
        
        results[f"window_{i+1}"] = {
            "period": f"{window_m2.index[0].date()} to {window_m2.index[-1].date()}",
            "difference": round(difference * 100, 2),
            "return_when_m2_accel": round(ret_when_accel * 100, 2),
            "return_when_m2_not": round(ret_when_not * 100, 2)
        }
        
        print(f"Window {i+1}: Difference {difference*100:+.2f}%")
    
    consistent = len(set(directions)) == 1 if len(directions) == 2 else False
    print(f"Result: {'PASS' if consistent else 'FAIL'}")
    
    return {"status": "PASS" if consistent else "FAIL", "windows": results, "consistent_direction": consistent}

def test_2_monte_carlo_permutation(crypto_data, macro_data, cross_asset_data, n_shuffles=1000):
    """Fast version: 1,000 shuffles instead of 10,000."""
    print("\n" + "="*60)
    print("TEST 2: MONTE CARLO PERMUTATION (FAST)")
    print("="*60)
    
    real_returns = run_backtest_with_cost(crypto_data, macro_data, cross_asset_data)
    real_sharpe = compute_sharpe(real_returns)
    
    print(f"Real Sharpe: {real_sharpe:.4f}")
    print(f"Running {n_shuffles} permutations...")
    
    btc_close = crypto_data["BTC"]["close"]
    confluence, _ = compute_confluence(
        btc_close, macro_data, cross_asset_data,
        sma_slow=ASSET_CONFIGS["BTC"]["sma_slow"],
        momentum_period=ASSET_CONFIGS["BTC"]["momentum_period"]
    )
    
    common_idx = real_returns.index.intersection(confluence.index)
    returns_aligned = real_returns.reindex(common_idx)
    confluence_aligned = confluence.reindex(common_idx)
    
    shuffle_sharpes = []
    
    for i in range(n_shuffles):
        if i % 200 == 0:
            print(f"  Progress: {i}/{n_shuffles}")
        
        shuffled_signals = confluence_aligned.sample(frac=1, random_state=i+42).values
        shuffled_confluence = pd.Series(shuffled_signals, index=confluence_aligned.index)
        
        leverage_series = shuffled_confluence.map(lambda x: LEVERAGE_MAP.get(int(x), 0.0))
        leverage_series = leverage_series.shift(1).fillna(0)
        
        portfolio_ret = pd.Series(0.0, index=common_idx)
        for asset, weight in WEIGHTS.items():
            if asset in crypto_data:
                asset_ret = crypto_data[asset]["close"].pct_change().reindex(common_idx, method='ffill').fillna(0)
                portfolio_ret += weight * leverage_series * asset_ret
        
        lev_changes = leverage_series.diff().abs().fillna(0)
        portfolio_ret -= lev_changes * TX_COST
        
        shuffle_sharpe = compute_sharpe(portfolio_ret)
        shuffle_sharpes.append(shuffle_sharpe)
    
    shuffle_sharpes = np.array(shuffle_sharpes)
    better_count = (shuffle_sharpes >= real_sharpe).sum()
    p_value = float(better_count / n_shuffles)
    
    p95 = float(np.percentile(shuffle_sharpes, 95))
    p99 = float(np.percentile(shuffle_sharpes, 99))
    
    print(f"Real: {real_sharpe:.4f}, P-value: {p_value:.4f}, 95th pct: {p95:.4f}")
    
    pass_test = p_value < 0.05
    print(f"Result: {'PASS' if pass_test else 'FAIL'}")
    
    return {
        "status": "PASS" if pass_test else "FAIL",
        "real_sharpe": round(real_sharpe, 4),
        "p_value": round(p_value, 4),
        "percentile_95": round(p95, 4),
        "percentile_99": round(p99, 4),
        "significant": pass_test
    }

def test_4_transaction_cost_sensitivity(crypto_data, macro_data, cross_asset_data):
    """Transaction cost sensitivity test."""
    print("\n" + "="*60)
    print("TEST 4: TRANSACTION COST SENSITIVITY")
    print("="*60)
    
    cost_levels = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
    results = []
    
    for cost in cost_levels:
        returns = run_backtest_with_cost(crypto_data, macro_data, cross_asset_data, tx_cost=cost)
        sharpe = compute_sharpe(returns)
        
        total_return = float((1 + returns).prod() - 1) * 100
        result = {"cost_pct": round(cost * 100, 3), "sharpe": round(sharpe, 4), "total_return": round(total_return, 2)}
        results.append(result)
        print(f"Cost {cost*100:.2f}%: Sharpe {sharpe:.4f}")
    
    # Find breakeven
    sharpes = [r["sharpe"] for r in results]
    costs = [r["cost_pct"]/100 for r in results]
    
    breakeven_cost = 0.0
    for i in range(len(sharpes)-1):
        if sharpes[i] > 0 and sharpes[i+1] <= 0:
            x1, y1 = costs[i], sharpes[i]
            x2, y2 = costs[i+1], sharpes[i+1]
            breakeven_cost = x1 - y1 * (x2 - x1) / (y2 - y1)
            break
    
    breakeven_pct = breakeven_cost * 100
    pass_test = breakeven_pct > 0.3
    
    print(f"Breakeven cost: {breakeven_pct:.3f}%")
    print(f"Result: {'PASS' if pass_test else 'FAIL'}")
    
    return {
        "status": "PASS" if pass_test else "FAIL",
        "cost_analysis": results,
        "breakeven_cost_pct": round(breakeven_pct, 3),
        "passes_3x_threshold": pass_test
    }

def test_6_out_of_universe(macro_data, cross_asset_data):
    """Out-of-universe test on SPY and GLD only for speed."""
    print("\n" + "="*60)
    print("TEST 6: OUT-OF-UNIVERSE TEST (FAST)")
    print("="*60)
    
    test_assets = {"SPY": "SPY", "GLD": "GLD"}  # Test fewer assets for speed
    results = {}
    wins = 0
    
    for asset_name, ticker in test_assets.items():
        print(f"Testing on {asset_name}...")
        
        try:
            data = yf.download(ticker, start="2020-01-01", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            asset_close = data["Close"]
            
            confluence, _ = compute_confluence(asset_close, macro_data, cross_asset_data, sma_slow=100, momentum_period=35)
            leverage = confluence.map(lambda x: LEVERAGE_MAP.get(int(x), 0.0))
            leverage = leverage.shift(1).fillna(0)
            
            asset_returns = asset_close.pct_change()
            strategy_returns = leverage * asset_returns
            lev_changes = leverage.diff().abs().fillna(0)
            strategy_returns -= lev_changes * TX_COST
            
            strategy_returns = strategy_returns.dropna()
            asset_returns_clean = asset_returns.reindex(strategy_returns.index).dropna()
            
            strategy_sharpe = compute_sharpe(strategy_returns)
            bh_sharpe = compute_sharpe(asset_returns_clean)
            
            beats_bh = strategy_sharpe > bh_sharpe
            if beats_bh:
                wins += 1
            
            results[asset_name] = {
                "strategy_sharpe": round(strategy_sharpe, 4),
                "bh_sharpe": round(bh_sharpe, 4),
                "beats_bh": beats_bh
            }
            
            print(f"  Strategy: {strategy_sharpe:.4f}, B&H: {bh_sharpe:.4f}, Beats: {beats_bh}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[asset_name] = {"error": str(e)}
    
    pass_test = wins >= 1  # Need 1 out of 2 for fast version
    print(f"Result: {'PASS' if pass_test else 'FAIL'} ({wins}/{len(test_assets)} wins)")
    
    return {
        "status": "PASS" if pass_test else "FAIL",
        "asset_results": results,
        "wins": wins,
        "total_assets": len(test_assets),
        "passes_threshold": pass_test
    }

def main():
    print("MEGA STRATEGY V3.1-H2 FAST VALIDATION BATTERY")
    print("🚀 STREAMLINED VERSION FOR QUICK VALIDATION")
    print("=" * 60)
    
    # Load data
    crypto_data, cross_asset_data, macro_data = load_data()
    
    if not crypto_data or len(macro_data) == 0:
        print("ERROR: Insufficient data for validation!")
        return
    
    # Run core tests
    test_results = {}
    
    try:
        test_results["test_1_regime_stationarity"] = test_1_regime_stationarity(crypto_data, macro_data, cross_asset_data)
    except Exception as e:
        print(f"Test 1 failed: {e}")
        test_results["test_1_regime_stationarity"] = {"status": "ERROR", "error": str(e)}
    
    try:
        test_results["test_2_monte_carlo"] = test_2_monte_carlo_permutation(crypto_data, macro_data, cross_asset_data)
    except Exception as e:
        print(f"Test 2 failed: {e}")
        test_results["test_2_monte_carlo"] = {"status": "ERROR", "error": str(e)}
    
    try:
        test_results["test_4_cost_sensitivity"] = test_4_transaction_cost_sensitivity(crypto_data, macro_data, cross_asset_data)
    except Exception as e:
        print(f"Test 4 failed: {e}")
        test_results["test_4_cost_sensitivity"] = {"status": "ERROR", "error": str(e)}
    
    try:
        test_results["test_6_out_of_universe"] = test_6_out_of_universe(macro_data, cross_asset_data)
    except Exception as e:
        print(f"Test 6 failed: {e}")
        test_results["test_6_out_of_universe"] = {"status": "ERROR", "error": str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("FAST VALIDATION SUMMARY")
    print("="*80)
    
    test_names = [
        ("Test 1: Regime Stationarity", "test_1_regime_stationarity"),
        ("Test 2: Monte Carlo Permutation", "test_2_monte_carlo"),
        ("Test 4: Transaction Cost Sensitivity", "test_4_cost_sensitivity"),
        ("Test 6: Out-of-Universe", "test_6_out_of_universe")
    ]
    
    passed_tests = 0
    total_tests = len(test_names)
    
    for test_desc, test_key in test_names:
        if test_key in test_results and test_results[test_key]["status"] == "PASS":
            passed_tests += 1
        status = test_results[test_key]["status"] if test_key in test_results else "ERROR"
        print(f"{test_desc:<40} {status}")
    
    pass_rate = passed_tests / total_tests
    
    if pass_rate >= 0.75:  # 3/4 tests
        verdict = "VALIDATED"
    elif pass_rate >= 0.50:
        verdict = "PARTIALLY VALIDATED"
    else:
        verdict = "NOT VALIDATED"
    
    print(f"\nFAST VALIDATION VERDICT: {verdict}")
    print(f"Tests Passed: {passed_tests}/{total_tests} ({pass_rate*100:.1f}%)")
    
    # Save results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "MegaStrategyV3.1-H2",
        "validation_type": "FAST",
        "validation_verdict": verdict,
        "tests_passed": passed_tests,
        "total_tests": total_tests,
        "pass_rate": round(pass_rate, 3),
        "test_results": test_results
    }
    
    results_file = RESULTS_DIR / "validation_battery_fast_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()