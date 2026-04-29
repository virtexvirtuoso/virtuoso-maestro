"""
MEGA STRATEGY V3.1-H2 — VALIDATION BATTERY PART 2 (FINAL)

Complete validation with:
- Test A: Synthetic Bootstrap (500 paths)  
- Test B: Signal Decay Analysis
- Test C: M2 Deep Dive across regimes
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
from fredapi import Fred
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

# Force output flushing
def print_flush(msg):
    print(msg, flush=True)

print_flush(f"🔬 VALIDATION BATTERY PART 2 STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_flush("=" * 80)

# Import strategy components
try:
    from strategies.composite.mega_strategy_v31 import run_full_strategy
    from strategies.composite.mega_strategy_v3 import compute_confluence, ASSET_CONFIGS
    print_flush("✅ Strategy imports successful")
except ImportError as e:
    print_flush(f"❌ Strategy import failed: {e}")
    sys.exit(1)

# Configuration
RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}
WEIGHTS = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15}

LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}
TX_COST = 0.001
FRED_API_KEY = os.getenv('FRED_API_KEY')

def load_data():
    """Load extended data from 2017-2024."""
    print_flush("\n📊 Loading extended data (2017-2024)...")
    
    crypto_data = {}
    print_flush("Loading crypto data...")
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            print_flush(f"  Loading {name}...")
            data = yf.download(ticker, start="2017-01-01", end="2025-01-01", progress=False)
            if len(data) > 100:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                df = pd.DataFrame({
                    'open': data['Open'], 'high': data['High'], 'low': data['Low'],
                    'close': data['Close'], 'volume': data['Volume']
                }, index=data.index)
                crypto_data[name] = df
                print_flush(f"    ✓ {name}: {len(df)} days")
        except Exception as e:
            print_flush(f"    ✗ {name}: {e}")
    
    print_flush("Loading cross-asset data...")
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            print_flush(f"  Loading {col}...")
            data = yf.download(ticker, start="2017-01-01", end="2025-01-01", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            cross_asset_data[col] = data["Close"]
            print_flush(f"    ✓ {col}: {len(data)} days")
        except Exception as e:
            print_flush(f"    ✗ {col}: {e}")
    
    print_flush("Loading macro data...")
    fred = Fred(api_key=FRED_API_KEY)
    macro_data = pd.DataFrame()
    for col, series in FRED_SERIES.items():
        try:
            print_flush(f"  Loading {col}...")
            data = fred.get_series(series, start="2015-01-01", end="2025-01-01")
            macro_data[col] = data
            print_flush(f"    ✓ {col}: {len(data)} points")
        except Exception as e:
            print_flush(f"    ✗ {col}: {e}")
    
    if len(macro_data) > 0:
        macro_data = macro_data.resample('D').ffill()
    
    print_flush(f"\n✅ Data loading complete:")
    print_flush(f"   Crypto: {len(crypto_data)} assets")
    print_flush(f"   Cross-asset: {len(cross_asset_data.columns)} series")
    print_flush(f"   Macro: {len(macro_data.columns)} series")
    
    return crypto_data, cross_asset_data, macro_data

def compute_sharpe(returns):
    """Compute annualized Sharpe ratio."""
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))

def block_bootstrap_returns(returns, block_size=20, n_paths=500, seed=42):
    """Generate synthetic price paths using block bootstrap."""
    np.random.seed(seed)
    returns_clean = returns.dropna()
    n_obs = len(returns_clean)
    
    if n_obs < block_size * 2:
        raise ValueError(f"Need at least {block_size * 2} observations, got {n_obs}")
    
    synthetic_paths = []
    
    for path_i in range(n_paths):
        synthetic_returns = []
        
        while len(synthetic_returns) < n_obs:
            start_idx = np.random.randint(0, n_obs - block_size + 1)
            block = returns_clean.iloc[start_idx:start_idx + block_size]
            synthetic_returns.extend(block.values)
        
        synthetic_returns = synthetic_returns[:n_obs]
        synthetic_paths.append(pd.Series(synthetic_returns, index=returns_clean.index))
    
    return synthetic_paths

def run_strategy_on_synthetic_data(crypto_data_original, macro_data, cross_asset_data, synthetic_btc_returns):
    """Run V3.1 strategy on synthetic BTC path with REAL macro data."""
    crypto_data_synthetic = crypto_data_original.copy()
    
    # Generate synthetic BTC prices
    btc_original = crypto_data_original["BTC"]["close"]
    initial_price = btc_original.iloc[0]
    synthetic_prices = initial_price * (1 + synthetic_btc_returns).cumprod()
    
    # Update synthetic BTC data
    crypto_data_synthetic["BTC"] = crypto_data_original["BTC"].copy()
    crypto_data_synthetic["BTC"]["close"] = synthetic_prices
    
    # Scale OHLC proportionally
    price_ratio = synthetic_prices / btc_original
    for col in ["open", "high", "low"]:
        crypto_data_synthetic["BTC"][col] = crypto_data_original["BTC"][col] * price_ratio
    
    try:
        portfolio_df, _ = run_full_strategy(
            crypto_data_synthetic, macro_data, cross_asset_data,
            leverage_map=LEVERAGE_MAP, weights=WEIGHTS, tx_cost=TX_COST
        )
        return portfolio_df["daily_pnl"]
    except Exception as e:
        return pd.Series(dtype=float)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST A: SYNTHETIC BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_synthetic_bootstrap(crypto_data, macro_data, cross_asset_data, n_paths=500):
    """Test A: Synthetic Bootstrap (500 paths)"""
    print_flush("\n" + "="*60)
    print_flush("TEST A: SYNTHETIC BOOTSTRAP (500 PATHS)")
    print_flush("="*60)
    
    if "BTC" not in crypto_data:
        return {"status": "ERROR", "reason": "No BTC data"}
    
    btc_returns = crypto_data["BTC"]["close"].pct_change().dropna()
    print_flush(f"Bootstrapping {n_paths} synthetic BTC paths (block_size=20)...")
    
    try:
        synthetic_paths = block_bootstrap_returns(btc_returns, block_size=20, n_paths=n_paths)
    except Exception as e:
        return {"status": "ERROR", "reason": f"Bootstrap failed: {e}"}
    
    print_flush(f"Running strategy on {len(synthetic_paths)} synthetic paths...")
    
    results = []
    strategy_wins = 0
    
    for i, synthetic_returns in enumerate(synthetic_paths):
        if i % 50 == 0:
            print_flush(f"  Progress: {i}/{n_paths} ({strategy_wins} wins so far)")
        
        try:
            # Run strategy on synthetic path
            strategy_returns = run_strategy_on_synthetic_data(
                crypto_data, macro_data, cross_asset_data, synthetic_returns
            )
            
            if len(strategy_returns) == 0:
                continue
                
            # Calculate Sharpe ratios
            strategy_sharpe = compute_sharpe(strategy_returns)
            bh_sharpe = compute_sharpe(synthetic_returns.reindex(strategy_returns.index))
            
            beats_bh = strategy_sharpe > bh_sharpe
            if beats_bh:
                strategy_wins += 1
            
            results.append({
                "path": i,
                "strategy_sharpe": round(strategy_sharpe, 4),
                "bh_sharpe": round(bh_sharpe, 4),
                "beats_bh": beats_bh
            })
            
        except Exception as e:
            continue
    
    if len(results) == 0:
        return {"status": "ERROR", "reason": "No valid bootstrap paths"}
    
    win_rate = strategy_wins / len(results)
    pass_test = win_rate > 0.60
    
    strategy_sharpes = [r["strategy_sharpe"] for r in results]
    bh_sharpes = [r["bh_sharpe"] for r in results]
    
    print_flush(f"\nBootstrap Results:")
    print_flush(f"  Valid paths: {len(results)}")
    print_flush(f"  Strategy wins: {strategy_wins}")
    print_flush(f"  Win rate: {win_rate:.1%}")
    print_flush(f"  Avg strategy Sharpe: {np.mean(strategy_sharpes):.3f}")
    print_flush(f"  Avg B&H Sharpe: {np.mean(bh_sharpes):.3f}")
    print_flush(f"  Result: {'PASS' if pass_test else 'FAIL'}")
    
    return {
        "status": "PASS" if pass_test else "FAIL",
        "n_paths": len(results),
        "strategy_wins": strategy_wins,
        "win_rate": round(win_rate, 3),
        "avg_strategy_sharpe": round(np.mean(strategy_sharpes), 4),
        "avg_bh_sharpe": round(np.mean(bh_sharpes), 4)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST B: SIGNAL DECAY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signal_components(crypto_data, macro_data, cross_asset_data):
    """Extract the 5 core signals for decay analysis."""
    btc_close = crypto_data["BTC"]["close"]
    signals = {}
    
    try:
        # 1. M2 Acceleration
        if "m2" in macro_data.columns:
            m2 = macro_data["m2"].reindex(btc_close.index, method="ffill")
            m2_yoy = m2.pct_change(252)
            m2_accel = m2_yoy > m2_yoy.rolling(90).mean()
            signals["m2_accel"] = m2_accel.astype(int)
        
        # 2. Liquidity Proxy (HYG relative performance)
        if "hyg" in cross_asset_data.columns:
            hyg_ret = cross_asset_data["hyg"].pct_change(20)
            hyg_rank = hyg_ret.rolling(252).rank(pct=True)
            signals["liq_proxy"] = (hyg_rank > 0.7).astype(int)
        
        # 3. Yield Curve
        if "yield_curve" in macro_data.columns:
            yield_curve = macro_data["yield_curve"].reindex(btc_close.index, method="ffill")
            yield_signal = yield_curve > yield_curve.rolling(90).mean()
            signals["yield_curve"] = yield_signal.astype(int)
        
        # 4. Cross-asset Momentum
        if "gold" in cross_asset_data.columns and "bonds" in cross_asset_data.columns:
            gold_mom = cross_asset_data["gold"].pct_change(20) > 0
            bonds_mom = cross_asset_data["bonds"].pct_change(20) > 0
            cross_mom = (gold_mom & bonds_mom).astype(int)
            signals["cross_asset_mom"] = cross_mom
        
        # 5. Crypto Momentum
        btc_mom = btc_close.pct_change(30) > 0
        signals["crypto_mom"] = btc_mom.astype(int)
        
    except Exception as e:
        print_flush(f"Error computing signals: {e}")
    
    return signals

def test_b_signal_decay(crypto_data, macro_data, cross_asset_data):
    """Test B: Signal Decay Analysis"""
    print_flush("\n" + "="*60)
    print_flush("TEST B: SIGNAL DECAY ANALYSIS")
    print_flush("="*60)
    
    if "BTC" not in crypto_data:
        return {"status": "ERROR", "reason": "No BTC data"}
    
    btc_close = crypto_data["BTC"]["close"]
    signals = compute_signal_components(crypto_data, macro_data, cross_asset_data)
    
    if len(signals) == 0:
        return {"status": "ERROR", "reason": "No signals computed"}
    
    horizons = [1, 5, 10, 20, 30, 60, 90]
    ic_results = {}
    
    for signal_name, signal_series in signals.items():
        print_flush(f"Analyzing {signal_name}...")
        signal_ics = {}
        
        for horizon in horizons:
            try:
                btc_fwd_ret = btc_close.pct_change(horizon).shift(-horizon)
                
                common_idx = signal_series.index.intersection(btc_fwd_ret.index)
                if len(common_idx) < 100:
                    continue
                
                signal_aligned = signal_series.reindex(common_idx)
                ret_aligned = btc_fwd_ret.reindex(common_idx)
                
                valid_mask = ~(signal_aligned.isna() | ret_aligned.isna())
                if valid_mask.sum() < 50:
                    continue
                
                signal_clean = signal_aligned[valid_mask]
                ret_clean = ret_aligned[valid_mask]
                
                ic, p_value = stats.spearmanr(signal_clean, ret_clean)
                
                signal_ics[f"{horizon}d"] = {
                    "ic": round(float(ic), 4),
                    "p_value": round(float(p_value), 4),
                    "significant": p_value < 0.05,
                    "n_obs": int(valid_mask.sum())
                }
                
            except Exception as e:
                continue
        
        ic_results[signal_name] = signal_ics
        
        sig_horizons = [h for h in signal_ics if signal_ics[h]["significant"]]
        print_flush(f"  Significant horizons: {sig_horizons}")
    
    # Check pass condition: 3+ signals significant at 20d+
    long_horizon_signals = []
    for signal_name, signal_ics in ic_results.items():
        has_long_significance = any(
            signal_ics[h]["significant"] 
            for h in ["20d", "30d", "60d", "90d"] 
            if h in signal_ics
        )
        if has_long_significance:
            long_horizon_signals.append(signal_name)
    
    pass_test = len(long_horizon_signals) >= 3
    
    print_flush(f"\nSignal Decay Summary:")
    print_flush(f"  Signals with 20d+ significance: {long_horizon_signals}")
    print_flush(f"  Count: {len(long_horizon_signals)}/5")
    print_flush(f"  Result: {'PASS' if pass_test else 'FAIL'}")
    
    return {
        "status": "PASS" if pass_test else "FAIL",
        "ic_analysis": ic_results,
        "long_horizon_significant": long_horizon_signals,
        "n_long_significant": len(long_horizon_signals),
        "passes_threshold": pass_test
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST C: M2 DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════

def test_c_m2_deep_dive(crypto_data, macro_data):
    """Test C: M2 Deep Dive across regime windows"""
    print_flush("\n" + "="*60)
    print_flush("TEST C: M2 DEEP DIVE")
    print_flush("="*60)
    
    if "BTC" not in crypto_data or "m2" not in macro_data.columns:
        return {"status": "ERROR", "reason": "Missing BTC or M2 data"}
    
    btc_close = crypto_data["BTC"]["close"]
    m2 = macro_data["m2"].reindex(btc_close.index, method="ffill")
    
    # Compute M2 acceleration signal
    m2_yoy = m2.pct_change(252)
    m2_accel = m2_yoy > m2_yoy.rolling(90).mean()
    
    # 30d forward BTC returns
    btc_fwd_30d = btc_close.pct_change(30).shift(-30)
    
    # Align data
    common_idx = m2_accel.index.intersection(btc_fwd_30d.index)
    m2_aligned = m2_accel.reindex(common_idx)
    ret_aligned = btc_fwd_30d.reindex(common_idx)
    
    valid_mask = ~(m2_aligned.isna() | ret_aligned.isna())
    m2_clean = m2_aligned[valid_mask]
    ret_clean = ret_aligned[valid_mask]
    
    # Define windows
    windows = {
        "pre_2017": ("2015-01-01", "2016-12-31"),
        "2017_2019": ("2017-01-01", "2019-12-31"), 
        "2020_2021": ("2020-01-01", "2021-12-31"),
        "2022_2024": ("2022-01-01", "2024-12-31")
    }
    
    window_analysis = {}
    
    print_flush("Analyzing M2 effectiveness by regime:")
    
    for window_name, (start_date, end_date) in windows.items():
        try:
            window_mask = (m2_clean.index >= start_date) & (m2_clean.index <= end_date)
            window_m2 = m2_clean[window_mask]
            window_ret = ret_clean[window_mask]
            
            if len(window_m2) < 20:
                continue
                
            if window_m2.sum() > 5 and (~window_m2).sum() > 5:
                correlation, p_val = stats.spearmanr(window_m2.astype(int), window_ret)
                
                ret_when_accel = window_ret[window_m2].mean() if window_m2.sum() > 0 else 0
                ret_when_not = window_ret[~window_m2].mean() if (~window_m2).sum() > 0 else 0
                
                hit_rate = (window_ret[window_m2] > 0).mean() if window_m2.sum() > 0 else 0
                
                window_analysis[window_name] = {
                    "period": f"{start_date} to {end_date}",
                    "n_obs": int(len(window_m2)),
                    "correlation": round(float(correlation), 4),
                    "correlation_p_value": round(float(p_val), 4),
                    "ret_when_m2_accel": round(float(ret_when_accel * 100), 2),
                    "ret_when_m2_not": round(float(ret_when_not * 100), 2),
                    "difference": round(float((ret_when_accel - ret_when_not) * 100), 2),
                    "hit_rate_when_accel": round(float(hit_rate), 3),
                    "m2_accel_days": int(window_m2.sum()),
                    "total_days": int(len(window_m2))
                }
                
                print_flush(f"  {window_name}: Corr={correlation:.3f}, Diff={ret_when_accel-ret_when_not:+.1%}")
            
        except Exception as e:
            print_flush(f"  {window_name}: ERROR - {e}")
            continue
    
    # Rolling 1-year correlation (simplified)
    print_flush("Computing rolling 1yr correlation...")
    rolling_corr = []
    
    try:
        for i in range(252, len(m2_clean), 21):  # Every 3 weeks
            if i + 252 > len(m2_clean):
                break
                
            window_data = pd.DataFrame({
                "m2": m2_clean.iloc[i-252:i].astype(int),
                "ret": ret_clean.iloc[i-252:i]
            }).dropna()
            
            if len(window_data) > 50:
                corr, _ = stats.spearmanr(window_data["m2"], window_data["ret"])
                rolling_corr.append({
                    "date": m2_clean.index[i],
                    "correlation": round(float(corr), 4)
                })
                
        corr_values = [r["correlation"] for r in rolling_corr]
        if corr_values:
            rolling_stats = {
                "mean_correlation": round(float(np.mean(corr_values)), 4),
                "std_correlation": round(float(np.std(corr_values)), 4),
                "periods_positive": int(sum(1 for c in corr_values if c > 0)),
                "total_periods": len(corr_values)
            }
        else:
            rolling_stats = {}
            
    except Exception as e:
        print_flush(f"Rolling correlation failed: {e}")
        rolling_corr = []
        rolling_stats = {}
    
    # Diagnosis
    diagnosis = []
    if len(window_analysis) >= 2:
        correlations = [(w, a["correlation"]) for w, a in window_analysis.items()]
        best_period = max(correlations, key=lambda x: x[1])
        worst_period = min(correlations, key=lambda x: x[1])
        
        diagnosis.extend([
            f"Best M2 period: {best_period[0]} (correlation: {best_period[1]})",
            f"Worst M2 period: {worst_period[0]} (correlation: {worst_period[1]})"
        ])
        
        if "2022_2024" in window_analysis:
            recent_corr = window_analysis["2022_2024"]["correlation"]
            if recent_corr > 0.1:
                diagnosis.append("M2 signal remains effective in recent period")
            else:
                diagnosis.append("M2 signal weak in recent period")
    
    print_flush(f"\nM2 Deep Dive Complete:")
    print_flush(f"  Windows analyzed: {len(window_analysis)}")
    print_flush(f"  Rolling periods: {len(rolling_corr)}")
    for diag in diagnosis:
        print_flush(f"  {diag}")
    
    return {
        "status": "COMPLETE",
        "window_analysis": window_analysis,
        "rolling_stats": rolling_stats,
        "diagnosis": diagnosis,
        "n_windows": len(window_analysis),
        "n_rolling_periods": len(rolling_corr)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_flush("MEGA STRATEGY V3.1-H2 VALIDATION BATTERY PART 2")
    print_flush("🧪 ADVANCED VALIDATION: BOOTSTRAP + SIGNAL DECAY + M2 DEEP DIVE")
    print_flush("=" * 80)
    
    # Load data
    crypto_data, cross_asset_data, macro_data = load_data()
    
    if not crypto_data or len(macro_data) == 0:
        print_flush("❌ Insufficient data for validation!")
        return
    
    # Run Part 2 tests
    test_results = {}
    
    print_flush(f"\n🔬 Starting Part 2 validation at {datetime.now().strftime('%H:%M:%S')}")
    
    # Test A: Synthetic Bootstrap
    try:
        print_flush("\n" + "🎲" * 20 + " TEST A " + "🎲" * 20)
        test_results["test_a_synthetic_bootstrap"] = test_a_synthetic_bootstrap(
            crypto_data, macro_data, cross_asset_data, n_paths=500
        )
    except Exception as e:
        print_flush(f"Test A failed: {e}")
        test_results["test_a_synthetic_bootstrap"] = {"status": "ERROR", "error": str(e)}
    
    # Test B: Signal Decay
    try:
        print_flush("\n" + "📉" * 20 + " TEST B " + "📉" * 20)
        test_results["test_b_signal_decay"] = test_b_signal_decay(
            crypto_data, macro_data, cross_asset_data
        )
    except Exception as e:
        print_flush(f"Test B failed: {e}")
        test_results["test_b_signal_decay"] = {"status": "ERROR", "error": str(e)}
    
    # Test C: M2 Deep Dive  
    try:
        print_flush("\n" + "💰" * 20 + " TEST C " + "💰" * 20)
        test_results["test_c_m2_deep_dive"] = test_c_m2_deep_dive(crypto_data, macro_data)
    except Exception as e:
        print_flush(f"Test C failed: {e}")
        test_results["test_c_m2_deep_dive"] = {"status": "ERROR", "error": str(e)}
    
    # Final Summary
    print_flush("\n" + "="*80)
    print_flush("VALIDATION BATTERY PART 2 SUMMARY")
    print_flush("="*80)
    
    test_names = [
        ("Test A: Synthetic Bootstrap", "test_a_synthetic_bootstrap"), 
        ("Test B: Signal Decay Analysis", "test_b_signal_decay"),
        ("Test C: M2 Deep Dive", "test_c_m2_deep_dive")
    ]
    
    passed_tests = 0
    total_tests = len(test_names)
    
    for test_desc, test_key in test_names:
        if test_key in test_results:
            status = test_results[test_key]["status"]
            if status == "PASS":
                passed_tests += 1
            elif status == "COMPLETE":
                status = "COMPLETE ✓"
                passed_tests += 0.5  # Partial credit for diagnostic test
        else:
            status = "ERROR"
        print_flush(f"{test_desc:<40} {status}")
    
    pass_rate = passed_tests / total_tests
    
    if pass_rate >= 0.75:
        verdict = "STRONGLY VALIDATED"
    elif pass_rate >= 0.50:
        verdict = "VALIDATED"
    else:
        verdict = "NEEDS IMPROVEMENT"
    
    print_flush(f"\nPART 2 VALIDATION VERDICT: {verdict}")
    print_flush(f"Tests Passed: {passed_tests:.1f}/{total_tests} ({pass_rate*100:.1f}%)")
    
    # Save comprehensive results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "MegaStrategyV3.1-H2",
        "validation_type": "PART_2_ADVANCED",
        "validation_verdict": verdict,
        "tests_passed": passed_tests,
        "total_tests": total_tests,
        "pass_rate": round(pass_rate, 3),
        "test_results": test_results,
        "data_period": "2017-2024",
        "methodology": {
            "test_a": "Block bootstrap with 500 synthetic BTC paths, strategy vs B&H comparison",
            "test_b": "Information Coefficient analysis of 5 signals across 7 horizons",
            "test_c": "M2 effectiveness analysis across 4 regime windows with rolling correlation"
        }
    }
    
    results_file = RESULTS_DIR / "validation_battery_part2_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print_flush(f"\n📊 Results saved to: {results_file}")
    print_flush(f"🏁 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return final_results

if __name__ == "__main__":
    main()