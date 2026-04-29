#!/usr/bin/env python3
"""
BTC Wiz On-Chain Signal Battery - Full Rigorous Methodology
============================================================
Tests on-chain signals from BGeometrics API against BTC price data.

Signals: MVRV, NUPL, SOPR, NVT, Reserve Risk + price-derived (Pi Cycle, 2Y MA)
Methodology: 14-fold expanding walk-forward, 500 permutation tests, Bonferroni correction
"""

import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
PRICE_FILE = Path.home() / "Desktop/maestro/data/spot/BTC_spot_daily.csv"
OUTPUT_FILE = Path.home() / "Desktop/maestro/data/backtest_results/btcwiz_full_battery.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

BGEOMETRICS_BASE = "https://bitcoin-data.com/v1"

# Signal thresholds from BTC Wiz research
THRESHOLDS = {
    'mvrv': {'buy': 1.0, 'sell': 3.7, 'name': 'MVRV Ratio'},
    'nupl': {'buy': 0.0, 'sell': 0.75, 'name': 'NUPL'},
    'sopr': {'buy': 1.0, 'sell': 1.0, 'name': 'SOPR'},  # < 1 = buy (capitulation), > 1 in bull = hold
    'nvt':  {'buy': 30, 'sell': 150, 'name': 'NVT Ratio'},
    'reserve_risk': {'buy': 0.0001, 'sell': 0.02, 'name': 'Reserve Risk'},
}

N_PERMS = 500
N_FOLDS = 14
ALPHA = 0.05

# ============================================================================
# DATA FETCHING
# ============================================================================
def fetch_onchain_data():
    """Load on-chain signals from local JSON files (pre-fetched from BGeometrics)."""
    signals = {}
    
    local_dir = Path.home() / "Desktop/maestro/data/onchain"
    
    endpoints = {
        'mvrv': ('mvrv', 'mvrv'),
        'nupl': ('nupl', 'nupl'),
        'sopr': ('sopr', 'sopr'),
        'nvt': ('nvt', 'nvt'),
        'reserve_risk': ('reserve_risk', 'reserveRisk'),
    }
    
    for name, (filename, col) in endpoints.items():
        local_file = local_dir / f"{filename}.json"
        print(f"  Loading {name}...")
        try:
            if local_file.exists():
                with open(local_file) as f:
                    data = json.load(f)
            else:
                # Fallback to API
                r = requests.get(f"{BGEOMETRICS_BASE}/{filename.replace('_','-')}", timeout=30)
                if r.status_code != 200:
                    print(f"    ✗ {name}: HTTP {r.status_code}")
                    continue
                data = r.json()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['d'])
            df[name] = pd.to_numeric(df[col], errors='coerce')
            df = df[['date', name]].dropna().set_index('date')
            signals[name] = df
            print(f"    ✓ {name}: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
        except Exception as e:
            print(f"    ✗ {name}: {e}")
    
    return signals

def load_price_data():
    """Load BTC spot price data."""
    df = pd.read_csv(PRICE_FILE)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    # Forward returns (next day)
    df['fwd_ret'] = df['close'].pct_change().shift(-1)
    print(f"  Price data: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
    return df

def compute_price_signals(price_df):
    """Compute price-derived on-chain-style signals."""
    close = price_df['close']
    signals = {}
    
    # Pi Cycle Top: 111d SMA vs 350d SMA × 2
    sma111 = close.rolling(111, min_periods=90).mean()
    sma350x2 = close.rolling(350, min_periods=280).mean() * 2
    # Distance metric: how close 111d is to crossing 350d×2 (normalized)
    pi_ratio = sma111 / sma350x2
    signals['pi_cycle'] = pd.DataFrame({'pi_cycle': pi_ratio}, index=close.index)
    THRESHOLDS['pi_cycle'] = {'buy': 0.0, 'sell': 1.0, 'name': 'Pi Cycle Ratio'}
    
    # 2-Year MA Multiplier position
    ma2y = close.rolling(730, min_periods=365).mean()
    ma2y_upper = ma2y * 5
    position = (close - ma2y) / (ma2y_upper - ma2y)
    position = position.clip(0, 1)
    signals['two_year_ma'] = pd.DataFrame({'two_year_ma': position}, index=close.index)
    THRESHOLDS['two_year_ma'] = {'buy': 0.1, 'sell': 0.8, 'name': '2Y MA Position'}
    
    # SMA50 baseline
    sma50 = close.rolling(50, min_periods=40).mean()
    signals['sma50_trend'] = pd.DataFrame({'sma50_trend': (close > sma50).astype(float)}, index=close.index)
    
    for name, df in signals.items():
        valid = df.dropna()
        if len(valid) > 0:
            print(f"    ✓ {name}: {len(valid)} days")
    
    return signals

# ============================================================================
# STRATEGY VARIANTS
# ============================================================================
def strategy_threshold(signal_series, buy_thresh, sell_thresh, signal_name):
    """Long when signal in buy zone, flat when in sell zone."""
    if signal_name == 'sopr':
        # SOPR < 1 = capitulation buy, >= 1 = hold in bull
        pos = (signal_series < buy_thresh).astype(float)
    elif signal_name == 'nvt':
        # NVT < 30 = undervalued (buy), > 150 = overvalued (sell)
        pos = (signal_series < buy_thresh).astype(float)
    elif signal_name in ('reserve_risk',):
        # Low reserve risk = buy zone
        pos = (signal_series < buy_thresh).astype(float)
    else:
        # MVRV < 1 = buy, > 3.7 = sell; NUPL < 0 = buy, > 0.75 = sell
        pos = pd.Series(0.0, index=signal_series.index)
        pos[signal_series < buy_thresh] = 1.0
        pos[signal_series > sell_thresh] = 0.0  # flat in sell zone
        # Between thresholds: maintain previous position
        pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
    return pos

def strategy_zscore(signal_series, window=30):
    """Long when z-score < -1.5, flat when > 1.5."""
    rolling_mean = signal_series.rolling(window, min_periods=10).mean()
    rolling_std = signal_series.rolling(window, min_periods=10).std()
    z = (signal_series - rolling_mean) / rolling_std.replace(0, np.nan)
    pos = pd.Series(np.nan, index=signal_series.index)
    pos[z < -1.5] = 1.0
    pos[z > 1.5] = 0.0
    pos = pos.ffill().fillna(0.0)
    return pos

def strategy_trend(signal_series, window=20):
    """Long when signal's 20d SMA is rising."""
    sma = signal_series.rolling(window, min_periods=10).mean()
    rising = sma.diff() > 0
    return rising.astype(float)

def strategy_v4_filter(signal_series, sma50_pos, buy_thresh, sell_thresh, signal_name):
    """SMA50 long, but reduce 50% when on-chain signals overheated."""
    pos = sma50_pos.copy()
    if signal_name in ('mvrv', 'nupl', 'two_year_ma', 'pi_cycle'):
        overheated = signal_series > sell_thresh
    elif signal_name == 'nvt':
        overheated = signal_series > sell_thresh
    elif signal_name == 'reserve_risk':
        overheated = signal_series > sell_thresh
    else:
        overheated = pd.Series(False, index=signal_series.index)
    pos[overheated] = pos[overheated] * 0.5
    return pos

def strategy_v4_boost(signal_series, sma50_pos, buy_thresh, signal_name):
    """SMA50 long + on-chain buy zone → increase position 50%."""
    pos = sma50_pos.copy()
    if signal_name == 'sopr':
        in_buy = signal_series < buy_thresh
    elif signal_name in ('mvrv', 'nupl'):
        in_buy = signal_series < buy_thresh
    elif signal_name == 'nvt':
        in_buy = signal_series < buy_thresh
    elif signal_name == 'reserve_risk':
        in_buy = signal_series < buy_thresh
    else:
        in_buy = signal_series < buy_thresh
    pos[in_buy] = pos[in_buy] * 1.5
    return pos

# ============================================================================
# BACKTEST ENGINE
# ============================================================================
def compute_metrics(returns):
    """Compute Sharpe, CAGR, MaxDD from a return series."""
    returns = returns.dropna()
    if len(returns) < 30:
        return {'sharpe': np.nan, 'cagr': np.nan, 'maxdd': np.nan, 'n_days': len(returns)}
    
    cum = (1 + returns).cumprod()
    n_years = len(returns) / 365.25
    
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 and cum.iloc[-1] > 0 else np.nan
    
    peak = cum.cummax()
    dd = (cum - peak) / peak
    maxdd = dd.min()
    
    sharpe = returns.mean() / returns.std() * np.sqrt(365.25) if returns.std() > 0 else 0.0
    
    return {
        'sharpe': round(float(sharpe), 4),
        'cagr': round(float(cagr), 4),
        'maxdd': round(float(maxdd), 4),
        'n_days': int(len(returns)),
    }

def walk_forward_backtest(positions, fwd_returns, n_folds=14):
    """
    Expanding walk-forward backtest.
    Signal bar N, trade bar N+1 (positions already aligned to signal day).
    """
    # Shift positions by 1 to avoid look-ahead
    shifted_pos = positions.shift(1)
    
    # Align
    common = shifted_pos.dropna().index.intersection(fwd_returns.dropna().index)
    pos = shifted_pos.loc[common]
    ret = fwd_returns.loc[common]
    
    if len(common) < 100:
        return None, None
    
    strat_returns = pos * ret
    
    # Walk-forward: expanding window
    fold_size = len(common) // n_folds
    oos_returns = []
    
    for i in range(1, n_folds):
        train_end = fold_size * i
        test_end = min(fold_size * (i + 1), len(common))
        if test_end <= train_end:
            continue
        oos_ret = strat_returns.iloc[train_end:test_end]
        oos_returns.append(oos_ret)
    
    if not oos_returns:
        return None, None
    
    oos_all = pd.concat(oos_returns)
    full_metrics = compute_metrics(oos_all)
    
    return full_metrics, oos_all

def permutation_test(positions, fwd_returns, observed_sharpe, n_perms=500):
    """Permutation test: shuffle signal-return alignment."""
    if np.isnan(observed_sharpe):
        return 1.0
    
    shifted_pos = positions.shift(1)
    common = shifted_pos.dropna().index.intersection(fwd_returns.dropna().index)
    pos = shifted_pos.loc[common].values
    ret = fwd_returns.loc[common].values
    
    if len(common) < 100:
        return 1.0
    
    count_better = 0
    rng = np.random.RandomState(42)
    
    for _ in range(n_perms):
        shuffled_ret = rng.permutation(ret)
        perm_strat = pos * shuffled_ret
        perm_strat_clean = perm_strat[~np.isnan(perm_strat)]
        if len(perm_strat_clean) < 30 or np.std(perm_strat_clean) == 0:
            continue
        perm_sharpe = np.mean(perm_strat_clean) / np.std(perm_strat_clean) * np.sqrt(365.25)
        if perm_sharpe >= observed_sharpe:
            count_better += 1
    
    return count_better / n_perms

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("BTC WIZ ON-CHAIN SIGNAL BATTERY")
    print("=" * 70)
    
    # 1. Load data
    print("\n📊 Loading data...")
    price_df = load_price_data()
    
    print("\n🔗 Fetching on-chain signals from BGeometrics...")
    onchain_signals = fetch_onchain_data()
    
    print("\n📈 Computing price-derived signals...")
    price_signals = compute_price_signals(price_df)
    
    # Merge all signals
    all_signals = {**onchain_signals, **price_signals}
    
    # SMA50 baseline positions
    sma50_pos = price_signals.get('sma50_trend', pd.DataFrame()).get('sma50_trend', pd.Series())
    
    # 2. Signal descriptions and availability
    print("\n" + "=" * 70)
    print("SIGNAL DESCRIPTIONS & DATA AVAILABILITY")
    print("=" * 70)
    for name, df in sorted(all_signals.items()):
        if name == 'sma50_trend':
            continue
        thresh = THRESHOLDS.get(name, {})
        desc = thresh.get('name', name)
        buy = thresh.get('buy', 'N/A')
        sell = thresh.get('sell', 'N/A')
        n = len(df)
        start = df.index.min().date() if hasattr(df.index.min(), 'date') else 'N/A'
        end = df.index.max().date() if hasattr(df.index.max(), 'date') else 'N/A'
        print(f"  {desc:25s} | {n:5d} days | {start} → {end} | Buy<{buy} Sell>{sell}")
    
    # 3. Run tests
    print("\n" + "=" * 70)
    print("RUNNING SIGNAL BATTERY")
    print("=" * 70)
    
    results = []
    total_tests = 0
    
    # Count total tests for Bonferroni
    for name in all_signals:
        if name == 'sma50_trend':
            continue
        total_tests += 5  # threshold, zscore, trend, v4_filter, v4_boost
    
    bonferroni_alpha = ALPHA / total_tests
    print(f"\nTotal tests: {total_tests}, Bonferroni α: {bonferroni_alpha:.6f}")
    
    fwd_ret = price_df['fwd_ret']
    
    # SMA50 baseline
    print("\n--- SMA50 BASELINE ---")
    if len(sma50_pos) > 0:
        base_metrics, _ = walk_forward_backtest(sma50_pos, fwd_ret)
        if base_metrics:
            print(f"  SMA50: Sharpe={base_metrics['sharpe']:.3f} CAGR={base_metrics['cagr']:.3%} MaxDD={base_metrics['maxdd']:.3%}")
            results.append({
                'signal': 'SMA50_baseline', 'variant': 'baseline',
                **base_metrics, 'p_value': None, 'significant': None
            })
    
    # Buy & Hold baseline
    bh_ret = fwd_ret.dropna()
    bh_metrics = compute_metrics(bh_ret)
    print(f"  Buy&Hold: Sharpe={bh_metrics['sharpe']:.3f} CAGR={bh_metrics['cagr']:.3%} MaxDD={bh_metrics['maxdd']:.3%}")
    results.append({
        'signal': 'BuyAndHold', 'variant': 'baseline',
        **bh_metrics, 'p_value': None, 'significant': None
    })
    
    # Test each signal
    for sig_name in sorted(all_signals.keys()):
        if sig_name == 'sma50_trend':
            continue
        
        sig_df = all_signals[sig_name]
        sig_col = sig_name
        if isinstance(sig_df, pd.DataFrame):
            sig_col = sig_df.columns[0]
            sig_series = sig_df[sig_col]
        else:
            sig_series = sig_df
        
        thresh = THRESHOLDS.get(sig_name, {})
        buy_t = thresh.get('buy', None)
        sell_t = thresh.get('sell', None)
        desc = thresh.get('name', sig_name)
        
        print(f"\n--- {desc.upper()} ---")
        
        # Align with price data
        common_idx = sig_series.index.intersection(price_df.index)
        if len(common_idx) < 200:
            print(f"  ⚠ Insufficient overlap: {len(common_idx)} days, skipping")
            continue
        
        sig_aligned = sig_series.loc[common_idx]
        
        variants = {}
        
        # 1. Threshold
        if buy_t is not None and sell_t is not None:
            variants['threshold'] = strategy_threshold(sig_aligned, buy_t, sell_t, sig_name)
        
        # 2. Z-score contrarian
        variants['zscore'] = strategy_zscore(sig_aligned)
        
        # 3. Trend following
        variants['trend'] = strategy_trend(sig_aligned)
        
        # 4. V4 + filter
        if sell_t is not None and len(sma50_pos) > 0:
            sma_aligned = sma50_pos.reindex(common_idx).fillna(0)
            variants['v4_filter'] = strategy_v4_filter(sig_aligned, sma_aligned, buy_t, sell_t, sig_name)
        
        # 5. V4 + boost
        if buy_t is not None and len(sma50_pos) > 0:
            sma_aligned = sma50_pos.reindex(common_idx).fillna(0)
            variants['v4_boost'] = strategy_v4_boost(sig_aligned, sma_aligned, buy_t, sig_name)
        
        for var_name, positions in variants.items():
            metrics, oos_ret = walk_forward_backtest(positions, fwd_ret)
            
            if metrics is None or np.isnan(metrics['sharpe']):
                print(f"  {var_name:15s}: insufficient data")
                continue
            
            # Permutation test
            p_val = permutation_test(positions, fwd_ret, metrics['sharpe'], N_PERMS)
            sig = p_val < bonferroni_alpha
            
            status = "✅ SIG" if sig else "  "
            print(f"  {var_name:15s}: Sharpe={metrics['sharpe']:+.3f} CAGR={metrics['cagr']:+.3%} MaxDD={metrics['maxdd']:.3%} p={p_val:.4f} {status}")
            
            results.append({
                'signal': sig_name,
                'variant': var_name,
                **metrics,
                'p_value': round(p_val, 6),
                'significant': bool(sig),
            })
    
    # 4. Composite signal
    print("\n" + "=" * 70)
    print("MULTI-SIGNAL COMPOSITE")
    print("=" * 70)
    
    # Build composite from top signals: normalized z-scores
    composite_signals = ['mvrv', 'nupl', 'sopr', 'reserve_risk']
    available = [s for s in composite_signals if s in all_signals]
    
    if len(available) >= 2:
        print(f"  Using signals: {available}")
        
        # Normalize each signal to z-score, then average
        z_scores = {}
        for s in available:
            df = all_signals[s]
            if isinstance(df, pd.DataFrame):
                series = df.iloc[:, 0]
            else:
                series = df
            z = (series - series.rolling(365, min_periods=90).mean()) / series.rolling(365, min_periods=90).std()
            z_scores[s] = z
        
        # Align all
        composite_df = pd.DataFrame(z_scores)
        composite_avg = composite_df.mean(axis=1).dropna()
        
        # Contrarian: low composite z = buy zone
        composite_pos = pd.Series(0.0, index=composite_avg.index)
        composite_pos[composite_avg < -0.5] = 1.0
        composite_pos[composite_avg > 1.0] = 0.0
        composite_pos = composite_pos.replace(0.0, np.nan).ffill().fillna(0.0)
        
        metrics, _ = walk_forward_backtest(composite_pos, fwd_ret)
        if metrics and not np.isnan(metrics['sharpe']):
            p_val = permutation_test(composite_pos, fwd_ret, metrics['sharpe'], N_PERMS)
            sig = p_val < bonferroni_alpha
            status = "✅ SIG" if sig else ""
            print(f"  Composite contrarian: Sharpe={metrics['sharpe']:+.3f} CAGR={metrics['cagr']:+.3%} MaxDD={metrics['maxdd']:.3%} p={p_val:.4f} {status}")
            results.append({
                'signal': 'composite_contrarian',
                'variant': 'multi_signal',
                **metrics,
                'p_value': round(p_val, 6),
                'significant': bool(sig),
            })
        
        # V4 overlay: SMA50 + composite filter/boost
        if len(sma50_pos) > 0:
            sma_aligned = sma50_pos.reindex(composite_avg.index).fillna(0)
            
            # Filter: reduce when composite overheated
            v4_filter_pos = sma_aligned.copy()
            v4_filter_pos[composite_avg > 1.0] = v4_filter_pos[composite_avg > 1.0] * 0.5
            
            metrics_f, _ = walk_forward_backtest(v4_filter_pos, fwd_ret)
            if metrics_f and not np.isnan(metrics_f['sharpe']):
                p_val_f = permutation_test(v4_filter_pos, fwd_ret, metrics_f['sharpe'], N_PERMS)
                sig_f = p_val_f < bonferroni_alpha
                status = "✅ SIG" if sig_f else ""
                print(f"  V4+composite filter: Sharpe={metrics_f['sharpe']:+.3f} CAGR={metrics_f['cagr']:+.3%} MaxDD={metrics_f['maxdd']:.3%} p={p_val_f:.4f} {status}")
                results.append({
                    'signal': 'composite_v4_filter',
                    'variant': 'v4_overlay',
                    **metrics_f,
                    'p_value': round(p_val_f, 6),
                    'significant': bool(sig_f),
                })
            
            # Boost: increase when composite in buy zone
            v4_boost_pos = sma_aligned.copy()
            v4_boost_pos[composite_avg < -0.5] = v4_boost_pos[composite_avg < -0.5] * 1.5
            
            metrics_b, _ = walk_forward_backtest(v4_boost_pos, fwd_ret)
            if metrics_b and not np.isnan(metrics_b['sharpe']):
                p_val_b = permutation_test(v4_boost_pos, fwd_ret, metrics_b['sharpe'], N_PERMS)
                sig_b = p_val_b < bonferroni_alpha
                status = "✅ SIG" if sig_b else ""
                print(f"  V4+composite boost:  Sharpe={metrics_b['sharpe']:+.3f} CAGR={metrics_b['cagr']:+.3%} MaxDD={metrics_b['maxdd']:.3%} p={p_val_b:.4f} {status}")
                results.append({
                    'signal': 'composite_v4_boost',
                    'variant': 'v4_overlay',
                    **metrics_b,
                    'p_value': round(p_val_b, 6),
                    'significant': bool(sig_b),
                })
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Signal':25s} {'Variant':15s} {'Sharpe':>8s} {'CAGR':>10s} {'MaxDD':>10s} {'p-value':>10s} {'Sig':>5s}")
    print("-" * 85)
    
    for r in sorted(results, key=lambda x: -x['sharpe'] if not np.isnan(x['sharpe']) else -999):
        sig_mark = "✅" if r.get('significant') else ""
        p_str = f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A"
        print(f"{r['signal']:25s} {r['variant']:15s} {r['sharpe']:+8.3f} {r['cagr']:+10.3%} {r['maxdd']:10.3%} {p_str:>10s} {sig_mark:>5s}")
    
    # Which beat SMA50?
    sma50_sharpe = next((r['sharpe'] for r in results if r['signal'] == 'SMA50_baseline'), None)
    if sma50_sharpe is not None:
        print(f"\n{'='*70}")
        print(f"SIGNALS BEATING SMA50 BASELINE (Sharpe {sma50_sharpe:.3f})")
        print(f"{'='*70}")
        beaters = [r for r in results if r['signal'] not in ('SMA50_baseline', 'BuyAndHold') 
                   and r['sharpe'] > sma50_sharpe]
        if beaters:
            for r in sorted(beaters, key=lambda x: -x['sharpe']):
                sig_mark = " ✅" if r.get('significant') else ""
                print(f"  {r['signal']:25s} {r['variant']:15s} Sharpe={r['sharpe']:+.3f}{sig_mark}")
        else:
            print("  None beat SMA50 baseline.")
    
    # Statistically significant results
    sig_results = [r for r in results if r.get('significant')]
    print(f"\n{'='*70}")
    print(f"STATISTICALLY SIGNIFICANT RESULTS (Bonferroni α={bonferroni_alpha:.6f})")
    print(f"{'='*70}")
    if sig_results:
        for r in sorted(sig_results, key=lambda x: -x['sharpe']):
            print(f"  {r['signal']:25s} {r['variant']:15s} Sharpe={r['sharpe']:+.3f} p={r['p_value']:.6f}")
    else:
        print("  No statistically significant results after Bonferroni correction.")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDED COMPOSITE FOR V4 LAYER 4")
    print(f"{'='*70}")
    
    # Find best V4 overlay
    v4_results = [r for r in results if 'v4' in r['variant'] or 'composite' in r['signal']]
    if v4_results:
        best_v4 = max(v4_results, key=lambda x: x['sharpe'] if not np.isnan(x['sharpe']) else -999)
        print(f"  Best V4 overlay: {best_v4['signal']} ({best_v4['variant']})")
        print(f"  Sharpe={best_v4['sharpe']:+.3f} CAGR={best_v4['cagr']:+.3%} MaxDD={best_v4['maxdd']:.3%}")
        if best_v4.get('significant'):
            print("  ✅ Statistically significant — RECOMMEND for production")
        else:
            print(f"  ⚠ Not significant after Bonferroni (p={best_v4['p_value']:.4f})")
            print("  Consider as informational overlay only, not hard signal")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'walk_forward_folds': N_FOLDS,
            'permutation_tests': N_PERMS,
            'bonferroni_alpha': bonferroni_alpha,
            'total_tests': total_tests,
            'look_ahead_prevention': 'signal bar N, trade bar N+1',
        },
        'results': results,
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == '__main__':
    main()
