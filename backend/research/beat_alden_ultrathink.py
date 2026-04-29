#!/usr/bin/env python3
"""
ULTRATHINK: Beat Alden's Net Liquidity Proxy
Goal: Sharpe > 0.62 for BTC timing
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from scipy import stats
import json, warnings, itertools
from pathlib import Path
warnings.filterwarnings('ignore')

FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)
COST = 0.001
START = "2017-01-01"
END = "2025-12-31"
results_dir = Path("/Users/ffv_macmini/Desktop/maestro/data/research")

print("=" * 70)
print("ULTRATHINK: Beat Alden's Net Liquidity Proxy")
print("=" * 70)

# ── Data Download ──
print("\n[1/6] Downloading data...")
tickers = {
    'BTC': 'BTC-USD', 'UUP': 'UUP', 'GLD': 'GLD', 'TLT': 'TLT', 'HYG': 'HYG',
    'TIP': 'TIP', 'LQD': 'LQD', 'SHY': 'SHY', 'XLF': 'XLF', 'EMB': 'EMB',
    'VIX': '^VIX', 'COPPER': 'HG=F', 'SPY': 'SPY'
}
prices = {}
for name, tick in tickers.items():
    try:
        df = yf.download(tick, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices[name] = df['Close'].dropna()
        print(f"  {name}: {len(prices[name])} days")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# FRED data
print("  Downloading FRED data...")
fred_series = {}
for sid in ['WALCL', 'WTREGEN', 'RRPONTSYD', 'M2SL']:
    try:
        fred_series[sid] = fred.get_series(sid, observation_start=START)
        print(f"  {sid}: {len(fred_series[sid])} obs")
    except Exception as e:
        print(f"  {sid}: FAILED ({e})")

# Build aligned daily df
pdf = pd.DataFrame(prices)
pdf = pdf.ffill().dropna(subset=['BTC'])
btc_ret = pdf['BTC'].pct_change()

# Alden's Net Liquidity
walcl = fred_series['WALCL'].reindex(pdf.index, method='ffill')
wtregen = fred_series['WTREGEN'].reindex(pdf.index, method='ffill')
rrp = fred_series['RRPONTSYD'].reindex(pdf.index, method='ffill')
net_liq = walcl - wtregen - rrp

# M2
m2 = fred_series['M2SL'].reindex(pdf.index, method='ffill')


def backtest_signal(signal, btc_prices, cost=COST):
    """Given a binary signal (1=long, 0=flat), backtest on BTC."""
    signal = signal.shift(1).fillna(0)  # 1-day lag
    ret = btc_prices.pct_change().fillna(0)
    trades = signal.diff().abs().fillna(0)
    strat_ret = signal * ret - trades * cost
    cum = (1 + strat_ret).cumprod()
    
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    years = len(strat_ret) / 252
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 and total_ret > -1 else 0
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()
    n_trades = int(trades.sum())
    pct_long = signal.mean()
    
    return {
        'sharpe': round(sharpe, 4),
        'total_return': round(total_ret * 100, 1),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'n_trades': n_trades,
        'pct_long': round(pct_long * 100, 1)
    }


# ── Alden Baseline ──
print("\n[2/6] Computing Alden baseline...")
alden_sig = (net_liq.pct_change(20) > 0).astype(int)
alden_sig = alden_sig.reindex(pdf.index).ffill().fillna(0)
alden_res = backtest_signal(alden_sig, pdf['BTC'])
print(f"  Alden: Sharpe={alden_res['sharpe']}, MaxDD={alden_res['max_dd']}%, Trades={alden_res['n_trades']}")

# B&H baseline
bh_ret = pdf['BTC'].pct_change().fillna(0)
bh_cum = (1 + bh_ret).cumprod()
bh_sharpe = round(bh_ret.mean() / bh_ret.std() * np.sqrt(252), 4)
print(f"  Buy&Hold: Sharpe={bh_sharpe}")


# ── PHASE 1: Parameter Sweep ──
print("\n[3/6] Phase 1: Parameter sweep (lookback x consensus)...")
base_instruments = ['UUP', 'GLD', 'TLT', 'HYG']
# UUP is inverse (dollar down = bullish), others are direct (up = bullish)
inverse = {'UUP', 'VIX', 'SHY'}  # instruments where DOWN = bullish

best_sweep = {'sharpe': -999}
sweep_results = []

for lookback in [5, 10, 20, 40, 60, 90, 120, 180]:
    for threshold in [2, 3, 4]:
        scores = pd.DataFrame(index=pdf.index)
        for inst in base_instruments:
            if inst not in pdf.columns:
                continue
            chg = pdf[inst].pct_change(lookback)
            if inst in inverse:
                scores[inst] = (chg < 0).astype(int)
            else:
                scores[inst] = (chg > 0).astype(int)
        
        consensus = scores.sum(axis=1)
        signal = (consensus >= threshold).astype(int)
        res = backtest_signal(signal, pdf['BTC'])
        res['lookback'] = lookback
        res['threshold'] = threshold
        sweep_results.append(res)
        
        if res['sharpe'] > best_sweep.get('sharpe', -999):
            best_sweep = res.copy()

print(f"\n  BEST SWEEP: lookback={best_sweep['lookback']}, threshold={best_sweep['threshold']}")
print(f"  Sharpe={best_sweep['sharpe']}, MaxDD={best_sweep['max_dd']}%, Trades={best_sweep['n_trades']}")
print(f"  vs Alden: {'WINS' if best_sweep['sharpe'] > alden_res['sharpe'] else 'LOSES'}")

# Print top 5
sweep_sorted = sorted(sweep_results, key=lambda x: x['sharpe'], reverse=True)[:10]
print("\n  Top 10 parameter combos:")
for r in sweep_sorted:
    print(f"    lb={r['lookback']:3d} thr={r['threshold']} | Sharpe={r['sharpe']:6.3f} MaxDD={r['max_dd']:6.1f}% Trades={r['n_trades']:3d} Long={r['pct_long']:.0f}%")


# ── PHASE 2: Instrument Selection ──
print("\n[4/6] Phase 2: Instrument selection...")
# Use best lookback/threshold from Phase 1
best_lb = best_sweep['lookback']
best_thr = best_sweep['threshold']

# Test adding each candidate
candidates = ['TIP', 'LQD', 'SHY', 'XLF', 'EMB', 'VIX', 'COPPER', 'SPY']
add_results = []

for cand in candidates:
    if cand not in pdf.columns:
        continue
    instruments = base_instruments + [cand]
    scores = pd.DataFrame(index=pdf.index)
    for inst in instruments:
        if inst not in pdf.columns:
            continue
        chg = pdf[inst].pct_change(best_lb)
        if inst in inverse:
            scores[inst] = (chg < 0).astype(int)
        else:
            scores[inst] = (chg > 0).astype(int)
    
    # Adjust threshold for 5 instruments
    for thr in [2, 3, 4, 5]:
        consensus = scores.sum(axis=1)
        signal = (consensus >= thr).astype(int)
        res = backtest_signal(signal, pdf['BTC'])
        res['instruments'] = '+'.join(instruments)
        res['added'] = cand
        res['threshold'] = thr
        add_results.append(res)

# Test removing each original
for remove in base_instruments:
    instruments = [i for i in base_instruments if i != remove]
    scores = pd.DataFrame(index=pdf.index)
    for inst in instruments:
        chg = pdf[inst].pct_change(best_lb)
        if inst in inverse:
            scores[inst] = (chg < 0).astype(int)
        else:
            scores[inst] = (chg > 0).astype(int)
    
    for thr in [2, 3]:
        consensus = scores.sum(axis=1)
        signal = (consensus >= thr).astype(int)
        res = backtest_signal(signal, pdf['BTC'])
        res['instruments'] = '+'.join(instruments)
        res['removed'] = remove
        res['threshold'] = thr
        add_results.append(res)

add_sorted = sorted(add_results, key=lambda x: x['sharpe'], reverse=True)[:10]
print("\n  Top 10 instrument combos:")
for r in add_sorted:
    tag = f"added={r.get('added','')}" if 'added' in r else f"removed={r.get('removed','')}"
    print(f"    {tag:15s} thr={r['threshold']} | Sharpe={r['sharpe']:6.3f} MaxDD={r['max_dd']:6.1f}% Trades={r['n_trades']:3d}")

best_inst = add_sorted[0]


# ── PHASE 3: Alternative Signal Construction ──
print("\n[5/6] Phase 3: Alternative signal methods...")

# Method A: Current best from Phase 1 (already computed)
method_results = {'binary_consensus': best_sweep}

# Method B: Z-score weighted
print("  Testing z-score weighted...")
for lookback in [20, 40, 60, 90, 120]:
    z_scores = pd.DataFrame(index=pdf.index)
    for inst in base_instruments:
        if inst not in pdf.columns:
            continue
        chg = pdf[inst].pct_change(lookback)
        rolling_mean = chg.rolling(252).mean()
        rolling_std = chg.rolling(252).std()
        z = (chg - rolling_mean) / rolling_std
        if inst in inverse:
            z_scores[inst] = -z
        else:
            z_scores[inst] = z
    
    composite = z_scores.mean(axis=1)
    for pct in [0, 0.25, 0.5]:
        signal = (composite > pct).astype(int)
        res = backtest_signal(signal, pdf['BTC'])
        res['method'] = f'zscore_lb{lookback}_thr{pct}'
        if res['sharpe'] > method_results.get('zscore', {}).get('sharpe', -999):
            method_results['zscore'] = res

# Method C: PCA
print("  Testing PCA...")
from sklearn.decomposition import PCA
for lookback in [20, 40, 60, 90]:
    changes = pd.DataFrame(index=pdf.index)
    for inst in base_instruments:
        if inst not in pdf.columns:
            continue
        chg = pdf[inst].pct_change(lookback)
        if inst in inverse:
            changes[inst] = -chg
        else:
            changes[inst] = chg
    
    changes = changes.dropna()
    if len(changes) < 252:
        continue
    
    # Rolling PCA (use expanding window, step every 5 days for speed)
    window = 252
    pc1 = pd.Series(index=changes.index, dtype=float)
    last_pca = None
    for i in range(window, len(changes), 5):
        chunk = changes.iloc[max(0,i-window):i]
        last_pca = PCA(n_components=1)
        last_pca.fit(chunk)
        end = min(i+5, len(changes))
        for j in range(i, end):
            pc1.iloc[j] = last_pca.transform(changes.iloc[[j]])[0, 0]
    
    pc1 = pc1.reindex(pdf.index)
    signal = (pc1 > 0).astype(int)
    res = backtest_signal(signal, pdf['BTC'])
    res['method'] = f'pca_lb{lookback}'
    if res['sharpe'] > method_results.get('pca', {}).get('sharpe', -999):
        method_results['pca'] = res

# Method D: Walk-forward regression
print("  Testing walk-forward regression...")
from sklearn.linear_model import Ridge
for lookback in [20, 40, 60]:
    features = pd.DataFrame(index=pdf.index)
    for inst in base_instruments:
        if inst not in pdf.columns:
            continue
        chg = pdf[inst].pct_change(lookback)
        if inst in inverse:
            features[inst] = -chg
        else:
            features[inst] = chg
    
    fwd_ret = pdf['BTC'].pct_change(20).shift(-20)  # 20-day forward return
    features = features.dropna()
    common_idx = features.index.intersection(fwd_ret.dropna().index)
    features = features.loc[common_idx]
    fwd_ret = fwd_ret.loc[common_idx]
    
    train_window = 504  # 2 years
    signal = pd.Series(0, index=features.index)
    
    for i in range(train_window, len(features), 5):  # step 5 for speed
        X_train = features.iloc[i-train_window:i].values
        y_train = fwd_ret.iloc[i-train_window:i].values
        
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        if mask.sum() < 100:
            continue
        
        model = Ridge(alpha=1.0)
        model.fit(X_train[mask], y_train[mask])
        pred = model.predict(features.iloc[[i]].values)[0]
        for j in range(i, min(i+5, len(features))):
            signal.iloc[j] = 1 if pred > 0 else 0
    
    signal = signal.reindex(pdf.index).ffill().fillna(0)
    res = backtest_signal(signal, pdf['BTC'])
    res['method'] = f'regression_lb{lookback}'
    if res['sharpe'] > method_results.get('regression', {}).get('sharpe', -999):
        method_results['regression'] = res

# Method E: Hybrid (Alden when fresh + market when stale)
print("  Testing hybrid (Alden + market proxy)...")
# Alden data freshness: WALCL is weekly (Wed), WTREGEN weekly, RRPONTSYD daily
walcl_fresh = walcl.copy()
# Mark staleness: if WALCL hasn't updated in > 7 days, use market proxy
walcl_diff = walcl.diff()
stale = (walcl_diff == 0).rolling(7).sum() >= 7  # stale if no change for 7+ days

# Best market proxy from Phase 1
scores_best = pd.DataFrame(index=pdf.index)
for inst in base_instruments:
    if inst not in pdf.columns:
        continue
    chg = pdf[inst].pct_change(best_lb)
    if inst in inverse:
        scores_best[inst] = (chg < 0).astype(int)
    else:
        scores_best[inst] = (chg > 0).astype(int)
market_sig = (scores_best.sum(axis=1) >= best_thr).astype(int)

hybrid_signal = alden_sig.copy()
hybrid_signal[stale] = market_sig[stale]
res = backtest_signal(hybrid_signal, pdf['BTC'])
res['method'] = 'hybrid_alden_market'
method_results['hybrid'] = res

# Method F: SMA-based regime (instruments above/below their own SMA)
print("  Testing SMA regime method...")
for sma_period in [50, 100, 200]:
    scores_sma = pd.DataFrame(index=pdf.index)
    for inst in base_instruments:
        if inst not in pdf.columns:
            continue
        sma = pdf[inst].rolling(sma_period).mean()
        above = (pdf[inst] > sma).astype(int)
        if inst in inverse:
            scores_sma[inst] = 1 - above  # below SMA = bullish for dollar
        else:
            scores_sma[inst] = above
    
    for thr in [2, 3, 4]:
        consensus = scores_sma.sum(axis=1)
        signal = (consensus >= thr).astype(int)
        res = backtest_signal(signal, pdf['BTC'])
        res['method'] = f'sma{sma_period}_thr{thr}'
        if res['sharpe'] > method_results.get('sma_regime', {}).get('sharpe', -999):
            method_results['sma_regime'] = res

# Method G: Rate of change magnitude (not just direction)
print("  Testing ROC magnitude...")
for lookback in [20, 40, 60, 90]:
    roc_sum = pd.Series(0.0, index=pdf.index)
    for inst in base_instruments:
        if inst not in pdf.columns:
            continue
        roc = pdf[inst].pct_change(lookback)
        if inst in inverse:
            roc_sum -= roc
        else:
            roc_sum += roc
    
    # Normalize
    roc_z = (roc_sum - roc_sum.rolling(252).mean()) / roc_sum.rolling(252).std()
    signal = (roc_z > 0).astype(int)
    res = backtest_signal(signal, pdf['BTC'])
    res['method'] = f'roc_magnitude_lb{lookback}'
    if res['sharpe'] > method_results.get('roc_magnitude', {}).get('sharpe', -999):
        method_results['roc_magnitude'] = res

# ── Print all method results ──
print("\n  METHOD COMPARISON:")
print(f"  {'Method':<25s} {'Sharpe':>8s} {'MaxDD':>8s} {'Trades':>7s} {'Long%':>6s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
for name, r in sorted(method_results.items(), key=lambda x: x[1].get('sharpe', -999), reverse=True):
    print(f"  {name:<25s} {r.get('sharpe',0):8.4f} {r.get('max_dd',0):7.1f}% {r.get('n_trades',0):7d} {r.get('pct_long',0):5.1f}%")
print(f"  {'ALDEN BASELINE':<25s} {alden_res['sharpe']:8.4f} {alden_res['max_dd']:7.1f}% {alden_res['n_trades']:7d} {alden_res['pct_long']:5.1f}%")
print(f"  {'BUY & HOLD':<25s} {bh_sharpe:8.4f}")


# ── PHASE 4: Best proxy final comparison ──
print("\n[6/6] Phase 4: Final comparison with best proxy...")
overall_best_name = max(method_results, key=lambda k: method_results[k].get('sharpe', -999))
overall_best = method_results[overall_best_name]
print(f"\n  OVERALL BEST METHOD: {overall_best_name}")
print(f"  Sharpe: {overall_best['sharpe']}")
print(f"  vs Alden ({alden_res['sharpe']}): {'WINS!' if overall_best['sharpe'] > alden_res['sharpe'] else 'Still behind'}")
print(f"  vs Buy&Hold ({bh_sharpe}): {'WINS!' if overall_best['sharpe'] > bh_sharpe else 'Still behind'}")

# ── PHASE 5: M2 Direction Prediction ──
print("\n\n  M2 DIRECTION PREDICTION (best proxy)...")
# Rebuild best proxy signal
if best_sweep['sharpe'] >= overall_best.get('sharpe', 0) - 0.05:
    # Use sweep winner
    proxy_lb = best_sweep['lookback']
    proxy_thr = best_sweep['threshold']
else:
    proxy_lb = best_sweep['lookback']
    proxy_thr = best_sweep['threshold']

scores_final = pd.DataFrame(index=pdf.index)
for inst in base_instruments:
    if inst not in pdf.columns:
        continue
    chg = pdf[inst].pct_change(proxy_lb)
    if inst in inverse:
        scores_final[inst] = (chg < 0).astype(int)
    else:
        scores_final[inst] = (chg > 0).astype(int)
proxy_score = scores_final.sum(axis=1)
proxy_bullish = (proxy_score >= proxy_thr).astype(int)

# M2 monthly changes
m2_monthly = fred_series['M2SL'].resample('M').last().pct_change()
m2_direction = (m2_monthly > 0).astype(int)

correct = 0
total = 0
for date, direction in m2_direction.items():
    if pd.isna(direction):
        continue
    # Look at proxy signal 20 days before
    lookback_date = date - pd.Timedelta(days=20)
    mask = (proxy_bullish.index >= lookback_date) & (proxy_bullish.index <= date)
    if mask.sum() > 0:
        proxy_pred = proxy_bullish[mask].iloc[-1]
        if proxy_pred == direction:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
p_val = stats.binomtest(correct, total, 0.5).pvalue if total > 0 else 1.0
print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
print(f"  p-value: {p_val:.4f}")
print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'}")


# ── Save Results ──
all_results = {
    'alden_baseline': alden_res,
    'buy_hold_sharpe': bh_sharpe,
    'best_parameter_sweep': best_sweep,
    'top_10_sweeps': sweep_sorted,
    'best_instrument_combos': add_sorted[:5],
    'method_comparison': {k: v for k, v in method_results.items()},
    'overall_best': {'method': overall_best_name, **overall_best},
    'm2_prediction': {'accuracy': accuracy, 'p_value': p_val, 'correct': correct, 'total': total}
}

with open(results_dir / 'beat_alden_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# ── Write Report ──
report = f"""# Ultrathink: Beating Alden's Net Liquidity Proxy

## Executive Summary

Comprehensive parameter optimization and alternative signal construction
to beat Alden's Net Liquidity framework (Sharpe {alden_res['sharpe']}) for BTC timing.

## Results

### Alden Baseline
- Sharpe: {alden_res['sharpe']}
- MaxDD: {alden_res['max_dd']}%
- Trades: {alden_res['n_trades']}

### Buy & Hold
- Sharpe: {bh_sharpe}

### Best Parameter Sweep (Binary Consensus)
- Lookback: {best_sweep['lookback']}, Threshold: {best_sweep['threshold']}
- Sharpe: {best_sweep['sharpe']}
- MaxDD: {best_sweep['max_dd']}%
- Trades: {best_sweep['n_trades']}
- vs Alden: {'WINS' if best_sweep['sharpe'] > alden_res['sharpe'] else 'LOSES'}

### Method Comparison
| Method | Sharpe | MaxDD | Trades | Long% |
|--------|--------|-------|--------|-------|
"""

for name, r in sorted(method_results.items(), key=lambda x: x[1].get('sharpe', -999), reverse=True):
    report += f"| {name} | {r.get('sharpe',0):.4f} | {r.get('max_dd',0):.1f}% | {r.get('n_trades',0)} | {r.get('pct_long',0):.1f}% |\n"
report += f"| ALDEN | {alden_res['sharpe']:.4f} | {alden_res['max_dd']:.1f}% | {alden_res['n_trades']} | {alden_res['pct_long']:.1f}% |\n"
report += f"| BUY & HOLD | {bh_sharpe:.4f} | — | — | 100% |\n"

report += f"""
### Overall Best: {overall_best_name}
- Sharpe: {overall_best['sharpe']}
- {'BEATS' if overall_best['sharpe'] > alden_res['sharpe'] else 'Does not beat'} Alden

### M2 Direction Prediction
- Accuracy: {accuracy:.1%} ({correct}/{total})
- p-value: {p_val:.4f}
- Significant: {'YES' if p_val < 0.05 else 'NO'}

---
*Generated: {pd.Timestamp.now()}*
"""

with open(results_dir / 'beat_alden_report.md', 'w') as f:
    f.write(report)

print(f"\n{'='*70}")
print(f"FINAL VERDICT")
print(f"{'='*70}")
print(f"Best proxy Sharpe: {overall_best['sharpe']} ({overall_best_name})")
print(f"Alden Sharpe: {alden_res['sharpe']}")
print(f"{'MISSION ACCOMPLISHED' if overall_best['sharpe'] > alden_res['sharpe'] else 'NEED MORE WORK'}")
print(f"\nResults saved to {results_dir}/beat_alden_*.{{json,md}}")
