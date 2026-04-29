#!/usr/bin/env python3
"""
Taker Buy/Sell Volume Imbalance - Signal Validation (Optimized)
"""

import json, warnings, sys
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

DB_PATH = Path.home() / "Desktop/maestro/data/maestro.duckdb"
OUT_PATH = Path.home() / "Desktop/maestro/data/backtest_results/taker_volume_test.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH), read_only=True)

ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'LINK', 'ADA', 'AVAX', 'DOT',
          'NEAR', 'UNI', 'ATOM', 'FIL', 'ZEC', 'DYDX', 'OP', 'INJ', 'APT', 'ARB',
          'SUI', 'CRV', 'SEI', 'TIA']

def load_data(symbol):
    taker = con.execute(f"SELECT * FROM cg_taker_volume WHERE symbol='{symbol}' ORDER BY date").df()
    price = con.execute(f"SELECT * FROM perps_daily WHERE symbol='{symbol}' ORDER BY date").df()
    taker['date'] = pd.to_datetime(taker['date']).dt.date
    price['date'] = pd.to_datetime(price['date']).dt.date
    df = pd.merge(taker, price, on='date', how='inner', suffixes=('', '_price'))
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df['buy_vol'] = df['taker_buy_volume_usd'].astype(float)
    df['sell_vol'] = df['taker_sell_volume_usd'].astype(float)
    df['ratio'] = df['buy_vol'] / df['sell_vol'].replace(0, np.nan)
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['total_vol'] = df['buy_vol'] + df['sell_vol']
    df['returns'] = df['close'].pct_change()
    df['fwd_returns'] = df['returns'].shift(-1)
    return df.dropna(subset=['ratio', 'returns']).reset_index(drop=True)

# ── Signal generators (return position: 1/0/-1) ──
def sig_imbalance_momentum(df):
    return (df['ratio'] > 1.05).astype(int)

def sig_imbalance_zscore(df):
    roll = df['ratio'].rolling(30)
    z = (df['ratio'] - roll.mean()) / roll.std()
    return (z > 1.5).astype(int)

def sig_volume_surge(df):
    avg_vol = df['total_vol'].rolling(20).mean()
    return ((df['total_vol'] > 2 * avg_vol) & (df['ratio'] > 1.0)).astype(int)

def sig_ctd(df):
    ctd = df['delta'].cumsum()
    return (ctd.rolling(5).mean() > ctd.rolling(20).mean()).astype(int)

def sig_contrarian_ls(df):
    hi = df['ratio'].expanding(60).quantile(0.9)
    lo = df['ratio'].expanding(60).quantile(0.1)
    pos = pd.Series(0, index=df.index)
    pos[df['ratio'] > hi] = -1
    pos[df['ratio'] < lo] = 1
    return pos

def sig_contrarian_lo(df):
    lo = df['ratio'].expanding(60).quantile(0.1)
    return (df['ratio'] < lo).astype(int)

def sig_sma50(df):
    return (df['close'] > df['close'].rolling(50).mean()).astype(int)

def sig_v4_confirm(df):
    trend = df['close'] > df['close'].rolling(50).mean()
    return (trend & (df['ratio'] > 1.0)).astype(int)

def sig_v4_exit(df):
    trend = df['close'] > df['close'].rolling(50).mean()
    sell_dom = (df['ratio'] < 1.0).astype(int)
    consec = sell_dom.rolling(3).sum() >= 3
    pos = trend.astype(int).copy()
    pos[trend & consec] = 0
    return pos

SIGNALS = {
    'S1_Imbalance_Mom': sig_imbalance_momentum,
    'S2_Imbalance_ZScore': sig_imbalance_zscore,
    'S3_Volume_Surge': sig_volume_surge,
    'S4_CTD_Momentum': sig_ctd,
    'S5_Contrarian_LS': sig_contrarian_ls,
    'S5b_Contrarian_LO': sig_contrarian_lo,
    'S6_V4+Taker_Confirm': sig_v4_confirm,
    'S7_V4+Taker_Exit': sig_v4_exit,
    'Baseline_SMA50': sig_sma50,
}

def compute_metrics(r):
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'maxdd': 0, 'n_days': len(r)}
    sharpe = r.mean() / r.std() * np.sqrt(365)
    cum = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    years = len(r) / 365
    cagr = (1 + total) ** (1/years) - 1 if total > -1 else -1
    maxdd = (cum / cum.cummax() - 1).min()
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr*100, 2), 
            'maxdd': round(maxdd*100, 2), 'n_days': len(r)}

def walk_forward(df, signal_func, n_folds=14):
    """Expanding walk-forward. Returns OOS strategy returns."""
    n = len(df)
    if n < 200: return None
    min_train = max(100, n // (n_folds + 1))
    fold_size = (n - min_train) // n_folds
    
    all_oos = []
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end: break
        
        signals = signal_func(df.iloc[:test_end])
        shifted = signals.shift(1)  # no look-ahead
        strat_ret = shifted.iloc[train_end:test_end] * df['returns'].iloc[train_end:test_end]
        all_oos.append(strat_ret)
    
    return pd.concat(all_oos) if all_oos else None

def fast_permutation_test(df, signal_func, observed_sharpe, n_perms=500):
    """Fast permutation: pre-compute signals, shuffle returns."""
    n = len(df)
    if n < 200: return 1.0
    
    # Pre-compute signal positions (these stay fixed)
    min_train = max(100, n // 15)
    signals = signal_func(df).shift(1)  # shifted signals for full dataset
    # Only use OOS portion (after min_train)
    sig_oos = signals.iloc[min_train:].values
    ret_oos = df['returns'].iloc[min_train:].values
    
    # Observed Sharpe on OOS
    strat = sig_oos * ret_oos
    valid = ~np.isnan(strat)
    strat = strat[valid]
    
    if len(strat) < 50 or np.std(strat) == 0:
        return 1.0
    
    obs = np.mean(strat) / np.std(strat) * np.sqrt(365)
    
    count = 0
    for _ in range(n_perms):
        perm_ret = np.random.permutation(ret_oos)
        perm_strat = sig_oos * perm_ret
        ps = perm_strat[valid]
        if np.std(ps) > 0:
            perm_sharpe = np.mean(ps) / np.std(ps) * np.sqrt(365)
            if perm_sharpe >= obs:
                count += 1
    
    return (count + 1) / (n_perms + 1)

# ── Main ──
print("=" * 80)
print("TAKER BUY/SELL VOLUME IMBALANCE - SIGNAL VALIDATION")
print("=" * 80)
sys.stdout.flush()

all_data = {}
for sym in ASSETS:
    try:
        df = load_data(sym)
        if len(df) >= 500:
            all_data[sym] = df
            print(f"  {sym}: {len(df)} days")
    except Exception as e:
        print(f"  {sym}: FAILED - {e}")
sys.stdout.flush()

# ── Descriptive Stats ──
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

desc_stats = {}
for sym, df in all_data.items():
    ratio = df['ratio'].dropna()
    merged = pd.concat([df['ratio'], df['fwd_returns']], axis=1).dropna()
    corr = merged.corr().iloc[0, 1] if len(merged) > 30 else np.nan
    ac1 = ratio.autocorr(1) if len(ratio) > 30 else np.nan
    
    desc_stats[sym] = {
        'mean_ratio': round(ratio.mean(), 4),
        'std_ratio': round(ratio.std(), 4),
        'median_ratio': round(ratio.median(), 4),
        'corr_fwd_ret': round(corr, 4) if not np.isnan(corr) else None,
        'autocorr_1': round(ac1, 4) if not np.isnan(ac1) else None,
        'avg_buy_M': round(df['buy_vol'].mean()/1e6, 1),
        'avg_sell_M': round(df['sell_vol'].mean()/1e6, 1),
        'n': len(df),
    }

desc_df = pd.DataFrame(desc_stats).T
print(desc_df.to_string())

print("\n--- Buy/Sell Ratio → Next-Day Return Correlation ---")
corrs = {k: v['corr_fwd_ret'] for k, v in desc_stats.items() if v['corr_fwd_ret'] is not None}
for sym in sorted(corrs, key=lambda x: abs(corrs[x]), reverse=True)[:10]:
    print(f"  {sym:6s}: {corrs[sym]:+.4f}")
print(f"  {'AVG':6s}: {np.mean(list(corrs.values())):+.4f}")
sys.stdout.flush()

# ── Backtests ──
print("\n" + "=" * 80)
print("WALK-FORWARD BACKTEST + PERMUTATION TESTS")
print("=" * 80)

n_tests = len(SIGNALS) * len(all_data)
bonf_alpha = 0.05 / n_tests
print(f"Tests: {n_tests}, Bonferroni α: {bonf_alpha:.6f}\n")
sys.stdout.flush()

rows = []
for sig_name, sig_func in SIGNALS.items():
    print(f"--- {sig_name} ---")
    sys.stdout.flush()
    for symbol in all_data:
        df = all_data[symbol]
        
        # Walk-forward
        oos = walk_forward(df, sig_func)
        if oos is None: continue
        metrics = compute_metrics(oos)
        
        # Buy-and-hold
        bh = compute_metrics(df['returns'].iloc[100:])
        
        # Permutation test
        p = fast_permutation_test(df, sig_func, metrics['sharpe'], n_perms=500)
        
        result = {
            'signal': sig_name, 'asset': symbol,
            **metrics,
            'bh_sharpe': bh['sharpe'], 'bh_cagr': bh['cagr'],
            'p_value': round(p, 4),
            'sig_bonf': p < bonf_alpha,
            'beats_sma50': False,  # filled later
        }
        rows.append(result)
        
        marker = "***" if p < bonf_alpha else ("*" if p < 0.05 else "")
        print(f"  {symbol:6s} Sharpe {metrics['sharpe']:+6.3f} CAGR {metrics['cagr']:+7.2f}% MaxDD {metrics['maxdd']:7.2f}% p={p:.4f} {marker}")
    sys.stdout.flush()

# ── Summary ──
print("\n" + "=" * 80)
print("SIGNALS BEATING SMA50 BASELINE")
print("=" * 80)

baseline = {r['asset']: r for r in rows if r['signal'] == 'Baseline_SMA50'}
winners = []
for r in rows:
    if r['signal'] == 'Baseline_SMA50': continue
    bl = baseline.get(r['asset'])
    if bl and r['sharpe'] > bl['sharpe']:
        r['beats_sma50'] = True
        winners.append(r)

if winners:
    winners.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"{'Signal':28s} {'Asset':6s} {'Sharpe':>8s} {'SMA50':>8s} {'CAGR':>8s} {'p-val':>8s}")
    for w in winners[:20]:
        bl = baseline[w['asset']]
        sig = "**" if w['sig_bonf'] else ("*" if w['p_value'] < 0.05 else "")
        print(f"{w['signal']:28s} {w['asset']:6s} {w['sharpe']:+8.3f} {bl['sharpe']:+8.3f} {w['cagr']:+7.2f}% {w['p_value']:8.4f} {sig}")
else:
    print("None.")

print("\n" + "=" * 80)
print("BONFERRONI SIGNIFICANT")
print("=" * 80)
sig_results = [r for r in rows if r['sig_bonf']]
if sig_results:
    for r in sorted(sig_results, key=lambda x: x['sharpe'], reverse=True):
        print(f"  {r['signal']:28s} {r['asset']:6s} Sharpe {r['sharpe']:+.3f} p={r['p_value']:.4f}")
else:
    print("None survived Bonferroni correction.")

nom = [r for r in rows if r['p_value'] < 0.05 and not r['sig_bonf']]
if nom:
    print(f"\nNominally significant (p<0.05, n={len(nom)}):")
    for r in sorted(nom, key=lambda x: x['sharpe'], reverse=True)[:10]:
        print(f"  {r['signal']:28s} {r['asset']:6s} Sharpe {r['sharpe']:+.3f} p={r['p_value']:.4f}")

print("\n" + "=" * 80)
print("AVG SHARPE BY SIGNAL")
print("=" * 80)
sig_groups = {}
for r in rows:
    sig_groups.setdefault(r['signal'], []).append(r['sharpe'])
for s in sorted(sig_groups, key=lambda x: np.mean(sig_groups[x]), reverse=True):
    v = sig_groups[s]
    print(f"  {s:28s} avg={np.mean(v):+.3f} med={np.median(v):+.3f} n={len(v)}")

# Save
output = {
    'metadata': {'run_date': datetime.now().isoformat(), 'n_assets': len(all_data),
                 'n_signals': len(SIGNALS), 'n_tests': n_tests, 'bonferroni_alpha': bonf_alpha},
    'descriptive_stats': desc_stats,
    'results': rows,
    'summary': {
        'avg_sharpe_by_signal': {k: round(np.mean(v), 3) for k, v in sig_groups.items()},
        'bonferroni_significant': len(sig_results),
        'nominally_significant': len(nom),
        'beating_sma50': len(winners),
    }
}
with open(OUT_PATH, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {OUT_PATH}")
print("✅ DONE")
