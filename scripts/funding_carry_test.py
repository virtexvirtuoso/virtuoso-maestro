#!/usr/bin/env python3
"""
Funding Rate Carry & Regime Signal Testing
CoinGlass 5yr funding data, 19 tokens, 2020-2026
7 signals with walk-forward validation and permutation tests
"""
import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Load Data ───
DB = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
con = duckdb.connect(DB)

fr_all = con.execute("SELECT date, symbol, close as funding_rate FROM cg_funding_rate ORDER BY symbol, date").df()
price_all = con.execute("SELECT date, symbol, close as price, open, high, low FROM perps_daily ORDER BY symbol, date").df()
con.close()

# Convert funding from % to decimal (0.01 = 0.01% = 0.0001)
fr_all['funding_rate'] = fr_all['funding_rate'] / 100  # now in decimal

# Merge funding + price per symbol
symbols = sorted(set(fr_all['symbol']) & set(price_all['symbol']))
print(f"Symbols with both funding & price: {symbols}")

data = {}
for sym in symbols:
    fr_s = fr_all[fr_all['symbol'] == sym][['date', 'funding_rate']].copy()
    pr_s = price_all[price_all['symbol'] == sym][['date', 'price']].copy()
    merged = pd.merge(fr_s, pr_s, on='date', how='inner').sort_values('date').reset_index(drop=True)
    if len(merged) >= 200:
        merged['ret'] = merged['price'].pct_change().shift(-1)  # next-day return
        merged['sma50'] = merged['price'].rolling(50).mean()
        merged['fr_7d'] = merged['funding_rate'].rolling(7).mean()
        merged['fr_30d_mean'] = merged['funding_rate'].rolling(30).mean()
        merged['fr_30d_std'] = merged['funding_rate'].rolling(30).std()
        merged['fr_zscore'] = (merged['funding_rate'] - merged['fr_30d_mean']) / merged['fr_30d_std']
        merged['price_ret_7d'] = merged['price'].pct_change(7)
        merged['fr_change_7d'] = merged['funding_rate'] - merged['funding_rate'].shift(7)
        data[sym] = merged
        
print(f"Tokens with >=200 days: {list(data.keys())} ({len(data)})")

# ─── Signal Generators ───
def sig_extreme_contrarian(df):
    """Long when funding < -0.01%, flat when > 0.05%"""
    sig = np.zeros(len(df))
    sig[df['funding_rate'] < -0.0001] = 1  # -0.01%
    sig[df['funding_rate'] > 0.0005] = 0   # 0.05%
    # between: hold previous
    for i in range(1, len(sig)):
        if sig[i] == 0 and df['funding_rate'].iloc[i] >= -0.0001 and df['funding_rate'].iloc[i] <= 0.0005:
            sig[i] = sig[i-1]
    return sig

def sig_mean_reversion(df):
    """Long when z-score < -2, flat when > +2"""
    sig = np.zeros(len(df))
    z = df['fr_zscore'].values
    sig[z < -2] = 1
    sig[z > 2] = -1  # flat/short
    for i in range(1, len(sig)):
        if sig[i] == 0:
            sig[i] = sig[i-1]
    return sig

def sig_funding_trend(df):
    """Long when 7d avg funding rising from negative to positive"""
    sig = np.zeros(len(df))
    fr7 = df['fr_7d'].values
    for i in range(1, len(sig)):
        if not np.isnan(fr7[i]) and not np.isnan(fr7[i-1]):
            if fr7[i] > 0 and fr7[i-1] <= 0:
                sig[i] = 1
            elif fr7[i] < 0 and fr7[i-1] >= 0:
                sig[i] = 0
            else:
                sig[i] = sig[i-1]
    return sig

def sig_divergence(df):
    """Price rising + funding declining = bearish. Price falling + funding rising = bullish."""
    sig = np.zeros(len(df))
    pr = df['price_ret_7d'].values
    fc = df['fr_change_7d'].values
    for i in range(7, len(sig)):
        if not np.isnan(pr[i]) and not np.isnan(fc[i]):
            if pr[i] < 0 and fc[i] > 0:  # bottom forming
                sig[i] = 1
            elif pr[i] > 0 and fc[i] < 0:  # potential top
                sig[i] = -1
            else:
                sig[i] = sig[i-1]
    return sig

def sig_sma50(df):
    """Baseline SMA50"""
    return (df['price'] > df['sma50']).astype(float).values

def sig_sma50_funding_filter(df):
    """SMA50 long, reduce 50% when funding > 0.05%"""
    base = sig_sma50(df)
    filtered = base.copy()
    filtered[df['funding_rate'].values > 0.0005] *= 0.5
    return filtered

def sig_sma50_neg_funding_boost(df):
    """SMA50 long + negative funding → 1.5x"""
    base = sig_sma50(df)
    boosted = base.copy()
    boosted[(base > 0) & (df['funding_rate'].values < 0)] *= 1.5
    return boosted

SIGNALS = {
    'extreme_contrarian': sig_extreme_contrarian,
    'mean_reversion': sig_mean_reversion,
    'funding_trend': sig_funding_trend,
    'divergence': sig_divergence,
    'sma50_baseline': sig_sma50,
    'sma50_funding_filter': sig_sma50_funding_filter,
    'sma50_neg_funding_boost': sig_sma50_neg_funding_boost,
}

# ─── Backtest Engine ───
def backtest_signal(signal, returns, start_idx=0):
    """Simple backtest: signal[i] * returns[i] (signal already shifted by construction)"""
    sig = signal[start_idx:]
    ret = returns[start_idx:]
    mask = ~np.isnan(ret) & ~np.isnan(sig)
    strat_ret = sig[mask] * ret[mask]
    if len(strat_ret) < 30:
        return None
    equity = (1 + strat_ret).cumprod()
    total_ret = equity[-1] - 1 if len(equity) > 0 else 0
    ann_ret = (1 + total_ret) ** (365 / len(strat_ret)) - 1 if len(strat_ret) > 0 else 0
    dd = equity / np.maximum.accumulate(equity) - 1
    max_dd = dd.min()
    sharpe = np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(365) if np.std(strat_ret) > 0 else 0
    win_rate = np.mean(strat_ret > 0) if len(strat_ret) > 0 else 0
    exposure = np.mean(np.abs(sig[mask]))
    return {
        'total_return': round(float(total_ret * 100), 2),
        'ann_return': round(float(ann_ret * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd * 100), 2),
        'win_rate': round(float(win_rate * 100), 1),
        'exposure': round(float(exposure * 100), 1),
        'n_days': int(len(strat_ret)),
        'strat_returns': strat_ret,
    }

# ─── Walk-Forward (14-fold expanding) ───
def walk_forward_test(signal_fn, df, n_folds=14):
    """Expanding walk-forward: train on first k folds, test on fold k+1"""
    n = len(df)
    fold_size = n // n_folds
    if fold_size < 30:
        return None
    
    oos_returns = []
    for k in range(3, n_folds):  # need at least 3 folds for training
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n)
        sig = signal_fn(df)
        test_sig = sig[test_start:test_end]
        test_ret = df['ret'].values[test_start:test_end]
        mask = ~np.isnan(test_ret) & ~np.isnan(test_sig)
        oos_returns.extend((test_sig[mask] * test_ret[mask]).tolist())
    
    if len(oos_returns) < 30:
        return None
    oos = np.array(oos_returns)
    equity = (1 + oos).cumprod()
    total_ret = equity[-1] - 1
    sharpe = np.mean(oos) / np.std(oos) * np.sqrt(365) if np.std(oos) > 0 else 0
    dd = equity / np.maximum.accumulate(equity) - 1
    return {
        'oos_return': round(float(total_ret * 100), 2),
        'oos_sharpe': round(float(sharpe), 3),
        'oos_max_dd': round(float(dd.min() * 100), 2),
        'oos_days': len(oos),
    }

# ─── Permutation Test ───
def permutation_test(signal, returns, n_perms=500):
    """Test if signal Sharpe is significantly better than random"""
    mask = ~np.isnan(returns) & ~np.isnan(signal)
    sig_m, ret_m = signal[mask], returns[mask]
    if len(sig_m) < 30:
        return 1.0
    actual_sharpe = np.mean(sig_m * ret_m) / np.std(sig_m * ret_m) * np.sqrt(365)
    count_better = 0
    for _ in range(n_perms):
        perm_sig = np.random.permutation(sig_m)
        perm_ret = perm_sig * ret_m
        perm_sharpe = np.mean(perm_ret) / np.std(perm_ret) * np.sqrt(365) if np.std(perm_ret) > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1
    return count_better / n_perms

# ─── Cross-Sectional Carry (Signal #1) ───
def test_carry_portfolio():
    """Rank tokens by funding, long bottom 5, weekly rebalance"""
    # Build panel
    all_dates = sorted(set.intersection(*[set(data[s]['date']) for s in data if len(data[s]) > 500]))
    if len(all_dates) < 100:
        # Use dates where at least 8 tokens have data
        date_counts = {}
        for s in data:
            for d in data[s]['date']:
                date_counts[d] = date_counts.get(d, 0) + 1
        all_dates = sorted([d for d, c in date_counts.items() if c >= 8])
    
    print(f"Carry portfolio: {len(all_dates)} dates with sufficient coverage")
    
    port_returns = []
    rebal_dates = all_dates[::7]  # weekly
    
    for i, rd in enumerate(rebal_dates[:-1]):
        next_rd = rebal_dates[i + 1]
        # Get funding rates on rebalance date
        fr_today = {}
        for sym in data:
            row = data[sym][data[sym]['date'] == rd]
            if len(row) > 0 and not np.isnan(row['funding_rate'].iloc[0]):
                fr_today[sym] = row['funding_rate'].iloc[0]
        
        if len(fr_today) < 8:
            continue
        
        # Rank: long bottom 5 (lowest funding)
        ranked = sorted(fr_today.items(), key=lambda x: x[1])
        longs = [s for s, _ in ranked[:5]]
        
        # Get returns for the week
        for sym in longs:
            df_sym = data[sym]
            week_data = df_sym[(df_sym['date'] > rd) & (df_sym['date'] <= next_rd)]
            if len(week_data) > 0:
                wk_ret = (1 + week_data['ret'].dropna()).prod() - 1
                port_returns.append(wk_ret / 5)  # equal weight
    
    if len(port_returns) < 10:
        return None
    
    pr = np.array(port_returns)
    equity = (1 + pr).cumprod()
    total_ret = equity[-1] - 1
    # Weekly returns -> annualize
    sharpe = np.mean(pr) / np.std(pr) * np.sqrt(52) if np.std(pr) > 0 else 0
    dd = equity / np.maximum.accumulate(equity) - 1
    
    return {
        'total_return': round(float(total_ret * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(dd.min() * 100), 2),
        'n_weeks': len(port_returns),
        'avg_weekly_ret': round(float(np.mean(pr) * 100), 3),
    }

# ─── Conditional Analysis ───
def conditional_analysis(sym):
    """Returns after extreme funding readings"""
    df = data[sym]
    fr = df['funding_rate'].values
    ret = df['ret'].values
    
    results = {}
    for label, cond in [
        ('funding < -0.03%', fr < -0.0003),
        ('funding < -0.01%', fr < -0.0001),
        ('funding 0-0.01%', (fr >= 0) & (fr <= 0.0001)),
        ('funding 0.01-0.05%', (fr > 0.0001) & (fr <= 0.0005)),
        ('funding > 0.05%', fr > 0.0005),
        ('funding > 0.1%', fr > 0.001),
    ]:
        r = ret[cond & ~np.isnan(ret)]
        if len(r) > 5:
            results[label] = {
                'count': int(len(r)),
                'avg_ret': round(float(np.mean(r) * 100), 4),
                'median_ret': round(float(np.median(r) * 100), 4),
                'win_rate': round(float(np.mean(r > 0) * 100), 1),
            }
    return results

# ─── Run Everything ───
print("\n" + "="*80)
print("FUNDING RATE SIGNAL TESTING — CoinGlass 5yr Data")
print("="*80)

results = {
    'metadata': {
        'run_date': datetime.now().isoformat(),
        'symbols': list(data.keys()),
        'n_symbols': len(data),
        'total_rows': sum(len(data[s]) for s in data),
        'date_range': {s: [str(data[s]['date'].min()), str(data[s]['date'].max())] for s in data},
    },
    'signal_results': {},
    'carry_portfolio': None,
    'conditional_analysis': {},
    'buy_and_hold': {},
}

# Buy & Hold baseline
for sym in data:
    df = data[sym]
    ret = df['ret'].dropna()
    eq = (1 + ret).cumprod()
    results['buy_and_hold'][sym] = {
        'total_return': round(float((eq.iloc[-1] - 1) * 100), 2),
        'sharpe': round(float(np.mean(ret) / np.std(ret) * np.sqrt(365)), 3),
    }

# Test each signal on each token
n_tests = len(SIGNALS) * len(data)
bonferroni_alpha = 0.05 / n_tests
print(f"\nBonferroni alpha: {bonferroni_alpha:.6f} ({n_tests} tests)")

for sig_name, sig_fn in SIGNALS.items():
    print(f"\n--- {sig_name} ---")
    results['signal_results'][sig_name] = {}
    
    for sym in data:
        df = data[sym]
        signal = sig_fn(df)
        
        # Full backtest
        bt = backtest_signal(signal, df['ret'].values, start_idx=50)
        if bt is None:
            continue
        
        # Walk-forward
        wf = walk_forward_test(sig_fn, df)
        
        # Permutation test
        pval = permutation_test(signal[50:], df['ret'].values[50:], n_perms=500)
        
        entry = {
            'full': {k: v for k, v in bt.items() if k != 'strat_returns'},
            'walk_forward': wf,
            'perm_pval': round(pval, 4),
            'significant': pval < bonferroni_alpha,
        }
        results['signal_results'][sig_name][sym] = entry
        
        sig_str = "***" if pval < bonferroni_alpha else ""
        print(f"  {sym:6s}: Sharpe={bt['sharpe']:+.3f}  Ret={bt['total_return']:+8.1f}%  DD={bt['max_dd']:+7.1f}%  p={pval:.3f} {sig_str}")

# Carry portfolio
print("\n--- Cross-Sectional Carry Portfolio ---")
carry = test_carry_portfolio()
results['carry_portfolio'] = carry
if carry:
    print(f"  Return: {carry['total_return']:+.1f}%  Sharpe: {carry['sharpe']:.3f}  MaxDD: {carry['max_dd']:.1f}%  Weeks: {carry['n_weeks']}")

# Conditional analysis for major tokens
for sym in ['BTC', 'ETH', 'SOL']:
    if sym in data:
        results['conditional_analysis'][sym] = conditional_analysis(sym)

# ─── Summary Table ───
print("\n" + "="*80)
print("SUMMARY: Average Sharpe by Signal (Full Period)")
print("="*80)
for sig_name in SIGNALS:
    sharpes = [results['signal_results'][sig_name][s]['full']['sharpe'] 
               for s in results['signal_results'][sig_name]]
    oos_sharpes = [results['signal_results'][sig_name][s]['walk_forward']['oos_sharpe'] 
                   for s in results['signal_results'][sig_name] 
                   if results['signal_results'][sig_name][s]['walk_forward']]
    n_sig = sum(1 for s in results['signal_results'][sig_name] 
                if results['signal_results'][sig_name][s]['significant'])
    print(f"  {sig_name:30s}  Sharpe={np.mean(sharpes):+.3f}  OOS_Sharpe={np.mean(oos_sharpes) if oos_sharpes else 0:+.3f}  Sig={n_sig}/{len(sharpes)}")

print("\n--- Conditional Analysis (BTC) ---")
if 'BTC' in results['conditional_analysis']:
    for label, stats in results['conditional_analysis']['BTC'].items():
        print(f"  {label:25s}  n={stats['count']:4d}  avg_ret={stats['avg_ret']:+.4f}%  win={stats['win_rate']:.0f}%")

# Save
outdir = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "funding_carry_test.json")
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
