#!/usr/bin/env python3
"""
LSR Full-Cycle Test: CoinGlass Long/Short Ratio signals (2020-2026)
Tests 6 signal types across BTC, ETH, SOL, BNB with walk-forward + permutation tests.
"""
import json, warnings, os
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime
warnings.filterwarnings('ignore')

# ── Config ──
ASSETS = ['BTC', 'ETH', 'SOL', 'BNB']
N_PERMS = 500
N_FOLDS = 14
BONFERRONI_N = 24  # ~6 signals × 4 assets
FUNDING_DAILY = 0.01 / 100  # ~0.01% daily drag for perps

# ── Load Data ──
con = duckdb.connect(os.path.expanduser('~/Desktop/maestro/data/maestro.duckdb'), read_only=True)

def load_asset(symbol):
    """Load and merge price + LSR data for an asset."""
    price = con.execute(f"SELECT date, open, high, low, close FROM perps_daily WHERE symbol='{symbol}' ORDER BY date").df()
    lsr_g = con.execute(f"SELECT date, global_account_long_short_ratio as lsr_global FROM cg_lsr_global WHERE symbol='{symbol}' ORDER BY date").df()
    lsr_top = con.execute(f"SELECT date, top_account_long_short_ratio as lsr_top FROM cg_lsr_top_account WHERE symbol='{symbol}' ORDER BY date").df()
    
    df = price.merge(lsr_g, on='date', how='inner').merge(lsr_top, on='date', how='left')
    df = df.sort_values('date').reset_index(drop=True)
    df['ret'] = df['close'].pct_change().shift(-1)  # next-day return (for signal on bar N, trade N+1)
    df['sma50'] = df['close'].rolling(50).mean()
    return df

# ── Signal Generators ──
def sig_contrarian(df):
    """Signal 1: Short when LSR > 80th pctl, Long when < 20th (rolling 90d)."""
    r = df['lsr_global'].rolling(90, min_periods=30)
    p80 = r.quantile(0.8)
    p20 = r.quantile(0.2)
    sig = pd.Series(0, index=df.index)
    sig[df['lsr_global'] > p80] = -1
    sig[df['lsr_global'] < p20] = 1
    return sig

def sig_momentum(df):
    """Signal 2: Long when 5d SMA of LSR > 20d SMA, else short."""
    sma5 = df['lsr_global'].rolling(5).mean()
    sma20 = df['lsr_global'].rolling(20).mean()
    sig = pd.Series(0, index=df.index)
    sig[sma5 > sma20] = 1
    sig[sma5 < sma20] = -1
    return sig

def sig_extreme_mr(df):
    """Signal 3: Long only when ratio < 1.0 (more shorts than longs)."""
    sig = pd.Series(0, index=df.index)
    sig[df['lsr_global'] < 1.0] = 1
    return sig

def sig_divergence(df):
    """Signal 4: Long when top > global (smart money long), short when top < global."""
    sig = pd.Series(0, index=df.index)
    sig[df['lsr_top'] > df['lsr_global']] = 1
    sig[df['lsr_top'] < df['lsr_global']] = -1
    return sig

def sig_v4_lsr_filter(df):
    """Signal 5: SMA50 long, but reduce 50% if LSR > 90th pctl."""
    r = df['lsr_global'].rolling(90, min_periods=30)
    p90 = r.quantile(0.9)
    trend_long = df['close'] > df['sma50']
    sig = pd.Series(0.0, index=df.index)
    sig[trend_long] = 1.0
    sig[trend_long & (df['lsr_global'] > p90)] = 0.5
    sig[~trend_long] = -1.0
    return sig

def sig_v4_smart(df):
    """Signal 6: SMA50 long + top>global → full, top<global → half."""
    trend_long = df['close'] > df['sma50']
    smart = df['lsr_top'] > df['lsr_global']
    sig = pd.Series(0.0, index=df.index)
    sig[trend_long & smart] = 1.0
    sig[trend_long & ~smart] = 0.5
    sig[~trend_long] = -1.0
    return sig

def sig_sma50_baseline(df):
    """Baseline: simple SMA50 trend."""
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > df['sma50']] = 1
    sig[df['close'] <= df['sma50']] = -1
    return sig

SIGNALS = {
    'LSR_Contrarian': sig_contrarian,
    'LSR_Momentum': sig_momentum,
    'LSR_ExtremeMR': sig_extreme_mr,
    'LSR_Divergence': sig_divergence,
    'V4_LSR_Filter': sig_v4_lsr_filter,
    'V4_SmartMoney': sig_v4_smart,
    'SMA50_Baseline': sig_sma50_baseline,
}

# ── Backtest Engine ──
def backtest_returns(signals, returns, funding_drag=True):
    """Compute strategy returns from signals and next-day returns."""
    strat_ret = signals * returns
    if funding_drag:
        # Apply funding drag when in position
        strat_ret -= np.abs(signals) * FUNDING_DAILY
    return strat_ret

def calc_metrics(strat_ret, ann=365):
    """Calculate Sharpe, CAGR, MaxDD from daily returns."""
    strat_ret = strat_ret.dropna()
    if len(strat_ret) < 30:
        return {'sharpe': np.nan, 'cagr': np.nan, 'maxdd': np.nan, 'n_days': len(strat_ret)}
    cum = (1 + strat_ret).cumprod()
    total_ret = cum.iloc[-1] / cum.iloc[0] - 1
    n_years = len(strat_ret) / ann
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = strat_ret.mean() / max(strat_ret.std(), 1e-10) * np.sqrt(ann)
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    maxdd = drawdown.min()
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr * 100, 1), 'maxdd': round(maxdd * 100, 1), 'n_days': len(strat_ret)}

# ── Walk-Forward ──
def walk_forward_test(df, sig_func, n_folds=N_FOLDS):
    """Expanding walk-forward: train on first k folds, test on k+1."""
    valid = df.dropna(subset=['ret', 'lsr_global']).copy()
    if len(valid) < 100:
        return None, None
    
    signals = sig_func(valid)
    fold_size = len(valid) // (n_folds + 1)
    
    oos_rets = []
    for k in range(1, n_folds + 1):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, len(valid))
        if test_start >= len(valid):
            break
        test_sig = signals.iloc[test_start:test_end]
        test_ret = valid['ret'].iloc[test_start:test_end]
        sr = backtest_returns(test_sig, test_ret)
        oos_rets.append(sr)
    
    if not oos_rets:
        return None, None
    
    all_oos = pd.concat(oos_rets)
    return all_oos, signals

# ── Permutation Test ──
def permutation_test(df, sig_func, observed_sharpe, n_perms=N_PERMS):
    """Shuffle returns, recompute sharpe n_perms times."""
    valid = df.dropna(subset=['ret', 'lsr_global']).copy()
    signals = sig_func(valid)
    
    count_better = 0
    for _ in range(n_perms):
        shuffled_ret = valid['ret'].sample(frac=1, replace=False).values
        perm_strat = backtest_returns(signals, pd.Series(shuffled_ret, index=valid.index))
        perm_metrics = calc_metrics(perm_strat)
        if not np.isnan(perm_metrics['sharpe']) and perm_metrics['sharpe'] >= observed_sharpe:
            count_better += 1
    
    return (count_better + 1) / (n_perms + 1)  # conservative p-value

# ── Conditional Analysis ──
def conditional_analysis(df):
    """What happens after extreme LSR readings."""
    valid = df.dropna(subset=['lsr_global', 'close']).copy()
    valid['fwd_1d'] = valid['close'].pct_change(1).shift(-1)
    valid['fwd_3d'] = valid['close'].pct_change(3).shift(-3)
    valid['fwd_7d'] = valid['close'].pct_change(7).shift(-7)
    valid['fwd_14d'] = valid['close'].pct_change(14).shift(-14)
    
    r90 = valid['lsr_global'].rolling(90, min_periods=30)
    p90 = r90.quantile(0.9)
    p10 = r90.quantile(0.1)
    
    results = {}
    for label, mask in [('LSR>90pctl', valid['lsr_global'] > p90), 
                         ('LSR<10pctl', valid['lsr_global'] < p10),
                         ('LSR<1.0', valid['lsr_global'] < 1.0),
                         ('All', pd.Series(True, index=valid.index))]:
        subset = valid[mask]
        r = {}
        for h in ['fwd_1d', 'fwd_3d', 'fwd_7d', 'fwd_14d']:
            vals = subset[h].dropna()
            r[h] = {'mean': round(vals.mean()*100, 3), 'median': round(vals.median()*100, 3), 
                     'n': len(vals), 'pct_positive': round((vals>0).mean()*100, 1)}
        results[label] = r
    return results

# ── Main ──
print("="*90)
print("LSR FULL-CYCLE TEST — CoinGlass Data (2020-2026)")
print("="*90)

all_results = {}
rows = []

for symbol in ASSETS:
    print(f"\n{'─'*40} {symbol} {'─'*40}")
    df = load_asset(symbol)
    print(f"  Data: {df['date'].min().date()} → {df['date'].max().date()} ({len(df)} bars)")
    print(f"  LSR range: {df['lsr_global'].min():.2f} – {df['lsr_global'].max():.2f}")
    if df['lsr_top'].notna().sum() > 0:
        print(f"  Top LSR range: {df['lsr_top'].min():.2f} – {df['lsr_top'].max():.2f}")
    
    # Buy & Hold
    bh_ret = df['ret'].dropna()
    bh_metrics = calc_metrics(bh_ret, ann=365)
    bh_metrics['signal'] = 'BuyHold'
    bh_metrics['symbol'] = symbol
    bh_metrics['p_value'] = np.nan
    bh_metrics['p_bonf'] = np.nan
    rows.append(bh_metrics)
    
    for sig_name, sig_func in SIGNALS.items():
        oos_ret, _ = walk_forward_test(df, sig_func)
        if oos_ret is None:
            print(f"  {sig_name}: insufficient data")
            continue
        
        m = calc_metrics(oos_ret)
        
        # Permutation test
        if not np.isnan(m['sharpe']):
            p_val = permutation_test(df, sig_func, m['sharpe'])
        else:
            p_val = 1.0
        
        p_bonf = min(p_val * BONFERRONI_N, 1.0)
        
        m['signal'] = sig_name
        m['symbol'] = symbol
        m['p_value'] = round(p_val, 4)
        m['p_bonf'] = round(p_bonf, 4)
        rows.append(m)
    
    # Conditional analysis
    cond = conditional_analysis(df)
    all_results[f'{symbol}_conditional'] = cond

con.close()

# ── Results Table ──
results_df = pd.DataFrame(rows)
results_df = results_df[['symbol', 'signal', 'sharpe', 'cagr', 'maxdd', 'n_days', 'p_value', 'p_bonf']]

print("\n" + "="*90)
print("FULL RESULTS TABLE")
print("="*90)
print(results_df.to_string(index=False))

# ── Highlight: beats SMA50? ──
print("\n" + "="*90)
print("SIGNALS THAT BEAT SMA50 BASELINE (by Sharpe)")
print("="*90)
for symbol in ASSETS:
    sym_df = results_df[results_df['symbol'] == symbol]
    baseline = sym_df[sym_df['signal'] == 'SMA50_Baseline']
    if baseline.empty:
        continue
    base_sharpe = baseline['sharpe'].values[0]
    winners = sym_df[(sym_df['sharpe'] > base_sharpe) & (~sym_df['signal'].isin(['BuyHold', 'SMA50_Baseline']))]
    if len(winners) > 0:
        for _, w in winners.iterrows():
            sig = "✅" if w['p_bonf'] < 0.05 else "⚠️"
            print(f"  {sig} {symbol} {w['signal']}: Sharpe {w['sharpe']:.3f} vs baseline {base_sharpe:.3f} (p_bonf={w['p_bonf']:.4f})")
    else:
        print(f"  ❌ {symbol}: No LSR signal beats SMA50 baseline")

# ── Conditional Analysis ──
print("\n" + "="*90)
print("CONDITIONAL ANALYSIS: Forward Returns After Extreme LSR")
print("="*90)
for symbol in ASSETS:
    cond = all_results[f'{symbol}_conditional']
    print(f"\n  {symbol}:")
    print(f"  {'Condition':<15} {'1d%':>8} {'3d%':>8} {'7d%':>8} {'14d%':>8} {'N':>6} {'%Pos(14d)':>10}")
    for label in ['LSR>90pctl', 'LSR<10pctl', 'LSR<1.0', 'All']:
        c = cond[label]
        n14 = c['fwd_14d']
        print(f"  {label:<15} {c['fwd_1d']['mean']:>7.3f}% {c['fwd_3d']['mean']:>7.3f}% {c['fwd_7d']['mean']:>7.3f}% {n14['mean']:>7.3f}% {n14['n']:>6} {n14['pct_positive']:>9.1f}%")

# ── Save JSON ──
output = {
    'timestamp': datetime.now().isoformat(),
    'config': {'assets': ASSETS, 'n_perms': N_PERMS, 'n_folds': N_FOLDS, 'bonferroni_n': BONFERRONI_N},
    'results': results_df.to_dict(orient='records'),
    'conditional': all_results,
}
os.makedirs(os.path.expanduser('~/Desktop/maestro/data/backtest_results'), exist_ok=True)
with open(os.path.expanduser('~/Desktop/maestro/data/backtest_results/lsr_fullcycle_test.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to ~/Desktop/maestro/data/backtest_results/lsr_fullcycle_test.json")
