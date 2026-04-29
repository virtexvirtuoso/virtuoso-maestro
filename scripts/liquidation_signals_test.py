#!/usr/bin/env python3
"""Liquidation-based trading signal backtest with walk-forward validation and permutation tests."""

import os, json, warnings
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Data Loading ───
DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'DOGE']
OUTPUT_PATH = os.path.expanduser("~/Desktop/maestro/data/backtest_results/liquidation_signals_test.json")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

N_FOLDS = 14
N_PERMS = 500
ALPHA = 0.05
TOTAL_TESTS = 6 * 6  # 6 signals x 6 symbols
BONF_ALPHA = ALPHA / TOTAL_TESTS

def load_data(symbol):
    con = duckdb.connect(DB_PATH, read_only=True)
    liq = con.execute(f"SELECT * FROM cg_liquidations WHERE symbol='{symbol}' ORDER BY date").df()
    price = con.execute(f"SELECT * FROM perps_daily WHERE symbol='{symbol}' ORDER BY date").df()
    con.close()
    
    liq = liq.rename(columns={'aggregated_long_liquidation_usd': 'long_liq', 
                               'aggregated_short_liquidation_usd': 'short_liq'})
    liq['total_liq'] = liq['long_liq'] + liq['short_liq']
    liq['date'] = pd.to_datetime(liq['date']).dt.normalize()
    price['date'] = pd.to_datetime(price['date']).dt.normalize()
    
    df = price.merge(liq[['date','long_liq','short_liq','total_liq']], on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    df['fwd_ret'] = df['returns'].shift(-1)  # next day return for signal evaluation
    return df

# ─── Signal Generators ───
def sig_flush_buy(df):
    """S1: Long when total_liq > 95th pct rolling 90d"""
    pct95 = df['total_liq'].rolling(90, min_periods=60).quantile(0.95)
    sig = pd.Series(0, index=df.index)
    sig[df['total_liq'] > pct95] = 1
    return sig

def sig_flush_direction(df):
    """S2: Long after long-liq spike, short after short-liq spike"""
    long_pct95 = df['long_liq'].rolling(90, min_periods=60).quantile(0.95)
    short_pct95 = df['short_liq'].rolling(90, min_periods=60).quantile(0.95)
    sig = pd.Series(0, index=df.index)
    sig[df['long_liq'] > long_pct95] = 1   # longs rekt = buy
    sig[df['short_liq'] > short_pct95] = -1  # shorts rekt = sell
    return sig

def sig_calm(df):
    """S3: Long when liq < 10th pct for 5+ consecutive days"""
    pct10 = df['total_liq'].rolling(90, min_periods=60).quantile(0.10)
    below = (df['total_liq'] < pct10).astype(int)
    # Count consecutive days below
    streak = below.copy()
    for i in range(1, len(streak)):
        if streak.iloc[i] == 1:
            streak.iloc[i] = streak.iloc[i-1] + 1
    sig = pd.Series(0, index=df.index)
    sig[streak >= 5] = 1
    return sig

def sig_divergence(df):
    """S4: Price new 20d low BUT liq declining = bullish"""
    price_low = df['close'] == df['close'].rolling(20).min()
    liq_declining = df['total_liq'].rolling(5).mean() < df['total_liq'].rolling(20).mean()
    sig = pd.Series(0, index=df.index)
    sig[price_low & liq_declining] = 1
    return sig

def sig_cascade_overlay(df):
    """S5: Cascade regime detector. 1=normal, 0=flat during cascade+3d"""
    avg30 = df['total_liq'].rolling(30, min_periods=20).mean()
    cascade_day = (df['total_liq'] > 3 * avg30).astype(int)
    # 2+ consecutive days
    cascade = pd.Series(0, index=df.index)
    for i in range(1, len(cascade)):
        if cascade_day.iloc[i] == 1 and cascade_day.iloc[i-1] == 1:
            cascade.iloc[i] = 1
            cascade.iloc[i-1] = 1
    # Extend flat period: cascade + 3 days after
    flat = cascade.copy()
    for i in range(len(flat)):
        if cascade.iloc[i] == 1:
            for j in range(1, 4):
                if i+j < len(flat):
                    flat.iloc[i+j] = 1
    # 1 = allow trading, 0 = go flat
    sig = 1 - flat
    return sig

def sig_ls_ratio(df):
    """S6: Long/short liq ratio contrarian"""
    ratio = df['long_liq'] / df['short_liq'].replace(0, np.nan)
    sig = pd.Series(0, index=df.index)
    sig[ratio > 3] = 1      # longs destroyed = buy
    sig[ratio < 0.33] = -1  # shorts destroyed = reduce
    return sig

SIGNALS = {
    'S1_FlushBuy': sig_flush_buy,
    'S2_FlushDirection': sig_flush_direction,
    'S3_Calm': sig_calm,
    'S4_Divergence': sig_divergence,
    'S5_CascadeOverlay': sig_cascade_overlay,
    'S6_LSRatio': sig_ls_ratio,
}

# ─── Baselines ───
def sma50_signal(df):
    sma = df['close'].rolling(50).mean()
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > sma] = 1
    sig[df['close'] <= sma] = -1
    return sig

def v4_overlay(df, liq_signal):
    """SMA50 + trailing stop + liquidation signal overlay"""
    sma = df['close'].rolling(50).mean()
    base = pd.Series(0, index=df.index)
    base[df['close'] > sma] = 1
    base[df['close'] <= sma] = -1
    
    # Trailing stop: exit if price drops 5% from rolling 20d high
    rolling_high = df['close'].rolling(20).max()
    trailing_stop = df['close'] < rolling_high * 0.95
    base[trailing_stop] = 0
    
    # Apply overlay
    if liq_signal.nunique() <= 2 and set(liq_signal.unique()) <= {0, 1}:
        # Binary overlay (S5 style): multiply
        combined = base * liq_signal
    else:
        # Signal overlay: use liq signal to confirm/override
        combined = base.copy()
        # Only go long when liq signal agrees or is neutral
        combined[(base == 1) & (liq_signal == -1)] = 0
        combined[(base == -1) & (liq_signal == 1)] = 0
    return combined

# ─── Backtest Engine ───
def calc_returns(signals, fwd_returns):
    """Calculate strategy returns. Signal on bar N, trade on bar N+1 (fwd_ret already shifted)."""
    strat_ret = signals * fwd_returns
    strat_ret = strat_ret.dropna()
    if len(strat_ret) == 0:
        return {'total_return': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'n_trades': 0, 'annual_return': 0}
    
    cum = (1 + strat_ret).cumprod()
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    n_years = len(strat_ret) / 252
    annual_ret = (1 + total_ret) ** (1/max(n_years, 0.1)) - 1
    
    # Sharpe
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    
    # Max DD
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    # Win rate
    trades = strat_ret[strat_ret != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    
    return {
        'total_return': round(float(total_ret), 4),
        'annual_return': round(float(annual_ret), 4),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd), 4),
        'win_rate': round(float(win_rate), 4),
        'n_trades': int((signals.diff().fillna(0) != 0).sum()),
    }

def walk_forward(df, signal_func, n_folds=14):
    """Expanding window walk-forward. Returns OOS results."""
    n = len(df)
    min_train = max(180, n // (n_folds + 1))
    fold_size = (n - min_train) // n_folds
    
    all_oos_rets = []
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break
        
        # Generate signal on full data up to train_end (for rolling calcs), evaluate OOS
        signals = signal_func(df.iloc[:test_end])
        oos_sig = signals.iloc[train_end:test_end]
        oos_fwd = df['fwd_ret'].iloc[train_end:test_end]
        
        fold_ret = oos_sig * oos_fwd
        all_oos_rets.append(fold_ret)
    
    if not all_oos_rets:
        return calc_returns(pd.Series(dtype=float), pd.Series(dtype=float))
    
    combined = pd.concat(all_oos_rets)
    # Reconstruct as if it were signal * fwd_ret
    cum = (1 + combined.dropna()).cumprod()
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    n_years = len(combined.dropna()) / 252
    annual_ret = (1 + total_ret) ** (1/max(n_years, 0.1)) - 1
    sharpe = combined.dropna().mean() / combined.dropna().std() * np.sqrt(252) if combined.dropna().std() > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() if len(dd) > 0 else 0
    trades = combined.dropna()
    trades_nz = trades[trades != 0]
    win_rate = (trades_nz > 0).mean() if len(trades_nz) > 0 else 0
    
    return {
        'total_return': round(float(total_ret), 4),
        'annual_return': round(float(annual_ret), 4),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd), 4),
        'win_rate': round(float(win_rate), 4),
        'n_days_oos': int(len(combined.dropna())),
    }

def permutation_test(df, signal_func, n_perms=500):
    """Permutation test: shuffle signal assignment, compare to actual."""
    signals = signal_func(df)
    actual_ret = (signals * df['fwd_ret']).dropna()
    actual_sharpe = actual_ret.mean() / actual_ret.std() * np.sqrt(252) if actual_ret.std() > 0 else 0
    
    count_better = 0
    for _ in range(n_perms):
        shuffled = signals.sample(frac=1).reset_index(drop=True)
        perm_ret = (shuffled * df['fwd_ret']).dropna()
        perm_sharpe = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1
    
    p_value = (count_better + 1) / (n_perms + 1)
    return round(p_value, 4)

# ─── Event Study ───
def event_study(df, signal_func, horizons=[1, 3, 7, 14]):
    """Average returns after signal fires."""
    signals = signal_func(df)
    long_events = df.index[signals == 1]
    
    results = {}
    for h in horizons:
        rets = []
        for idx in long_events:
            if idx + h < len(df):
                r = df['close'].iloc[idx + h] / df['close'].iloc[idx] - 1
                rets.append(r)
        if rets:
            results[f'{h}d_mean'] = round(np.mean(rets), 4)
            results[f'{h}d_median'] = round(np.median(rets), 4)
            results[f'{h}d_winrate'] = round(np.mean([r > 0 for r in rets]), 4)
            results[f'{h}d_n'] = len(rets)
        else:
            results[f'{h}d_mean'] = None
    return results

# ─── Main ───
print("="*80)
print("LIQUIDATION SIGNALS BACKTEST")
print("="*80)

all_results = {}
event_studies = {}
beats_sma50 = []

for sym in SYMBOLS:
    print(f"\n{'─'*60}")
    print(f"  {sym}")
    print(f"{'─'*60}")
    
    df = load_data(sym)
    print(f"  Data: {df['date'].min().date()} → {df['date'].max().date()}, {len(df)} rows")
    
    # Baselines
    sma_sig = sma50_signal(df)
    bh_ret = calc_returns(pd.Series(1, index=df.index), df['fwd_ret'])
    sma_wf = walk_forward(df, sma50_signal, N_FOLDS)
    
    all_results[sym] = {'buy_hold': bh_ret, 'sma50_wf': sma_wf}
    event_studies[sym] = {}
    
    print(f"  Buy&Hold: {bh_ret['total_return']:.1%}  Sharpe={bh_ret['sharpe']:.2f}")
    print(f"  SMA50 WF: {sma_wf['total_return']:.1%}  Sharpe={sma_wf['sharpe']:.2f}")
    
    for sig_name, sig_func in SIGNALS.items():
        # Walk-forward
        wf = walk_forward(df, sig_func, N_FOLDS)
        
        # Permutation test
        p_val = permutation_test(df, sig_func, N_PERMS)
        significant = p_val < BONF_ALPHA
        
        # V4 overlay
        v4_func = lambda d, sf=sig_func: v4_overlay(d, sf(d))
        v4_wf = walk_forward(df, v4_func, N_FOLDS)
        
        # Event study (for S1 flush events)
        if sig_name in ['S1_FlushBuy', 'S2_FlushDirection', 'S4_Divergence', 'S6_LSRatio']:
            es = event_study(df, sig_func)
            event_studies[sym][sig_name] = es
        
        all_results[sym][sig_name] = {
            'wf': wf,
            'p_value': p_val,
            'significant_bonferroni': significant,
            'v4_overlay': v4_wf,
        }
        
        beat = wf['sharpe'] > sma_wf['sharpe']
        marker = "✓" if beat else " "
        sig_mark = "★" if significant else " "
        print(f"  {sig_name:25s} WF: {wf['total_return']:>8.1%}  Sharpe={wf['sharpe']:>6.2f}  p={p_val:.3f} {sig_mark}  V4: {v4_wf['sharpe']:>6.2f} {marker}")
        
        if beat:
            beats_sma50.append(f"{sym}/{sig_name} (Sharpe {wf['sharpe']:.2f} vs SMA50 {sma_wf['sharpe']:.2f})")

# ─── Results Table ───
print("\n" + "="*80)
print("FULL RESULTS TABLE")
print("="*80)
header = f"{'Symbol':<6} {'Signal':<25} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'WinRate':>7} {'p-val':>7} {'Sig':>3} {'V4_Sharpe':>9}"
print(header)
print("─" * len(header))

for sym in SYMBOLS:
    r = all_results[sym]
    print(f"{sym:<6} {'Buy&Hold':<25} {r['buy_hold']['total_return']:>8.1%} {r['buy_hold']['sharpe']:>7.2f} {r['buy_hold']['max_dd']:>8.1%} {r['buy_hold']['win_rate']:>7.1%}")
    print(f"{sym:<6} {'SMA50 (WF)':<25} {r['sma50_wf']['total_return']:>8.1%} {r['sma50_wf']['sharpe']:>7.2f} {r['sma50_wf']['max_dd']:>8.1%} {r['sma50_wf']['win_rate']:>7.1%}")
    for sig_name in SIGNALS:
        sr = r[sig_name]
        wf = sr['wf']
        v4 = sr['v4_overlay']
        sig_str = "★" if sr['significant_bonferroni'] else ""
        print(f"{sym:<6} {sig_name:<25} {wf['total_return']:>8.1%} {wf['sharpe']:>7.2f} {wf['max_dd']:>8.1%} {wf['win_rate']:>7.1%} {sr['p_value']:>7.3f} {sig_str:>3} {v4['sharpe']:>9.2f}")
    print()

# ─── Event Study ───
print("\n" + "="*80)
print("EVENT STUDY: Average Returns After Liquidation Events")
print("="*80)
for sym in SYMBOLS:
    es = event_studies.get(sym, {})
    if not es:
        continue
    print(f"\n{sym}:")
    for sig_name, data in es.items():
        print(f"  {sig_name}:")
        for h in [1, 3, 7, 14]:
            m = data.get(f'{h}d_mean')
            med = data.get(f'{h}d_median')
            wr = data.get(f'{h}d_winrate')
            n = data.get(f'{h}d_n', 0)
            if m is not None:
                print(f"    {h:>2}d: mean={m:>+.2%}  median={med:>+.2%}  win_rate={wr:.0%}  n={n}")

# ─── Signals That Beat SMA50 ───
print("\n" + "="*80)
print("SIGNALS THAT BEAT SMA50 (by Sharpe)")
print("="*80)
if beats_sma50:
    for b in beats_sma50:
        print(f"  ✓ {b}")
else:
    print("  None")

# ─── Save JSON ───
# Convert for JSON serialization
output = {
    'metadata': {
        'run_date': datetime.now().isoformat(),
        'symbols': SYMBOLS,
        'n_folds': N_FOLDS,
        'n_permutations': N_PERMS,
        'bonferroni_alpha': BONF_ALPHA,
        'signals': list(SIGNALS.keys()),
    },
    'results': all_results,
    'event_studies': event_studies,
    'beats_sma50': beats_sma50,
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to {OUTPUT_PATH}")
