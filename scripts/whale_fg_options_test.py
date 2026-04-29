#!/usr/bin/env python3
"""Whale Copy-Trading, Fear & Greed, and Options Max Pain backtests."""

import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Load Data ───────────────────────────────────────────────────────────
DB = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
con = duckdb.connect(DB, read_only=True)

whales = con.execute("SELECT * FROM whale_trades ORDER BY date").df()
options = con.execute("SELECT * FROM cg_options").df()

# Price data for all symbols we need
symbols_needed = ['BTC', 'ETH', 'SOL']
prices = {}
for sym in symbols_needed:
    df = con.execute(f"SELECT * FROM perps_daily WHERE symbol='{sym}' ORDER BY date").df()
    if len(df) > 0:
        df = df.set_index('date').sort_index()
        df['ret'] = df['close'].pct_change()
        df['sma50'] = df['close'].rolling(50).mean()
        prices[sym] = df
con.close()

# Fear & Greed from API (already fetched)
import subprocess, json as _json
resp = subprocess.run(['curl', '-s', '-H', 'CG-API-KEY: ***REMOVED***',
    'https://open-api-v4.coinglass.com/api/index/fear-greed-history?limit=1000'],
    capture_output=True, text=True)
fg_raw = _json.loads(resp.stdout)['data']
fg_df = pd.DataFrame({
    'date': pd.to_datetime(fg_raw['time_list'], unit='ms').normalize(),
    'fg': fg_raw['data_list'],
    'fg_price': fg_raw['price_list']
}).set_index('date').sort_index()
fg_df = fg_df[~fg_df.index.duplicated(keep='last')]

print(f"Fear & Greed: {fg_df.index.min().date()} to {fg_df.index.max().date()}, {len(fg_df)} days")
print(f"Whale trades: {whales['date'].min().date()} to {whales['date'].max().date()}, {len(whales)} trades, {whales['trader_name'].nunique()} traders")
print(f"Price data: {', '.join(f'{s}: {len(prices[s])}d' for s in prices)}")

# ─── Helper Functions ────────────────────────────────────────────────────
def backtest_signals(signal_series, price_df, hold_days=7):
    """Given a signal series (1=long, 0=flat) aligned to price_df dates,
    compute returns. Signal on bar N → enter bar N+1."""
    common = signal_series.index.intersection(price_df.index)
    sig = signal_series.reindex(common).fillna(0)
    ret = price_df['ret'].reindex(common).fillna(0)
    # Shift signal by 1 day (no look-ahead)
    pos = sig.shift(1).fillna(0)
    strat_ret = pos * ret
    
    total_ret = (1 + strat_ret).prod() - 1
    n_days = len(strat_ret)
    if n_days < 10:
        return None
    ann_ret = (1 + total_ret) ** (365 / max(n_days, 1)) - 1
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(365) if strat_ret.std() > 0 else 0
    max_dd = ((1 + strat_ret).cumprod() / (1 + strat_ret).cumprod().cummax() - 1).min()
    long_pct = (pos > 0).mean()
    bnh_ret = (1 + ret).prod() - 1
    
    return {
        'total_return': round(float(total_ret * 100), 2),
        'ann_return': round(float(ann_ret * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd * 100), 2),
        'exposure': round(float(long_pct * 100), 1),
        'n_days': n_days,
        'bnh_return': round(float(bnh_ret * 100), 2),
    }

def permutation_test(signal_series, price_df, n_perms=500):
    """Permutation test: shuffle signal, compare Sharpe."""
    common = signal_series.index.intersection(price_df.index)
    sig = signal_series.reindex(common).fillna(0)
    ret = price_df['ret'].reindex(common).fillna(0)
    pos = sig.shift(1).fillna(0)
    real_sharpe = (pos * ret).mean() / (pos * ret).std() * np.sqrt(365) if (pos * ret).std() > 0 else 0
    
    count_better = 0
    for _ in range(n_perms):
        perm_pos = pos.sample(frac=1).values
        perm_ret = perm_pos * ret.values
        perm_sharpe = perm_ret.mean() / perm_ret.std() * np.sqrt(365) if perm_ret.std() > 0 else 0
        if perm_sharpe >= real_sharpe:
            count_better += 1
    return count_better / n_perms

def walk_forward(signal_func, price_df, n_folds=None, **kwargs):
    """Walk-forward: apply signal_func on expanding window."""
    dates = price_df.index
    n = len(dates)
    if n_folds is None:
        n_folds = min(14, max(3, n // 30))
    fold_size = n // (n_folds + 1)
    
    oos_results = []
    for i in range(n_folds):
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        test_dates = dates[test_start:test_end]
        sig = signal_func(price_df, **kwargs)
        sig_oos = sig.reindex(test_dates)
        r = backtest_signals(sig_oos, price_df)
        if r:
            oos_results.append(r)
    
    if not oos_results:
        return None
    avg = {}
    for k in oos_results[0]:
        avg[k] = round(np.mean([r[k] for r in oos_results]), 3)
    avg['n_folds'] = len(oos_results)
    return avg

# ─── PART 1: Whale Trade Signals ─────────────────────────────────────────
print("\n" + "="*80)
print("PART 1: WHALE TRADE SIGNALS")
print("="*80)

# Whale trader statistics
print("\n--- Whale Trader Statistics ---")
# Identify opens and closes per trader
whale_opens = whales[whales['action'].isin(['flip', 'increase'])].copy()
whale_closes = whales[whales['action'] == 'close'].copy()

# Per-trader stats from closes
trader_stats = []
for trader in whales['trader_name'].unique():
    t_trades = whales[whales['trader_name'] == trader]
    t_closes = t_trades[t_trades['action'] == 'close']
    t_opens = t_trades[t_trades['action'].isin(['flip', 'increase'])]
    
    if len(t_closes) < 2:
        continue
    
    # Estimate win rate: compare entry_price to actual close price on close date
    wins = 0
    total_eval = 0
    for _, row in t_closes.iterrows():
        sym = row['symbol']
        if sym not in prices:
            continue
        close_date = pd.Timestamp(row['date'])
        if close_date in prices[sym].index:
            mkt_price = prices[sym].loc[close_date, 'close']
            total_eval += 1
            if row['position_side'] == 'LONG' and mkt_price > row['entry_price']:
                wins += 1
            elif row['position_side'] == 'SHORT' and mkt_price < row['entry_price']:
                wins += 1
    
    wr = wins / total_eval if total_eval > 0 else 0
    
    # Avg hold time estimate from date spans
    dates = sorted(t_trades['date'].unique())
    avg_hold = np.mean(np.diff([d.timestamp() for d in pd.to_datetime(dates)])) / 86400 if len(dates) > 1 else 7
    
    trader_stats.append({
        'trader': trader,
        'tier': t_trades['trader_tier'].iloc[0],
        'n_trades': len(t_trades),
        'n_closes': len(t_closes),
        'win_rate': round(wr * 100, 1),
        'symbols': list(t_trades['symbol'].unique()),
        'avg_hold_days': round(avg_hold, 1),
    })

ts_df = pd.DataFrame(trader_stats).sort_values('n_trades', ascending=False)
print(f"\nTotal traders with 2+ closes: {len(ts_df)}")
print(f"Elite traders (>60% WR): {len(ts_df[ts_df['win_rate'] > 60])}")
print(f"\nTop 15 traders by trade count:")
print(ts_df.head(15)[['trader','tier','n_trades','win_rate','avg_hold_days']].to_string(index=False))

elite_traders = set(ts_df[ts_df['win_rate'] > 60]['trader'].values)

# Build daily whale signals per symbol
def whale_daily_signal(sym, price_df, filter_elite=False, consensus_min=1, require_sma=False):
    """Build daily signal from whale trades."""
    wt = whales[whales['symbol'] == sym].copy()
    if filter_elite:
        wt = wt[wt['trader_name'].isin(elite_traders)]
    
    # For each day, count net whale direction from opens/flips
    opens = wt[wt['action'].isin(['flip', 'increase'])]
    
    signal = pd.Series(0.0, index=price_df.index)
    
    for date in opens['date'].unique():
        date = pd.Timestamp(date)
        if date not in signal.index:
            continue
        day_trades = opens[opens['date'] == date]
        longs = (day_trades['position_side'] == 'LONG').sum()
        shorts = (day_trades['position_side'] == 'SHORT').sum()
        net = longs - shorts
        
        if consensus_min > 1:
            if longs >= consensus_min:
                net = 1
            elif shorts >= consensus_min:
                net = -1
            else:
                net = 0
        
        if net > 0:
            # Hold for 7 days
            idx = signal.index.get_loc(date)
            end_idx = min(idx + 7, len(signal))
            signal.iloc[idx:end_idx] = 1
    
    if require_sma:
        sma_bull = (price_df['close'] > price_df['sma50']).astype(float)
        signal = signal * sma_bull
    
    return signal

results = {}
for sym in ['BTC', 'ETH', 'SOL']:
    if sym not in prices:
        continue
    pdf = prices[sym]
    whale_sym = whales[whales['symbol'] == sym]
    if len(whale_sym) < 10:
        continue
    
    print(f"\n--- {sym} Whale Signals ---")
    print(f"  Whale trades: {len(whale_sym)}")
    
    # Restrict price to whale data period
    wstart = whale_sym['date'].min()
    wend = whale_sym['date'].max()
    pdf_w = pdf[(pdf.index >= wstart) & (pdf.index <= wend)]
    
    # Strategy 1: Whale Copy
    sig1 = whale_daily_signal(sym, pdf_w)
    r1 = backtest_signals(sig1, pdf_w)
    if r1:
        p1 = permutation_test(sig1, pdf_w)
        r1['p_value'] = round(p1, 4)
        results[f'{sym}_whale_copy'] = r1
        print(f"  1. Whale Copy: ret={r1['total_return']}% sharpe={r1['sharpe']} p={p1:.3f}")
    
    # Strategy 2: Whale Consensus (3+)
    sig2 = whale_daily_signal(sym, pdf_w, consensus_min=3)
    r2 = backtest_signals(sig2, pdf_w)
    if r2:
        p2 = permutation_test(sig2, pdf_w)
        r2['p_value'] = round(p2, 4)
        results[f'{sym}_whale_consensus'] = r2
        print(f"  2. Whale Consensus: ret={r2['total_return']}% sharpe={r2['sharpe']} p={p2:.3f}")
    
    # Strategy 3: Elite Whale Only
    sig3 = whale_daily_signal(sym, pdf_w, filter_elite=True)
    r3 = backtest_signals(sig3, pdf_w)
    if r3:
        p3 = permutation_test(sig3, pdf_w)
        r3['p_value'] = round(p3, 4)
        results[f'{sym}_elite_whale'] = r3
        print(f"  3. Elite Whale: ret={r3['total_return']}% sharpe={r3['sharpe']} p={p3:.3f}")
    
    # Strategy 4: Whale + SMA50
    sig4 = whale_daily_signal(sym, pdf_w, require_sma=True)
    r4 = backtest_signals(sig4, pdf_w)
    if r4:
        p4 = permutation_test(sig4, pdf_w)
        r4['p_value'] = round(p4, 4)
        results[f'{sym}_whale_sma50'] = r4
        print(f"  4. Whale+SMA50: ret={r4['total_return']}% sharpe={r4['sharpe']} p={p4:.3f}")
    
    # SMA50 baseline
    sma_sig = (pdf_w['close'] > pdf_w['sma50']).astype(float)
    r_sma = backtest_signals(sma_sig, pdf_w)
    if r_sma:
        results[f'{sym}_sma50_baseline'] = r_sma
        print(f"  Baseline SMA50: ret={r_sma['total_return']}% sharpe={r_sma['sharpe']}")

# ─── PART 2: Fear & Greed ────────────────────────────────────────────────
print("\n" + "="*80)
print("PART 2: FEAR & GREED SIGNALS")
print("="*80)

# F&G distribution
print("\n--- F&G Distribution ---")
fg_bins = pd.cut(fg_df['fg'], bins=[0,20,40,50,60,80,100], labels=['0-20','20-40','40-50','50-60','60-80','80-100'])
print(fg_bins.value_counts().sort_index())

# Conditional returns at each F&G level
btc = prices['BTC'].copy()
btc_fg = btc.join(fg_df[['fg']], how='inner')
print("\n--- Conditional BTC Returns by F&G Level ---")
for label, (lo, hi) in [('Extreme Fear 0-20', (0,20)), ('Fear 20-40', (20,40)),
                          ('Neutral 40-60', (40,60)), ('Greed 60-80', (60,80)),
                          ('Extreme Greed 80-100', (80,100))]:
    mask = (btc_fg['fg'] >= lo) & (btc_fg['fg'] < hi)
    if mask.sum() > 0:
        fwd = btc_fg.loc[mask, 'ret'].shift(-1)  # next day return
        print(f"  {label}: n={mask.sum()}, avg_next_day={fwd.mean()*100:.3f}%, med={fwd.median()*100:.3f}%")

for sym in ['BTC', 'ETH', 'SOL']:
    if sym not in prices:
        continue
    pdf = prices[sym]
    pdf_fg = pdf.join(fg_df[['fg']], how='inner')
    if len(pdf_fg) < 100:
        continue
    
    print(f"\n--- {sym} F&G Strategies ---")
    
    # 5. F&G Contrarian
    sig5 = pd.Series(0.0, index=pdf_fg.index)
    sig5[pdf_fg['fg'] < 20] = 1.0  # long in extreme fear
    # flat when > 80 (already 0)
    r5 = backtest_signals(sig5, pdf_fg)
    if r5:
        p5 = permutation_test(sig5, pdf_fg)
        r5['p_value'] = round(p5, 4)
        results[f'{sym}_fg_contrarian'] = r5
        print(f"  5. F&G Contrarian: ret={r5['total_return']}% sharpe={r5['sharpe']} exp={r5['exposure']}% p={p5:.3f}")
    
    # 6. F&G Momentum
    fg_cross = pdf_fg['fg'].copy()
    sig6 = pd.Series(0.0, index=pdf_fg.index)
    above50 = fg_cross > 50
    cross_up = above50 & (~above50.shift(1).fillna(False))
    cross_down = (~above50) & (above50.shift(1).fillna(True))
    state = 0
    for i, dt in enumerate(sig6.index):
        if cross_up.iloc[i]:
            state = 1
        elif cross_down.iloc[i]:
            state = 0
        sig6.iloc[i] = state
    r6 = backtest_signals(sig6, pdf_fg)
    if r6:
        p6 = permutation_test(sig6, pdf_fg)
        r6['p_value'] = round(p6, 4)
        results[f'{sym}_fg_momentum'] = r6
        print(f"  6. F&G Momentum: ret={r6['total_return']}% sharpe={r6['sharpe']} exp={r6['exposure']}% p={p6:.3f}")
    
    # 7. V4 (SMA50) + F&G Filter (flat when >85)
    sig7 = (pdf_fg['close'] > pdf_fg['sma50']).astype(float)
    sig7[pdf_fg['fg'] > 85] = 0
    r7 = backtest_signals(sig7, pdf_fg)
    if r7:
        p7 = permutation_test(sig7, pdf_fg)
        r7['p_value'] = round(p7, 4)
        results[f'{sym}_v4_fg_filter'] = r7
        print(f"  7. V4+F&G Filter: ret={r7['total_return']}% sharpe={r7['sharpe']} exp={r7['exposure']}% p={p7:.3f}")
    
    # 8. F&G + SMA50 (only SMA longs when F&G < 60)
    sig8 = (pdf_fg['close'] > pdf_fg['sma50']).astype(float)
    sig8[pdf_fg['fg'] >= 60] = 0
    r8 = backtest_signals(sig8, pdf_fg)
    if r8:
        p8 = permutation_test(sig8, pdf_fg)
        r8['p_value'] = round(p8, 4)
        results[f'{sym}_fg_sma50'] = r8
        print(f"  8. F&G+SMA50: ret={r8['total_return']}% sharpe={r8['sharpe']} exp={r8['exposure']}% p={p8:.3f}")
    
    # SMA50 baseline for full F&G period
    sma_base = (pdf_fg['close'] > pdf_fg['sma50']).astype(float)
    r_base = backtest_signals(sma_base, pdf_fg)
    if r_base:
        results[f'{sym}_sma50_fg_period'] = r_base
        print(f"  SMA50 baseline: ret={r_base['total_return']}% sharpe={r_base['sharpe']}")
    
    bnh = backtest_signals(pd.Series(1.0, index=pdf_fg.index), pdf_fg)
    if bnh:
        results[f'{sym}_bnh_fg_period'] = bnh
        print(f"  B&H baseline: ret={bnh['total_return']}%")

# ─── PART 3: Options Max Pain ────────────────────────────────────────────
print("\n" + "="*80)
print("PART 3: OPTIONS MAX PAIN")
print("="*80)

# Parse max pain data
maxpain_btc = options[options['source_file'] == 'options_maxpain_BTC.csv'][['max_pain_price', 'date']].dropna()
maxpain_eth = options[options['source_file'] == 'options_maxpain_ETH.csv'][['max_pain_price', 'date']].dropna()

print(f"\nMax Pain BTC: {len(maxpain_btc)} expiry dates")
print(f"Max Pain ETH: {len(maxpain_eth)} expiry dates")

# Date format appears to be YYMMDD
for _, row in maxpain_btc.iterrows():
    d = int(row['date'])
    dt = pd.Timestamp(f"20{d//10000:02d}-{(d%10000)//100:02d}-{d%100:02d}")
    print(f"  Expiry: {dt.date()}, Max Pain: ${row['max_pain_price']:,.0f}")

print("\nNote: Options max pain data is snapshot-only (future expiry dates as of today).")
print("No historical max pain data available for backtesting strategies 9-10.")
print("Current max pain levels can be used as forward-looking reference only.")

# Current price vs max pain analysis
btc_latest = prices['BTC']['close'].iloc[-1]
eth_latest = prices['ETH']['close'].iloc[-1]
print(f"\nCurrent BTC: ${btc_latest:,.0f}")
print(f"Current ETH: ${eth_latest:,.0f}")

for _, row in maxpain_btc.iterrows():
    d = int(row['date'])
    dt = pd.Timestamp(f"20{d//10000:02d}-{(d%10000)//100:02d}-{d%100:02d}")
    mp = row['max_pain_price']
    dev = (btc_latest - mp) / mp * 100
    print(f"  BTC {dt.date()} expiry: MP=${mp:,.0f}, deviation={dev:+.1f}%")

# ─── Walk-Forward for F&G strategies (longer data) ──────────────────────
print("\n" + "="*80)
print("WALK-FORWARD VALIDATION (F&G strategies, BTC)")
print("="*80)

btc_fg = prices['BTC'].join(fg_df[['fg']], how='inner')
if len(btc_fg) > 200:
    def fg_contrarian_func(pdf, **kw):
        fg_vals = fg_df['fg'].reindex(pdf.index)
        sig = pd.Series(0.0, index=pdf.index)
        sig[fg_vals < 20] = 1.0
        return sig
    
    def fg_momentum_func(pdf, **kw):
        fg_vals = fg_df['fg'].reindex(pdf.index).fillna(50)
        above50 = fg_vals > 50
        cross_up = above50 & (~above50.shift(1).fillna(False))
        cross_down = (~above50) & (above50.shift(1).fillna(True))
        sig = pd.Series(0.0, index=pdf.index)
        state = 0
        for i in range(len(sig)):
            if cross_up.iloc[i]: state = 1
            elif cross_down.iloc[i]: state = 0
            sig.iloc[i] = state
        return sig
    
    def v4_fg_func(pdf, **kw):
        fg_vals = fg_df['fg'].reindex(pdf.index).fillna(50)
        sig = (pdf['close'] > pdf['sma50']).astype(float)
        sig[fg_vals > 85] = 0
        return sig
    
    for name, func in [('F&G Contrarian', fg_contrarian_func),
                        ('F&G Momentum', fg_momentum_func),
                        ('V4+F&G Filter', v4_fg_func)]:
        wf = walk_forward(func, btc_fg, n_folds=10)
        if wf:
            print(f"  {name} WF: sharpe={wf['sharpe']} ret={wf['total_return']}% folds={wf['n_folds']}")
            results[f'BTC_{name.replace(" ","_").replace("+","_")}_WF'] = wf

# ─── Summary Table ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary = pd.DataFrame(results).T
cols = ['total_return', 'sharpe', 'max_dd', 'exposure', 'p_value', 'bnh_return', 'n_days']
available_cols = [c for c in cols if c in summary.columns]
print(summary[available_cols].to_string())

# Bonferroni correction
n_tests = sum(1 for r in results.values() if 'p_value' in r)
if n_tests > 0:
    bonf_threshold = 0.05 / n_tests
    print(f"\nBonferroni threshold (α=0.05, {n_tests} tests): p < {bonf_threshold:.4f}")
    sig_strats = [k for k, v in results.items() if v.get('p_value', 1) < bonf_threshold]
    print(f"Significant after Bonferroni: {sig_strats if sig_strats else 'None'}")

# ─── Save Results ────────────────────────────────────────────────────────
outdir = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "whale_fg_options_test.json")

output = {
    'generated': datetime.now().isoformat(),
    'results': results,
    'whale_trader_stats': trader_stats[:20],
    'fg_distribution': fg_bins.value_counts().sort_index().to_dict(),
    'options_note': 'Max pain data is snapshot-only (future expiries). No historical data for backtesting.',
    'methodology': {
        'signal_delay': '1 day (bar N signal, bar N+1 trade)',
        'permutation_tests': 500,
        'bonferroni_correction': True,
        'whale_hold_period': '7 days',
    }
}
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
