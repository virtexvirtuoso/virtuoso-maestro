#!/usr/bin/env python3
"""
ULTRATHINK V2 — Validation #4: Cross-Asset Lead-Lag
====================================================
Do Gold, HY Spreads, DXY, Copper/Gold lead crypto by days/weeks?
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

print("=" * 70)
print("ULTRATHINK V2 — #4: Cross-Asset Lead-Lag → Crypto")
print("=" * 70)

# --- Load data ---
print("\n📥 Downloading cross-asset data...")
assets = {
    'BTC': 'BTC-USD',
    'Gold': 'GC=F',
    'DXY': 'DX-Y.NYB',
    'HYG': 'HYG',      # High yield bond ETF (inverse of spreads)
    'Copper': 'HG=F',
    'TLT': 'TLT',      # Long-term treasuries
    'VIX': '^VIX',
}

data = {}
for name, ticker in assets.items():
    try:
        d = yf.download(ticker, start='2020-01-01', end='2026-02-12', progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)
        data[name] = d['Close'].dropna()
        print(f"  {name}: {len(data[name])} days")
    except:
        print(f"  {name}: FAILED")

# Also try FRED HY spread
try:
    from datasource.fred_loader import MacroDataLoader
    loader = MacroDataLoader()
    hy_spread = loader.get_series('BAMLH0A0HYM2')
    data['HY_Spread'] = hy_spread
    print(f"  HY_Spread (FRED): {len(hy_spread)} obs")
except:
    pass

# --- Align ---
df = pd.DataFrame()
for name, series in data.items():
    df[name] = series

df = df.ffill().dropna()
print(f"\n  Aligned: {len(df)} days")

# --- Test 1: Lead-lag correlation ---
print("\n📊 Test 1: Lead-Lag Correlations (asset 20d return → BTC future return)")

df['btc_fwd_5d'] = df['BTC'].pct_change(5).shift(-5)
df['btc_fwd_10d'] = df['BTC'].pct_change(10).shift(-10)
df['btc_fwd_20d'] = df['BTC'].pct_change(20).shift(-20)

leaders = ['Gold', 'DXY', 'HYG', 'Copper', 'TLT', 'VIX']
if 'HY_Spread' in df.columns:
    leaders.append('HY_Spread')

print(f"\n  {'Asset 20d ret':<20} {'→ BTC 5d fwd':>15} {'→ BTC 10d fwd':>15} {'→ BTC 20d fwd':>15}")
print(f"  {'-'*70}")

for asset in leaders:
    if asset not in df.columns:
        continue
    if asset == 'HY_Spread':
        asset_ret = -df[asset].diff(20)  # negative spread change = bullish
    else:
        asset_ret = df[asset].pct_change(20)
    
    results = []
    for col in ['btc_fwd_5d', 'btc_fwd_10d', 'btc_fwd_20d']:
        valid = pd.DataFrame({'x': asset_ret, 'y': df[col]}).dropna()
        if len(valid) > 50:
            corr, pval = stats.pearsonr(valid['x'], valid['y'])
            star = '*' if pval < 0.05 else ''
            results.append(f"{corr:>6.3f}{star}")
        else:
            results.append("   N/A")
    
    print(f"  {asset:<20} {'  '.join(results)}")

# --- Test 2: Gold breakout → BTC follow ---
print("\n📊 Test 2: Gold Breakout → BTC Follow (specific events)")

df['gold_20d'] = df['Gold'].pct_change(20)
df['gold_breakout'] = df['gold_20d'] > df['gold_20d'].rolling(120).quantile(0.9)

breakout_days = df[df['gold_breakout']].index
if len(breakout_days) > 0:
    fwd_rets = []
    for day in breakout_days:
        loc = df.index.get_loc(day)
        for lag in [5, 10, 20]:
            if loc + lag < len(df):
                ret = (df['BTC'].iloc[loc + lag] / df['BTC'].iloc[loc] - 1)
                fwd_rets.append({'lag': lag, 'ret': ret})
    
    fwd_df = pd.DataFrame(fwd_rets)
    for lag in [5, 10, 20]:
        subset = fwd_df[fwd_df['lag'] == lag]['ret']
        print(f"  Gold breakout → BTC +{lag}d: mean={subset.mean()*100:.2f}%, median={subset.median()*100:.2f}%, n={len(subset)}, hit_rate={( subset>0).mean()*100:.0f}%")

# --- Test 3: HY Spread widening → BTC crash ---
print("\n📊 Test 3: HY Spread Widening → BTC Drawdown Warning")

if 'HY_Spread' in df.columns:
    df['hy_change_20d'] = df['HY_Spread'].diff(20)
    df['hy_widening'] = df['hy_change_20d'] > df['hy_change_20d'].rolling(120).quantile(0.85)
    
    # When HY spreads widen rapidly, what happens to BTC?
    for label, mask in [('HY Widening (>85th pct)', df['hy_widening'] == True),
                         ('HY Normal', df['hy_widening'] == False)]:
        rets = df.loc[mask, 'BTC'].pct_change().dropna()
        if len(rets) > 20:
            ann = rets.mean() * 252 * 100
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            print(f"  {label:<35} days={len(rets):>5}  ret={ann:>7.1f}%/yr  Sharpe={sharpe:>5.2f}")
else:
    # Use HYG as proxy
    df['hyg_20d'] = df['HYG'].pct_change(20)
    df['hyg_crash'] = df['hyg_20d'] < df['hyg_20d'].rolling(120).quantile(0.15)
    
    for label, mask in [('HYG Crashing (<15th pct)', df['hyg_crash'] == True),
                         ('HYG Normal', df['hyg_crash'] == False)]:
        rets = df.loc[mask, 'BTC'].pct_change().dropna()
        if len(rets) > 20:
            ann = rets.mean() * 252 * 100
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            print(f"  {label:<35} days={len(rets):>5}  ret={ann:>7.1f}%/yr  Sharpe={sharpe:>5.2f}")

# --- Test 4: Composite cross-asset signal ---
print("\n📊 Test 4: Composite Cross-Asset Signal → BTC")

# Bullish: Gold up, DXY down, HYG up (spreads tight), VIX low/falling
df['xsig_gold'] = (df['Gold'].pct_change(20) > 0).astype(int)
df['xsig_dxy'] = (df['DXY'].pct_change(20) < 0).astype(int)
df['xsig_hyg'] = (df['HYG'].pct_change(20) > 0).astype(int)
df['xsig_vix'] = (df['VIX'] < df['VIX'].rolling(60).median()).astype(int)

df['xsig_total'] = df['xsig_gold'] + df['xsig_dxy'] + df['xsig_hyg'] + df['xsig_vix']

df['btc_daily'] = df['BTC'].pct_change()

print(f"\n  {'Cross-Asset Score':<25} {'Days':>8} {'Ann Ret':>10} {'Sharpe':>8}")
print(f"  {'-'*55}")
for score in range(5):
    mask = df['xsig_total'] == score
    rets = df.loc[mask, 'btc_daily'].dropna()
    if len(rets) > 20:
        ann = rets.mean() * 252 * 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        print(f"  Score = {score:<20} {len(rets):>8} {ann:>9.1f}% {sharpe:>8.2f}")

# Strategy: long when score >= 3
df['xsig_strat'] = (df['xsig_total'] >= 3).shift(1).fillna(0).astype(int) * df['btc_daily']
strat = df['xsig_strat'].dropna()
bh = df['btc_daily'].dropna()

s_sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
b_sharpe = bh.mean() / bh.std() * np.sqrt(252) if bh.std() > 0 else 0
cum_s = (1 + strat).cumprod()
cum_b = (1 + bh).cumprod()

print(f"\n  Cross-Asset Strategy (score>=3):")
print(f"    Sharpe: {s_sharpe:.2f} vs B&H {b_sharpe:.2f}")
print(f"    Total: {(cum_s.iloc[-1]-1)*100:.1f}% vs B&H {(cum_b.iloc[-1]-1)*100:.1f}%")
print(f"    MaxDD: {(cum_s/cum_s.cummax()-1).min()*100:.1f}% vs B&H {(cum_b/cum_b.cummax()-1).min()*100:.1f}%")
print(f"    Exposure: {(df['xsig_total']>=3).mean()*100:.1f}%")

print("\n✅ VALIDATION COMPLETE")
print("=" * 70)
