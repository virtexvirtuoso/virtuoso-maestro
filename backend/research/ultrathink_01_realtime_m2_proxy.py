#!/usr/bin/env python3
"""
ULTRATHINK V2 — Validation #1: Real-Time Daily M2 Proxy
========================================================
Can we build a daily composite that tracks M2 acceleration
using DXY, Gold, US10Y, HY spreads?

Tests:
1. Correlation of daily proxy with actual monthly M2 changes
2. Predictive power: proxy → forward BTC returns
3. Conditional Sharpe: long BTC only when proxy is positive
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

print("=" * 70)
print("ULTRATHINK V2 — #1: Real-Time M2 Proxy")
print("=" * 70)

# --- Load daily market data ---
tickers = {
    'DXY': 'DX-Y.NYB',
    'Gold': 'GC=F',
    'US10Y': '^TNX',
    'HYSpread': 'HYG',  # proxy: HYG price (inverse of spread)
    'BTC': 'BTC-USD',
}

print("\n📥 Downloading data...")
data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2020-01-01', end='2026-02-12', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        data[name] = df['Close'].dropna()
        print(f"  {name}: {len(data[name])} days")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# --- Load M2 from FRED ---
try:
    from datasource.fred_loader import MacroDataLoader
    loader = MacroDataLoader()
    m2 = loader.get_series('M2SL')
    m2_mom = m2.pct_change()  # monthly change
    m2_accel = m2_mom.diff()  # acceleration
    print(f"  M2: {len(m2)} monthly observations")
except Exception as e:
    print(f"  M2 FRED load failed: {e}")
    print("  Using manual calculation...")
    m2 = None

# --- Build daily proxy ---
print("\n🔧 Building daily M2 proxy composite...")

# Align all to common dates
common_idx = data['BTC'].index
for name in ['DXY', 'Gold', 'US10Y', 'HYSpread']:
    if name in data:
        common_idx = common_idx.intersection(data[name].index)

df = pd.DataFrame(index=common_idx)
for name in data:
    df[name] = data[name].reindex(common_idx)

df = df.dropna()
print(f"  Common dates: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

# Z-score each signal (60-day rolling)
window = 60

# Signals: negative DXY change, positive Gold change, negative yield change, positive HYG change (= tightening spreads)
df['sig_dxy'] = -df['DXY'].pct_change(20).rolling(window).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10))
df['sig_gold'] = df['Gold'].pct_change(20).rolling(window).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10))
df['sig_10y'] = -df['US10Y'].diff(20).rolling(window).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10))
df['sig_hyg'] = df['HYSpread'].pct_change(20).rolling(window).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10))

# Equal-weight composite
df['m2_proxy'] = (df['sig_dxy'] + df['sig_gold'] + df['sig_10y'] + df['sig_hyg']) / 4
df = df.dropna()

# --- Test 1: Correlation with actual M2 ---
if m2 is not None:
    print("\n📊 Test 1: Correlation with actual M2 acceleration")
    # Resample proxy to monthly, compare with M2
    proxy_monthly = df['m2_proxy'].resample('ME').last()
    m2_accel_aligned = m2_accel.reindex(proxy_monthly.index, method='ffill')
    
    both = pd.DataFrame({'proxy': proxy_monthly, 'm2_accel': m2_accel_aligned}).dropna()
    if len(both) > 10:
        corr, pval = stats.pearsonr(both['proxy'], both['m2_accel'])
        print(f"  Correlation: {corr:.3f} (p={pval:.4f})")
        print(f"  N months: {len(both)}")
    else:
        print("  Insufficient overlap")

# --- Test 2: Predictive power for BTC ---
print("\n📊 Test 2: Proxy → Forward BTC returns")
df['btc_ret_1d'] = df['BTC'].pct_change().shift(-1)
df['btc_ret_5d'] = df['BTC'].pct_change(5).shift(-5)
df['btc_ret_20d'] = df['BTC'].pct_change(20).shift(-20)

for horizon, col in [(1, 'btc_ret_1d'), (5, 'btc_ret_5d'), (20, 'btc_ret_20d')]:
    valid = df[['m2_proxy', col]].dropna()
    corr, pval = stats.pearsonr(valid['m2_proxy'], valid[col])
    print(f"  {horizon}d forward: corr={corr:.4f}, p={pval:.4f}, n={len(valid)}")

# --- Test 3: Conditional Sharpe ---
print("\n📊 Test 3: Conditional Sharpe (long when proxy > 0)")
df['btc_daily_ret'] = df['BTC'].pct_change()

# Strategy: long when proxy > 0, flat otherwise
df['signal'] = (df['m2_proxy'] > 0).astype(int)
df['strat_ret'] = df['signal'].shift(1) * df['btc_daily_ret']

# Calculate metrics
strat = df['strat_ret'].dropna()
bh = df['btc_daily_ret'].dropna()

strat_sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
bh_sharpe = bh.mean() / bh.std() * np.sqrt(252) if bh.std() > 0 else 0

strat_annual = strat.mean() * 252
bh_annual = bh.mean() * 252

exposure = df['signal'].mean()

# Max drawdown
cum_strat = (1 + strat).cumprod()
strat_dd = (cum_strat / cum_strat.cummax() - 1).min()

cum_bh = (1 + bh).cumprod()
bh_dd = (cum_bh / cum_bh.cummax() - 1).min()

print(f"\n  {'Metric':<25} {'Strategy':<15} {'Buy & Hold':<15}")
print(f"  {'-'*55}")
print(f"  {'Annual Return':<25} {strat_annual*100:>12.1f}% {bh_annual*100:>12.1f}%")
print(f"  {'Sharpe Ratio':<25} {strat_sharpe:>12.2f} {bh_sharpe:>12.2f}")
print(f"  {'Max Drawdown':<25} {strat_dd*100:>12.1f}% {bh_dd*100:>12.1f}%")
print(f"  {'Exposure':<25} {exposure*100:>12.1f}%")
print(f"  {'Total Return':<25} {(cum_strat.iloc[-1]-1)*100:>12.1f}% {(cum_bh.iloc[-1]-1)*100:>12.1f}%")

# --- Test 4: Regime analysis ---
print("\n📊 Test 4: Regime Analysis")
for label, mask in [('Proxy > 0.5 (strong bull)', df['m2_proxy'] > 0.5),
                     ('Proxy 0 to 0.5 (mild bull)', (df['m2_proxy'] > 0) & (df['m2_proxy'] <= 0.5)),
                     ('Proxy -0.5 to 0 (mild bear)', (df['m2_proxy'] <= 0) & (df['m2_proxy'] > -0.5)),
                     ('Proxy < -0.5 (strong bear)', df['m2_proxy'] < -0.5)]:
    regime_rets = df.loc[mask, 'btc_daily_ret'].dropna()
    if len(regime_rets) > 20:
        ann_ret = regime_rets.mean() * 252
        ann_sharpe = regime_rets.mean() / regime_rets.std() * np.sqrt(252) if regime_rets.std() > 0 else 0
        print(f"  {label:<35} days={len(regime_rets):>5}  ret={ann_ret*100:>7.1f}%/yr  Sharpe={ann_sharpe:>5.2f}")

print("\n✅ VALIDATION COMPLETE")
print("=" * 70)
