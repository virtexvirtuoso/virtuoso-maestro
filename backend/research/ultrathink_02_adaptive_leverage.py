#!/usr/bin/env python3
"""
ULTRATHINK V2 — Validation #2: Adaptive Leverage via Signal Confluence
======================================================================
When multiple independent signals align, use Kelly-inspired leverage.
Tests whether multi-signal confluence predicts returns and improves Sharpe.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

print("=" * 70)
print("ULTRATHINK V2 — #2: Adaptive Leverage (Signal Confluence)")
print("=" * 70)

# --- Load data ---
print("\n📥 Downloading data...")
btc = yf.download('BTC-USD', start='2020-01-01', end='2026-02-12', progress=False)
gold = yf.download('GC=F', start='2020-01-01', end='2026-02-12', progress=False)
dxy = yf.download('DX-Y.NYB', start='2020-01-01', end='2026-02-12', progress=False)
tnx = yf.download('^TNX', start='2020-01-01', end='2026-02-12', progress=False)

for df in [btc, gold, dxy, tnx]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

# Load M2
try:
    from datasource.fred_loader import MacroDataLoader
    loader = MacroDataLoader()
    m2 = loader.get_series('M2SL')
    m2_yoy = m2.pct_change(12)
    m2_accel = m2_yoy.diff()
    has_m2 = True
    print(f"  M2: {len(m2)} obs")
except:
    has_m2 = False
    print("  M2: unavailable, using 4 signals")

# Load yield curve
try:
    yc = loader.get_series('T10Y2Y')
    has_yc = True
    print(f"  Yield Curve: {len(yc)} obs")
except:
    has_yc = False

# Load funding rates
funding_path = os.path.expanduser("~/Desktop/maestro/data/derivatives/btc_funding.csv")
try:
    fund = pd.read_csv(funding_path, parse_dates=['timestamp'])
    fund = fund.set_index('timestamp')['fundingRate'].resample('D').mean()
    has_funding = True
    print(f"  Funding: {len(fund)} days")
except:
    has_funding = False

# --- Build signals ---
print("\n🔧 Building 5 independent signals...")

df = pd.DataFrame(index=btc.index)
df['btc_close'] = btc['Close']
df['btc_ret'] = btc['Close'].pct_change()

# Signal 1: Price trend (close > 200d EMA)
df['ema200'] = df['btc_close'].ewm(span=200).mean()
df['sig_trend'] = (df['btc_close'] > df['ema200']).astype(int)

# Signal 2: Gold momentum (20d positive)
df['gold'] = gold['Close'].reindex(df.index, method='ffill')
df['sig_gold'] = (df['gold'].pct_change(20) > 0).astype(int)

# Signal 3: DXY weakness (20d negative)
df['dxy'] = dxy['Close'].reindex(df.index, method='ffill')
df['sig_dxy'] = (df['dxy'].pct_change(20) < 0).astype(int)

# Signal 4: M2 accelerating
if has_m2:
    m2_daily = m2_accel.reindex(df.index, method='ffill')
    df['sig_m2'] = (m2_daily > 0).astype(int)
else:
    # Fallback: yield change negative (easing)
    df['tnx'] = tnx['Close'].reindex(df.index, method='ffill')
    df['sig_m2'] = (df['tnx'].diff(20) < 0).astype(int)

# Signal 5: Funding not extreme (below 75th percentile)
if has_funding:
    fund_daily = fund.reindex(df.index, method='ffill')
    fund_75 = fund_daily.rolling(120).quantile(0.75)
    df['sig_funding'] = (fund_daily < fund_75).astype(int)
else:
    df['sig_funding'] = 1  # neutral if no data

# Confluence count
df['confluence'] = df['sig_trend'] + df['sig_gold'] + df['sig_dxy'] + df['sig_m2'] + df['sig_funding']
df = df.dropna()

# --- Test: Confluence vs Forward Returns ---
print("\n📊 Test 1: Confluence Count → Forward Returns")
df['fwd_5d'] = df['btc_close'].pct_change(5).shift(-5)
df['fwd_20d'] = df['btc_close'].pct_change(20).shift(-20)

for n in range(6):
    mask = df['confluence'] == n
    if mask.sum() > 20:
        ret_5d = df.loc[mask, 'fwd_5d'].mean() * 100
        ret_20d = df.loc[mask, 'fwd_20d'].mean() * 100
        sharpe = df.loc[mask, 'btc_ret'].mean() / df.loc[mask, 'btc_ret'].std() * np.sqrt(252) if df.loc[mask, 'btc_ret'].std() > 0 else 0
        print(f"  Confluence={n}: days={mask.sum():>5}  5d_ret={ret_5d:>6.2f}%  20d_ret={ret_20d:>6.2f}%  daily_Sharpe={sharpe:>5.2f}")

# --- Test: Adaptive leverage strategy ---
print("\n📊 Test 2: Adaptive Leverage Strategy")

# Leverage map
leverage_map = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}
df['leverage'] = df['confluence'].map(leverage_map)
df['strat_ret'] = df['leverage'].shift(1) * df['btc_ret']

# Compare strategies
strategies = {
    'Adaptive Leverage': df['strat_ret'],
    'Fixed 1x (always long)': df['btc_ret'],
    'Binary (long if confluence>=3)': (df['confluence'] >= 3).shift(1).fillna(0).astype(int) * df['btc_ret'],
}

print(f"\n  {'Strategy':<35} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>10} {'Exposure':>10}")
print(f"  {'-'*75}")

for name, rets in strategies.items():
    rets = rets.dropna()
    ann_ret = rets.mean() * 252
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cum = (1 + rets).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    exposure = (rets != 0).mean()
    print(f"  {name:<35} {ann_ret*100:>9.1f}% {sharpe:>8.2f} {dd*100:>9.1f}% {exposure*100:>9.1f}%")

# --- Test: Signal independence ---
print("\n📊 Test 3: Signal Independence (correlation matrix)")
sig_cols = ['sig_trend', 'sig_gold', 'sig_dxy', 'sig_m2', 'sig_funding']
corr_matrix = df[sig_cols].corr()
print(corr_matrix.round(2).to_string())

# Average pairwise correlation
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
avg_corr = upper.stack().mean()
print(f"\n  Average pairwise correlation: {avg_corr:.3f}")
print(f"  (Lower = more independent = better for confluence)")

print("\n✅ VALIDATION COMPLETE")
print("=" * 70)
