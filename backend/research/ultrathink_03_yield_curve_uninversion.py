#!/usr/bin/env python3
"""
ULTRATHINK V2 — Validation #3: Yield Curve Uninversion Front-Run
================================================================
Does yield curve steepening (from inverted) predict crypto rallies?
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

print("=" * 70)
print("ULTRATHINK V2 — #3: Yield Curve Uninversion → Crypto")
print("=" * 70)

# --- Load data ---
print("\n📥 Loading data...")
btc = yf.download('BTC-USD', start='2019-01-01', end='2026-02-12', progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.droplevel(1)

try:
    from datasource.fred_loader import MacroDataLoader
    loader = MacroDataLoader()
    yc = loader.get_series('T10Y2Y')  # 10Y-2Y spread
    ff = loader.get_series('FEDFUNDS')
    print(f"  Yield curve: {len(yc)} obs")
    print(f"  Fed Funds: {len(ff)} obs")
except Exception as e:
    print(f"  FRED failed: {e}")
    sys.exit(1)

# --- Build signals ---
df = pd.DataFrame(index=btc.index)
df['btc'] = btc['Close']
df['btc_ret'] = btc['Close'].pct_change()

# Yield curve (daily FRED)
df['yc'] = yc.reindex(df.index, method='ffill')
df['yc_ma20'] = df['yc'].rolling(20).mean()
df['yc_change_60d'] = df['yc'].diff(60)  # 60-day change in spread

# Fed funds (monthly)
df['ff'] = ff.reindex(df.index, method='ffill')
df['ff_change_3m'] = df['ff'].diff(63)  # ~3 month change

df = df.dropna()

# --- Test 1: Yield curve regime → BTC returns ---
print("\n📊 Test 1: Yield Curve Regime → BTC Returns")

regimes = {
    'Deeply Inverted (<-0.5)': df['yc'] < -0.5,
    'Mildly Inverted (-0.5 to 0)': (df['yc'] >= -0.5) & (df['yc'] < 0),
    'Mildly Positive (0 to 0.5)': (df['yc'] >= 0) & (df['yc'] < 0.5),
    'Strongly Positive (>0.5)': df['yc'] >= 0.5,
}

for name, mask in regimes.items():
    rets = df.loc[mask, 'btc_ret'].dropna()
    if len(rets) > 20:
        ann = rets.mean() * 252 * 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        print(f"  {name:<40} days={len(rets):>5}  ret={ann:>7.1f}%/yr  Sharpe={sharpe:>5.2f}")

# --- Test 2: Yield curve CHANGE → BTC returns ---
print("\n📊 Test 2: Yield Curve 60d Change → Forward BTC Returns")

df['fwd_20d'] = df['btc'].pct_change(20).shift(-20)
df['fwd_60d'] = df['btc'].pct_change(60).shift(-60)

# Steepening vs flattening
steep = df['yc_change_60d'] > 0.1  # steepening
flat = df['yc_change_60d'] < -0.1  # flattening

for label, mask in [('Steepening (>0.1)', steep), ('Flattening (<-0.1)', flat)]:
    valid = df.loc[mask].dropna(subset=['fwd_20d', 'fwd_60d'])
    if len(valid) > 20:
        print(f"  {label}: n={len(valid)}, 20d_fwd={valid['fwd_20d'].mean()*100:.2f}%, 60d_fwd={valid['fwd_60d'].mean()*100:.2f}%")

# --- Test 3: The Uninversion Signal ---
print("\n📊 Test 3: Uninversion Events (cross from negative to positive)")

# Find uninversion events
df['was_inverted'] = (df['yc'].shift(20) < 0).astype(int)
df['now_positive'] = (df['yc'] > 0).astype(int)
df['uninversion'] = (df['was_inverted'] == 1) & (df['now_positive'] == 1) & (df['now_positive'].shift(1) == 0)

events = df[df['uninversion']].index
print(f"  Found {len(events)} uninversion events")

for event in events:
    # Forward returns after uninversion
    loc = df.index.get_loc(event)
    for days in [5, 20, 60, 90, 120]:
        if loc + days < len(df):
            fwd_ret = (df['btc'].iloc[loc + days] / df['btc'].iloc[loc] - 1) * 100
            print(f"    {event.date()}: +{days}d → {fwd_ret:+.1f}%")

# --- Test 4: Combined signal: YC steepening + Fed cutting ---
print("\n📊 Test 4: Combined — YC Steepening + Fed Cutting")

df['signal_combined'] = ((df['yc_change_60d'] > 0) & (df['ff_change_3m'] < 0)).astype(int)

for label, sig_val in [('Both active', 1), ('Neither', 0)]:
    mask = df['signal_combined'] == sig_val
    rets = df.loc[mask, 'btc_ret'].dropna()
    if len(rets) > 20:
        ann = rets.mean() * 252 * 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        print(f"  {label}: days={len(rets):>5}  ret={ann:>7.1f}%/yr  Sharpe={sharpe:>5.2f}")

# --- Strategy backtest ---
print("\n📊 Test 5: Strategy — Long BTC when YC improving from inversion")

# Signal: YC was inverted in last 120 days AND is now improving (20d MA rising)
df['yc_improving'] = (df['yc_ma20'].diff(20) > 0).astype(int)
df['recent_inversion'] = df['yc'].rolling(120).min().lt(0).astype(int)
df['entry_signal'] = df['yc_improving'] & df['recent_inversion']

# Also include: normal positive YC regime
df['bull_signal'] = ((df['yc'] > 0) & (df['yc_change_60d'] > -0.2)).astype(int)
df['final_signal'] = ((df['entry_signal'] == 1) | (df['bull_signal'] == 1)).astype(int)

df['strat_ret'] = df['final_signal'].shift(1) * df['btc_ret']

strat = df['strat_ret'].dropna()
bh = df['btc_ret'].dropna()

s_sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
b_sharpe = bh.mean() / bh.std() * np.sqrt(252) if bh.std() > 0 else 0

cum_s = (1 + strat).cumprod()
cum_b = (1 + bh).cumprod()

print(f"\n  {'Metric':<25} {'Strategy':<15} {'Buy & Hold':<15}")
print(f"  {'-'*55}")
print(f"  {'Ann Return':<25} {strat.mean()*252*100:>12.1f}% {bh.mean()*252*100:>12.1f}%")
print(f"  {'Sharpe':<25} {s_sharpe:>12.2f} {b_sharpe:>12.2f}")
print(f"  {'Max DD':<25} {(cum_s/cum_s.cummax()-1).min()*100:>12.1f}% {(cum_b/cum_b.cummax()-1).min()*100:>12.1f}%")
print(f"  {'Exposure':<25} {df['final_signal'].mean()*100:>12.1f}%")
print(f"  {'Total Return':<25} {(cum_s.iloc[-1]-1)*100:>12.1f}% {(cum_b.iloc[-1]-1)*100:>12.1f}%")

print("\n✅ VALIDATION COMPLETE")
print("=" * 70)
