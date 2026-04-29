#!/usr/bin/env python3
"""
ULTRATHINK V2 — Validation #5: Funding Rate Carry + Short Side
==============================================================
Can we profit from funding carry and controlled shorting in bear regimes?
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

print("=" * 70)
print("ULTRATHINK V2 — #5: Funding Carry + Short Side")
print("=" * 70)

# --- Load data ---
print("\n📥 Loading data...")

btc = yf.download('BTC-USD', start='2023-01-01', end='2026-02-12', progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.droplevel(1)

# Funding rates
funding_path = os.path.expanduser("~/Desktop/maestro/data/derivatives/btc_funding.csv")
fund = pd.read_csv(funding_path, parse_dates=['timestamp'])
fund = fund.set_index('timestamp')['fundingRate']
fund_daily = fund.resample('D').mean()
print(f"  Funding: {len(fund_daily)} daily obs")
print(f"  BTC: {len(btc)} daily obs")

# M2 for regime
try:
    from datasource.fred_loader import MacroDataLoader
    loader = MacroDataLoader()
    m2 = loader.get_series('M2SL')
    m2_accel = m2.pct_change(12).diff()
    has_m2 = True
except:
    has_m2 = False

# --- Build signals ---
df = pd.DataFrame(index=btc.index)
df['btc'] = btc['Close']
df['btc_ret'] = btc['Close'].pct_change()
df['funding'] = fund_daily.reindex(df.index, method='ffill')

# Rolling z-score of funding
df['fund_ma'] = df['funding'].rolling(60).mean()
df['fund_std'] = df['funding'].rolling(60).std()
df['fund_z'] = (df['funding'] - df['fund_ma']) / (df['fund_std'] + 1e-10)

# Regime
df['ema200'] = df['btc'].ewm(span=200).mean()
df['bull'] = (df['btc'] > df['ema200']).astype(int)

if has_m2:
    df['m2_accel'] = m2_accel.reindex(df.index, method='ffill')
    df['m2_bull'] = (df['m2_accel'] > 0).astype(int)

df = df.dropna()

# --- Test 1: Funding rate → forward returns ---
print("\n📊 Test 1: Funding Z-Score → Forward BTC Returns")

df['fwd_1d'] = df['btc'].pct_change().shift(-1)
df['fwd_5d'] = df['btc'].pct_change(5).shift(-5)

for label, lo, hi in [('Very Neg (z<-1.5)', -999, -1.5),
                       ('Negative (-1.5 to -0.5)', -1.5, -0.5),
                       ('Normal (-0.5 to 0.5)', -0.5, 0.5),
                       ('Elevated (0.5 to 1.5)', 0.5, 1.5),
                       ('Extreme (z>1.5)', 1.5, 999)]:
    mask = (df['fund_z'] >= lo) & (df['fund_z'] < hi)
    if mask.sum() > 10:
        r1 = df.loc[mask, 'fwd_1d'].mean() * 100
        r5 = df.loc[mask, 'fwd_5d'].mean() * 100
        print(f"  {label:<30} n={mask.sum():>5}  1d={r1:>6.3f}%  5d={r5:>6.3f}%")

# --- Test 2: Funding carry ---
print("\n📊 Test 2: Funding Carry Value")

# Annualized carry from funding
avg_funding = df['funding'].mean()
ann_carry = avg_funding * 3 * 365 * 100  # 3x daily, annualized
print(f"  Average daily funding: {avg_funding*100:.4f}%")
print(f"  Annualized carry (short collects): {ann_carry:.1f}%")

# Carry by regime
for label, mask in [('Bull (>200EMA)', df['bull'] == 1), ('Bear (<200EMA)', df['bull'] == 0)]:
    avg_f = df.loc[mask, 'funding'].mean()
    ann_c = avg_f * 3 * 365 * 100
    print(f"  {label}: avg funding={avg_f*100:.4f}%, ann carry={ann_c:.1f}%")

# --- Test 3: Short strategy in bear + high funding ---
print("\n📊 Test 3: Short Strategy (bear regime + elevated funding)")

# Strategy: short when bear AND funding z > 0.5 (longs paying, crowded)
df['short_signal'] = ((df['bull'] == 0) & (df['fund_z'] > 0.5)).astype(int)

# Short P&L = -price_change + funding_carry
df['short_ret'] = df['short_signal'].shift(1) * (-df['btc_ret'] + df['funding'].abs())

# Long strategy (our existing): long when bull
df['long_ret'] = df['bull'].shift(1) * df['btc_ret']

# Combined: long in bull, short in bear+high_funding, flat otherwise
df['combined_ret'] = df['long_ret'] + df['short_ret']

strategies = {
    'Long Only (bull)': df['long_ret'],
    'Short Only (bear+hi_fund)': df['short_ret'],
    'Combined L+S': df['combined_ret'],
    'Buy & Hold': df['btc_ret'],
}

print(f"\n  {'Strategy':<30} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>10} {'Exposure':>10}")
print(f"  {'-'*70}")

for name, rets in strategies.items():
    rets = rets.dropna()
    ann = rets.mean() * 252 * 100
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cum = (1 + rets).cumprod()
    dd = (cum / cum.cummax() - 1).min() * 100
    exp = (rets != 0).mean() * 100
    print(f"  {name:<30} {ann:>9.1f}% {sharpe:>8.2f} {dd:>9.1f}% {exp:>9.1f}%")

# --- Test 4: Funding extremes as entry signals ---
print("\n📊 Test 4: Extreme Funding as Contrarian Entry")

# When funding is extreme negative → buy signal (shorts overleveraged)
df['extreme_neg_fund'] = (df['fund_z'] < -1.5).astype(int)
df['extreme_pos_fund'] = (df['fund_z'] > 1.5).astype(int)

for label, sig_col in [('After extreme neg funding (buy)', 'extreme_neg_fund'),
                        ('After extreme pos funding (sell)', 'extreme_pos_fund')]:
    mask = df[sig_col] == 1
    if mask.sum() > 5:
        fwd = df.loc[mask, 'fwd_5d'].dropna()
        print(f"  {label}: n={len(fwd)}, mean_5d={fwd.mean()*100:.3f}%, hit_rate={(fwd>0).mean()*100:.0f}%")

# --- Test 5: Combined with M2 ---
if has_m2:
    print("\n📊 Test 5: Full System (M2 regime + funding filter)")
    
    df['full_long'] = ((df['m2_bull'] == 1) & (df['fund_z'] < 1.5)).astype(int)
    df['full_short'] = ((df['m2_bull'] == 0) & (df['fund_z'] > 0.5)).astype(int)
    df['full_ret'] = (df['full_long'].shift(1) * df['btc_ret'] + 
                      df['full_short'].shift(1) * (-df['btc_ret'] + df['funding'].abs()))
    
    full = df['full_ret'].dropna()
    ann = full.mean() * 252 * 100
    sharpe = full.mean() / full.std() * np.sqrt(252) if full.std() > 0 else 0
    cum = (1 + full).cumprod()
    dd = (cum / cum.cummax() - 1).min() * 100
    
    print(f"  M2+Funding System: Ann={ann:.1f}%, Sharpe={sharpe:.2f}, MaxDD={dd:.1f}%")
    print(f"  Long exposure: {df['full_long'].mean()*100:.1f}%, Short exposure: {df['full_short'].mean()*100:.1f}%")

print("\n✅ VALIDATION COMPLETE")
print("=" * 70)
