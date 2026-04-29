"""
Quick Validation #3: M2 Liquidity Growth + Volatility Regime
Thesis: Global liquidity (M2) growth rate drives crypto cycles. When M2 is expanding,
crypto momentum works better. Combine with vol regime (low vol → breakout coming).
"""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader

loader = StockDataLoader()
macro = MacroDataLoader()

# BTC daily
btc = loader.get_ohlcv("BTC-USD", timeframe="1d", start_date="2020-01-01", end_date="2026-02-01")
btc = btc[['close']]
btc.index = pd.DatetimeIndex(btc.index).tz_localize(None)

# M2 money supply (monthly) 
m2 = macro.get_series('M2SL', start_date='2019-01-01')
m2 = m2.to_frame('m2')
m2.index = pd.DatetimeIndex(m2.index).tz_localize(None)
m2['m2_yoy'] = m2['m2'].pct_change(12)  # YoY growth
m2['m2_mom'] = m2['m2'].pct_change(1)   # MoM growth
m2['m2_accel'] = m2['m2_yoy'].diff()     # Acceleration

# VIX (daily)
try:
    vix = macro.get_series('VIXCLS', start_date='2020-01-01')
    vix = vix.to_frame('vix')
    vix.index = pd.DatetimeIndex(vix.index).tz_localize(None)
except:
    # Fallback: compute BTC realized vol
    vix = None

# Forward fill M2 to daily
m2_daily = m2.resample('D').ffill()

df = btc.join(m2_daily, how='left').ffill()
if vix is not None:
    df = df.join(vix, how='left').ffill()

df['returns'] = df['close'].pct_change()
df['mom_20'] = df['close'].pct_change(20)
df['vol_20'] = df['returns'].rolling(20).std()
df['vol_z'] = (df['vol_20'] - df['vol_20'].rolling(120).mean()) / df['vol_20'].rolling(120).std()

df = df.dropna(subset=['m2_yoy', 'returns', 'mom_20'])

# Regimes
df['m2_regime'] = np.where(df['m2_yoy'] > 0, 'expanding', 'contracting')
df['vol_regime'] = np.where(df['vol_z'] < -0.5, 'low_vol', np.where(df['vol_z'] > 0.5, 'high_vol', 'normal'))

# Strategy: Full momentum when M2 expanding, half when contracting
# Boost position when vol is compressed (breakout setup)
df['signal_base'] = np.where(df['mom_20'] > 0, 1, -1)
df['position_size'] = 1.0
df.loc[df['m2_regime'] == 'contracting', 'position_size'] = 0.5
df.loc[(df['vol_regime'] == 'low_vol') & (df['mom_20'] > 0), 'position_size'] = 1.5  # vol breakout boost

df['strat_m2_vol'] = df['signal_base'].shift(1) * df['position_size'].shift(1) * df['returns']
df['strat_mom'] = df['signal_base'].shift(1) * df['returns']
df['bh'] = df['returns']

df = df.dropna()

def stats(rets, name):
    ann_ret = rets.mean() * 365
    ann_vol = rets.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod().iloc[-1] - 1
    dd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
    print(f"{name:30s} | Return: {cum:8.1%} | Sharpe: {sharpe:.2f} | Max DD: {dd:.1%}")

print(f"\n=== M2 Liquidity + Vol Regime Validation (BTC, {df.index[0].date()} to {df.index[-1].date()}) ===")
stats(df['strat_m2_vol'], "M2+Vol Momentum")
stats(df['strat_mom'], "Plain Momentum")
stats(df['bh'], "Buy & Hold")

# Regime analysis
print(f"\n--- Performance by M2 Regime ---")
for regime in ['expanding', 'contracting']:
    mask = df['m2_regime'] == regime
    r = df.loc[mask, 'returns'].dropna()
    print(f"  {regime:15s}: mean={r.mean()*365:.1%}/yr, sharpe={r.mean()/r.std()*np.sqrt(365):.2f}, days={len(r)}")

print(f"\n--- Performance by Vol Regime ---")
for regime in ['low_vol', 'normal', 'high_vol']:
    mask = df['vol_regime'] == regime
    r = df.loc[mask, 'returns'].dropna()
    if len(r) > 10:
        print(f"  {regime:15s}: mean={r.mean()*365:.1%}/yr, sharpe={r.mean()/r.std()*np.sqrt(365):.2f}, days={len(r)}")

# M2 acceleration signal
df['m2_accel_signal'] = np.where(df['m2_accel'] > 0, 'accelerating', 'decelerating')
print(f"\n--- BTC returns by M2 acceleration ---")
for regime in ['accelerating', 'decelerating']:
    mask = df['m2_accel_signal'] == regime
    r = df.loc[mask, 'returns'].dropna()
    print(f"  {regime:15s}: mean={r.mean()*365:.1%}/yr, sharpe={r.mean()/r.std()*np.sqrt(365):.2f}, days={len(r)}")
