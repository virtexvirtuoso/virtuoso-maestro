"""
Quick Validation #1: Carry + Momentum
Thesis: Combine funding rate carry (positive = long bias in market) with price momentum.
Go long when momentum is positive AND funding is not extreme (avoid crowded trades).
Go short when momentum is negative AND funding is high (crowded longs unwinding).
"""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

# Load BTC price
from datasource.yfinance_loader import StockDataLoader
loader = StockDataLoader()
btc = loader.get_ohlcv("BTC-USD", timeframe="1d", start_date="2023-06-01", end_date="2026-02-01")
btc = btc[['close']]
btc.index = pd.DatetimeIndex(btc.index).tz_localize(None)

# Load funding rates - resample to daily
funding = pd.read_csv(os.path.expanduser("~/Desktop/maestro/data/derivatives/btc_funding.csv"), parse_dates=['timestamp'])
funding = funding.set_index('timestamp')
funding.index = funding.index.tz_localize(None)
# Daily mean funding rate
funding_daily = funding['fundingRate'].resample('1D').mean().to_frame('funding')

# Merge
df = btc.join(funding_daily, how='inner')
df = df.dropna()

# Signals
df['returns'] = df['close'].pct_change()
df['mom_20'] = df['close'].pct_change(20)  # 20-day momentum
df['funding_ma7'] = df['funding'].rolling(7).mean()  # smoothed funding
df['funding_z'] = (df['funding_ma7'] - df['funding_ma7'].rolling(60).mean()) / df['funding_ma7'].rolling(60).std()

# Strategy: Long when momentum > 0 and funding not extremely high (z < 1.5)
# Short when momentum < 0 and funding high (z > 1) — crowded longs
df['signal'] = 0
df.loc[(df['mom_20'] > 0) & (df['funding_z'] < 1.5), 'signal'] = 1
df.loc[(df['mom_20'] < 0) & (df['funding_z'] > 1.0), 'signal'] = -1

# Also test pure momentum for comparison
df['signal_mom'] = np.where(df['mom_20'] > 0, 1, -1)

# Strategy returns (next-day)
df['strat_carry_mom'] = df['signal'].shift(1) * df['returns']
df['strat_mom_only'] = df['signal_mom'].shift(1) * df['returns']
df['buy_hold'] = df['returns']

df = df.dropna()

def stats(rets, name):
    ann_ret = rets.mean() * 365
    ann_vol = rets.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod().iloc[-1] - 1
    dd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
    print(f"{name:25s} | Return: {cum:7.1%} | Ann Sharpe: {sharpe:.2f} | Max DD: {dd:.1%} | Exposure: {(rets != 0).mean():.1%}")

print(f"\n=== Carry + Momentum Validation (BTC, {df.index[0].date()} to {df.index[-1].date()}) ===")
print(f"Observations: {len(df)}")
stats(df['strat_carry_mom'], "Carry+Momentum")
stats(df['strat_mom_only'], "Momentum Only")
stats(df['buy_hold'], "Buy & Hold")

# Signal analysis
print(f"\nSignal distribution:")
print(df['signal'].value_counts().to_string())
print(f"\nFunding Z-score stats: mean={df['funding_z'].mean():.2f}, std={df['funding_z'].std():.2f}")
