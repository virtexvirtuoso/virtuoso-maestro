"""Test script for StockDataLoader + VectorBT compatibility."""

import sys
sys.path.insert(0, ".")

from datasource.yfinance_loader import StockDataLoader, BTC_PROXIES
import vectorbt as vbt
import pandas as pd
from datetime import datetime, timedelta

loader = StockDataLoader()

# 1. SPY 5 years daily
print("=" * 60)
print("1. SPY - 5 years daily")
start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
spy = loader.get_ohlcv("SPY", "1d", start_date=start)
print(f"   Shape: {spy.shape}, Range: {spy.index[0].date()} → {spy.index[-1].date()}")
print(f"   Close: {spy['close'].iloc[0]:.2f} → {spy['close'].iloc[-1]:.2f}")

# 2. MSTR max history
print("\n2. MSTR - max history")
mstr = loader.get_ohlcv("MSTR")
print(f"   Shape: {mstr.shape}, Range: {mstr.index[0].date()} → {mstr.index[-1].date()}")
print(f"   Close: {mstr['close'].iloc[0]:.2f} → {mstr['close'].iloc[-1]:.2f}")

# 3. All BTC_PROXIES
print("\n3. BTC_PROXIES")
proxies = loader.get_multiple(BTC_PROXIES)
for sym, df in proxies.items():
    if not df.empty:
        print(f"   {sym:5s}: {df.shape[0]:>5d} rows, {df.index[0].date()} → {df.index[-1].date()}, last={df['close'].iloc[-1]:.2f}")
    else:
        print(f"   {sym:5s}: NO DATA")

# 4. VectorBT SMA crossover on SPY
print("\n4. VectorBT SMA Crossover on SPY")
print("=" * 60)
close = spy["close"]
fast_ma = vbt.MA.run(close, window=20)
slow_ma = vbt.MA.run(close, window=50)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)
pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=100_000, freq="1D")
print(f"   Total Return:  {pf.total_return():.2%}")
print(f"   Sharpe Ratio:  {pf.sharpe_ratio():.3f}")
print(f"   Max Drawdown:  {pf.max_drawdown():.2%}")
print(f"   Win Rate:      {pf.trades.win_rate():.2%}")
print(f"   # Trades:      {pf.trades.count()}")
print(f"   Final Value:   ${pf.final_value():,.2f}")
print("\n✅ VectorBT compatibility confirmed!")
