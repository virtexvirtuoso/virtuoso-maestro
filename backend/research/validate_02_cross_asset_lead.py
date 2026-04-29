"""
Quick Validation #2: Cross-Asset Lead-Lag (MSTR/COIN leading BTC)
Thesis: BTC proxy stocks trade during US hours and may lead BTC price moves
due to institutional flow. MSTR premium/discount contains info about demand.
"""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
loader = StockDataLoader()

btc = loader.get_ohlcv("BTC-USD", timeframe="1d", start_date="2022-01-01", end_date="2026-02-01")
mstr = loader.get_ohlcv("MSTR", timeframe="1d", start_date="2022-01-01", end_date="2026-02-01")
coin = loader.get_ohlcv("COIN", timeframe="1d", start_date="2022-01-01", end_date="2026-02-01")

# Align
btc_c = btc[['close']].rename(columns={'close': 'btc'})
mstr_c = mstr[['close']].rename(columns={'close': 'mstr'})
coin_c = coin[['close']].rename(columns={'close': 'coin'})

for s in [btc_c, mstr_c, coin_c]:
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)

df = btc_c.join(mstr_c, how='inner').join(coin_c, how='inner').dropna()

df['btc_ret'] = df['btc'].pct_change()
df['mstr_ret'] = df['mstr'].pct_change()
df['coin_ret'] = df['coin'].pct_change()

# Lead-lag: does today's MSTR return predict tomorrow's BTC return?
from scipy import stats as scipy_stats

for proxy, name in [('mstr_ret', 'MSTR'), ('coin_ret', 'COIN')]:
    for lag in [1, 2, 3]:
        x = df[proxy].dropna()
        y = df['btc_ret'].shift(-lag).reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        corr, pval = scipy_stats.pearsonr(x[common], y[common])
        print(f"{name}(t) → BTC(t+{lag}): corr={corr:.4f}, p={pval:.4f}")

# Strategy: Use MSTR relative strength as signal
# If MSTR outperforms BTC (proxy premium expanding) → bullish
df['mstr_rel'] = df['mstr_ret'] - df['btc_ret']  # MSTR excess return
df['mstr_rel_5d'] = df['mstr_rel'].rolling(5).sum()  # 5-day cumulative relative strength

df['signal'] = np.where(df['mstr_rel_5d'] > 0.02, 1,    # MSTR leading by >2%
               np.where(df['mstr_rel_5d'] < -0.02, -1, 0))  # MSTR lagging

df['strat_ret'] = df['signal'].shift(1) * df['btc_ret']
df['bh_ret'] = df['btc_ret']
df = df.dropna()

def stats(rets, name):
    ann_ret = rets.mean() * 365
    ann_vol = rets.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod().iloc[-1] - 1
    dd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
    exp = (rets != 0).mean()
    print(f"{name:25s} | Return: {cum:7.1%} | Sharpe: {sharpe:.2f} | Max DD: {dd:.1%} | Exposure: {exp:.1%}")

print(f"\n=== Cross-Asset Lead-Lag Validation (BTC, {df.index[0].date()} to {df.index[-1].date()}) ===")
print(f"Observations: {len(df)}")
stats(df['strat_ret'], "MSTR Rel Strength")
stats(df['bh_ret'], "BTC Buy & Hold")

print(f"\nSignal distribution:")
print(df['signal'].value_counts().to_string())

# ETH/BTC ratio as regime
eth = loader.get_ohlcv("ETH-USD", timeframe="1d", start_date="2022-01-01", end_date="2026-02-01")
eth_c = eth[['close']].rename(columns={'close': 'eth'})
eth_c.index = pd.DatetimeIndex(eth_c.index).tz_localize(None)
df2 = btc_c.join(eth_c, how='inner').dropna()
df2['eth_btc'] = df2['eth'] / df2['btc']
df2['eth_btc_ma50'] = df2['eth_btc'].rolling(50).mean()
df2['btc_ret'] = df2['btc'].pct_change()
# When ETH/BTC rising (risk-on altseason) → BTC momentum may differ
df2['ethbtc_trend'] = np.where(df2['eth_btc'] > df2['eth_btc_ma50'], 'risk_on', 'risk_off')
for regime in ['risk_on', 'risk_off']:
    mask = df2['ethbtc_trend'] == regime
    rets = df2.loc[mask, 'btc_ret'].dropna()
    print(f"\nBTC daily return in {regime}: mean={rets.mean()*365:.1%}/yr, vol={rets.std()*np.sqrt(365):.1%}, sharpe={rets.mean()/rets.std()*np.sqrt(365):.2f}, n={len(rets)}")
