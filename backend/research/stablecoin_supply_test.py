"""
Stablecoin Supply Signal Test — does stablecoin supply growth predict BTC returns?
Default thresholds only (supply_growth > 0), no optimization.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import numpy as np
import pandas as pd
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.stablecoin_loader import StablecoinLoader
from datasource.fred_loader import MacroDataLoader

TX_COST = 0.001
SHARPE_SCALE = np.sqrt(365)

# ── Load data ──────────────────────────────────────────────────────────────────

stock_loader = StockDataLoader()
stable_loader = StablecoinLoader()

btc = stock_loader.get_ohlcv("BTC-USD", timeframe="1d", start_date="2019-01-01")
btc = btc[["close"]].copy()
btc.index = pd.DatetimeIndex(btc.index).tz_localize(None)

supply_growth = stable_loader.get_supply_growth(period=30)
supply_growth.index = pd.DatetimeIndex(supply_growth.index).tz_localize(None)

# Merge
df = btc.join(supply_growth, how="inner")
df["returns"] = df["close"].pct_change()
df["fwd_30d_ret"] = df["close"].pct_change(30).shift(-30)

# Signal: supply growing (lagged 1 day for no-lookahead)
df["supply_growing"] = (df["supply_growth_30d"] > 0).astype(int).shift(1)
df = df.dropna(subset=["returns", "supply_growing"])

start_date = df.index[0].strftime("%Y-%m-%d")
end_date = df.index[-1].strftime("%Y-%m-%d")
n_days = len(df)

# ── Conditional returns ────────────────────────────────────────────────────────

def regime_stats(rets):
    ann_ret = rets.mean() * 365
    ann_vol = rets.std() * SHARPE_SCALE
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    return ann_ret, sharpe

growing_mask = df["supply_growing"] == 1
ret_growing = df.loc[growing_mask, "returns"]
ret_not_growing = df.loc[~growing_mask, "returns"]

ann_grow, sharpe_grow = regime_stats(ret_growing)
ann_notgrow, sharpe_notgrow = regime_stats(ret_not_growing)
spread = ann_grow - ann_notgrow

# ── Simple long/flat backtest ──────────────────────────────────────────────────

trades = df["supply_growing"].diff().abs().sum() / 2
avg_daily_trades = trades / n_days
daily_tc = avg_daily_trades * TX_COST

strat_ret = df["supply_growing"] * df["returns"] - daily_tc
equity = (1 + strat_ret).cumprod()
cagr = equity.iloc[-1] ** (365 / n_days) - 1
maxdd = (equity / equity.cummax() - 1).min()
sharpe_strat = strat_ret.mean() / strat_ret.std() * SHARPE_SCALE if strat_ret.std() > 0 else 0
time_in_mkt = df["supply_growing"].mean()

# ── Correlation analysis ───────────────────────────────────────────────────────

corr_df = df.dropna(subset=["supply_growth_30d", "fwd_30d_ret"])
pearson_r, pearson_p = stats.pearsonr(corr_df["supply_growth_30d"], corr_df["fwd_30d_ret"])
spearman_r, spearman_p = stats.spearmanr(corr_df["supply_growth_30d"], corr_df["fwd_30d_ret"])

# ── M2 comparison ─────────────────────────────────────────────────────────────

try:
    macro = MacroDataLoader()
    m2 = macro.get_series("M2SL", start_date="2018-01-01")
    m2.index = pd.DatetimeIndex(m2.index).tz_localize(None)
    m2_daily = m2.resample("D").ffill()
    m2_accel = m2_daily.pct_change(12 * 30).diff()  # ~YoY acceleration on daily-ffilled

    df2 = df.copy()
    df2 = df2.join(m2_accel.rename("m2_accel"), how="left").ffill()
    df2 = df2.dropna(subset=["m2_accel", "returns", "supply_growing"])

    m2_signal = (df2["m2_accel"] > 0).astype(int).shift(1)
    combined_signal = ((df2["supply_growing"] == 1) & (m2_signal == 1)).astype(int)

    m2_ret = m2_signal * df2["returns"]
    combined_ret = combined_signal * df2["returns"]

    sharpe_m2 = m2_ret.mean() / m2_ret.std() * SHARPE_SCALE if m2_ret.std() > 0 else 0
    sharpe_combined = combined_ret.mean() / combined_ret.std() * SHARPE_SCALE if combined_ret.std() > 0 else 0
    m2_available = True
except Exception as e:
    m2_available = False
    m2_err = str(e)

# ── Output ─────────────────────────────────────────────────────────────────────

print(f"""
STABLECOIN SUPPLY SIGNAL TEST
{'=' * 54}
Data: USDT+USDC supply, {start_date} to {end_date} ({n_days} days)

Conditional Returns:
  Supply Growing:     {ann_grow:6.1%}/yr annualized, Sharpe {sharpe_grow:.2f}
  Supply Not Growing: {ann_notgrow:6.1%}/yr annualized, Sharpe {sharpe_notgrow:.2f}
  Spread:             {spread:6.1%}/yr

Simple Long/Flat Backtest:
  Sharpe:       {sharpe_strat:.2f}
  CAGR:         {cagr:6.1%}
  MaxDD:        {maxdd:7.1%}
  Time in Mkt:  {time_in_mkt:6.1%}

Correlation (supply_growth_30d vs fwd_30d_btc_return):
  Pearson:  {pearson_r:.2f} (p={pearson_p:.3f})
  Spearman: {spearman_r:.2f} (p={spearman_p:.3f})
""")

if m2_available:
    print(f"""Comparison with M2 Acceleration:
  M2 Accel Signal:          Sharpe {sharpe_m2:.2f}
  Stablecoin Supply Signal: Sharpe {sharpe_strat:.2f}
  Combined (AND):           Sharpe {sharpe_combined:.2f}
""")
else:
    print(f"Comparison with M2 Acceleration: SKIPPED ({m2_err})")
