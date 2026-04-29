"""
Macro-Filtered Crypto Momentum Backtest
Tests whether macro regime awareness improves crypto momentum trading.
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add backend to path
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))
from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from datasource.factor_loader import FactorDataLoader

START = "2017-01-01"
END = "2026-02-12"

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 70)
print("MACRO-FILTERED CRYPTO MOMENTUM BACKTEST")
print("=" * 70)

print("\n📥 Loading data...")

stock_loader = StockDataLoader()
macro_loader = MacroDataLoader()
factor_loader = FactorDataLoader()

btc = stock_loader.get_ohlcv("BTC-USD", "1d", START, END)
eth = stock_loader.get_ohlcv("ETH-USD", "1d", START, END)
spy = stock_loader.get_ohlcv("SPY", "1d", START, END)

print(f"  BTC: {len(btc)} days ({btc.index[0].date()} to {btc.index[-1].date()})")
print(f"  ETH: {len(eth)} days ({eth.index[0].date()} to {eth.index[-1].date()})")
print(f"  SPY: {len(spy)} days")

# FRED macro series
macro_series = {
    'T10Y2Y': 'T10Y2Y',
    'M2SL': 'M2SL',
    'FEDFUNDS': 'FEDFUNDS',
    'CPIAUCSL': 'CPIAUCSL',
    'DCOILWTICO': 'DCOILWTICO',
    'GOLDAMGBD228NLBM': 'GOLDAMGBD228NLBM',
    'BAMLH0A0HYM2': 'BAMLH0A0HYM2',
}
macro_df = macro_loader.get_multiple(macro_series, start_date="2015-01-01", end_date=END)
print(f"  Macro: {len(macro_df)} observations, {list(macro_df.columns)}")

# Fama-French
try:
    ff3 = factor_loader.get_ff3()
    print(f"  FF3: {len(ff3)} months")
except Exception as e:
    print(f"  FF3: failed ({e}), skipping")
    ff3 = None

# ── 2. COMPUTE MACRO REGIME SCORE ─────────────────────────────────────────────

print("\n📊 Computing macro regime score...")

# Resample macro to daily and forward-fill
macro_daily = macro_df.resample('D').last().ffill()

# Yield curve positive
yield_pos = (macro_daily['T10Y2Y'] > 0).astype(int)

# M2 YoY > 0
m2_yoy = macro_daily['M2SL'].pct_change(365)  # approx YoY for daily-ffilled monthly
m2_expanding = (m2_yoy > 0).astype(int)

# M2 accelerating: YoY > 6mo MA
m2_yoy_6mo_ma = m2_yoy.rolling(180).mean()
m2_accelerating = (m2_yoy > m2_yoy_6mo_ma).astype(int)

# CPI declining: YoY < 3mo MA
cpi_yoy = macro_daily['CPIAUCSL'].pct_change(365)
cpi_yoy_3mo_ma = cpi_yoy.rolling(90).mean()
cpi_declining = (cpi_yoy < cpi_yoy_3mo_ma).astype(int)

# Fed not hiking: 3mo change <= 0
fed_3mo_chg = macro_daily['FEDFUNDS'].diff(90)
fed_not_hiking = (fed_3mo_chg <= 0).astype(int)

# HY spread tightening: spread < 3mo MA
hy_3mo_ma = macro_daily['BAMLH0A0HYM2'].rolling(90).mean()
hy_tightening = (macro_daily['BAMLH0A0HYM2'] < hy_3mo_ma).astype(int)

# Combine
macro_score = yield_pos + m2_expanding + m2_accelerating + cpi_declining + fed_not_hiking + hy_tightening
macro_score = macro_score.reindex(btc.index).ffill().fillna(0)

print(f"  Macro score distribution:")
for v in range(7):
    pct = (macro_score == v).mean() * 100
    if pct > 0:
        print(f"    Score {v}: {pct:.1f}%")

# ── 3. COMPUTE CRYPTO MOMENTUM SIGNALS ────────────────────────────────────────

print("\n📈 Computing crypto momentum signals...")

btc_close = btc['close']
eth_close = eth['close']

# BTC signals
btc_sma200 = btc_close.rolling(200).mean()
btc_sma50 = btc_close.rolling(50).mean()
btc_trend = btc_close > btc_sma200
btc_momentum = btc_close.pct_change(30) > 0
btc_golden_cross = btc_sma50 > btc_sma200

# RSI
delta = btc_close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
btc_rsi = 100 - (100 / (1 + rs))
btc_rsi_ok = (btc_rsi > 30) & (btc_rsi < 70)

# ETH signals
eth_sma200 = eth_close.rolling(200).mean()
eth_trend = eth_close > eth_sma200
eth_momentum = eth_close.pct_change(30) > 0

print(f"  BTC trend (>200SMA): {btc_trend.mean()*100:.1f}% of days")
print(f"  BTC momentum (30d ROC>0): {btc_momentum.mean()*100:.1f}% of days")
print(f"  BTC golden cross: {btc_golden_cross.mean()*100:.1f}% of days")

# ── 4. STRATEGY RETURNS ───────────────────────────────────────────────────────

print("\n🔧 Running strategies...")

btc_ret = btc_close.pct_change().fillna(0)
eth_ret = eth_close.pct_change().fillna(0)

# Align macro_score to btc index
ms = macro_score.reindex(btc.index).ffill().fillna(0)

def position_size_by_macro(score):
    """Map macro score to position size."""
    sz = pd.Series(0.0, index=score.index)
    sz[score >= 5] = 1.0
    sz[(score >= 3) & (score < 5)] = 0.6
    sz[(score >= 1) & (score < 3)] = 0.3
    sz[score == 0] = 0.0
    return sz

macro_pos_size = position_size_by_macro(ms)

# Strategy 1: Buy & Hold
s1_ret = btc_ret.copy()

# Strategy 2: Simple Momentum
s2_signal = (btc_trend & btc_momentum).astype(float)
s2_signal = s2_signal.shift(1).fillna(0)  # no lookahead
s2_ret = btc_ret * s2_signal

# Strategy 3: Momentum + Macro Binary
s3_signal = (btc_trend & btc_momentum & (ms >= 3)).astype(float)
s3_signal = s3_signal.shift(1).fillna(0)
s3_ret = btc_ret * s3_signal

# Strategy 4: Momentum + Macro Position Sizing
s4_signal = (btc_trend & btc_momentum).astype(float)
s4_pos = (s4_signal * macro_pos_size).shift(1).fillna(0)
s4_ret = btc_ret * s4_pos

# Strategy 5: Full System with trailing stop
s5_entry_signal = (btc_trend & btc_momentum & btc_golden_cross).shift(1).fillna(False)
s5_pos = pd.Series(0.0, index=btc.index)
s5_in_trade = False
s5_peak = 0.0
for i in range(len(btc.index)):
    idx = btc.index[i]
    price = btc_close.iloc[i]
    ms_val = ms.iloc[i] if i < len(ms) else 0
    
    if s5_in_trade:
        if price > s5_peak:
            s5_peak = price
        if price < s5_peak * 0.85:  # 15% trailing stop
            s5_in_trade = False
            s5_pos.iloc[i] = 0.0
            continue
        # Update position size by macro
        if ms_val >= 5: s5_pos.iloc[i] = 1.0
        elif ms_val >= 3: s5_pos.iloc[i] = 0.6
        elif ms_val >= 1: s5_pos.iloc[i] = 0.3
        else: s5_pos.iloc[i] = 0.0
    else:
        if i > 0 and s5_entry_signal.iloc[i]:
            s5_in_trade = True
            s5_peak = price
            if ms_val >= 5: s5_pos.iloc[i] = 1.0
            elif ms_val >= 3: s5_pos.iloc[i] = 0.6
            elif ms_val >= 1: s5_pos.iloc[i] = 0.3
            else: s5_pos.iloc[i] = 0.0

s5_ret = btc_ret * s5_pos

# Strategy 6: ETH with macro position sizing
eth_ms = macro_score.reindex(eth.index).ffill().fillna(0)
eth_macro_pos = position_size_by_macro(eth_ms)
s6_signal = (eth_trend & eth_momentum).astype(float)
s6_pos = (s6_signal * eth_macro_pos).shift(1).fillna(0)
s6_ret = eth_ret * s6_pos

# ── 5. PERFORMANCE METRICS ────────────────────────────────────────────────────

def calc_metrics(returns, name, pos_series=None):
    """Calculate comprehensive performance metrics."""
    equity = (1 + returns).cumprod()
    total_ret = equity.iloc[-1] - 1
    
    n_years = len(returns) / 252
    cagr = (equity.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    daily_rf = 0.0
    excess = returns - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    
    downside = returns[returns < 0].std()
    sortino = returns.mean() / downside * np.sqrt(252) if downside > 0 else 0
    
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Trades: count signal changes
    if pos_series is not None:
        in_market = (pos_series > 0)
        trades = (in_market.astype(int).diff().abs() > 0).sum() // 2
        time_in_market = in_market.mean() * 100
        avg_pos = pos_series[pos_series > 0].mean() if (pos_series > 0).any() else 0
    else:
        trades = 0
        time_in_market = 100.0
        avg_pos = 1.0
    
    # Win rate (monthly)
    monthly = returns.resample('ME').sum()
    win_rate = (monthly > 0).mean() * 100
    
    # Yearly returns
    yearly = returns.resample('YE').apply(lambda x: (1+x).prod() - 1)
    best_year = f"{yearly.idxmax().year}: {yearly.max()*100:.1f}%" if len(yearly) > 0 else "N/A"
    worst_year = f"{yearly.idxmin().year}: {yearly.min()*100:.1f}%" if len(yearly) > 0 else "N/A"
    
    return {
        'name': name,
        'total_return': round(total_ret * 100, 1),
        'cagr': round(cagr * 100, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 2),
        'win_rate_monthly': round(win_rate, 1),
        'trades': int(trades),
        'time_in_market': round(time_in_market, 1),
        'avg_position': round(avg_pos * 100, 1),
        'best_year': best_year,
        'worst_year': worst_year,
    }

# Compute all strategy metrics
strategies = {
    'S1: Buy & Hold BTC': calc_metrics(s1_ret, 'Buy & Hold BTC'),
    'S2: Simple Momentum': calc_metrics(s2_ret, 'Simple Momentum', s2_signal.shift(1).fillna(0)),
    'S3: Momentum+Macro Binary': calc_metrics(s3_ret, 'Momentum+Macro Binary', s3_signal),
    'S4: Momentum+Macro Sizing': calc_metrics(s4_ret, 'Momentum+Macro Sizing', s4_pos),
    'S5: Full System': calc_metrics(s5_ret, 'Full System', s5_pos),
    'S6: ETH Macro Sizing': calc_metrics(s6_ret, 'ETH Macro Sizing', s6_pos),
}

# ── PRINT COMPARISON TABLE ────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("STRATEGY COMPARISON")
print("=" * 100)
header = f"{'Strategy':<28} {'Return%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'WinR%':>6} {'InMkt%':>7}"
print(header)
print("-" * 100)
for k, v in strategies.items():
    print(f"{k:<28} {v['total_return']:>8.1f} {v['cagr']:>7.1f} {v['sharpe']:>7.2f} {v['sortino']:>8.2f} {v['max_dd']:>7.1f} {v['calmar']:>7.2f} {v['win_rate_monthly']:>6.1f} {v['time_in_market']:>7.1f}")

# Additional details
print("\n" + "-" * 70)
for k, v in strategies.items():
    print(f"  {k}: Best={v['best_year']}, Worst={v['worst_year']}, Trades={v['trades']}, AvgPos={v['avg_position']}%")

# ── 6. ADDITIONAL ANALYSIS ────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ADDITIONAL ANALYSIS")
print("=" * 70)

# Correlation of macro score with future BTC returns
btc_fwd_30d = btc_close.pct_change(30).shift(-30)
ms_aligned = ms.reindex(btc_fwd_30d.index)
valid = pd.DataFrame({'macro_score': ms_aligned, 'fwd_30d': btc_fwd_30d}).dropna()
corr = valid['macro_score'].corr(valid['fwd_30d'])
print(f"\n🔍 Macro Score → Next 30-day BTC Return Correlation: {corr:.4f}")

# Average forward return by macro score
print("\n  Avg 30-day forward BTC return by macro score:")
for score_val in range(7):
    subset = valid[valid['macro_score'] == score_val]
    if len(subset) > 0:
        print(f"    Score {score_val}: {subset['fwd_30d'].mean()*100:+.2f}% (n={len(subset)})")

# Drawdown analysis for major crashes
print("\n📉 Crash Drawdown Comparison:")
crashes = {
    '2018 Bear (-84%)': ('2018-01-01', '2018-12-15'),
    'COVID Mar 2020 (-50%)': ('2020-02-15', '2020-03-23'),
    'May 2021 (-55%)': ('2021-04-14', '2021-06-22'),
    '2022 Bear (-77%)': ('2022-01-01', '2022-12-31'),
}

for crash_name, (start, end) in crashes.items():
    mask = (btc.index >= start) & (btc.index <= end)
    if mask.sum() == 0:
        continue
    bh_dd = (btc_close[mask].iloc[-1] / btc_close[mask].iloc[0] - 1) * 100
    
    s2_eq = (1 + s2_ret[mask]).cumprod()
    s2_dd = (s2_eq.iloc[-1] - 1) * 100
    
    s4_eq = (1 + s4_ret[mask]).cumprod()
    s4_dd = (s4_eq.iloc[-1] - 1) * 100
    
    s5_eq = (1 + s5_ret[mask]).cumprod()
    s5_dd = (s5_eq.iloc[-1] - 1) * 100
    
    print(f"  {crash_name}:")
    print(f"    Buy&Hold: {bh_dd:+.1f}% | Momentum: {s2_dd:+.1f}% | Macro Sizing: {s4_dd:+.1f}% | Full System: {s5_dd:+.1f}%")

# Monthly return heatmap for Strategy 4
print("\n📅 Strategy 4 Monthly Returns (%):")
s4_monthly = s4_ret.resample('ME').apply(lambda x: (1+x).prod() - 1) * 100
s4_pivot = pd.DataFrame({
    'year': s4_monthly.index.year,
    'month': s4_monthly.index.month,
    'ret': s4_monthly.values
})
heatmap = s4_pivot.pivot_table(values='ret', index='year', columns='month', aggfunc='sum')
heatmap.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(heatmap.round(1).to_string())

# ── 7. SAVE RESULTS ───────────────────────────────────────────────────────────

results_dir = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
results_dir.mkdir(parents=True, exist_ok=True)

# Monthly heatmap as dict
heatmap_dict = {}
for yr in heatmap.index:
    heatmap_dict[str(yr)] = {m: round(v, 2) for m, v in heatmap.loc[yr].items() if not pd.isna(v)}

# Crash analysis dict
crash_analysis = {}
for crash_name, (start, end) in crashes.items():
    mask = (btc.index >= start) & (btc.index <= end)
    if mask.sum() == 0:
        continue
    crash_analysis[crash_name] = {
        'buy_hold': round((btc_close[mask].iloc[-1] / btc_close[mask].iloc[0] - 1) * 100, 1),
        'momentum': round(((1 + s2_ret[mask]).cumprod().iloc[-1] - 1) * 100, 1),
        'macro_sizing': round(((1 + s4_ret[mask]).cumprod().iloc[-1] - 1) * 100, 1),
        'full_system': round(((1 + s5_ret[mask]).cumprod().iloc[-1] - 1) * 100, 1),
    }

results = {
    'run_date': datetime.now().isoformat(),
    'period': f"{START} to {END}",
    'strategies': strategies,
    'macro_btc_correlation_30d': round(corr, 4),
    'avg_fwd_return_by_macro_score': {
        str(s): round(valid[valid['macro_score']==s]['fwd_30d'].mean()*100, 2)
        for s in range(7) if len(valid[valid['macro_score']==s]) > 0
    },
    'crash_analysis': crash_analysis,
    'monthly_heatmap_s4': heatmap_dict,
}

output_path = results_dir / "macro_crypto_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✅ Results saved to {output_path}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

s1 = strategies['S1: Buy & Hold BTC']
s2 = strategies['S2: Simple Momentum']
s4 = strategies['S4: Momentum+Macro Sizing']
s5 = strategies['S5: Full System']

print(f"""
1. MOMENTUM ALONE (S2 vs S1):
   Sharpe: {s2['sharpe']} vs {s1['sharpe']} | MaxDD: {s2['max_dd']}% vs {s1['max_dd']}%
   → Momentum {"improves" if s2['sharpe'] > s1['sharpe'] else "does not improve"} risk-adjusted returns

2. MACRO POSITION SIZING (S4 vs S2):
   Sharpe: {s4['sharpe']} vs {s2['sharpe']} | MaxDD: {s4['max_dd']}% vs {s2['max_dd']}%
   → Macro sizing {"improves" if s4['sharpe'] > s2['sharpe'] else "does not improve"} risk-adjusted returns
   → Time in market: {s4['time_in_market']}% vs {s2['time_in_market']}%

3. FULL SYSTEM (S5):
   Sharpe: {s5['sharpe']} | MaxDD: {s5['max_dd']}% | CAGR: {s5['cagr']}%
   → {"Best risk-adjusted" if s5['sharpe'] >= max(s2['sharpe'], s4['sharpe']) else "Not the best risk-adjusted"} returns

4. MACRO PREDICTS CRYPTO: Correlation = {corr:.4f}
   → {"Weak but positive" if 0 < corr < 0.1 else "Meaningful" if corr >= 0.1 else "Negligible/negative"} predictive power

5. VERDICT: Macro regime awareness {"DOES" if s4['sharpe'] > s2['sharpe'] else "DOES NOT"} improve crypto momentum trading.
""")
