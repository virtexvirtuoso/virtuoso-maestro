"""
TEST 1: Head-to-Head Proxy Comparison for Patent Validation
Compare 3 liquidity proxies for BTC position sizing (2018-2025):
  A) Our 4-instrument proxy (DXY/Gold/TLT/HYG, 3-of-4 consensus)
  B) Alden-style Net Liquidity (WALCL - WTREGEN - RRPONTSYD)
  C) Simple M2 only (acceleration)
"""
import sys, json, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from pathlib import Path

# --- Config ---
START = "2018-01-01"
END = "2025-12-31"
TX_COST = 0.001
LOOKBACK = 20
OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))

# --- Load FRED API key ---
def get_fred_key():
    key = os.environ.get("FRED_API_KEY")
    if key: return key
    env_path = os.path.expanduser("~/Desktop/btc_wiz/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("FRED_API_KEY="):
                return line.split("=",1)[1].strip()
    raise ValueError("No FRED_API_KEY found")

fred = Fred(api_key=get_fred_key())

# --- Download data ---
print("=" * 60)
print("DOWNLOADING DATA")
print("=" * 60)

# BTC
print("Downloading BTC-USD...")
btc = yf.download("BTC-USD", start=START, end=END, progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)
btc_close = btc["Close"].copy()
btc_close.index = pd.DatetimeIndex(btc_close.index).tz_localize(None)
btc_ret = btc_close.pct_change().fillna(0)
print(f"  BTC: {len(btc_close)} days, {btc_close.index[0].date()} to {btc_close.index[-1].date()}")

# Cross-asset for Proxy A
print("Downloading cross-asset ETFs (UUP, GLD, TLT, HYG)...")
etfs = yf.download(["UUP", "GLD", "TLT", "HYG"], start="2017-01-01", end=END, progress=False)
if isinstance(etfs.columns, pd.MultiIndex):
    close_etfs = etfs["Close"]
else:
    close_etfs = etfs[["Close"]]
for c in close_etfs.columns:
    close_etfs[c] = pd.to_numeric(close_etfs[c], errors='coerce')
close_etfs.index = pd.DatetimeIndex(close_etfs.index).tz_localize(None)
close_etfs = close_etfs.ffill()
print(f"  ETFs: {len(close_etfs)} days")

# FRED for Proxy B
print("Downloading FRED series (WALCL, WTREGEN, RRPONTSYD, M2SL)...")
walcl = fred.get_series("WALCL", observation_start="2017-01-01")
wtregen = fred.get_series("WTREGEN", observation_start="2017-01-01")
rrp = fred.get_series("RRPONTSYD", observation_start="2017-01-01")
m2 = fred.get_series("M2SL", observation_start="2017-01-01")

for s in [walcl, wtregen, rrp, m2]:
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)

print(f"  WALCL: {len(walcl)}, WTREGEN: {len(wtregen)}, RRP: {len(rrp)}, M2: {len(m2)}")

# --- Build signals ---
print("\n" + "=" * 60)
print("COMPUTING PROXY SIGNALS")
print("=" * 60)

idx = btc_close.index

# PROXY A: 4-instrument real-time proxy (from mega_strategy_v3.py signal 2)
print("\nProxy A: 4-Instrument Real-Time Proxy")
dxy = close_etfs["UUP"].reindex(idx, method="ffill").ffill()
gold = close_etfs["GLD"].reindex(idx, method="ffill").ffill()
bonds = close_etfs["TLT"].reindex(idx, method="ffill").ffill()
hyg = close_etfs["HYG"].reindex(idx, method="ffill").ffill()

liq_score = pd.Series(0.0, index=idx)
liq_score += (dxy.pct_change(LOOKBACK) < 0).astype(float).fillna(0)   # DXY declining
liq_score += (gold.pct_change(LOOKBACK) > 0).astype(float).fillna(0)  # Gold rising
liq_score += (bonds.pct_change(LOOKBACK) > 0).astype(float).fillna(0) # TLT rising (yields falling)
liq_score += (hyg.pct_change(LOOKBACK) > 0).astype(float).fillna(0)   # HYG rising (spreads tight)
sig_a = (liq_score >= 3).astype(int).shift(1).fillna(0).astype(int)
print(f"  Bullish days: {sig_a.sum()} / {len(sig_a)} ({100*sig_a.mean():.1f}%)")

# PROXY B: Alden-style Net Liquidity
print("\nProxy B: Alden Net Liquidity (WALCL - TGA - RRP)")
net_liq = walcl.reindex(idx, method="ffill").ffill() - \
          wtregen.reindex(idx, method="ffill").ffill() - \
          rrp.reindex(idx, method="ffill").ffill()
net_liq_roc = net_liq.pct_change(LOOKBACK)
sig_b = (net_liq_roc > 0).astype(int).shift(1).fillna(0).astype(int)
print(f"  Bullish days: {sig_b.sum()} / {len(sig_b)} ({100*sig_b.mean():.1f}%)")

# PROXY C: Simple M2 acceleration
print("\nProxy C: M2 Acceleration")
m2_daily = m2.reindex(idx, method="ffill").ffill()
m2_3m = m2_daily.pct_change(63)   # ~3 months
m2_6m = m2_daily.pct_change(126)  # ~6 months
sig_c = (m2_3m > m2_6m).astype(int).shift(1).fillna(0).astype(int)
print(f"  Bullish days: {sig_c.sum()} / {len(sig_c)} ({100*sig_c.mean():.1f}%)")

# --- Backtest function ---
def backtest(signal, btc_ret, btc_close, name):
    """Long BTC when signal=1, flat when signal=0. 0.1% tx cost per trade."""
    signal = signal.reindex(btc_ret.index).fillna(0).astype(int)
    trades = signal.diff().abs().fillna(0)
    costs = trades * TX_COST
    daily_pnl = signal * btc_ret - costs
    equity = (1 + daily_pnl).cumprod()
    
    total_ret = equity.iloc[-1] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1
    
    # Sharpe
    ann_ret = daily_pnl.mean() * 252
    ann_vol = daily_pnl.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Max DD
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    
    n_trades = int(trades.sum())
    
    print(f"\n  {name}:")
    print(f"    Total Return: {total_ret*100:.1f}%")
    print(f"    CAGR:         {cagr*100:.1f}%")
    print(f"    Sharpe:       {sharpe:.2f}")
    print(f"    Max DD:       {max_dd*100:.1f}%")
    print(f"    Trades:       {n_trades}")
    
    return {
        "name": name,
        "total_return": round(float(total_ret), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd), 4),
        "n_trades": n_trades,
        "bullish_pct": round(float(signal.mean()), 4),
        "equity": equity,
        "signal": signal,
    }

# --- Run backtests ---
print("\n" + "=" * 60)
print("BACKTEST RESULTS")
print("=" * 60)

# Buy and hold baseline
bh_eq = (1 + btc_ret).cumprod()
bh_years = (bh_eq.index[-1] - bh_eq.index[0]).days / 365.25
bh_cagr = bh_eq.iloc[-1] ** (1/bh_years) - 1
bh_sharpe = (btc_ret.mean() * 252) / (btc_ret.std() * np.sqrt(252))
bh_dd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()
print(f"\n  Buy & Hold BTC:")
print(f"    Total Return: {(bh_eq.iloc[-1]-1)*100:.1f}%")
print(f"    CAGR:         {bh_cagr*100:.1f}%")
print(f"    Sharpe:       {bh_sharpe:.2f}")
print(f"    Max DD:       {bh_dd*100:.1f}%")

res_a = backtest(sig_a, btc_ret, btc_close, "Proxy A: 4-Instrument (Ours)")
res_b = backtest(sig_b, btc_ret, btc_close, "Proxy B: Alden Net Liquidity")
res_c = backtest(sig_c, btc_ret, btc_close, "Proxy C: M2 Acceleration")

# --- Signal correlations ---
print("\n" + "=" * 60)
print("SIGNAL CORRELATIONS")
print("=" * 60)

corr_ab = sig_a.corr(sig_b)
corr_ac = sig_a.corr(sig_c)
corr_bc = sig_b.corr(sig_c)
print(f"  A vs B (Ours vs Alden):  {corr_ab:.3f}")
print(f"  A vs C (Ours vs M2):     {corr_ac:.3f}")
print(f"  B vs C (Alden vs M2):    {corr_bc:.3f}")

# --- Agreement analysis ---
print("\n" + "=" * 60)
print("AGREEMENT ANALYSIS")
print("=" * 60)
agree_ab = (sig_a == sig_b).mean()
agree_ac = (sig_a == sig_c).mean()
agree_bc = (sig_b == sig_c).mean()
all_agree = ((sig_a == sig_b) & (sig_b == sig_c)).mean()
print(f"  A agrees with B: {agree_ab*100:.1f}%")
print(f"  A agrees with C: {agree_ac*100:.1f}%")
print(f"  B agrees with C: {agree_bc*100:.1f}%")
print(f"  All three agree: {all_agree*100:.1f}%")

# --- Monthly returns ---
monthly_a = res_a["equity"].resample("M").last().pct_change().dropna()
monthly_b = res_b["equity"].resample("M").last().pct_change().dropna()
monthly_c = res_c["equity"].resample("M").last().pct_change().dropna()
monthly_bh = bh_eq.resample("M").last().pct_change().dropna()

# --- Save results ---
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results = {
    "test": "proxy_comparison",
    "period": f"{START} to {END}",
    "buy_and_hold": {
        "total_return": round(float(bh_eq.iloc[-1]-1), 4),
        "cagr": round(float(bh_cagr), 4),
        "sharpe": round(float(bh_sharpe), 2),
        "max_drawdown": round(float(bh_dd), 4),
    },
    "proxy_a_4instrument": {k: v for k, v in res_a.items() if k not in ("equity", "signal")},
    "proxy_b_alden": {k: v for k, v in res_b.items() if k not in ("equity", "signal")},
    "proxy_c_m2": {k: v for k, v in res_c.items() if k not in ("equity", "signal")},
    "correlations": {
        "a_vs_b": round(float(corr_ab), 3),
        "a_vs_c": round(float(corr_ac), 3),
        "b_vs_c": round(float(corr_bc), 3),
    },
    "agreement": {
        "a_b": round(float(agree_ab), 3),
        "a_c": round(float(agree_ac), 3),
        "b_c": round(float(agree_bc), 3),
        "all_three": round(float(all_agree), 3),
    },
}

out_path = OUTPUT_DIR / "proxy_comparison_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {out_path}")

# --- Key conclusion ---
print("\n" + "=" * 60)
print("KEY FINDING")
print("=" * 60)
proxies = [("A (4-Instrument)", res_a), ("B (Alden)", res_b), ("C (M2)", res_c)]
best = max(proxies, key=lambda x: x[1]["sharpe"])
print(f"  Best Sharpe: {best[0]} = {best[1]['sharpe']:.2f}")
if best[0].startswith("A"):
    print("  ✅ Our 4-instrument proxy BEATS Alden's framework with fewer inputs!")
    print("  This is the 'unexpected result' for the patent.")
else:
    print(f"  Note: {best[0]} has the best Sharpe. Analyze further.")

# Compare A vs B specifically
print(f"\n  Proxy A Sharpe: {res_a['sharpe']:.2f} vs Proxy B Sharpe: {res_b['sharpe']:.2f}")
if res_a['sharpe'] >= res_b['sharpe']:
    print("  ✅ 4 inputs >= 3 FRED inputs. Parsimony advantage confirmed.")
    print(f"  Max DD: A={res_a['max_drawdown']*100:.1f}% vs B={res_b['max_drawdown']*100:.1f}%")
print("\nDone!")
