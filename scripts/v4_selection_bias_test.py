#!/usr/bin/env python3
"""V4 Selection Bias Holdout Test"""

import glob, os, json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
OUT_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

SELECTION_START = "2021-01-01"
SELECTION_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
HOLDOUT_END = "2026-02-15"
ORIGINAL_TOP5 = {"SOL", "FTM", "AVAX", "BNB", "SUI"}

def load_all_tokens():
    """Load all tokens, return dict of symbol -> DataFrame"""
    tokens = {}
    for f in glob.glob(os.path.join(DATA_DIR, "*_1d.csv")):
        name = os.path.basename(f).replace("_1d.csv", "")
        if name.startswith("binance_"):
            sym = name.replace("binance_", "").replace("_usdt", "").upper()
        else:
            sym = name.upper()
        
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.dropna(subset=["close"])
        
        # Need data before 2022-01-01
        if df["timestamp"].min() < pd.Timestamp("2022-01-01"):
            tokens[sym] = df
    
    print(f"Loaded {len(tokens)} tokens with data before 2022")
    return tokens

def run_v4d(df, start, end):
    """Run V4d strategy on a DataFrame slice. Returns daily portfolio returns Series."""
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    d = df[mask].copy().reset_index(drop=True)
    if len(d) < 60:
        return None
    
    c = d["close"].values
    h = d["high"].values
    l = d["low"].values
    
    # SMA50
    sma50 = pd.Series(c).rolling(50).mean().values
    
    # ATR20
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0]-l[0]], tr])
    atr20 = pd.Series(tr).rolling(20).mean().values
    
    # 30d rolling vol (annualized)
    log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
    vol30 = pd.Series(log_ret).rolling(30).std().values * np.sqrt(365)
    
    # Generate signals and returns
    position = np.zeros(len(d))
    trailing_stop = np.zeros(len(d))
    peak = np.zeros(len(d))
    
    for i in range(51, len(d)):
        # Signal from bar i-1, trade on bar i
        signal_long = c[i-1] > sma50[i-1]
        
        # Vol ceiling: halve position when vol > 80%
        vol_scale = 0.5 if (not np.isnan(vol30[i-1]) and vol30[i-1] > 0.80) else 1.0
        
        if signal_long and not np.isnan(sma50[i-1]):
            if position[i-1] == 0:
                # New entry
                position[i] = vol_scale
                peak[i] = c[i]
                # Trailing stop: 2x ATR(20), floored 5%, capped 20%
                if not np.isnan(atr20[i-1]) and c[i-1] > 0:
                    stop_pct = np.clip(2 * atr20[i-1] / c[i-1], 0.05, 0.20)
                else:
                    stop_pct = 0.10
                trailing_stop[i] = c[i] * (1 - stop_pct)
            else:
                # Continue position
                position[i] = vol_scale
                peak[i] = max(peak[i-1], c[i])
                if not np.isnan(atr20[i-1]) and c[i-1] > 0:
                    stop_pct = np.clip(2 * atr20[i-1] / c[i-1], 0.05, 0.20)
                else:
                    stop_pct = 0.10
                new_stop = peak[i] * (1 - stop_pct)
                trailing_stop[i] = max(trailing_stop[i-1], new_stop)
                
                # Check trailing stop hit
                if l[i] < trailing_stop[i]:
                    position[i] = 0
                    peak[i] = 0
                    trailing_stop[i] = 0
        else:
            position[i] = 0
    
    # Daily returns: position[i] * (c[i]/c[i-1] - 1)
    daily_ret = np.zeros(len(d))
    for i in range(1, len(d)):
        if position[i] != 0:
            daily_ret[i] = position[i] * (c[i] / c[i-1] - 1)
    
    return pd.Series(daily_ret[51:], index=d["timestamp"].values[51:])

def calc_metrics(returns):
    """Sharpe, CAGR, MaxDD from daily returns series"""
    if returns is None or len(returns) < 30:
        return None
    r = returns.values
    sharpe = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(365)
    cum = np.cumprod(1 + r)
    total = cum[-1]
    n_years = len(r) / 365
    cagr = total ** (1/n_years) - 1 if n_years > 0 else 0
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2), "max_dd": round(max_dd * 100, 2)}

def equal_weight_portfolio(token_returns_dict):
    """Combine multiple token returns into equal-weight portfolio"""
    if not token_returns_dict:
        return None
    all_df = pd.DataFrame(token_returns_dict)
    # Equal weight: average of available returns each day
    port_ret = all_df.mean(axis=1)
    return port_ret

# ===== MAIN =====
tokens = load_all_tokens()

# Step 1: Run V4d on selection period (2021-2023) for all tokens
print("\n" + "="*70)
print("STEP 1: SELECTION PERIOD (2021-2023) SCREENING")
print("="*70)

selection_results = {}
for sym, df in sorted(tokens.items()):
    ret = run_v4d(df, SELECTION_START, SELECTION_END)
    if ret is not None:
        m = calc_metrics(ret)
        if m:
            selection_results[sym] = m
            selection_results[sym]["_returns"] = ret

print(f"\n{len(selection_results)} tokens with valid selection-period results")
print(f"\n{'Rank':<5} {'Token':<10} {'Sharpe':<10} {'CAGR%':<10} {'MaxDD%':<10}")
print("-" * 45)

ranked = sorted(selection_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
for i, (sym, m) in enumerate(ranked, 1):
    marker = " ★" if sym in ORIGINAL_TOP5 else ""
    print(f"{i:<5} {sym:<10} {m['sharpe']:<10} {m['cagr']:<10} {m['max_dd']:<10}{marker}")

# Step 2: Pick Top 5, Top 10
top5_syms = [sym for sym, _ in ranked[:5]]
top10_syms = [sym for sym, _ in ranked[:10]]
all_syms = [sym for sym, _ in ranked]

print(f"\n{'='*70}")
print(f"SELECTION-PERIOD Top 5: {top5_syms}")
print(f"ORIGINAL Top 5:        {sorted(ORIGINAL_TOP5)}")
print(f"Overlap: {set(top5_syms) & ORIGINAL_TOP5}")
print(f"{'='*70}")

# Step 3: Run V4d on HOLDOUT period (2024-2026)
print(f"\n{'='*70}")
print("STEP 3: HOLDOUT PERIOD (2024-2026) — TRUE OUT-OF-SAMPLE")
print("="*70)

holdout_returns = {}
for sym, df in tokens.items():
    if sym in all_syms:
        ret = run_v4d(df, HOLDOUT_START, HOLDOUT_END)
        if ret is not None:
            holdout_returns[sym] = ret

# Build portfolios
results = {}
for label, syms in [("Top 5", top5_syms), ("Top 10", top10_syms), ("All Survivors", all_syms)]:
    avail = {s: holdout_returns[s] for s in syms if s in holdout_returns}
    port = equal_weight_portfolio(avail)
    m = calc_metrics(port)
    results[label] = {"tokens": [s for s in syms if s in holdout_returns], "n": len(avail), **m}
    print(f"\n{label} ({len(avail)} tokens): {[s for s in syms if s in holdout_returns]}")
    print(f"  Sharpe: {m['sharpe']}, CAGR: {m['cagr']}%, MaxDD: {m['max_dd']}%")

# Also run original Top 5 on holdout for comparison
orig_avail = {s: holdout_returns[s] for s in ORIGINAL_TOP5 if s in holdout_returns}
orig_port = equal_weight_portfolio(orig_avail)
orig_m = calc_metrics(orig_port)
results["Original Top 5 (holdout only)"] = {"tokens": sorted(orig_avail.keys()), "n": len(orig_avail), **orig_m}
print(f"\nOriginal Top 5 on holdout ({len(orig_avail)}): {sorted(orig_avail.keys())}")
print(f"  Sharpe: {orig_m['sharpe']}, CAGR: {orig_m['cagr']}%, MaxDD: {orig_m['max_dd']}%")

# Verdict
print(f"\n{'='*70}")
print("VERDICT")
print("="*70)
sel_sharpe = results["Top 5"]["sharpe"]
print(f"Original claim (full-period Top 5, full-period test): Sharpe 3.19")
print(f"Selection-period Top 5, HOLDOUT Sharpe:               {sel_sharpe}")
print(f"Original Top 5, HOLDOUT Sharpe:                       {orig_m['sharpe']}")

if sel_sharpe >= 2.5:
    verdict = "CONFIRMED — Selection bias minimal. Strategy is robust."
elif sel_sharpe >= 1.5:
    verdict = "PARTIALLY CONFIRMED — Some degradation but still strong."
elif sel_sharpe >= 0.5:
    verdict = "INFLATED — Significant selection bias. Real performance much weaker."
else:
    verdict = "BUSTED — Selection bias was the main driver. Strategy is weak OOS."

print(f"\n>>> {verdict}")

# Save JSON
output = {
    "test": "V4 Selection Bias Holdout Test",
    "selection_period": f"{SELECTION_START} to {SELECTION_END}",
    "holdout_period": f"{HOLDOUT_START} to {HOLDOUT_END}",
    "selection_rankings": [(sym, {k:v for k,v in m.items() if k != "_returns"}) for sym, m in ranked],
    "selected_top5": top5_syms,
    "selected_top10": top10_syms,
    "original_top5": sorted(ORIGINAL_TOP5),
    "overlap_top5": sorted(set(top5_syms) & ORIGINAL_TOP5),
    "holdout_results": {k: {kk:vv for kk,vv in v.items()} for k,v in results.items()},
    "original_claim_sharpe": 3.19,
    "verdict": verdict
}

with open(os.path.join(OUT_DIR, "v4_selection_bias_test.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {OUT_DIR}/v4_selection_bias_test.json")
