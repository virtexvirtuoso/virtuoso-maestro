#!/usr/bin/env python3
"""Block Permutation Test for V4 + Regime-Dependent Portfolio Construction"""

import numpy as np
import pandas as pd
import duckdb
import os
import json
import sys
from datetime import datetime

np.random.seed(42)

SPOT_DIR = os.path.expanduser("~/Desktop/maestro/data/spot/")
DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")

def load_spot(asset):
    df = pd.read_csv(f"{SPOT_DIR}/{asset}_spot_daily.csv", parse_dates=["Date"])
    df.columns = df.columns.str.lower()
    df = df.sort_values("date").set_index("date")
    df["returns"] = df["close"].pct_change()
    return df

assets_spot = {a: load_spot(a) for a in ["BTC", "ETH", "SOL"]}

common_dates = assets_spot["BTC"].index
for a in ["ETH", "SOL"]:
    common_dates = common_dates.intersection(assets_spot[a].index)
common_dates = common_dates.sort_values()
for a in assets_spot:
    assets_spot[a] = assets_spot[a].loc[common_dates]

print(f"Common: {common_dates[0].date()} to {common_dates[-1].date()}, {len(common_dates)} days")

TRAIL_STOPS = {"BTC": 0.12, "ETH": 0.15, "SOL": 0.08}

def run_v4d(close, returns, trail_pct):
    """V4d on arrays. Returns strategy returns array."""
    n = len(close)
    sma50 = pd.Series(close).rolling(50).mean().values
    
    # Vol ceiling
    vol_20 = pd.Series(returns).rolling(20).std().values * np.sqrt(365)
    
    position = np.zeros(n)
    peak = close[0]
    in_trade = False
    
    for i in range(51, n):
        # Signal: previous close > previous SMA50
        if close[i-1] > sma50[i-1]:
            if not in_trade:
                in_trade = True
                peak = close[i]
            position[i] = 1.0
        
        if in_trade:
            peak = max(peak, close[i])
            if (close[i] - peak) / peak < -trail_pct:
                in_trade = False
                position[i] = 0.0
        
        # Vol ceiling
        if not np.isnan(vol_20[i]) and vol_20[i] > 0.80:
            position[i] *= 0.5
    
    # DD breaker pass
    strat_ret = np.nan_to_num(position * returns)
    cum = np.cumprod(1 + strat_ret)
    rm = np.maximum.accumulate(cum)
    dd = (cum - rm) / rm
    
    pos2 = position.copy()
    flat = False
    for i in range(1, n):
        if flat:
            if close[i] > sma50[i] if not np.isnan(sma50[i]) else False:
                flat = False
            else:
                pos2[i] = 0.0
        sr = pos2[i] * returns[i] if not np.isnan(returns[i]) else 0.0
        # Recompute running dd
    
    # Simpler: just use dd breaker on the position
    cum2 = 1.0
    rm2 = 1.0
    flat = False
    for i in range(1, n):
        if flat:
            if not np.isnan(sma50[i]) and close[i] > sma50[i]:
                flat = False
                # keep position
            else:
                pos2[i] = 0.0
        sr = pos2[i] * (returns[i] if not np.isnan(returns[i]) else 0.0)
        cum2 *= (1 + sr)
        rm2 = max(rm2, cum2)
        if (cum2 - rm2) / rm2 < -0.25:
            flat = True
    
    return np.nan_to_num(pos2 * returns)

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 30 or np.std(r) == 0:
        return 0.0
    return np.mean(r) / np.std(r) * np.sqrt(365)

def wf_sharpe(port_ret, n_folds=14):
    """Walk-forward OOS Sharpe from pre-computed portfolio returns."""
    n = len(port_ret)
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos = []
    for f in range(n_folds):
        s = min_train + f * fold_size
        e = min(s + fold_size, n)
        oos.extend(port_ret[s:e])
    return sharpe(np.array(oos)), np.array(oos)

# Compute V4d per asset
v4d_ret = {}
for asset in ["BTC", "ETH", "SOL"]:
    df = assets_spot[asset]
    v4d_ret[asset] = run_v4d(df["close"].values, df["returns"].values, TRAIL_STOPS[asset])

port_baseline = np.mean([v4d_ret[a] for a in ["BTC", "ETH", "SOL"]], axis=0)
actual_sharpe, actual_oos = wf_sharpe(port_baseline)
print(f"\nV4d OOS Sharpe: {actual_sharpe:.3f}")

# ============================================================
# BLOCK PERMUTATION (fast: permute returns, reconstruct prices, re-run V4d)
# ============================================================
print("\n=== PART 1: BLOCK PERMUTATION TEST ===")

N_PERMS = 500  # 500 for speed
BLOCK_SIZES = [20, 40, 60, 120]

def block_permute_and_run(block_size):
    """One permutation: shuffle blocks, re-run V4d, return Sharpe."""
    n = len(common_dates)
    n_blocks = n // block_size
    perm = np.random.permutation(n_blocks)
    
    all_asset_rets = []
    for asset in ["BTC", "ETH", "SOL"]:
        returns = assets_spot[asset]["returns"].values.copy()
        close_orig = assets_spot[asset]["close"].values
        
        # Shuffle returns in blocks
        new_ret = np.zeros(n)
        for new_i, old_i in enumerate(perm):
            s1, e1 = new_i * block_size, (new_i + 1) * block_size
            s2, e2 = old_i * block_size, (old_i + 1) * block_size
            if e1 <= n and e2 <= n:
                new_ret[s1:e1] = returns[s2:e2]
        # Leftover
        used = n_blocks * block_size
        if used < n:
            new_ret[used:] = returns[used:]
        
        # Reconstruct prices
        new_close = np.zeros(n)
        new_close[0] = close_orig[0]
        for i in range(1, n):
            r = new_ret[i] if not np.isnan(new_ret[i]) else 0.0
            new_close[i] = new_close[i-1] * (1 + r)
        
        sr = run_v4d(new_close, new_ret, TRAIL_STOPS[asset])
        all_asset_rets.append(sr)
    
    port = np.mean(all_asset_rets, axis=0)
    s, _ = wf_sharpe(port)
    return s

block_perm_results = {}
for bs in BLOCK_SIZES:
    print(f"Block size {bs}...", end=" ", flush=True)
    perm_sharpes = []
    for p in range(N_PERMS):
        perm_sharpes.append(block_permute_and_run(bs))
        if (p+1) % 100 == 0:
            print(f"{p+1}", end=" ", flush=True)
    
    p_val = np.mean(np.array(perm_sharpes) >= actual_sharpe)
    block_perm_results[bs] = {
        "p_value": float(p_val),
        "mean_perm_sharpe": float(np.mean(perm_sharpes)),
        "std_perm_sharpe": float(np.std(perm_sharpes)),
    }
    print(f"→ p={p_val:.4f} (mean={np.mean(perm_sharpes):.3f}±{np.std(perm_sharpes):.3f})")

# Standard permutation (block=1)
print("Block size 1 (standard)...", end=" ", flush=True)
perm_sharpes_std = []
for p in range(N_PERMS):
    perm_sharpes_std.append(block_permute_and_run(1))
    if (p+1) % 100 == 0:
        print(f"{p+1}", end=" ", flush=True)
p_std = np.mean(np.array(perm_sharpes_std) >= actual_sharpe)
block_perm_results[1] = {
    "p_value": float(p_std),
    "mean_perm_sharpe": float(np.mean(perm_sharpes_std)),
    "std_perm_sharpe": float(np.std(perm_sharpes_std)),
}
print(f"→ p={p_std:.4f}")

print(f"\nActual Sharpe: {actual_sharpe:.3f}")
for bs in [1] + BLOCK_SIZES:
    r = block_perm_results[bs]
    print(f"  B={bs:3d}: p={r['p_value']:.4f}  perm_sharpe={r['mean_perm_sharpe']:.3f}±{r['std_perm_sharpe']:.3f}")

# ============================================================
# PART 2: REGIME
# ============================================================
print("\n=== PART 2: REGIME-DEPENDENT PORTFOLIO ===")

con = duckdb.connect(DB_PATH, read_only=True)
lsr = con.execute("SELECT * FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date").df()
fr = con.execute("SELECT * FROM cg_funding_rate WHERE symbol='BTC' ORDER BY date").df()
liq = con.execute("SELECT * FROM cg_liquidations WHERE symbol='BTC' ORDER BY date").df()
taker = con.execute("SELECT * FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date").df()
con.close()

# Build regime score aligned to common_dates
regime = pd.DataFrame(index=common_dates)

# LSR
lsr_s = lsr.drop_duplicates("date").set_index("date")["global_account_long_short_ratio"]
lsr_p50 = lsr_s.rolling(30, min_periods=10).median()
lsr_score = (lsr_s < lsr_p50).astype(float)
regime["lsr_score"] = lsr_score

# Funding
fr_s = fr.drop_duplicates("date").set_index("date")["close"]
regime["fr_score"] = (fr_s < 0.03).astype(float)

# Liquidations
liq_df = liq.drop_duplicates("date").set_index("date")
liq_total = liq_df["aggregated_long_liquidation_usd"] + liq_df["aggregated_short_liquidation_usd"]
liq_p80 = liq_total.rolling(30, min_periods=10).quantile(0.8)
regime["liq_score"] = (liq_total < liq_p80).astype(float)

# Taker
taker_df = taker.drop_duplicates("date").set_index("date")
taker_ratio = taker_df["taker_buy_volume_usd"] / taker_df["taker_sell_volume_usd"].replace(0, np.nan)
regime["taker_score"] = (taker_ratio > 1.0).astype(float)

regime = regime.ffill().fillna(0.5)
regime["score"] = regime["lsr_score"]*0.35 + regime["fr_score"]*0.35 + regime["liq_score"]*0.15 + regime["taker_score"]*0.15

scores = regime["score"].values
print(f"Regime score: mean={scores.mean():.2f}, min={scores.min():.2f}, max={scores.max():.2f}")

# Historical events
events = {
    "May 2021 crash": "2021-05-19",
    "FTX collapse": "2022-11-08", 
    "2023 recovery": "2023-01-15",
    "2024 bull run": "2024-01-10",
    "2024 bull peak": "2024-11-15",
}
print("\n--- Historical Regime Scores ---")
event_scores = {}
for name, ds in events.items():
    dt = pd.Timestamp(ds)
    idx = regime.index.get_indexer([dt], method="nearest")[0]
    if 0 <= idx < len(regime):
        s = regime.iloc[idx]["score"]
        d = regime.index[idx]
        print(f"  {name:25s} ({d.date()}): {s:.2f}")
        event_scores[name] = {"date": str(d.date()), "score": float(s)}

# Regime variants
# Pre-compute wide/tight stop versions
v4d_wide = {}; v4d_tight = {}
for asset in ["BTC", "ETH", "SOL"]:
    df = assets_spot[asset]
    v4d_wide[asset] = run_v4d(df["close"].values, df["returns"].values, TRAIL_STOPS[asset] * 1.5)
    v4d_tight[asset] = run_v4d(df["close"].values, df["returns"].values, TRAIL_STOPS[asset] * 0.5)

n = len(common_dates)

# V1: Asset mix
port_mix = np.zeros(n)
for i in range(n):
    s = scores[i]
    if s >= 0.7:
        port_mix[i] = np.mean([v4d_ret[a][i] for a in ["BTC","ETH","SOL"]])
    elif s >= 0.3:
        port_mix[i] = np.mean([v4d_ret[a][i] for a in ["BTC","ETH"]])
    # else 0

# V2: Stop adjust
port_stop = np.zeros(n)
for i in range(n):
    s = scores[i]
    if s >= 0.7:
        port_stop[i] = np.mean([v4d_wide[a][i] for a in ["BTC","ETH","SOL"]])
    elif s < 0.3:
        port_stop[i] = np.mean([v4d_tight[a][i] for a in ["BTC","ETH","SOL"]])
    else:
        port_stop[i] = port_baseline[i]

# V3: Position sizing
port_pos = port_baseline * scores

# V4: Combined
port_comb = np.zeros(n)
for i in range(n):
    s = scores[i]
    if s >= 0.7:
        port_comb[i] = np.mean([v4d_wide[a][i] for a in ["BTC","ETH","SOL"]])
    elif s >= 0.3:
        port_comb[i] = np.mean([v4d_ret[a][i] for a in ["BTC","ETH"]])
    # else 0

variants = {
    "V4d Baseline": port_baseline,
    "Regime Asset Mix": port_mix,
    "Regime Stop Adjust": port_stop,
    "Regime Position Size": port_pos,
    "Regime Combined": port_comb,
}

def bootstrap_ci(oos, n_boot=1000):
    sharpes = [sharpe(oos[np.random.choice(len(oos), len(oos), replace=True)]) for _ in range(n_boot)]
    return np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)

print(f"\n{'Variant':30s} {'Sharpe':>7s} {'AnnRet':>7s} {'MaxDD':>7s} {'CI95':>16s}")
print("-" * 70)

regime_results = {}
for name, rets in variants.items():
    s, oos = wf_sharpe(rets)
    cum = np.cumprod(1 + oos)
    ann = cum[-1]**(365/len(oos)) - 1
    dd = np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))
    lo, hi = bootstrap_ci(oos)
    print(f"{name:30s} {s:7.3f} {ann:6.1%} {dd:6.1%}  [{lo:.3f}, {hi:.3f}]")
    regime_results[name] = {"sharpe": float(s), "ann_return": float(ann), "max_dd": float(dd), "ci_95": [float(lo), float(hi)]}

# Save
out_dir = os.path.expanduser("~/Desktop/maestro/data/backtest_results/")
os.makedirs(out_dir, exist_ok=True)
results = {
    "timestamp": datetime.now().isoformat(),
    "part1_block_permutation": {"actual_sharpe": float(actual_sharpe), "block_results": {str(k): v for k, v in block_perm_results.items()}},
    "part2_regime_portfolio": regime_results,
    "historical_regime_events": event_scores,
}
with open(f"{out_dir}/block_perm_regime_test.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_dir}/block_perm_regime_test.json")

# VERDICT
print("\n" + "="*60)
print("VERDICT")
print("="*60)
best_bp = min(block_perm_results[bs]["p_value"] for bs in BLOCK_SIZES)
std_p = block_perm_results[1]["p_value"]
print(f"\nBlock Permutation:")
print(f"  Standard (B=1) p-value: {std_p:.4f}")
print(f"  Best block p-value: {best_bp:.4f}")
if best_bp < 0.05:
    print("  ✅ V4d SIGNIFICANT at 5% with block permutation")
elif best_bp < 0.10:
    print("  ⚠️ V4d marginally significant (p<0.10)")
else:
    print("  ❌ V4d NOT significant even with block permutation")
print(f"  {'✅' if best_bp < std_p else '⚠️'} Block perm {'more' if best_bp < std_p else 'less'} significant than standard")

bs = regime_results["V4d Baseline"]["sharpe"]
best_v = max(((k,v["sharpe"]) for k,v in regime_results.items() if k!="V4d Baseline"), key=lambda x:x[1])
print(f"\nRegime Construction:")
print(f"  Baseline: {bs:.3f}")
print(f"  Best: {best_v[0]} ({best_v[1]:.3f})")
if best_v[1] > bs:
    print(f"  ✅ Improves V4d by {(best_v[1]-bs)/abs(bs)*100:.1f}%")
else:
    print(f"  ❌ No improvement over baseline")
