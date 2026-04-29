"""Debug the walk-forward engine."""
import pandas as pd
import numpy as np

from backend.config.data_paths import BARS_1M_V1

DATA_DIR = BARS_1M_V1

df = pd.read_csv(DATA_DIR / "btcusdt_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
df4h = df.resample("4h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum",
}).dropna(subset=["open"])
df4h["return"] = df4h["close"].pct_change()
returns = df4h["return"].values

print(f"4h bars: {len(df4h)}")
print(f"Returns: mean={np.nanmean(returns)*100:.5f}%, std={np.nanstd(returns)*100:.3f}%")
print(f"Buy & hold Sharpe: {np.nanmean(returns)/np.nanstd(returns)*np.sqrt(252*6):.3f}")

# Test 1: Always long (signal = all 1s)
always_long = np.ones(len(returns))
sig = np.roll(always_long, 1); sig[0] = 0

# Simple returns
strat_ret = returns * sig
print(f"\nAlways long (no comm):")
print(f"  Mean: {np.nanmean(strat_ret)*100:.5f}%")
print(f"  Sharpe: {np.nanmean(strat_ret)/np.nanstd(strat_ret)*np.sqrt(252*6):.3f}")

# WF version
n = len(returns)
n_folds = 7
fold_size = n // (n_folds + 1)
print(f"\nWF params: fold_size={fold_size}, total_bars={n}")

oos_all = []
for i in range(n_folds):
    ts = fold_size * (i + 2)
    te = min(ts + fold_size, n)
    fr = returns[ts:te] * sig[ts:te]
    oos_all.extend(fr.tolist())
    mean_r = np.mean(fr)
    print(f"  Fold {i}: [{ts}:{te}] ({te-ts} bars) mean_ret={mean_r*100:.5f}% dates={df4h.index[ts]} to {df4h.index[min(te-1,n-1)]}")

oos = np.array(oos_all)
print(f"\nOOS combined: {len(oos)} bars, mean={np.mean(oos)*100:.5f}%")
print(f"OOS Sharpe: {np.mean(oos)/np.std(oos)*np.sqrt(252*6):.3f}")

# Test 2: Perfect signal WITHOUT np.roll
print(f"\n{'='*50}")
print("PERFECT SIGNAL DEBUG")
perfect = np.where(returns > 0, 1, -1)
perfect[0] = 0

# Without roll
strat_no_roll = returns * perfect
print(f"No roll (cheating): mean={np.nanmean(strat_no_roll)*100:.5f}%")
print(f"  Sharpe: {np.nanmean(strat_no_roll)/np.nanstd(strat_no_roll)*np.sqrt(252*6):.3f}")

# With roll (correct — can't see future)
perfect_rolled = np.roll(perfect, 1); perfect_rolled[0] = 0
strat_rolled = returns * perfect_rolled
print(f"With roll (lagged): mean={np.nanmean(strat_rolled)*100:.5f}%")
print(f"  Sharpe: {np.nanmean(strat_rolled)/np.nanstd(strat_rolled)*np.sqrt(252*6):.3f}")

# Check: is np.roll doing what we think?
print(f"\n  returns[100:103] = {returns[100:103]}")
print(f"  perfect[100:103] = {perfect[100:103]}")
print(f"  rolled [100:103] = {perfect_rolled[100:103]}")
print(f"  product[100:103] = {strat_rolled[100:103]}")
print(f"  Note: rolled signal at bar i = perfect signal at bar i-1")
print(f"  So we're trading bar i with the signal from bar i-1's RETURN direction")
print(f"  This is autocorrelation test, NOT perfect foresight")

# Test 3: ACTUAL perfect foresight
print(f"\n{'='*50}")
print("ACTUAL PERFECT FORESIGHT (use NEXT bar's return to set THIS bar's signal)")
fwd_ret = np.roll(returns, -1); fwd_ret[-1] = 0
perfect_foresight = np.where(fwd_ret > 0, 1, -1)
strat_foresight = returns * perfect_foresight
# But wait — in WF, the roll happens INSIDE. So:
# signal[i] = sign(returns[i+1])  → after roll: sig[i] = signal[i-1] = sign(returns[i])
# That's CURRENT bar foresight, not next bar.
# To get actual foresight through the WF engine, signal[i] should = sign(returns[i])
# because after roll, sig[i] = signal[i-1], and returns[i] * sig[i] = returns[i] * signal[i-1]
# We need signal[i] = sign(returns[i+1]) so that after roll, sig[i] = sign(returns[i])

# Let's just bypass the WF to check
print(f"Direct (no WF): mean={np.nanmean(strat_foresight)*100:.5f}%")
print(f"  Sharpe: {np.nanmean(strat_foresight)/np.nanstd(strat_foresight)*np.sqrt(252*6):.3f}")

# For WF: signal[i] = sign(return[i+1]) → after roll → sig[i] = sign(return[i]) = perfect
foresight_for_wf = np.where(np.roll(returns, -1) > 0, 1, -1)
oos_f = []
for i in range(n_folds):
    ts = fold_size * (i + 2)
    te = min(ts + fold_size, n)
    sig_f = np.roll(foresight_for_wf, 1); sig_f[0] = 0
    fr = returns[ts:te] * sig_f[ts:te]
    oos_f.extend(fr.tolist())
oos_f = np.array(oos_f)
print(f"WF foresight: Sharpe={np.mean(oos_f)/np.std(oos_f)*np.sqrt(252*6):.3f}")

# Test 4: Commission impact
print(f"\n{'='*50}")
print("COMMISSION IMPACT")
# How many trades does a random signal generate?
rand_sig = np.random.choice([-1, 0, 1], size=len(returns), p=[0.3, 0.4, 0.3])
rolled = np.roll(rand_sig, 1); rolled[0] = 0
changes = np.abs(np.diff(np.concatenate([[0], rolled])))
n_changes = np.sum(changes > 0)
total_comm = np.sum(changes) * 0.0006
print(f"  Random signal: {int(n_changes)} position changes")
print(f"  Total commission: {total_comm*100:.3f}%")
print(f"  Commission per bar: {total_comm/len(returns)*100:.6f}%")
print(f"  Mean return per bar: {np.nanmean(returns)*100:.5f}%")
print(f"  Commission/return ratio: {total_comm/len(returns) / abs(np.nanmean(returns[1:])):.1f}x")
