"""
Run HSAKA with EXACT VPS walk-forward methodology to confirm results match.
Then run orderflow signals through the same engine.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
from pathlib import Path

from backend.config.data_paths import BARS_1M_V1

OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")
OF_DIR = BARS_1M_V1

def walk_forward_vps(rets, signals, n_folds=14, commission=0.0006):
    """EXACT copy of VPS walk_forward."""
    n = len(rets)
    fold_size = n // (n_folds + 1)
    fold_sharpes = []
    for fold in range(n_folds):
        oos_start = fold_size * (fold + 1)
        oos_end = fold_size * (fold + 2) if fold < n_folds - 1 else n
        sigs = signals[oos_start:oos_end]
        r = rets[oos_start:oos_end]
        trades = np.diff(sigs, prepend=sigs[0])
        costs = np.abs(trades) * commission
        sr = sigs[:-1] * r[1:] - costs[1:]
        if len(sr) > 5 and np.std(sr) > 0:
            fold_sharpes.append(np.mean(sr) / np.std(sr) * np.sqrt(365 * 6))
        else:
            fold_sharpes.append(0)
    oos_sharpe = np.mean(fold_sharpes)
    pos_folds = sum(1 for s in fold_sharpes if s > 0)
    t_stat, p_val = stats.ttest_1samp(fold_sharpes, 0)
    p_val = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    return {"oos_sharpe": round(oos_sharpe, 3), "p_value": round(p_val, 4),
            "pos_folds": pos_folds, "n_folds": n_folds,
            "fold_sharpes": [round(s, 3) for s in fold_sharpes]}

def walk_forward_local(rets, signals, n_folds=7, commission=0.0006):
    """Our local WF (from magus_orderflow.py)."""
    sig = np.roll(signals, 1); sig[0] = 0
    n = len(rets)
    fold_size = n // (n_folds + 1)
    if fold_size < 50: return None
    oos = []
    for i in range(n_folds):
        ts = fold_size * (i + 2)
        te = min(ts + fold_size, n)
        if te <= ts: break
        fr = rets[ts:te] * sig[ts:te]
        sc = np.abs(np.diff(np.concatenate([[0], sig[ts:te]])))
        fr = fr - sc * commission
        oos.extend(fr.tolist())
    if len(oos) < 100: return None
    oos = np.array(oos)
    if np.std(oos) < 1e-10: return None
    sharpe = np.mean(oos) / np.std(oos) * np.sqrt(252 * 6)
    boot = np.array([np.mean(oos[np.random.randint(0, len(oos), len(oos))]) for _ in range(3000)])
    p = np.mean(boot <= 0)
    return {"oos_sharpe": round(sharpe, 3), "p_value": round(p, 4)}

def hsaka_signal(close, high, low, order=7, sfp_lb=20, ct=2):
    highs = argrelextrema(high, np.greater_equal, order=order)[0]
    lows = argrelextrema(low, np.less_equal, order=order)[0]
    structure = np.zeros(len(close))
    events = sorted([(i, 1) for i in highs] + [(i, -1) for i in lows])
    for idx, trend in events:
        structure[idx:] = trend
    sfp = np.zeros(len(close))
    for i in range(sfp_lb, len(close)):
        prev_high = np.max(high[i-sfp_lb:i])
        prev_low = np.min(low[i-sfp_lb:i])
        if high[i] > prev_high and close[i] < prev_high:
            sfp[i] = -2
        elif low[i] < prev_low and close[i] > prev_low:
            sfp[i] = 2
    score = structure * 2 + sfp
    return np.where(score >= ct, 1, np.where(score <= -ct, -1, 0))

# ============================================================
# TEST 1: HSAKA on full OHLCV data (same as VPS)
# ============================================================
print("=" * 70)
print("TEST 1: HSAKA on BTC 4h OHLCV (full range)")
df = pd.read_csv(OHLCV_DIR / "binance_btc_usdt_4h.csv", parse_dates=["timestamp"])
rets = df["close"].pct_change().values
sig = hsaka_signal(df["close"].values, df["high"].values, df["low"].values)

print(f"  Bars: {len(df)}, Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"  Signal: long={np.mean(sig==1)*100:.1f}%, short={np.mean(sig==-1)*100:.1f}%, flat={np.mean(sig==0)*100:.1f}%")
print(f"  Trades: {np.sum(np.abs(np.diff(sig))>0)}")

vps_result = walk_forward_vps(rets, sig)
local_result = walk_forward_local(rets, sig)

print(f"\n  VPS method (14 folds, t-test):   Sharpe={vps_result['oos_sharpe']}, p={vps_result['p_value']}, pos_folds={vps_result['pos_folds']}/{vps_result['n_folds']}")
print(f"  Local method (7 folds, bootstrap): Sharpe={local_result['oos_sharpe']}, p={local_result['p_value']}")
print(f"\n  VPS fold Sharpes: {vps_result['fold_sharpes']}")

# ============================================================
# TEST 2: Controlled comparison — both methods, same folds
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: Both methods with 14 folds")
local14 = walk_forward_local(rets, sig, n_folds=14)
print(f"  VPS 14-fold:   Sharpe={vps_result['oos_sharpe']}, p={vps_result['p_value']}")
print(f"  Local 14-fold: Sharpe={local14['oos_sharpe']}, p={local14['p_value']}")

# ============================================================
# TEST 3: Always long through both engines
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: Always long (trivial signal)")
always_long = np.ones(len(rets))
vps_long = walk_forward_vps(rets, always_long)
local_long = walk_forward_local(rets, always_long)
print(f"  VPS:   Sharpe={vps_long['oos_sharpe']}, p={vps_long['p_value']}")
print(f"  Local: Sharpe={local_long['oos_sharpe']}, p={local_long['p_value']}")

# ============================================================
# TEST 4: Check if returns alignment is the issue
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: Manual return alignment check")
print(f"  VPS does: signals[i] * returns[i+1] (forward-looking return)")
print(f"  Local does: returns[i] * rolled_signal[i] = returns[i] * signal[i-1]")
print(f"  These SHOULD be the same: signal[i]*ret[i+1] == signal[i]*ret[i+1]")
print(f"  Let's verify...")

# Manual VPS-style calculation
sig_trimmed = sig[:-1]
ret_shifted = rets[1:]
vps_pnl = sig_trimmed * ret_shifted
costs = np.abs(np.diff(sig, prepend=sig[0]))[1:] * 0.0006
vps_pnl_net = vps_pnl - costs
vps_sharpe = np.mean(vps_pnl_net) / np.std(vps_pnl_net) * np.sqrt(365*6)
print(f"  VPS-style full IS Sharpe: {vps_sharpe:.3f}")

# Manual local-style calculation
rolled = np.roll(sig, 1); rolled[0] = 0
local_pnl = rets * rolled
sc = np.abs(np.diff(np.concatenate([[0], rolled])))
local_pnl_net = local_pnl - sc * 0.0006
local_sharpe = np.mean(local_pnl_net) / np.std(local_pnl_net) * np.sqrt(252*6)
print(f"  Local-style full IS Sharpe: {local_sharpe:.3f}")

# Compare the actual P&L arrays
# VPS: sig[0]*ret[1], sig[1]*ret[2], ... sig[n-2]*ret[n-1]
# Local: ret[0]*rolled[0]=0, ret[1]*sig[0], ret[2]*sig[1], ... ret[n-1]*sig[n-2]
# Local[1:] should match VPS exactly
diff = np.abs(vps_pnl[:100] - local_pnl[1:101])
print(f"  P&L difference (first 100): max={np.max(diff):.10f}")
if np.max(diff) < 1e-10:
    print(f"  ✅ P&L arrays match — it's NOT an alignment issue")
else:
    print(f"  ❌ P&L arrays DIFFER")

# ============================================================
# TEST 5: The real difference — annualization factor
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Annualization factor")
print(f"  VPS uses: sqrt(365 * 6) = {np.sqrt(365*6):.3f}")
print(f"  Local uses: sqrt(252 * 6) = {np.sqrt(252*6):.3f}")
print(f"  Ratio: {np.sqrt(365*6)/np.sqrt(252*6):.3f}")

# Recompute local with VPS annualization
local_365 = walk_forward_local(rets, sig, n_folds=14)
# Manual recompute with 365
sig_r = np.roll(sig, 1); sig_r[0] = 0
n = len(rets); fs = n // 15
oos = []
for i in range(14):
    ts = fs * (i + 2); te = min(ts + fs, n)
    if te <= ts: break
    fr = rets[ts:te] * sig_r[ts:te]
    sc = np.abs(np.diff(np.concatenate([[0], sig_r[ts:te]])))
    oos.extend((fr - sc * 0.0006).tolist())
oos = np.array(oos)
sharpe_365 = np.mean(oos) / np.std(oos) * np.sqrt(365*6)
sharpe_252 = np.mean(oos) / np.std(oos) * np.sqrt(252*6)
print(f"  Same OOS data, sqrt(365*6): Sharpe={sharpe_365:.3f}")
print(f"  Same OOS data, sqrt(252*6): Sharpe={sharpe_252:.3f}")

# ============================================================
# TEST 6: THE FOLD OFFSET DIFFERENCE
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: Fold offset (THIS IS LIKELY THE KEY)")
print(f"  VPS: fold starts at fold_size*(fold+1), so fold 0 OOS = [{fs}:{fs*2}]")
print(f"  Local: fold starts at fold_size*(i+2), so fold 0 OOS = [{fs*2}:{fs*3}]")
print(f"  VPS uses fold 0 as OOS (no IS before it!)")
print(f"  Local skips fold 0 (reserves fold 0 + fold 1 as IS)")

# Check: what does VPS fold 0 look like?
print(f"\n  VPS fold 0: bars [{fs}:{fs*2}] = {df['timestamp'].iloc[fs]} to {df['timestamp'].iloc[min(fs*2-1, len(df)-1)]}")
print(f"  VPS fold 1: bars [{fs*2}:{fs*3}] = {df['timestamp'].iloc[fs*2]} to {df['timestamp'].iloc[min(fs*3-1, len(df)-1)]}")
print(f"  Local fold 0: bars [{fs*2}:{fs*3}] (same as VPS fold 1)")

# Run VPS method but skip fold 0 (like local does)
print(f"\n  VPS with fold 0 skipped:")
fold_sharpes_no0 = vps_result['fold_sharpes'][1:]
mean_no0 = np.mean(fold_sharpes_no0)
print(f"  Mean of folds 1-13: {mean_no0:.3f}")
print(f"  vs VPS all folds: {vps_result['oos_sharpe']:.3f}")

# ============================================================
# TEST 7: Orderflow signals through CORRECT VPS engine
# ============================================================
print(f"\n{'='*70}")
print("TEST 7: Orderflow signals through VPS engine")

# Load orderflow 4h
df_of = pd.read_csv(OF_DIR / "btcusdt_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
df4h = df_of.resample("4h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "buy_vol": "sum", "sell_vol": "sum", "delta": "sum",
}).dropna(subset=["open"])
rets_of = df4h["close"].pct_change().values

# CVD trend
cvd = np.cumsum(df4h["delta"].values)
cvd_sma = pd.Series(cvd).rolling(20).mean().values
cvd_prev = np.roll(cvd, 1); cvd_prev[0] = 0
cvd_sig = np.zeros(len(cvd))
cvd_sig[(cvd > cvd_sma) & (cvd > cvd_prev)] = 1
cvd_sig[(cvd < cvd_sma) & (cvd < cvd_prev)] = -1
cvd_sig[:20] = 0

# Volume imbalance
buy_pct = df4h["buy_vol"].values / np.clip(df4h["volume"].values, 1, None)
bpct_sma = pd.Series(buy_pct).rolling(10).mean().values
imb_sig = np.zeros(len(buy_pct))
imb_sig[bpct_sma > 0.55] = 1
imb_sig[bpct_sma < 0.45] = -1
imb_sig[:10] = 0

for name, signal in [("CVD trend", cvd_sig), ("Vol imbalance", imb_sig)]:
    vps_r = walk_forward_vps(rets_of, signal)
    print(f"  {name}: Sharpe={vps_r['oos_sharpe']:.3f}, p={vps_r['p_value']:.4f}, pos_folds={vps_r['pos_folds']}/14")

# HSAKA on orderflow data through VPS engine
hsaka_of = hsaka_signal(df4h["close"].values, df4h["high"].values, df4h["low"].values)
vps_hsaka = walk_forward_vps(rets_of, hsaka_of)
print(f"  HSAKA Lean: Sharpe={vps_hsaka['oos_sharpe']:.3f}, p={vps_hsaka['p_value']:.4f}, pos_folds={vps_hsaka['pos_folds']}/14")

# HSAKA on OHLCV data through VPS engine (should be ~2.0)
print(f"\n  Reference — HSAKA on original OHLCV:")
vps_hsaka_ohlcv = walk_forward_vps(rets, sig)
print(f"  HSAKA (full OHLCV): Sharpe={vps_hsaka_ohlcv['oos_sharpe']:.3f}, p={vps_hsaka_ohlcv['p_value']:.4f}")
print(f"  Folds: {vps_hsaka_ohlcv['fold_sharpes']}")
