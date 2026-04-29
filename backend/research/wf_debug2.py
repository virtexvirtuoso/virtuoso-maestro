"""Debug: why HSAKA works on VPS OHLCV but not on orderflow-derived OHLCV."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import argrelextrema

from backend.config.data_paths import BARS_1M_V1

OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")
OF_DIR = BARS_1M_V1

def hsaka_signal(close, high, low, order=7, sfp_lb=20, ct=2):
    """HSAKA Lean: structure + SFP."""
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

def walk_forward(returns, signal, comm=0.0006, ann_factor=252*6, n_folds=7):
    sig = np.roll(signal, 1); sig[0] = 0
    n = len(returns)
    fold_size = n // (n_folds + 1)
    if fold_size < 50: return None
    oos = []
    for i in range(n_folds):
        ts = fold_size * (i + 2)
        te = min(ts + fold_size, n)
        if te <= ts: break
        fr = returns[ts:te] * sig[ts:te]
        sc = np.abs(np.diff(np.concatenate([[0], sig[ts:te]])))
        fr = fr - sc * comm
        oos.extend(fr.tolist())
    if len(oos) < 100: return None
    oos = np.array(oos)
    if np.std(oos) < 1e-10: return None
    sharpe = np.mean(oos) / np.std(oos) * np.sqrt(ann_factor)
    boot = np.array([np.mean(oos[np.random.randint(0, len(oos), len(oos))]) for _ in range(3000)])
    p = np.mean(boot <= 0)
    invested = np.mean(np.abs(sig)) * 100
    n_trades = int(np.sum(np.abs(np.diff(sig)) > 0))
    return {"sharpe": round(sharpe, 3), "p": round(p, 4), "inv": round(invested,1), "trades": n_trades}

# Source 1: OHLCV file (what VPS uses)
print("=== SOURCE 1: OHLCV FILE ===")
df1 = pd.read_csv(OHLCV_DIR / "binance_btc_usdt_4h.csv", parse_dates=["timestamp"])
df1["return"] = df1["close"].pct_change()
print(f"Bars: {len(df1)}, Range: {df1['timestamp'].iloc[0]} → {df1['timestamp'].iloc[-1]}")
print(f"Returns: mean={df1['return'].mean()*100:.5f}%, std={df1['return'].std()*100:.3f}%")

sig1 = hsaka_signal(df1["close"].values, df1["high"].values, df1["low"].values)
wf1 = walk_forward(df1["return"].values, sig1)
print(f"HSAKA: {wf1}")

# Source 2: Orderflow-derived
print(f"\n=== SOURCE 2: ORDERFLOW-DERIVED ===")
df2 = pd.read_csv(OF_DIR / "btcusdt_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
df2_4h = df2.resample("4h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna(subset=["open"])
df2_4h["return"] = df2_4h["close"].pct_change()
print(f"Bars: {len(df2_4h)}, Range: {df2_4h.index[0]} → {df2_4h.index[-1]}")
print(f"Returns: mean={df2_4h['return'].mean()*100:.5f}%, std={df2_4h['return'].std()*100:.3f}%")

sig2 = hsaka_signal(df2_4h["close"].values, df2_4h["high"].values, df2_4h["low"].values)
wf2 = walk_forward(df2_4h["return"].values, sig2)
print(f"HSAKA: {wf2}")

# Compare signals
print(f"\n=== SIGNAL COMPARISON ===")
# Align dates
common_start = max(df1["timestamp"].iloc[0], df2_4h.index[0])
common_end = min(df1["timestamp"].iloc[-1], df2_4h.index[-1])
print(f"Common range: {common_start} → {common_end}")

mask1 = (df1["timestamp"] >= common_start) & (df1["timestamp"] <= common_end)
d1 = df1[mask1].reset_index(drop=True)
d2 = df2_4h[(df2_4h.index >= common_start) & (df2_4h.index <= common_end)].reset_index()

print(f"  OHLCV bars in range: {len(d1)}")
print(f"  Orderflow bars in range: {len(d2)}")

# Check close price correlation
if len(d1) > 0 and len(d2) > 0:
    # Merge on timestamp
    d1_ts = d1.set_index("timestamp")["close"]
    d2_ts = d2.set_index("timestamp")["close"]
    common = d1_ts.index.intersection(d2_ts.index)
    print(f"  Matching timestamps: {len(common)}")
    if len(common) > 10:
        corr = np.corrcoef(d1_ts[common].values, d2_ts[common].values)[0,1]
        diff = (d1_ts[common].values - d2_ts[common].values)
        print(f"  Close price corr: {corr:.6f}")
        print(f"  Close price diff: mean={np.mean(diff):.2f}, max={np.max(np.abs(diff)):.2f}")
        print(f"  Close price diff %: mean={np.mean(np.abs(diff)/d1_ts[common].values)*100:.4f}%")

# Run HSAKA on OHLCV data trimmed to same range as orderflow
print(f"\n=== HSAKA ON SAME DATE RANGE ===")
d1_trim = df1[mask1]
sig1_trim = hsaka_signal(d1_trim["close"].values, d1_trim["high"].values, d1_trim["low"].values)
wf1_trim = walk_forward(d1_trim["return"].values, sig1_trim)
print(f"OHLCV (trimmed to 2024-2026): {wf1_trim}")
print(f"Orderflow (2024-2026): {wf2}")

# Signal stats
print(f"\n=== SIGNAL STATS ===")
print(f"OHLCV full: trades={np.sum(np.abs(np.diff(sig1))>0)}, invested={np.mean(np.abs(sig1))*100:.1f}%")
print(f"OHLCV trim: trades={np.sum(np.abs(np.diff(sig1_trim))>0)}, invested={np.mean(np.abs(sig1_trim))*100:.1f}%")
print(f"Orderflow:  trades={np.sum(np.abs(np.diff(sig2))>0)}, invested={np.mean(np.abs(sig2))*100:.1f}%")
