"""
Magus Orderflow Strategy — Built from tick-level aggTrades data.

Signals derived from ACTUAL orderflow (buy/sell volume split):
1. CVD Trend: Cumulative Volume Delta direction (net aggressive buying/selling)
2. Delta Divergence: Price makes new high but CVD doesn't (or vice versa) — trapped traders
3. Absorption: High aggressive selling but price holds = passive bid wall (and vice versa)
4. Puke Detection: Sudden spike in one-sided aggressive volume = forced liquidation
5. Volume Imbalance: Buy/sell ratio extreme at key levels

Test on 5m, 15m, 1h, 4h timeframes.
Walk-forward 7 folds, bootstrap p-values.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from backend.config.data_paths import BARS_1M_V1

DATA_DIR = BARS_1M_V1
OUT_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")

ASSETS = ["btcusdt", "ethusdt"]
TIMEFRAMES = ["15min", "1h", "4h"]
COMMISSIONS = {"5min": 0.0008, "15min": 0.0004, "1h": 0.0006, "4h": 0.0006}
ANN_FACTORS = {"5min": 252*24*12, "15min": 252*24*4, "1h": 252*24, "4h": 252*6}


def load_and_resample(asset, tf):
    """Load 1m orderflow data and resample to target timeframe."""
    path = DATA_DIR / f"{asset}_1m.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    
    if tf == "1min":
        return df
    
    resampled = df.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "dollar_volume": "sum",
        "buy_vol": "sum",
        "sell_vol": "sum",
        "buy_dollar": "sum",
        "sell_dollar": "sum",
        "trade_count": "sum",
        "delta": "sum",
        "delta_dollar": "sum",
        "buy_pct": "mean",
    })
    resampled = resampled.dropna(subset=["open"])
    return resampled


def cvd_trend(delta, lookback=20):
    """Component 1: CVD direction — cumulative delta trend.
    Rising CVD = aggressive buyers dominating = bullish. Vectorized.
    """
    cvd = np.cumsum(delta)
    cvd_sma = pd.Series(cvd).rolling(lookback).mean().values
    cvd_prev = np.roll(cvd, 1); cvd_prev[0] = 0
    
    sig = np.zeros(len(delta))
    sig[(cvd > cvd_sma) & (cvd > cvd_prev)] = 1
    sig[(cvd < cvd_sma) & (cvd < cvd_prev)] = -1
    sig[:lookback] = 0
    return sig


def delta_divergence(close, delta, lookback=20):
    """Component 2: Price/CVD divergence — trapped traders. Vectorized."""
    cvd = np.cumsum(delta)
    cs = pd.Series(close)
    cvds = pd.Series(cvd)
    
    price_max = cs.rolling(lookback).max().values
    price_min = cs.rolling(lookback).min().values
    cvd_mean = cvds.rolling(lookback).mean().values
    
    sig = np.zeros(len(close))
    # Bearish div: price at rolling high, CVD below its mean
    bear = (close >= price_max * 0.999) & (cvd < cvd_mean)
    # Bullish div: price at rolling low, CVD above its mean
    bull = (close <= price_min * 1.001) & (cvd > cvd_mean)
    
    sig[bear] = -1
    sig[bull] = 1
    sig[:lookback] = 0
    return sig


def absorption(close, buy_vol, sell_vol, high, low, lookback=10):
    """Component 3: Absorption detection. Vectorized.
    High sell volume but price doesn't drop = passive bid absorption = bullish.
    High buy volume but price doesn't rise = passive ask absorption = bearish.
    """
    total_vol = buy_vol + sell_vol
    sell_pct = sell_vol / np.clip(total_vol, 1, None)
    buy_pct = buy_vol / np.clip(total_vol, 1, None)
    
    atr = pd.Series(high - low).rolling(lookback).mean().values
    price_change = np.diff(close, prepend=close[0])
    norm_change = np.abs(price_change) / np.clip(atr, 1e-10, None)
    
    sell_pct_sma = pd.Series(sell_pct).rolling(lookback).mean().values
    buy_pct_sma = pd.Series(buy_pct).rolling(lookback).mean().values
    
    sig = np.zeros(len(close))
    # Bullish absorption: heavy selling, price holds
    bull = (sell_pct > 0.60) & (sell_pct > sell_pct_sma * 1.2) & ((norm_change < 0.5) | (price_change > 0))
    # Bearish absorption: heavy buying, price holds
    bear = (buy_pct > 0.60) & (buy_pct > buy_pct_sma * 1.2) & ((norm_change < 0.5) | (price_change < 0))
    
    sig[bull] = 1
    sig[bear] = -1
    sig[:lookback] = 0
    return sig


def puke_detection(delta, volume, lookback=20):
    """Component 4: Puke detection — sudden aggressive liquidation flow.
    Massive one-sided delta spike = forced liquidation = FADE it.
    Vectorized z-score computation.
    """
    ds = pd.Series(delta)
    vs = pd.Series(volume)
    
    d_mean = ds.rolling(lookback).mean().values
    d_std = ds.rolling(lookback).std().values
    v_mean = vs.rolling(lookback).mean().values
    v_std = vs.rolling(lookback).std().values
    
    delta_z = (delta - d_mean) / np.clip(d_std, 1e-10, None)
    vol_z = (volume - v_mean) / np.clip(v_std, 1e-10, None)
    
    sig = np.zeros(len(delta))
    # Massive sell puke → fade = go long
    sig[(delta_z < -2.0) & (vol_z > 1.5)] = 1
    # Massive buy puke → fade = go short
    sig[(delta_z > 2.0) & (vol_z > 1.5)] = -1
    sig[:lookback] = 0
    
    return sig


def volume_imbalance(buy_vol, sell_vol, lookback=10):
    """Component 5: Sustained volume imbalance. Vectorized."""
    buy_pct = buy_vol / np.clip(buy_vol + sell_vol, 1, None)
    buy_pct_sma = pd.Series(buy_pct).rolling(lookback).mean().values
    
    sig = np.zeros(len(buy_vol))
    sig[buy_pct_sma > 0.55] = 1
    sig[buy_pct_sma < 0.45] = -1
    sig[:lookback] = 0
    return sig


def generate_signal(df, cvd_lb=20, div_lb=20, abs_lb=10, puke_lb=20, imb_lb=10, ct=2):
    """Generate full Magus orderflow signal."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    d = df["delta"].values
    v = df["volume"].values
    bv = df["buy_vol"].values
    sv = df["sell_vol"].values
    
    s1 = cvd_trend(d, cvd_lb)
    s2 = delta_divergence(c, d, div_lb)
    s3 = absorption(c, bv, sv, h, l, abs_lb)
    s4 = puke_detection(d, v, puke_lb)
    s5 = volume_imbalance(bv, sv, imb_lb)
    
    score = s1 + s2 + s3 + s4 + s5
    signal = np.where(score >= ct, 1, np.where(score <= -ct, -1, 0))
    
    components = {
        "cvd_trend": s1, "delta_div": s2, "absorption": s3,
        "puke": s4, "vol_imbalance": s5
    }
    return signal, score, components


def walk_forward(returns, signal, comm, ann_factor, n_folds=7):
    """Walk-forward with bootstrap p-value."""
    sig = np.roll(signal, 1)
    sig[0] = 0
    
    n = len(returns)
    fold_size = n // (n_folds + 1)
    if fold_size < 50:
        return None
    
    oos = []
    for i in range(n_folds):
        ts = fold_size * (i + 2)
        te = min(ts + fold_size, n)
        if te <= ts:
            break
        fr = returns[ts:te] * sig[ts:te]
        sc = np.abs(np.diff(np.concatenate([[0], sig[ts:te]])))
        fr = fr - sc * comm
        oos.extend(fr.tolist())
    
    if len(oos) < 100:
        return None
    
    oos = np.array(oos)
    if np.std(oos) < 1e-10:
        return None
    
    sharpe = np.mean(oos) / np.std(oos) * np.sqrt(ann_factor)
    
    boot = np.array([
        np.mean(oos[np.random.randint(0, len(oos), len(oos))])
        for _ in range(5000)
    ])
    p = np.mean(boot <= 0)
    
    cum = np.cumprod(1 + oos)
    peak = np.maximum.accumulate(cum)
    max_dd = np.min((cum - peak) / peak)
    invested = np.mean(np.abs(sig)) * 100
    n_trades = int(np.sum(np.abs(np.diff(sig)) > 0))
    
    return {
        "oos_sharpe": round(sharpe, 3),
        "p_value": round(p, 4),
        "max_dd": round(max_dd * 100, 1),
        "invested_pct": round(invested, 1),
        "n_trades": n_trades,
    }


def main():
    results = []
    
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            comm = COMMISSIONS[tf]
            ann = ANN_FACTORS[tf]
            
            print(f"\n{'='*60}")
            print(f"=== {asset.upper()} {tf} (comm={comm*10000:.0f}bps) ===")
            
            df = load_and_resample(asset, tf)
            df["return"] = df["close"].pct_change()
            returns = df["return"].values
            print(f"  {len(df):,} bars")
            
            # Component fire rates at default params
            signal, score, comps = generate_signal(df)
            rates = " | ".join(
                f"{k}={np.mean(np.abs(v))*100:.0f}%"
                for k, v in comps.items()
            )
            print(f"  Fire rates: {rates}")
            print(f"  Score dist: " + " ".join(
                f"{s}:{np.mean(score==s)*100:.0f}%"
                for s in range(-5, 6)
            ))
            
            best = None
            # Parameter grid
            for cvd_lb in [10, 20, 40]:
                for puke_lb in [10, 20]:
                    for ct in [1, 2, 3]:
                        sig, _, _ = generate_signal(
                            df, cvd_lb=cvd_lb, puke_lb=puke_lb, ct=ct
                        )
                        wf = walk_forward(returns, sig, comm, ann)
                        if wf is None:
                            continue
                        
                        r = {
                            "asset": asset, "tf": tf,
                            "cvd_lb": cvd_lb, "puke_lb": puke_lb, "ct": ct,
                            **wf
                        }
                        results.append(r)
                        
                        if wf["p_value"] < 0.05:
                            if best is None or wf["oos_sharpe"] > best["oos_sharpe"]:
                                best = r
                            print(f"  ✅ cvd={cvd_lb} puke={puke_lb} ct={ct}: "
                                  f"OOS {wf['oos_sharpe']:.3f} p={wf['p_value']:.4f} "
                                  f"dd={wf['max_dd']}% inv={wf['invested_pct']}% "
                                  f"trades={wf['n_trades']}")
            
            if best:
                print(f"  🏆 BEST: OOS {best['oos_sharpe']:.3f} p={best['p_value']:.4f}")
            else:
                print(f"  ❌ No significant results")
    
    # Ablation on best asset/tf combo
    print(f"\n{'='*60}")
    print("=== ABLATION (BTC 4h, default params) ===")
    df = load_and_resample("btcusdt", "4h")
    df["return"] = df["close"].pct_change()
    returns = df["return"].values
    
    signal, score, comps = generate_signal(df, ct=2)
    base_wf = walk_forward(returns, signal, COMMISSIONS["4h"], ANN_FACTORS["4h"])
    if base_wf:
        print(f"  Full (5 comp): OOS {base_wf['oos_sharpe']:.3f} p={base_wf['p_value']:.4f}")
    
    comp_names = list(comps.keys())
    for remove in comp_names:
        reduced_score = sum(v for k, v in comps.items() if k != remove)
        reduced_sig = np.where(reduced_score >= 2, 1, np.where(reduced_score <= -2, -1, 0))
        wf = walk_forward(returns, reduced_sig, COMMISSIONS["4h"], ANN_FACTORS["4h"])
        if wf and base_wf:
            delta = wf["oos_sharpe"] - base_wf["oos_sharpe"]
            print(f"  Remove {remove:<14s}: OOS {wf['oos_sharpe']:.3f} p={wf['p_value']:.4f} Δ{delta:+.3f}")
        elif wf:
            print(f"  Remove {remove:<14s}: OOS {wf['oos_sharpe']:.3f} p={wf['p_value']:.4f}")
    
    # Also do ablation on 15m
    print(f"\n=== ABLATION (BTC 15m, default params) ===")
    df = load_and_resample("btcusdt", "15min")
    df["return"] = df["close"].pct_change()
    returns = df["return"].values
    
    signal, score, comps = generate_signal(df, ct=2)
    base_wf = walk_forward(returns, signal, COMMISSIONS["15min"], ANN_FACTORS["15min"])
    if base_wf:
        print(f"  Full (5 comp): OOS {base_wf['oos_sharpe']:.3f} p={base_wf['p_value']:.4f}")
    
    for remove in comp_names:
        reduced_score = sum(v for k, v in comps.items() if k != remove)
        reduced_sig = np.where(reduced_score >= 2, 1, np.where(reduced_score <= -2, -1, 0))
        wf = walk_forward(returns, reduced_sig, COMMISSIONS["15min"], ANN_FACTORS["15min"])
        if wf and base_wf:
            delta = wf["oos_sharpe"] - base_wf["oos_sharpe"]
            print(f"  Remove {remove:<14s}: OOS {wf['oos_sharpe']:.3f} p={wf['p_value']:.4f} Δ{delta:+.3f}")
    
    # Save
    out = OUT_DIR / "magus_orderflow_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    total = len(results)
    sig = [r for r in results if r["p_value"] < 0.05]
    bonf = 0.05 / total if total else 1
    bonf_sig = [r for r in results if r["p_value"] < bonf]
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total}, SIGNIFICANT (p<0.05): {len(sig)} ({len(sig)/total*100:.0f}%)")
    print(f"BONFERRONI (p<{bonf:.6f}): {len(bonf_sig)}")
    
    if sig:
        by_key = defaultdict(list)
        for r in sig:
            by_key[f"{r['asset']}_{r['tf']}"].append(r)
        print(f"\nPer asset/TF:")
        for k in sorted(by_key.keys()):
            best = max(by_key[k], key=lambda x: x["oos_sharpe"])
            bf = "🏆" if best["p_value"] < bonf else "✅"
            print(f"  {bf} {k.upper()}: OOS {best['oos_sharpe']:.3f} "
                  f"p={best['p_value']:.4f} ct={best['ct']} "
                  f"inv={best['invested_pct']}% trades={best['n_trades']} "
                  f"({len(by_key[k])} sig)")


if __name__ == "__main__":
    main()
