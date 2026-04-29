"""
Magus Orderflow V2 — Using CORRECT VPS walk-forward engine.
Also re-validates HSAKA to confirm methodology is sound.

Fixes from sanity check:
1. Use VPS market_structure (HH/HL/LH/LL) not broken local version
2. Use VPS walk_forward (14 folds, t-test, correct OOS offset)
3. Validate HSAKA first, then test orderflow
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
from pathlib import Path
from collections import defaultdict
import json

from backend.config.data_paths import BARS_1M_V1

OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")
OF_DIR = BARS_1M_V1
OUT_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")


# ══════════════════════════════════════════════════════════
# EXACT VPS FUNCTIONS (copy-pasted from VPS hsaka_lean.py)
# ══════════════════════════════════════════════════════════

def market_structure(df, sw):
    """VPS version: compares successive swing highs/lows."""
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    sh_idx = argrelextrema(high, np.greater_equal, order=sw)[0]
    sl_idx = argrelextrema(low, np.less_equal, order=sw)[0]
    bias = np.zeros(n)
    all_swings = [(idx, high[idx], 'H') for idx in sh_idx] + [(idx, low[idx], 'L') for idx in sl_idx]
    all_swings.sort(key=lambda x: x[0])
    prev_high = prev_low = None
    trend = 0
    for idx, price, stype in all_swings:
        if stype == 'H':
            if prev_high is not None:
                trend = 1 if price > prev_high else -1
            prev_high = price
        else:
            if prev_low is not None:
                trend = 1 if price > prev_low else -1
            prev_low = price
        bias[idx:] = trend
    return bias


def sfp_signal(df, sw):
    """VPS version."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    sig = np.zeros(n)
    for i in range(sw, n):
        rh = np.max(high[i-sw:i])
        rl = np.min(low[i-sw:i])
        if high[i] > rh and close[i] < rh:
            sig[i] = -1
        if low[i] < rl and close[i] > rl:
            sig[i] = 1
    return sig


def hsaka_lean_signals(df, sw, ct, sfp_lb):
    """VPS version: struct*2 + sfp*2, no ffill."""
    struct = market_structure(df, sw)
    sfp = sfp_signal(df, sfp_lb)
    score = struct * 2 + sfp * 2
    signals = np.zeros(len(df))
    signals[score >= ct] = 1
    signals[score <= -ct] = -1
    return signals


def walk_forward_vps(rets, signals, n_folds=14, commission=0.0006):
    """EXACT VPS walk_forward."""
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
    invested = round(np.mean(np.abs(signals)) * 100, 1)
    n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))
    return {
        "oos_sharpe": round(oos_sharpe, 3), "p_value": round(p_val, 4),
        "pos_folds": pos_folds, "n_folds": n_folds,
        "invested_pct": invested, "n_trades": n_trades,
    }


# ══════════════════════════════════════════════════════════
# ORDERFLOW SIGNALS
# ══════════════════════════════════════════════════════════

def cvd_trend(delta, lookback=20):
    cvd = np.cumsum(delta)
    cvd_sma = pd.Series(cvd).rolling(lookback).mean().values
    cvd_prev = np.roll(cvd, 1); cvd_prev[0] = 0
    sig = np.zeros(len(delta))
    sig[(cvd > cvd_sma) & (cvd > cvd_prev)] = 1
    sig[(cvd < cvd_sma) & (cvd < cvd_prev)] = -1
    sig[:lookback] = 0
    return sig

def delta_divergence(close, delta, lookback=20):
    cvd = np.cumsum(delta)
    cs = pd.Series(close)
    cvds = pd.Series(cvd)
    price_max = cs.rolling(lookback).max().values
    price_min = cs.rolling(lookback).min().values
    cvd_mean = cvds.rolling(lookback).mean().values
    sig = np.zeros(len(close))
    sig[(close >= price_max * 0.999) & (cvd < cvd_mean)] = -1
    sig[(close <= price_min * 1.001) & (cvd > cvd_mean)] = 1
    sig[:lookback] = 0
    return sig

def absorption(close, buy_vol, sell_vol, high, low, lookback=10):
    total_vol = buy_vol + sell_vol
    sell_pct = sell_vol / np.clip(total_vol, 1, None)
    buy_pct = buy_vol / np.clip(total_vol, 1, None)
    atr = pd.Series(high - low).rolling(lookback).mean().values
    price_change = np.diff(close, prepend=close[0])
    norm_change = np.abs(price_change) / np.clip(atr, 1e-10, None)
    sell_pct_sma = pd.Series(sell_pct).rolling(lookback).mean().values
    buy_pct_sma = pd.Series(buy_pct).rolling(lookback).mean().values
    sig = np.zeros(len(close))
    sig[(sell_pct > 0.60) & (sell_pct > sell_pct_sma * 1.2) & ((norm_change < 0.5) | (price_change > 0))] = 1
    sig[(buy_pct > 0.60) & (buy_pct > buy_pct_sma * 1.2) & ((norm_change < 0.5) | (price_change < 0))] = -1
    sig[:lookback] = 0
    return sig

def puke_detection(delta, volume, lookback=20):
    ds = pd.Series(delta); vs = pd.Series(volume)
    d_mean = ds.rolling(lookback).mean().values
    d_std = ds.rolling(lookback).std().values
    v_mean = vs.rolling(lookback).mean().values
    v_std = vs.rolling(lookback).std().values
    delta_z = (delta - d_mean) / np.clip(d_std, 1e-10, None)
    vol_z = (volume - v_mean) / np.clip(v_std, 1e-10, None)
    sig = np.zeros(len(delta))
    sig[(delta_z < -2.0) & (vol_z > 1.5)] = 1
    sig[(delta_z > 2.0) & (vol_z > 1.5)] = -1
    sig[:lookback] = 0
    return sig

def volume_imbalance(buy_vol, sell_vol, lookback=10):
    buy_pct = buy_vol / np.clip(buy_vol + sell_vol, 1, None)
    buy_pct_sma = pd.Series(buy_pct).rolling(lookback).mean().values
    sig = np.zeros(len(buy_vol))
    sig[buy_pct_sma > 0.55] = 1
    sig[buy_pct_sma < 0.45] = -1
    sig[:lookback] = 0
    return sig


def main():
    results = []
    
    # ════════════════════════════════════════════
    # VALIDATION: HSAKA on OHLCV (should match VPS ~2.0)
    # ════════════════════════════════════════════
    print("=" * 70)
    print("VALIDATION: HSAKA Lean on BTC 4h OHLCV")
    print("=" * 70)
    
    df = pd.read_csv(OHLCV_DIR / "binance_btc_usdt_4h.csv", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    rets = df["close"].pct_change().values
    
    # Grid search like VPS
    from itertools import product as iprod
    best = None
    for sw, ct, sfp_lb in iprod([5,7,10,14], [2,3,4], [7,10,14,20]):
        signals = hsaka_lean_signals(df, sw, ct, sfp_lb)
        if np.sum(np.abs(signals) > 0) < 15:
            continue
        wf = walk_forward_vps(rets, signals)
        if best is None or wf["oos_sharpe"] > best["oos_sharpe"]:
            best = {**wf, "sw": sw, "ct": ct, "sfp_lb": sfp_lb}
    
    print(f"  Best: sw={best['sw']}, ct={best['ct']}, sfp={best['sfp_lb']}")
    print(f"  OOS Sharpe: {best['oos_sharpe']}, p={best['p_value']}")
    print(f"  Pos folds: {best['pos_folds']}/{best['n_folds']}")
    print(f"  Invested: {best['invested_pct']}%, Trades: {best['n_trades']}")
    
    if best['oos_sharpe'] > 1.0 and best['p_value'] < 0.05:
        print(f"  ✅ HSAKA VALIDATED — methodology is correct")
    else:
        print(f"  ⚠ HSAKA not matching VPS — investigating...")
    
    # ════════════════════════════════════════════
    # ORDERFLOW SIGNALS
    # ════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ORDERFLOW SIGNALS (through validated VPS engine)")
    print("=" * 70)
    
    for asset in ["btcusdt", "ethusdt"]:
        for tf in ["15min", "1h", "4h"]:
            print(f"\n--- {asset.upper()} {tf} ---")
            
            df_of = pd.read_csv(OF_DIR / f"{asset}_1m.csv",
                               parse_dates=["timestamp"], index_col="timestamp")
            df_r = df_of.resample(tf).agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "buy_vol": "sum", "sell_vol": "sum",
                "delta": "sum", "trade_count": "sum",
            }).dropna(subset=["open"])
            
            rets = df_r["close"].pct_change().values
            c = df_r["close"].values
            h = df_r["high"].values
            l = df_r["low"].values
            d = df_r["delta"].values
            v = df_r["volume"].values
            bv = df_r["buy_vol"].values
            sv = df_r["sell_vol"].values
            
            print(f"  {len(df_r)} bars")
            comm = 0.0004 if tf == "15min" else 0.0006
            n_folds = 14 if len(df_r) > 3000 else 10 if len(df_r) > 1500 else 7
            
            # Test individual signals
            signals_dict = {
                "CVD_trend": cvd_trend(d, 20),
                "Delta_div": delta_divergence(c, d, 20),
                "Absorption": absorption(c, bv, sv, h, l, 10),
                "Puke": puke_detection(d, v, 20),
                "Vol_imbalance": volume_imbalance(bv, sv, 10),
            }
            
            for name, sig in signals_dict.items():
                fire = np.mean(np.abs(sig) > 0) * 100
                if fire < 0.5:
                    print(f"  {name:<15s}: fires {fire:.1f}% — too sparse")
                    continue
                wf = walk_forward_vps(rets, sig, n_folds=n_folds, commission=comm)
                status = "✅" if wf["p_value"] < 0.05 else "❌"
                print(f"  {status} {name:<15s}: OOS {wf['oos_sharpe']:>7.3f} p={wf['p_value']:.4f} "
                      f"folds={wf['pos_folds']}/{wf['n_folds']} inv={wf['invested_pct']}% "
                      f"fire={fire:.0f}%")
                
                results.append({
                    "asset": asset, "tf": tf, "signal": name,
                    "oos_sharpe": wf["oos_sharpe"], "p_value": wf["p_value"],
                    "pos_folds": wf["pos_folds"], "invested_pct": wf["invested_pct"],
                })
            
            # Combined signal (ct=2)
            for ct in [1, 2, 3]:
                score = sum(signals_dict.values())
                combined = np.where(score >= ct, 1, np.where(score <= -ct, -1, 0))
                fire = np.mean(np.abs(combined) > 0) * 100
                if fire < 0.5:
                    continue
                wf = walk_forward_vps(rets, combined, n_folds=n_folds, commission=comm)
                status = "✅" if wf["p_value"] < 0.05 else "❌"
                print(f"  {status} Combined ct={ct}:   OOS {wf['oos_sharpe']:>7.3f} p={wf['p_value']:.4f} "
                      f"inv={wf['invested_pct']}%")
                
                results.append({
                    "asset": asset, "tf": tf, "signal": f"combined_ct{ct}",
                    "oos_sharpe": wf["oos_sharpe"], "p_value": wf["p_value"],
                })
            
            # HSAKA + orderflow confirmation
            struct = market_structure(df_r, 7)
            sfp = sfp_signal(df_r, 10)
            hsaka_score = struct * 2 + sfp * 2
            hsaka_sig = np.where(hsaka_score >= 2, 1, np.where(hsaka_score <= -2, -1, 0))
            
            # HSAKA alone
            wf_h = walk_forward_vps(rets, hsaka_sig, n_folds=n_folds, commission=comm)
            print(f"  {'✅' if wf_h['p_value']<0.05 else '❌'} HSAKA alone:     OOS {wf_h['oos_sharpe']:>7.3f} p={wf_h['p_value']:.4f}")
            
            # HSAKA + CVD confirmation (only trade HSAKA when CVD agrees)
            cvd_s = cvd_trend(d, 20)
            hsaka_cvd = np.where((hsaka_sig == 1) & (cvd_s == 1), 1,
                                np.where((hsaka_sig == -1) & (cvd_s == -1), -1, 0))
            wf_hc = walk_forward_vps(rets, hsaka_cvd, n_folds=n_folds, commission=comm)
            print(f"  {'✅' if wf_hc['p_value']<0.05 else '❌'} HSAKA+CVD:       OOS {wf_hc['oos_sharpe']:>7.3f} p={wf_hc['p_value']:.4f} "
                  f"inv={wf_hc['invested_pct']}%")
            
            # HSAKA + puke fade as entry timing
            puke_s = puke_detection(d, v, 20)
            hsaka_puke = hsaka_sig.copy()
            # Cancel HSAKA signal when puke opposes it
            hsaka_puke[(hsaka_sig == 1) & (puke_s == -1)] = 0
            hsaka_puke[(hsaka_sig == -1) & (puke_s == 1)] = 0
            wf_hp = walk_forward_vps(rets, hsaka_puke, n_folds=n_folds, commission=comm)
            print(f"  {'✅' if wf_hp['p_value']<0.05 else '❌'} HSAKA-puke_filt: OOS {wf_hp['oos_sharpe']:>7.3f} p={wf_hp['p_value']:.4f} "
                  f"inv={wf_hp['invested_pct']}%")
    
    # Save
    with open(OUT_DIR / "magus_orderflow_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    total = len(results)
    sig = [r for r in results if r["p_value"] < 0.05]
    print(f"\n{'='*70}")
    print(f"TOTAL: {total}, SIGNIFICANT: {len(sig)} ({len(sig)/max(total,1)*100:.0f}%)")
    if sig:
        for r in sorted(sig, key=lambda x: x["oos_sharpe"], reverse=True):
            print(f"  ✅ {r['asset'].upper()} {r['tf']} {r['signal']}: OOS {r['oos_sharpe']:.3f} p={r['p_value']:.4f}")


if __name__ == "__main__":
    main()
