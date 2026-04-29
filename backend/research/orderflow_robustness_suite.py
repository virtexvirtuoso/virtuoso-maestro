"""
Orderflow Robustness Suite — Exhaustive Testing
================================================
Closes ALL gaps from the initial component validation:

1. MULTI-TIMEFRAME: 15m, 1h, 4h (not just 1h)
2. MULTI-HORIZON: 5, 10, 30, 60 bar holds
3. V2 MICROSTRUCTURE: large trade delta, clusters, price impact, arrival rate
4. CONDITIONAL SIGNALS: high-volume regime, swing points, session-filtered
5. SIGNAL COMBINATIONS: best 2-3 component blends
6. REGIME-FILTERED: trending vs ranging market overlay

Methodology: 14-fold expanding WF, non-overlapping trades, 10bps costs, bootstrap CI
If NOTHING passes here, orderflow is definitively dead for systematic trading.

Author: Maestro 🎼
Date: 2026-03-07
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import argrelextrema
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

from backend.config.data_paths import BARS_1M_V1, BARS_1M_V2

OF_DIR = BARS_1M_V1
OF_V2_DIR = BARS_1M_V2
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")
COST_BPS = 10


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_of(asset, tf="1h", v2=False):
    """Load orderflow data at given timeframe."""
    suffix = "_v2" if v2 else ""
    base = OF_V2_DIR if v2 else OF_DIR
    suffix2 = "_v2" if v2 else ""
    path = base / f"{asset}_1m{suffix2}.csv"
    if not path.exists():
        return None
    
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    
    # Build aggregation dict based on available columns
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "trade_count": "sum", "dollar_volume": "sum",
        "buy_vol": "sum", "sell_vol": "sum", "delta": "sum",
    }
    
    # Add optional columns if they exist
    for col in ["buy_dollar", "sell_dollar", "delta_dollar"]:
        if col in df.columns:
            agg[col] = "sum"
    
    # V2 specific columns
    if v2:
        v2_sum_cols = ["large_trade_count", "large_trade_vol", "large_buy_vol", 
                       "large_sell_vol", "buy_cluster_count", "sell_cluster_count",
                       "large_delta"]
        v2_mean_cols = ["avg_trade_size", "price_impact", "trades_per_sec_avg"]
        v2_max_cols = ["max_trade_size", "trades_per_sec_max"]
        
        for col in v2_sum_cols:
            if col in df.columns:
                agg[col] = "sum"
        for col in v2_mean_cols:
            if col in df.columns:
                agg[col] = "mean"
        for col in v2_max_cols:
            if col in df.columns:
                agg[col] = "max"
    
    if tf != "1min":
        df = df.resample(tf).agg(agg).dropna(subset=["open"])
    
    df["buy_pct"] = df["buy_vol"] / df["volume"].clip(lower=1e-10)
    df["return"] = df["close"].pct_change()
    return df


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════

def walk_forward(df, signal_col, hold_bars=30, n_folds=14, cost_bps=COST_BPS):
    """14-fold WF with non-overlapping trades + bootstrap CI."""
    total = len(df)
    if total < 500:
        return {"error": "Too few bars", "n_bars": total}
    
    date_range = df.index[-1] - df.index[0]
    months = max(date_range.days / 30.44, 1)
    bpm = int(total / months)
    
    min_train = bpm * 3
    test_size = bpm
    if test_size < 10:
        return {"error": "Test size too small"}
    
    available = total - min_train
    n_folds = min(n_folds, max(available // test_size, 0))
    
    if n_folds < 3:
        return {"error": "Insufficient data for WF", "n_folds": n_folds}
    
    signals = df[signal_col].values
    closes = df["close"].values
    all_trades = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = min(test_start + test_size, total)
        if test_end > total:
            break
        
        i = test_start
        while i < test_end - hold_bars:
            sig = signals[i]
            if np.isnan(sig) or sig == 0:
                i += 1
                continue
            
            pos = np.sign(sig)
            raw = (closes[min(i + hold_bars, total - 1)] / closes[i] - 1) * pos
            net = raw - (cost_bps / 10000)
            all_trades.append(net)
            i += hold_bars
    
    if len(all_trades) < 15:
        return {"error": "Too few trades", "n": len(all_trades)}
    
    net = np.array(all_trades)
    t_stat, p_val = sp_stats.ttest_1samp(net, 0)
    
    # Annualization factor depends on timeframe
    # We'll use a generic one based on hold_bars
    # Assume ~8760 hours/year; for 1h data with 30-bar hold = 292 trades/year
    tpy = 8760 / hold_bars  # rough trades per year for hourly
    sharpe = (np.mean(net) / np.std(net)) * np.sqrt(tpy) if np.std(net) > 0 else 0
    
    # Bootstrap
    bs = []
    for _ in range(1000):
        s = np.random.choice(net, size=len(net), replace=True)
        bs.append((np.mean(s) / np.std(s)) * np.sqrt(tpy) if np.std(s) > 0 else 0)
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
    
    pf = abs(np.sum(net[net > 0]) / np.sum(net[net <= 0])) if np.sum(net[net <= 0]) != 0 else float("inf")
    
    return {
        "n_trades": len(all_trades),
        "mean_pct": float(np.mean(net) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "sharpe": float(sharpe),
        "sharpe_ci": [float(ci_lo), float(ci_hi)],
        "hit_rate": float(np.mean(net > 0)),
        "pf": float(pf),
        "total_ret_pct": float(np.sum(net) * 100),
    }


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════

# --- Core signals (both directions) ---

def sig_cvd_regime(df):
    cvd = df["delta"].cumsum()
    return ((cvd.rolling(20, min_periods=10).mean() > cvd.rolling(50, min_periods=25).mean()).astype(float) * 2 - 1)

def sig_trade_flow(df):
    return (df["buy_pct"].rolling(5, min_periods=3).mean() - 0.5) * 2

def sig_pressure(df):
    vp = (df["buy_vol"] - df["sell_vol"]) / df["volume"].clip(lower=1e-10)
    return vp.rolling(5, min_periods=3).mean().clip(-1, 1)

def sig_imbalance(df):
    r = df["buy_pct"].rolling(5, min_periods=3).mean()
    m = df["buy_pct"].rolling(12, min_periods=6).mean()
    o = df["buy_pct"].rolling(25, min_periods=12).mean()
    return ((r - 0.5) * 0.40 + (m - 0.5) * 0.35 + (o - 0.5) * 0.25) * 2

def sig_stacked(df):
    buy_dom = (df["buy_pct"] > 0.55).astype(int).rolling(5, min_periods=3).sum()
    sell_dom = (df["buy_pct"] < 0.45).astype(int).rolling(5, min_periods=3).sum()
    return (buy_dom - sell_dom) / 5

def sig_vol_shock(df):
    mu = df["volume"].rolling(50, min_periods=20).mean()
    sigma = df["volume"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    shock = ((df["volume"] - mu) / sigma > 1.5).astype(float)
    raw = shock * np.sign(df["delta"])
    return raw.replace(0, np.nan).ffill().fillna(0)


# --- V2 Microstructure signals ---

def sig_large_trade_delta(df):
    """Large trade delta — institutional flow direction."""
    if "large_delta" not in df.columns:
        return None
    ld = df["large_delta"]
    mu = ld.rolling(50, min_periods=20).mean()
    sigma = ld.rolling(50, min_periods=20).std().clip(lower=1e-10)
    return ((ld - mu) / sigma).clip(-3, 3) / 3

def sig_large_trade_delta_contra(df):
    s = sig_large_trade_delta(df)
    return -s if s is not None else None

def sig_large_trade_pct(df):
    """When large trades are a bigger share of volume AND buying → signal."""
    if "large_pct_of_vol" not in df.columns or "large_delta" not in df.columns:
        return None
    lpv = df["large_pct_of_vol"].rolling(10, min_periods=5).mean()
    lpv_z = (lpv - lpv.rolling(50, min_periods=20).mean()) / lpv.rolling(50, min_periods=20).std().clip(lower=1e-10)
    direction = np.sign(df["large_delta"].rolling(5, min_periods=3).mean())
    return (lpv_z.clip(0, 3) / 3 * direction).clip(-1, 1)

def sig_cluster_imbalance(df):
    """Buy cluster count vs sell cluster count — spatial clustering."""
    if "buy_cluster_count" not in df.columns:
        return None
    bc = df["buy_cluster_count"].rolling(5, min_periods=3).mean()
    sc = df["sell_cluster_count"].rolling(5, min_periods=3).mean()
    total = (bc + sc).clip(lower=1)
    return ((bc - sc) / total).clip(-1, 1)

def sig_cluster_imbalance_contra(df):
    s = sig_cluster_imbalance(df)
    return -s if s is not None else None

def sig_price_impact(df):
    """Price impact z-score × delta direction — high impact + buying = institutional."""
    if "price_impact" not in df.columns:
        return None
    pi = df["price_impact"].rolling(10, min_periods=5).mean()
    mu = pi.rolling(50, min_periods=20).mean()
    sigma = pi.rolling(50, min_periods=20).std().clip(lower=1e-10)
    pi_z = (pi - mu) / sigma
    direction = np.sign(df["delta"].rolling(5, min_periods=3).mean())
    return (pi_z.clip(0, 3) / 3 * direction).clip(-1, 1)

def sig_arrival_rate(df):
    """Trade arrival rate acceleration — sudden increase in trades/sec."""
    if "trades_per_sec_avg" not in df.columns:
        return None
    tps = df["trades_per_sec_avg"].rolling(10, min_periods=5).mean()
    mu = tps.rolling(50, min_periods=20).mean()
    sigma = tps.rolling(50, min_periods=20).std().clip(lower=1e-10)
    tps_z = (tps - mu) / sigma
    direction = np.sign(df["delta"].rolling(5, min_periods=3).mean())
    return (tps_z.clip(0, 3) / 3 * direction).clip(-1, 1)


# --- Conditional signals ---

def sig_cvd_high_volume_only(df):
    """CVD regime but only active during high-volume bars."""
    vol_z = (df["volume"] - df["volume"].rolling(50, min_periods=20).mean()) / \
            df["volume"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    regime = sig_cvd_regime(df)
    # Zero out signal when volume is below average
    return regime * (vol_z > 0).astype(float)

def sig_pressure_at_swings(df):
    """Pressure signal only at swing highs/lows."""
    order = max(5, min(20, len(df) // 200))
    highs = argrelextrema(df["high"].values, np.greater_equal, order=order)[0]
    lows = argrelextrema(df["low"].values, np.less_equal, order=order)[0]
    
    swing_mask = np.zeros(len(df), dtype=bool)
    for idx in np.concatenate([highs, lows]):
        # Active within 3 bars of swing
        start = max(0, idx - 3)
        end = min(len(df), idx + 4)
        swing_mask[start:end] = True
    
    pressure = sig_pressure(df)
    result = pressure.copy()
    result[~swing_mask] = 0
    # Forward fill to maintain position between swings
    return result.replace(0, np.nan).ffill().fillna(0)

def sig_delta_european_session(df):
    """Delta contrarian only during European session (strongest from deep dive)."""
    hour = df.index.hour
    is_europe = (hour >= 8) & (hour < 14)
    
    mu = df["delta"].rolling(50, min_periods=20).mean()
    sigma = df["delta"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    z = (df["delta"] - mu) / sigma
    contra = -z.clip(-3, 3) / 3
    
    result = contra.copy()
    result[~is_europe] = 0
    return result.replace(0, np.nan).ffill().fillna(0)


# --- Combination signals ---

def sig_combo_best3(df):
    """Combination: vol_shock + stacked + pressure (top 3 from initial test)."""
    s1 = sig_vol_shock(df)
    s2 = sig_stacked(df)
    s3 = sig_pressure(df)
    return (s1 + s2 + s3) / 3

def sig_combo_vol_delta(df):
    """Volume shock direction + CVD regime confirmation."""
    s1 = sig_vol_shock(df)
    s2 = sig_cvd_regime(df)
    return (s1 + s2) / 2

def sig_combo_flow_stacked(df):
    """Trade flow + stacked imbalance."""
    s1 = sig_trade_flow(df)
    s2 = sig_stacked(df)
    return (s1 + s2) / 2


# --- Regime-filtered ---

def sig_trend_filtered_pressure(df):
    """Pressure signal only when price is trending (SMA20 vs SMA50)."""
    sma20 = df["close"].rolling(20, min_periods=10).mean()
    sma50 = df["close"].rolling(50, min_periods=25).mean()
    trending = (abs(sma20 - sma50) / sma50 > 0.02)  # >2% divergence = trending
    
    pressure = sig_pressure(df)
    result = pressure.copy()
    result[~trending] = 0
    return result.replace(0, np.nan).ffill().fillna(0)

def sig_range_filtered_contra(df):
    """Contrarian delta only in ranging markets."""
    sma20 = df["close"].rolling(20, min_periods=10).mean()
    sma50 = df["close"].rolling(50, min_periods=25).mean()
    ranging = (abs(sma20 - sma50) / sma50 < 0.01)  # <1% = ranging
    
    mu = df["delta"].rolling(50, min_periods=20).mean()
    sigma = df["delta"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    contra = -(df["delta"] - mu) / sigma
    contra = contra.clip(-3, 3) / 3
    
    result = contra.copy()
    result[~ranging] = 0
    return result.replace(0, np.nan).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # Find all assets
    assets_v1 = sorted([f.stem.replace("_1m", "")
                        for f in OF_DIR.glob("*_1m.csv")
                        if "v2" not in f.stem])
    assets_v2 = sorted([f.stem.replace("_1m_v2", "")
                         for f in OF_V2_DIR.glob("*_1m_v2.csv")])
    
    print("=" * 110)
    print("ORDERFLOW ROBUSTNESS SUITE — EXHAUSTIVE TESTING")
    print(f"V1 assets: {len(assets_v1)} ({', '.join(assets_v1)})")
    print(f"V2 assets: {len(assets_v2)} ({', '.join(assets_v2)})")
    print("=" * 110)
    
    all_rows = []
    
    # ═══════════════════════════════════════════
    # BLOCK 1: Multi-TF × Multi-Horizon on core signals
    # ═══════════════════════════════════════════
    print(f"\n{'█' * 110}")
    print("█ BLOCK 1: MULTI-TIMEFRAME × MULTI-HORIZON (core signals)")
    print(f"{'█' * 110}")
    
    core_signals = {
        "cvd_regime": sig_cvd_regime,
        "trade_flow": sig_trade_flow,
        "pressure": sig_pressure,
        "vol_shock": sig_vol_shock,
        "stacked": sig_stacked,
    }
    
    for asset in assets_v1:
        for tf, tf_label in [("15min", "15m"), ("1h", "1h"), ("4h", "4h")]:
            df = load_of(asset, tf)
            if df is None or len(df) < 500:
                continue
            
            for sig_name, sig_func in core_signals.items():
                df[sig_name] = sig_func(df)
                
                # Adjust hold bars by timeframe
                # Target roughly same calendar time across TFs
                for hold_hours in [5, 10, 30, 60]:
                    hold_bars_map = {"15min": hold_hours * 4, "1h": hold_hours, "4h": max(hold_hours // 4, 2)}
                    hb = hold_bars_map[tf]
                    
                    result = walk_forward(df, sig_name, hold_bars=hb)
                    
                    passes = not ("error" in result) and result.get("p_value", 1) < 0.05 and result.get("sharpe", 0) > 0.5
                    
                    row = {
                        "block": "multi_tf_horizon",
                        "asset": asset, "tf": tf_label, "signal": sig_name,
                        "hold_hours": hold_hours, "hold_bars": hb,
                    }
                    if "error" not in result:
                        row.update(result)
                        row["verdict"] = "PASS" if passes else "FAIL"
                    else:
                        row["verdict"] = "ERROR"
                        row["error"] = result["error"]
                    
                    all_rows.append(row)
                    
                    if passes:
                        print(f"  ✅ {asset} {tf_label} {sig_name} h={hold_hours}hrs: "
                              f"Sharpe={result['sharpe']:+.2f} t={result['t_stat']:.2f} p={result['p_value']:.4f}")
    
    b1_pass = sum(1 for r in all_rows if r.get("verdict") == "PASS" and r["block"] == "multi_tf_horizon")
    b1_total = sum(1 for r in all_rows if r["block"] == "multi_tf_horizon" and r.get("verdict") != "ERROR")
    print(f"\n  Block 1: {b1_pass}/{b1_total} pass")
    
    # ═══════════════════════════════════════════
    # BLOCK 2: V2 Microstructure signals
    # ═══════════════════════════════════════════
    print(f"\n{'█' * 110}")
    print("█ BLOCK 2: V2 MICROSTRUCTURE SIGNALS (BTC, ETH only)")
    print(f"{'█' * 110}")
    
    v2_signals = {
        "large_delta": sig_large_trade_delta,
        "large_delta_contra": sig_large_trade_delta_contra,
        "large_trade_pct": sig_large_trade_pct,
        "cluster_imbalance": sig_cluster_imbalance,
        "cluster_imb_contra": sig_cluster_imbalance_contra,
        "price_impact": sig_price_impact,
        "arrival_rate": sig_arrival_rate,
    }
    
    for asset in assets_v2:
        for tf, tf_label in [("15min", "15m"), ("1h", "1h"), ("4h", "4h")]:
            df = load_of(asset, tf, v2=True)
            if df is None or len(df) < 500:
                continue
            
            for sig_name, sig_func in v2_signals.items():
                signal = sig_func(df)
                if signal is None:
                    continue
                df[sig_name] = signal
                
                for hold_hours in [5, 10, 30]:
                    hold_bars_map = {"15min": hold_hours * 4, "1h": hold_hours, "4h": max(hold_hours // 4, 2)}
                    hb = hold_bars_map[tf]
                    
                    result = walk_forward(df, sig_name, hold_bars=hb)
                    
                    passes = not ("error" in result) and result.get("p_value", 1) < 0.05 and result.get("sharpe", 0) > 0.5
                    
                    row = {
                        "block": "v2_microstructure",
                        "asset": asset, "tf": tf_label, "signal": sig_name,
                        "hold_hours": hold_hours, "hold_bars": hb,
                    }
                    if "error" not in result:
                        row.update(result)
                        row["verdict"] = "PASS" if passes else "FAIL"
                    else:
                        row["verdict"] = "ERROR"
                        row["error"] = result["error"]
                    
                    all_rows.append(row)
                    
                    if passes:
                        print(f"  ✅ {asset} {tf_label} {sig_name} h={hold_hours}hrs: "
                              f"Sharpe={result['sharpe']:+.2f} t={result['t_stat']:.2f} p={result['p_value']:.4f}")
    
    b2_pass = sum(1 for r in all_rows if r.get("verdict") == "PASS" and r["block"] == "v2_microstructure")
    b2_total = sum(1 for r in all_rows if r["block"] == "v2_microstructure" and r.get("verdict") != "ERROR")
    print(f"\n  Block 2: {b2_pass}/{b2_total} pass")
    
    # ═══════════════════════════════════════════
    # BLOCK 3: Conditional signals
    # ═══════════════════════════════════════════
    print(f"\n{'█' * 110}")
    print("█ BLOCK 3: CONDITIONAL SIGNALS")
    print(f"{'█' * 110}")
    
    cond_signals = {
        "cvd_highvol_only": sig_cvd_high_volume_only,
        "pressure_at_swings": sig_pressure_at_swings,
        "delta_europe_contra": sig_delta_european_session,
        "trend_filtered_pressure": sig_trend_filtered_pressure,
        "range_filtered_contra": sig_range_filtered_contra,
    }
    
    for asset in assets_v1:
        df = load_of(asset, "1h")
        if df is None or len(df) < 500:
            continue
        
        for sig_name, sig_func in cond_signals.items():
            try:
                df[sig_name] = sig_func(df)
            except Exception as e:
                continue
            
            for hold_hours in [10, 30]:
                result = walk_forward(df, sig_name, hold_bars=hold_hours)
                
                passes = not ("error" in result) and result.get("p_value", 1) < 0.05 and result.get("sharpe", 0) > 0.5
                
                row = {
                    "block": "conditional",
                    "asset": asset, "tf": "1h", "signal": sig_name,
                    "hold_hours": hold_hours, "hold_bars": hold_hours,
                }
                if "error" not in result:
                    row.update(result)
                    row["verdict"] = "PASS" if passes else "FAIL"
                else:
                    row["verdict"] = "ERROR"
                    row["error"] = result["error"]
                
                all_rows.append(row)
                
                if passes:
                    print(f"  ✅ {asset} 1h {sig_name} h={hold_hours}hrs: "
                          f"Sharpe={result['sharpe']:+.2f} t={result['t_stat']:.2f} p={result['p_value']:.4f}")
    
    b3_pass = sum(1 for r in all_rows if r.get("verdict") == "PASS" and r["block"] == "conditional")
    b3_total = sum(1 for r in all_rows if r["block"] == "conditional" and r.get("verdict") != "ERROR")
    print(f"\n  Block 3: {b3_pass}/{b3_total} pass")
    
    # ═══════════════════════════════════════════
    # BLOCK 4: Signal combinations
    # ═══════════════════════════════════════════
    print(f"\n{'█' * 110}")
    print("█ BLOCK 4: SIGNAL COMBINATIONS")
    print(f"{'█' * 110}")
    
    combo_signals = {
        "combo_best3": sig_combo_best3,
        "combo_vol_cvd": sig_combo_vol_delta,
        "combo_flow_stack": sig_combo_flow_stacked,
    }
    
    for asset in assets_v1:
        df = load_of(asset, "1h")
        if df is None or len(df) < 500:
            continue
        
        for sig_name, sig_func in combo_signals.items():
            df[sig_name] = sig_func(df)
            
            for hold_hours in [10, 30]:
                result = walk_forward(df, sig_name, hold_bars=hold_hours)
                
                passes = not ("error" in result) and result.get("p_value", 1) < 0.05 and result.get("sharpe", 0) > 0.5
                
                row = {
                    "block": "combinations",
                    "asset": asset, "tf": "1h", "signal": sig_name,
                    "hold_hours": hold_hours, "hold_bars": hold_hours,
                }
                if "error" not in result:
                    row.update(result)
                    row["verdict"] = "PASS" if passes else "FAIL"
                else:
                    row["verdict"] = "ERROR"
                    row["error"] = result["error"]
                
                all_rows.append(row)
                
                if passes:
                    print(f"  ✅ {asset} 1h {sig_name} h={hold_hours}hrs: "
                          f"Sharpe={result['sharpe']:+.2f} t={result['t_stat']:.2f} p={result['p_value']:.4f}")
    
    b4_pass = sum(1 for r in all_rows if r.get("verdict") == "PASS" and r["block"] == "combinations")
    b4_total = sum(1 for r in all_rows if r["block"] == "combinations" and r.get("verdict") != "ERROR")
    print(f"\n  Block 4: {b4_pass}/{b4_total} pass")
    
    # ═══════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════
    sdf = pd.DataFrame(all_rows)
    valid = sdf[sdf["verdict"].isin(["PASS", "FAIL"])]
    n_pass = (valid["verdict"] == "PASS").sum()
    n_total = len(valid)
    expected_false_pos = n_total * 0.05
    
    print(f"\n\n{'=' * 110}")
    print("FINAL SUMMARY — ORDERFLOW ROBUSTNESS SUITE")
    print(f"{'=' * 110}")
    print(f"\n  Total tests:        {n_total}")
    print(f"  Passes (p<0.05 & Sharpe>0.5): {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"  Expected false positives at 5%: {expected_false_pos:.0f}")
    print(f"  Pass rate vs expectation: {'ABOVE' if n_pass > expected_false_pos * 1.5 else 'AT OR BELOW'} random chance")
    
    # Bonferroni
    bonf = 0.05 / n_total
    bonf_pass = sum(1 for _, r in valid.iterrows() if r.get("p_value", 1) < bonf and r.get("sharpe", 0) > 0.5)
    print(f"  Bonferroni corrected (p < {bonf:.6f}): {bonf_pass} pass")
    
    # By block
    print(f"\n  By block:")
    for block in ["multi_tf_horizon", "v2_microstructure", "conditional", "combinations"]:
        bv = valid[valid["block"] == block]
        bp = (bv["verdict"] == "PASS").sum()
        bt = len(bv)
        if bt > 0:
            print(f"    {block:<25s}: {bp}/{bt} ({bp/bt*100:.1f}%)")
    
    # Top results regardless of block
    if n_pass > 0:
        print(f"\n  All passing tests:")
        passes = valid[valid["verdict"] == "PASS"].sort_values("sharpe", ascending=False)
        for _, r in passes.iterrows():
            print(f"    {r['asset']} {r['tf']} {r['signal']} h={r['hold_hours']}hrs: "
                  f"Sharpe={r['sharpe']:+.2f} t={r['t_stat']:.2f} p={r['p_value']:.4f} n={r.get('n_trades', '?')}")
    
    # Save
    out_json = RESULTS_DIR / "orderflow_robustness_suite_20260307.json"
    sdf.to_json(out_json, orient="records", indent=2)
    print(f"\n  JSON: {out_json}")
    
    out_csv = RESULTS_DIR / "orderflow_robustness_suite_20260307.csv"
    sdf.to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}")


if __name__ == "__main__":
    main()
