"""
Systematic Walk-Forward Validation of ALL Orderflow Components
==============================================================
Tests each of Virtuoso's 8 orderflow indicator components independently
across all available assets with actual orderflow data.

Components (from orderflow_indicators.py):
1. CVD — net buy-sell volume direction (already tested, included for completeness)
2. Trade Flow — buy/sell ratio (buy_pct)
3. Imbalance — temporal buy/sell split (recent 25% vs medium 50% vs overall)
4. Open Interest — OI changes (requires derivatives data, proxy via volume trend)
5. Pressure — volume-weighted + value-weighted + count + large trade bias
6. Liquidity — trade frequency + volume scoring
7. Liquidity Zones — SMC-style S/R (needs multi-TF, simplified here)
8. Stacked Imbalance — consecutive bar-level buy/sell clustering

Methodology:
- 14 expanding-window folds, non-overlapping trades
- 10bps transaction costs
- 30-bar hold at 1h
- Bootstrap 95% CI on Sharpe
- All available orderflow assets

Author: Maestro 🎼
Date: 2026-03-07
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

from backend.config.data_paths import BARS_1M_V1

OF_DIR = BARS_1M_V1
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 10
HOLD_BARS = 30  # ~30 hours at 1h


def load_1h(asset):
    """Load and resample to 1h."""
    path = OF_DIR / f"{asset}_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "buy_vol": "sum", "sell_vol": "sum",
        "delta": "sum", "trade_count": "sum",
        "dollar_volume": "sum",
        "buy_dollar": "sum", "sell_dollar": "sum",
        "delta_dollar": "sum",
    }).dropna(subset=["open"])
    df["buy_pct"] = df["buy_vol"] / df["volume"].clip(lower=1e-10)
    df["return_1h"] = df["close"].pct_change()
    return df


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATORS — one per component
# Each returns a Series of signal values (positive = bullish)
# ═══════════════════════════════════════════════════════════

def signal_cvd_regime(df):
    """Component 1: CVD Regime (SMA20>SMA50 on cumulative delta)."""
    cvd = df["delta"].cumsum()
    sma20 = cvd.rolling(20, min_periods=10).mean()
    sma50 = cvd.rolling(50, min_periods=25).mean()
    return (sma20 > sma50).astype(float) * 2 - 1


def signal_cvd_contrarian(df):
    """Component 1b: CVD bar-level contrarian (negate delta z-score)."""
    mu = df["delta"].rolling(50, min_periods=20).mean()
    sigma = df["delta"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    z = (df["delta"] - mu) / sigma
    return -z.clip(-3, 3) / 3  # Normalize to [-1, 1]


def signal_trade_flow(df):
    """Component 2: Trade Flow — buy percentage deviation from 50%."""
    # Virtuoso: buy_pct maps to -1..+1 around 0.5
    # Smooth over 5 bars to reduce noise
    bp = df["buy_pct"].rolling(5, min_periods=3).mean()
    return (bp - 0.5) * 2  # 0.6 buy_pct → +0.2 signal


def signal_trade_flow_contrarian(df):
    """Component 2b: Trade Flow contrarian — fade buy percentage."""
    bp = df["buy_pct"].rolling(5, min_periods=3).mean()
    return -(bp - 0.5) * 2


def signal_imbalance(df):
    """Component 3: Temporal Imbalance (recent vs medium vs overall buy/sell)."""
    # Mirrors Virtuoso: 40% recent (5 bars), 35% medium (12 bars), 25% overall (25 bars)
    recent = df["buy_pct"].rolling(5, min_periods=3).mean()
    medium = df["buy_pct"].rolling(12, min_periods=6).mean()
    overall = df["buy_pct"].rolling(25, min_periods=12).mean()
    
    combined = (recent - 0.5) * 0.40 + (medium - 0.5) * 0.35 + (overall - 0.5) * 0.25
    return combined * 2  # Scale to [-1, 1] range


def signal_imbalance_contrarian(df):
    """Component 3b: Imbalance contrarian."""
    return -signal_imbalance(df)


def signal_pressure(df):
    """Component 4: Pressure — volume-weighted + value-weighted + count + large trade."""
    # Volume pressure
    vol_pressure = (df["buy_vol"] - df["sell_vol"]) / df["volume"].clip(lower=1e-10)
    
    # Value (dollar) pressure
    val_pressure = (df["buy_dollar"] - df["sell_dollar"]) / df["dollar_volume"].clip(lower=1e-10)
    
    # Combined (simplified from Virtuoso's 4-component blend)
    # 40% volume, 30% value, 30% count (we don't have separate count data)
    combined = vol_pressure * 0.5 + val_pressure * 0.5
    
    # Smooth
    return combined.rolling(5, min_periods=3).mean().clip(-1, 1)


def signal_pressure_contrarian(df):
    """Component 4b: Pressure contrarian."""
    return -signal_pressure(df)


def signal_liquidity(df):
    """Component 5: Liquidity — trade frequency + volume z-score as regime filter.
    High liquidity + buying = more confident bullish; low liquidity = uncertain.
    """
    # Trade frequency z-score
    tc_mu = df["trade_count"].rolling(50, min_periods=20).mean()
    tc_sigma = df["trade_count"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    freq_z = (df["trade_count"] - tc_mu) / tc_sigma
    
    # Volume z-score
    vol_mu = df["volume"].rolling(50, min_periods=20).mean()
    vol_sigma = df["volume"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    vol_z = (df["volume"] - vol_mu) / vol_sigma
    
    # Liquidity regime: high activity = trust the delta more
    liq_regime = (freq_z + vol_z) / 2
    
    # Direction from delta
    delta_sign = np.sign(df["delta"])
    
    # Signal: delta direction × liquidity confidence
    return (delta_sign * liq_regime.clip(0, 3) / 3).clip(-1, 1)


def signal_stacked_imbalance(df):
    """Component 6: Stacked Imbalance — consecutive bars of same-side dominance.
    Counts consecutive bars where buy_pct > threshold (or < 1-threshold).
    Spatial clustering = institutional footprint.
    """
    threshold = 0.55  # Bar is "buy dominant" if buy_pct > 55%
    
    buy_dom = (df["buy_pct"] > threshold).astype(int)
    sell_dom = (df["buy_pct"] < (1 - threshold)).astype(int)
    
    # Count consecutive bars of same dominance (rolling window of 5)
    buy_stack = buy_dom.rolling(5, min_periods=3).sum()  # 0-5 consecutive buy bars
    sell_stack = sell_dom.rolling(5, min_periods=3).sum()
    
    # Signal: net stacking
    signal = (buy_stack - sell_stack) / 5  # Normalize to [-1, 1]
    return signal


def signal_stacked_imbalance_contrarian(df):
    """Component 6b: Stacked Imbalance contrarian."""
    return -signal_stacked_imbalance(df)


def signal_delta_acceleration(df):
    """Component 7: CVD Acceleration (2nd derivative) — rate of change of buying."""
    cvd = df["delta"].cumsum()
    cvd_vel = cvd.diff()  # 1st derivative = delta itself
    cvd_acc = cvd_vel.diff()  # 2nd derivative
    
    # Z-score the acceleration
    acc_mu = cvd_acc.rolling(50, min_periods=20).mean()
    acc_sigma = cvd_acc.rolling(50, min_periods=20).std().clip(lower=1e-10)
    acc_z = (cvd_acc - acc_mu) / acc_sigma
    
    return acc_z.clip(-3, 3) / 3


def signal_volume_shock(df):
    """Component 8: Volume Shock — extreme volume bars as entry trigger.
    High volume bars with directional bias.
    """
    vol_mu = df["volume"].rolling(50, min_periods=20).mean()
    vol_sigma = df["volume"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    vol_z = (df["volume"] - vol_mu) / vol_sigma
    
    # Only trigger on high volume (z > 1.5)
    is_shock = (vol_z > 1.5).astype(float)
    
    # Direction from delta on shock bars
    delta_dir = np.sign(df["delta"])
    
    # Signal only on shock bars, hold direction
    raw = is_shock * delta_dir
    # Forward fill the signal for non-shock bars (hold position)
    return raw.replace(0, np.nan).ffill().fillna(0)


def signal_volume_shock_contrarian(df):
    """Component 8b: Volume Shock contrarian — fade the shock."""
    vol_mu = df["volume"].rolling(50, min_periods=20).mean()
    vol_sigma = df["volume"].rolling(50, min_periods=20).std().clip(lower=1e-10)
    vol_z = (df["volume"] - vol_mu) / vol_sigma
    
    is_shock = (vol_z > 1.5).astype(float)
    delta_dir = np.sign(df["delta"])
    
    raw = is_shock * (-delta_dir)  # FADE the direction
    return raw.replace(0, np.nan).ffill().fillna(0)


def signal_large_trade_ratio(df):
    """Component 9: Large Trade Ratio — are big trades buying or selling?
    Uses v2 data if available, otherwise approximates from dollar_volume/trade_count.
    """
    # Approximate large trade activity: dollar_volume / trade_count = avg trade size
    avg_size = df["dollar_volume"] / df["trade_count"].clip(lower=1)
    size_mu = avg_size.rolling(50, min_periods=20).mean()
    size_sigma = avg_size.rolling(50, min_periods=20).std().clip(lower=1e-10)
    size_z = (avg_size - size_mu) / size_sigma
    
    # When avg trade size is large AND delta is positive → institutional buying
    delta_dir = np.sign(df["delta"])
    
    # Only count when size is above normal
    large_bar = (size_z > 0.5).astype(float)
    
    return (large_bar * delta_dir * size_z.clip(0, 3) / 3).clip(-1, 1)


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE (reused from deep validation)
# ═══════════════════════════════════════════════════════════

def simulate_nonoverlap(df, signal_col, hold_bars=HOLD_BARS, cost_bps=COST_BPS):
    """Non-overlapping trade simulation."""
    signals = df[signal_col].values
    closes = df["close"].values
    n = len(df)
    trades = []
    i = 50
    
    while i < n - hold_bars:
        sig = signals[i]
        if np.isnan(sig) or sig == 0:
            i += 1
            continue
        
        pos = np.sign(sig)
        entry = closes[i]
        exit_p = closes[min(i + hold_bars, n - 1)]
        raw_ret = (exit_p / entry - 1) * pos
        net_ret = raw_ret - (cost_bps / 10000)
        
        trades.append({"position": int(pos), "raw_return": float(raw_ret), "net_return": float(net_ret)})
        i += hold_bars
    
    return trades


def walk_forward(df, signal_col, n_folds=14):
    """14-fold expanding-window WF with non-overlapping trades."""
    total = len(df)
    date_range = df.index[-1] - df.index[0]
    months = date_range.days / 30.44
    bpm = int(total / months)
    
    min_train = bpm * 3
    test_size = bpm
    available = total - min_train
    n_folds = min(n_folds, available // test_size)
    
    if n_folds < 3:
        return {"error": "Insufficient data", "n_folds": n_folds}
    
    all_trades = []
    fold_results = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = min(test_start + test_size, total)
        if test_end > total:
            break
        
        test_df = df.iloc[test_start:test_end]
        trades = simulate_nonoverlap(test_df, signal_col)
        
        if len(trades) < 2:
            continue
        
        net = [t["net_return"] for t in trades]
        tpy = 365 * 24 / HOLD_BARS
        fold_sharpe = (np.mean(net) / np.std(net)) * np.sqrt(tpy) if np.std(net) > 0 else 0
        
        fold_results.append({
            "fold": fold + 1,
            "n_trades": len(trades),
            "sharpe": float(fold_sharpe),
            "hit_rate": float(np.mean([r > 0 for r in net])),
        })
        all_trades.extend(trades)
    
    if len(all_trades) < 10:
        return {"error": "Too few trades", "n": len(all_trades)}
    
    net = np.array([t["net_return"] for t in all_trades])
    raw = np.array([t["raw_return"] for t in all_trades])
    
    t_stat, p_val = sp_stats.ttest_1samp(net, 0)
    tpy = 365 * 24 / HOLD_BARS
    sharpe = (np.mean(net) / np.std(net)) * np.sqrt(tpy) if np.std(net) > 0 else 0
    
    # Bootstrap CI
    bs_sharpes = []
    for _ in range(1000):
        s = np.random.choice(net, size=len(net), replace=True)
        bs = (np.mean(s) / np.std(s)) * np.sqrt(tpy) if np.std(s) > 0 else 0
        bs_sharpes.append(bs)
    ci_lo, ci_hi = np.percentile(bs_sharpes, [2.5, 97.5])
    
    pos_folds = sum(1 for f in fold_results if f["sharpe"] > 0)
    
    return {
        "n_folds": len(fold_results),
        "n_trades": len(all_trades),
        "mean_net_pct": float(np.mean(net) * 100),
        "mean_raw_pct": float(np.mean(raw) * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "sharpe": float(sharpe),
        "sharpe_ci": [float(ci_lo), float(ci_hi)],
        "hit_rate": float(np.mean(net > 0)),
        "total_return_pct": float(np.sum(net) * 100),
        "positive_folds": f"{pos_folds}/{len(fold_results)}",
        "profit_factor": float(abs(np.sum(net[net > 0]) / np.sum(net[net <= 0]))) if np.sum(net[net <= 0]) != 0 else float("inf"),
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # All signals to test
    SIGNALS = {
        # Confirmation variants (Virtuoso default direction)
        "cvd_regime": signal_cvd_regime,
        "trade_flow": signal_trade_flow,
        "imbalance": signal_imbalance,
        "pressure": signal_pressure,
        "liquidity": signal_liquidity,
        "stacked_imbalance": signal_stacked_imbalance,
        "delta_acceleration": signal_delta_acceleration,
        "volume_shock": signal_volume_shock,
        "large_trade_ratio": signal_large_trade_ratio,
        
        # Contrarian variants
        "cvd_contrarian": signal_cvd_contrarian,
        "trade_flow_contra": signal_trade_flow_contrarian,
        "imbalance_contra": signal_imbalance_contrarian,
        "pressure_contra": signal_pressure_contrarian,
        "stacked_imb_contra": signal_stacked_imbalance_contrarian,
        "vol_shock_contra": signal_volume_shock_contrarian,
    }
    
    # Find all orderflow assets
    assets = sorted([f.stem.replace("_1m", "")
                     for f in OF_DIR.glob("*_1m.csv")
                     if "v2" not in f.stem])
    
    print("=" * 100)
    print("SYSTEMATIC ORDERFLOW COMPONENT VALIDATION")
    print(f"Assets: {len(assets)} | Signals: {len(SIGNALS)} | Tests: {len(assets) * len(SIGNALS)}")
    print(f"Methodology: 14-fold WF, non-overlapping, {COST_BPS}bps costs, {HOLD_BARS}-bar hold @ 1h")
    print(f"Assets: {', '.join(assets)}")
    print("=" * 100)
    
    all_results = {}
    summary_rows = []
    
    for asset in assets:
        print(f"\n{'#' * 80}")
        print(f"# {asset.upper()}")
        print(f"{'#' * 80}")
        
        df = load_1h(asset)
        if df is None:
            print(f"  SKIP: no data")
            continue
        
        print(f"  {len(df):,} bars | {df.index[0]} → {df.index[-1]}")
        
        asset_results = {}
        
        for sig_name, sig_func in SIGNALS.items():
            try:
                df[sig_name] = sig_func(df)
            except Exception as e:
                print(f"  {sig_name}: ERROR computing signal: {e}")
                continue
            
            result = walk_forward(df, sig_name)
            asset_results[sig_name] = result
            
            if "error" in result:
                print(f"  {sig_name:<25s}: {result['error']}")
                summary_rows.append({
                    "asset": asset, "signal": sig_name, "sharpe": None,
                    "t_stat": None, "p_value": None, "verdict": "ERROR"
                })
                continue
            
            passes = result["p_value"] < 0.05 and result["sharpe"] > 0.5
            verdict = "✅ PASS" if passes else "❌ FAIL"
            
            print(f"  {sig_name:<25s}: Sharpe={result['sharpe']:+6.2f} "
                  f"[{result['sharpe_ci'][0]:+.1f},{result['sharpe_ci'][1]:+.1f}] "
                  f"t={result['t_stat']:+5.2f} p={result['p_value']:.4f} "
                  f"hit={result['hit_rate']:.0%} PF={result['profit_factor']:.2f} "
                  f"folds={result['positive_folds']} n={result['n_trades']} {verdict}")
            
            summary_rows.append({
                "asset": asset, "signal": sig_name,
                "sharpe": result["sharpe"], "t_stat": result["t_stat"],
                "p_value": result["p_value"], "hit_rate": result["hit_rate"],
                "n_trades": result["n_trades"], "profit_factor": result["profit_factor"],
                "positive_folds": result["positive_folds"],
                "total_return_pct": result["total_return_pct"],
                "verdict": "PASS" if passes else "FAIL"
            })
        
        all_results[asset] = asset_results
    
    # ═══════════════════════════════════════════════
    # SUMMARY TABLES
    # ═══════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print("SUMMARY BY SIGNAL (averaged across assets)")
    print(f"{'=' * 100}")
    
    sdf = pd.DataFrame(summary_rows)
    valid = sdf[sdf["sharpe"].notna()]
    
    if len(valid) > 0:
        sig_summary = valid.groupby("signal").agg(
            avg_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            avg_t=("t_stat", "mean"),
            pass_rate=("verdict", lambda x: (x == "PASS").mean()),
            n_assets=("asset", "count"),
            avg_pf=("profit_factor", "mean"),
            avg_hit=("hit_rate", "mean"),
        ).sort_values("avg_sharpe", ascending=False)
        
        print(f"\n{'Signal':<25s} {'Avg Sharpe':>10s} {'Med Sharpe':>10s} {'Avg t':>8s} {'Pass%':>7s} {'Avg PF':>8s} {'Avg Hit':>8s} {'Assets':>7s}")
        print("-" * 95)
        for sig, row in sig_summary.iterrows():
            marker = "⭐" if row["pass_rate"] > 0.3 else "🔸" if row["avg_sharpe"] > 0 else "❌"
            print(f"{sig:<25s} {row['avg_sharpe']:>+10.2f} {row['median_sharpe']:>+10.2f} "
                  f"{row['avg_t']:>+8.2f} {row['pass_rate']:>6.0%} {row['avg_pf']:>8.2f} "
                  f"{row['avg_hit']:>7.0%} {int(row['n_assets']):>7d} {marker}")
    
    # Best individual results
    print(f"\n\n{'=' * 100}")
    print("TOP 10 INDIVIDUAL RESULTS (by Sharpe)")
    print(f"{'=' * 100}")
    
    top10 = valid.nlargest(10, "sharpe")
    print(f"{'Asset':<12s} {'Signal':<25s} {'Sharpe':>8s} {'t-stat':>8s} {'p-value':>8s} {'Hit%':>6s} {'PF':>6s} {'Verdict'}")
    print("-" * 85)
    for _, row in top10.iterrows():
        v = "✅" if row["verdict"] == "PASS" else "❌"
        print(f"{row['asset']:<12s} {row['signal']:<25s} {row['sharpe']:>+8.2f} "
              f"{row['t_stat']:>+8.2f} {row['p_value']:>8.4f} {row['hit_rate']:>5.0%} "
              f"{row['profit_factor']:>6.2f} {v}")
    
    # Pass count
    n_pass = (valid["verdict"] == "PASS").sum()
    n_total = len(valid)
    print(f"\n\nOVERALL: {n_pass}/{n_total} tests pass ({n_pass/n_total*100:.1f}%)")
    
    # Multiple testing correction (Bonferroni)
    bonf_threshold = 0.05 / n_total
    bonf_pass = (valid["p_value"] < bonf_threshold).sum() & (valid["sharpe"] > 0.5)
    print(f"After Bonferroni correction (p < {bonf_threshold:.6f}): {bonf_pass} pass")
    
    # Save results
    out_json = RESULTS_DIR / "orderflow_components_validation_20260307.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON: {out_json}")
    
    # Save summary CSV
    out_csv = RESULTS_DIR / "orderflow_components_summary_20260307.csv"
    sdf.to_csv(out_csv, index=False)
    print(f"CSV: {out_csv}")
    
    # Save markdown
    out_md = RESULTS_DIR / "orderflow_components_validation_20260307.md"
    with open(out_md, "w") as f:
        f.write("# Systematic Orderflow Component Validation\n\n")
        f.write(f"**Date:** 2026-03-07\n")
        f.write(f"**Assets:** {len(assets)} ({', '.join(assets)})\n")
        f.write(f"**Signals:** {len(SIGNALS)} (9 confirmation + 6 contrarian)\n")
        f.write(f"**Total tests:** {n_total}\n")
        f.write(f"**Pass rate:** {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)\n\n")
        
        f.write("## Signal Rankings (averaged across assets)\n\n")
        f.write("| Signal | Avg Sharpe | Med Sharpe | Avg t | Pass% | Avg PF | Avg Hit |\n")
        f.write("|--------|-----------|-----------|-------|-------|--------|--------|\n")
        if len(valid) > 0:
            for sig, row in sig_summary.iterrows():
                f.write(f"| {sig} | {row['avg_sharpe']:+.2f} | {row['median_sharpe']:+.2f} | "
                        f"{row['avg_t']:+.2f} | {row['pass_rate']:.0%} | {row['avg_pf']:.2f} | {row['avg_hit']:.0%} |\n")
        
        f.write("\n## All Results\n\n")
        f.write("| Asset | Signal | Sharpe | t-stat | p-value | Hit% | PF | Verdict |\n")
        f.write("|-------|--------|--------|--------|---------|------|----|---------|\n")
        for _, row in sdf.iterrows():
            if row["sharpe"] is not None:
                f.write(f"| {row['asset']} | {row['signal']} | {row['sharpe']:+.2f} | "
                        f"{row['t_stat']:+.2f} | {row['p_value']:.4f} | {row.get('hit_rate', 0):.0%} | "
                        f"{row.get('profit_factor', 0):.2f} | {row['verdict']} |\n")
    
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()
