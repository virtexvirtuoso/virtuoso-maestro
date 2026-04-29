"""
Multi-Timeframe Directional Validation
=======================================
Tests ALL indicator signals across ALL available timeframes:
  5m, 15m, 1h, 4h, 1d

Key hypothesis: signal might exist at daily but not intraday
(aligns with V4 Honest System which works on daily, OOS Sharpe 1.52)

Uses the best signals from each category + classic trend following.
Assets: all available per timeframe (up to 44 for daily)

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
import talib

from backend.config.data_paths import BARS_1M_V1

OF_DIR = BARS_1M_V1
OHLCV_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/ohlcv")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/research")
COST_BPS = 10


def load_ohlcv(filepath):
    """Load any OHLCV file."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    for col in ["timestamp", "date", "datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    df = df.sort_index()
    # Ensure numeric
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def load_of_resampled(asset, tf):
    """Load orderflow data resampled to target TF."""
    path = OF_DIR / f"{asset}_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "trade_count": "sum", "buy_vol": "sum", 
           "sell_vol": "sum", "delta": "sum"}
    df = df.resample(tf).agg(agg).dropna(subset=["open"])
    df["buy_pct"] = df["buy_vol"] / df["volume"].clip(lower=1e-10)
    return df


def walk_forward(df, signal_col, hold_bars=10, n_folds=14, cost_bps=COST_BPS):
    """Generic WF engine."""
    total = len(df)
    if total < 200:
        return {"error": "Too few bars", "n": total}
    
    date_range = (df.index[-1] - df.index[0]).days
    if date_range < 180:
        return {"error": "Less than 6 months"}
    
    months = max(date_range / 30.44, 1)
    bpm = int(total / months)
    min_train = max(bpm * 3, 60)  # At least 60 bars for training
    test_size = max(bpm, 20)  # At least 20 bars per fold
    
    available = total - min_train
    n_folds = min(n_folds, max(available // test_size, 0))
    if n_folds < 3:
        return {"error": "Insufficient data for WF"}
    
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
            exit_idx = min(i + hold_bars, total - 1)
            raw = (closes[exit_idx] / closes[i] - 1) * pos
            net = raw - (cost_bps / 10000)
            all_trades.append(net)
            i += hold_bars
    
    if len(all_trades) < 10:
        return {"error": "Too few trades", "n": len(all_trades)}
    
    net = np.array(all_trades)
    t_stat, p_val = sp_stats.ttest_1samp(net, 0)
    
    # Annualization: estimate trades per year
    trades_per_fold = len(all_trades) / n_folds
    folds_per_year = 365.25 / (date_range / n_folds) if date_range > 0 else 1
    tpy = max(trades_per_fold * folds_per_year, 1)
    
    sharpe = (np.mean(net) / np.std(net)) * np.sqrt(tpy) if np.std(net) > 0 else 0
    
    # Bootstrap CI
    bs = []
    for _ in range(500):
        s = np.random.choice(net, size=len(net), replace=True)
        bs.append((np.mean(s) / np.std(s)) * np.sqrt(tpy) if np.std(s) > 0 else 0)
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
    
    pf = abs(np.sum(net[net > 0]) / np.sum(net[net <= 0])) if np.sum(net[net <= 0]) != 0 else float("inf")
    
    return {
        "n_trades": len(all_trades), "mean_pct": float(np.mean(net) * 100),
        "t_stat": float(t_stat), "p_value": float(p_val),
        "sharpe": float(sharpe), "sharpe_ci": [float(ci_lo), float(ci_hi)],
        "hit_rate": float(np.mean(net > 0)), "pf": float(pf),
        "total_ret_pct": float(np.sum(net) * 100),
    }


# ═══════════════════════════════════════════════════════════
# SIGNALS — best from each category + classic trend following
# ═══════════════════════════════════════════════════════════

def sig_sma_cross(df, fast=20, slow=50):
    sma_f = df["close"].rolling(fast, min_periods=fast//2).mean()
    sma_s = df["close"].rolling(slow, min_periods=slow//2).mean()
    return (sma_f > sma_s).astype(float) * 2 - 1

def sig_sma_cross_10_30(df):
    return sig_sma_cross(df, 10, 30)

def sig_sma_cross_50_200(df):
    return sig_sma_cross(df, 50, 200)

def sig_ema_cross(df, fast=12, slow=26):
    ema_f = df["close"].ewm(span=fast).mean()
    ema_s = df["close"].ewm(span=slow).mean()
    return (ema_f > ema_s).astype(float) * 2 - 1

def sig_macd(df):
    macd, signal, hist = talib.MACD(df["close"])
    return np.sign(hist).fillna(0)

def sig_rsi_momentum(df):
    rsi = talib.RSI(df["close"], timeperiod=14)
    return ((rsi - 50) / 50).clip(-1, 1)

def sig_rsi_reversion(df):
    rsi = talib.RSI(df["close"], timeperiod=14)
    s = pd.Series(0.0, index=df.index)
    s[rsi < 30] = 1.0; s[rsi > 70] = -1.0
    return s.replace(0, np.nan).ffill().fillna(0)

def sig_adx_trend(df):
    adx = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    plus_di = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=14)
    minus_di = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=14)
    direction = np.sign(plus_di - minus_di)
    strength = (adx / 50).clip(0, 1)
    return (direction * strength).fillna(0)

def sig_donchian(df, period=20):
    high_n = df["high"].rolling(period).max().shift(1)
    low_n = df["low"].rolling(period).min().shift(1)
    return ((df["close"] > high_n).astype(float) - (df["close"] < low_n).astype(float))

def sig_obv_trend(df):
    obv = talib.OBV(df["close"], df["volume"])
    sma20 = obv.rolling(20).mean()
    sma50 = obv.rolling(50).mean()
    return (sma20 > sma50).astype(float) * 2 - 1

def sig_cmf(df, period=20):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).clip(lower=1e-10)
    mfv = mfm * df["volume"]
    return (mfv.rolling(period).sum() / df["volume"].rolling(period).sum().clip(lower=1e-10)).clip(-1, 1)

def sig_bbands_revert(df):
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    pos = (df["close"] - lower) / (upper - lower).clip(lower=1e-10)
    return (-(pos - 0.5) * 2).clip(-1, 1).fillna(0)

def sig_trend_position(df):
    s20 = df["close"].rolling(20, min_periods=10).mean()
    s50 = df["close"].rolling(50, min_periods=25).mean()
    s200 = df["close"].rolling(200, min_periods=100).mean()
    above = ((df["close"] > s20).astype(float) + (df["close"] > s50).astype(float) + 
             (df["close"] > s200).astype(float)) / 3
    return (above - 0.5) * 2

def sig_structure_break(df, lookback=20):
    high_n = df["high"].rolling(lookback).max().shift(1)
    low_n = df["low"].rolling(lookback).min().shift(1)
    return ((df["close"] > high_n).astype(float) - (df["close"] < low_n).astype(float)).replace(0, np.nan).ffill().fillna(0)

def sig_atr_breakout(df):
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    atr_sma = atr.rolling(50).mean()
    expanding = (atr > atr_sma * 1.2).astype(float)
    direction = np.sign(df["close"].diff(5))
    return (expanding * direction).replace(0, np.nan).ffill().fillna(0)

def sig_momentum(df, period=10):
    """Pure momentum — past N-bar return as signal."""
    ret = df["close"].pct_change(period)
    return np.sign(ret).fillna(0)

def sig_momentum_20(df):
    return sig_momentum(df, 20)

def sig_tsmom(df, lookback=20):
    """Time-series momentum — sign of past return, vol-scaled."""
    ret = df["close"].pct_change(lookback)
    vol = df["close"].pct_change().rolling(lookback).std().clip(lower=1e-10)
    return (ret / vol).clip(-3, 3) / 3

def sig_mean_reversion(df, period=20):
    sma = df["close"].rolling(period).mean()
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    z = (df["close"] - sma) / atr.clip(lower=1e-10)
    s = pd.Series(0.0, index=df.index)
    s[z > 2] = -1.0; s[z < -2] = 1.0
    return s.replace(0, np.nan).ffill().fillna(0)

def sig_cci(df):
    cci = talib.CCI(df["high"], df["low"], df["close"], timeperiod=20)
    return (cci / 200).clip(-1, 1)

def sig_williams_r(df):
    wr = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=14)
    s = pd.Series(0.0, index=df.index)
    s[wr < -80] = 1.0; s[wr > -20] = -1.0
    return s.replace(0, np.nan).ffill().fillna(0)


SIGNALS = {
    # Trend following
    "sma_20_50": sig_sma_cross,
    "sma_10_30": sig_sma_cross_10_30,
    "sma_50_200": sig_sma_cross_50_200,
    "ema_12_26": sig_ema_cross,
    "macd": sig_macd,
    "adx_trend": sig_adx_trend,
    "donchian_20": sig_donchian,
    "trend_position": sig_trend_position,
    "structure_break": sig_structure_break,
    "atr_breakout": sig_atr_breakout,
    
    # Momentum
    "momentum_10": sig_momentum,
    "momentum_20": sig_momentum_20,
    "tsmom_20": sig_tsmom,
    
    # Mean reversion
    "rsi_reversion": sig_rsi_reversion,
    "rsi_momentum": sig_rsi_momentum,
    "bbands_revert": sig_bbands_revert,
    "mean_reversion": sig_mean_reversion,
    "williams_r": sig_williams_r,
    "cci": sig_cci,
    
    # Volume
    "obv_trend": sig_obv_trend,
    "cmf": sig_cmf,
}


def main():
    # Define timeframes and their hold periods + data sources
    TF_CONFIG = {
        "5m": {"suffix": "5m", "holds": [12, 36, 72], "hold_labels": ["1h", "3h", "6h"]},    # 12×5m=1h
        "15m": {"suffix": "15m", "holds": [4, 16, 48], "hold_labels": ["1h", "4h", "12h"]},
        "1h": {"suffix": "1h", "holds": [4, 12, 24], "hold_labels": ["4h", "12h", "24h"]},
        "4h": {"suffix": "4h", "holds": [3, 6, 18], "hold_labels": ["12h", "24h", "3d"]},
        "1d": {"suffix": "1d", "holds": [3, 7, 14], "hold_labels": ["3d", "1w", "2w"]},
    }
    
    # Find all available assets per TF
    def find_assets(tf_suffix):
        files = list(OHLCV_DIR.glob(f"*_{tf_suffix}.csv"))
        assets = []
        for f in files:
            name = f.stem.replace(f"_{tf_suffix}", "")
            if name.startswith("kucoin_") or "_btc" in name:
                continue  # Skip non-USDT pairs
            assets.append((name, f))
        return sorted(assets)
    
    print("=" * 120)
    print("MULTI-TIMEFRAME DIRECTIONAL VALIDATION")
    print(f"Signals: {len(SIGNALS)} | Timeframes: {len(TF_CONFIG)}")
    print(f"Methodology: 14-fold WF, non-overlapping, {COST_BPS}bps costs")
    print("=" * 120)
    
    all_rows = []
    
    for tf, config in TF_CONFIG.items():
        assets = find_assets(config["suffix"])
        
        print(f"\n{'█' * 120}")
        print(f"█ TIMEFRAME: {tf} | {len(assets)} assets | holds: {config['hold_labels']}")
        print(f"{'█' * 120}")
        
        for asset_name, asset_path in assets:
            df = load_ohlcv(asset_path)
            if df is None or len(df) < 200:
                continue
            
            # Ensure we have volume
            if "volume" not in df.columns or df["volume"].sum() == 0:
                continue
            
            for sig_name, sig_func in SIGNALS.items():
                try:
                    df[sig_name] = sig_func(df)
                except Exception:
                    continue
                
                for hold, hold_label in zip(config["holds"], config["hold_labels"]):
                    wf = walk_forward(df, sig_name, hold_bars=hold)
                    
                    passes = not ("error" in wf) and wf.get("p_value", 1) < 0.05 and wf.get("sharpe", 0) > 0.5
                    
                    row = {"tf": tf, "asset": asset_name, "signal": sig_name,
                           "hold_bars": hold, "hold_label": hold_label}
                    
                    if "error" not in wf:
                        row.update(wf)
                        row["verdict"] = "PASS" if passes else "FAIL"
                    else:
                        row["verdict"] = "ERROR"
                    
                    all_rows.append(row)
                    
                    if passes:
                        print(f"  ✅ {asset_name} {sig_name} h={hold_label}: "
                              f"Sharpe={wf['sharpe']:+.2f} t={wf['t_stat']:.2f} p={wf['p_value']:.4f} "
                              f"n={wf['n_trades']} hit={wf['hit_rate']:.0%}")
    
    # ═══════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════
    sdf = pd.DataFrame(all_rows)
    valid = sdf[sdf["verdict"].isin(["PASS", "FAIL"])]
    n_pass = (valid["verdict"] == "PASS").sum()
    n_total = len(valid)
    expected_fp = n_total * 0.05
    
    print(f"\n\n{'=' * 120}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 120}")
    print(f"Total tests: {n_total}")
    print(f"Passes: {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"Expected false positives at 5%: {expected_fp:.0f}")
    print(f"Pass rate vs random: {'ABOVE ⭐' if n_pass > expected_fp * 1.5 else 'AT/BELOW random'}")
    
    bonf = 0.05 / n_total
    bonf_pass = sum(1 for _, r in valid.iterrows()
                    if r.get("p_value", 1) < bonf and r.get("sharpe", 0) > 0.5)
    print(f"Bonferroni corrected (p < {bonf:.6f}): {bonf_pass}")
    
    # By timeframe
    print(f"\n--- BY TIMEFRAME ---")
    for tf in ["5m", "15m", "1h", "4h", "1d"]:
        tf_df = valid[valid["tf"] == tf]
        tp = (tf_df["verdict"] == "PASS").sum()
        tt = len(tf_df)
        if tt > 0:
            efp = tt * 0.05
            marker = "⭐" if tp > efp * 1.5 else ""
            print(f"  {tf:>4s}: {tp:>4d}/{tt:>5d} ({tp/tt*100:>5.1f}%) — expected FP: {efp:.0f} {marker}")
    
    # By signal (averaged across all TFs and assets)
    print(f"\n--- TOP 20 SIGNALS (by avg Sharpe across all TFs/assets) ---")
    sig_avg = valid.groupby("signal").agg(
        avg_sharpe=("sharpe", "mean"),
        med_sharpe=("sharpe", "median"),
        pass_rate=("verdict", lambda x: (x == "PASS").mean()),
        n=("asset", "count"),
        avg_hit=("hit_rate", "mean"),
    ).sort_values("avg_sharpe", ascending=False)
    
    print(f"  {'Signal':<25s} {'Avg Sharpe':>10s} {'Med':>8s} {'Pass%':>7s} {'Hit%':>6s} {'N':>5s}")
    print(f"  {'-'*65}")
    for sig, row in sig_avg.head(20).iterrows():
        marker = "⭐" if row["pass_rate"] > 0.10 else "🔸" if row["avg_sharpe"] > 0.3 else ""
        print(f"  {sig:<25s} {row['avg_sharpe']:>+10.2f} {row['med_sharpe']:>+8.2f} "
              f"{row['pass_rate']:>6.0%} {row['avg_hit']:>5.0%} {int(row['n']):>5d} {marker}")
    
    # By signal × timeframe
    print(f"\n--- SIGNAL × TIMEFRAME HEATMAP (avg Sharpe) ---")
    if len(valid) > 0:
        pivot = valid.pivot_table(index="signal", columns="tf", values="sharpe", aggfunc="mean")
        # Reorder columns
        tf_order = [t for t in ["5m", "15m", "1h", "4h", "1d"] if t in pivot.columns]
        pivot = pivot[tf_order]
        
        # Show top signals
        pivot["avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg", ascending=False)
        
        print(f"  {'Signal':<25s}", end="")
        for tf in tf_order:
            print(f" {tf:>8s}", end="")
        print(f" {'AVG':>8s}")
        print(f"  {'-'*75}")
        for sig, row in pivot.head(21).iterrows():
            print(f"  {sig:<25s}", end="")
            for tf in tf_order:
                val = row.get(tf, np.nan)
                if np.isnan(val):
                    print(f" {'N/A':>8s}", end="")
                else:
                    marker = "⭐" if val > 1.0 else "🔸" if val > 0.5 else "  "
                    print(f" {val:>+6.2f}{marker}", end="")
            print(f" {row['avg']:>+8.2f}")
    
    # All passes
    if n_pass > 0:
        print(f"\n--- ALL PASSING TESTS (sorted by Sharpe) ---")
        passes_df = valid[valid["verdict"] == "PASS"].sort_values("sharpe", ascending=False)
        print(f"  {'TF':>4s} {'Asset':<20s} {'Signal':<25s} {'Hold':>6s} {'Sharpe':>8s} {'t':>6s} "
              f"{'p':>8s} {'Hit%':>6s} {'PF':>6s} {'N':>5s}")
        print(f"  {'-'*100}")
        for _, r in passes_df.head(50).iterrows():
            print(f"  {r['tf']:>4s} {r['asset']:<20s} {r['signal']:<25s} {r.get('hold_label',''):>6s} "
                  f"{r['sharpe']:>+8.2f} {r['t_stat']:>+6.2f} {r['p_value']:>8.4f} "
                  f"{r['hit_rate']:>5.0%} {r['pf']:>6.2f} {r['n_trades']:>5d}")
    
    # Save
    out_json = RESULTS_DIR / "multitf_direction_validation_20260307.json"
    sdf.to_json(out_json, orient="records", indent=2)
    
    out_csv = RESULTS_DIR / "multitf_direction_validation_20260307.csv"
    sdf.to_csv(out_csv, index=False)
    
    print(f"\nJSON: {out_json}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
