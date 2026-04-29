#!/usr/bin/env python3
"""
VPIN Directional Study v2 — Rolling Walk-Forward, Perp-Native
=============================================================
Fixes from v1:
1. Chart features computed from perp tick data (not spot candles)
2. Rolling walk-forward: 5 folds (train 6d, test 6d, expanding)
3. Fixed nan issue with constant-valued features
4. Forward AND backward OOS validation

Features tested:
- Buy/sell pressure (from tick data)
- Chart positioning (from perp 1-min bars → resampled to 1h/4h)
- Orderbook imbalance + spread (from L2 snapshots)
- Microstructure (large trades, trade acceleration)

Run: source ~/tick_collector/venv/bin/activate && python3 vpin_directional_v2.py
"""

import os, sys, glob, gc, warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
warnings.filterwarnings("ignore")

from backend.config.data_paths import TICK_TRADES, TICK_ORDERBOOK
DATA_DIR = str(TICK_TRADES)
OB_DIR = str(TICK_ORDERBOOK)

CONFIGS = {
    "BTCUSDT": {"bucket_size": 2000, "lookback": 100},
    "SOLUSDT": {"bucket_size": 200000, "lookback": 50},
}

FORWARD_BUCKETS = [10, 25, 50]
VPIN_PCTILE = 70
N_FOLDS = 5
MIN_TRAIN_DAYS = 6
FOLD_SIZE_DAYS = 6  # rolling test window


def aggregate_to_1min(symbol):
    """Load tick data → 1-min bars with buy/sell/large trade breakdown."""
    files = sorted(glob.glob(f"{DATA_DIR}/{symbol}/*.parquet"))
    if not files:
        return None
    all_bars = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "price", "size", "side"])
        df["minute"] = (df["timestamp"] // 60000) * 60000
        is_buy = df["side"] == "Buy"
        bars = df.groupby("minute").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("size", "sum"), n_trades=("size", "count"),
        )
        bars["buy_vol"] = df[is_buy].groupby("minute")["size"].sum()
        bars["sell_vol"] = df[~is_buy].groupby("minute")["size"].sum()
        p95 = df["size"].quantile(0.95)
        large = df[df["size"] >= p95]
        bars["large_buy"] = large[large["side"] == "Buy"].groupby(
            (large["timestamp"] // 60000) * 60000)["size"].sum()
        bars["large_sell"] = large[large["side"] == "Sell"].groupby(
            (large["timestamp"] // 60000) * 60000)["size"].sum()
        bars = bars.fillna(0)
        all_bars.append(bars)
        del df, large; gc.collect()
    result = pd.concat(all_bars).sort_index()
    result = result.groupby(level=0).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "n_trades": "sum", "buy_vol": "sum", "sell_vol": "sum",
        "large_buy": "sum", "large_sell": "sum",
    })
    print(f"  {symbol}: {len(result):,} 1-min bars")
    return result


def make_volume_buckets(bars_1m, bucket_size):
    """1-min bars → equal-volume buckets."""
    cumvol = bars_1m["volume"].values.cumsum()
    bid = (cumvol // bucket_size).astype(np.int64)
    tmp = pd.DataFrame({
        "bid": bid,
        "buy_vol": bars_1m["buy_vol"].values,
        "sell_vol": bars_1m["sell_vol"].values,
        "total_vol": bars_1m["volume"].values,
        "dollar_vol": bars_1m["volume"].values * bars_1m["close"].values,
        "close": bars_1m["close"].values,
        "high": bars_1m["high"].values,
        "low": bars_1m["low"].values,
        "n_trades": bars_1m["n_trades"].values,
        "large_buy": bars_1m["large_buy"].values,
        "large_sell": bars_1m["large_sell"].values,
        "ts": bars_1m.index.values,
    })
    g = tmp.groupby("bid", sort=True)
    b = pd.DataFrame({
        "buy_vol": g["buy_vol"].sum().values,
        "sell_vol": g["sell_vol"].sum().values,
        "total_vol": g["total_vol"].sum().values,
        "vwap": (g["dollar_vol"].sum() / g["total_vol"].sum()).values,
        "close": g["close"].last().values,
        "high": g["high"].max().values,
        "low": g["low"].min().values,
        "n_trades": g["n_trades"].sum().values,
        "large_buy": g["large_buy"].sum().values,
        "large_sell": g["large_sell"].sum().values,
        "time_start": g["ts"].first().values,
        "time_end": g["ts"].last().values,
    })
    b["oi"] = np.abs(b["buy_vol"] - b["sell_vol"]) / b["total_vol"]
    b["net_delta"] = b["buy_vol"] - b["sell_vol"]
    del tmp; gc.collect()
    return b


def resample_1min_to_bars(bars_1m, freq_min):
    """Resample 1-min bars to larger timeframe for chart features."""
    bars = bars_1m.copy()
    bars.index = pd.to_datetime(bars.index, unit="ms")
    rule = f"{freq_min}min"
    resampled = bars.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return resampled


def compute_chart_indicators(candles):
    """Compute technical indicators on OHLCV candles."""
    c = candles.copy()
    c["sma20"] = c["close"].rolling(20).mean()
    c["sma50"] = c["close"].rolling(50).mean()
    
    # RSI
    delta = c["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    c["rsi"] = 100 - (100 / (1 + rs))
    
    # ATR
    tr = pd.DataFrame({
        "hl": c["high"] - c["low"],
        "hc": (c["high"] - c["close"].shift(1)).abs(),
        "lc": (c["low"] - c["close"].shift(1)).abs(),
    }).max(axis=1)
    c["atr"] = tr.rolling(14).mean()
    
    # Range percentile
    roll_low = c["low"].rolling(20).min()
    roll_high = c["high"].rolling(20).max()
    c["range_pctile"] = (c["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan)
    
    # Trend strength
    c["dist_sma20"] = (c["close"] - c["sma20"]) / c["atr"].replace(0, np.nan)
    c["dist_sma50"] = (c["close"] - c["sma50"]) / c["atr"].replace(0, np.nan)
    c["sma20_slope"] = c["sma20"].diff(5) / c["atr"].replace(0, np.nan)
    
    # MACD normalized
    ema12 = c["close"].ewm(span=12).mean()
    ema26 = c["close"].ewm(span=26).mean()
    c["macd_norm"] = (ema12 - ema26) / c["atr"].replace(0, np.nan)
    
    # Trend binary
    c["above_sma20"] = (c["close"] > c["sma20"]).astype(float)
    c["above_sma50"] = (c["close"] > c["sma50"]).astype(float)
    
    # Bollinger position
    bb_std = c["close"].rolling(20).std()
    c["bb_position"] = (c["close"] - c["sma20"]) / (2 * bb_std).replace(0, np.nan)
    
    return c


def add_all_features(buckets, bars_1m, symbol):
    """Add all feature categories to buckets."""
    b = buckets
    
    # === PRESSURE FEATURES (from tick data) ===
    b["buy_ratio"] = b["buy_vol"] / b["total_vol"]
    b["delta_ma5"] = b["net_delta"].rolling(5).mean()
    b["delta_ma20"] = b["net_delta"].rolling(20).mean()
    b["delta_momentum"] = b["delta_ma5"] - b["delta_ma20"]
    # Normalize delta momentum
    b["delta_mom_z"] = (b["delta_momentum"] - b["delta_momentum"].rolling(50).mean()) / \
                        b["delta_momentum"].rolling(50).std().replace(0, np.nan)
    
    # CVD slope
    b["cvd"] = b["net_delta"].cumsum()
    b["cvd_slope"] = b["cvd"].diff(10) / 10  # simple slope over 10 buckets
    b["cvd_slope_z"] = (b["cvd_slope"] - b["cvd_slope"].rolling(50).mean()) / \
                        b["cvd_slope"].rolling(50).std().replace(0, np.nan)
    
    # Large trade bias
    large_total = (b["large_buy"] + b["large_sell"]).replace(0, np.nan)
    b["large_ratio"] = b["large_buy"] / large_total
    b["large_ratio_ma10"] = b["large_ratio"].rolling(10).mean()
    b["large_net_z"] = ((b["large_buy"] - b["large_sell"]).rolling(10).mean())
    lnz_std = b["large_net_z"].rolling(50).std().replace(0, np.nan)
    b["large_net_z"] = (b["large_net_z"] - b["large_net_z"].rolling(50).mean()) / lnz_std
    
    # Trade acceleration
    b["trades_ma5"] = b["n_trades"].rolling(5).mean()
    b["trades_ma20"] = b["n_trades"].rolling(20).mean()
    b["trade_accel"] = b["trades_ma5"] / b["trades_ma20"].replace(0, np.nan)
    
    # === CHART FEATURES (from perp 1-min bars resampled to 1h) ===
    for tf_min, tf_label in [(60, "1h"), (240, "4h")]:
        candles = resample_1min_to_bars(bars_1m, tf_min)
        indicators = compute_chart_indicators(candles)
        
        # Convert indicator index back to ms for alignment
        ind_ts = indicators.index.astype(np.int64) // 10**6
        bucket_ts = b["time_start"].values
        idx = np.searchsorted(ind_ts.values, bucket_ts, side="right") - 1
        idx = np.clip(idx, 0, len(indicators) - 1)
        
        for col in ["rsi", "range_pctile", "dist_sma20", "dist_sma50",
                     "sma20_slope", "macd_norm", "above_sma20", "above_sma50", "bb_position"]:
            if col in indicators.columns:
                b[f"chart_{tf_label}_{col}"] = indicators[col].values[idx]
    
    # === ORDERBOOK FEATURES (from L2 snapshots, memory-efficient) ===
    ob_files = sorted(glob.glob(f"{OB_DIR}/{symbol}/*.parquet"))
    if ob_files:
        all_ob = []
        for f in ob_files:
            try:
                ob = pd.read_parquet(f, columns=["timestamp", "spread", "imbalance"])
                ob["minute"] = (ob["timestamp"] // 60000) * 60000
                ob_min = ob.groupby("minute").agg(
                    ob_imbalance=("imbalance", "mean"),
                    ob_spread=("spread", "mean"),
                )
                all_ob.append(ob_min)
                del ob; gc.collect()
            except:
                continue
        if all_ob:
            ob_all = pd.concat(all_ob).sort_index().groupby(level=0).mean()
            ob_ts = ob_all.index.values
            idx = np.searchsorted(ob_ts, bucket_ts, side="right") - 1
            idx = np.clip(idx, 0, len(ob_all) - 1)
            b["ob_imbalance"] = ob_all["ob_imbalance"].values[idx]
            b["ob_spread"] = ob_all["ob_spread"].values[idx]
            b["ob_spread_z"] = (b["ob_spread"] - b["ob_spread"].rolling(100).mean()) / \
                                b["ob_spread"].rolling(100).std().replace(0, np.nan)
            b["ob_imb_ma5"] = b["ob_imbalance"].rolling(5).mean()
            b["ob_imb_ma20"] = b["ob_imbalance"].rolling(20).mean()
            # Imbalance centered (0.5 = neutral)
            b["ob_imb_centered"] = b["ob_imb_ma5"] - 0.5
            del ob_all, all_ob; gc.collect()
            print(f"    Orderbook features added")
    
    return b


def get_feature_lists(buckets):
    """Return categorized feature lists based on available columns."""
    pressure = [c for c in ["buy_ratio", "delta_mom_z", "cvd_slope_z",
                            "large_ratio_ma10", "large_net_z", "trade_accel"]
                if c in buckets.columns]
    
    chart_1h = [c for c in buckets.columns if c.startswith("chart_1h_") and "above_" not in c]
    chart_4h = [c for c in buckets.columns if c.startswith("chart_4h_") and "above_" not in c]
    chart_trend = [c for c in buckets.columns if "above_sma" in c]
    
    ob = [c for c in ["ob_imbalance", "ob_spread_z", "ob_imb_ma5",
                       "ob_imb_ma20", "ob_imb_centered"]
          if c in buckets.columns]
    
    return {
        "pressure": pressure,
        "chart_1h": chart_1h,
        "chart_4h": chart_4h,
        "chart_trend": chart_trend,
        "orderbook": ob,
    }


def evaluate_feature(feat_vals, fwd_ret, feature_name):
    """Test if feature predicts forward return direction."""
    valid = ~(np.isnan(feat_vals) | np.isnan(fwd_ret))
    feat = feat_vals[valid]
    ret = fwd_ret[valid]
    n = len(feat)
    
    if n < 30:
        return {"n": n, "acc": np.nan, "rho": np.nan, "p": np.nan, "mean_bull": np.nan, "mean_bear": np.nan}
    
    # Check for constant feature (causes nan correlation)
    if feat.std() < 1e-10:
        return {"n": n, "acc": np.nan, "rho": np.nan, "p": np.nan, "mean_bull": np.nan, "mean_bear": np.nan}
    
    corr, pval = stats.spearmanr(feat, ret)
    
    # Direction prediction
    if "above_" in feature_name or "range_pctile" in feature_name:
        pred_up = feat > 0.5
    elif "imbalance" in feature_name and "centered" not in feature_name:
        pred_up = feat > 0.5
    else:
        med = np.median(feat)
        pred_up = feat > med
    
    if pred_up.sum() == 0 or pred_up.sum() == n:
        return {"n": n, "acc": np.nan, "rho": corr, "p": pval, "mean_bull": np.nan, "mean_bear": np.nan}
    
    correct = (pred_up & (ret > 0)) | (~pred_up & (ret < 0))
    acc = correct.mean()
    mean_bull = ret[pred_up].mean()
    mean_bear = ret[~pred_up].mean()
    
    return {"n": n, "acc": round(acc, 4), "rho": round(corr, 4), "p": pval,
            "mean_bull": mean_bull, "mean_bear": mean_bear}


def evaluate_combo(buckets, features, fwd_ret, high_vpin_mask, fold_mask):
    """Z-score combo of features → predict direction."""
    mask = high_vpin_mask & fold_mask
    for f in features:
        mask = mask & buckets[f].notna()
    mask = mask & ~np.isnan(fwd_ret)
    
    if mask.sum() < 30:
        return {"n": mask.sum(), "acc": np.nan, "rho": np.nan, "p": np.nan}
    
    # Z-score each feature within the fold
    z_sum = np.zeros(mask.sum())
    for f in features:
        vals = buckets.loc[mask, f].values
        std = vals.std()
        if std < 1e-10:
            continue
        z_sum += (vals - vals.mean()) / std
    
    ret = fwd_ret[mask]
    if np.std(z_sum) < 1e-10:
        return {"n": mask.sum(), "acc": np.nan, "rho": np.nan, "p": np.nan}
    
    corr, pval = stats.spearmanr(z_sum, ret)
    pred_up = z_sum > 0
    if pred_up.sum() == 0 or pred_up.sum() == len(pred_up):
        return {"n": mask.sum(), "acc": np.nan, "rho": corr, "p": pval}
    
    correct = (pred_up & (ret > 0)) | (~pred_up & (ret < 0))
    return {"n": mask.sum(), "acc": round(correct.mean(), 4), "rho": round(corr, 4), "p": pval}


def rolling_walk_forward(buckets, all_features, feature_cats, fwd_buckets):
    """Rolling walk-forward with expanding IS window."""
    dates = sorted(buckets["date"].unique())
    n_dates = len(dates)
    
    results = []
    
    # Create folds: expanding train, fixed test
    folds = []
    test_start = MIN_TRAIN_DAYS
    while test_start + FOLD_SIZE_DAYS <= n_dates:
        train_dates = dates[:test_start]
        test_dates = dates[test_start:test_start + FOLD_SIZE_DAYS]
        folds.append((train_dates, test_dates))
        test_start += FOLD_SIZE_DAYS
    
    # Also add backward test (train on later, test on earlier) for robustness
    # Reverse: last N days = train, earlier = test
    if n_dates >= MIN_TRAIN_DAYS + FOLD_SIZE_DAYS:
        bwd_train = dates[-(MIN_TRAIN_DAYS + FOLD_SIZE_DAYS):]
        bwd_test = dates[:FOLD_SIZE_DAYS]
        folds.append((bwd_train, bwd_test))
    
    print(f"  {len(folds)} folds ({len(folds)-1} forward + 1 backward)")
    
    for fold_i, (train_dates, test_dates) in enumerate(folds):
        is_backward = fold_i == len(folds) - 1
        fold_label = f"bwd" if is_backward else f"fwd{fold_i+1}"
        
        train_mask = buckets["date"].isin(set(train_dates))
        test_mask = buckets["date"].isin(set(test_dates))
        
        # VPIN threshold from train
        vpin_train = buckets.loc[train_mask & buckets["vpin"].notna(), "vpin"]
        if len(vpin_train) < 50:
            continue
        vpin_thresh = vpin_train.quantile(VPIN_PCTILE / 100)
        high_vpin = buckets["vpin"] >= vpin_thresh
        
        n_test_high = (high_vpin & test_mask).sum()
        if n_test_high < 20:
            continue
        
        for fwd_k in fwd_buckets:
            fwd_ret = (buckets["vwap"].shift(-fwd_k) / buckets["vwap"] - 1).values
            
            # Individual features
            for feat in all_features:
                if feat not in buckets.columns:
                    continue
                
                # Test set evaluation
                test_high = high_vpin & test_mask
                feat_vals = buckets[feat].values.copy()
                test_vals = np.where(test_high, feat_vals, np.nan)
                test_ret = np.where(test_high, fwd_ret, np.nan)
                
                # Only evaluate on valid test points
                test_idx = test_high.values
                if test_idx.sum() < 20:
                    continue
                
                res = evaluate_feature(feat_vals[test_idx], fwd_ret[test_idx], feat)
                
                # Also train eval for overfit check
                train_idx = (high_vpin & train_mask).values
                train_res = evaluate_feature(feat_vals[train_idx], fwd_ret[train_idx], feat) if train_idx.sum() > 20 else {"acc": np.nan, "rho": np.nan}
                
                results.append({
                    "fold": fold_label,
                    "type": "individual",
                    "feature": feat,
                    "fwd_k": fwd_k,
                    "train_n": train_idx.sum(),
                    "train_acc": train_res.get("acc", np.nan),
                    "train_rho": train_res.get("rho", np.nan),
                    "test_n": res["n"],
                    "test_acc": res["acc"],
                    "test_rho": res["rho"],
                    "test_p": res["p"],
                    "mean_bull": res.get("mean_bull", np.nan),
                    "mean_bear": res.get("mean_bear", np.nan),
                })
    
    return pd.DataFrame(results)


def run():
    print("=" * 70)
    print("VPIN DIRECTIONAL v2 — Rolling WF, Perp-Native")
    print(f"Folds: expanding IS ({MIN_TRAIN_DAYS}d min) + {FOLD_SIZE_DAYS}d test + backward")
    print("=" * 70)
    
    all_results = []
    
    for symbol, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")
        
        print("  Loading trades...")
        bars = aggregate_to_1min(symbol)
        if bars is None:
            continue
        
        print("  Building buckets...")
        buckets = make_volume_buckets(bars, cfg["bucket_size"])
        print(f"  {len(buckets):,} buckets")
        
        # VPIN
        buckets["vpin"] = buckets["oi"].rolling(cfg["lookback"], min_periods=cfg["lookback"]).mean()
        
        print("  Computing features (perp-native)...")
        buckets = add_all_features(buckets, bars, symbol)
        del bars; gc.collect()
        
        buckets["date"] = pd.to_datetime(buckets["time_start"], unit="ms").dt.date
        
        # Get all features
        feat_cats = get_feature_lists(buckets)
        all_feats = []
        for cat, feats in feat_cats.items():
            all_feats.extend(feats)
            print(f"    {cat}: {len(feats)} features")
        print(f"    Total: {len(all_feats)} features")
        
        # Rolling walk-forward
        print(f"\n  Running rolling walk-forward...")
        df = rolling_walk_forward(buckets, all_feats, feat_cats, FORWARD_BUCKETS)
        df["symbol"] = symbol
        all_results.append(df)
        
        del buckets; gc.collect()
    
    # Combine
    results = pd.concat(all_results, ignore_index=True)
    
    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("RESULTS — Individual Features")
    print("=" * 70)
    
    ind = results[results["type"] == "individual"].copy()
    valid = ind.dropna(subset=["test_acc"])
    n = len(valid)
    
    print(f"Total tests: {n}")
    print(f"Test accuracy > 55%: {(valid['test_acc'] > 0.55).sum()} ({100*(valid['test_acc'] > 0.55).mean():.1f}%)")
    print(f"Test accuracy > 60%: {(valid['test_acc'] > 0.60).sum()}")
    bonf = 0.05 / max(n, 1)
    n_sig = (valid["test_p"] < 0.05).sum()
    n_bonf = (valid["test_p"] < bonf).sum()
    print(f"Significant (p<0.05): {n_sig}")
    print(f"Bonferroni (p<{bonf:.6f}): {n_bonf}")
    
    # Feature ranking: mean test accuracy across ALL folds and horizons
    print(f"\n{'─'*70}")
    print("FEATURE RANKING (mean test accuracy, all folds)")
    print(f"{'─'*70}")
    
    rank = valid.groupby(["feature"]).agg(
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_rho=("test_rho", "mean"),
        n_sig=("test_p", lambda x: (x < 0.05).sum()),
        n_tests=("test_p", "count"),
        n_above55=("test_acc", lambda x: (x > 0.55).sum()),
    ).sort_values("mean_acc", ascending=False)
    
    for feat, r in rank.iterrows():
        sig_str = f"{int(r['n_sig'])}/{int(r['n_tests'])}"
        print(f"  {feat:35s}  acc={r['mean_acc']:.1%} +/-{r['std_acc']:.1%}  "
              f"rho={r['mean_rho']:+.3f}  sig={sig_str:>5s}  >55%={int(r['n_above55'])}")
    
    # Per-symbol breakdown
    for sym in CONFIGS:
        print(f"\n{'─'*70}")
        print(f"  {sym} — Top features by mean accuracy")
        print(f"{'─'*70}")
        sym_data = valid[valid["symbol"] == sym]
        sym_rank = sym_data.groupby("feature").agg(
            mean_acc=("test_acc", "mean"),
            mean_rho=("test_rho", "mean"),
            n_sig=("test_p", lambda x: (x < 0.05).sum()),
            n_tests=("test_p", "count"),
        ).sort_values("mean_acc", ascending=False).head(15)
        
        for feat, r in sym_rank.iterrows():
            print(f"    {feat:35s}  acc={r['mean_acc']:.1%}  rho={r['mean_rho']:+.3f}  sig={int(r['n_sig'])}/{int(r['n_tests'])}")
    
    # Forward vs Backward comparison
    print(f"\n{'─'*70}")
    print("FORWARD vs BACKWARD OOS (robustness check)")
    print(f"{'─'*70}")
    
    for direction in ["fwd", "bwd"]:
        mask = valid["fold"].str.startswith(direction)
        sub = valid[mask]
        if len(sub) > 0:
            print(f"\n  {direction.upper()}: {len(sub)} tests, "
                  f"mean acc={sub['test_acc'].mean():.1%}, "
                  f">55%={( sub['test_acc'] > 0.55).sum()}, "
                  f"sig={( sub['test_p'] < 0.05).sum()}")
    
    # Consistency check: features that work in BOTH forward and backward
    print(f"\n{'─'*70}")
    print("CONSISTENT FEATURES (>55% in both fwd AND bwd)")
    print(f"{'─'*70}")
    
    for feat in rank.index:
        fwd_data = valid[(valid["feature"] == feat) & valid["fold"].str.startswith("fwd")]
        bwd_data = valid[(valid["feature"] == feat) & valid["fold"].str.startswith("bwd")]
        if len(fwd_data) > 0 and len(bwd_data) > 0:
            fwd_acc = fwd_data["test_acc"].mean()
            bwd_acc = bwd_data["test_acc"].mean()
            if fwd_acc > 0.55 and bwd_acc > 0.55:
                print(f"  {feat:35s}  fwd={fwd_acc:.1%}  bwd={bwd_acc:.1%}  CONSISTENT")
            elif fwd_acc > 0.55 or bwd_acc > 0.55:
                print(f"  {feat:35s}  fwd={fwd_acc:.1%}  bwd={bwd_acc:.1%}  partial")
    
    # Best individual results
    print(f"\n{'─'*70}")
    print("TOP 20 INDIVIDUAL TEST RESULTS (by accuracy)")
    print(f"{'─'*70}")
    
    top = valid.nlargest(20, "test_acc")
    for _, r in top.iterrows():
        print(f"  {r['symbol']:10s} {r['fold']:5s} {r['feature']:30s} fwd={int(r['fwd_k']):3d}: "
              f"train={r['train_acc']:.1%} test={r['test_acc']:.1%} rho={r['test_rho']:+.3f} (p={r['test_p']:.2e})")
    
    # Bull vs bear return spread
    print(f"\n{'─'*70}")
    print("RETURN SPREAD (mean return when feature bullish vs bearish)")
    print(f"{'─'*70}")
    
    spread = valid.groupby("feature").agg(
        mean_bull=("mean_bull", "mean"),
        mean_bear=("mean_bear", "mean"),
    )
    spread["spread"] = spread["mean_bull"] - spread["mean_bear"]
    spread = spread.sort_values("spread", ascending=False)
    for feat, r in spread.head(15).iterrows():
        print(f"  {feat:35s}  bull={r['mean_bull']:+.4%}  bear={r['mean_bear']:+.4%}  spread={r['spread']:+.4%}")
    
    # Save
    out = os.path.expanduser("~/Desktop/maestro/backend/research/vpin_results/vpin_directional_v2_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run()
