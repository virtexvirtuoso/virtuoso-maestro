#!/usr/bin/env python3
"""
VPIN Directional Study — Can we predict WHERE the big move goes?
================================================================
VPIN tells us WHEN a big move is coming (validated: rho=0.37, 70 Bonferroni).
This study tests what additional signals predict the DIRECTION.

Approach:
- At each high-VPIN moment, capture contextual features
- Test which features (alone + combined) predict forward return SIGN
- Walk-forward: IS=20 days, OOS=10 days

Feature Categories:
1. BUY/SELL PRESSURE (from tick data — already in VPIN buckets)
   - Net delta (buy_vol - sell_vol) 
   - Delta momentum (change in net delta over recent buckets)
   - Buy ratio (buy_vol / total_vol)
   
2. CHART POSITIONING (from spot candles)
   - Price vs SMA20/50/200 (trend context)
   - RSI (overbought/oversold)
   - Price percentile in recent range (near high vs low)
   - ATR-normalized distance from MA (how extended)
   
3. ORDERBOOK IMBALANCE (from L2 snapshots)
   - Bid/ask imbalance at time of high VPIN
   - Spread (wide spread = uncertainty)

4. MICROSTRUCTURE (from tick data)
   - Trade size trend (are large trades buying or selling?)
   - Trade frequency acceleration

Run on VPS: source ~/tick_collector/venv/bin/activate && python3 vpin_directional_study.py
"""

import os, sys, glob, gc, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

from backend.config.data_paths import TICK_TRADES, TICK_ORDERBOOK, MAESTRO_DATA_ROOT
DATA_DIR = str(TICK_TRADES)
OB_DIR = str(TICK_ORDERBOOK)
CANDLE_DIR = str(MAESTRO_DATA_ROOT)  # legacy spot CSVs (if any)

# BTC config (best from VPIN study)
CONFIGS = {
    "BTCUSDT": {"bucket_size": 2000, "lookback": 100, "candle_prefix": "BTC"},
    "SOLUSDT": {"bucket_size": 200000, "lookback": 50, "candle_prefix": "SOL"},
}

IS_DAYS = 20
FORWARD_BUCKETS = [10, 25, 50]  # predict direction over these horizons
VPIN_THRESHOLD_PCTILE = 70  # only look at high-VPIN moments (top 30%)


def aggregate_to_1min(symbol):
    """Load tick data day-by-day, aggregate to 1-min bars."""
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
        
        # Large trade features (trades > 95th percentile size)
        p95 = df["size"].quantile(0.95)
        large = df[df["size"] >= p95]
        bars["large_buy_vol"] = large[large["side"] == "Buy"].groupby(
            (large["timestamp"] // 60000) * 60000)["size"].sum()
        bars["large_sell_vol"] = large[large["side"] == "Sell"].groupby(
            (large["timestamp"] // 60000) * 60000)["size"].sum()
        
        bars = bars.fillna(0)
        all_bars.append(bars)
        del df, large
        gc.collect()
    
    result = pd.concat(all_bars).sort_index()
    result = result.groupby(level=0).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "n_trades": "sum", "buy_vol": "sum", "sell_vol": "sum",
        "large_buy_vol": "sum", "large_sell_vol": "sum",
    })
    print(f"  {symbol}: {len(result):,} 1-min bars")
    return result


def make_volume_buckets(bars_1m, bucket_size):
    """Convert 1-min bars into volume buckets with rich features."""
    cumvol = bars_1m["volume"].values.cumsum()
    bucket_id = (cumvol // bucket_size).astype(np.int64)
    
    tmp = pd.DataFrame({
        "bid": bucket_id,
        "buy_vol": bars_1m["buy_vol"].values,
        "sell_vol": bars_1m["sell_vol"].values,
        "total_vol": bars_1m["volume"].values,
        "dollar_vol": bars_1m["volume"].values * bars_1m["close"].values,
        "close": bars_1m["close"].values,
        "high": bars_1m["high"].values,
        "low": bars_1m["low"].values,
        "n_trades": bars_1m["n_trades"].values,
        "large_buy": bars_1m["large_buy_vol"].values,
        "large_sell": bars_1m["large_sell_vol"].values,
        "ts": bars_1m.index.values,
    })
    
    g = tmp.groupby("bid", sort=True)
    buckets = pd.DataFrame({
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
    
    # Core VPIN
    buckets["oi"] = np.abs(buckets["buy_vol"] - buckets["sell_vol"]) / buckets["total_vol"]
    buckets["net_delta"] = buckets["buy_vol"] - buckets["sell_vol"]
    
    del tmp
    gc.collect()
    return buckets


def add_pressure_features(buckets, lookback):
    """Buy/sell pressure features from tick data."""
    b = buckets.copy()
    
    # VPIN
    b["vpin"] = b["oi"].rolling(lookback, min_periods=lookback).mean()
    
    # 1. Net delta (raw buy-sell pressure)
    b["buy_ratio"] = b["buy_vol"] / b["total_vol"]
    
    # 2. Delta momentum — is buying accelerating?
    b["delta_ma5"] = b["net_delta"].rolling(5).mean()
    b["delta_ma20"] = b["net_delta"].rolling(20).mean()
    b["delta_momentum"] = b["delta_ma5"] - b["delta_ma20"]
    
    # 3. Cumulative delta (CVD) direction
    b["cvd"] = b["net_delta"].cumsum()
    b["cvd_slope"] = b["cvd"].rolling(20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=False
    )
    
    # 4. Large trade bias — are whales buying or selling?
    b["large_net"] = b["large_buy"] - b["large_sell"]
    b["large_ratio"] = b["large_buy"] / (b["large_buy"] + b["large_sell"]).replace(0, np.nan)
    b["large_ratio_ma10"] = b["large_ratio"].rolling(10).mean()
    
    # 5. Trade intensity (acceleration)
    b["trades_ma5"] = b["n_trades"].rolling(5).mean()
    b["trades_ma20"] = b["n_trades"].rolling(20).mean()
    b["trade_accel"] = b["trades_ma5"] / b["trades_ma20"]
    
    return b


def add_chart_features(buckets, symbol, candle_prefix):
    """Technical/chart positioning from spot candles."""
    # Load 1h candles and align to bucket timestamps
    candle_file = os.path.join(CANDLE_DIR, f"{candle_prefix}_spot_1h.csv")
    if not os.path.exists(candle_file):
        print(f"    No 1h candles for {candle_prefix}")
        return buckets
    
    candles = pd.read_csv(candle_file, parse_dates=["Date"])
    candles["ts"] = candles["Date"].astype(np.int64) // 10**6  # to ms
    candles = candles.sort_values("ts").set_index("ts")
    
    # Compute indicators on 1h candles
    candles["sma20"] = candles["Close"].rolling(20).mean()
    candles["sma50"] = candles["Close"].rolling(50).mean()
    candles["sma200"] = candles["Close"].rolling(200).mean()
    
    # RSI
    delta = candles["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    candles["rsi"] = 100 - (100 / (1 + rs))
    
    # ATR
    tr = pd.DataFrame({
        "hl": candles["High"] - candles["Low"],
        "hc": (candles["High"] - candles["Close"].shift(1)).abs(),
        "lc": (candles["Low"] - candles["Close"].shift(1)).abs(),
    }).max(axis=1)
    candles["atr"] = tr.rolling(14).mean()
    
    # Price position in 20-bar range
    candles["range_pctile"] = (
        (candles["Close"] - candles["Low"].rolling(20).min()) /
        (candles["High"].rolling(20).max() - candles["Low"].rolling(20).min()).replace(0, np.nan)
    )
    
    # Distance from SMA (ATR-normalized)
    candles["dist_sma20_atr"] = (candles["Close"] - candles["sma20"]) / candles["atr"]
    candles["dist_sma50_atr"] = (candles["Close"] - candles["sma50"]) / candles["atr"]
    
    # Trend strength (SMA20 slope normalized by ATR)
    candles["sma20_slope"] = candles["sma20"].diff(5) / candles["atr"]
    
    # EMA 12/26 for MACD-like momentum
    candles["ema12"] = candles["Close"].ewm(span=12).mean()
    candles["ema26"] = candles["Close"].ewm(span=26).mean()
    candles["macd_norm"] = (candles["ema12"] - candles["ema26"]) / candles["atr"]
    
    # Map candle features to buckets using time_start
    # Find the most recent 1h candle for each bucket
    candle_ts = candles.index.values
    bucket_ts = buckets["time_start"].values
    
    # For each bucket, find the last candle_ts <= bucket_ts
    idx = np.searchsorted(candle_ts, bucket_ts, side="right") - 1
    idx = np.clip(idx, 0, len(candles) - 1)
    
    feature_cols = ["rsi", "range_pctile", "dist_sma20_atr", "dist_sma50_atr",
                    "sma20_slope", "macd_norm"]
    
    # Trend context: above/below MAs
    candles["above_sma20"] = (candles["Close"] > candles["sma20"]).astype(float)
    candles["above_sma50"] = (candles["Close"] > candles["sma50"]).astype(float)
    candles["above_sma200"] = (candles["Close"] > candles["sma200"]).astype(float)
    feature_cols += ["above_sma20", "above_sma50", "above_sma200"]
    
    for col in feature_cols:
        buckets[f"chart_{col}"] = candles[col].values[idx]
    
    print(f"    Chart features added ({len(feature_cols)} indicators)")
    return buckets


def add_orderbook_features(buckets, symbol):
    """L2 orderbook imbalance aligned to bucket timestamps."""
    ob_files = sorted(glob.glob(f"{OB_DIR}/{symbol}/*.parquet"))
    if not ob_files:
        print(f"    No orderbook data for {symbol}")
        return buckets
    
    # Load orderbook day-by-day, extract summary per minute to save memory
    all_ob = []
    for f in ob_files:
        try:
            ob = pd.read_parquet(f, columns=["timestamp", "mid_price", "spread", "imbalance"])
            ob["minute"] = (ob["timestamp"] // 60000) * 60000
            ob_min = ob.groupby("minute").agg(
                ob_imbalance=("imbalance", "mean"),
                ob_spread=("spread", "mean"),
            )
            all_ob.append(ob_min)
            del ob
            gc.collect()
        except Exception as e:
            print(f"    Error loading {f}: {e}")
            continue
    
    if not all_ob:
        return buckets
    
    ob_all = pd.concat(all_ob).sort_index()
    ob_all = ob_all.groupby(level=0).mean()
    
    # Map to buckets
    ob_ts = ob_all.index.values
    bucket_ts = buckets["time_start"].values
    idx = np.searchsorted(ob_ts, bucket_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ob_all) - 1)
    
    buckets["ob_imbalance"] = ob_all["ob_imbalance"].values[idx]
    buckets["ob_spread"] = ob_all["ob_spread"].values[idx]
    
    # Normalized spread (z-score)
    buckets["ob_spread_z"] = (
        (buckets["ob_spread"] - buckets["ob_spread"].rolling(100).mean()) /
        buckets["ob_spread"].rolling(100).std()
    )
    
    # Imbalance momentum
    buckets["ob_imbalance_ma5"] = buckets["ob_imbalance"].rolling(5).mean()
    buckets["ob_imbalance_ma20"] = buckets["ob_imbalance"].rolling(20).mean()
    
    print(f"    Orderbook features added (imbalance, spread)")
    del ob_all, all_ob
    gc.collect()
    return buckets


def evaluate_directional(buckets, feature_name, fwd_k, vpin_threshold, is_mask):
    """
    Test: when VPIN is high, does this feature predict forward return direction?
    Returns IS and OOS accuracy + stats.
    """
    fwd_ret = buckets["vwap"].shift(-fwd_k) / buckets["vwap"] - 1
    high_vpin = buckets["vpin"] >= vpin_threshold
    
    valid = high_vpin & buckets[feature_name].notna() & fwd_ret.notna()
    
    results = {}
    for label, mask in [("is", valid & is_mask), ("oos", valid & ~is_mask)]:
        if mask.sum() < 30:
            results[label] = {"n": mask.sum(), "accuracy": np.nan, "spearman": np.nan, "p": np.nan}
            continue
        
        feat = buckets.loc[mask, feature_name].values
        ret = fwd_ret[mask].values
        
        # Spearman correlation: feature vs forward return
        corr, pval = stats.spearmanr(feat, ret)
        
        # Directional accuracy: positive feature → positive return
        if feature_name.startswith("chart_above") or feature_name == "chart_range_pctile":
            # For binary/range features: high = bullish
            pred_up = feat > 0.5
        elif "imbalance" in feature_name:
            # Book imbalance > 0.5 = more bids = bullish
            pred_up = feat > 0.5
        else:
            # For continuous features: positive = bullish
            pred_up = feat > np.median(feat)
        
        correct = (pred_up & (ret > 0)) | (~pred_up & (ret < 0))
        accuracy = correct.mean()
        
        # Mean return when feature is bullish vs bearish
        mean_ret_bull = ret[pred_up].mean() if pred_up.sum() > 0 else np.nan
        mean_ret_bear = ret[~pred_up].mean() if (~pred_up).sum() > 0 else np.nan
        
        results[label] = {
            "n": mask.sum(),
            "accuracy": round(accuracy, 4),
            "spearman": round(corr, 4),
            "p": pval,
            "mean_ret_bull": mean_ret_bull,
            "mean_ret_bear": mean_ret_bear,
        }
    
    return results


def evaluate_combined(buckets, features, fwd_k, vpin_threshold, is_mask):
    """
    Combine multiple features into a composite score and test direction.
    Simple approach: z-score each feature, sum them, test if composite predicts direction.
    """
    fwd_ret = buckets["vwap"].shift(-fwd_k) / buckets["vwap"] - 1
    high_vpin = buckets["vpin"] >= vpin_threshold
    
    # Z-score each feature
    z_scores = pd.DataFrame()
    for feat in features:
        col = buckets[feat]
        z = (col - col.rolling(100).mean()) / col.rolling(100).std()
        z_scores[feat] = z
    
    composite = z_scores.mean(axis=1)
    valid = high_vpin & composite.notna() & fwd_ret.notna()
    
    results = {}
    for label, mask in [("is", valid & is_mask), ("oos", valid & ~is_mask)]:
        if mask.sum() < 30:
            results[label] = {"n": mask.sum(), "accuracy": np.nan, "spearman": np.nan, "p": np.nan}
            continue
        
        comp = composite[mask].values
        ret = fwd_ret[mask].values
        
        corr, pval = stats.spearmanr(comp, ret)
        pred_up = comp > 0
        correct = (pred_up & (ret > 0)) | (~pred_up & (ret < 0))
        accuracy = correct.mean()
        
        results[label] = {
            "n": mask.sum(),
            "accuracy": round(accuracy, 4),
            "spearman": round(corr, 4),
            "p": pval,
        }
    
    return results


def run():
    print("=" * 70)
    print("VPIN DIRECTIONAL STUDY")
    print("Can we predict WHERE the big move goes?")
    print(f"IS: {IS_DAYS} days, OOS: remaining")
    print("=" * 70)
    
    all_results = []
    combo_results = []
    
    for symbol, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {symbol} (bucket={cfg['bucket_size']}, lb={cfg['lookback']})")
        print(f"{'='*60}")
        
        # 1. Build VPIN buckets with pressure features
        print("  Loading trades...")
        bars = aggregate_to_1min(symbol)
        if bars is None:
            continue
        
        print("  Building volume buckets...")
        buckets = make_volume_buckets(bars, cfg["bucket_size"])
        print(f"  {len(buckets):,} buckets")
        del bars; gc.collect()
        
        print("  Adding pressure features...")
        buckets = add_pressure_features(buckets, cfg["lookback"])
        
        print("  Adding chart features...")
        buckets = add_chart_features(buckets, symbol, cfg["candle_prefix"])
        
        print("  Adding orderbook features...")
        buckets = add_orderbook_features(buckets, symbol)
        
        # IS/OOS split
        buckets["date"] = pd.to_datetime(buckets["time_start"], unit="ms").dt.date
        dates = sorted(buckets["date"].unique())
        is_cutoff = dates[IS_DAYS - 1]
        is_mask = buckets["date"] <= is_cutoff
        
        # VPIN threshold (from IS data)
        vpin_valid = buckets.loc[is_mask & buckets["vpin"].notna(), "vpin"]
        vpin_thresh = vpin_valid.quantile(VPIN_THRESHOLD_PCTILE / 100)
        n_high = (buckets["vpin"] >= vpin_thresh).sum()
        print(f"\n  VPIN threshold (p{VPIN_THRESHOLD_PCTILE}): {vpin_thresh:.4f}")
        print(f"  High-VPIN buckets: {n_high}")
        
        # Feature list
        pressure_feats = ["buy_ratio", "delta_momentum", "cvd_slope",
                         "large_ratio_ma10", "trade_accel"]
        chart_feats = [c for c in buckets.columns if c.startswith("chart_")]
        ob_feats = [c for c in buckets.columns if c.startswith("ob_") and c != "ob_spread"]
        
        all_feats = pressure_feats + chart_feats + ob_feats
        
        print(f"\n  Testing {len(all_feats)} features x {len(FORWARD_BUCKETS)} horizons...")
        print(f"  {'─'*55}")
        
        # Test each feature individually
        for feat in all_feats:
            if feat not in buckets.columns:
                continue
            for fwd_k in FORWARD_BUCKETS:
                res = evaluate_directional(buckets, feat, fwd_k, vpin_thresh, is_mask)
                
                row = {
                    "symbol": symbol,
                    "feature": feat,
                    "fwd_k": fwd_k,
                    "type": "individual",
                    "is_n": res["is"]["n"],
                    "is_acc": res["is"]["accuracy"],
                    "is_rho": res["is"]["spearman"],
                    "is_p": res["is"]["p"],
                    "oos_n": res["oos"]["n"],
                    "oos_acc": res["oos"]["accuracy"],
                    "oos_rho": res["oos"]["spearman"],
                    "oos_p": res["oos"]["p"],
                }
                all_results.append(row)
                
                # Print notable
                oos = res["oos"]
                if oos["accuracy"] is not np.nan and not np.isnan(oos.get("accuracy", np.nan)):
                    if oos["accuracy"] > 0.55 or (oos["p"] is not None and not np.isnan(oos["p"]) and oos["p"] < 0.05):
                        tag = "**" if oos["accuracy"] > 0.58 else "* "
                        print(f"  {tag} {feat:30s} fwd={fwd_k:3d}: "
                              f"IS acc={res['is']['accuracy']:.1%} rho={res['is']['spearman']:+.3f} | "
                              f"OOS acc={oos['accuracy']:.1%} rho={oos['spearman']:+.3f} (p={oos['p']:.4f})")
        
        # Test combinations
        print(f"\n  Testing feature combinations...")
        print(f"  {'─'*55}")
        
        # Best pressure + best chart + orderbook
        combos = {
            "pressure_all": [f for f in pressure_feats if f in buckets.columns],
            "chart_trend": [f for f in ["chart_above_sma20", "chart_above_sma50", "chart_sma20_slope", "chart_macd_norm"] if f in buckets.columns],
            "chart_mean_rev": [f for f in ["chart_rsi", "chart_range_pctile", "chart_dist_sma20_atr"] if f in buckets.columns],
            "pressure+trend": [f for f in ["buy_ratio", "delta_momentum", "cvd_slope", "chart_above_sma50", "chart_macd_norm"] if f in buckets.columns],
            "pressure+ob": [f for f in ["buy_ratio", "delta_momentum", "ob_imbalance_ma5"] if f in buckets.columns],
            "all_bullish": [f for f in ["buy_ratio", "delta_momentum", "cvd_slope", "large_ratio_ma10",
                                        "chart_above_sma50", "chart_macd_norm", "chart_sma20_slope",
                                        "ob_imbalance_ma5"] if f in buckets.columns],
            "whale+trend": [f for f in ["large_ratio_ma10", "chart_above_sma50", "chart_macd_norm"] if f in buckets.columns],
        }
        
        for combo_name, features in combos.items():
            if len(features) < 2:
                continue
            for fwd_k in FORWARD_BUCKETS:
                res = evaluate_combined(buckets, features, fwd_k, vpin_thresh, is_mask)
                
                row = {
                    "symbol": symbol,
                    "feature": combo_name,
                    "fwd_k": fwd_k,
                    "type": "combo",
                    "n_features": len(features),
                    "is_n": res["is"]["n"],
                    "is_acc": res["is"]["accuracy"],
                    "is_rho": res["is"]["spearman"],
                    "is_p": res["is"]["p"],
                    "oos_n": res["oos"]["n"],
                    "oos_acc": res["oos"]["accuracy"],
                    "oos_rho": res["oos"]["spearman"],
                    "oos_p": res["oos"]["p"],
                }
                combo_results.append(row)
                
                oos = res["oos"]
                if oos["accuracy"] is not np.nan and not np.isnan(oos.get("accuracy", np.nan)):
                    tag = "**" if oos["accuracy"] > 0.58 else "* " if oos["accuracy"] > 0.55 else "  "
                    print(f"  {tag} {combo_name:30s} fwd={fwd_k:3d} ({len(features)}F): "
                          f"IS acc={res['is']['accuracy']:.1%} rho={res['is']['spearman']:+.3f} | "
                          f"OOS acc={oos['accuracy']:.1%} rho={oos['spearman']:+.3f} (p={oos['p']:.4f})")
        
        del buckets; gc.collect()
    
    # === SUMMARY ===
    df = pd.DataFrame(all_results)
    df_combo = pd.DataFrame(combo_results)
    df_all = pd.concat([df, df_combo], ignore_index=True)
    
    print("\n" + "=" * 70)
    print("SUMMARY — Individual Features")
    print("=" * 70)
    
    valid = df.dropna(subset=["oos_acc"])
    n = len(valid)
    print(f"Total tests: {n}")
    print(f"OOS accuracy > 55%: {(valid['oos_acc'] > 0.55).sum()} ({100*(valid['oos_acc'] > 0.55).mean():.1f}%)")
    print(f"OOS accuracy > 58%: {(valid['oos_acc'] > 0.58).sum()}")
    print(f"OOS accuracy > 60%: {(valid['oos_acc'] > 0.60).sum()}")
    print(f"OOS rho significant (p<0.05): {(valid['oos_p'] < 0.05).sum()}")
    bonf = 0.05 / max(n, 1)
    print(f"Bonferroni (p<{bonf:.5f}): {(valid['oos_p'] < bonf).sum()}")
    
    if (valid['oos_p'] < 0.05).sum() > 0:
        print("\nTop individual features (by OOS p-value):")
        top = valid[valid['oos_p'] < 0.05].nsmallest(15, "oos_p")
        for _, r in top.iterrows():
            print(f"  {r['symbol']:10s} {r['feature']:30s} fwd={int(r['fwd_k']):3d}: "
                  f"OOS acc={r['oos_acc']:.1%} rho={r['oos_rho']:+.3f} (p={r['oos_p']:.2e})")
    
    print("\nTop by OOS accuracy (>55%):")
    top_acc = valid[valid['oos_acc'] > 0.55].nlargest(15, "oos_acc")
    for _, r in top_acc.iterrows():
        print(f"  {r['symbol']:10s} {r['feature']:30s} fwd={int(r['fwd_k']):3d}: "
              f"acc={r['oos_acc']:.1%} rho={r['oos_rho']:+.3f} (p={r['oos_p']:.2e})")
    
    if len(df_combo) > 0:
        print("\n" + "=" * 70)
        print("SUMMARY — Feature Combinations")
        print("=" * 70)
        vc = df_combo.dropna(subset=["oos_acc"])
        print(f"Total combos: {len(vc)}")
        print(f"OOS accuracy > 55%: {(vc['oos_acc'] > 0.55).sum()}")
        print(f"OOS accuracy > 58%: {(vc['oos_acc'] > 0.58).sum()}")
        
        print("\nAll combos (sorted by OOS accuracy):")
        for _, r in vc.nlargest(20, "oos_acc").iterrows():
            print(f"  {r['symbol']:10s} {r['feature']:30s} fwd={int(r['fwd_k']):3d}: "
                  f"acc={r['oos_acc']:.1%} rho={r['oos_rho']:+.3f} (p={r['oos_p']:.4f})")
    
    # Feature importance ranking
    print("\n" + "=" * 70)
    print("FEATURE RANKING (mean OOS accuracy across all horizons)")
    print("=" * 70)
    feat_rank = valid.groupby("feature").agg(
        mean_acc=("oos_acc", "mean"),
        mean_rho=("oos_rho", "mean"),
        n_sig=("oos_p", lambda x: (x < 0.05).sum()),
        n_tests=("oos_p", "count"),
    ).sort_values("mean_acc", ascending=False)
    
    for feat, r in feat_rank.iterrows():
        print(f"  {feat:35s}: acc={r['mean_acc']:.1%}  rho={r['mean_rho']:+.3f}  sig={int(r['n_sig'])}/{int(r['n_tests'])}")
    
    # Save
    out = os.path.expanduser("~/Desktop/maestro/backend/research/vpin_results/vpin_directional_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_all.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run()
