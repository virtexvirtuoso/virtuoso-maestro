#!/usr/bin/env python3
"""
VPIN Local Study — BTC + SOL from per-tick data
====================================================
Adapted from vpin_study.py + vpin_directional_v2.py to run locally.
Data paths resolved via backend.config.data_paths (single source of truth).

Walk-forward: IS=25 days, OOS=remaining (~13 days for BTC/SOL)
Tests:
  1. VPIN → forward volatility (Spearman ρ, quintile ratio)
  2. VPIN → directional accuracy (high VPIN + net buy/sell → price direction)
  3. Feature ranking (pressure, chart, orderbook features)
  4. Rolling WF with expanding IS + backward validation

Usage: python3 vpin_local_study.py
"""

import os, sys, glob, gc, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

# === PATHS (central registry) ===
from backend.config.data_paths import TICK_TRADES, TICK_ORDERBOOK
DATA_DIR = str(TICK_TRADES)
OB_DIR = str(TICK_ORDERBOOK)
OUT_DIR = os.path.expanduser("~/Desktop/maestro/backend/research/vpin_results")

CONFIGS = {
    "BTCUSDT": {
        "bucket_sizes": [100, 500, 2000],
        "lookbacks": [25, 50, 100],
        "dir_bucket": 2000,
        "dir_lookback": 100,
    },
    "SOLUSDT": {
        "bucket_sizes": [10000, 50000, 200000],
        "lookbacks": [25, 50, 100],
        "dir_bucket": 200000,
        "dir_lookback": 50,
    },
}

FORWARD_WINDOWS = [10, 25, 50]
IS_DAYS = 25
VPIN_PCTILE = 70
MIN_TRAIN_DAYS = 6
FOLD_SIZE_DAYS = 6


# ──────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────

def aggregate_to_1min(symbol):
    """Load tick data day-by-day → 1-min bars with buy/sell/large trade breakdown."""
    files = sorted(glob.glob(f"{DATA_DIR}/{symbol}/*.parquet"))
    if not files:
        print(f"  {symbol}: no trade files found")
        return None

    all_bars = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "price", "size", "side"])
        df["minute"] = (df["timestamp"] // 60000) * 60000
        is_buy = df["side"] == "Buy"

        bars = df.groupby("minute").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            n_trades=("size", "count"),
        )
        bars["buy_vol"] = df[is_buy].groupby("minute")["size"].sum()
        bars["sell_vol"] = df[~is_buy].groupby("minute")["size"].sum()

        # Large trades (>= p95)
        p95 = df["size"].quantile(0.95)
        large = df[df["size"] >= p95]
        bars["large_buy"] = large[large["side"] == "Buy"].groupby(
            (large["timestamp"] // 60000) * 60000)["size"].sum()
        bars["large_sell"] = large[large["side"] == "Sell"].groupby(
            (large["timestamp"] // 60000) * 60000)["size"].sum()

        bars = bars.fillna(0)
        all_bars.append(bars)
        del df, large
        gc.collect()

    result = pd.concat(all_bars).sort_index()
    result = result.groupby(level=0).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "n_trades": "sum", "buy_vol": "sum", "sell_vol": "sum",
        "large_buy": "sum", "large_sell": "sum",
    })

    day0 = files[0].split("/")[-1].replace(".parquet", "")
    dayN = files[-1].split("/")[-1].replace(".parquet", "")
    print(f"  {symbol}: {len(result):,} 1-min bars ({day0} → {dayN})")
    return result


# ──────────────────────────────────────────────────────
# VPIN CORE
# ──────────────────────────────────────────────────────

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
    del tmp
    gc.collect()
    return b


def compute_vpin(buckets, lookback):
    return buckets["oi"].rolling(lookback, min_periods=lookback).mean()


def compute_forward_vol(buckets, k):
    log_ret = np.log(buckets["vwap"] / buckets["vwap"].shift(1))
    return log_ret.rolling(k, min_periods=k).std().shift(-k)


def compute_forward_ret(buckets, k):
    return buckets["vwap"].shift(-k) / buckets["vwap"] - 1


# ──────────────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────────────

def resample_1min_to_bars(bars_1m, freq_min):
    bars = bars_1m.copy()
    bars.index = pd.to_datetime(bars.index, unit="ms")
    resampled = bars.resample(f"{freq_min}min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return resampled


def compute_chart_indicators(candles):
    c = candles.copy()
    c["sma20"] = c["close"].rolling(20).mean()
    c["sma50"] = c["close"].rolling(50).mean()

    delta = c["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    c["rsi"] = 100 - (100 / (1 + rs))

    tr = pd.DataFrame({
        "hl": c["high"] - c["low"],
        "hc": (c["high"] - c["close"].shift(1)).abs(),
        "lc": (c["low"] - c["close"].shift(1)).abs(),
    }).max(axis=1)
    c["atr"] = tr.rolling(14).mean()

    roll_low = c["low"].rolling(20).min()
    roll_high = c["high"].rolling(20).max()
    c["range_pctile"] = (c["close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan)

    c["dist_sma20"] = (c["close"] - c["sma20"]) / c["atr"].replace(0, np.nan)
    c["dist_sma50"] = (c["close"] - c["sma50"]) / c["atr"].replace(0, np.nan)
    c["sma20_slope"] = c["sma20"].diff(5) / c["atr"].replace(0, np.nan)

    ema12 = c["close"].ewm(span=12).mean()
    ema26 = c["close"].ewm(span=26).mean()
    c["macd_norm"] = (ema12 - ema26) / c["atr"].replace(0, np.nan)

    c["above_sma20"] = (c["close"] > c["sma20"]).astype(float)
    c["above_sma50"] = (c["close"] > c["sma50"]).astype(float)

    bb_std = c["close"].rolling(20).std()
    c["bb_position"] = (c["close"] - c["sma20"]) / (2 * bb_std).replace(0, np.nan)

    return c


def add_all_features(buckets, bars_1m, symbol):
    b = buckets

    # Pressure features
    b["buy_ratio"] = b["buy_vol"] / b["total_vol"]
    b["delta_ma5"] = b["net_delta"].rolling(5).mean()
    b["delta_ma20"] = b["net_delta"].rolling(20).mean()
    b["delta_momentum"] = b["delta_ma5"] - b["delta_ma20"]
    b["delta_mom_z"] = (b["delta_momentum"] - b["delta_momentum"].rolling(50).mean()) / \
                        b["delta_momentum"].rolling(50).std().replace(0, np.nan)

    b["cvd"] = b["net_delta"].cumsum()
    b["cvd_slope"] = b["cvd"].diff(10) / 10
    b["cvd_slope_z"] = (b["cvd_slope"] - b["cvd_slope"].rolling(50).mean()) / \
                        b["cvd_slope"].rolling(50).std().replace(0, np.nan)

    large_total = (b["large_buy"] + b["large_sell"]).replace(0, np.nan)
    b["large_ratio"] = b["large_buy"] / large_total
    b["large_ratio_ma10"] = b["large_ratio"].rolling(10).mean()
    b["large_net_z"] = ((b["large_buy"] - b["large_sell"]).rolling(10).mean())
    lnz_std = b["large_net_z"].rolling(50).std().replace(0, np.nan)
    b["large_net_z"] = (b["large_net_z"] - b["large_net_z"].rolling(50).mean()) / lnz_std

    b["trades_ma5"] = b["n_trades"].rolling(5).mean()
    b["trades_ma20"] = b["n_trades"].rolling(20).mean()
    b["trade_accel"] = b["trades_ma5"] / b["trades_ma20"].replace(0, np.nan)

    # Chart features (1h + 4h)
    bucket_ts = b["time_start"].values
    for tf_min, tf_label in [(60, "1h"), (240, "4h")]:
        candles = resample_1min_to_bars(bars_1m, tf_min)
        indicators = compute_chart_indicators(candles)
        ind_ts = indicators.index.astype(np.int64) // 10**6
        idx = np.searchsorted(ind_ts.values, bucket_ts, side="right") - 1
        idx = np.clip(idx, 0, len(indicators) - 1)

        for col in ["rsi", "range_pctile", "dist_sma20", "dist_sma50",
                     "sma20_slope", "macd_norm", "above_sma20", "above_sma50", "bb_position"]:
            if col in indicators.columns:
                b[f"chart_{tf_label}_{col}"] = indicators[col].values[idx]

    # Orderbook features
    ob_files = sorted(glob.glob(f"{OB_DIR}/{symbol}/*.parquet"))
    ob_files = [f for f in ob_files if not f.endswith(".aria2")]
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
                del ob
                gc.collect()
            except Exception as e:
                print(f"    Warning: skipping {f}: {e}")
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
            b["ob_imb_centered"] = b["ob_imb_ma5"] - 0.5
            del ob_all, all_ob
            gc.collect()
            print(f"    Orderbook features added ({len(ob_files)} files)")

    return b


# ──────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────

def evaluate_vol(vpin, target):
    valid = pd.DataFrame({"vpin": vpin, "target": target}).dropna()
    if len(valid) < 50:
        return {"n": len(valid), "spearman": np.nan, "p": np.nan, "ratio": np.nan}
    corr, pval = stats.spearmanr(valid["vpin"], valid["target"])
    q20 = valid["vpin"].quantile(0.2)
    q80 = valid["vpin"].quantile(0.8)
    low = valid.loc[valid["vpin"] <= q20, "target"].mean()
    high = valid.loc[valid["vpin"] >= q80, "target"].mean()
    ratio = high / low if low != 0 else np.nan
    return {"n": len(valid), "spearman": round(corr, 4), "p": pval, "ratio": round(ratio, 3)}


def evaluate_feature(feat_vals, fwd_ret, feature_name):
    valid = ~(np.isnan(feat_vals) | np.isnan(fwd_ret))
    feat = feat_vals[valid]
    ret = fwd_ret[valid]
    n = len(feat)
    if n < 30 or feat.std() < 1e-10:
        return {"n": n, "acc": np.nan, "rho": np.nan, "p": np.nan, "mean_bull": np.nan, "mean_bear": np.nan}

    corr, pval = stats.spearmanr(feat, ret)

    if "above_" in feature_name or "range_pctile" in feature_name:
        pred_up = feat > 0.5
    elif "imbalance" in feature_name and "centered" not in feature_name:
        pred_up = feat > 0.5
    else:
        pred_up = feat > np.median(feat)

    if pred_up.sum() == 0 or pred_up.sum() == n:
        return {"n": n, "acc": np.nan, "rho": corr, "p": pval, "mean_bull": np.nan, "mean_bear": np.nan}

    correct = (pred_up & (ret > 0)) | (~pred_up & (ret < 0))
    return {
        "n": n, "acc": round(correct.mean(), 4), "rho": round(corr, 4), "p": pval,
        "mean_bull": ret[pred_up].mean(), "mean_bear": ret[~pred_up].mean(),
    }


# ──────────────────────────────────────────────────────
# PART 1: VPIN → VOLATILITY (walk-forward)
# ──────────────────────────────────────────────────────

def run_volatility_study(symbol, bars, cfg):
    print(f"\n  ── Part 1: VPIN → Volatility ──")
    results = []

    bars["date"] = pd.to_datetime(bars.index, unit="ms").date
    dates = sorted(bars["date"].unique())
    is_cutoff = dates[IS_DAYS - 1] if len(dates) > IS_DAYS else dates[len(dates)//2]

    for bsize in cfg["bucket_sizes"]:
        buckets = make_volume_buckets(bars.drop(columns=["date"]), bsize)
        n_buckets = len(buckets)
        avg_dur = (buckets["time_end"] - buckets["time_start"]).mean() / 60000
        print(f"\n    bucket={bsize}: {n_buckets:,} buckets (avg {avg_dur:.1f} min)")

        buckets["date"] = pd.to_datetime(buckets["time_start"], unit="ms").dt.date
        is_mask = buckets["date"] <= is_cutoff

        for lb in cfg["lookbacks"]:
            vpin = compute_vpin(buckets, lb)

            for fk in FORWARD_WINDOWS:
                fwd_vol = compute_forward_vol(buckets, fk)
                fwd_ret = compute_forward_ret(buckets, fk)

                is_vol = evaluate_vol(vpin[is_mask], fwd_vol[is_mask])
                oos_vol = evaluate_vol(vpin[~is_mask], fwd_vol[~is_mask])

                # Directional test
                net_buy = buckets["buy_vol"] - buckets["sell_vol"]
                oos_mask = ~is_mask & vpin.notna() & fwd_ret.notna()
                dir_acc = np.nan
                if oos_mask.sum() > 50:
                    q80 = vpin[oos_mask].quantile(0.8)
                    high_vpin = oos_mask & (vpin >= q80)
                    if high_vpin.sum() > 10:
                        correct = ((net_buy[high_vpin] > 0) & (fwd_ret[high_vpin] > 0)) | \
                                  ((net_buy[high_vpin] < 0) & (fwd_ret[high_vpin] < 0))
                        dir_acc = correct.mean()

                row = {
                    "symbol": symbol, "bucket": bsize, "lookback": lb, "fwd_k": fk,
                    "is_n": is_vol["n"], "is_rho": is_vol["spearman"], "is_p": is_vol["p"],
                    "is_vol_ratio": is_vol["ratio"],
                    "oos_n": oos_vol["n"], "oos_rho": oos_vol["spearman"], "oos_p": oos_vol["p"],
                    "oos_vol_ratio": oos_vol["ratio"],
                    "oos_dir_acc": round(dir_acc, 4) if not np.isnan(dir_acc) else np.nan,
                }
                results.append(row)

                if oos_vol["p"] is not None and not np.isnan(oos_vol["p"]) and oos_vol["p"] < 0.05:
                    tag = "✅" if oos_vol["spearman"] > 0.05 else "⚠️"
                    dir_str = f" dir={dir_acc:.1%}" if not np.isnan(dir_acc) else ""
                    print(f"      {tag} lb={lb} fwd={fk}: IS ρ={is_vol['spearman']:.3f} "
                          f"OOS ρ={oos_vol['spearman']:.3f}(p={oos_vol['p']:.2e}) "
                          f"vol_ratio={oos_vol['ratio']:.2f}{dir_str}")

        del buckets
        gc.collect()

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────
# PART 2: DIRECTIONAL FEATURES (rolling WF)
# ──────────────────────────────────────────────────────

def run_directional_study(symbol, bars, cfg):
    print(f"\n  ── Part 2: Directional Features (rolling WF) ──")

    bsize = cfg["dir_bucket"]
    lb = cfg["dir_lookback"]

    buckets = make_volume_buckets(bars, bsize)
    buckets["vpin"] = compute_vpin(buckets, lb)
    print(f"    {len(buckets):,} buckets (bucket={bsize}, lookback={lb})")

    print("    Computing features...")
    buckets = add_all_features(buckets, bars, symbol)

    buckets["date"] = pd.to_datetime(buckets["time_start"], unit="ms").dt.date
    dates = sorted(buckets["date"].unique())
    n_dates = len(dates)

    # Feature list
    pressure = [c for c in ["buy_ratio", "delta_mom_z", "cvd_slope_z",
                            "large_ratio_ma10", "large_net_z", "trade_accel"]
                if c in buckets.columns]
    chart_1h = [c for c in buckets.columns if c.startswith("chart_1h_") and "above_" not in c]
    chart_4h = [c for c in buckets.columns if c.startswith("chart_4h_") and "above_" not in c]
    ob = [c for c in ["ob_imbalance", "ob_spread_z", "ob_imb_ma5",
                       "ob_imb_ma20", "ob_imb_centered"]
          if c in buckets.columns]
    all_feats = pressure + chart_1h + chart_4h + ob
    print(f"    Features: pressure={len(pressure)}, chart_1h={len(chart_1h)}, "
          f"chart_4h={len(chart_4h)}, orderbook={len(ob)}, total={len(all_feats)}")

    # Rolling folds
    folds = []
    test_start = MIN_TRAIN_DAYS
    while test_start + FOLD_SIZE_DAYS <= n_dates:
        train_dates = dates[:test_start]
        test_dates = dates[test_start:test_start + FOLD_SIZE_DAYS]
        folds.append((f"fwd{len(folds)+1}", train_dates, test_dates))
        test_start += FOLD_SIZE_DAYS

    if n_dates >= MIN_TRAIN_DAYS + FOLD_SIZE_DAYS:
        bwd_train = dates[-(MIN_TRAIN_DAYS + FOLD_SIZE_DAYS):]
        bwd_test = dates[:FOLD_SIZE_DAYS]
        folds.append(("bwd", bwd_train, bwd_test))

    print(f"    {len(folds)} folds ({len(folds)-1} forward + 1 backward)")

    results = []
    for fold_label, train_dates, test_dates in folds:
        train_mask = buckets["date"].isin(set(train_dates))
        test_mask = buckets["date"].isin(set(test_dates))

        vpin_train = buckets.loc[train_mask & buckets["vpin"].notna(), "vpin"]
        if len(vpin_train) < 50:
            continue
        vpin_thresh = vpin_train.quantile(VPIN_PCTILE / 100)
        high_vpin = buckets["vpin"] >= vpin_thresh
        n_test_high = (high_vpin & test_mask).sum()
        if n_test_high < 20:
            continue

        for fwd_k in FORWARD_WINDOWS:
            fwd_ret = (buckets["vwap"].shift(-fwd_k) / buckets["vwap"] - 1).values

            for feat in all_feats:
                test_high = high_vpin & test_mask
                test_idx = test_high.values
                if test_idx.sum() < 20:
                    continue

                res = evaluate_feature(buckets[feat].values[test_idx], fwd_ret[test_idx], feat)

                train_idx = (high_vpin & train_mask).values
                train_res = evaluate_feature(
                    buckets[feat].values[train_idx], fwd_ret[train_idx], feat
                ) if train_idx.sum() > 20 else {"acc": np.nan, "rho": np.nan}

                results.append({
                    "symbol": symbol, "fold": fold_label, "feature": feat, "fwd_k": fwd_k,
                    "train_n": train_idx.sum(), "train_acc": train_res.get("acc", np.nan),
                    "train_rho": train_res.get("rho", np.nan),
                    "test_n": res["n"], "test_acc": res["acc"],
                    "test_rho": res["rho"], "test_p": res["p"],
                    "mean_bull": res.get("mean_bull", np.nan),
                    "mean_bear": res.get("mean_bear", np.nan),
                })

    del buckets
    gc.collect()
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────

def print_vol_summary(vol_df):
    print(f"\n{'='*70}")
    print("PART 1 SUMMARY: VPIN → Volatility")
    print(f"{'='*70}")

    n = len(vol_df)
    valid = vol_df.dropna(subset=["oos_rho"])
    n_sig = (valid["oos_p"] < 0.05).sum()
    n_pos = ((valid["oos_p"] < 0.05) & (valid["oos_rho"] > 0)).sum()
    bonf = 0.05 / max(n, 1)
    n_bonf = (valid["oos_p"] < bonf).sum()

    print(f"Total tests: {n}")
    print(f"OOS significant (p<0.05): {n_sig} ({100*n_sig/max(n,1):.1f}%)")
    print(f"OOS significant + positive ρ: {n_pos}")
    print(f"Bonferroni (p<{bonf:.5f}): {n_bonf}")

    if n_pos > 0:
        print("\nTop OOS results:")
        top = valid[(valid["oos_p"] < 0.05) & (valid["oos_rho"] > 0)].nsmallest(15, "oos_p")
        for _, r in top.iterrows():
            dir_str = f" dir={r['oos_dir_acc']:.1%}" if not np.isnan(r['oos_dir_acc']) else ""
            print(f"  {r['symbol']} bkt={int(r['bucket'])} lb={int(r['lookback'])} fwd={int(r['fwd_k'])}: "
                  f"IS ρ={r['is_rho']:.3f} OOS ρ={r['oos_rho']:.3f} (p={r['oos_p']:.2e}) "
                  f"vol_ratio={r['oos_vol_ratio']:.2f}{dir_str}")

    # Per-symbol
    for sym in CONFIGS:
        s = valid[(valid["symbol"] == sym) & (valid["oos_p"] < 0.05) & (valid["oos_rho"] > 0)]
        if len(s) > 0:
            print(f"\n  {sym}: {len(s)} sig tests, mean ρ={s['oos_rho'].mean():.3f}, "
                  f"mean vol_ratio={s['oos_vol_ratio'].mean():.2f}")


def print_dir_summary(dir_df):
    print(f"\n{'='*70}")
    print("PART 2 SUMMARY: Directional Features")
    print(f"{'='*70}")

    valid = dir_df.dropna(subset=["test_acc"])
    n = len(valid)
    if n == 0:
        print("No valid directional results")
        return

    bonf = 0.05 / max(n, 1)
    n_sig = (valid["test_p"] < 0.05).sum()
    n_bonf = (valid["test_p"] < bonf).sum()

    print(f"Total tests: {n}")
    print(f"Test accuracy > 55%: {(valid['test_acc'] > 0.55).sum()} ({100*(valid['test_acc'] > 0.55).mean():.1f}%)")
    print(f"Test accuracy > 60%: {(valid['test_acc'] > 0.60).sum()}")
    print(f"Significant (p<0.05): {n_sig}")
    print(f"Bonferroni (p<{bonf:.6f}): {n_bonf}")

    # Feature ranking
    print(f"\n{'─'*70}")
    print("FEATURE RANKING (mean test accuracy, all folds)")
    print(f"{'─'*70}")

    rank = valid.groupby("feature").agg(
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_rho=("test_rho", "mean"),
        n_sig=("test_p", lambda x: (x < 0.05).sum()),
        n_tests=("test_p", "count"),
    ).sort_values("mean_acc", ascending=False)

    for feat, r in rank.iterrows():
        print(f"  {feat:35s}  acc={r['mean_acc']:.1%} ±{r['std_acc']:.1%}  "
              f"rho={r['mean_rho']:+.3f}  sig={int(r['n_sig'])}/{int(r['n_tests'])}")

    # Per-symbol top features
    for sym in CONFIGS:
        sym_data = valid[valid["symbol"] == sym]
        if len(sym_data) == 0:
            continue
        print(f"\n{'─'*70}")
        print(f"  {sym} — Top 10 features")
        print(f"{'─'*70}")
        sym_rank = sym_data.groupby("feature").agg(
            mean_acc=("test_acc", "mean"),
            mean_rho=("test_rho", "mean"),
            n_sig=("test_p", lambda x: (x < 0.05).sum()),
            n_tests=("test_p", "count"),
        ).sort_values("mean_acc", ascending=False).head(10)
        for feat, r in sym_rank.iterrows():
            print(f"    {feat:35s}  acc={r['mean_acc']:.1%}  rho={r['mean_rho']:+.3f}  "
                  f"sig={int(r['n_sig'])}/{int(r['n_tests'])}")

    # Consistency (fwd vs bwd)
    print(f"\n{'─'*70}")
    print("CONSISTENT FEATURES (>55% in both fwd AND bwd)")
    print(f"{'─'*70}")

    found_any = False
    for feat in rank.index:
        fwd_data = valid[(valid["feature"] == feat) & valid["fold"].str.startswith("fwd")]
        bwd_data = valid[(valid["feature"] == feat) & valid["fold"].str.startswith("bwd")]
        if len(fwd_data) > 0 and len(bwd_data) > 0:
            fwd_acc = fwd_data["test_acc"].mean()
            bwd_acc = bwd_data["test_acc"].mean()
            if fwd_acc > 0.55 and bwd_acc > 0.55:
                print(f"  {feat:35s}  fwd={fwd_acc:.1%}  bwd={bwd_acc:.1%}  ✅ CONSISTENT")
                found_any = True
            elif fwd_acc > 0.55 or bwd_acc > 0.55:
                print(f"  {feat:35s}  fwd={fwd_acc:.1%}  bwd={bwd_acc:.1%}  partial")
                found_any = True

    if not found_any:
        print("  None found — no features consistently > 55% in both directions")

    # Top 20 individual
    print(f"\n{'─'*70}")
    print("TOP 20 INDIVIDUAL RESULTS (by accuracy)")
    print(f"{'─'*70}")

    top = valid.nlargest(20, "test_acc")
    for _, r in top.iterrows():
        print(f"  {r['symbol']:10s} {r['fold']:5s} {r['feature']:30s} fwd={int(r['fwd_k']):3d}: "
              f"train={r['train_acc']:.1%} test={r['test_acc']:.1%} rho={r['test_rho']:+.3f} (p={r['test_p']:.2e})")


def run():
    t0 = time.time()
    print("=" * 70)
    print("VPIN LOCAL STUDY — BTC + SOL (G-Drive data)")
    print(f"IS: {IS_DAYS} days, Directional: rolling WF ({MIN_TRAIN_DAYS}d train, {FOLD_SIZE_DAYS}d test)")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_vol = []
    all_dir = []

    for symbol, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        print("  Loading trades...")
        bars = aggregate_to_1min(symbol)
        if bars is None:
            continue

        # Part 1: Volatility
        vol_df = run_volatility_study(symbol, bars, cfg)
        all_vol.append(vol_df)

        # Part 2: Directional
        bars_clean = bars.drop(columns=["date"], errors="ignore")
        dir_df = run_directional_study(symbol, bars_clean, cfg)
        all_dir.append(dir_df)

        del bars, bars_clean
        gc.collect()

    # Combine and summarize
    vol_results = pd.concat(all_vol, ignore_index=True) if all_vol else pd.DataFrame()
    dir_results = pd.concat(all_dir, ignore_index=True) if all_dir else pd.DataFrame()

    if len(vol_results) > 0:
        print_vol_summary(vol_results)
        vol_out = f"{OUT_DIR}/vpin_volatility_results.csv"
        vol_results.to_csv(vol_out, index=False)
        print(f"\nSaved: {vol_out}")

    if len(dir_results) > 0:
        print_dir_summary(dir_results)
        dir_out = f"{OUT_DIR}/vpin_directional_results.csv"
        dir_results.to_csv(dir_out, index=False)
        print(f"\nSaved: {dir_out}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    run()
