#!/usr/bin/env python3
"""
Sprint 2: VPIN Validation
=========================
Volume-Synchronized Probability of Informed Trading (Easley, López de Prado, O'Hara 2012)

VPIN = |buy_vol - sell_vol| / total_vol, computed over volume buckets.

Hypothesis: High VPIN predicts elevated volatility and/or adverse price moves.
We test VPIN as:
  1. Volatility predictor (does high VPIN → high next-period RV?)
  2. Directional signal (does high VPIN predict direction? Probably not, but test anyway)
  3. Regime filter (does conditioning on VPIN regime improve a base strategy?)

Walk-forward: 14 folds, expanding window, multi-asset (BTC + ETH).
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime

from backend.config.data_paths import BARS_1M_V1

OFLOW_DIR = BARS_1M_V1
OHLCV_DIR = Path.home() / "Desktop/maestro/data/ohlcv"
OUTPUT_DIR = Path.home() / "Desktop/maestro/backend/research"

N_FOLDS = 14
ANNUALIZE_FACTOR = np.sqrt(365)
TRANSACTION_COST = 0.0006

# ============================================================
# VPIN COMPUTATION
# ============================================================

def compute_vpin(df_1m, volume_bucket_size=None, n_buckets=50):
    """
    Compute VPIN from 1-minute orderflow data.
    
    1. Aggregate 1m bars into volume buckets (each bucket = fixed BTC/ETH volume)
    2. For each bucket: VPIN_bucket = |buy_vol - sell_vol| / total_vol
    3. VPIN = rolling mean of last n_buckets
    
    Returns a time-indexed series of VPIN values (one per volume bucket).
    """
    df = df_1m[["timestamp", "close", "volume", "buy_vol", "sell_vol"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Auto-size buckets if not specified: total_volume / ~5000 buckets
    if volume_bucket_size is None:
        total_vol = df["volume"].sum()
        volume_bucket_size = total_vol / 5000
    
    # Aggregate into volume buckets
    buckets = []
    cum_vol = 0
    cum_buy = 0
    cum_sell = 0
    bucket_start = df["timestamp"].iloc[0]
    
    for _, row in df.iterrows():
        cum_vol += row["volume"]
        cum_buy += row["buy_vol"]
        cum_sell += row["sell_vol"]
        
        if cum_vol >= volume_bucket_size:
            buckets.append({
                "timestamp": row["timestamp"],
                "bucket_start": bucket_start,
                "volume": cum_vol,
                "buy_vol": cum_buy,
                "sell_vol": cum_sell,
                "close": row["close"],
                "order_imbalance": abs(cum_buy - cum_sell) / cum_vol if cum_vol > 0 else 0,
                "signed_imbalance": (cum_buy - cum_sell) / cum_vol if cum_vol > 0 else 0,
            })
            cum_vol = 0
            cum_buy = 0
            cum_sell = 0
            bucket_start = row["timestamp"]
    
    bucket_df = pd.DataFrame(buckets)
    bucket_df.set_index("timestamp", inplace=True)
    
    # VPIN = rolling mean of order_imbalance over last n_buckets
    bucket_df["vpin"] = bucket_df["order_imbalance"].rolling(n_buckets).mean()
    bucket_df["signed_vpin"] = bucket_df["signed_imbalance"].rolling(n_buckets).mean()
    
    return bucket_df.dropna()

def vpin_to_daily(vpin_df):
    """Resample VPIN to daily frequency (last value per day)."""
    daily = vpin_df.resample("D").agg({
        "vpin": "last",
        "signed_vpin": "last",
        "order_imbalance": "mean",  # avg bucket imbalance that day
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    # Also compute intraday VPIN stats
    vpin_stats = vpin_df["vpin"].resample("D").agg(["mean", "std", "max", "min"])
    vpin_stats.columns = ["vpin_mean", "vpin_std", "vpin_max", "vpin_min"]
    daily = daily.join(vpin_stats)
    
    return daily

# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def expanding_window_splits(n, n_folds=14, min_train_pct=0.3):
    min_train = int(n * min_train_pct)
    total_test = n - min_train
    test_per_fold = total_test // n_folds
    
    if test_per_fold < 10:
        n_folds = max(total_test // 15, 3)
        test_per_fold = total_test // n_folds
    
    splits = []
    for i in range(n_folds):
        test_start = min_train + i * test_per_fold
        test_end = min_train + (i + 1) * test_per_fold
        if i == n_folds - 1:
            test_end = n
        if test_end > n:
            break
        splits.append((list(range(0, test_start)), list(range(test_start, test_end))))
    
    return splits

# ============================================================
# TEST 1: VPIN → Volatility Prediction
# ============================================================

def test_vpin_vol_prediction(daily, n_folds=14):
    """
    Does VPIN predict next-day realized volatility?
    Compare: RV model with vs without VPIN features.
    """
    df = daily.copy()
    df["rv"] = np.log(df["close"] / df["close"].shift(1)).abs()
    df["rv_lag1"] = df["rv"].shift(1)
    df["rv_5d"] = df["rv"].rolling(5).mean().shift(1)
    df["rv_22d"] = df["rv"].rolling(22).mean().shift(1)
    df["vpin_lag1"] = df["vpin"].shift(1)
    df["vpin_mean_lag1"] = df["vpin_mean"].shift(1)
    df["vpin_max_lag1"] = df["vpin_max"].shift(1)
    df["vpin_std_lag1"] = df["vpin_std"].shift(1)
    df["next_rv"] = df["rv"].shift(-1)
    df = df.dropna()
    
    base_features = ["rv_lag1", "rv_5d", "rv_22d"]
    vpin_features = ["vpin_lag1", "vpin_mean_lag1", "vpin_max_lag1"]
    all_features = base_features + vpin_features
    
    splits = expanding_window_splits(len(df), n_folds)
    actual_folds = len(splits)
    
    base_maes = []
    enh_maes = []
    fold_details = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_train = train["next_rv"].values
        y_test = test["next_rv"].values
        
        # Baseline: RV lags only
        X_train_b = np.column_stack([np.ones(len(train)), train[base_features].values])
        X_test_b = np.column_stack([np.ones(len(test)), test[base_features].values])
        beta_b = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
        pred_b = X_test_b @ beta_b
        mae_b = np.mean(np.abs(y_test - pred_b))
        
        # Enhanced: + VPIN
        X_train_e = np.column_stack([np.ones(len(train)), train[all_features].values])
        X_test_e = np.column_stack([np.ones(len(test)), test[all_features].values])
        beta_e = np.linalg.lstsq(X_train_e, y_train, rcond=None)[0]
        pred_e = X_test_e @ beta_e
        mae_e = np.mean(np.abs(y_test - pred_e))
        
        base_maes.append(mae_b)
        enh_maes.append(mae_e)
        fold_details.append({
            "fold": i+1,
            "mae_base": round(mae_b, 6),
            "mae_enhanced": round(mae_e, 6),
            "improvement_pct": round((mae_b - mae_e) / mae_b * 100, 2),
            "test_start": str(df.index[test_idx[0]].date()),
            "test_end": str(df.index[test_idx[-1]].date()),
            "n_days": len(test_idx),
        })
    
    # Aggregate
    improvements = [(b - e) / b * 100 for b, e in zip(base_maes, enh_maes)]
    t_stat, p_val = stats.ttest_rel(base_maes, enh_maes) if len(base_maes) > 2 else (0, 1)
    
    agg = {
        "avg_improvement_pct": round(np.mean(improvements), 2),
        "median_improvement_pct": round(np.median(improvements), 2),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "n_folds": actual_folds,
        "pct_improved": round(sum(1 for i in improvements if i > 0) / len(improvements) * 100, 1),
    }
    
    return fold_details, agg

# ============================================================
# TEST 2: VPIN Quintile Analysis (non-parametric)
# ============================================================

def test_vpin_quintiles(daily):
    """Non-parametric: sort by VPIN quintile, measure next-day RV and returns."""
    df = daily.copy()
    df["rv"] = np.log(df["close"] / df["close"].shift(1)).abs()
    df["next_rv"] = df["rv"].shift(-1)
    df["next_ret"] = df["close"].pct_change().shift(-1)
    df["vpin_lag1"] = df["vpin"].shift(1)
    df = df.dropna()
    
    df["vpin_quintile"] = pd.qcut(df["vpin_lag1"], 5, labels=["Q1_low","Q2","Q3","Q4","Q5_high"])
    
    results = []
    for q in ["Q1_low","Q2","Q3","Q4","Q5_high"]:
        sub = df[df["vpin_quintile"] == q]
        results.append({
            "quintile": q,
            "n": len(sub),
            "mean_next_rv": round(sub["next_rv"].mean(), 6),
            "median_next_rv": round(sub["next_rv"].median(), 6),
            "mean_next_ret": round(sub["next_ret"].mean(), 6),
            "std_next_ret": round(sub["next_ret"].std(), 6),
            "mean_vpin": round(sub["vpin_lag1"].mean(), 4),
        })
    
    # Monotonicity test: Q5 should have higher vol than Q1
    q5_rv = df[df["vpin_quintile"]=="Q5_high"]["next_rv"]
    q1_rv = df[df["vpin_quintile"]=="Q1_low"]["next_rv"]
    t_q, p_q = stats.ttest_ind(q5_rv, q1_rv)
    
    # Spearman correlation: VPIN_lag → next_rv
    rho, p_rho = stats.spearmanr(df["vpin_lag1"], df["next_rv"])
    
    return results, {
        "q5_q1_ratio": round(q5_rv.mean() / q1_rv.mean(), 2),
        "q5_vs_q1_ttest": round(t_q, 3),
        "q5_vs_q1_pvalue": round(p_q, 4),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_rho, 4),
    }

# ============================================================
# TEST 3: VPIN as Directional Signal
# ============================================================

def test_vpin_directional(daily, n_folds=14):
    """
    Test signed VPIN as directional signal.
    High signed VPIN (net buying) → long. Low signed VPIN (net selling) → short.
    """
    df = daily.copy()
    df["ret"] = df["close"].pct_change()
    df["signed_vpin_lag1"] = df["signed_vpin"].shift(1)
    df = df.dropna()
    
    splits = expanding_window_splits(len(df), n_folds)
    actual_folds = len(splits)
    
    fold_results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Thresholds from training data
        threshold_long = train["signed_vpin_lag1"].quantile(0.7)
        threshold_short = train["signed_vpin_lag1"].quantile(0.3)
        
        signal = pd.Series(0, index=test.index)
        signal[test["signed_vpin_lag1"] > threshold_long] = 1
        signal[test["signed_vpin_lag1"] < threshold_short] = -1
        
        strat_ret = signal.shift(1) * test["ret"]
        trades = signal.diff().abs()
        strat_ret = strat_ret - trades * TRANSACTION_COST
        strat_ret = strat_ret.dropna()
        
        if len(strat_ret) > 0 and strat_ret.std() > 0:
            sharpe = strat_ret.mean() / strat_ret.std() * ANNUALIZE_FACTOR
            total_ret = (1 + strat_ret).prod() - 1
            win_rate = (strat_ret > 0).sum() / len(strat_ret)
        else:
            sharpe = 0; total_ret = 0; win_rate = 0
        
        fold_results.append({
            "fold": i+1,
            "sharpe": round(sharpe, 4),
            "total_return": round(total_ret * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "test_start": str(df.index[test_idx[0]].date()),
            "test_end": str(df.index[test_idx[-1]].date()),
        })
    
    sharpes = [r["sharpe"] for r in fold_results]
    t_stat, p_val = stats.ttest_1samp(sharpes, 0) if len(sharpes) > 2 else (0, 1)
    p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    
    agg = {
        "mean_sharpe": round(np.mean(sharpes), 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_one, 4),
        "pct_positive": round(sum(1 for s in sharpes if s > 0) / len(sharpes) * 100, 1),
        "n_folds": actual_folds,
    }
    
    return fold_results, agg

# ============================================================
# TEST 4: VPIN Extreme Filter (vol-timing)
# ============================================================

def test_vpin_extreme_filter(daily, n_folds=14):
    """
    Use VPIN as a risk filter:
    - When VPIN > 90th percentile (from training): go flat (risk-off)
    - Otherwise: stay long (buy & hold)
    Compare to pure buy & hold.
    """
    df = daily.copy()
    df["ret"] = df["close"].pct_change()
    df["vpin_lag1"] = df["vpin"].shift(1)
    df = df.dropna()
    
    splits = expanding_window_splits(len(df), n_folds)
    actual_folds = len(splits)
    
    fold_results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Risk-off threshold from training
        threshold_90 = train["vpin_lag1"].quantile(0.90)
        threshold_95 = train["vpin_lag1"].quantile(0.95)
        
        # Signal: 1 = long (normal), 0 = flat (high VPIN)
        signal_90 = pd.Series(1, index=test.index)
        signal_90[test["vpin_lag1"] > threshold_90] = 0
        
        signal_95 = pd.Series(1, index=test.index)
        signal_95[test["vpin_lag1"] > threshold_95] = 0
        
        # Buy & hold returns
        bh_ret = test["ret"]
        
        # Filtered returns (90th)
        filt_ret_90 = signal_90.shift(1) * test["ret"]
        trades_90 = signal_90.diff().abs()
        filt_ret_90 = filt_ret_90 - trades_90 * TRANSACTION_COST
        filt_ret_90 = filt_ret_90.dropna()
        
        # Filtered returns (95th)
        filt_ret_95 = signal_95.shift(1) * test["ret"]
        trades_95 = signal_95.diff().abs()
        filt_ret_95 = filt_ret_95 - trades_95 * TRANSACTION_COST
        filt_ret_95 = filt_ret_95.dropna()
        
        bh_ret = bh_ret.loc[filt_ret_90.index]
        
        def sharpe(r):
            return r.mean() / r.std() * ANNUALIZE_FACTOR if r.std() > 0 else 0
        
        pct_flat_90 = (signal_90 == 0).sum() / len(signal_90) * 100
        
        fold_results.append({
            "fold": i+1,
            "bh_sharpe": round(sharpe(bh_ret), 4),
            "filt90_sharpe": round(sharpe(filt_ret_90), 4),
            "filt95_sharpe": round(sharpe(filt_ret_95), 4),
            "bh_return": round((1+bh_ret).prod() - 1, 4) * 100,
            "filt90_return": round(((1+filt_ret_90).prod() - 1) * 100, 2),
            "filt95_return": round(((1+filt_ret_95).prod() - 1) * 100, 2),
            "pct_flat_90": round(pct_flat_90, 1),
            "test_start": str(df.index[test_idx[0]].date()),
            "test_end": str(df.index[test_idx[-1]].date()),
        })
    
    # Does filtering improve Sharpe?
    bh_sharpes = [r["bh_sharpe"] for r in fold_results]
    f90_sharpes = [r["filt90_sharpe"] for r in fold_results]
    f95_sharpes = [r["filt95_sharpe"] for r in fold_results]
    
    t90, p90 = stats.ttest_rel(f90_sharpes, bh_sharpes) if len(bh_sharpes) > 2 else (0, 1)
    t95, p95 = stats.ttest_rel(f95_sharpes, bh_sharpes) if len(bh_sharpes) > 2 else (0, 1)
    
    agg = {
        "mean_bh_sharpe": round(np.mean(bh_sharpes), 4),
        "mean_f90_sharpe": round(np.mean(f90_sharpes), 4),
        "mean_f95_sharpe": round(np.mean(f95_sharpes), 4),
        "f90_vs_bh_t": round(t90, 4),
        "f90_vs_bh_p": round(p90, 4),
        "f95_vs_bh_t": round(t95, 4),
        "f95_vs_bh_p": round(p95, 4),
        "pct_f90_beats_bh": round(sum(1 for f, b in zip(f90_sharpes, bh_sharpes) if f > b) / len(bh_sharpes) * 100, 1),
        "n_folds": actual_folds,
    }
    
    return fold_results, agg

# ============================================================
# TEST 5: Raw correlation analysis
# ============================================================

def test_vpin_correlations(daily):
    """Full-sample correlation analysis between VPIN features and future outcomes."""
    df = daily.copy()
    df["rv"] = np.log(df["close"] / df["close"].shift(1)).abs()
    df["next_rv"] = df["rv"].shift(-1)
    df["next_ret"] = df["close"].pct_change().shift(-1)
    df["next_abs_ret"] = df["next_ret"].abs()
    df["next_5d_rv"] = df["rv"].rolling(5).mean().shift(-5)
    df["vpin_lag1"] = df["vpin"].shift(1)
    df["vpin_mean_lag1"] = df["vpin_mean"].shift(1)
    df["vpin_max_lag1"] = df["vpin_max"].shift(1)
    df["signed_vpin_lag1"] = df["signed_vpin"].shift(1)
    df = df.dropna()
    
    results = {}
    for feat in ["vpin_lag1", "vpin_mean_lag1", "vpin_max_lag1", "signed_vpin_lag1"]:
        for target in ["next_rv", "next_abs_ret", "next_ret", "next_5d_rv"]:
            r, p = stats.pearsonr(df[feat], df[target])
            sr, sp = stats.spearmanr(df[feat], df[target])
            results[f"{feat}_vs_{target}"] = {
                "pearson_r": round(r, 4), "pearson_p": round(p, 4),
                "spearman_rho": round(sr, 4), "spearman_p": round(sp, 4),
            }
    
    return results

# ============================================================
# MAIN
# ============================================================

def process_asset(asset, symbol):
    """Run full VPIN validation for one asset."""
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()}")
    print(f"{'='*60}")
    
    # Load 1m orderflow
    from backend.config.data_paths import BARS_1M_V2
    fp = BARS_1M_V2 / f"{symbol}_1m_v2.csv"
    print(f"  Loading {fp.name}...")
    df_1m = pd.read_csv(fp, parse_dates=["timestamp"])
    print(f"  Loaded {len(df_1m):,} 1-minute bars")
    
    # Compute VPIN with different bucket sizes
    total_vol = df_1m["volume"].sum()
    n_days = (df_1m["timestamp"].max() - df_1m["timestamp"].min()).days
    avg_daily_vol = total_vol / n_days
    
    # Standard: bucket = 1/50th of daily volume (so ~50 buckets/day)
    bucket_size = avg_daily_vol / 50
    print(f"  Avg daily volume: {avg_daily_vol:.0f} {asset.upper()}")
    print(f"  Volume bucket size: {bucket_size:.2f} {asset.upper()}")
    
    print("  Computing VPIN...")
    vpin_df = compute_vpin(df_1m, volume_bucket_size=bucket_size, n_buckets=50)
    print(f"  VPIN computed: {len(vpin_df):,} volume buckets")
    
    daily = vpin_to_daily(vpin_df)
    print(f"  Daily VPIN: {len(daily)} days, {daily.index[0].date()} to {daily.index[-1].date()}")
    print(f"  VPIN stats: mean={daily['vpin'].mean():.4f}, std={daily['vpin'].std():.4f}, "
          f"min={daily['vpin'].min():.4f}, max={daily['vpin'].max():.4f}")
    
    report = []
    report.append(f"## {asset.upper()}\n\n")
    report.append(f"**Data:** {len(df_1m):,} 1-minute bars, {df_1m['timestamp'].min().date()} to {df_1m['timestamp'].max().date()}  \n")
    report.append(f"**Volume bucket:** {bucket_size:.2f} {asset.upper()} (~50 buckets/day)  \n")
    report.append(f"**VPIN rolling window:** 50 buckets (~1 day)  \n")
    report.append(f"**Daily VPIN:** {len(daily)} days  \n")
    report.append(f"**VPIN stats:** mean={daily['vpin'].mean():.4f}, std={daily['vpin'].std():.4f}  \n\n")
    
    # TEST 1: Correlations
    print("\n  --- Test 1: Correlation Analysis ---")
    corr_results = test_vpin_correlations(daily)
    
    report.append("### Test 1: VPIN Correlations with Future Outcomes\n\n")
    report.append("| Feature | Target | Pearson r | p | Spearman ρ | p |\n")
    report.append("|---------|--------|-----------|---|------------|---|\n")
    for key, vals in corr_results.items():
        feat, target = key.split("_vs_")
        sig = "✅" if vals["spearman_p"] < 0.05 else "⚠️" if vals["spearman_p"] < 0.1 else ""
        report.append(f"| {feat} | {target} | {vals['pearson_r']:+.4f} | {vals['pearson_p']:.4f} | "
                      f"{vals['spearman_rho']:+.4f} | {vals['spearman_p']:.4f} {sig} |\n")
        if "next_rv" in key and "vpin_lag1_vs" in key:
            print(f"    {feat} → {target}: r={vals['pearson_r']:+.4f} (p={vals['pearson_p']:.4f}), "
                  f"ρ={vals['spearman_rho']:+.4f} (p={vals['spearman_p']:.4f})")
    
    # TEST 2: Quintile Analysis
    print("\n  --- Test 2: VPIN Quintile Analysis ---")
    quintile_results, quintile_agg = test_vpin_quintiles(daily)
    
    report.append(f"\n### Test 2: VPIN Quintile → Next-Day Outcomes\n\n")
    report.append("| Quintile | n | Mean VPIN | Mean Next RV | Median Next RV | Mean Next Ret |\n")
    report.append("|----------|---|-----------|-------------|----------------|---------------|\n")
    for r in quintile_results:
        report.append(f"| {r['quintile']} | {r['n']} | {r['mean_vpin']:.4f} | {r['mean_next_rv']:.6f} | "
                      f"{r['median_next_rv']:.6f} | {r['mean_next_ret']:+.6f} |\n")
    
    report.append(f"\n**Q5/Q1 RV ratio:** {quintile_agg['q5_q1_ratio']}x  \n")
    report.append(f"**Q5 vs Q1 t-test:** t={quintile_agg['q5_vs_q1_ttest']}, p={quintile_agg['q5_vs_q1_pvalue']}  \n")
    report.append(f"**Spearman (VPIN→next_rv):** ρ={quintile_agg['spearman_rho']}, p={quintile_agg['spearman_p']}  \n\n")
    
    print(f"    Q5/Q1 ratio: {quintile_agg['q5_q1_ratio']}x, t={quintile_agg['q5_vs_q1_ttest']}, p={quintile_agg['q5_vs_q1_pvalue']}")
    print(f"    Spearman: ρ={quintile_agg['spearman_rho']}, p={quintile_agg['spearman_p']}")
    
    # TEST 3: Vol Prediction WF
    print("\n  --- Test 3: VPIN Vol Prediction (Walk-Forward) ---")
    vol_folds, vol_agg = test_vpin_vol_prediction(daily, N_FOLDS)
    
    report.append(f"### Test 3: VPIN Vol Prediction (Walk-Forward, {vol_agg['n_folds']} folds)\n\n")
    report.append("| Fold | Test Period | MAE Base | MAE +VPIN | Δ% |\n")
    report.append("|------|------------|----------|-----------|----|\n")
    for d in vol_folds:
        report.append(f"| {d['fold']} | {d['test_start']} → {d['test_end']} | {d['mae_base']:.6f} | "
                      f"{d['mae_enhanced']:.6f} | {d['improvement_pct']:+.2f}% |\n")
    
    report.append(f"\n**Avg MAE improvement:** {vol_agg['avg_improvement_pct']:+.2f}%  \n")
    report.append(f"**Paired t-test:** t={vol_agg['t_stat']}, p={vol_agg['p_value']}  \n")
    report.append(f"**% folds improved:** {vol_agg['pct_improved']}%  \n\n")
    
    print(f"    Avg improvement: {vol_agg['avg_improvement_pct']:+.2f}%, t={vol_agg['t_stat']}, p={vol_agg['p_value']}")
    
    # TEST 4: Directional Signal WF
    print("\n  --- Test 4: Signed VPIN Directional Signal ---")
    dir_folds, dir_agg = test_vpin_directional(daily, N_FOLDS)
    
    report.append(f"### Test 4: Signed VPIN Directional Signal ({dir_agg['n_folds']} folds)\n\n")
    report.append("| Fold | Test Period | Sharpe | Return | Win% |\n")
    report.append("|------|------------|--------|--------|------|\n")
    for r in dir_folds:
        report.append(f"| {r['fold']} | {r['test_start']} → {r['test_end']} | {r['sharpe']:.2f} | {r['total_return']:.1f}% | {r['win_rate']:.1f}% |\n")
    
    report.append(f"\n**Mean Sharpe:** {dir_agg['mean_sharpe']:.2f} | **t:** {dir_agg['t_stat']:.2f} | **p:** {dir_agg['p_value']:.3f} | **% positive:** {dir_agg['pct_positive']:.0f}%  \n\n")
    
    print(f"    Mean Sharpe: {dir_agg['mean_sharpe']:.2f}, t={dir_agg['t_stat']:.2f}, p={dir_agg['p_value']:.3f}")
    
    # TEST 5: VPIN Extreme Filter
    print("\n  --- Test 5: VPIN Risk-Off Filter ---")
    filt_folds, filt_agg = test_vpin_extreme_filter(daily, N_FOLDS)
    
    report.append(f"### Test 5: VPIN Risk-Off Filter ({filt_agg['n_folds']} folds)\n\n")
    report.append("*Go flat when VPIN > training 90th/95th percentile, otherwise hold long.*\n\n")
    report.append("| Fold | Test Period | B&H Sharpe | Filt90 Sharpe | Filt95 Sharpe | % Flat |\n")
    report.append("|------|------------|-----------|--------------|--------------|--------|\n")
    for r in filt_folds:
        report.append(f"| {r['fold']} | {r['test_start']} → {r['test_end']} | {r['bh_sharpe']:.2f} | "
                      f"{r['filt90_sharpe']:.2f} | {r['filt95_sharpe']:.2f} | {r['pct_flat_90']:.0f}% |\n")
    
    report.append(f"\n**Mean B&H Sharpe:** {filt_agg['mean_bh_sharpe']:.2f}  \n")
    report.append(f"**Mean Filt90 Sharpe:** {filt_agg['mean_f90_sharpe']:.2f} (t={filt_agg['f90_vs_bh_t']:.2f}, p={filt_agg['f90_vs_bh_p']:.3f})  \n")
    report.append(f"**Mean Filt95 Sharpe:** {filt_agg['mean_f95_sharpe']:.2f} (t={filt_agg['f95_vs_bh_t']:.2f}, p={filt_agg['f95_vs_bh_p']:.3f})  \n")
    report.append(f"**% folds Filt90 beats B&H:** {filt_agg['pct_f90_beats_bh']}%  \n\n")
    
    print(f"    B&H Sharpe: {filt_agg['mean_bh_sharpe']:.2f}")
    print(f"    Filt90: {filt_agg['mean_f90_sharpe']:.2f} (t={filt_agg['f90_vs_bh_t']:.2f}, p={filt_agg['f90_vs_bh_p']:.3f})")
    print(f"    Filt95: {filt_agg['mean_f95_sharpe']:.2f} (t={filt_agg['f95_vs_bh_t']:.2f}, p={filt_agg['f95_vs_bh_p']:.3f})")
    
    return report, {
        "correlations": corr_results,
        "quintiles": quintile_agg,
        "vol_prediction": vol_agg,
        "directional": dir_agg,
        "extreme_filter": filt_agg,
    }

def main():
    print("=" * 70)
    print("SPRINT 2: VPIN VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Method: 14-fold expanding WF, volume-bucketed VPIN")
    print("=" * 70)
    
    full_report = []
    full_report.append("# Sprint 2: VPIN Validation Results\n\n")
    full_report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
    full_report.append(f"**Method:** {N_FOLDS}-fold expanding walk-forward  \n")
    full_report.append(f"**VPIN:** Volume-bucketed (Easley, López de Prado, O'Hara 2012)  \n")
    full_report.append(f"**Transaction cost:** {TRANSACTION_COST*100:.2f}%  \n\n")
    full_report.append("---\n\n")
    
    all_summaries = {}
    
    for asset, symbol in [("btc", "btcusdt"), ("eth", "ethusdt")]:
        asset_report, summary = process_asset(asset, symbol)
        full_report.extend(asset_report)
        full_report.append("---\n\n")
        all_summaries[asset] = summary
    
    # Cross-asset summary
    full_report.append("## Cross-Asset Summary\n\n")
    
    full_report.append("### Vol Prediction (VPIN → next-day RV)\n\n")
    full_report.append("| Asset | Avg Δ MAE% | t-stat | p-value | % Improved |\n")
    full_report.append("|-------|-----------|--------|---------|------------|\n")
    for asset, s in all_summaries.items():
        v = s["vol_prediction"]
        sig = "✅" if v["p_value"] < 0.05 else "⚠️" if v["p_value"] < 0.1 else "❌"
        full_report.append(f"| {asset.upper()} | {v['avg_improvement_pct']:+.2f}% | {v['t_stat']:.2f} | {v['p_value']:.3f} | {v['pct_improved']}% {sig} |\n")
    
    full_report.append("\n### Quintile Analysis (Q5/Q1 vol ratio)\n\n")
    full_report.append("| Asset | Q5/Q1 Ratio | Q5 vs Q1 p | Spearman ρ | Spearman p |\n")
    full_report.append("|-------|------------|-----------|------------|------------|\n")
    for asset, s in all_summaries.items():
        q = s["quintiles"]
        sig = "✅" if q["spearman_p"] < 0.05 else "⚠️" if q["spearman_p"] < 0.1 else "❌"
        full_report.append(f"| {asset.upper()} | {q['q5_q1_ratio']}x | {q['q5_vs_q1_pvalue']} | {q['spearman_rho']} | {q['spearman_p']} {sig} |\n")
    
    full_report.append("\n### Directional Signal\n\n")
    full_report.append("| Asset | Mean Sharpe | t-stat | p-value | % Positive |\n")
    full_report.append("|-------|------------|--------|---------|------------|\n")
    for asset, s in all_summaries.items():
        d = s["directional"]
        sig = "✅" if d["p_value"] < 0.05 and d["mean_sharpe"] > 0.5 else "❌"
        full_report.append(f"| {asset.upper()} | {d['mean_sharpe']:.2f} | {d['t_stat']:.2f} | {d['p_value']:.3f} | {d['pct_positive']}% {sig} |\n")
    
    full_report.append("\n### Risk-Off Filter (VPIN > 90th percentile → flat)\n\n")
    full_report.append("| Asset | B&H Sharpe | Filt90 Sharpe | Improvement | p-value |\n")
    full_report.append("|-------|-----------|--------------|-------------|----------|\n")
    for asset, s in all_summaries.items():
        f = s["extreme_filter"]
        delta = f["mean_f90_sharpe"] - f["mean_bh_sharpe"]
        sig = "✅" if f["f90_vs_bh_p"] < 0.05 and delta > 0 else "⚠️" if f["f90_vs_bh_p"] < 0.1 and delta > 0 else "❌"
        full_report.append(f"| {asset.upper()} | {f['mean_bh_sharpe']:.2f} | {f['mean_f90_sharpe']:.2f} | {delta:+.2f} | {f['f90_vs_bh_p']:.3f} {sig} |\n")
    
    # Final verdicts
    full_report.append("\n## Final Verdicts\n\n")
    full_report.append("| Use Case | Verdict | Evidence |\n")
    full_report.append("|----------|---------|----------|\n")
    
    # Determine verdicts
    vol_pass = sum(1 for s in all_summaries.values() if s["vol_prediction"]["p_value"] < 0.1)
    dir_pass = sum(1 for s in all_summaries.values() if s["directional"]["p_value"] < 0.1 and s["directional"]["mean_sharpe"] > 0.3)
    filt_pass = sum(1 for s in all_summaries.values() if s["extreme_filter"]["f90_vs_bh_p"] < 0.1)
    quint_pass = sum(1 for s in all_summaries.values() if s["quintiles"]["spearman_p"] < 0.05)
    
    full_report.append(f"| Vol Prediction | {'✅ IMPLEMENT' if vol_pass == 2 else '⚠️ PARTIAL' if vol_pass == 1 else '❌ REJECTED'} | {vol_pass}/2 assets pass |\n")
    full_report.append(f"| Directional Signal | {'✅ IMPLEMENT' if dir_pass == 2 else '⚠️ PARTIAL' if dir_pass == 1 else '❌ REJECTED'} | {dir_pass}/2 assets pass |\n")
    full_report.append(f"| Risk-Off Filter | {'✅ IMPLEMENT' if filt_pass == 2 else '⚠️ PARTIAL' if filt_pass == 1 else '❌ REJECTED'} | {filt_pass}/2 assets pass |\n")
    full_report.append(f"| Quintile Monotonicity | {'✅ CONFIRMED' if quint_pass == 2 else '⚠️ PARTIAL' if quint_pass == 1 else '❌ NOT FOUND'} | {quint_pass}/2 assets significant |\n")
    
    full_report.append(f"\n---\n\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by sprint2_vpin_validation.py*\n")
    
    # Save
    output_path = OUTPUT_DIR / "sprint2_vpin_results.md"
    with open(output_path, "w") as f:
        f.write("".join(full_report))
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
