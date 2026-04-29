#!/usr/bin/env python3
"""
Sprint 1 Academic Paper Validation Framework
=============================================
Walk-forward OOS validation of Sprint 1 signals on actual VPS data.
No in-sample metrics. No fake results. Honest assessment.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime
import json

# Paths
OHLCV_DIR = Path.home() / "Desktop/maestro/data/ohlcv"
DERIV_DIR = Path.home() / "Desktop/maestro/data/derivatives"
OUTPUT_DIR = Path.home() / "Desktop/maestro/backend/research"

# Constants
TRANSACTION_COST = 0.0006  # 0.06% round-trip
MIN_FOLDS = 5
TRAIN_RATIO = 0.7

# ============================================================
# DATA LOADING
# ============================================================

def load_btc_daily():
    """Load BTC daily OHLCV."""
    df = pd.read_csv(OHLCV_DIR / "binance_btc_usdt_1d.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df

def load_derivatives(asset="btc"):
    """Load all derivatives data for an asset."""
    data = {}
    
    # Funding
    fp = DERIV_DIR / f"{asset}_funding_full.csv"
    if fp.exists():
        df = pd.read_csv(fp, parse_dates=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        data["funding"] = df
    
    # OI
    fp = DERIV_DIR / f"{asset}_oi_daily_full.csv"
    if fp.exists():
        df = pd.read_csv(fp, parse_dates=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        df.columns = ["oi_open", "oi_high", "oi_low", "oi_close"]
        data["oi"] = df
    
    # Liquidations
    fp = DERIV_DIR / f"{asset}_liquidations_daily.csv"
    if fp.exists():
        df = pd.read_csv(fp, parse_dates=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        data["liquidations"] = df
    
    # LSR
    fp = DERIV_DIR / f"{asset}_lsr_daily_full.csv"
    if fp.exists():
        df = pd.read_csv(fp, parse_dates=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        data["lsr"] = df
    
    # Taker
    fp = DERIV_DIR / f"{asset}_taker.csv"
    if fp.exists():
        df = pd.read_csv(fp, parse_dates=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        data["taker"] = df
    
    return data

def merge_ohlcv_derivatives(ohlcv, derivatives, exclude_taker=True):
    """Merge OHLCV with derivatives on date. Uses left join on OHLCV, then drops rows with all-NaN derivatives."""
    merged = ohlcv.copy()
    
    for name, df in derivatives.items():
        if name == "taker" and exclude_taker:
            continue  # Skip taker — only ~21 days, ruins join
        if name == "taker":
            df_daily = df.resample("D").agg({
                "buySellRatio": "mean",
                "sellVol": "sum", 
                "buyVol": "sum"
            })
            merged = merged.join(df_daily, how="left")
        else:
            merged = merged.join(df, how="left")
    
    # Keep only rows where we have at least funding_rate (our primary derivatives column)
    if "funding_rate" in merged.columns:
        merged = merged.dropna(subset=["funding_rate"])
    
    return merged

# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def expanding_window_splits(n, n_folds=5, min_train_pct=0.3):
    """
    Generate expanding window train/test splits.
    Each fold: train on first X%, test on next chunk.
    """
    # Reserve enough data so each test fold has meaningful size
    test_size = int(n * (1 - TRAIN_RATIO) / n_folds * 2)  # rough
    # Actually: divide the latter portion into n_folds test segments
    # Train expands, test is fixed-size chunks
    
    total_test = int(n * (1 - TRAIN_RATIO))
    test_per_fold = total_test // n_folds
    min_train = int(n * min_train_pct)
    
    splits = []
    for i in range(n_folds):
        test_end = min_train + (i + 1) * test_per_fold
        test_start = min_train + i * test_per_fold
        train_end = test_start
        
        if test_end > n:
            break
        
        splits.append((
            list(range(0, train_end)),
            list(range(test_start, test_end))
        ))
    
    return splits

def compute_metrics(returns, annualize=True):
    """Compute strategy metrics from a return series."""
    if len(returns) == 0 or returns.std() == 0:
        return {
            "sharpe": 0.0, "total_return": 0.0, "win_rate": 0.0,
            "max_dd": 0.0, "profit_factor": 0.0, "n_trades": 0
        }
    
    factor = np.sqrt(365) if annualize else 1.0
    sharpe = returns.mean() / returns.std() * factor
    total_ret = (1 + returns).prod() - 1
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')
    
    # Count trades (signal changes)
    n_trades = len(returns[returns != 0])
    
    return {
        "sharpe": round(sharpe, 4),
        "total_return": round(total_ret * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "n_trades": n_trades
    }

def apply_signal_to_returns(signal, price, cost=TRANSACTION_COST):
    """
    Convert signal series to strategy returns.
    Signal: -1 (short), 0 (flat), +1 (long)
    """
    # Daily returns of underlying
    underlying_ret = price.pct_change()
    
    # Align
    signal = signal.reindex(underlying_ret.index).fillna(0)
    
    # Strategy returns = signal_t-1 * return_t (signal at close, execute next day)
    strat_ret = signal.shift(1) * underlying_ret
    
    # Transaction costs on signal changes
    trades = signal.diff().abs()
    strat_ret = strat_ret - trades * cost
    
    return strat_ret.dropna()

def walk_forward_validate(signal_func, data, n_folds=5, label="Signal"):
    """
    Run walk-forward validation on a signal function.
    signal_func(train_data, test_data) -> signal series for test period
    """
    n = len(data)
    splits = expanding_window_splits(n, n_folds)
    
    results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        try:
            signal = signal_func(train, test)
            strat_returns = apply_signal_to_returns(signal, test["close"])
            metrics = compute_metrics(strat_returns)
            metrics["train_start"] = str(train.index[0].date())
            metrics["train_end"] = str(train.index[-1].date())
            metrics["test_start"] = str(test.index[0].date())
            metrics["test_end"] = str(test.index[-1].date())
            metrics["fold"] = i + 1
            results.append(metrics)
        except Exception as e:
            print(f"  Fold {i+1} failed: {e}")
            results.append({
                "fold": i+1, "sharpe": 0, "total_return": 0, "win_rate": 0,
                "max_dd": 0, "profit_factor": 0, "n_trades": 0,
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "error": str(e)
            })
    
    return results

def aggregate_results(fold_results):
    """Compute aggregate statistics from fold results."""
    sharpes = [r["sharpe"] for r in fold_results]
    returns = [r["total_return"] for r in fold_results]
    
    n = len(sharpes)
    mean_sharpe = np.mean(sharpes)
    
    # t-test: is mean Sharpe > 0?
    if n > 1 and np.std(sharpes) > 0:
        t_stat, p_value = stats.ttest_1samp(sharpes, 0)
        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    else:
        t_stat, p_one_sided = 0, 1.0
    
    return {
        "mean_sharpe": round(mean_sharpe, 4),
        "median_sharpe": round(np.median(sharpes), 4),
        "min_sharpe": round(np.min(sharpes), 4),
        "max_sharpe": round(np.max(sharpes), 4),
        "pct_profitable": round(sum(1 for r in returns if r > 0) / n * 100, 1),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_one_sided, 4),
        "n_folds": n
    }

# ============================================================
# SIGNAL IMPLEMENTATIONS
# ============================================================

def signal_1_ofi_proxy(train, test):
    """
    Signal 1: OFI proxy using OI change / market_cap
    Since taker data is only 21 days, use OI change as proxy.
    market_cap ≈ close * circulating_supply (use close * constant as proxy)
    """
    # OI change as order flow proxy
    if "oi_close" not in test.columns:
        raise ValueError("No OI data available")
    
    # Compute OI change
    all_data = pd.concat([train, test])
    oi_change = all_data["oi_close"].diff()
    
    # Normalize by price (proxy for market cap — same direction, proportional)
    ofi_mcap = oi_change / all_data["close"]
    
    # Use training period to set thresholds
    train_ofi = ofi_mcap.loc[train.index]
    threshold_long = train_ofi.quantile(0.7)
    threshold_short = train_ofi.quantile(0.3)
    
    # Generate test signals
    test_ofi = ofi_mcap.loc[test.index]
    signal = pd.Series(0, index=test.index)
    signal[test_ofi > threshold_long] = 1
    signal[test_ofi < threshold_short] = -1
    
    return signal

def signal_2_dar_funding(train, test):
    """
    Signal 2: DAR(1) Funding Rate Prediction
    funding_t = c + φ*funding_{t-1} + ε_t, var(ε) = α + β*funding_{t-1}²
    Simple: fit AR(1) on training, predict one-step-ahead on test.
    If predicted funding > threshold → short (crowded longs). If < -threshold → long.
    """
    if "funding_rate" not in train.columns:
        raise ValueError("No funding data")
    
    fr_train = train["funding_rate"].dropna()
    
    if len(fr_train) < 30:
        raise ValueError("Insufficient training data for DAR")
    
    # Fit simple AR(1): funding_t = c + phi * funding_{t-1}
    y = fr_train.iloc[1:].values
    x = fr_train.iloc[:-1].values
    
    # OLS
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phi = beta[0], beta[1]
    
    # Residuals for variance model
    residuals = y - (c + phi * x)
    
    # DAR variance: var = alpha + beta * funding_{t-1}^2
    res_sq = residuals ** 2
    X_var = np.column_stack([np.ones(len(x)), x**2])
    var_beta = np.linalg.lstsq(X_var, res_sq, rcond=None)[0]
    
    # Thresholds from training: use mean ± 0.5*std of funding
    fr_mean = fr_train.mean()
    fr_std = fr_train.std()
    threshold = fr_std * 0.5
    
    # Predict one-step-ahead on test
    all_fr = pd.concat([train["funding_rate"], test["funding_rate"]])
    signal = pd.Series(0.0, index=test.index)
    
    for i, dt in enumerate(test.index):
        # Previous funding rate
        idx_pos = all_fr.index.get_loc(dt)
        prev_fr = all_fr.iloc[idx_pos - 1]
        
        # Predicted funding
        pred_fr = c + phi * prev_fr
        
        # Signal: high predicted funding → crowded longs → short
        if pred_fr > fr_mean + threshold:
            signal.iloc[i] = -1
        elif pred_fr < fr_mean - threshold:
            signal.iloc[i] = 1
    
    return signal

def signal_3_har_vol(train, test):
    """
    Signal 3: HAR Funding→Vol
    Not a directional signal — measures vol forecast improvement.
    Returns dummy signal; we report forecast accuracy instead.
    """
    raise NotImplementedError("VOL_FORECAST")

def har_vol_analysis(data, n_folds=5):
    """
    HAR model with/without funding rate for vol forecasting.
    RV_t = c + β_d*RV_{t-1} + β_w*mean(RV_{t-5:t-1}) + β_m*mean(RV_{t-22:t-1}) [+ γ*FR_{t-1}]
    """
    # Compute realized volatility (daily absolute returns as proxy)
    data = data.copy()
    data["rv"] = np.log(data["close"] / data["close"].shift(1)).abs()
    data["rv_lag1"] = data["rv"].shift(1)
    data["rv_week"] = data["rv"].rolling(5).mean().shift(1)
    data["rv_month"] = data["rv"].rolling(22).mean().shift(1)
    
    has_funding = "funding_rate" in data.columns
    if has_funding:
        data["fr_lag1"] = data["funding_rate"].shift(1)
    
    data = data.dropna()
    
    splits = expanding_window_splits(len(data), n_folds)
    
    results_base = []
    results_enhanced = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        # Baseline HAR (no funding)
        features_base = ["rv_lag1", "rv_week", "rv_month"]
        X_train = train[features_base].values
        y_train = train["rv"].values
        X_test = test[features_base].values
        y_test = test["rv"].values
        
        X_train_c = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_c = np.column_stack([np.ones(len(X_test)), X_test])
        
        beta_base = np.linalg.lstsq(X_train_c, y_train, rcond=None)[0]
        pred_base = X_test_c @ beta_base
        
        mae_base = np.mean(np.abs(y_test - pred_base))
        rmse_base = np.sqrt(np.mean((y_test - pred_base)**2))
        
        fold_result_base = {
            "fold": i+1, "mae": round(mae_base, 6), "rmse": round(rmse_base, 6),
            "test_start": str(data.index[test_idx[0]].date()),
            "test_end": str(data.index[test_idx[-1]].date()),
        }
        results_base.append(fold_result_base)
        
        # Enhanced HAR (with funding)
        if has_funding:
            features_enh = ["rv_lag1", "rv_week", "rv_month", "fr_lag1"]
            X_train_e = train[features_enh].values
            X_test_e = test[features_enh].values
            
            X_train_ec = np.column_stack([np.ones(len(X_train_e)), X_train_e])
            X_test_ec = np.column_stack([np.ones(len(X_test_e)), X_test_e])
            
            beta_enh = np.linalg.lstsq(X_train_ec, y_train, rcond=None)[0]
            pred_enh = X_test_ec @ beta_enh
            
            mae_enh = np.mean(np.abs(y_test - pred_enh))
            rmse_enh = np.sqrt(np.mean((y_test - pred_enh)**2))
            
            fold_result_enh = {
                "fold": i+1, "mae": round(mae_enh, 6), "rmse": round(rmse_enh, 6),
                "test_start": fold_result_base["test_start"],
                "test_end": fold_result_base["test_end"],
            }
            results_enhanced.append(fold_result_enh)
    
    return results_base, results_enhanced

def signal_4_lgbm_vol(data, n_folds=5):
    """
    Signal 4: LightGBM Vol Enhancement
    Compare LightGBM vol forecast with/without derivatives features.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None, "LightGBM not installed"
    
    data = data.copy()
    data["rv"] = np.log(data["close"] / data["close"].shift(1)).abs()
    data["rv_lag1"] = data["rv"].shift(1)
    data["rv_lag2"] = data["rv"].shift(2)
    data["rv_lag3"] = data["rv"].shift(3)
    data["rv_week"] = data["rv"].rolling(5).mean().shift(1)
    data["rv_month"] = data["rv"].rolling(22).mean().shift(1)
    data["volume_lag1"] = data["volume"].shift(1)
    
    # Derivatives features
    deriv_features = []
    if "funding_rate" in data.columns:
        data["fr_lag1"] = data["funding_rate"].shift(1)
        deriv_features.append("fr_lag1")
    if "oi_close" in data.columns:
        data["oi_change"] = data["oi_close"].pct_change().shift(1)
        deriv_features.append("oi_change")
    if "long_ratio" in data.columns:
        data["lsr_lag1"] = data["long_ratio"].shift(1)
        deriv_features.append("lsr_lag1")
    if "long_liq" in data.columns:
        data["total_liq_lag1"] = (data["long_liq"] + data["short_liq"]).shift(1)
        deriv_features.append("total_liq_lag1")
    
    data = data.dropna()
    
    base_features = ["rv_lag1", "rv_lag2", "rv_lag3", "rv_week", "rv_month", "volume_lag1"]
    all_features = base_features + deriv_features
    
    splits = expanding_window_splits(len(data), n_folds)
    
    results_base = []
    results_enhanced = []
    
    lgb_params = {
        "objective": "regression", "metric": "mae", "verbosity": -1,
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42
    }
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        y_train = train["rv"].values
        y_test = test["rv"].values
        
        # Baseline
        model_base = lgb.LGBMRegressor(**lgb_params)
        model_base.fit(train[base_features], y_train)
        pred_base = model_base.predict(test[base_features])
        
        mae_base = np.mean(np.abs(y_test - pred_base))
        rmse_base = np.sqrt(np.mean((y_test - pred_base)**2))
        
        results_base.append({
            "fold": i+1, "mae": round(mae_base, 6), "rmse": round(rmse_base, 6),
            "test_start": str(data.index[test_idx[0]].date()),
            "test_end": str(data.index[test_idx[-1]].date()),
        })
        
        # Enhanced
        if deriv_features:
            model_enh = lgb.LGBMRegressor(**lgb_params)
            model_enh.fit(train[all_features], y_train)
            pred_enh = model_enh.predict(test[all_features])
            
            mae_enh = np.mean(np.abs(y_test - pred_enh))
            rmse_enh = np.sqrt(np.mean((y_test - pred_enh)**2))
            
            results_enhanced.append({
                "fold": i+1, "mae": round(mae_enh, 6), "rmse": round(rmse_enh, 6),
                "test_start": results_base[-1]["test_start"],
                "test_end": results_base[-1]["test_end"],
            })
            
            # Feature importance for last fold
            if i == len(splits) - 1:
                importances = dict(zip(all_features, model_enh.feature_importances_))
    
    return results_base, results_enhanced

def audit_tpe_settings():
    """Signal 6: Audit Optuna TPE settings against paper recommendations."""
    # Read the walk_forward_optuna.py to check settings
    config_file = Path.home() / "Desktop/maestro/backend/engine_v2/walk_forward_optuna.py"
    
    findings = []
    with open(config_file) as f:
        content = f.read()
    
    # Check TPESampler usage
    if "TPESampler" in content:
        findings.append("✅ Using TPESampler (correct)")
    else:
        findings.append("❌ Not using TPESampler")
    
    if "multivariate=True" in content:
        findings.append("✅ multivariate=True (recommended by paper)")
    elif "multivariate" in content:
        findings.append("⚠️ multivariate found but not set to True")
    else:
        findings.append("❌ multivariate not set — paper recommends True for correlated params")
    
    if "n_startup_trials" in content:
        # Extract value
        import re
        match = re.search(r'n_startup_trials\s*[=:]\s*(\d+)', content)
        if match:
            val = int(match.group(1))
            if val >= 10:
                findings.append(f"✅ n_startup_trials={val} (paper recommends ≥10)")
            else:
                findings.append(f"⚠️ n_startup_trials={val} (paper recommends ≥10)")
    
    if "HyperbandPruner" in content or "MedianPruner" in content:
        findings.append("✅ Pruning enabled (Hyperband/Median)")
    
    if "group=True" in content:
        findings.append("✅ group=True for TPE (recommended)")
    elif "group" not in content:
        findings.append("⚠️ group not set — paper recommends group=True for structured search")
    
    return findings

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_all():
    print("=" * 70)
    print("SPRINT 1 VALIDATION FRAMEWORK")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    report = []
    report.append("# Sprint 1 Academic Paper Validation Results\n")
    report.append(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"**Transaction costs:** {TRANSACTION_COST*100:.2f}% round-trip\n")
    report.append(f"**Walk-forward:** {MIN_FOLDS} folds, expanding window, {TRAIN_RATIO:.0%} train\n")
    report.append("---\n")
    
    # Load data
    print("\nLoading data...")
    btc_daily = load_btc_daily()
    derivatives = load_derivatives("btc")
    merged = merge_ohlcv_derivatives(btc_daily, derivatives)
    
    print(f"  BTC daily: {len(btc_daily)} rows ({btc_daily.index[0].date()} to {btc_daily.index[-1].date()})")
    print(f"  Merged with derivatives: {len(merged)} rows ({merged.index[0].date()} to {merged.index[-1].date()})")
    print(f"  Columns: {list(merged.columns)}")
    
    # ========================================
    # Signal 1: OFI Proxy
    # ========================================
    print("\n" + "=" * 50)
    print("Signal 1: Matched Filter OFI (Market-Cap Normalization)")
    print("=" * 50)
    
    # Check taker data availability
    taker_rows = len(derivatives.get("taker", pd.DataFrame()))
    report.append("\n## Signal 1: Matched Filter OFI (Market-Cap Normalization)\n")
    
    if taker_rows < 100:
        print(f"  Taker data: only {taker_rows} hourly rows (~{taker_rows//24} days) — INSUFFICIENT for walk-forward")
        print("  Using OI change / price as proxy for OFI...")
    
    if "oi_close" in merged.columns:
        fold_results = walk_forward_validate(signal_1_ofi_proxy, merged, MIN_FOLDS, "OFI Proxy")
        agg = aggregate_results(fold_results)
        
        report.append(f"**Data:** BTC OHLCV + OI daily, {merged.index[0].date()} to {merged.index[-1].date()}, {len(merged)} rows\n")
        report.append(f"**Method:** OI change / price as OFI proxy. Thresholds: 30th/70th percentile from training.\n")
        report.append(f"**Note:** Original paper uses L2 order book OFI normalized by market cap. We use OI change as proxy.\n")
        report.append(f"**Walk-Forward:** {MIN_FOLDS} folds, expanding window\n\n")
        
        report.append("| Fold | Train Period | Test Period | OOS Sharpe | OOS Return | Win Rate | MaxDD |\n")
        report.append("|------|-------------|-------------|------------|------------|----------|-------|\n")
        for r in fold_results:
            report.append(f"| {r['fold']} | {r.get('train_start','')} to {r.get('train_end','')} | {r.get('test_start','')} to {r.get('test_end','')} | {r['sharpe']:.2f} | {r['total_return']:.1f}% | {r['win_rate']:.1f}% | {r['max_dd']:.1f}% |\n")
        
        report.append(f"\n**Aggregate:**\n")
        report.append(f"- Mean OOS Sharpe: {agg['mean_sharpe']:.2f}\n")
        report.append(f"- Median OOS Sharpe: {agg['median_sharpe']:.2f}\n")
        report.append(f"- Min OOS Sharpe: {agg['min_sharpe']:.2f}\n")
        report.append(f"- % Folds Profitable: {agg['pct_profitable']:.0f}%\n")
        report.append(f"- t-stat (Sharpe > 0): {agg['t_stat']:.2f} (p={agg['p_value']:.3f})\n")
        
        verdict = "IMPLEMENT" if agg['mean_sharpe'] > 0.5 and agg['p_value'] < 0.1 else \
                  "NEEDS MORE DATA" if agg['p_value'] > 0.3 else "SKIP"
        report.append(f"\n**VERDICT: {verdict}**\n")
        report.append(f"**Reason:** {'OI proxy may not capture true OFI dynamics. Original requires L2 book data.' if verdict != 'IMPLEMENT' else 'Signal shows consistent OOS performance.'}\n")
        
        print_fold_table(fold_results, agg)
    else:
        report.append("**VERDICT: SKIP — No OI data available**\n")
    
    # ========================================
    # Signal 2: DAR Funding Rate
    # ========================================
    print("\n" + "=" * 50)
    print("Signal 2: DAR Funding Rate Prediction (Inan)")
    print("=" * 50)
    
    report.append("\n---\n\n## Signal 2: DAR Funding Rate Prediction (Inan)\n")
    
    if "funding_rate" in merged.columns:
        fold_results = walk_forward_validate(signal_2_dar_funding, merged, MIN_FOLDS, "DAR Funding")
        agg = aggregate_results(fold_results)
        
        report.append(f"**Data:** BTC OHLCV + funding rate, {merged.index[0].date()} to {merged.index[-1].date()}, {len(merged)} rows\n")
        report.append(f"**Method:** DAR(1): funding_t = c + φ*funding_{{t-1}}. Signal: predicted FR > mean+0.5σ → short; < mean-0.5σ → long.\n")
        report.append(f"**Walk-Forward:** {MIN_FOLDS} folds, expanding window\n\n")
        
        report.append("| Fold | Train Period | Test Period | OOS Sharpe | OOS Return | Win Rate | MaxDD |\n")
        report.append("|------|-------------|-------------|------------|------------|----------|-------|\n")
        for r in fold_results:
            report.append(f"| {r['fold']} | {r.get('train_start','')} to {r.get('train_end','')} | {r.get('test_start','')} to {r.get('test_end','')} | {r['sharpe']:.2f} | {r['total_return']:.1f}% | {r['win_rate']:.1f}% | {r['max_dd']:.1f}% |\n")
        
        report.append(f"\n**Aggregate:**\n")
        report.append(f"- Mean OOS Sharpe: {agg['mean_sharpe']:.2f}\n")
        report.append(f"- Median OOS Sharpe: {agg['median_sharpe']:.2f}\n")
        report.append(f"- Min OOS Sharpe: {agg['min_sharpe']:.2f}\n")
        report.append(f"- % Folds Profitable: {agg['pct_profitable']:.0f}%\n")
        report.append(f"- t-stat (Sharpe > 0): {agg['t_stat']:.2f} (p={agg['p_value']:.3f})\n")
        
        verdict = "IMPLEMENT" if agg['mean_sharpe'] > 0.5 and agg['p_value'] < 0.1 else \
                  "NEEDS MORE DATA" if agg['mean_sharpe'] > 0 and agg['p_value'] > 0.2 else "SKIP"
        report.append(f"\n**VERDICT: {verdict}**\n")
        reason = "Funding rate mean-reversion via DAR shows "
        reason += "consistent OOS alpha." if verdict == "IMPLEMENT" else "weak/inconsistent signal."
        report.append(f"**Reason:** {reason}\n")
        
        print_fold_table(fold_results, agg)
    else:
        report.append("**VERDICT: SKIP — No funding data in merged set**\n")
    
    # ========================================
    # Signal 3: HAR Funding→Vol
    # ========================================
    print("\n" + "=" * 50)
    print("Signal 3: HAR Funding→Vol (Kim)")
    print("=" * 50)
    
    report.append("\n---\n\n## Signal 3: HAR Funding→Vol (Kim)\n")
    
    if "funding_rate" in merged.columns:
        results_base, results_enh = har_vol_analysis(merged, MIN_FOLDS)
        
        report.append(f"**Data:** BTC OHLCV + funding rate, {merged.index[0].date()} to {merged.index[-1].date()}\n")
        report.append(f"**Method:** HAR model: RV_t = c + β_d·RV_{{t-1}} + β_w·RV_{{t-5}} + β_m·RV_{{t-22}} [+ γ·FR_{{t-1}}]\n")
        report.append(f"**Walk-Forward:** {MIN_FOLDS} folds, expanding window\n\n")
        report.append("**This is a vol forecast model, not a directional signal.**\n\n")
        
        report.append("### Baseline HAR (no funding)\n")
        report.append("| Fold | Test Period | MAE | RMSE |\n")
        report.append("|------|------------|-----|------|\n")
        for r in results_base:
            report.append(f"| {r['fold']} | {r['test_start']} to {r['test_end']} | {r['mae']:.6f} | {r['rmse']:.6f} |\n")
        
        if results_enh:
            report.append("\n### Enhanced HAR (with funding rate)\n")
            report.append("| Fold | Test Period | MAE | RMSE |\n")
            report.append("|------|------------|-----|------|\n")
            for r in results_enh:
                report.append(f"| {r['fold']} | {r['test_start']} to {r['test_end']} | {r['mae']:.6f} | {r['rmse']:.6f} |\n")
            
            # Compare
            base_maes = [r['mae'] for r in results_base]
            enh_maes = [r['mae'] for r in results_enh]
            improvement = [(b - e) / b * 100 for b, e in zip(base_maes, enh_maes)]
            avg_imp = np.mean(improvement)
            
            report.append(f"\n**MAE Improvement:** {avg_imp:.2f}% average across folds\n")
            
            # Diebold-Mariano style test
            if len(base_maes) > 2:
                t_dm, p_dm = stats.ttest_rel(base_maes, enh_maes)
                report.append(f"**Paired t-test (baseline vs enhanced MAE):** t={t_dm:.2f}, p={p_dm:.3f}\n")
            
            verdict = "IMPLEMENT" if avg_imp > 2 else "MARGINAL" if avg_imp > 0 else "SKIP"
            report.append(f"\n**VERDICT: {verdict}**\n")
            report.append(f"**Reason:** Funding rate {'improves' if avg_imp > 0 else 'does not improve'} vol forecasts by {abs(avg_imp):.1f}% on average OOS.\n")
        
        # Print to console
        print("  Baseline HAR MAE:", [r['mae'] for r in results_base])
        if results_enh:
            print("  Enhanced HAR MAE:", [r['mae'] for r in results_enh])
            print(f"  Avg improvement: {avg_imp:.2f}%")
    
    # ========================================
    # Signal 4: LightGBM Vol Enhancement
    # ========================================
    print("\n" + "=" * 50)
    print("Signal 4: LightGBM Vol Enhancement")
    print("=" * 50)
    
    report.append("\n---\n\n## Signal 4: LightGBM Vol Enhancement\n")
    
    result = signal_4_lgbm_vol(merged, MIN_FOLDS)
    if result[0] is None:
        report.append(f"**VERDICT: SKIP — {result[1]}**\n")
        print(f"  {result[1]}")
    else:
        results_base, results_enh = result
        
        report.append(f"**Data:** BTC OHLCV + all derivatives, {merged.index[0].date()} to {merged.index[-1].date()}\n")
        report.append(f"**Method:** LightGBM regression predicting next-day RV.\n")
        report.append(f"**Base features:** lagged RV (1,2,3), weekly RV, monthly RV, volume\n")
        report.append(f"**Enhanced features:** + funding rate, OI change, LSR, liquidations\n")
        report.append(f"**Walk-Forward:** {MIN_FOLDS} folds, expanding window\n\n")
        
        report.append("### Baseline (OHLCV only)\n")
        report.append("| Fold | Test Period | MAE | RMSE |\n")
        report.append("|------|------------|-----|------|\n")
        for r in results_base:
            report.append(f"| {r['fold']} | {r['test_start']} to {r['test_end']} | {r['mae']:.6f} | {r['rmse']:.6f} |\n")
        
        if results_enh:
            report.append("\n### Enhanced (+ derivatives)\n")
            report.append("| Fold | Test Period | MAE | RMSE |\n")
            report.append("|------|------------|-----|------|\n")
            for r in results_enh:
                report.append(f"| {r['fold']} | {r['test_start']} to {r['test_end']} | {r['mae']:.6f} | {r['rmse']:.6f} |\n")
            
            base_maes = [r['mae'] for r in results_base]
            enh_maes = [r['mae'] for r in results_enh]
            improvement = [(b - e) / b * 100 for b, e in zip(base_maes, enh_maes)]
            avg_imp = np.mean(improvement)
            
            report.append(f"\n**MAE Improvement:** {avg_imp:.2f}% average across folds\n")
            
            if len(base_maes) > 2:
                t_dm, p_dm = stats.ttest_rel(base_maes, enh_maes)
                report.append(f"**Paired t-test:** t={t_dm:.2f}, p={p_dm:.3f}\n")
            
            verdict = "IMPLEMENT" if avg_imp > 3 else "MARGINAL" if avg_imp > 0 else "SKIP"
            report.append(f"\n**VERDICT: {verdict}**\n")
            report.append(f"**Reason:** Derivatives features {'improve' if avg_imp > 0 else 'do not improve'} LightGBM vol forecasts by {abs(avg_imp):.1f}% on average OOS.\n")
        
        print("  Baseline MAE:", [r['mae'] for r in results_base])
        if results_enh:
            print("  Enhanced MAE:", [r['mae'] for r in results_enh])
            print(f"  Avg improvement: {avg_imp:.2f}%")
    
    # ========================================
    # Signal 5: LWI
    # ========================================
    print("\n" + "=" * 50)
    print("Signal 5: LWI (Liquidity Withdrawal Index)")
    print("=" * 50)
    
    report.append("\n---\n\n## Signal 5: LWI (Liquidity Withdrawal Index)\n")
    report.append("**VERDICT: CANNOT IMPLEMENT**\n\n")
    report.append("**Reason:** LWI = cancellations / (depth + additions) requires L2 order book data with:\n")
    report.append("- Order cancellation events\n")
    report.append("- Book depth snapshots\n")
    report.append("- Order additions/modifications\n\n")
    report.append("We have no L2 order book data in our dataset. Would need a live or historical LOB feed ")
    report.append("(e.g., Tardis.dev, Kaiko, or direct exchange websocket recording).\n\n")
    report.append("**Data needed:** L2 order book snapshots or event-level book data at sub-second granularity.\n")
    report.append("**Estimated cost:** $200-500/mo for historical LOB data provider.\n")
    print("  SKIP — No L2 order book data available")
    
    # ========================================
    # Signal 6: TPE Audit
    # ========================================
    print("\n" + "=" * 50)
    print("Signal 6: TPE Configuration Audit")
    print("=" * 50)
    
    report.append("\n---\n\n## Signal 6: TPE Configuration Audit\n")
    
    findings = audit_tpe_settings()
    report.append("**Audit of `walk_forward_optuna.py` against paper recommendations:**\n\n")
    for f in findings:
        report.append(f"- {f}\n")
        print(f"  {f}")
    
    report.append("\n**VERDICT: AUDIT COMPLETE — see findings above**\n")
    
    # ========================================
    # Summary
    # ========================================
    report.append("\n---\n\n## Summary\n\n")
    report.append("| Signal | Type | Verdict | Key Metric |\n")
    report.append("|--------|------|---------|------------|\n")
    report.append("| 1. OFI Proxy | Directional | See above | OOS Sharpe |\n")
    report.append("| 2. DAR Funding | Directional | See above | OOS Sharpe |\n")
    report.append("| 3. HAR Vol | Vol Forecast | See above | MAE improvement |\n")
    report.append("| 4. LightGBM Vol | Vol Forecast | See above | MAE improvement |\n")
    report.append("| 5. LWI | Directional | CANNOT IMPLEMENT | No data |\n")
    report.append("| 6. TPE Audit | Config | COMPLETE | N/A |\n")
    
    # Save report
    output_path = OUTPUT_DIR / "sprint1_validation_results.md"
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    print(f"\n\nResults saved to: {output_path}")
    return report

def print_fold_table(fold_results, agg):
    """Print fold results to console."""
    print(f"  {'Fold':>4} | {'Sharpe':>7} | {'Return':>8} | {'WinRate':>7} | {'MaxDD':>7}")
    print(f"  {'-'*4} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*7}")
    for r in fold_results:
        print(f"  {r['fold']:>4} | {r['sharpe']:>7.2f} | {r['total_return']:>7.1f}% | {r['win_rate']:>6.1f}% | {r['max_dd']:>6.1f}%")
    print(f"\n  Mean Sharpe: {agg['mean_sharpe']:.2f} | Median: {agg['median_sharpe']:.2f} | t={agg['t_stat']:.2f} (p={agg['p_value']:.3f})")

if __name__ == "__main__":
    run_all()
