#!/usr/bin/env python3
"""
Sprint 1 Revalidation — Canonical 14-fold WF, Multi-Asset
==========================================================
Addresses issues found in methodology audit:
- Uses 14 folds (matching VPS canonical methodology)
- Tests across 5 assets: BTC, ETH, SOL, SUI, LINK
- Properly labels OFI as OI-proxy (not true OFI)
- Reports per-asset and cross-asset aggregate stats
- Uses t-test with sqrt(365) annualization
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
N_FOLDS = 14  # Canonical VPS methodology
MIN_TRAIN_PCT = 0.3
ANNUALIZE_FACTOR = np.sqrt(365)

ASSETS = ["btc", "eth", "sol", "sui", "link"]

# ============================================================
# DATA LOADING
# ============================================================

def load_ohlcv(asset):
    """Load daily OHLCV for an asset."""
    fp = OHLCV_DIR / f"binance_{asset}_usdt_1d.csv"
    if not fp.exists():
        raise FileNotFoundError(f"No OHLCV for {asset}")
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df

def load_derivatives(asset):
    """Load all derivatives data for an asset."""
    data = {}
    
    for dtype, suffix, cols in [
        ("funding", "funding_full", None),
        ("oi", "oi_daily_full", ["oi_open", "oi_high", "oi_low", "oi_close"]),
        ("liquidations", "liquidations_daily", None),
        ("lsr", "lsr_daily_full", None),
    ]:
        fp = DERIV_DIR / f"{asset}_{suffix}.csv"
        if fp.exists():
            df = pd.read_csv(fp, parse_dates=["timestamp"]).sort_values("timestamp")
            df.set_index("timestamp", inplace=True)
            if cols:
                df.columns = cols
            data[dtype] = df
    
    return data

def merge_data(ohlcv, derivatives):
    """Merge OHLCV with all derivatives on date."""
    merged = ohlcv.copy()
    for name, df in derivatives.items():
        merged = merged.join(df, how="left")
    
    # Keep only rows with funding_rate (our primary derivatives column)
    if "funding_rate" in merged.columns:
        merged = merged.dropna(subset=["funding_rate"])
    
    return merged

def load_asset(asset):
    """Load and merge everything for one asset."""
    ohlcv = load_ohlcv(asset)
    derivs = load_derivatives(asset)
    merged = merge_data(ohlcv, derivs)
    return merged

# ============================================================
# WALK-FORWARD ENGINE (14 folds, expanding window)
# ============================================================

def expanding_window_splits(n, n_folds=14, min_train_pct=0.3):
    """
    Generate expanding window train/test splits.
    Total test region = last (1-min_train_pct) of data, divided into n_folds chunks.
    """
    min_train = int(n * min_train_pct)
    total_test = n - min_train
    test_per_fold = total_test // n_folds
    
    if test_per_fold < 10:
        # Not enough data for this many folds, reduce
        n_folds = max(total_test // 15, 3)
        test_per_fold = total_test // n_folds
    
    splits = []
    for i in range(n_folds):
        test_start = min_train + i * test_per_fold
        test_end = min_train + (i + 1) * test_per_fold
        if i == n_folds - 1:
            test_end = n  # Last fold gets remainder
        
        if test_end > n:
            break
        
        splits.append((
            list(range(0, test_start)),
            list(range(test_start, test_end))
        ))
    
    return splits

def compute_metrics(returns):
    """Compute strategy metrics from return series."""
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "win_rate": 0.0, "max_dd": 0.0, "n_days": 0}
    
    sharpe = returns.mean() / returns.std() * ANNUALIZE_FACTOR
    total_ret = (1 + returns).prod() - 1
    win_rate = (returns > 0).sum() / len(returns)
    
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return {
        "sharpe": round(sharpe, 4),
        "total_return": round(total_ret * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "n_days": len(returns)
    }

def apply_signal(signal, price, cost=TRANSACTION_COST):
    """Convert signal to strategy returns. Signal at t, execute at t+1."""
    underlying_ret = price.pct_change()
    signal = signal.reindex(underlying_ret.index).fillna(0)
    strat_ret = signal.shift(1) * underlying_ret
    trades = signal.diff().abs()
    strat_ret = strat_ret - trades * cost
    return strat_ret.dropna()

# ============================================================
# SIGNAL IMPLEMENTATIONS
# ============================================================

def signal_oi_proxy(train, test):
    """OI change / price as OFI proxy. NOT true OFI (requires L2 book)."""
    if "oi_close" not in test.columns:
        raise ValueError("No OI data")
    
    all_data = pd.concat([train, test])
    oi_change = all_data["oi_close"].diff()
    ofi_proxy = oi_change / all_data["close"]
    
    train_vals = ofi_proxy.loc[train.index].dropna()
    threshold_long = train_vals.quantile(0.7)
    threshold_short = train_vals.quantile(0.3)
    
    test_vals = ofi_proxy.loc[test.index]
    signal = pd.Series(0, index=test.index)
    signal[test_vals > threshold_long] = 1
    signal[test_vals < threshold_short] = -1
    return signal

def signal_dar_funding(train, test):
    """DAR(1) funding rate prediction → directional signal."""
    if "funding_rate" not in train.columns:
        raise ValueError("No funding data")
    
    fr_train = train["funding_rate"].dropna()
    if len(fr_train) < 30:
        raise ValueError("Insufficient training data")
    
    y = fr_train.iloc[1:].values
    x = fr_train.iloc[:-1].values
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phi = beta[0], beta[1]
    
    fr_mean = fr_train.mean()
    fr_std = fr_train.std()
    threshold = fr_std * 0.5
    
    all_fr = pd.concat([train["funding_rate"], test["funding_rate"]])
    signal = pd.Series(0.0, index=test.index)
    
    for i, dt in enumerate(test.index):
        idx_pos = all_fr.index.get_loc(dt)
        prev_fr = all_fr.iloc[idx_pos - 1]
        pred_fr = c + phi * prev_fr
        
        if pred_fr > fr_mean + threshold:
            signal.iloc[i] = -1
        elif pred_fr < fr_mean - threshold:
            signal.iloc[i] = 1
    
    return signal

def run_directional_wf(signal_func, data, label="Signal"):
    """Run walk-forward on a directional signal."""
    n = len(data)
    splits = expanding_window_splits(n, N_FOLDS)
    actual_folds = len(splits)
    
    results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        try:
            signal = signal_func(train, test)
            strat_returns = apply_signal(signal, test["close"])
            metrics = compute_metrics(strat_returns)
            metrics["fold"] = i + 1
            metrics["test_start"] = str(data.index[test_idx[0]].date())
            metrics["test_end"] = str(data.index[test_idx[-1]].date())
            results.append(metrics)
        except Exception as e:
            results.append({
                "fold": i+1, "sharpe": 0, "total_return": 0, "win_rate": 0,
                "max_dd": 0, "n_days": 0, "error": str(e),
                "test_start": str(data.index[test_idx[0]].date()),
                "test_end": str(data.index[test_idx[-1]].date()),
            })
    
    return results, actual_folds

def run_vol_forecast_wf(data, use_lgbm=False):
    """Run walk-forward on vol forecast (HAR or LightGBM)."""
    data = data.copy()
    data["rv"] = np.log(data["close"] / data["close"].shift(1)).abs()
    data["rv_lag1"] = data["rv"].shift(1)
    data["rv_lag2"] = data["rv"].shift(2)
    data["rv_lag3"] = data["rv"].shift(3)
    data["rv_week"] = data["rv"].rolling(5).mean().shift(1)
    data["rv_month"] = data["rv"].rolling(22).mean().shift(1)
    
    if use_lgbm:
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
    if "long_liq" in data.columns and "short_liq" in data.columns:
        data["total_liq_lag1"] = (data["long_liq"] + data["short_liq"]).shift(1)
        deriv_features.append("total_liq_lag1")
    
    data = data.dropna()
    
    if use_lgbm:
        base_features = ["rv_lag1", "rv_lag2", "rv_lag3", "rv_week", "rv_month", "volume_lag1"]
    else:
        base_features = ["rv_lag1", "rv_week", "rv_month"]
    
    all_features = base_features + deriv_features
    
    splits = expanding_window_splits(len(data), N_FOLDS)
    actual_folds = len(splits)
    
    base_maes = []
    enh_maes = []
    fold_details = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        y_train = train["rv"].values
        y_test = test["rv"].values
        
        if use_lgbm:
            try:
                import lightgbm as lgb
                lgb_params = {
                    "objective": "regression", "metric": "mae", "verbosity": -1,
                    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42
                }
                
                model_base = lgb.LGBMRegressor(**lgb_params)
                model_base.fit(train[base_features], y_train)
                pred_base = model_base.predict(test[base_features])
                
                model_enh = lgb.LGBMRegressor(**lgb_params)
                model_enh.fit(train[all_features], y_train)
                pred_enh = model_enh.predict(test[all_features])
            except ImportError:
                return None, None, None, "LightGBM not installed"
        else:
            # OLS HAR
            X_train_b = np.column_stack([np.ones(len(train)), train[base_features].values])
            X_test_b = np.column_stack([np.ones(len(test)), test[base_features].values])
            beta_b = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
            pred_base = X_test_b @ beta_b
            
            X_train_e = np.column_stack([np.ones(len(train)), train[all_features].values])
            X_test_e = np.column_stack([np.ones(len(test)), test[all_features].values])
            beta_e = np.linalg.lstsq(X_train_e, y_train, rcond=None)[0]
            pred_enh = X_test_e @ beta_e
        
        mae_b = np.mean(np.abs(y_test - pred_base))
        mae_e = np.mean(np.abs(y_test - pred_enh))
        base_maes.append(mae_b)
        enh_maes.append(mae_e)
        
        fold_details.append({
            "fold": i+1,
            "mae_base": round(mae_b, 6),
            "mae_enhanced": round(mae_e, 6),
            "improvement_pct": round((mae_b - mae_e) / mae_b * 100, 2),
            "test_start": str(data.index[test_idx[0]].date()),
            "test_end": str(data.index[test_idx[-1]].date()),
            "n_days": len(test_idx),
        })
    
    return base_maes, enh_maes, fold_details, actual_folds

def aggregate_sharpes(fold_results):
    """Aggregate directional signal results."""
    sharpes = [r["sharpe"] for r in fold_results if "error" not in r]
    if len(sharpes) < 3:
        return {"mean_sharpe": 0, "t_stat": 0, "p_value": 1.0, "n_folds": len(sharpes), "pct_positive": 0}
    
    mean_s = np.mean(sharpes)
    t_stat, p_val = stats.ttest_1samp(sharpes, 0)
    p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    
    return {
        "mean_sharpe": round(mean_s, 4),
        "median_sharpe": round(np.median(sharpes), 4),
        "std_sharpe": round(np.std(sharpes), 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_one, 4),
        "n_folds": len(sharpes),
        "pct_positive": round(sum(1 for s in sharpes if s > 0) / len(sharpes) * 100, 1),
    }

def aggregate_vol(base_maes, enh_maes):
    """Aggregate vol forecast results."""
    improvements = [(b - e) / b * 100 for b, e in zip(base_maes, enh_maes)]
    avg_imp = np.mean(improvements)
    
    if len(base_maes) > 2:
        t_stat, p_val = stats.ttest_rel(base_maes, enh_maes)
    else:
        t_stat, p_val = 0, 1.0
    
    return {
        "avg_improvement_pct": round(avg_imp, 2),
        "median_improvement_pct": round(np.median(improvements), 2),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "n_folds": len(base_maes),
        "pct_improved": round(sum(1 for i in improvements if i > 0) / len(improvements) * 100, 1),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SPRINT 1 REVALIDATION — 14-Fold WF, Multi-Asset")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Assets: {', '.join(a.upper() for a in ASSETS)}")
    print(f"Folds: {N_FOLDS} (expanding window)")
    print(f"Transaction cost: {TRANSACTION_COST*100:.2f}%")
    print(f"Annualization: sqrt(365)")
    print("=" * 70)
    
    report = []
    report.append("# Sprint 1 Revalidation — 14-Fold WF, Multi-Asset\n\n")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
    report.append(f"**Assets:** {', '.join(a.upper() for a in ASSETS)}  \n")
    report.append(f"**Folds:** {N_FOLDS} target (expanding window, min 30% train)  \n")
    report.append(f"**Transaction cost:** {TRANSACTION_COST*100:.2f}% round-trip  \n")
    report.append(f"**Annualization:** √365  \n\n")
    report.append("---\n\n")
    
    # Store cross-asset results
    all_results = {
        "oi_proxy": {},
        "dar_funding": {},
        "har_vol": {},
        "lgbm_vol": {},
    }
    
    for asset in ASSETS:
        print(f"\n{'='*50}")
        print(f"  ASSET: {asset.upper()}")
        print(f"{'='*50}")
        
        try:
            data = load_asset(asset)
        except Exception as e:
            print(f"  ERROR loading {asset}: {e}")
            continue
        
        print(f"  Data: {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Columns: {[c for c in data.columns if c not in ['open','high','low','volume']]}")
        
        report.append(f"## {asset.upper()}\n\n")
        report.append(f"**Data:** {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}  \n")
        report.append(f"**Available features:** {', '.join(c for c in data.columns if c not in ['open','high','low','close','volume'])}  \n\n")
        
        # --- Signal 1: OI Proxy ---
        print(f"\n  Signal 1: OI Proxy (NOT true OFI)")
        try:
            folds, actual_n = run_directional_wf(signal_oi_proxy, data, "OI Proxy")
            agg = aggregate_sharpes(folds)
            all_results["oi_proxy"][asset] = agg
            
            report.append(f"### Signal 1: OI Proxy\n\n")
            report.append(f"*Note: This uses OI change / price as proxy. NOT true L2 book OFI.*  \n")
            report.append(f"Actual folds: {actual_n}  \n\n")
            report.append(f"| Fold | Test Period | Sharpe | Return | Win% | MaxDD |\n")
            report.append(f"|------|------------|--------|--------|------|-------|\n")
            for r in folds:
                err = f" ⚠️{r.get('error','')}" if 'error' in r else ""
                report.append(f"| {r['fold']} | {r['test_start']} → {r['test_end']} | {r['sharpe']:.2f} | {r['total_return']:.1f}% | {r['win_rate']:.1f}% | {r['max_dd']:.1f}%{err} |\n")
            
            report.append(f"\n**Mean Sharpe:** {agg['mean_sharpe']:.2f} | **t:** {agg['t_stat']:.2f} | **p:** {agg['p_value']:.3f} | **% positive:** {agg['pct_positive']:.0f}%  \n\n")
            
            print(f"    Mean Sharpe: {agg['mean_sharpe']:.2f}, t={agg['t_stat']:.2f}, p={agg['p_value']:.3f}, {agg['pct_positive']:.0f}% positive folds")
        except Exception as e:
            print(f"    FAILED: {e}")
            report.append(f"### Signal 1: OI Proxy — FAILED: {e}\n\n")
        
        # --- Signal 2: DAR Funding ---
        print(f"\n  Signal 2: DAR Funding")
        try:
            folds, actual_n = run_directional_wf(signal_dar_funding, data, "DAR Funding")
            agg = aggregate_sharpes(folds)
            all_results["dar_funding"][asset] = agg
            
            report.append(f"### Signal 2: DAR Funding\n\n")
            report.append(f"Actual folds: {actual_n}  \n\n")
            report.append(f"| Fold | Test Period | Sharpe | Return | Win% | MaxDD |\n")
            report.append(f"|------|------------|--------|--------|------|-------|\n")
            for r in folds:
                err = f" ⚠️{r.get('error','')}" if 'error' in r else ""
                report.append(f"| {r['fold']} | {r['test_start']} → {r['test_end']} | {r['sharpe']:.2f} | {r['total_return']:.1f}% | {r['win_rate']:.1f}% | {r['max_dd']:.1f}%{err} |\n")
            
            report.append(f"\n**Mean Sharpe:** {agg['mean_sharpe']:.2f} | **t:** {agg['t_stat']:.2f} | **p:** {agg['p_value']:.3f} | **% positive:** {agg['pct_positive']:.0f}%  \n\n")
            
            print(f"    Mean Sharpe: {agg['mean_sharpe']:.2f}, t={agg['t_stat']:.2f}, p={agg['p_value']:.3f}, {agg['pct_positive']:.0f}% positive folds")
        except Exception as e:
            print(f"    FAILED: {e}")
            report.append(f"### Signal 2: DAR Funding — FAILED: {e}\n\n")
        
        # --- Signal 3: HAR Vol ---
        print(f"\n  Signal 3: HAR Vol")
        try:
            base_m, enh_m, details, actual_n = run_vol_forecast_wf(data, use_lgbm=False)
            if base_m is None:
                raise ValueError(str(actual_n))
            agg = aggregate_vol(base_m, enh_m)
            all_results["har_vol"][asset] = agg
            
            report.append(f"### Signal 3: HAR Vol (funding enhancement)\n\n")
            report.append(f"Actual folds: {actual_n}  \n\n")
            report.append(f"| Fold | Test Period | MAE Base | MAE Enh | Δ% |\n")
            report.append(f"|------|------------|----------|---------|----|\n")
            for d in details:
                report.append(f"| {d['fold']} | {d['test_start']} → {d['test_end']} | {d['mae_base']:.6f} | {d['mae_enhanced']:.6f} | {d['improvement_pct']:+.2f}% |\n")
            
            report.append(f"\n**Avg improvement:** {agg['avg_improvement_pct']:.2f}% | **t:** {agg['t_stat']:.2f} | **p:** {agg['p_value']:.3f} | **% improved:** {agg['pct_improved']:.0f}%  \n\n")
            
            print(f"    Avg improvement: {agg['avg_improvement_pct']:.2f}%, t={agg['t_stat']:.2f}, p={agg['p_value']:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            report.append(f"### Signal 3: HAR Vol — FAILED: {e}\n\n")
        
        # --- Signal 4: LightGBM Vol ---
        print(f"\n  Signal 4: LightGBM Vol")
        try:
            base_m, enh_m, details, actual_n = run_vol_forecast_wf(data, use_lgbm=True)
            if base_m is None:
                raise ValueError(str(actual_n))
            agg = aggregate_vol(base_m, enh_m)
            all_results["lgbm_vol"][asset] = agg
            
            report.append(f"### Signal 4: LightGBM Vol (derivatives enhancement)\n\n")
            report.append(f"Actual folds: {actual_n}  \n\n")
            report.append(f"| Fold | Test Period | MAE Base | MAE Enh | Δ% |\n")
            report.append(f"|------|------------|----------|---------|----|\n")
            for d in details:
                report.append(f"| {d['fold']} | {d['test_start']} → {d['test_end']} | {d['mae_base']:.6f} | {d['mae_enhanced']:.6f} | {d['improvement_pct']:+.2f}% |\n")
            
            report.append(f"\n**Avg improvement:** {agg['avg_improvement_pct']:.2f}% | **t:** {agg['t_stat']:.2f} | **p:** {agg['p_value']:.3f} | **% improved:** {agg['pct_improved']:.0f}%  \n\n")
            
            print(f"    Avg improvement: {agg['avg_improvement_pct']:.2f}%, t={agg['t_stat']:.2f}, p={agg['p_value']:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            report.append(f"### Signal 4: LightGBM Vol — FAILED: {e}\n\n")
        
        report.append("---\n\n")
    
    # ========================================
    # Cross-Asset Summary
    # ========================================
    report.append("## Cross-Asset Summary\n\n")
    
    for signal_name, label in [
        ("oi_proxy", "OI Proxy (directional)"),
        ("dar_funding", "DAR Funding (directional)"),
        ("har_vol", "HAR Vol Enhancement"),
        ("lgbm_vol", "LightGBM Vol Enhancement"),
    ]:
        results = all_results[signal_name]
        report.append(f"### {label}\n\n")
        
        if signal_name in ["oi_proxy", "dar_funding"]:
            report.append(f"| Asset | Mean Sharpe | t-stat | p-value | % Positive Folds |\n")
            report.append(f"|-------|------------|--------|---------|------------------|\n")
            for asset, agg in results.items():
                verdict = "✅" if agg['p_value'] < 0.05 and agg['mean_sharpe'] > 0.5 else "⚠️" if agg['p_value'] < 0.1 else "❌"
                report.append(f"| {asset.upper()} | {agg['mean_sharpe']:.2f} | {agg['t_stat']:.2f} | {agg['p_value']:.3f} | {agg['pct_positive']:.0f}% {verdict} |\n")
            
            # Cross-asset aggregate
            all_sharpes = []
            for agg in results.values():
                all_sharpes.append(agg['mean_sharpe'])
            if all_sharpes:
                cross_mean = np.mean(all_sharpes)
                report.append(f"\n**Cross-asset mean Sharpe:** {cross_mean:.2f}  \n")
        else:
            report.append(f"| Asset | Avg Δ MAE% | t-stat | p-value | % Improved Folds |\n")
            report.append(f"|-------|-----------|--------|---------|------------------|\n")
            for asset, agg in results.items():
                verdict = "✅" if agg['p_value'] < 0.05 and agg['avg_improvement_pct'] > 2 else "⚠️" if agg['p_value'] < 0.1 else "❌"
                report.append(f"| {asset.upper()} | {agg['avg_improvement_pct']:+.2f}% | {agg['t_stat']:.2f} | {agg['p_value']:.3f} | {agg['pct_improved']:.0f}% {verdict} |\n")
            
            all_imps = [agg['avg_improvement_pct'] for agg in results.values()]
            if all_imps:
                cross_imp = np.mean(all_imps)
                report.append(f"\n**Cross-asset mean improvement:** {cross_imp:+.2f}%  \n")
        
        report.append("\n")
    
    # ========================================
    # Final Verdicts
    # ========================================
    report.append("## Final Verdicts\n\n")
    report.append("| Signal | Original Verdict | Revalidation Verdict | Confidence |\n")
    report.append("|--------|-----------------|---------------------|------------|\n")
    
    # Determine verdicts based on cross-asset results
    for signal_name, orig_verdict in [
        ("oi_proxy", "❌ REJECTED"),
        ("dar_funding", "❌ REJECTED"),
        ("har_vol", "⚠️ MARGINAL"),
        ("lgbm_vol", "✅ IMPLEMENT"),
    ]:
        results = all_results[signal_name]
        if not results:
            report.append(f"| {signal_name} | {orig_verdict} | NO DATA | — |\n")
            continue
        
        if signal_name in ["oi_proxy", "dar_funding"]:
            passing = sum(1 for a in results.values() if a['p_value'] < 0.1 and a['mean_sharpe'] > 0.3)
            total = len(results)
            if passing >= total * 0.6:
                new_verdict = "✅ IMPLEMENT"
                conf = "HIGH" if passing == total else "MODERATE"
            elif passing >= 1:
                new_verdict = "⚠️ MARGINAL"
                conf = "LOW"
            else:
                new_verdict = "❌ REJECTED"
                conf = "HIGH"
        else:
            passing = sum(1 for a in results.values() if a['p_value'] < 0.1 and a['avg_improvement_pct'] > 1)
            total = len(results)
            if passing >= total * 0.6:
                new_verdict = "✅ IMPLEMENT"
                conf = "HIGH" if passing == total else "MODERATE"
            elif passing >= 1:
                new_verdict = "⚠️ ASSET-SPECIFIC"
                conf = "LOW"
            else:
                new_verdict = "❌ REJECTED"
                conf = "HIGH"
        
        report.append(f"| {signal_name} | {orig_verdict} | {new_verdict} | {conf} |\n")
    
    report.append("\n---\n\n")
    report.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by sprint1_revalidation.py*\n")
    
    # Save
    output_path = OUTPUT_DIR / "sprint1_revalidation_results.md"
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    print(f"\n\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
