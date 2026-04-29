#!/usr/bin/env python3
"""
Walk-Forward Validation for MacroMomentum Strategy — FIXED Parameters (No Optimization)

The "optimization paradox": unoptimized strategies often outperform Optuna-tuned ones OOS.
This script validates the Momentum+Macro system (claimed Sharpe 1.40) using the same
13-fold WFO rigor applied to V3.1, but with NO parameter tuning at all.

If Sharpe 1.40 is full-sample, the real OOS number will be lower.
"""
import sys, os, json, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import numpy as np
import pandas as pd

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_momentum import generate_signals, position_size
from strategies.composite.macro_score_builder import compute_macro_score

# ── Config ────────────────────────────────────────────────────
N_FOLDS = 13
TRAIN_DAYS = 504   # 2 years — NOT optimized, just SMA warmup
TEST_DAYS = 126    # 6 months OOS
TX_COST = 0.001    # 10 bps round-trip
N_PERM = 500       # permutation test iterations
N_BOOT = 1000      # bootstrap CI iterations

# Fixed parameters — NO Optuna, NO tuning
PARAMS = {
    "sma_slow": 200,
    "sma_fast": 50,
    "momentum_period": 30,
    "trailing_stop_pct": 0.15,
    "roc_threshold": 0.0,
}


# ── Backtest Engine ───────────────────────────────────────────
def backtest(df, signals, sizes, tx_cost=TX_COST):
    """Vectorized backtest with transaction cost on position changes."""
    close = df["close"].values
    sig = signals.values.astype(float)
    sz = sizes.values.astype(float)
    pos = sig * sz  # generate_signals already does shift(1), no double-shift

    ret = np.diff(close) / close[:-1]
    pos_trunc = pos[:-1]

    # Transaction cost on position changes
    pos_change = np.abs(np.diff(pos))
    tc = np.concatenate([[0], pos_change]) * tx_cost
    tc = tc[:-1]

    pos_ret = pos_trunc * ret - tc

    if len(pos_ret) == 0 or np.all(pos_ret == 0):
        return {
            "sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0,
            "calmar": 0.0, "trades": 0, "win_rate": 0.0,
            "daily_returns": [],
        }

    cum = (1 + pos_ret).cumprod()
    total_ret = cum[-1] - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0.0

    sig_diff = np.diff(signals.values)
    trades = int(np.count_nonzero(sig_diff))

    active = pos_ret[pos_ret != 0]
    daily_mean = np.mean(active) if len(active) > 0 else 0.0
    daily_std = np.std(active) if len(active) > 0 else 1.0
    sharpe = (daily_mean / daily_std) * np.sqrt(365) if daily_std > 0 else 0.0

    winning = np.sum(pos_ret > 0)
    total_active = np.sum(pos_ret != 0)
    win_rate = winning / total_active if total_active > 0 else 0.0

    annual_ret = total_ret / (len(pos_ret) / 365)
    calmar = annual_ret / max_dd if max_dd > 0 else 0.0

    return {
        "sharpe": sharpe, "total_return": total_ret, "max_dd": max_dd,
        "calmar": calmar, "trades": trades, "win_rate": win_rate,
        "daily_returns": pos_ret.tolist(),
    }


def run_strategy(df, macro_sc):
    """Run strategy with fixed params, return metrics."""
    sig = generate_signals(
        df,
        sma_slow=PARAMS["sma_slow"],
        sma_fast=PARAMS["sma_fast"],
        momentum_period=PARAMS["momentum_period"],
        trailing_stop_pct=PARAMS["trailing_stop_pct"],
        roc_threshold=PARAMS["roc_threshold"],
    )
    sz = position_size(sig, macro_sc)
    return backtest(df, sig, sz)


# ── Statistical Tests ─────────────────────────────────────────
def permutation_test(oos_returns, n_perm=N_PERM):
    """p-value via random permutation of daily returns."""
    if len(oos_returns) < 2:
        return 1.0
    observed_sharpe = np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(365)
    count = 0
    for _ in range(n_perm):
        shuffled = np.random.permutation(oos_returns)
        perm_sharpe = np.mean(shuffled) / np.std(shuffled) * np.sqrt(365)
        if perm_sharpe >= observed_sharpe:
            count += 1
    return count / n_perm


def bootstrap_ci(returns, n_boot=N_BOOT, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    if len(returns) < 2:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        std = np.std(sample)
        if std > 0:
            sharpes.append(np.mean(sample) / std * np.sqrt(365))
    if not sharpes:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return {"mean": round(float(np.mean(sharpes)), 4),
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4)}


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading BTC-USD and FRED macro data...")
    stock_loader = StockDataLoader()
    macro_loader = MacroDataLoader()

    btc = stock_loader.get_ohlcv("BTC-USD", "1d", "2017-01-01")
    macro_score_full = compute_macro_score(macro_loader, start_date="2015-01-01")
    macro_score_full = macro_score_full.reindex(btc.index, method="ffill").fillna(0)

    print(f"BTC: {len(btc)} bars, {btc.index[0].date()} to {btc.index[-1].date()}")
    print(f"Fixed params: {PARAMS}")

    n = len(btc)
    folds = []
    all_oos_returns = []
    fold_num = 0
    start = 0

    print(f"\nWALK-FORWARD VALIDATION (Fixed Params - No Optimization)")
    print(f"  Train: {TRAIN_DAYS}d (warmup only) | Test: {TEST_DAYS}d | {N_FOLDS} target folds\n")

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        fold_num += 1
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        # Need full data up to test_end for SMA warmup, but only score test period
        full_df = btc.iloc[:test_end].copy()
        full_macro = macro_score_full.iloc[:test_end].copy()

        test_df = btc.iloc[train_end:test_end].copy()

        # Generate signals on full history (SMA200 needs warmup), extract test period
        sig_full = generate_signals(full_df, **PARAMS)
        sz_full = position_size(sig_full, full_macro)

        # Slice to test period only
        test_sig = sig_full.iloc[train_end:test_end]
        test_sz = sz_full.iloc[train_end:test_end]

        metrics = backtest(test_df, test_sig, test_sz)
        all_oos_returns.extend(metrics.get("daily_returns", []))

        test_start_date = test_df.index[0].strftime("%Y-%m-%d")
        test_end_date = test_df.index[-1].strftime("%Y-%m-%d")

        fold_result = {
            "fold": fold_num,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "sharpe": round(metrics["sharpe"], 4),
            "total_return": round(metrics["total_return"], 4),
            "max_dd": round(metrics["max_dd"], 4),
            "calmar": round(metrics["calmar"], 4),
            "trades": metrics["trades"],
            "win_rate": round(metrics["win_rate"], 4),
        }
        folds.append(fold_result)

        print(f"  Fold {fold_num:2d}: {test_start_date} to {test_end_date} | "
              f"Sharpe={metrics['sharpe']:+6.2f} | "
              f"Ret={metrics['total_return']*100:+7.1f}% | "
              f"MaxDD={metrics['max_dd']*100:-6.1f}% | "
              f"Trades={metrics['trades']}")

        start += TEST_DAYS

    # ── Full-Sample Metrics ───────────────────────────────────
    print("\nComputing full-sample metrics...")
    sig_all = generate_signals(btc, **PARAMS)
    sz_all = position_size(sig_all, macro_score_full)
    full_metrics = backtest(btc, sig_all, sz_all)

    # ── Statistical Tests ─────────────────────────────────────
    oos_daily = np.array(all_oos_returns)
    oos_active = oos_daily[oos_daily != 0] if len(oos_daily) > 0 else oos_daily

    print("Running permutation test (500 iterations)...")
    p_value = permutation_test(oos_active, n_perm=N_PERM) if len(oos_active) > 1 else 1.0

    print("Running bootstrap CI (1000 samples)...")
    boot_ci = bootstrap_ci(oos_active, n_boot=N_BOOT)

    # ── Aggregate OOS Stats ───────────────────────────────────
    oos_sharpes = [f["sharpe"] for f in folds]
    oos_returns_list = [f["total_return"] for f in folds]
    oos_dds = [f["max_dd"] for f in folds]
    positive_folds = sum(1 for s in oos_sharpes if s > 0)

    aggregate = {
        "mean_sharpe": round(float(np.mean(oos_sharpes)), 4),
        "median_sharpe": round(float(np.median(oos_sharpes)), 4),
        "std_sharpe": round(float(np.std(oos_sharpes)), 4),
        "positive_folds": f"{positive_folds}/{len(folds)}",
        "p_value": round(float(p_value), 4),
        "bootstrap_ci": boot_ci,
    }

    full_sample = {
        "sharpe": round(full_metrics["sharpe"], 4),
        "total_return": round(full_metrics["total_return"], 4),
        "max_dd": round(full_metrics["max_dd"], 4),
        "calmar": round(full_metrics["calmar"], 4),
        "trades": full_metrics["trades"],
    }

    # ── Console Report ────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("WALK-FORWARD RESULTS (Fixed Params -- No Optimization)")
    print(sep)
    print(f"{'Fold':>4}  | {'Test Period':<23} | {'Sharpe':>7} | {'Return':>8} | {'MaxDD':>7}")
    print("-" * 70)
    for f in folds:
        print(f"  {f['fold']:2d}  | {f['test_start']} to {f['test_end']} | "
              f"{f['sharpe']:+7.2f} | {f['total_return']*100:+7.1f}% | "
              f"{f['max_dd']*100:6.1f}%")
    print("-" * 70)
    print(f" AVG  | {'':23s} | {aggregate['mean_sharpe']:+7.2f} | "
          f"{np.mean(oos_returns_list)*100:+7.1f}% | {np.mean(oos_dds)*100:6.1f}%")
    print(sep)

    print(f"\nMean OOS Sharpe: {aggregate['mean_sharpe']:.4f}")
    print(f"Median OOS Sharpe: {aggregate['median_sharpe']:.4f}")
    print(f"Positive folds: {aggregate['positive_folds']}")
    print(f"p-value (permutation): {p_value:.4f}")
    print(f"Bootstrap 95% CI: [{boot_ci['lower']:.4f}, {boot_ci['upper']:.4f}]")

    print(f"\nFull-sample: Sharpe={full_sample['sharpe']:.4f}, "
          f"Return={full_sample['total_return']*100:.1f}%, "
          f"MaxDD={full_sample['max_dd']*100:.1f}%, "
          f"Calmar={full_sample['calmar']:.2f}")

    # ── Save Results ──────────────────────────────────────────
    results = {
        "strategy": "MacroMomentum_FixedParams",
        "params": PARAMS,
        "n_folds": len(folds),
        "folds": folds,
        "aggregate_oos": aggregate,
        "full_sample": full_sample,
    }

    save_dir = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "macro_momentum_fixed_wf.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
