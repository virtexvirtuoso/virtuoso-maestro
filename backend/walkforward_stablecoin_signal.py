#!/usr/bin/env python3
"""
Walk-Forward Validation for Stablecoin Supply Signal — FIXED Parameters (No Optimization)

Full-sample showed Sharpe 1.09, spread 90.1%/yr. Is this real or another inflated number?
Same WF rigor as Momentum+Macro validation: rolling 126-day OOS folds, permutation test,
bootstrap CI. Fixed threshold (supply_growth_30d > 0), no tuning.
"""
import sys, os, json, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import numpy as np
import pandas as pd

from datasource.yfinance_loader import StockDataLoader
from datasource.stablecoin_loader import StablecoinLoader

# ── Config ────────────────────────────────────────────────────
TRAIN_DAYS = 504   # 2 years warmup (need 30d for supply growth calc)
TEST_DAYS = 126    # 6 months OOS
TX_COST = 0.001    # 10 bps round-trip
N_PERM = 500
N_BOOT = 1000
SUPPLY_GROWTH_PERIOD = 30  # fixed, no tuning


# ── Signal Generation ────────────────────────────────────────
def generate_signal(close: pd.Series, supply_growth: pd.Series) -> pd.Series:
    """Long when 30d stablecoin supply growth > 0, shifted 1 day for no-lookahead."""
    signal = (supply_growth > 0).astype(float)
    signal = signal.reindex(close.index, method="ffill").fillna(0)
    return signal.shift(1).fillna(0)


# ── Backtest Engine ──────────────────────────────────────────
def backtest_fold(close_vals, position_vals, tx_cost=TX_COST):
    """Compute metrics for a single fold."""
    ret = np.diff(close_vals) / close_vals[:-1]
    pos = position_vals[:-1]

    pos_change = np.abs(np.diff(position_vals))
    tc = np.concatenate([[0], pos_change]) * tx_cost
    tc = tc[:-1]

    pos_ret = pos * ret - tc

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

    trades = int(np.count_nonzero(np.diff(position_vals)))

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


# ── Statistical Tests ────────────────────────────────────────
def permutation_test(oos_returns, n_perm=N_PERM):
    if len(oos_returns) < 2:
        return 1.0
    observed_sharpe = np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(365)
    count = 0
    for _ in range(n_perm):
        shuffled = np.random.permutation(oos_returns)
        std = np.std(shuffled)
        if std > 0:
            perm_sharpe = np.mean(shuffled) / std * np.sqrt(365)
            if perm_sharpe >= observed_sharpe:
                count += 1
    return count / n_perm


def bootstrap_ci(returns, n_boot=N_BOOT, ci=0.95):
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


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading BTC-USD and stablecoin supply data...")
    stock_loader = StockDataLoader()
    stable_loader = StablecoinLoader()

    btc = stock_loader.get_ohlcv("BTC-USD", "1d", "2017-01-01")
    supply_growth = stable_loader.get_supply_growth(period=SUPPLY_GROWTH_PERIOD)

    # Align indexes
    btc.index = pd.DatetimeIndex(btc.index).tz_localize(None)
    supply_growth.index = pd.DatetimeIndex(supply_growth.index).tz_localize(None)

    # Generate signal on full dataset
    signal_full = generate_signal(btc["close"], supply_growth)

    # Restrict to where we have supply data
    valid_start = supply_growth.dropna().index[0]
    btc = btc[btc.index >= valid_start]
    signal_full = signal_full.reindex(btc.index).fillna(0)

    n = len(btc)
    print(f"BTC: {n} bars, {btc.index[0].date()} to {btc.index[-1].date()}")
    print(f"Fixed params: supply_growth_{SUPPLY_GROWTH_PERIOD}d > 0, TX_COST={TX_COST}")

    folds = []
    all_oos_returns = []
    fold_num = 0
    start = 0

    print(f"\nWALK-FORWARD VALIDATION (Stablecoin Supply Signal - Fixed Params)")
    print(f"  Train: {TRAIN_DAYS}d (warmup) | Test: {TEST_DAYS}d\n")

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        fold_num += 1
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        test_close = btc["close"].iloc[train_end:test_end].values
        test_pos = signal_full.iloc[train_end:test_end].values

        metrics = backtest_fold(test_close, test_pos)
        all_oos_returns.extend(metrics.get("daily_returns", []))

        test_start_date = btc.index[train_end].strftime("%Y-%m-%d")
        test_end_date = btc.index[test_end - 1].strftime("%Y-%m-%d")

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

    # ── Full-Sample Metrics ──────────────────────────────────
    print("\nComputing full-sample metrics...")
    full_close = btc["close"].values
    full_pos = signal_full.values
    full_metrics = backtest_fold(full_close, full_pos)

    # ── Statistical Tests ────────────────────────────────────
    oos_daily = np.array(all_oos_returns)
    oos_active = oos_daily[oos_daily != 0] if len(oos_daily) > 0 else oos_daily

    print(f"Running permutation test ({N_PERM} iterations)...")
    p_value = permutation_test(oos_active) if len(oos_active) > 1 else 1.0

    print(f"Running bootstrap CI ({N_BOOT} samples)...")
    boot_ci = bootstrap_ci(oos_active)

    # ── Aggregate OOS Stats ──────────────────────────────────
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

    # ── Console Report ───────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("WALK-FORWARD RESULTS (Stablecoin Supply Signal — Fixed Params)")
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

    # ── Save Results ─────────────────────────────────────────
    results = {
        "strategy": "StablecoinSupply_FixedParams",
        "params": {"supply_growth_period": SUPPLY_GROWTH_PERIOD, "threshold": 0.0},
        "n_folds": len(folds),
        "folds": folds,
        "aggregate_oos": aggregate,
        "full_sample": full_sample,
    }

    save_dir = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "stablecoin_supply_wf.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
