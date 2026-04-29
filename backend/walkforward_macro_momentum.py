#!/usr/bin/env python3
"""
Walk-Forward Validation for MacroMomentum Strategy
Training: 2 years (504 days), Test: 6 months (126 days), rolling forward.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import numpy as np
import pandas as pd
import optuna
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_momentum import generate_signals, position_size
from strategies.composite.macro_score_builder import compute_macro_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_DAYS = 504
TEST_DAYS = 126
OPTUNA_TRIALS = 50


# ── Backtest Engine (same as optimizer) ───────────────────────
def backtest(df, signals, sizes=None):
    close = df["close"].values
    sig = signals.values.astype(float)
    if sizes is not None:
        sig = sig * sizes.values

    ret = np.diff(close) / close[:-1]
    pos_ret = sig[:-1] * ret

    if len(pos_ret) == 0 or np.all(pos_ret == 0):
        return {"sharpe": 0, "total_return": 0, "max_dd": 0, "calmar": 0,
                "trades": 0, "win_rate": 0, "daily_returns": []}

    cum = (1 + pos_ret).cumprod()
    total_ret = cum[-1] - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    sig_diff = np.diff(signals.values)
    trades = np.count_nonzero(sig_diff)

    active = pos_ret[pos_ret != 0]
    daily_mean = np.mean(active) if len(active) > 0 else 0
    daily_std = np.std(active) if len(active) > 0 else 1
    sharpe = (daily_mean / daily_std) * np.sqrt(365) if daily_std > 0 else 0

    winning = np.sum(pos_ret > 0)
    total_active = np.sum(pos_ret != 0)
    win_rate = winning / total_active if total_active > 0 else 0

    annual_ret = total_ret / (len(pos_ret) / 365)
    calmar = annual_ret / max_dd if max_dd > 0 else 0

    return {"sharpe": sharpe, "total_return": total_ret, "max_dd": max_dd,
            "calmar": calmar, "trades": trades, "win_rate": win_rate,
            "daily_returns": pos_ret.tolist()}


def run_strategy(df, macro_sc, params):
    """Run strategy with given params, return metrics."""
    sig = generate_signals(df, sma_slow=params["sma_slow"], sma_fast=params["sma_fast"],
                           momentum_period=params["momentum_period"],
                           trailing_stop_pct=params["trailing_stop_pct"],
                           roc_threshold=params["roc_threshold"])
    sz = position_size(sig, macro_sc,
                       threshold_high=params["macro_threshold_high"],
                       threshold_mid=params["macro_threshold_mid"],
                       threshold_low=params["macro_threshold_low"],
                       size_high=params["position_size_high"],
                       size_mid=params["position_size_mid"],
                       size_low=params["position_size_low"])
    return backtest(df, sig, sz)


# ── Data Loading ──────────────────────────────────────────────
print("Loading data...")
stock_loader = StockDataLoader()
macro_loader = MacroDataLoader()

btc = stock_loader.get_ohlcv("BTC-USD", "1d", "2017-01-01", "2026-02-12")
macro_score_full = compute_macro_score(macro_loader, start_date="2015-01-01", end_date="2026-02-12")
macro_score_full = macro_score_full.reindex(btc.index, method="ffill").fillna(0)

print(f"BTC: {len(btc)} bars, {btc.index[0].date()} to {btc.index[-1].date()}")


# ── Walk-Forward Loop ─────────────────────────────────────────
def make_objective(train_df, train_macro):
    def objective(trial):
        params = {
            "sma_slow": trial.suggest_int("sma_slow", 100, 300, step=10),
            "sma_fast": trial.suggest_int("sma_fast", 20, 100, step=5),
            "momentum_period": trial.suggest_int("momentum_period", 10, 60, step=5),
            "trailing_stop_pct": trial.suggest_float("trailing_stop_pct", 0.08, 0.25),
            "roc_threshold": trial.suggest_float("roc_threshold", 0.0, 0.05),
            "macro_threshold_high": trial.suggest_int("macro_threshold_high", 4, 6),
            "macro_threshold_mid": trial.suggest_int("macro_threshold_mid", 2, 4),
            "macro_threshold_low": trial.suggest_int("macro_threshold_low", 1, 2),
            "position_size_high": trial.suggest_float("position_size_high", 0.8, 1.0),
            "position_size_mid": trial.suggest_float("position_size_mid", 0.4, 0.7),
            "position_size_low": trial.suggest_float("position_size_low", 0.1, 0.4),
        }
        if params["sma_fast"] >= params["sma_slow"]:
            return -10.0
        m = run_strategy(train_df, train_macro, params)
        score = m["sharpe"]
        if m["max_dd"] > 0.30:
            score -= 2.0
        if m["trades"] < 5:
            score -= 3.0
        return score
    return objective


if __name__ == "__main__":
    n = len(btc)
    folds = []
    fold_num = 0
    start = 0

    print(f"\n🔄 Walk-Forward Validation")
    print(f"   Train: {TRAIN_DAYS}d | Test: {TEST_DAYS}d | Optuna: {OPTUNA_TRIALS} trials/fold\n")

    all_oos_returns = []

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        fold_num += 1
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        train_df = btc.iloc[start:train_end].copy()
        test_df = btc.iloc[train_end:test_end].copy()
        train_macro = macro_score_full.iloc[start:train_end].copy()
        test_macro = macro_score_full.iloc[train_end:test_end].copy()

        print(f"  Fold {fold_num}: Train {train_df.index[0].date()}→{train_df.index[-1].date()} | "
              f"Test {test_df.index[0].date()}→{test_df.index[-1].date()}")

        # Optimize on training data
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(train_df, train_macro), n_trials=OPTUNA_TRIALS)
        best = study.best_params

        # IS metrics
        is_metrics = run_strategy(train_df, train_macro, best)
        # OOS metrics
        oos_metrics = run_strategy(test_df, test_macro, best)

        all_oos_returns.extend(oos_metrics.get("daily_returns", []))

        fold_result = {
            "fold": fold_num,
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "best_params": best,
            "is_sharpe": round(is_metrics["sharpe"], 4),
            "is_return": round(is_metrics["total_return"], 4),
            "is_max_dd": round(is_metrics["max_dd"], 4),
            "is_win_rate": round(is_metrics["win_rate"], 4),
            "is_calmar": round(is_metrics["calmar"], 4),
            "oos_sharpe": round(oos_metrics["sharpe"], 4),
            "oos_return": round(oos_metrics["total_return"], 4),
            "oos_max_dd": round(oos_metrics["max_dd"], 4),
            "oos_win_rate": round(oos_metrics["win_rate"], 4),
            "oos_calmar": round(oos_metrics["calmar"], 4),
            "oos_trades": int(oos_metrics["trades"]),
        }
        folds.append(fold_result)

        print(f"         IS: Sharpe={is_metrics['sharpe']:.3f} Ret={is_metrics['total_return']*100:.1f}% DD={is_metrics['max_dd']*100:.1f}%")
        print(f"        OOS: Sharpe={oos_metrics['sharpe']:.3f} Ret={oos_metrics['total_return']*100:.1f}% DD={oos_metrics['max_dd']*100:.1f}%")

        start += TEST_DAYS

    # ── Summary Report ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 WALK-FORWARD RESULTS SUMMARY")
    print("=" * 80)

    header = f"{'Fold':>4} | {'IS Sharpe':>10} {'OOS Sharpe':>11} {'Degrad':>7} | {'IS Ret%':>8} {'OOS Ret%':>9} | {'OOS DD%':>8} {'OOS WR%':>8}"
    print(header)
    print("-" * len(header))

    is_sharpes = [f["is_sharpe"] for f in folds]
    oos_sharpes = [f["oos_sharpe"] for f in folds]
    oos_returns = [f["oos_return"] for f in folds]
    oos_dds = [f["oos_max_dd"] for f in folds]

    for f in folds:
        degrad = f["oos_sharpe"] / f["is_sharpe"] if f["is_sharpe"] != 0 else 0
        print(f"  {f['fold']:2d} | {f['is_sharpe']:10.4f} {f['oos_sharpe']:11.4f} {degrad:7.2f} | "
              f"{f['is_return']*100:8.1f} {f['oos_return']*100:9.1f} | "
              f"{f['oos_max_dd']*100:8.1f} {f['oos_win_rate']*100:8.1f}")

    print("-" * len(header))
    avg_is_sharpe = np.mean(is_sharpes)
    avg_oos_sharpe = np.mean(oos_sharpes)
    avg_oos_ret = np.mean(oos_returns)
    avg_oos_dd = np.mean(oos_dds)
    avg_degrad = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0

    print(f" AVG | {avg_is_sharpe:10.4f} {avg_oos_sharpe:11.4f} {avg_degrad:7.2f} | "
          f"{np.mean([f['is_return'] for f in folds])*100:8.1f} {avg_oos_ret*100:9.1f} | "
          f"{avg_oos_dd*100:8.1f} {np.mean([f['oos_win_rate'] for f in folds])*100:8.1f}")

    # Statistical significance
    oos_daily = np.array(all_oos_returns)
    oos_daily = oos_daily[oos_daily != 0] if len(oos_daily) > 0 else oos_daily
    if len(oos_daily) > 1:
        t_stat, p_value = stats.ttest_1samp(oos_daily, 0)
        print(f"\n📈 Statistical Significance:")
        print(f"  OOS daily returns t-test: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"  Significant at 5%: {'YES ✅' if p_value < 0.05 else 'NO ❌'}")
        print(f"  Significant at 1%: {'YES ✅' if p_value < 0.01 else 'NO ❌'}")

    # Best OOS fold
    best_fold = max(folds, key=lambda f: f["oos_sharpe"])
    print(f"\n🏆 Best OOS Fold: {best_fold['fold']} (Sharpe={best_fold['oos_sharpe']:.4f})")
    print(f"   Recommended params:")
    for k, v in best_fold["best_params"].items():
        print(f"     {k:25s} = {v}")

    # Save results
    save_dir = os.path.expanduser("~/Desktop/maestro/data/optimization")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "walkforward_results.json")

    results = {
        "folds": folds,
        "summary": {
            "avg_is_sharpe": round(avg_is_sharpe, 4),
            "avg_oos_sharpe": round(avg_oos_sharpe, 4),
            "avg_oos_return": round(avg_oos_ret, 4),
            "avg_oos_max_dd": round(avg_oos_dd, 4),
            "degradation_ratio": round(avg_degrad, 4),
            "p_value": round(float(p_value), 6) if len(oos_daily) > 1 else None,
            "n_folds": len(folds),
        },
        "recommended_params": best_fold["best_params"],
    }

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {save_path}")
