#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for MacroMomentum Strategy
Maximizes Sharpe ratio on BTC-USD daily data with macro scoring.
"""
import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import numpy as np
import pandas as pd
import optuna
from optuna.importance import get_param_importances

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_momentum import generate_signals, position_size
from strategies.composite.macro_score_builder import compute_macro_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Data Loading ──────────────────────────────────────────────
print("Loading data...")
stock_loader = StockDataLoader()
macro_loader = MacroDataLoader()

btc = stock_loader.get_ohlcv("BTC-USD", "1d", "2017-01-01", "2026-02-12")
macro_score = compute_macro_score(macro_loader, start_date="2015-01-01", end_date="2026-02-12")
macro_score = macro_score.reindex(btc.index, method="ffill").fillna(0)

print(f"BTC data: {btc.index[0].date()} to {btc.index[-1].date()} ({len(btc)} bars)")
print(f"Macro score range: {macro_score.min()}-{macro_score.max()}, mean={macro_score.mean():.2f}")


# ── Backtest Engine ───────────────────────────────────────────
def backtest(df, signals, sizes=None):
    """Simple vectorized backtest. Returns dict of metrics."""
    close = df["close"].values
    sig = signals.values.astype(float)

    if sizes is not None:
        sig = sig * sizes.values

    # Daily returns
    ret = np.diff(close) / close[:-1]
    # Position returns (signal shifted already in generate_signals)
    pos_ret = sig[:-1] * ret

    if len(pos_ret) == 0 or np.all(pos_ret == 0):
        return {"sharpe": -10, "total_return": 0, "max_dd": 0, "calmar": -10, "trades": 0, "win_rate": 0}

    cum = (1 + pos_ret).cumprod()
    total_ret = cum[-1] - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    # Count trades (signal changes)
    sig_diff = np.diff(signals.values)
    trades = np.count_nonzero(sig_diff)

    # Sharpe (annualized)
    daily_mean = np.mean(pos_ret[pos_ret != 0]) if np.any(pos_ret != 0) else 0
    daily_std = np.std(pos_ret[pos_ret != 0]) if np.any(pos_ret != 0) else 1
    sharpe = (daily_mean / daily_std) * np.sqrt(365) if daily_std > 0 else 0

    # Win rate
    winning = np.sum(pos_ret > 0)
    total_active = np.sum(pos_ret != 0)
    win_rate = winning / total_active if total_active > 0 else 0

    # Calmar
    annual_ret = total_ret / (len(pos_ret) / 365)
    calmar = annual_ret / max_dd if max_dd > 0 else 0

    return {
        "sharpe": sharpe, "total_return": total_ret, "max_dd": max_dd,
        "calmar": calmar, "trades": trades, "win_rate": win_rate,
    }


# ── Optuna Objective ──────────────────────────────────────────
def objective(trial):
    params = {
        "sma_slow": trial.suggest_int("sma_slow", 100, 300, step=10),
        "sma_fast": trial.suggest_int("sma_fast", 20, 100, step=5),
        "momentum_period": trial.suggest_int("momentum_period", 10, 60, step=5),
        "trailing_stop_pct": trial.suggest_float("trailing_stop_pct", 0.08, 0.25),
        "roc_threshold": trial.suggest_float("roc_threshold", 0.0, 0.05),
    }
    # Position sizing params
    threshold_high = trial.suggest_int("macro_threshold_high", 4, 6)
    threshold_mid = trial.suggest_int("macro_threshold_mid", 2, 4)
    threshold_low = trial.suggest_int("macro_threshold_low", 1, 2)
    size_high = trial.suggest_float("position_size_high", 0.8, 1.0)
    size_mid = trial.suggest_float("position_size_mid", 0.4, 0.7)
    size_low = trial.suggest_float("position_size_low", 0.1, 0.4)

    if params["sma_fast"] >= params["sma_slow"]:
        return -10.0

    signals = generate_signals(btc, **params)
    sizes = position_size(
        signals, macro_score,
        threshold_high=threshold_high, threshold_mid=threshold_mid,
        threshold_low=threshold_low, size_high=size_high,
        size_mid=size_mid, size_low=size_low,
    )

    metrics = backtest(btc, signals, sizes)

    # Penalties
    score = metrics["sharpe"]
    if metrics["max_dd"] > 0.30:
        score -= 2.0
    if metrics["trades"] < 10:
        score -= 3.0

    return score


# ── Run Optimization ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Running Optuna optimization (200 trials)...")
    study = optuna.create_study(direction="maximize", study_name="macro_momentum")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    # Save study
    save_dir = os.path.expanduser("~/Desktop/maestro/data/optimization")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "macro_momentum_study.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\n💾 Study saved to {save_path}")

    # Best params
    print("\n" + "=" * 70)
    print("🏆 BEST PARAMETERS")
    print("=" * 70)
    for k, v in study.best_params.items():
        print(f"  {k:25s} = {v}")
    print(f"\n  Best Sharpe (penalized): {study.best_value:.4f}")

    # Run backtest with best params to get full metrics
    bp = study.best_params
    sig = generate_signals(btc, sma_slow=bp["sma_slow"], sma_fast=bp["sma_fast"],
                           momentum_period=bp["momentum_period"],
                           trailing_stop_pct=bp["trailing_stop_pct"],
                           roc_threshold=bp["roc_threshold"])
    sz = position_size(sig, macro_score,
                       threshold_high=bp["macro_threshold_high"],
                       threshold_mid=bp["macro_threshold_mid"],
                       threshold_low=bp["macro_threshold_low"],
                       size_high=bp["position_size_high"],
                       size_mid=bp["position_size_mid"],
                       size_low=bp["position_size_low"])
    m = backtest(btc, sig, sz)
    print(f"\n📊 Full Backtest Metrics:")
    print(f"  Sharpe:       {m['sharpe']:.4f}")
    print(f"  Total Return: {m['total_return']*100:.2f}%")
    print(f"  Max Drawdown: {m['max_dd']*100:.2f}%")
    print(f"  Calmar:       {m['calmar']:.4f}")
    print(f"  Trades:       {m['trades']}")
    print(f"  Win Rate:     {m['win_rate']*100:.1f}%")

    # Top 10 trials
    print("\n" + "=" * 70)
    print("📋 TOP 10 TRIALS")
    print("=" * 70)
    trials_df = study.trials_dataframe().sort_values("value", ascending=False).head(10)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number']):3d} | Sharpe={row['value']:.4f} | "
              f"sma_slow={int(row.get('params_sma_slow', 0))} "
              f"sma_fast={int(row.get('params_sma_fast', 0))} "
              f"mom={int(row.get('params_momentum_period', 0))} "
              f"stop={row.get('params_trailing_stop_pct', 0):.3f}")

    # Parameter importance
    print("\n" + "=" * 70)
    print("🎯 PARAMETER IMPORTANCE")
    print("=" * 70)
    try:
        importances = get_param_importances(study)
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"  {param:25s} {imp:.4f} {bar}")
    except Exception as e:
        print(f"  Could not compute importance: {e}")
