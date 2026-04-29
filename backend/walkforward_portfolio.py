"""
Walk-Forward Validation for MacroMomentum Portfolio.

2yr train / 6mo test, rolling.
Per fold: optimize portfolio weights + meta params (30 trials).
Per-asset params stay fixed from per-asset optimization.
"""
import sys, os, warnings, json
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_score_builder import compute_macro_score
from strategies.composite.macro_momentum_portfolio import (
    ASSET_CONFIGS, run_portfolio_backtest,
)

RESULTS_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/optimization/walkforward_portfolio_results.json"))
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = 2
TEST_MONTHS = 6


def load_data():
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()
    
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
    asset_data = {}
    for name, ticker in tickers.items():
        df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01", end_date="2026-02-01")
        if len(df) > 100:
            asset_data[name] = df
    
    macro_score = compute_macro_score(fred_loader, start_date="2015-01-01", end_date="2026-02-01")
    m2 = fred_loader.get_series("M2SL", start_date="2015-01-01", end_date="2026-02-01")
    m2_yoy = m2.pct_change(12)
    m2_yoy_ma6 = m2_yoy.rolling(6).mean()
    m2_acc = (m2_yoy > m2_yoy_ma6).resample("D").ffill().fillna(False)
    
    return asset_data, macro_score, m2_acc


def wf_objective(trial, asset_data, macro_score, m2_acc, start, end):
    """Quick portfolio optimization objective for walk-forward."""
    max_lev = trial.suggest_float("max_portfolio_leverage", 0.8, 1.5)
    rebal = trial.suggest_categorical("rebalance_frequency", ["daily", "weekly", "monthly"])
    alloc_mode = trial.suggest_categorical("allocation_mode", ["equal", "risk_parity", "momentum_weighted"])
    corr_filter = trial.suggest_categorical("correlation_filter", [True, False])
    vol_target = trial.suggest_float("vol_target", 0.15, 0.40)
    
    try:
        result = run_portfolio_backtest(
            asset_data, macro_score, m2_acc,
            mode=alloc_mode, max_portfolio_leverage=max_lev,
            vol_target=vol_target, rebalance_freq=rebal,
            correlation_filter=corr_filter,
            start_date=str(start.date()), end_date=str(end.date()),
        )
        
        if "error" in result:
            return -10
        
        sharpe = result["sharpe"]
        max_dd = result["max_dd"]
        penalty = max(0, (abs(max_dd) - 20) * 0.1) if max_dd < -20 else 0
        return sharpe - penalty
    except:
        return -10


def run_oos(asset_data, macro_score, m2_acc, best_params, start, end):
    """Run OOS with optimized params."""
    try:
        result = run_portfolio_backtest(
            asset_data, macro_score, m2_acc,
            mode=best_params.get("allocation_mode", "equal"),
            max_portfolio_leverage=best_params.get("max_portfolio_leverage", 1.0),
            vol_target=best_params.get("vol_target", 0.25),
            rebalance_freq=best_params.get("rebalance_frequency", "monthly"),
            correlation_filter=best_params.get("correlation_filter", False),
            start_date=str(start.date()), end_date=str(end.date()),
        )
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    print("Loading data...")
    asset_data, macro_score, m2_acc = load_data()
    
    # Get common range
    starts = [df.index[0] for df in asset_data.values()]
    ends = [df.index[-1] for df in asset_data.values()]
    data_start = max(starts)
    data_end = min(ends)
    print(f"Common range: {data_start.date()} to {data_end.date()}")
    
    # Generate folds
    folds = []
    train_start = data_start
    while True:
        train_end = train_start + timedelta(days=TRAIN_YEARS * 365)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=TEST_MONTHS * 30)
        
        if test_end > data_end:
            break
        
        folds.append((train_start, train_end, test_start, test_end))
        train_start += timedelta(days=TEST_MONTHS * 30)  # Roll forward by test period
    
    print(f"Generated {len(folds)} walk-forward folds")
    
    results = []
    
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"\nFold {i+1}/{len(folds)}: Train {tr_s.date()}-{tr_e.date()} | Test {te_s.date()}-{te_e.date()}")
        
        # Phase: optimize on training set (30 trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: wf_objective(trial, asset_data, macro_score, m2_acc, tr_s, tr_e),
            n_trials=30,
        )
        
        is_sharpe = study.best_value
        best_params = study.best_params
        
        # Evaluate IS with best params
        is_result = run_oos(asset_data, macro_score, m2_acc, best_params, tr_s, tr_e)
        is_metrics = {
            "sharpe": is_result.get("sharpe", 0),
            "return": is_result.get("total_return", 0),
            "max_dd": is_result.get("max_dd", 0),
        } if "error" not in is_result else {"sharpe": 0, "return": 0, "max_dd": 0}
        
        # Run OOS
        oos_result = run_oos(asset_data, macro_score, m2_acc, best_params, te_s, te_e)
        oos_metrics = {
            "sharpe": oos_result.get("sharpe", 0),
            "return": oos_result.get("total_return", 0),
            "max_dd": oos_result.get("max_dd", 0),
        } if "error" not in oos_result else {"sharpe": 0, "return": 0, "max_dd": 0}
        
        fold_result = {
            "fold": i + 1,
            "train_start": str(tr_s.date()),
            "train_end": str(tr_e.date()),
            "test_start": str(te_s.date()),
            "test_end": str(te_e.date()),
            "best_params": {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                           for k, v in best_params.items()},
            "is_sharpe": round(is_metrics["sharpe"], 3),
            "is_return": round(is_metrics["return"], 2),
            "is_max_dd": round(is_metrics["max_dd"], 2),
            "oos_sharpe": round(oos_metrics["sharpe"], 3),
            "oos_return": round(oos_metrics["return"], 2),
            "oos_max_dd": round(oos_metrics["max_dd"], 2),
        }
        
        results.append(fold_result)
        print(f"  IS: Sharpe={is_metrics['sharpe']:.3f}, Return={is_metrics['return']:.1f}%, DD={is_metrics['max_dd']:.1f}%")
        print(f"  OOS: Sharpe={oos_metrics['sharpe']:.3f}, Return={oos_metrics['return']:.1f}%, DD={oos_metrics['max_dd']:.1f}%")
        print(f"  Best: mode={best_params.get('allocation_mode')}, lev={best_params.get('max_portfolio_leverage', 0):.2f}")
    
    # Summary
    oos_sharpes = [r["oos_sharpe"] for r in results]
    oos_returns = [r["oos_return"] for r in results]
    oos_dds = [r["oos_max_dd"] for r in results]
    is_sharpes = [r["is_sharpe"] for r in results]
    
    active_folds = sum(1 for s in oos_sharpes if s != 0)
    
    # T-test on OOS returns
    from scipy import stats
    if len(oos_returns) > 1 and np.std(oos_returns) > 0:
        t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
    else:
        t_stat, p_value = 0, 1
    
    summary = {
        "n_folds": len(results),
        "active_folds": active_folds,
        "mean_is_sharpe": round(float(np.mean(is_sharpes)), 3),
        "mean_oos_sharpe": round(float(np.mean(oos_sharpes)), 3),
        "median_oos_sharpe": round(float(np.median(oos_sharpes)), 3),
        "mean_oos_return": round(float(np.mean(oos_returns)), 2),
        "mean_oos_max_dd": round(float(np.mean(oos_dds)), 2),
        "worst_oos_dd": round(float(min(oos_dds)), 2),
        "oos_return_tstat": round(float(t_stat), 3),
        "oos_return_pvalue": round(float(p_value), 4),
        "sharpe_decay": round(float(np.mean(is_sharpes) - np.mean(oos_sharpes)), 3),
    }
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD PORTFOLIO RESULTS")
    print("=" * 80)
    print(f"Folds: {len(results)} total, {active_folds} active")
    print(f"Mean IS Sharpe:  {summary['mean_is_sharpe']:.3f}")
    print(f"Mean OOS Sharpe: {summary['mean_oos_sharpe']:.3f}")
    print(f"Sharpe Decay:    {summary['sharpe_decay']:.3f}")
    print(f"Mean OOS Return: {summary['mean_oos_return']:.2f}%")
    print(f"Mean OOS MaxDD:  {summary['mean_oos_max_dd']:.2f}%")
    print(f"Worst OOS DD:    {summary['worst_oos_dd']:.2f}%")
    print(f"OOS t-stat:      {summary['oos_return_tstat']:.3f} (p={summary['oos_return_pvalue']:.4f})")
    
    # Per-fold table
    print(f"\n{'Fold':<6} {'Train Period':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Ret%':>9} {'OOS DD%':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['fold']:<6} {r['train_start']} - {r['train_end']:<5} "
              f"{r['is_sharpe']:>10.3f} {r['oos_sharpe']:>11.3f} "
              f"{r['oos_return']:>9.1f} {r['oos_max_dd']:>8.1f}")
    
    # Save
    output = {"summary": summary, "folds": results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
