"""
Walk-Forward Validation for MacroMomentum V2
Training: 2 years, Test: 6 months, rolling.
Compare to V1 walk-forward (golden cross was too restrictive).
"""
import sys, os, json, warnings, pickle
import pandas as pd
import numpy as np
import optuna
from pathlib import Path

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_score_builder import compute_macro_score
from strategies.composite.macro_momentum_v2 import generate_signals
from strategies.composite.macro_momentum import generate_signals as v1_signals

RESULTS_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/optimization/walkforward_v2_results.json"))
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

TX_COST = 0.001
TRAIN_DAYS = 730  # 2 years
TEST_DAYS = 182   # 6 months
N_TRIALS = 50


def load_data():
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()
    btc = stock_loader.get_ohlcv("BTC-USD", "1d", start_date="2017-01-01", end_date="2026-02-01")
    macro_score = compute_macro_score(fred_loader, start_date="2015-01-01", end_date="2026-02-01")
    
    m2 = fred_loader.get_series("M2SL", start_date="2015-01-01", end_date="2026-02-01")
    m2_yoy = m2.pct_change(12)
    m2_yoy_ma6 = m2_yoy.rolling(6).mean()
    m2_acc = (m2_yoy > m2_yoy_ma6).resample("D").ffill().fillna(False)
    
    try:
        mstr = stock_loader.get_ohlcv("MSTR", "1d", start_date="2017-01-01", end_date="2026-02-01")
        btc_ret5 = btc["close"].pct_change(5)
        mstr_ret5 = mstr["close"].pct_change(5)
        mstr_excess = (mstr_ret5 - btc_ret5.reindex(mstr_ret5.index, method="ffill")).reindex(btc.index, method="ffill")
    except Exception:
        mstr_excess = None
    
    return btc, macro_score, m2_acc, mstr_excess


def run_strategy(btc_slice, macro_score, m2_acc, mstr_excess, params):
    """Run V2 and return (sharpe, returns, position_sizes)."""
    res = generate_signals(btc_slice, macro_score=macro_score, m2_accelerating=m2_acc,
                          mstr_excess=mstr_excess, **params)
    daily_ret = btc_slice["close"].pct_change().fillna(0)
    pos = res["position_size"].fillna(0)
    pos_change = pos.diff().fillna(0).abs()
    strat_ret = daily_ret * pos - pos_change * TX_COST
    
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = ((1 + strat_ret).cumprod() / (1 + strat_ret).cumprod().cummax() - 1).min()
    entries = res["entry_type"].str.startswith("entry_").sum()
    time_in = (pos > 0).mean()
    
    return sharpe, max_dd, entries, time_in, strat_ret


def run_v1(btc_slice, macro_score):
    """Run V1 for comparison."""
    sig = v1_signals(btc_slice)
    from strategies.composite.macro_momentum import position_size as v1_pos
    pos = v1_pos(sig, macro_score)
    daily_ret = btc_slice["close"].pct_change().fillna(0)
    strat_ret = daily_ret * pos - pos.diff().fillna(0).abs() * TX_COST
    
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    time_in = (pos > 0).mean()
    entries = ((sig == 1) & (sig.shift(1) != 1)).sum()
    
    return sharpe, time_in, entries


def optimize_fold(btc_train, macro_score, m2_acc, mstr_excess):
    """Optimize on training data, return best params."""
    def obj(trial):
        params = {
            "sma_slow": trial.suggest_int("sma_slow", 80, 250, step=10),
            "momentum_period": trial.suggest_int("momentum_period", 10, 50, step=5),
            "roc_threshold": trial.suggest_float("roc_threshold", 0.0, 0.05),
            "rsi_entry": trial.suggest_int("rsi_entry", 25, 45),
            "rsi_exit": trial.suggest_int("rsi_exit", 65, 80),
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_std": trial.suggest_float("bb_std", 1.5, 2.5),
            "trail_stop_pct": trial.suggest_float("trail_stop_pct", 0.06, 0.20),
            "initial_size": trial.suggest_float("initial_size", 0.2, 0.5),
            "pyramid_size": trial.suggest_float("pyramid_size", 0.1, 0.35),
            "max_position": trial.suggest_float("max_position", 0.8, 1.5),
            "trim_pct": trial.suggest_float("trim_pct", 0.15, 0.40),
            "atr_exit_mult": trial.suggest_float("atr_exit_mult", 1.5, 3.0),
            "macro_weight": trial.suggest_float("macro_weight", 0.5, 1.0),
        }
        sharpe, max_dd, entries, _, _ = run_strategy(btc_train, macro_score, m2_acc, mstr_excess, params)
        penalty = 0
        if max_dd < -0.25:
            penalty += (abs(max_dd) - 0.25) * 5
        if entries < 5:
            penalty += (5 - entries) * 0.2
        return sharpe - penalty
    
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=N_TRIALS)
    return study.best_params, study.best_value


def main():
    print("Loading data...")
    btc, macro_score, m2_acc, mstr_excess = load_data()
    
    dates = btc.index
    total_days = len(dates)
    
    folds = []
    fold_num = 0
    start_idx = 0
    
    while start_idx + TRAIN_DAYS + TEST_DAYS <= total_days:
        train_end = start_idx + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, total_days)
        
        train_start_date = dates[start_idx]
        train_end_date = dates[train_end - 1]
        test_start_date = dates[train_end]
        test_end_date = dates[test_end - 1]
        
        folds.append((start_idx, train_end, test_end, 
                      train_start_date, train_end_date, test_start_date, test_end_date))
        start_idx += TEST_DAYS
        fold_num += 1
    
    print(f"Total folds: {len(folds)}")
    
    results = []
    v1_results = []
    
    for i, (si, te, tse, ts, ted, tss, tsed) in enumerate(folds):
        print(f"\nFold {i+1}/{len(folds)}: Train {ts.date()}→{ted.date()} | Test {tss.date()}→{tsed.date()}")
        
        btc_train = btc.iloc[si:te]
        btc_test = btc.iloc[te:tse]
        
        # Optimize V2 on train
        best_params, is_sharpe = optimize_fold(btc_train, macro_score, m2_acc, mstr_excess)
        
        # Evaluate V2 on test
        oos_sharpe, oos_dd, oos_entries, oos_time_in, oos_ret = run_strategy(
            btc_test, macro_score, m2_acc, mstr_excess, best_params)
        
        # V1 comparison on test
        v1_sharpe, v1_time_in, v1_entries = run_v1(btc_test, macro_score)
        
        print(f"  V2 IS Sharpe: {is_sharpe:.2f} → OOS Sharpe: {oos_sharpe:.2f} | "
              f"OOS entries: {oos_entries}, time_in: {oos_time_in:.1%}")
        print(f"  V1 OOS Sharpe: {v1_sharpe:.2f} | entries: {v1_entries}, time_in: {v1_time_in:.1%}")
        
        results.append({
            "fold": i + 1,
            "train_start": str(ts.date()),
            "train_end": str(ted.date()),
            "test_start": str(tss.date()),
            "test_end": str(tsed.date()),
            "is_sharpe": round(is_sharpe, 3),
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_max_dd": round(oos_dd * 100, 1) if oos_dd else 0,
            "oos_entries": int(oos_entries),
            "oos_time_in_market": round(oos_time_in * 100, 1),
            "best_params": {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()},
        })
        
        v1_results.append({
            "fold": i + 1,
            "v1_oos_sharpe": round(v1_sharpe, 3),
            "v1_oos_entries": int(v1_entries),
            "v1_oos_time_in": round(v1_time_in * 100, 1),
        })
    
    # Summary
    print(f"\n{'='*100}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*100}")
    
    is_sharpes = [r["is_sharpe"] for r in results]
    oos_sharpes = [r["oos_sharpe"] for r in results]
    oos_entries = [r["oos_entries"] for r in results]
    oos_time = [r["oos_time_in_market"] for r in results]
    
    v1_oos_sharpes = [r["v1_oos_sharpe"] for r in v1_results]
    v1_oos_entries_list = [r["v1_oos_entries"] for r in v1_results]
    v1_oos_time_list = [r["v1_oos_time_in"] for r in v1_results]
    
    print(f"\nV2 Walk-Forward:")
    print(f"  Mean IS Sharpe:  {np.mean(is_sharpes):.3f} ± {np.std(is_sharpes):.3f}")
    print(f"  Mean OOS Sharpe: {np.mean(oos_sharpes):.3f} ± {np.std(oos_sharpes):.3f}")
    print(f"  IS→OOS Degradation: {(1 - np.mean(oos_sharpes)/np.mean(is_sharpes))*100:.1f}%")
    print(f"  Mean OOS Entries: {np.mean(oos_entries):.1f}")
    print(f"  Mean OOS Time in Market: {np.mean(oos_time):.1f}%")
    print(f"  Positive OOS Sharpe: {sum(1 for s in oos_sharpes if s > 0)}/{len(oos_sharpes)}")
    
    print(f"\nV1 Walk-Forward (comparison):")
    print(f"  Mean OOS Sharpe: {np.mean(v1_oos_sharpes):.3f} ± {np.std(v1_oos_sharpes):.3f}")
    print(f"  Mean OOS Entries: {np.mean(v1_oos_entries_list):.1f}")
    print(f"  Mean OOS Time in Market: {np.mean(v1_oos_time_list):.1f}%")
    
    print(f"\nV2 vs V1 Improvement:")
    print(f"  OOS Sharpe: {np.mean(oos_sharpes):.3f} vs {np.mean(v1_oos_sharpes):.3f}")
    print(f"  OOS Activity: {np.mean(oos_time):.1f}% vs {np.mean(v1_oos_time_list):.1f}%")
    print(f"  OOS Entries: {np.mean(oos_entries):.1f} vs {np.mean(v1_oos_entries_list):.1f}")
    
    # Statistical significance (t-test on OOS Sharpes)
    from scipy import stats
    if len(oos_sharpes) > 2:
        t_stat, p_val = stats.ttest_rel(oos_sharpes, v1_oos_sharpes)
        print(f"\n  Paired t-test V2 vs V1: t={t_stat:.2f}, p={p_val:.4f}")
        print(f"  Significant at 5%: {'YES' if p_val < 0.05 else 'NO'}")
    
    # Per-fold table
    print(f"\n{'Fold':<6} {'Train Period':<28} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Ent':>8} {'OOS Time%':>10} {'V1 OOS':>8}")
    print("-" * 90)
    for r, v1r in zip(results, v1_results):
        print(f"{r['fold']:<6} {r['train_start']}→{r['test_end']:<12} "
              f"{r['is_sharpe']:>10.3f} {r['oos_sharpe']:>11.3f} {r['oos_entries']:>8} "
              f"{r['oos_time_in_market']:>10.1f} {v1r['v1_oos_sharpe']:>8.3f}")
    
    # Save
    save_data = {
        "v2_folds": results,
        "v1_comparison": v1_results,
        "summary": {
            "v2_mean_is_sharpe": round(np.mean(is_sharpes), 3),
            "v2_mean_oos_sharpe": round(np.mean(oos_sharpes), 3),
            "v2_degradation_pct": round((1 - np.mean(oos_sharpes)/max(np.mean(is_sharpes), 0.001))*100, 1),
            "v1_mean_oos_sharpe": round(np.mean(v1_oos_sharpes), 3),
            "v2_mean_oos_entries": round(np.mean(oos_entries), 1),
            "v1_mean_oos_entries": round(np.mean(v1_oos_entries_list), 1),
            "v2_mean_oos_time_in": round(np.mean(oos_time), 1),
            "v1_mean_oos_time_in": round(np.mean(v1_oos_time_list), 1),
        }
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
