"""
Walk-Forward Validation — Multi-Asset with loosened filters.
2yr train / 6mo test windows. Compare activity to V2's 4/14 active folds.
"""
import sys, os, json, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from datasource.yfinance_loader import StockDataLoader

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2017-01-01"
END = "2026-02-01"
TX_COST = 0.001
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
TRAIN_DAYS = 730  # 2 years
TEST_DAYS = 182   # 6 months
OPTUNA_TRIALS = 50  # per fold


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_bb(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid - std * s, mid, mid + std * s

def compute_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_backtest(df, params):
    """Run strategy, return (sharpe, total_return, max_dd, n_entries, time_in_mkt)."""
    close = df["close"]
    n = len(close)
    sma_slow = int(params["sma_slow"])
    
    if n < sma_slow + 50:
        return 0, 0, 0, 0, 0
    
    sma = close.rolling(sma_slow).mean()
    roc = close.pct_change(int(params["momentum_period"]))
    rsi = compute_rsi(close, 14)
    ema = close.ewm(span=int(params["ema_period"]), adjust=False).mean()
    bb_lower, _, bb_upper = compute_bb(close, 20, params["bb_std"])
    atr = compute_atr(df)
    
    regime = (close > sma) & (roc > 0)  # SMA + Momentum only
    rsi_dip = rsi < params["rsi_entry"]
    ema_dip = close < ema
    bb_dip = close <= bb_lower
    any_dip = rsi_dip | ema_dip | bb_dip
    
    regime = regime.shift(1).fillna(False)
    any_dip = any_dip.shift(1).fillna(False)
    rsi_hot = (rsi > params["rsi_exit"]).shift(1).fillna(False)
    bb_hot = (close > bb_upper).shift(1).fillna(False)
    atr_s = atr.shift(1).fillna(0)
    
    pos = 0.0; avg_entry = 0.0; equity = 1.0; peak = 1.0; total_cost = 0.0
    equities = np.ones(n); positions = np.zeros(n); n_entries = 0
    warmup = sma_slow + 5
    
    for i in range(warmup, n):
        price = close.iloc[i]; prev = close.iloc[i-1]
        if pos > 0 and prev > 0:
            equity *= (1 + (price - prev)/prev * pos)
        peak = max(peak, equity); equities[i] = equity
        
        if pos > 0 and (1 - equity/peak) >= params["trail_stop"]:
            equity *= (1 - pos * TX_COST); pos = 0; avg_entry = 0; total_cost = 0; continue
        
        if pos > 0:
            trim = 0
            if rsi_hot.iloc[i]: trim = pos * 0.25
            elif bb_hot.iloc[i]: trim = pos * 0.25
            elif avg_entry > 0 and atr_s.iloc[i] > 0 and (price-avg_entry)/atr_s.iloc[i] > params["atr_exit_mult"]:
                trim = pos * 0.25
            if trim > 0:
                equity *= (1 - trim * TX_COST); pos -= trim
                if pos < 0.01: pos = 0; avg_entry = 0; total_cost = 0
        
        if regime.iloc[i] and any_dip.iloc[i]:
            if pos == 0:
                add = min(params["initial_size"], params["max_position"])
                equity *= (1 - add * TX_COST); pos = add; avg_entry = price; total_cost = price*add; n_entries += 1
            elif pos < params["max_position"]:
                add = min(params["pyramid_size"], params["max_position"] - pos)
                if add > 0.01:
                    equity *= (1 - add * TX_COST); total_cost += price*add; pos += add; avg_entry = total_cost/pos
        positions[i] = pos
    
    eq_s = pd.Series(equities, index=df.index)
    dr = eq_s.pct_change().fillna(0)
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = (dr.mean()*252) / ann_vol if ann_vol > 1e-8 else 0
    total_ret = equity - 1
    max_dd = (eq_s / eq_s.cummax() - 1).min()
    time_in = (pd.Series(positions) > 0).mean() * 100
    
    return sharpe, total_ret * 100, max_dd * 100, n_entries, time_in


def optimize_on_train(df_train):
    """Optimize params on training data."""
    def objective(trial):
        params = {
            "sma_slow": trial.suggest_int("sma_slow", 50, 150, step=10),
            "momentum_period": trial.suggest_int("momentum_period", 15, 40, step=5),
            "rsi_entry": trial.suggest_int("rsi_entry", 30, 55),
            "rsi_exit": trial.suggest_int("rsi_exit", 65, 85),
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "bb_std": trial.suggest_float("bb_std", 1.5, 2.5, step=0.25),
            "trail_stop": trial.suggest_float("trail_stop", 0.10, 0.30, step=0.02),
            "atr_exit_mult": trial.suggest_float("atr_exit_mult", 1.5, 3.0, step=0.25),
            "max_position": trial.suggest_float("max_position", 0.8, 2.0, step=0.1),
            "initial_size": trial.suggest_float("initial_size", 0.3, 0.7, step=0.1),
            "pyramid_size": trial.suggest_float("pyramid_size", 0.1, 0.5, step=0.1),
        }
        sharpe, _, _, _, _ = run_backtest(df_train, params)
        return sharpe
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    print("=" * 100)
    print("WALK-FORWARD VALIDATION — Multi-Asset (SMA + Momentum filter)")
    print(f"Train: {TRAIN_DAYS}d | Test: {TEST_DAYS}d | Optuna: {OPTUNA_TRIALS} trials/fold")
    print("=" * 100)
    
    loader = StockDataLoader()
    all_wf_results = {}
    
    for asset in ASSETS:
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD: {asset}")
        print(f"{'='*80}")
        
        df = loader.get_ohlcv(asset, "1d", start_date=START, end_date=END)
        print(f"Data: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        
        n = len(df)
        fold = 0
        folds = []
        
        start_idx = 0
        while start_idx + TRAIN_DAYS + TEST_DAYS <= n:
            train_end = start_idx + TRAIN_DAYS
            test_end = min(train_end + TEST_DAYS, n)
            
            df_train = df.iloc[start_idx:train_end]
            df_test = df.iloc[train_end:test_end]
            
            fold += 1
            train_period = f"{df_train.index[0].strftime('%Y-%m')}->{df_train.index[-1].strftime('%Y-%m')}"
            test_period = f"{df_test.index[0].strftime('%Y-%m')}->{df_test.index[-1].strftime('%Y-%m')}"
            
            print(f"\n  Fold {fold}: Train {train_period} | Test {test_period}")
            
            # Optimize on train
            best_params, is_sharpe = optimize_on_train(df_train)
            
            # Test OOS
            oos_sharpe, oos_ret, oos_dd, oos_entries, oos_time = run_backtest(df_test, best_params)
            
            active = oos_entries > 0
            
            print(f"    IS Sharpe: {is_sharpe:.2f} | OOS Sharpe: {oos_sharpe:.2f} | "
                  f"OOS Ret: {oos_ret:.1f}% | OOS DD: {oos_dd:.1f}% | "
                  f"Entries: {oos_entries} | Active: {'YES' if active else 'NO'}")
            
            folds.append({
                "fold": fold,
                "train": train_period,
                "test": test_period,
                "is_sharpe": round(is_sharpe, 2),
                "oos_sharpe": round(oos_sharpe, 2),
                "oos_return": round(oos_ret, 1),
                "oos_max_dd": round(oos_dd, 1),
                "oos_entries": oos_entries,
                "oos_time_in_mkt": round(oos_time, 1),
                "active": active,
                "params": best_params,
            })
            
            start_idx += TEST_DAYS  # rolling forward by test window
        
        # Summary
        active_folds = sum(1 for f in folds if f["active"])
        total_folds = len(folds)
        oos_sharpes = [f["oos_sharpe"] for f in folds if f["active"]]
        oos_rets = [f["oos_return"] for f in folds if f["active"]]
        
        mean_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
        median_oos_sharpe = np.median(oos_sharpes) if oos_sharpes else 0
        std_oos_sharpe = np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0
        mean_oos_ret = np.mean(oos_rets) if oos_rets else 0
        
        # Statistical significance: t-test of OOS Sharpe > 0
        from scipy import stats
        if len(oos_sharpes) > 2:
            t_stat, p_value = stats.ttest_1samp(oos_sharpes, 0)
        else:
            t_stat, p_value = 0, 1.0
        
        print(f"\n  {'='*60}")
        print(f"  SUMMARY: {asset}")
        print(f"  {'='*60}")
        print(f"  Active folds: {active_folds}/{total_folds} ({active_folds/total_folds*100:.0f}%)")
        print(f"  Mean OOS Sharpe: {mean_oos_sharpe:.2f} (std={std_oos_sharpe:.2f})")
        print(f"  Median OOS Sharpe: {median_oos_sharpe:.2f}")
        print(f"  Mean OOS Return: {mean_oos_ret:.1f}%")
        print(f"  t-stat: {t_stat:.2f}, p-value: {p_value:.4f}")
        print(f"  Significant at 5%: {'YES ✓' if p_value < 0.05 else 'NO ✗'}")
        
        all_wf_results[asset] = {
            "folds": folds,
            "active_folds": active_folds,
            "total_folds": total_folds,
            "pct_active": round(active_folds/total_folds*100, 0),
            "mean_oos_sharpe": round(mean_oos_sharpe, 2),
            "median_oos_sharpe": round(median_oos_sharpe, 2),
            "std_oos_sharpe": round(std_oos_sharpe, 2),
            "mean_oos_return": round(mean_oos_ret, 1),
            "t_stat": round(t_stat, 2),
            "p_value": round(p_value, 4),
        }
    
    # Final comparison
    print("\n" + "=" * 100)
    print("WALK-FORWARD COMPARISON — All Assets")
    print("=" * 100)
    
    header = f"{'Asset':<12} {'Active':>10} {'MeanShp':>10} {'MedShp':>10} {'StdShp':>10} {'MeanRet%':>10} {'p-val':>10} {'Sig?':>8}"
    print(header)
    print("-" * 100)
    
    for asset in ASSETS:
        r = all_wf_results[asset]
        sig = "YES ✓" if r["p_value"] < 0.05 else "NO ✗"
        print(f"{asset.replace('-USD',''):<12} {r['active_folds']}/{r['total_folds']:>3} ({r['pct_active']:.0f}%) "
              f"{r['mean_oos_sharpe']:>10.2f} {r['median_oos_sharpe']:>10.2f} {r['std_oos_sharpe']:>10.2f} "
              f"{r['mean_oos_return']:>10.1f} {r['p_value']:>10.4f} {sig:>8}")
    
    print(f"\n  V2 (with M2 filter) baseline: 4/14 active folds (29%)")
    print(f"  Loosened filter improvement:")
    for asset in ASSETS:
        r = all_wf_results[asset]
        print(f"    {asset.replace('-USD','')}: {r['active_folds']}/{r['total_folds']} active ({r['pct_active']:.0f}%) — "
              f"{'BETTER' if r['pct_active'] > 29 else 'SAME/WORSE'} than V2")
    
    # Save
    save_data = {}
    for asset in all_wf_results:
        r = all_wf_results[asset].copy()
        for fold in r["folds"]:
            fold["params"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                             for k, v in fold["params"].items()}
        save_data[asset] = r
    
    with open(RESULTS_DIR / "walkforward_multi_asset_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {RESULTS_DIR / 'walkforward_multi_asset_results.json'}")


if __name__ == "__main__":
    main()
