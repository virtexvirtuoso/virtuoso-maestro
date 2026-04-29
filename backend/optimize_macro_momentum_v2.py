"""
Optuna Optimization for MacroMomentum V2
Maximize Sharpe with penalties for excessive drawdown or too few trades.
"""
import sys, os, warnings, pickle
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

STUDY_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/optimization/macro_momentum_v2_study.pkl"))
STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)

TX_COST = 0.001


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


def evaluate(btc, macro_score, m2_acc, mstr_excess, params):
    """Run strategy and return (sharpe, max_dd, n_trades)."""
    res = generate_signals(btc, macro_score=macro_score, m2_accelerating=m2_acc,
                          mstr_excess=mstr_excess, **params)
    
    daily_ret = btc["close"].pct_change().fillna(0)
    pos = res["position_size"].fillna(0)
    pos_change = pos.diff().fillna(0).abs()
    strat_ret = daily_ret * pos - pos_change * TX_COST
    
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    equity = (1 + strat_ret).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    
    entries = res["entry_type"].str.startswith("entry_").sum()
    
    return sharpe, max_dd, entries


def objective(trial, btc, macro_score, m2_acc, mstr_excess):
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
    
    sharpe, max_dd, n_trades = evaluate(btc, macro_score, m2_acc, mstr_excess, params)
    
    # Penalties
    penalty = 0
    if max_dd < -0.25:
        penalty += (abs(max_dd) - 0.25) * 5
    if n_trades < 15:
        penalty += (15 - n_trades) * 0.1
    
    return sharpe - penalty


def main():
    print("Loading data...")
    btc, macro_score, m2_acc, mstr_excess = load_data()
    print(f"BTC: {len(btc)} days")
    
    study = optuna.create_study(direction="maximize", study_name="macro_momentum_v2")
    
    print("Running 200 trials...")
    study.optimize(
        lambda trial: objective(trial, btc, macro_score, m2_acc, mstr_excess),
        n_trials=200,
        show_progress_bar=True,
    )
    
    # Save study
    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    
    # Best params
    print(f"\n{'='*80}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Best Sharpe (penalized): {study.best_value:.3f}")
    print(f"\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Evaluate best
    sharpe, max_dd, n_trades = evaluate(btc, macro_score, m2_acc, mstr_excess, study.best_params)
    print(f"\nRaw Sharpe: {sharpe:.3f}")
    print(f"Max DD: {max_dd*100:.1f}%")
    print(f"Entries: {n_trades}")
    
    # Top 10 trials
    print(f"\nTop 10 Trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    for i, t in enumerate(trials_sorted[:10]):
        print(f"  #{i+1}: Sharpe={t.value:.3f} | trail={t.params.get('trail_stop_pct', 0):.2f} "
              f"sma={t.params.get('sma_slow', 0)} rsi_e={t.params.get('rsi_entry', 0)}")
    
    print(f"\nStudy saved to {STUDY_PATH}")


if __name__ == "__main__":
    main()
