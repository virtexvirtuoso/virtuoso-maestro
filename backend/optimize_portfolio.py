"""
Optuna Optimization for MacroMomentum Portfolio System.

Phase 1: Per-asset param optimization for LINK (BTC/ETH/SOL already done)
Phase 2: Portfolio-level optimization (weights, allocation mode, meta-params)
"""
import sys, os, warnings, pickle, json
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
from strategies.composite.macro_momentum_portfolio import (
    ASSET_CONFIGS, run_portfolio_backtest, TX_COST,
)

STUDY_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/optimization/portfolio_study.pkl"))
RESULTS_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/optimization/optimize_multi_asset_results.json"))
STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_all_data():
    """Load all asset + macro data."""
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()
    
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
    asset_data = {}
    for name, ticker in tickers.items():
        df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01", end_date="2026-02-01")
        if len(df) > 100:
            asset_data[name] = df
            print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
    
    macro_score = compute_macro_score(fred_loader, start_date="2015-01-01", end_date="2026-02-01")
    
    m2 = fred_loader.get_series("M2SL", start_date="2015-01-01", end_date="2026-02-01")
    m2_yoy = m2.pct_change(12)
    m2_yoy_ma6 = m2_yoy.rolling(6).mean()
    m2_acc = (m2_yoy > m2_yoy_ma6).resample("D").ffill().fillna(False)
    
    return asset_data, macro_score, m2_acc


def evaluate_single_asset(df, macro_score, m2_acc, params):
    """Evaluate single asset strategy, return (sharpe, max_dd, n_trades)."""
    res = generate_signals(df, macro_score=macro_score, m2_accelerating=m2_acc, **params)
    daily_ret = df["close"].pct_change().fillna(0)
    pos = res["position_size"].fillna(0)
    pos_change = pos.diff().fillna(0).abs()
    strat_ret = daily_ret * pos - pos_change * TX_COST
    
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    equity = (1 + strat_ret).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    entries = res["entry_type"].str.startswith("entry_").sum()
    
    return float(sharpe), float(max_dd), int(entries)


# ─── Phase 1: LINK optimization ─────────────────────────────────────────────

def link_objective(trial, df, macro_score, m2_acc):
    params = {
        "sma_slow": trial.suggest_int("sma_slow", 60, 200, step=10),
        "momentum_period": trial.suggest_int("momentum_period", 10, 50, step=5),
        "roc_threshold": trial.suggest_float("roc_threshold", 0.0, 0.05),
        "rsi_entry": trial.suggest_int("rsi_entry", 25, 55),
        "rsi_exit": trial.suggest_int("rsi_exit", 65, 85),
        "ema_period": trial.suggest_int("ema_period", 15, 30),
        "bb_period": trial.suggest_int("bb_period", 15, 25),
        "bb_std": trial.suggest_float("bb_std", 1.5, 2.5),
        "trail_stop_pct": trial.suggest_float("trail_stop_pct", 0.06, 0.25),
        "initial_size": trial.suggest_float("initial_size", 0.3, 0.8),
        "pyramid_size": trial.suggest_float("pyramid_size", 0.1, 0.35),
        "max_position": trial.suggest_float("max_position", 0.8, 1.8),
        "trim_pct": trial.suggest_float("trim_pct", 0.15, 0.40),
        "atr_exit_mult": trial.suggest_float("atr_exit_mult", 1.5, 3.0),
        "macro_weight": trial.suggest_float("macro_weight", 0.5, 1.0),
    }
    
    sharpe, max_dd, n_trades = evaluate_single_asset(df, macro_score, m2_acc, params)
    
    penalty = 0
    if max_dd < -0.25:
        penalty += (abs(max_dd) - 0.25) * 5
    if n_trades < 10:
        penalty += (10 - n_trades) * 0.1
    
    return sharpe - penalty


def run_phase1(asset_data, macro_score, m2_acc):
    """Optimize LINK params."""
    print("\n" + "=" * 80)
    print("PHASE 1: LINK Per-Asset Optimization (100 trials)")
    print("=" * 80)
    
    if "LINK" not in asset_data:
        print("LINK data not available, skipping Phase 1")
        return ASSET_CONFIGS["LINK"]
    
    df = asset_data["LINK"]
    study = optuna.create_study(direction="maximize", study_name="link_optimization")
    study.optimize(
        lambda trial: link_objective(trial, df, macro_score, m2_acc),
        n_trials=100,
        show_progress_bar=True,
    )
    
    best = study.best_params
    sharpe, max_dd, n_trades = evaluate_single_asset(df, macro_score, m2_acc, best)
    
    print(f"\nLINK Best Sharpe (penalized): {study.best_value:.3f}")
    print(f"  Raw Sharpe: {sharpe:.3f}, Max DD: {max_dd*100:.1f}%, Trades: {n_trades}")
    print(f"  Params: sma_slow={best['sma_slow']}, momentum={best['momentum_period']}, "
          f"rsi_entry={best['rsi_entry']}, trail={best['trail_stop_pct']:.2f}")
    
    return best


# ─── Phase 2: Portfolio optimization ─────────────────────────────────────────

def portfolio_objective(trial, asset_data, macro_score, m2_acc, asset_configs):
    # Portfolio weights
    w_btc = trial.suggest_float("weight_btc", 0.15, 0.40)
    w_eth = trial.suggest_float("weight_eth", 0.15, 0.35)
    w_sol = trial.suggest_float("weight_sol", 0.10, 0.30)
    w_link = trial.suggest_float("weight_link", 0.05, 0.25)
    total = w_btc + w_eth + w_sol + w_link
    weights = {"BTC": w_btc/total, "ETH": w_eth/total, "SOL": w_sol/total, "LINK": w_link/total}
    
    # Meta params
    max_lev = trial.suggest_float("max_portfolio_leverage", 0.8, 1.5)
    rebal = trial.suggest_categorical("rebalance_frequency", ["daily", "weekly", "monthly"])
    alloc_mode = trial.suggest_categorical("allocation_mode", ["equal", "risk_parity", "momentum_weighted"])
    corr_filter = trial.suggest_categorical("correlation_filter", [True, False])
    vol_target = trial.suggest_float("vol_target", 0.15, 0.40)
    
    try:
        result = run_portfolio_backtest(
            asset_data, macro_score, m2_acc,
            weights=weights if alloc_mode == "custom" else None,
            mode=alloc_mode if alloc_mode != "custom" else "equal",
            max_portfolio_leverage=max_lev,
            vol_target=vol_target,
            rebalance_freq=rebal,
            correlation_filter=corr_filter,
            asset_configs=asset_configs,
        )
        
        if "error" in result:
            return -10
        
        sharpe = result["sharpe"]
        max_dd = result["max_dd"]
        
        # Penalty for drawdown > 20%
        penalty = 0
        if max_dd < -20:
            penalty += (abs(max_dd) - 20) * 0.1
        
        return sharpe - penalty
        
    except Exception as e:
        print(f"  Trial failed: {e}")
        return -10


def run_phase2(asset_data, macro_score, m2_acc, asset_configs):
    """Portfolio-level optimization."""
    print("\n" + "=" * 80)
    print("PHASE 2: Portfolio-Level Optimization (200 trials)")
    print("=" * 80)
    
    study = optuna.create_study(direction="maximize", study_name="portfolio_optimization")
    study.optimize(
        lambda trial: portfolio_objective(trial, asset_data, macro_score, m2_acc, asset_configs),
        n_trials=200,
        show_progress_bar=True,
    )
    
    # Save study
    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    
    print(f"\nBest Portfolio Sharpe (penalized): {study.best_value:.3f}")
    print(f"\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Top 5 trials
    print(f"\nTop 5 Trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        print(f"  #{i+1}: Sharpe={t.value:.3f} | mode={t.params.get('allocation_mode')} "
              f"rebal={t.params.get('rebalance_frequency')} lev={t.params.get('max_portfolio_leverage', 0):.2f}")
    
    return study


def main():
    print("Loading all asset data...")
    asset_data, macro_score, m2_acc = load_all_data()
    
    # Phase 1: Optimize LINK
    link_params = run_phase1(asset_data, macro_score, m2_acc)
    
    # Update LINK config
    updated_configs = dict(ASSET_CONFIGS)
    link_cfg = dict(ASSET_CONFIGS["LINK"])
    for k, v in link_params.items():
        if k in link_cfg:
            link_cfg[k] = v
    updated_configs["LINK"] = link_cfg
    
    print("\nUpdated LINK params:")
    for k in ["sma_slow", "momentum_period", "rsi_entry", "trail_stop_pct", "max_position", "initial_size"]:
        print(f"  {k}: {link_cfg.get(k)}")
    
    # Phase 2: Portfolio optimization
    study = run_phase2(asset_data, macro_score, m2_acc, updated_configs)
    
    # Save results
    results = {
        "link_optimized_params": {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) 
                                   for k, v in link_params.items()},
        "portfolio_best_params": {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) 
                                   for k, v in study.best_params.items()},
        "portfolio_best_sharpe": float(study.best_value),
        "n_trials_phase1": 100,
        "n_trials_phase2": 200,
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Study saved to {STUDY_PATH}")


if __name__ == "__main__":
    main()
