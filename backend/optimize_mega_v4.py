"""
Optuna Optimization for MegaStrategyV4 — 200 trials.
"""
import sys, os, json, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from datetime import datetime

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v4 import run_mega_v4

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/optimization"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}


def load_data():
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()

    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
        except:
            pass

    cross_asset_data = pd.DataFrame()
    for col_name, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col_name] = df["close"]
        except:
            pass

    macro_data = fred_loader.get_multiple(FRED_SERIES, start_date="2015-01-01")

    ff_data = None
    try:
        from datasource.factor_loader import FactorDataLoader
        ff_data = FactorDataLoader().get_ff5()
    except:
        pass

    return crypto_data, cross_asset_data, macro_data, ff_data


def objective(trial, crypto_data, cross_asset_data, macro_data, ff_data):
    # Risk budgets
    rb_v3 = trial.suggest_float("risk_budget_v3", 0.35, 0.60)
    rb_vol = trial.suggest_float("risk_budget_vol", 0.05, 0.25)
    rb_ema = trial.suggest_float("risk_budget_ema", 0.05, 0.25)
    rb_ff = trial.suggest_float("risk_budget_ff", 0.05, 0.20)
    rb_mtf = trial.suggest_float("risk_budget_mtf", 0.05, 0.20)

    # Normalize
    total = rb_v3 + rb_vol + rb_ema + rb_ff + rb_mtf
    risk_budgets = {
        "v3_core": rb_v3 / total,
        "vol_breakout": rb_vol / total,
        "ema_ribbon": rb_ema / total,
        "ff_bridge": rb_ff / total,
        "multitf": rb_mtf / total,
    }

    # Other params
    params = {
        "max_total_leverage": trial.suggest_float("max_total_leverage", 1.5, 3.0),
        "vol_target": trial.suggest_float("vol_target", 0.15, 0.50),
        "module_dd_breaker": trial.suggest_float("module_dd_breaker", 0.05, 0.15),
        "corr_threshold": trial.suggest_float("corr_threshold", 0.5, 0.8),
        "bb_squeeze_pctile": trial.suggest_int("bb_squeeze_pctile", 10, 30),
        "bb_atr_mult": trial.suggest_float("bb_atr_mult", 1.0, 2.5),
        "ff_lookback_months": trial.suggest_int("ff_lookback", 1, 6),
        "mtf_fast": trial.suggest_int("mtf_fast", 3, 10),
        "mtf_medium": trial.suggest_int("mtf_medium", 15, 40),
        "mtf_slow": trial.suggest_int("mtf_slow", 60, 180),
    }

    try:
        result = run_mega_v4(
            crypto_data, macro_data, cross_asset_data, ff_data,
            risk_budgets=risk_budgets, **params,
        )
        returns = result["portfolio_returns"]

        if len(returns) < 100 or returns.std() == 0:
            return -10.0

        # Metrics
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        equity = (1 + returns).cumprod()
        max_dd = float((equity / equity.cummax() - 1).min())

        total_trades = sum(r.get("trades", 0) for r in result["module_results"].values())

        # Penalties
        penalty = 0
        if max_dd < -0.15:
            penalty += (abs(max_dd) - 0.15) * 5
        if total_trades < 30:
            penalty += (30 - total_trades) * 0.01

        return sharpe - penalty

    except Exception as e:
        return -10.0


def main():
    print("=" * 70)
    print("MEGA STRATEGY V4 — OPTUNA OPTIMIZATION (200 trials)")
    print("=" * 70)

    print("\nLoading data...")
    crypto_data, cross_asset_data, macro_data, ff_data = load_data()
    print(f"  Assets: {list(crypto_data.keys())}")

    study = optuna.create_study(
        direction="maximize",
        study_name="mega_v4_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, crypto_data, cross_asset_data, macro_data, ff_data),
        n_trials=200,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    best = study.best_trial
    print(f"\nBest Sharpe (penalized): {best.value:.4f}")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Normalize risk budgets for display
    rb_keys = [k for k in best.params if k.startswith("risk_budget_")]
    rb_total = sum(best.params[k] for k in rb_keys)
    print(f"\nNormalized Risk Budgets:")
    for k in rb_keys:
        print(f"  {k}: {best.params[k]/rb_total:.3f}")

    # Top 10 trials
    print(f"\nTop 10 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    print(f"{'Rank':>5} {'Sharpe':>8} {'MaxLev':>8} {'VolTgt':>8} {'DDBreak':>8}")
    for i, t in enumerate(trials_sorted[:10]):
        if t.value is not None:
            print(f"{i+1:>5} {t.value:>8.3f} {t.params.get('max_total_leverage', 0):>8.2f} "
                  f"{t.params.get('vol_target', 0):>8.3f} {t.params.get('module_dd_breaker', 0):>8.3f}")

    # Save study
    study_path = RESULTS_DIR / "mega_v4_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\nStudy saved to {study_path}")

    # Also save best params as JSON
    params_path = RESULTS_DIR / "mega_v4_best_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "best_value": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Best params saved to {params_path}")


if __name__ == "__main__":
    main()
