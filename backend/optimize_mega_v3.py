"""
Optuna optimization for MegaStrategyV3 — 200 trials.
Optimizes leverage map, short params, and safety params.
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
from strategies.composite.mega_strategy_v3 import run_full_strategy, ASSET_CONFIGS

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    return crypto_data, cross_asset_data, macro_data


def compute_metrics(returns: pd.Series):
    if len(returns) < 30 or returns.std() == 0:
        return 0, 0, 0, 0
    equity = (1 + returns).cumprod()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    n_years = len(returns) / 252
    cagr = float(equity.iloc[-1] ** (1 / max(n_years, 0.1)) - 1)
    return sharpe, max_dd, cagr, len(returns)


def objective(trial, crypto_data, macro_data, cross_asset_data):
    # Leverage params
    lev_5 = trial.suggest_float("leverage_5", 1.5, 2.5)
    lev_4 = trial.suggest_float("leverage_4", 1.2, 2.0)
    lev_3 = trial.suggest_float("leverage_3", 0.8, 1.5)
    lev_2 = trial.suggest_float("leverage_2", 0.3, 0.8)
    lev_1 = trial.suggest_float("leverage_1", 0.1, 0.5)
    leverage_map = {5: lev_5, 4: lev_4, 3: lev_3, 2: lev_2, 1: lev_1, 0: 0.0}

    # Short params
    short_max_size = trial.suggest_float("short_max_size", 0.2, 0.8)
    short_rsi_entry = trial.suggest_int("short_rsi_entry", 60, 75)
    short_rsi_exit = trial.suggest_int("short_rsi_exit", 20, 35)
    short_trail_stop = trial.suggest_float("short_trail_stop", 0.08, 0.20)

    # Safety params
    vol_ceiling = trial.suggest_float("vol_ceiling", 0.6, 1.2)
    dd_reduction_threshold = trial.suggest_float("dd_reduction_threshold", 0.08, 0.15)
    funding_carry_daily = trial.suggest_float("funding_carry_weight", 0.0, 0.02)

    # Per-asset SMA variation (±20%)
    asset_configs = {}
    for asset, base_cfg in ASSET_CONFIGS.items():
        cfg = dict(base_cfg)
        sma_base = cfg["sma_slow"]
        mom_base = cfg["momentum_period"]
        cfg["sma_slow"] = trial.suggest_int(f"{asset}_sma", int(sma_base * 0.8), int(sma_base * 1.2))
        cfg["momentum_period"] = trial.suggest_int(f"{asset}_mom", int(mom_base * 0.8), int(mom_base * 1.2))
        asset_configs[asset] = cfg

    try:
        portfolio_df, per_asset = run_full_strategy(
            crypto_data, macro_data, cross_asset_data,
            leverage_map=leverage_map,
            asset_configs=asset_configs,
            enable_long=True, enable_short=True,
            enable_adaptive_leverage=True,
            funding_carry_daily=funding_carry_daily,
            short_max_size=short_max_size,
            short_rsi_entry=short_rsi_entry,
            short_rsi_exit=short_rsi_exit,
            short_trail_stop=short_trail_stop,
            vol_ceiling=vol_ceiling,
            dd_reduction_threshold=dd_reduction_threshold,
        )

        returns = portfolio_df["daily_pnl"]
        sharpe, max_dd, cagr, n_days = compute_metrics(returns)

        # Penalties
        penalty = 0
        if max_dd < -0.15:
            penalty += 0.3
        if max_dd < -0.25:
            penalty += 0.5

        # Trade count
        total_trades = 0
        for a, res in per_asset.items():
            long_changes = (res["long_position"].diff().abs() > 0.01).sum()
            short_changes = (res["short_position"].diff().abs() > 0.01).sum()
            total_trades += long_changes + short_changes
        if total_trades < 30:
            penalty += 0.5

        # Short side must be profitable
        short_returns = portfolio_df["short_pnl"] + portfolio_df["funding_pnl"]
        if len(short_returns) > 30 and short_returns.std() > 0:
            short_sharpe = float(short_returns.mean() / short_returns.std() * np.sqrt(252))
            if short_sharpe < 0:
                penalty += 0.3

        trial.set_user_attr("max_dd", max_dd)
        trial.set_user_attr("cagr", cagr)
        trial.set_user_attr("total_trades", int(total_trades))

        return sharpe - penalty

    except Exception as e:
        return -10.0


def main():
    print("\n" + "#" * 70)
    print("#  MegaStrategyV3 — Optuna Optimization (200 trials)")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    print("\nLoading data...")
    crypto_data, cross_asset_data, macro_data = load_data()
    print(f"  Assets: {list(crypto_data.keys())}")

    study = optuna.create_study(direction="maximize", study_name="mega_v3_optimization")

    print("\nRunning 200 trials...")
    study.optimize(
        lambda trial: objective(trial, crypto_data, macro_data, cross_asset_data),
        n_trials=200,
        show_progress_bar=True,
    )

    # Results
    best = study.best_trial
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"  Best Sharpe (penalized): {best.value:.3f}")
    print(f"  Max DD:                  {best.user_attrs.get('max_dd', 'N/A')}")
    print(f"  CAGR:                    {best.user_attrs.get('cagr', 'N/A')}")
    print(f"  Total trades:            {best.user_attrs.get('total_trades', 'N/A')}")

    print("\n  Best Parameters:")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            print(f"    {k:<30} {v:.4f}")
        else:
            print(f"    {k:<30} {v}")

    # Top 10 trials
    print("\n  Top 10 trials:")
    print(f"  {'#':>4} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8}")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    for i, t in enumerate(sorted_trials[:10]):
        md = t.user_attrs.get("max_dd", 0)
        cagr = t.user_attrs.get("cagr", 0)
        print(f"  {t.number:>4} {t.value:>8.3f} {md:>8.3f} {cagr:>8.3f}")

    # Save study
    study_path = RESULTS_DIR / "mega_v3_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\nStudy saved to {study_path}")

    # Save best params as JSON too
    params_path = RESULTS_DIR / "mega_v3_best_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "best_value": best.value,
            "best_params": best.params,
            "user_attrs": best.user_attrs,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"Best params saved to {params_path}")

    print("\n" + "#" * 70)
    print("#  OPTIMIZATION COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
