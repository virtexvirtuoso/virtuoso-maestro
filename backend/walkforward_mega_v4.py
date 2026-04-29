"""
Walk-Forward Validation for MegaStrategyV4 — 2yr train / 6mo test, rolling.
"""
import sys, os, json, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from datetime import datetime
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v4 import run_mega_v4
from strategies.composite.mega_strategy_v3 import run_full_strategy as run_v3_full

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


def slice_data(crypto_data, start, end):
    sliced = {}
    for name, df in crypto_data.items():
        mask = (df.index >= start) & (df.index <= end)
        s = df[mask]
        if len(s) > 30:
            sliced[name] = s
    return sliced


def compute_sharpe(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() * 252 / (returns.std() * np.sqrt(252)))


def compute_maxdd(returns):
    if len(returns) < 2:
        return 0.0
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def optimize_fold(crypto_train, cross_asset, macro_data, ff_data, n_trials=50):
    """Optimize on training data, return best params."""
    def obj(trial):
        rb_v3 = trial.suggest_float("risk_budget_v3", 0.35, 0.60)
        rb_vol = trial.suggest_float("risk_budget_vol", 0.05, 0.25)
        rb_ema = trial.suggest_float("risk_budget_ema", 0.05, 0.25)
        rb_ff = trial.suggest_float("risk_budget_ff", 0.05, 0.20)
        rb_mtf = trial.suggest_float("risk_budget_mtf", 0.05, 0.20)
        total = rb_v3 + rb_vol + rb_ema + rb_ff + rb_mtf
        risk_budgets = {
            "v3_core": rb_v3/total, "vol_breakout": rb_vol/total,
            "ema_ribbon": rb_ema/total, "ff_bridge": rb_ff/total, "multitf": rb_mtf/total,
        }
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
            result = run_mega_v4(crypto_train, macro_data, cross_asset, ff_data,
                                risk_budgets=risk_budgets, **params)
            ret = result["portfolio_returns"]
            sharpe = compute_sharpe(ret)
            maxdd = compute_maxdd(ret)
            penalty = max(0, (abs(maxdd) - 0.15) * 5) if maxdd < -0.15 else 0
            return sharpe - penalty
        except:
            return -10.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials)
    return study.best_params, study.best_value


def run_with_params(crypto_data, cross_asset, macro_data, ff_data, params):
    """Run V4 with specific params, return returns series."""
    # Extract risk budgets from params
    rb_keys = ["risk_budget_v3", "risk_budget_vol", "risk_budget_ema", "risk_budget_ff", "risk_budget_mtf"]
    rb_vals = [params.get(k, 0.2) for k in rb_keys]
    total = sum(rb_vals)
    risk_budgets = {
        "v3_core": rb_vals[0]/total, "vol_breakout": rb_vals[1]/total,
        "ema_ribbon": rb_vals[2]/total, "ff_bridge": rb_vals[3]/total, "multitf": rb_vals[4]/total,
    }
    other_params = {k: v for k, v in params.items() if not k.startswith("risk_budget_")}
    # Map ff_lookback -> ff_lookback_months
    if "ff_lookback" in other_params:
        other_params["ff_lookback_months"] = other_params.pop("ff_lookback")

    result = run_mega_v4(crypto_data, macro_data, cross_asset, ff_data,
                        risk_budgets=risk_budgets, **other_params)
    return result["portfolio_returns"], result


def main():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION — MegaStrategyV4")
    print("2yr train / 6mo test, rolling")
    print("=" * 70)

    print("\nLoading data...")
    crypto_data, cross_asset_data, macro_data, ff_data = load_data()

    # Find common date range
    all_indices = [crypto_data[a].index for a in crypto_data]
    common_start = max(idx[0] for idx in all_indices)
    common_end = min(idx[-1] for idx in all_indices)
    print(f"Common range: {common_start.date()} to {common_end.date()}")

    # Generate folds: 2yr train + 6mo test
    train_days = 252 * 2  # ~2 years
    test_days = 126  # ~6 months
    step_days = 126  # roll by 6 months

    total_days = (common_end - common_start).days
    folds = []
    current = common_start + pd.Timedelta(days=train_days)

    while current + pd.Timedelta(days=test_days) <= common_end:
        train_start = current - pd.Timedelta(days=train_days)
        train_end = current
        test_start = current
        test_end = current + pd.Timedelta(days=test_days)
        folds.append((train_start, train_end, test_start, test_end))
        current += pd.Timedelta(days=step_days)

    print(f"Number of folds: {len(folds)}")

    fold_results = []
    v3_oos_sharpes = []
    v4_oos_sharpes = []
    v4_oos_returns_all = []

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        print(f"\n--- Fold {i+1}/{len(folds)}: train {tr_start.date()}-{tr_end.date()} | test {te_start.date()}-{te_end.date()} ---")

        crypto_train = slice_data(crypto_data, tr_start, tr_end)
        crypto_test = slice_data(crypto_data, te_start, te_end)

        if not crypto_train or not crypto_test:
            print("  Skipping (insufficient data)")
            continue

        # Optimize V4 on training data
        print(f"  Optimizing V4 (50 trials)...")
        try:
            best_params, is_value = optimize_fold(crypto_train, cross_asset_data, macro_data, ff_data, n_trials=50)
            print(f"  IS Sharpe: {is_value:.3f}")
        except Exception as e:
            print(f"  Optimization failed: {e}")
            continue

        # Run V4 OOS
        try:
            v4_oos_ret, v4_oos_result = run_with_params(crypto_test, cross_asset_data, macro_data, ff_data, best_params)
            v4_oos_sharpe = compute_sharpe(v4_oos_ret)
            v4_oos_maxdd = compute_maxdd(v4_oos_ret)
            print(f"  V4 OOS Sharpe: {v4_oos_sharpe:.3f}, MaxDD: {v4_oos_maxdd*100:.1f}%")
            v4_oos_sharpes.append(v4_oos_sharpe)
            v4_oos_returns_all.append(v4_oos_ret)
        except Exception as e:
            print(f"  V4 OOS failed: {e}")
            v4_oos_sharpe = 0
            v4_oos_maxdd = 0

        # Run V3 OOS (fixed params, no optimization needed)
        try:
            v3_port, _ = run_v3_full(crypto_test, macro_data, cross_asset_data,
                                      enable_long=True, enable_short=True, enable_adaptive_leverage=True)
            v3_oos_ret = v3_port["daily_pnl"]
            v3_oos_sharpe = compute_sharpe(v3_oos_ret)
            print(f"  V3 OOS Sharpe: {v3_oos_sharpe:.3f}")
            v3_oos_sharpes.append(v3_oos_sharpe)
        except Exception as e:
            print(f"  V3 OOS failed: {e}")
            v3_oos_sharpe = 0
            v3_oos_sharpes.append(0)

        # Per-module OOS contribution
        module_contribs = {}
        if isinstance(v4_oos_result.get("module_contributions"), pd.DataFrame):
            mc = v4_oos_result["module_contributions"]
            for col in mc.columns:
                module_contribs[col] = round(float((1 + mc[col]).prod() - 1) * 100, 2)

        fold_results.append({
            "fold": i + 1,
            "train": f"{tr_start.date()} to {tr_end.date()}",
            "test": f"{te_start.date()} to {te_end.date()}",
            "is_sharpe": round(is_value, 3),
            "v4_oos_sharpe": round(v4_oos_sharpe, 3),
            "v4_oos_maxdd": round(v4_oos_maxdd * 100, 1),
            "v3_oos_sharpe": round(v3_oos_sharpe, 3),
            "best_params": {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()},
            "module_contributions": module_contribs,
        })

    # Summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    if fold_results:
        print(f"\n{'Fold':>5} {'IS Sharpe':>10} {'V4 OOS':>10} {'V3 OOS':>10} {'V4 MaxDD%':>10} {'V4>V3?':>8}")
        print("-" * 55)
        for fr in fold_results:
            v4_wins = "✓" if fr["v4_oos_sharpe"] > fr["v3_oos_sharpe"] else "✗"
            print(f"{fr['fold']:>5} {fr['is_sharpe']:>10.3f} {fr['v4_oos_sharpe']:>10.3f} "
                  f"{fr['v3_oos_sharpe']:>10.3f} {fr['v4_oos_maxdd']:>10.1f} {v4_wins:>8}")

        mean_v4 = np.mean(v4_oos_sharpes) if v4_oos_sharpes else 0
        mean_v3 = np.mean(v3_oos_sharpes) if v3_oos_sharpes else 0
        mean_is = np.mean([f["is_sharpe"] for f in fold_results])

        print(f"\nMean IS Sharpe:     {mean_is:.3f}")
        print(f"Mean V4 OOS Sharpe: {mean_v4:.3f}")
        print(f"Mean V3 OOS Sharpe: {mean_v3:.3f}")
        print(f"V3 WF Benchmark:    0.84")
        print(f"V4 beats V3:        {sum(1 for v4, v3 in zip(v4_oos_sharpes, v3_oos_sharpes) if v4 > v3)}/{len(v4_oos_sharpes)} folds")

        # Statistical significance (paired t-test)
        if len(v4_oos_sharpes) >= 3 and len(v3_oos_sharpes) >= 3:
            t_stat, p_value = stats.ttest_rel(v4_oos_sharpes[:len(v3_oos_sharpes)], v3_oos_sharpes[:len(v4_oos_sharpes)])
            print(f"\nPaired t-test (V4 vs V3):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
            print(f"  Significant at 10%: {'Yes' if p_value < 0.10 else 'No'}")

        # Module contributions across folds
        print(f"\nPer-Module Mean OOS Contribution:")
        all_modules = set()
        for fr in fold_results:
            all_modules.update(fr.get("module_contributions", {}).keys())
        for mod in sorted(all_modules):
            vals = [fr["module_contributions"].get(mod, 0) for fr in fold_results if mod in fr.get("module_contributions", {})]
            if vals:
                print(f"  {mod:<20}: {np.mean(vals):>8.2f}%")

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": len(fold_results),
        "mean_v4_oos_sharpe": round(mean_v4, 3) if fold_results else 0,
        "mean_v3_oos_sharpe": round(mean_v3, 3) if fold_results else 0,
        "v3_benchmark_oos_sharpe": 0.84,
        "folds": fold_results,
    }
    out_path = RESULTS_DIR / "walkforward_v4_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
