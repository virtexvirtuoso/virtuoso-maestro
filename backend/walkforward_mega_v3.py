"""
Walk-Forward Validation for MegaStrategyV3
2yr train / 6mo test, optimize leverage + short params per fold.
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


def slice_data(crypto_data, cross_asset_data, macro_data, start, end):
    """Slice all data to date range."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cd = {}
    for name, df in crypto_data.items():
        sliced = df[(df.index >= s) & (df.index <= e)]
        if len(sliced) > 50:
            cd[name] = sliced
    ca = cross_asset_data[(cross_asset_data.index >= s) & (cross_asset_data.index <= e)]
    md = macro_data[(macro_data.index >= s) & (macro_data.index <= e)]
    return cd, ca, md


def compute_metrics(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0, 0, 0
    equity = (1 + returns).cumprod()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    dd = float((equity / equity.cummax() - 1).min())
    n_years = len(returns) / 252
    cagr = float(equity.iloc[-1] ** (1 / max(n_years, 0.1)) - 1)
    return sharpe, dd, cagr


def optimize_fold(crypto_data, macro_data, cross_asset_data, n_trials=50):
    """Optimize leverage + short params on training data."""

    def objective(trial):
        lev_5 = trial.suggest_float("leverage_5", 1.5, 2.5)
        lev_4 = trial.suggest_float("leverage_4", 1.2, 2.0)
        lev_3 = trial.suggest_float("leverage_3", 0.8, 1.5)
        lev_2 = trial.suggest_float("leverage_2", 0.3, 0.8)
        lev_1 = trial.suggest_float("leverage_1", 0.1, 0.5)
        leverage_map = {5: lev_5, 4: lev_4, 3: lev_3, 2: lev_2, 1: lev_1, 0: 0.0}

        short_max_size = trial.suggest_float("short_max_size", 0.2, 0.8)
        short_rsi_entry = trial.suggest_int("short_rsi_entry", 60, 75)
        short_rsi_exit = trial.suggest_int("short_rsi_exit", 20, 35)
        short_trail_stop = trial.suggest_float("short_trail_stop", 0.08, 0.20)
        vol_ceiling = trial.suggest_float("vol_ceiling", 0.6, 1.2)
        dd_thresh = trial.suggest_float("dd_reduction_threshold", 0.08, 0.15)

        try:
            portfolio_df, _ = run_full_strategy(
                crypto_data, macro_data, cross_asset_data,
                leverage_map=leverage_map,
                enable_long=True, enable_short=True,
                enable_adaptive_leverage=True,
                short_max_size=short_max_size,
                short_rsi_entry=short_rsi_entry,
                short_rsi_exit=short_rsi_exit,
                short_trail_stop=short_trail_stop,
                vol_ceiling=vol_ceiling,
                dd_reduction_threshold=dd_thresh,
            )
            sharpe, max_dd, _ = compute_metrics(portfolio_df["daily_pnl"])
            penalty = 0
            if max_dd < -0.15:
                penalty += 0.3
            if max_dd < -0.25:
                penalty += 0.5
            return sharpe - penalty
        except:
            return -10.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_with_params(crypto_data, macro_data, cross_asset_data, params):
    """Run strategy with given params, return portfolio_df."""
    leverage_map = {
        5: params.get("leverage_5", 2.0),
        4: params.get("leverage_4", 1.5),
        3: params.get("leverage_3", 1.0),
        2: params.get("leverage_2", 0.6),
        1: params.get("leverage_1", 0.3),
        0: 0.0,
    }
    override = {k: v for k, v in params.items()
                if k in ("short_max_size", "short_rsi_entry", "short_rsi_exit",
                         "short_trail_stop", "vol_ceiling", "dd_reduction_threshold")}

    portfolio_df, per_asset = run_full_strategy(
        crypto_data, macro_data, cross_asset_data,
        leverage_map=leverage_map,
        enable_long=True, enable_short=True,
        enable_adaptive_leverage=True,
        **override,
    )
    return portfolio_df, per_asset


def main():
    print("\n" + "#" * 70)
    print("#  MegaStrategyV3 — Walk-Forward Validation")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#  2yr train / 6mo test, 50 trials per fold")
    print("#" * 70)

    print("\nLoading data...")
    crypto_data, cross_asset_data, macro_data = load_data()

    # Determine date range from BTC data
    btc = crypto_data.get("BTC")
    if btc is None:
        print("ERROR: No BTC data!")
        return

    start_date = btc.index[0]
    end_date = btc.index[-1]
    print(f"  Data range: {start_date.date()} to {end_date.date()}")

    # Generate folds: 2yr train, 6mo test
    train_months = 24
    test_months = 6
    folds = []
    current = start_date
    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end_date:
            break
        folds.append({
            "train_start": current,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })
        current += pd.DateOffset(months=test_months)

    print(f"  Generated {len(folds)} folds")

    # Run walk-forward
    fold_results = []
    all_oos_returns = []

    for i, fold in enumerate(folds):
        print(f"\n--- Fold {i+1}/{len(folds)} ---")
        print(f"  Train: {fold['train_start'].date()} → {fold['train_end'].date()}")
        print(f"  Test:  {fold['test_start'].date()} → {fold['test_end'].date()}")

        # Slice data
        train_crypto, train_ca, train_macro = slice_data(
            crypto_data, cross_asset_data, macro_data,
            fold["train_start"], fold["train_end"])
        test_crypto, test_ca, test_macro = slice_data(
            crypto_data, cross_asset_data, macro_data,
            fold["test_start"], fold["test_end"])

        if not train_crypto or not test_crypto:
            print("  SKIP — insufficient data")
            continue

        # Optimize on training data
        print(f"  Optimizing (50 trials)...")
        best_params = optimize_fold(train_crypto, train_macro, train_ca, n_trials=50)

        # In-sample performance
        try:
            is_df, _ = run_with_params(train_crypto, train_macro, train_ca, best_params)
            is_sharpe, is_dd, is_cagr = compute_metrics(is_df["daily_pnl"])
        except:
            is_sharpe, is_dd, is_cagr = 0, 0, 0

        # Out-of-sample performance
        try:
            oos_df, oos_per_asset = run_with_params(test_crypto, test_macro, test_ca, best_params)
            oos_sharpe, oos_dd, oos_cagr = compute_metrics(oos_df["daily_pnl"])

            # Long vs short OOS
            long_oos = oos_df["long_pnl"]
            short_oos = oos_df["short_pnl"] + oos_df["funding_pnl"]
            long_sharpe = float(long_oos.mean() / long_oos.std() * np.sqrt(252)) if long_oos.std() > 0 else 0
            short_sharpe = float(short_oos.mean() / short_oos.std() * np.sqrt(252)) if short_oos.std() > 0 else 0

            all_oos_returns.append(oos_df["daily_pnl"])
        except:
            oos_sharpe, oos_dd, oos_cagr = 0, 0, 0
            long_sharpe, short_sharpe = 0, 0

        fold_result = {
            "fold": i + 1,
            "train_period": f"{fold['train_start'].date()} → {fold['train_end'].date()}",
            "test_period": f"{fold['test_start'].date()} → {fold['test_end'].date()}",
            "is_sharpe": round(is_sharpe, 3),
            "is_max_dd": round(is_dd * 100, 2),
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_max_dd": round(oos_dd * 100, 2),
            "oos_cagr": round(oos_cagr * 100, 2),
            "oos_long_sharpe": round(long_sharpe, 3),
            "oos_short_sharpe": round(short_sharpe, 3),
            "best_params": {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()},
        }
        fold_results.append(fold_result)

        print(f"  IS Sharpe: {is_sharpe:.3f} | OOS Sharpe: {oos_sharpe:.3f} | "
              f"OOS DD: {oos_dd*100:.1f}% | Long: {long_sharpe:.3f} | Short: {short_sharpe:.3f}")

    # Aggregate OOS
    print("\n" + "=" * 90)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n{'Fold':>4} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS DD%':>8} {'Long Sharpe':>12} {'Short Sharpe':>13}")
    print("-" * 90)
    for r in fold_results:
        print(f"{r['fold']:>4} {r['is_sharpe']:>10.3f} {r['oos_sharpe']:>11.3f} "
              f"{r['oos_max_dd']:>8.1f} {r['oos_long_sharpe']:>12.3f} {r['oos_short_sharpe']:>13.3f}")

    if fold_results:
        oos_sharpes = [r["oos_sharpe"] for r in fold_results]
        oos_dds = [r["oos_max_dd"] for r in fold_results]
        is_sharpes = [r["is_sharpe"] for r in fold_results]

        print(f"\n  Aggregate OOS Sharpe (mean): {np.mean(oos_sharpes):.3f}")
        print(f"  Aggregate OOS Sharpe (median): {np.median(oos_sharpes):.3f}")
        print(f"  OOS Sharpe std: {np.std(oos_sharpes):.3f}")
        print(f"  Worst OOS fold: {min(oos_sharpes):.3f}")
        print(f"  Best OOS fold: {max(oos_sharpes):.3f}")
        print(f"  Avg OOS MaxDD: {np.mean(oos_dds):.1f}%")
        print(f"  IS→OOS decay: {np.mean(is_sharpes):.3f} → {np.mean(oos_sharpes):.3f} "
              f"({(1 - np.mean(oos_sharpes)/max(np.mean(is_sharpes), 0.001))*100:.0f}% decay)")

        # Combined OOS equity curve
        if all_oos_returns:
            combined = pd.concat(all_oos_returns)
            combined_sharpe, combined_dd, combined_cagr = compute_metrics(combined)
            print(f"\n  Combined OOS Sharpe: {combined_sharpe:.3f}")
            print(f"  Combined OOS MaxDD:  {combined_dd*100:.1f}%")
            print(f"  Combined OOS CAGR:   {combined_cagr*100:.1f}%")

        # Statistical significance (t-test on OOS Sharpes > 0)
        if len(oos_sharpes) >= 3:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(oos_sharpes, 0)
            print(f"\n  Statistical significance:")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: {p_value:.4f}")
            print(f"    Significant at 5%: {'YES ✓' if p_value < 0.05 else 'NO ✗'}")
            print(f"    Significant at 10%: {'YES ✓' if p_value < 0.10 else 'NO ✗'}")

        # Comparison to V2
        print(f"\n  Comparison to V2 walk-forward:")
        print(f"    V2 OOS Sharpe: 0.578")
        print(f"    V3 OOS Sharpe: {np.mean(oos_sharpes):.3f}")
        improvement = (np.mean(oos_sharpes) - 0.578) / 0.578 * 100
        print(f"    Improvement: {improvement:+.1f}%")

        # Does short side work OOS?
        short_sharpes = [r["oos_short_sharpe"] for r in fold_results]
        short_positive = sum(1 for s in short_sharpes if s > 0)
        print(f"\n  Short side OOS analysis:")
        print(f"    Mean short Sharpe: {np.mean(short_sharpes):.3f}")
        print(f"    Positive folds: {short_positive}/{len(short_sharpes)}")
        print(f"    Short adds value: {'YES ✓' if np.mean(short_sharpes) > 0 else 'INCONCLUSIVE'}")

    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": len(fold_results),
        "train_months": train_months,
        "test_months": test_months,
        "fold_results": fold_results,
        "aggregate": {
            "oos_sharpe_mean": round(np.mean(oos_sharpes), 3) if fold_results else 0,
            "oos_sharpe_median": round(np.median(oos_sharpes), 3) if fold_results else 0,
            "oos_sharpe_std": round(np.std(oos_sharpes), 3) if fold_results else 0,
            "oos_max_dd_mean": round(np.mean(oos_dds), 2) if fold_results else 0,
            "v2_comparison_sharpe": 0.578,
        },
    }

    out_path = RESULTS_DIR / "walkforward_mega_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "#" * 70)
    print("#  WALK-FORWARD COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
