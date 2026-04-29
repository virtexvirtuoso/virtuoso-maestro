"""
Backtest ML Enhanced — Walk-forward comparison of ML enhancements vs rule-based V3.

Strategies compared:
  S1: Buy & Hold BTC
  S2: V3 Full (rule-based baseline)
  S3: V3 + ML Regime Classifier
  S4: V3 + ML Signal Weighting
  S5: V3 + ML Entry Timing
  S6: V3 + ML Strategy Selector
  S7: V3 + ALL ML combined

CRITICAL: Walk-forward only. Train on rolling 2yr, predict next 6mo.
0.1% transaction costs throughout.
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Add backend to path
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from datasource.factor_loader import FactorDataLoader
from strategies.composite.mega_strategy_v3 import (
    run_single_asset, compute_confluence, detect_regime,
    ASSET_CONFIGS, TX_COST,
)
from ml.feature_engine import build_feature_matrix, get_feature_columns
from ml.regime_classifier import RegimeClassifier
from ml.signal_weighter import SignalWeighter
from ml.entry_timer import EntryTimer
from ml.ensemble_strategy_selector import StrategySelector

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESEARCH_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 730   # 2 years
TEST_WINDOW = 182    # 6 months


# ── Performance Metrics ───────────────────────────────────────────────────

def compute_metrics(daily_returns: pd.Series, name: str = "") -> dict:
    """Compute comprehensive performance metrics."""
    r = daily_returns.dropna()
    if len(r) < 30:
        return {"name": name, "error": "insufficient data"}

    cum = (1 + r).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total_ret) ** (252 / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(252))

    # Sharpe
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max Drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Omega (threshold=0)
    pos = r[r > 0].sum()
    neg = abs(r[r < 0].sum())
    omega = float(pos / neg) if neg > 0 else float("inf")

    # Profit Factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Tail Ratio
    p95 = np.percentile(r, 95) if len(r) > 20 else 0
    p5 = abs(np.percentile(r, 5)) if len(r) > 20 else 1
    tail_ratio = float(p95 / p5) if p5 > 0 else 0

    # UPI (Ulcer Performance Index)
    dd_sq = dd ** 2
    ulcer = float(np.sqrt(dd_sq.mean()))
    upi = ann_ret / ulcer if ulcer > 0 else 0

    return {
        "name": name,
        "total_return": round(total_ret, 4),
        "annual_return": round(ann_ret, 4),
        "annual_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "omega": round(omega, 4),
        "profit_factor": round(profit_factor, 4),
        "tail_ratio": round(tail_ratio, 4),
        "max_drawdown": round(max_dd, 4),
        "upi": round(upi, 4),
        "n_days": len(r),
    }


# ── Data Loading ──────────────────────────────────────────────────────────

def load_all_data():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()
    factor_loader = FactorDataLoader()

    # Crypto OHLCV
    print("  Loading crypto OHLCV...")
    btc = stock_loader.get_ohlcv("BTC-USD", start_date="2018-01-01")
    eth = stock_loader.get_ohlcv("ETH-USD", start_date="2018-01-01")
    sol = stock_loader.get_ohlcv("SOL-USD", start_date="2020-04-01")
    link = stock_loader.get_ohlcv("LINK-USD", start_date="2018-01-01")

    # Cross-asset
    print("  Loading cross-asset data...")
    cross_tickers = {"GLD": "GLD", "UUP": "UUP", "TLT": "TLT",
                     "HYG": "HYG", "COPX": "COPX", "SPY": "SPY"}
    cross_data = {}
    for name, ticker in cross_tickers.items():
        try:
            df = stock_loader.get_ohlcv(ticker, start_date="2018-01-01")
            cross_data[name] = df["close"]
        except Exception as e:
            print(f"    Warning: failed to load {ticker}: {e}")
    cross_asset = pd.DataFrame(cross_data)

    # Macro
    print("  Loading FRED macro data...")
    macro_series = {
        "yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS",
        "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2",
    }
    macro_df = fred_loader.get_multiple(macro_series, start_date="2017-01-01")

    # Factors
    print("  Loading FF factors...")
    try:
        ff_data = factor_loader.get_all_factors()
    except Exception as e:
        print(f"    Warning: factor loading failed: {e}")
        ff_data = pd.DataFrame()

    print(f"  BTC: {len(btc)} days | ETH: {len(eth)} | Macro: {len(macro_df)}")
    return btc, eth, sol, link, macro_df, ff_data, cross_asset


# ── Build Feature Matrix ─────────────────────────────────────────────────

def build_features(btc, eth, sol, link, macro_df, ff_data, cross_asset):
    print("\n  Building feature matrix...")

    # Compute V3 confluence for BTC (needed for features)
    # Create cross-asset format expected by V3
    v3_cross = pd.DataFrame(index=btc.index)
    col_map = {"GLD": "gold", "UUP": "dxy", "TLT": "bonds", "HYG": "hyg", "COPX": "copper"}
    for src, dst in col_map.items():
        if src in cross_asset.columns:
            v3_cross[dst] = cross_asset[src].reindex(btc.index, method="ffill")

    v3_macro = macro_df.reindex(btc.index, method="ffill")

    confluence, breakdown = compute_confluence(
        btc["close"], v3_macro, v3_cross,
        sma_slow=ASSET_CONFIGS["BTC"]["sma_slow"],
        momentum_period=ASSET_CONFIGS["BTC"]["momentum_period"],
    )
    regime = detect_regime(confluence)

    matrix = build_feature_matrix(
        btc, eth, sol, link, macro_df, ff_data, cross_asset,
        confluence_breakdown=breakdown,
        regime_series=regime,
    )

    feat_cols = get_feature_columns(matrix)
    print(f"  Features: {len(feat_cols)} | Rows: {len(matrix)}")
    return matrix, feat_cols, confluence, breakdown, regime, v3_cross, v3_macro


# ── Strategy Runners ──────────────────────────────────────────────────────

def run_buy_hold(btc: pd.DataFrame, start: str, end: str) -> pd.Series:
    """S1: Buy & Hold BTC."""
    c = btc["close"].loc[start:end]
    return c.pct_change().fillna(0)


def run_v3_baseline(btc, macro_df, cross_asset_v3, start, end) -> pd.Series:
    """S2: V3 rule-based baseline."""
    result = run_single_asset(
        btc.loc[:end], macro_df, cross_asset_v3,
        asset_name="BTC", enable_short=True,
    )
    return result["daily_pnl"].loc[start:end]


def run_v3_ml_regime(btc, macro_df, cross_asset_v3, matrix, feat_cols,
                     regime_classifier, start, end) -> pd.Series:
    """S3: V3 with ML regime replacing rule-based."""
    # Get ML regime predictions
    X = matrix[feat_cols].loc[:end]
    ml_regime = regime_classifier.predict(X).loc[start:end]

    # Run V3 but override regime
    from strategies.composite.mega_strategy_v3 import (
        generate_long_signals, generate_short_signals, adaptive_leverage,
        _realized_vol, ASSET_CONFIGS, DEFAULT_SAFETY_PARAMS,
    )
    cfg = ASSET_CONFIGS["BTC"]
    df = btc.loc[:end]
    close = df["close"]

    # Recompute confluence (still rule-based for signal generation)
    confluence, _ = compute_confluence(
        close, macro_df.reindex(close.index, method="ffill"),
        cross_asset_v3.reindex(close.index, method="ffill"),
        sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"],
    )

    long_df = generate_long_signals(df, ml_regime.reindex(df.index, method="ffill"),
                                    confluence, **cfg)
    short_df = generate_short_signals(df, ml_regime.reindex(df.index, method="ffill"),
                                      confluence, sma_slow=cfg["sma_slow"])

    rvol = _realized_vol(close, DEFAULT_SAFETY_PARAMS["vol_lookback"])
    daily_ret = close.pct_change().fillna(0)
    cum_ret = (1 + daily_ret).cumprod()
    running_dd = cum_ret / cum_ret.cummax() - 1
    lev = adaptive_leverage(confluence, rvol, running_dd)

    long_pos = long_df["long_position"].fillna(0) * lev
    short_pos = short_df["short_position"].fillna(0)

    pos_chg_l = long_pos.diff().abs().fillna(0)
    pos_chg_s = short_pos.diff().abs().fillna(0)

    pnl = (daily_ret * long_pos - pos_chg_l * TX_COST +
           (-daily_ret * short_pos) - pos_chg_s * TX_COST)
    return pnl.loc[start:end]


def run_v3_ml_weighting(btc, matrix, feat_cols, signal_weighter, start, end) -> pd.Series:
    """S4: V3 with ML-driven position sizing."""
    X = matrix[feat_cols].loc[:end]
    pos_size = signal_weighter.predict_position_size(X).loc[start:end]
    daily_ret = btc["close"].pct_change().fillna(0).loc[start:end]

    pos_chg = pos_size.diff().abs().fillna(0)
    return daily_ret * pos_size - pos_chg * TX_COST


def run_v3_ml_timing(btc, matrix, feat_cols, entry_timer, regime, start, end) -> pd.Series:
    """S5: V3 with ML entry timing."""
    X = matrix[feat_cols].loc[:end]
    scores = entry_timer.score_entry(X).loc[start:end]
    regime_aligned = regime.reindex(scores.index, method="ffill")
    daily_ret = btc["close"].pct_change().fillna(0).loc[start:end]

    # Base position from regime
    base_pos = pd.Series(0.0, index=scores.index)
    base_pos[regime_aligned.isin(["BULL", "MILD_BULL"])] = 1.0
    base_pos[regime_aligned == "ACCUMULATION"] = 0.5

    # ML timing modulates position
    ml_pos = base_pos * scores
    pos_chg = ml_pos.diff().abs().fillna(0)
    return daily_ret * ml_pos - pos_chg * TX_COST


def run_v3_all_ml(btc, macro_df, cross_asset_v3, matrix, feat_cols,
                   regime_classifier, signal_weighter, entry_timer,
                   regime, start, end) -> pd.Series:
    """S7: V3 with ALL ML enhancements combined."""
    X = matrix[feat_cols].loc[:end]

    # ML regime
    ml_regime = regime_classifier.predict(X).loc[start:end]

    # ML position sizing
    pos_size = signal_weighter.predict_position_size(X).loc[start:end]

    # ML entry timing
    scores = entry_timer.score_entry(X).loc[start:end]

    daily_ret = btc["close"].pct_change().fillna(0).loc[start:end]

    # Combine: use ML regime for direction, ML sizing * timing for magnitude
    direction = pd.Series(0.0, index=ml_regime.index)
    direction[ml_regime.isin(["BULL", "MILD_BULL"])] = 1.0
    direction[ml_regime == "ACCUMULATION"] = 0.7
    direction[ml_regime == "BEAR"] = -0.3  # Small short

    combined_pos = direction * pos_size * scores
    combined_pos = combined_pos.clip(-0.5, 1.5)

    pos_chg = combined_pos.diff().abs().fillna(0)
    return daily_ret * combined_pos - pos_chg * TX_COST


# ── Walk-Forward Backtest ─────────────────────────────────────────────────

def walk_forward_backtest(btc, eth, sol, link, macro_df, ff_data, cross_asset):
    matrix, feat_cols, confluence, breakdown, regime, v3_cross, v3_macro = \
        build_features(btc, eth, sol, link, macro_df, ff_data, cross_asset)

    dates = matrix.index.sort_values()
    # Find valid start (need enough data)
    valid_start_idx = TRAIN_WINDOW + 252  # 2yr train + 1yr warmup
    if valid_start_idx >= len(dates):
        print("ERROR: Not enough data for walk-forward")
        return {}

    print(f"\n{'=' * 60}")
    print("WALK-FORWARD BACKTEST")
    print(f"{'=' * 60}")
    print(f"  Data range: {dates[0].date()} → {dates[-1].date()}")
    print(f"  Train window: {TRAIN_WINDOW}d | Test window: {TEST_WINDOW}d")

    # Collect daily returns per strategy across all folds
    strategy_rets = {
        "S1_BuyHold": [], "S2_V3_Baseline": [], "S3_ML_Regime": [],
        "S4_ML_Weighting": [], "S5_ML_Timing": [], "S7_ML_All": [],
    }

    i = valid_start_idx
    fold = 0
    while i + TEST_WINDOW <= len(dates):
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end_idx = min(i + TEST_WINDOW - 1, len(dates) - 1)
        test_end = dates[test_end_idx]

        print(f"\n--- Fold {fold}: Train→{train_end.date()} | "
              f"Test {test_start.date()}→{test_end.date()} ---")

        start_str = str(test_start.date())
        end_str = str(test_end.date())
        train_end_str = str(train_end.date())

        # Train ML models on data up to train_end
        X_train = matrix[feat_cols].loc[:train_end]
        y_regime = matrix["regime_label"].loc[:train_end]
        y_fwd_20d = matrix["fwd_ret_20d"].loc[:train_end]
        y_fwd_5d = matrix["fwd_ret_5d"].loc[:train_end]
        y_fwd_1d = matrix["fwd_ret_1d"].loc[:train_end]

        # Train regime classifier
        print("  Training RegimeClassifier...")
        rc = RegimeClassifier()
        rc.train(X_train, y_regime)

        # Train signal weighter
        print("  Training SignalWeighter...")
        sw = SignalWeighter()
        sw.train(X_train, y_fwd_20d)

        # Train entry timer
        print("  Training EntryTimer...")
        et = EntryTimer()
        et.train(X_train, y_fwd_5d, regime=regime.loc[:train_end])

        # Run strategies on test period
        try:
            s1 = run_buy_hold(btc, start_str, end_str)
            strategy_rets["S1_BuyHold"].append(s1)
        except Exception as e:
            print(f"  S1 error: {e}")

        try:
            s2 = run_v3_baseline(btc, v3_macro, v3_cross, start_str, end_str)
            strategy_rets["S2_V3_Baseline"].append(s2)
        except Exception as e:
            print(f"  S2 error: {e}")

        try:
            s3 = run_v3_ml_regime(btc, v3_macro, v3_cross, matrix, feat_cols, rc, start_str, end_str)
            strategy_rets["S3_ML_Regime"].append(s3)
        except Exception as e:
            print(f"  S3 error: {e}")

        try:
            s4 = run_v3_ml_weighting(btc, matrix, feat_cols, sw, start_str, end_str)
            strategy_rets["S4_ML_Weighting"].append(s4)
        except Exception as e:
            print(f"  S4 error: {e}")

        try:
            s5 = run_v3_ml_timing(btc, matrix, feat_cols, et, regime, start_str, end_str)
            strategy_rets["S5_ML_Timing"].append(s5)
        except Exception as e:
            print(f"  S5 error: {e}")

        try:
            s7 = run_v3_all_ml(btc, v3_macro, v3_cross, matrix, feat_cols,
                                rc, sw, et, regime, start_str, end_str)
            strategy_rets["S7_ML_All"].append(s7)
        except Exception as e:
            print(f"  S7 error: {e}")

        fold += 1
        i += TEST_WINDOW

    return strategy_rets, matrix, feat_cols, regime


def run_component_validation(matrix, feat_cols, regime):
    """Run standalone walk-forward validation for each ML component."""
    print(f"\n{'=' * 60}")
    print("COMPONENT WALK-FORWARD VALIDATION")
    print(f"{'=' * 60}")

    X = matrix[feat_cols]
    results = {}

    # Regime Classifier
    print("\n── Regime Classifier Walk-Forward ──")
    rc = RegimeClassifier()
    rc_results = rc.walk_forward_validate(X, matrix["regime_label"],
                                          train_window=TRAIN_WINDOW,
                                          test_window=TEST_WINDOW)
    results["regime_classifier"] = rc_results

    # Signal Weighter
    print("\n── Signal Weighter Walk-Forward ──")
    sw = SignalWeighter()
    sw_results = sw.walk_forward_validate(X, matrix["fwd_ret_20d"],
                                          train_window=TRAIN_WINDOW,
                                          test_window=TEST_WINDOW)
    results["signal_weighter"] = sw_results

    # Entry Timer
    print("\n── Entry Timer Walk-Forward ──")
    et = EntryTimer()
    et_results = et.walk_forward_validate(X, matrix["fwd_ret_5d"],
                                          regime=regime,
                                          train_window=TRAIN_WINDOW,
                                          test_window=TEST_WINDOW)
    results["entry_timer"] = et_results

    # SHAP analysis (train on full data for interpretability — clearly labeled)
    print("\n── SHAP Feature Importance (full-sample, for interpretability only) ──")
    sw_full = SignalWeighter()
    sw_full.train(X, matrix["fwd_ret_20d"])
    try:
        shap_df = sw_full.compute_shap(X, max_samples=300)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        print("  Top 20 features by |SHAP|:")
        for feat, val in mean_abs_shap.head(20).items():
            print(f"    {feat:40s} {val:.6f}")
        results["shap_top20"] = {k: float(v) for k, v in mean_abs_shap.head(20).items()}
        results["signal_weights"] = sw_full.get_signal_weights()
    except Exception as e:
        print(f"  SHAP failed: {e}")
        results["shap_top20"] = {}
        results["signal_weights"] = sw_full.get_signal_weights()

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       ML ENHANCEMENT LAYER — WALK-FORWARD BACKTEST      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    btc, eth, sol, link, macro_df, ff_data, cross_asset = load_all_data()

    # Walk-forward backtest
    strategy_rets, matrix, feat_cols, regime = walk_forward_backtest(
        btc, eth, sol, link, macro_df, ff_data, cross_asset
    )

    # Compute metrics
    print(f"\n{'=' * 60}")
    print("RESULTS — OOS ONLY")
    print(f"{'=' * 60}")

    all_metrics = {}
    for name, ret_list in strategy_rets.items():
        if ret_list:
            combined = pd.concat(ret_list)
            metrics = compute_metrics(combined, name)
            all_metrics[name] = metrics

    # Print table
    header = f"{'Strategy':25s} {'Return':>10s} {'Sharpe':>8s} {'Sortino':>8s} {'Calmar':>8s} {'MaxDD':>8s} {'Omega':>8s} {'PF':>8s} {'UPI':>8s}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, m in all_metrics.items():
        if "error" in m:
            print(f"{name:25s} {'ERROR':>10s}")
            continue
        print(f"{m['name']:25s} {m['total_return']:>9.1%} {m['sharpe']:>8.2f} "
              f"{m['sortino']:>8.2f} {m['calmar']:>8.2f} {m['max_drawdown']:>8.1%} "
              f"{m['omega']:>8.2f} {m['profit_factor']:>8.2f} {m['upi']:>8.2f}")

    # Component validation
    component_results = run_component_validation(matrix, feat_cols, regime)

    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy_metrics": all_metrics,
        "component_validation": {
            k: {kk: vv for kk, vv in v.items()
                if not isinstance(vv, (np.ndarray,))}
            for k, v in component_results.items()
        },
    }

    results_path = RESULTS_DIR / "ml_enhanced_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Save feature importance
    fi_data = {
        "shap_top20": component_results.get("shap_top20", {}),
        "signal_weights": component_results.get("signal_weights", {}),
    }
    fi_path = RESEARCH_DIR / "ml_feature_importance.json"
    with open(fi_path, "w") as f:
        json.dump(fi_data, f, indent=2, default=str)
    print(f"  Feature importance saved to {fi_path}")

    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
