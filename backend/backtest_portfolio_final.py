"""
Comprehensive Final Backtest — 7 Strategy Comparison for MacroMomentum Portfolio.

S1: Buy & Hold BTC
S2: Buy & Hold equal-weight BTC+ETH+SOL+LINK
S3: V2 BTC only (optimized params)
S4: Portfolio equal-weight
S5: Portfolio risk-parity
S6: Portfolio momentum-weighted
S7: Portfolio Optuna-optimized
"""
import sys, os, warnings, json, pickle
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro/backend"))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.macro_score_builder import compute_macro_score
from strategies.composite.macro_momentum_v2 import generate_signals
from strategies.composite.macro_momentum_portfolio import (
    ASSET_CONFIGS, run_portfolio_backtest, TX_COST, _compute_metrics, _risk_metrics, _crash_analysis,
)

RESULTS_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results/portfolio_final_results.json"))
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
STUDY_PATH = Path(os.path.expanduser("~/Desktop/maestro/data/optimization/portfolio_study.pkl"))


def load_data():
    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()
    
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
    asset_data = {}
    for name, ticker in tickers.items():
        df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01", end_date="2026-02-01")
        if len(df) > 100:
            asset_data[name] = df
            print(f"  {name}: {len(df)} days")
    
    macro_score = compute_macro_score(fred_loader, start_date="2015-01-01", end_date="2026-02-01")
    m2 = fred_loader.get_series("M2SL", start_date="2015-01-01", end_date="2026-02-01")
    m2_yoy = m2.pct_change(12)
    m2_yoy_ma6 = m2_yoy.rolling(6).mean()
    m2_acc = (m2_yoy > m2_yoy_ma6).resample("D").ffill().fillna(False)
    
    return asset_data, macro_score, m2_acc


def get_common_range(asset_data):
    """Get common date range across all assets."""
    starts = [df.index[0] for df in asset_data.values()]
    ends = [df.index[-1] for df in asset_data.values()]
    return max(starts), min(ends)


def bh_strategy(close: pd.Series) -> dict:
    """Buy & hold metrics."""
    daily_ret = close.pct_change().fillna(0)
    equity = (1 + daily_ret).cumprod()
    metrics = _compute_metrics(daily_ret, equity, close.index)
    metrics["risk"] = _risk_metrics(daily_ret)
    return metrics, daily_ret, equity


def bh_portfolio_strategy(asset_data: dict, common_start, common_end) -> dict:
    """Equal-weight buy & hold portfolio."""
    assets = list(asset_data.keys())
    w = 1.0 / len(assets)
    
    portfolio_ret = None
    asset_rets_dict = {}
    for a in assets:
        close = asset_data[a]["close"]
        close = close[(close.index >= common_start) & (close.index <= common_end)]
        r = close.pct_change().fillna(0) * w
        asset_rets_dict[a] = r
        if portfolio_ret is None:
            portfolio_ret = r.copy()
        else:
            common = portfolio_ret.index.intersection(r.index)
            portfolio_ret = portfolio_ret.reindex(common).fillna(0) + r.reindex(common).fillna(0)
    
    equity = (1 + portfolio_ret).cumprod()
    metrics = _compute_metrics(portfolio_ret, equity, portfolio_ret.index)
    metrics["risk"] = _risk_metrics(portfolio_ret)
    return metrics, portfolio_ret, equity


def v2_btc_only(btc_df, macro_score, m2_acc):
    """V2 strategy on BTC only with optimized params."""
    cfg = {k: v for k, v in ASSET_CONFIGS["BTC"].items() if k != "ticker"}
    res = generate_signals(btc_df, macro_score=macro_score, m2_accelerating=m2_acc, **cfg)
    
    daily_ret = btc_df["close"].pct_change().fillna(0)
    pos = res["position_size"].fillna(0)
    pos_change = pos.diff().fillna(0).abs()
    strat_ret = daily_ret * pos - pos_change * TX_COST
    equity = (1 + strat_ret).cumprod()
    
    metrics = _compute_metrics(strat_ret, equity, btc_df.index)
    metrics["risk"] = _risk_metrics(strat_ret)
    trades = int((res["entry_type"].str.startswith("entry_")).sum() + (res["entry_type"] == "trailing_stop").sum())
    metrics["total_trades"] = trades
    return metrics, strat_ret, equity


def extract_serializable(result):
    """Extract JSON-serializable metrics from portfolio backtest result."""
    exclude = {"equity_curve", "daily_returns", "asset_returns"}
    out = {}
    for k, v in result.items():
        if k in exclude:
            continue
        if isinstance(v, (dict, list, str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = float(v)
    return out


def print_comparison(strategies: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON — MacroMomentum Portfolio Final Backtest")
    print("=" * 120)
    
    header = f"{'Strategy':<35} {'Return%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>7} {'WinRate':>8} {'Trades':>7}"
    print(header)
    print("-" * 120)
    
    for name, data in strategies.items():
        m = data.get("metrics", data)
        trades = m.get("total_trades", m.get("trades", "-"))
        print(f"{name:<35} {m.get('total_return', 0):>8.1f} {m.get('cagr', 0):>7.1f} "
              f"{m.get('sharpe', 0):>7.3f} {m.get('sortino', 0):>8.3f} "
              f"{m.get('max_dd', 0):>8.1f} {m.get('calmar', 0):>7.3f} "
              f"{m.get('win_rate_monthly', 0):>7.1f}% {str(trades):>7}")
    
    print("=" * 120)


def print_crash_analysis(strategies: dict):
    """Print crash period analysis."""
    print("\n" + "=" * 100)
    print("CRASH ANALYSIS")
    print("=" * 100)
    
    periods = ["2018_bear", "covid_mar2020", "may2021_crash", "2022_bear"]
    period_labels = {"2018_bear": "2018 Bear", "covid_mar2020": "COVID Mar 2020",
                     "may2021_crash": "May 2021 Crash", "2022_bear": "2022 Bear"}
    
    for period in periods:
        print(f"\n--- {period_labels.get(period, period)} ---")
        header = f"{'Strategy':<35} {'Return%':>10} {'MaxDD%':>10}"
        print(header)
        
        for name, data in strategies.items():
            crash = data.get("crash_analysis", {}).get(period, {})
            if crash:
                print(f"{name:<35} {crash.get('portfolio_return', 0):>10.1f} {crash.get('portfolio_max_dd', 0):>10.1f}")


def print_risk_analysis(strategies: dict):
    """Print risk metrics."""
    print("\n" + "=" * 100)
    print("RISK ANALYSIS")
    print("=" * 100)
    
    header = f"{'Strategy':<35} {'VaR95%':>8} {'VaR99%':>8} {'CVaR95%':>9} {'CVaR99%':>9} {'LongestDD':>10}"
    print(header)
    print("-" * 100)
    
    for name, data in strategies.items():
        m = data.get("metrics", data)
        risk = m.get("risk", data.get("risk", {}))
        longest = m.get("longest_dd_days", data.get("longest_dd_days", 0))
        print(f"{name:<35} {risk.get('var_95_daily', 0):>8.3f} {risk.get('var_99_daily', 0):>8.3f} "
              f"{risk.get('cvar_95_daily', 0):>9.3f} {risk.get('cvar_99_daily', 0):>9.3f} "
              f"{longest:>9}d")


def main():
    print("Loading data...")
    asset_data, macro_score, m2_acc = load_data()
    
    common_start, common_end = get_common_range(asset_data)
    print(f"\nCommon range: {common_start.date()} to {common_end.date()}")
    
    strategies = {}
    
    # S1: Buy & Hold BTC
    print("\nRunning S1: Buy & Hold BTC...")
    btc_common = asset_data["BTC"][(asset_data["BTC"].index >= common_start) & (asset_data["BTC"].index <= common_end)]
    m1, r1, e1 = bh_strategy(btc_common["close"])
    m1["total_trades"] = 1
    crash1 = _crash_analysis(e1, r1, {"BTC": r1}, r1.index)
    strategies["S1: B&H BTC"] = {"metrics": m1, "crash_analysis": crash1}
    
    # S2: Buy & Hold equal-weight
    print("Running S2: Buy & Hold Equal-Weight...")
    m2, r2, e2 = bh_portfolio_strategy(asset_data, common_start, common_end)
    m2["total_trades"] = len(asset_data)
    crash2 = _crash_analysis(e2, r2, {a: asset_data[a]["close"].reindex(r2.index).pct_change().fillna(0) / len(asset_data) for a in asset_data}, r2.index)
    strategies["S2: B&H Equal-Weight"] = {"metrics": m2, "crash_analysis": crash2}
    
    # S3: V2 BTC Only
    print("Running S3: V2 BTC Only...")
    m3, r3, e3 = v2_btc_only(btc_common, macro_score, m2_acc)
    crash3 = _crash_analysis(e3, r3, {"BTC": r3}, r3.index)
    strategies["S3: V2 BTC Only"] = {"metrics": m3, "crash_analysis": crash3}
    
    # S4: Portfolio Equal-Weight
    print("Running S4: Portfolio Equal-Weight...")
    res4 = run_portfolio_backtest(
        asset_data, macro_score, m2_acc, mode="equal",
        max_portfolio_leverage=1.0, rebalance_freq="monthly",
        start_date=str(common_start.date()), end_date=str(common_end.date()),
    )
    strategies["S4: Portfolio Equal"] = res4
    
    # S5: Portfolio Risk-Parity
    print("Running S5: Portfolio Risk-Parity...")
    res5 = run_portfolio_backtest(
        asset_data, macro_score, m2_acc, mode="risk_parity",
        max_portfolio_leverage=1.0, rebalance_freq="monthly",
        start_date=str(common_start.date()), end_date=str(common_end.date()),
    )
    strategies["S5: Portfolio Risk-Parity"] = res5
    
    # S6: Portfolio Momentum-Weighted
    print("Running S6: Portfolio Momentum-Weighted...")
    res6 = run_portfolio_backtest(
        asset_data, macro_score, m2_acc, mode="momentum_weighted",
        max_portfolio_leverage=1.0, rebalance_freq="monthly",
        start_date=str(common_start.date()), end_date=str(common_end.date()),
    )
    strategies["S6: Portfolio Momentum"] = res6
    
    # S7: Portfolio Optuna-Optimized
    print("Running S7: Portfolio Optuna-Optimized...")
    if STUDY_PATH.exists():
        with open(STUDY_PATH, "rb") as f:
            study = pickle.load(f)
        bp = study.best_params
        
        w_total = bp.get("weight_btc", 0.25) + bp.get("weight_eth", 0.25) + bp.get("weight_sol", 0.25) + bp.get("weight_link", 0.25)
        opt_weights = {
            "BTC": bp.get("weight_btc", 0.25) / w_total,
            "ETH": bp.get("weight_eth", 0.25) / w_total,
            "SOL": bp.get("weight_sol", 0.25) / w_total,
            "LINK": bp.get("weight_link", 0.25) / w_total,
        }
        
        res7 = run_portfolio_backtest(
            asset_data, macro_score, m2_acc,
            mode=bp.get("allocation_mode", "equal"),
            max_portfolio_leverage=bp.get("max_portfolio_leverage", 1.0),
            vol_target=bp.get("vol_target", 0.25),
            rebalance_freq=bp.get("rebalance_frequency", "monthly"),
            correlation_filter=bp.get("correlation_filter", False),
            start_date=str(common_start.date()), end_date=str(common_end.date()),
        )
        strategies["S7: Portfolio Optuna"] = res7
    else:
        print("  WARNING: No Optuna study found. Run optimize_portfolio.py first.")
    
    # Print results
    # Normalize structure for printing
    print_data = {}
    for name, data in strategies.items():
        if "metrics" in data:
            print_data[name] = data["metrics"]
            print_data[name]["crash_analysis"] = data.get("crash_analysis", {})
        else:
            print_data[name] = data
    
    print_comparison(print_data)
    
    # Crash analysis
    crash_data = {}
    for name, data in strategies.items():
        if "crash_analysis" in data:
            crash_data[name] = data
        elif "metrics" in data and "crash_analysis" in data:
            crash_data[name] = data
    print_crash_analysis(strategies)
    
    # Risk analysis
    print_risk_analysis(print_data)
    
    # Per-asset contribution for portfolio strategies
    print("\n" + "=" * 100)
    print("PER-ASSET CONTRIBUTION TO P&L")
    print("=" * 100)
    for name, data in strategies.items():
        if "per_asset" in data:
            print(f"\n{name}:")
            for a, am in data["per_asset"].items():
                print(f"  {a}: Sharpe={am.get('sharpe', 0):.3f}, Return={am.get('total_return', 0):.1f}%, "
                      f"Contribution={am.get('contribution', 0)*100:.1f}%, Trades={am.get('trades', 0)}")
    
    # Save results (JSON serializable)
    save_data = {}
    for name, data in strategies.items():
        if "metrics" in data:
            save_data[name] = {"metrics": data["metrics"], "crash_analysis": data.get("crash_analysis", {})}
        else:
            save_data[name] = extract_serializable(data)
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
