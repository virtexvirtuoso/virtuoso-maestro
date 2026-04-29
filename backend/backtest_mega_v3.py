"""
Backtest MegaStrategyV3 — 10-strategy comparison with full reporting.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v3 import (
    run_full_strategy, ASSET_CONFIGS, DEFAULT_LEVERAGE_MAP,
    TX_COST, SHORT_BORROW_COST_DAILY, FUNDING_CARRY_DAILY,
)

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}


def load_data():
    """Load all data needed for backtesting."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    stock_loader = StockDataLoader()
    fred_loader = MacroDataLoader()

    # Crypto
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Cross-asset
    cross_asset_data = pd.DataFrame()
    for col_name, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock_loader.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col_name] = df["close"]
            print(f"  {col_name} ({ticker}): {len(df)} days")
        except Exception as e:
            print(f"  {col_name}: FAILED - {e}")

    # Macro
    macro_data = fred_loader.get_multiple(FRED_SERIES, start_date="2015-01-01")
    print(f"  Macro data: {len(macro_data)} days, columns: {list(macro_data.columns)}")

    return crypto_data, cross_asset_data, macro_data


def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Compute performance metrics from daily returns."""
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["total_return", "cagr", "sharpe", "sortino", "max_dd", "calmar", "win_rate"]}

    equity = (1 + returns).cumprod()
    n_years = len(returns) / 252
    total_ret = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / max(n_years, 0.1)) - 1)
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    monthly = returns.resample("ME").sum()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0

    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
    }


def run_buy_hold(crypto_data, assets=None):
    """Buy & Hold strategy returns."""
    if assets is None:
        assets = list(crypto_data.keys())
    # Equal weight
    common_idx = None
    for a in assets:
        if a in crypto_data:
            idx = crypto_data[a].index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or len(common_idx) < 10:
        return pd.Series(dtype=float)

    portfolio_ret = pd.Series(0.0, index=common_idx)
    w = 1.0 / len(assets)
    for a in assets:
        if a in crypto_data:
            ret = crypto_data[a]["close"].reindex(common_idx).pct_change().fillna(0)
            portfolio_ret += ret * w
    return portfolio_ret


def run_v3_variant(crypto_data, macro_data, cross_asset_data,
                   enable_long=True, enable_short=True,
                   enable_adaptive_leverage=True,
                   leverage_map=None, vol_ceiling=None,
                   label="V3"):
    """Run a V3 strategy variant."""
    override = {}
    if vol_ceiling is not None:
        override["vol_ceiling"] = vol_ceiling

    portfolio_df, per_asset = run_full_strategy(
        crypto_data, macro_data, cross_asset_data,
        leverage_map=leverage_map,
        enable_long=enable_long,
        enable_short=enable_short,
        enable_adaptive_leverage=enable_adaptive_leverage,
        **override,
    )
    return portfolio_df, per_asset


def crash_analysis(portfolio_df: pd.DataFrame) -> dict:
    """Analyze P&L during crash periods."""
    periods = {
        "2018_bear": ("2018-01-01", "2018-12-31"),
        "covid_mar2020": ("2020-02-15", "2020-04-15"),
        "may2021_crash": ("2021-04-15", "2021-07-20"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
    }
    results = {}
    for name, (start, end) in periods.items():
        mask = (portfolio_df.index >= pd.Timestamp(start)) & (portfolio_df.index <= pd.Timestamp(end))
        if mask.sum() < 5:
            continue
        pnl = portfolio_df.loc[mask, "daily_pnl"]
        eq = (1 + pnl).cumprod()
        total = float(eq.iloc[-1] - 1) * 100
        dd = float((eq / eq.cummax() - 1).min()) * 100

        long_pnl = portfolio_df.loc[mask, "long_pnl"].sum() * 100 if "long_pnl" in portfolio_df else 0
        short_pnl = portfolio_df.loc[mask, "short_pnl"].sum() * 100 if "short_pnl" in portfolio_df else 0

        results[name] = {
            "total_return_pct": round(total, 2),
            "max_dd_pct": round(dd, 2),
            "long_contrib_pct": round(float(long_pnl), 2),
            "short_contrib_pct": round(float(short_pnl), 2),
        }
    return results


def regime_breakdown(portfolio_df: pd.DataFrame) -> dict:
    """Per-regime performance for S7."""
    results = {}
    for reg in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"]:
        mask = portfolio_df["regime"] == reg
        if mask.sum() < 5:
            results[reg] = {"days": 0, "pct_time": 0}
            continue
        pnl = portfolio_df.loc[mask, "daily_pnl"]
        ann_ret = pnl.mean() * 252 * 100
        ann_vol = pnl.std() * np.sqrt(252) * 100
        sharpe = (pnl.mean() * 252) / (pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0
        results[reg] = {
            "days": int(mask.sum()),
            "pct_time": round(mask.mean() * 100, 1),
            "ann_return_pct": round(float(ann_ret), 2),
            "ann_vol_pct": round(float(ann_vol), 2),
            "sharpe": round(float(sharpe), 3),
        }
    return results


def monthly_heatmap(portfolio_df: pd.DataFrame) -> dict:
    """Monthly returns heatmap data."""
    monthly = portfolio_df["daily_pnl"].resample("ME").sum() * 100
    heatmap = {}
    for dt, val in monthly.items():
        yr = str(dt.year)
        mo = str(dt.month)
        if yr not in heatmap:
            heatmap[yr] = {}
        heatmap[yr][mo] = round(float(val), 2)
    return heatmap


def print_comparison_table(results: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 110)
    print("STRATEGY COMPARISON — MegaStrategyV3 Backtest")
    print("=" * 110)
    header = f"{'Strategy':<40} {'Return%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>7} {'WinR%':>6}"
    print(header)
    print("-" * 110)

    for name, metrics in results.items():
        row = (f"{name:<40} {metrics['total_return']:>8.1f} {metrics['cagr']:>7.1f} "
               f"{metrics['sharpe']:>7.3f} {metrics['sortino']:>8.3f} "
               f"{metrics['max_dd']:>8.1f} {metrics['calmar']:>7.3f} {metrics['win_rate']:>6.1f}")
        print(row)
    print("=" * 110)


def print_regime_table(regime_data: dict):
    """Print regime breakdown."""
    print("\n" + "=" * 90)
    print("REGIME BREAKDOWN — S7 (V3 Full)")
    print("=" * 90)
    header = f"{'Regime':<18} {'Days':>6} {'% Time':>7} {'Ann Ret%':>9} {'Ann Vol%':>9} {'Sharpe':>7}"
    print(header)
    print("-" * 90)
    for reg, data in regime_data.items():
        if data["days"] == 0:
            continue
        print(f"{reg:<18} {data['days']:>6} {data['pct_time']:>7.1f} "
              f"{data.get('ann_return_pct', 0):>9.1f} {data.get('ann_vol_pct', 0):>9.1f} "
              f"{data.get('sharpe', 0):>7.3f}")
    print("=" * 90)


def print_crash_table(crash_data: dict):
    """Print crash analysis."""
    print("\n" + "=" * 80)
    print("CRASH ANALYSIS — S7 (V3 Full)")
    print("=" * 80)
    header = f"{'Period':<20} {'Return%':>9} {'MaxDD%':>8} {'Long%':>8} {'Short%':>8}"
    print(header)
    print("-" * 80)
    for period, data in crash_data.items():
        print(f"{period:<20} {data['total_return_pct']:>9.1f} {data['max_dd_pct']:>8.1f} "
              f"{data['long_contrib_pct']:>8.1f} {data['short_contrib_pct']:>8.1f}")
    print("=" * 80)


def print_monthly_heatmap(heatmap: dict):
    """Print monthly returns heatmap."""
    print("\n" + "=" * 100)
    print("MONTHLY RETURNS HEATMAP — S7 (V3 Full)")
    print("=" * 100)
    months = [str(i) for i in range(1, 13)]
    header = f"{'Year':<6}" + "".join(f"{'M'+m:>7}" for m in months) + f"{'Total':>8}"
    print(header)
    print("-" * 100)
    for yr in sorted(heatmap.keys()):
        row = f"{yr:<6}"
        total = 0
        for m in months:
            val = heatmap[yr].get(m, 0)
            total += val
            if val > 0:
                row += f"{val:>7.1f}"
            elif val < 0:
                row += f"{val:>7.1f}"
            else:
                row += f"{'--':>7}"
        row += f"{total:>8.1f}"
        print(row)
    print("=" * 100)


def main():
    print("\n" + "#" * 70)
    print("#  MegaStrategyV3 — Comprehensive Backtest")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    crypto_data, cross_asset_data, macro_data = load_data()

    if not crypto_data:
        print("ERROR: No crypto data loaded!")
        return

    comparison = {}

    # S1: Buy & Hold BTC
    print("\n[S1] Buy & Hold BTC...")
    bh_btc = run_buy_hold(crypto_data, ["BTC"])
    comparison["S1: Buy & Hold BTC"] = compute_metrics(bh_btc)

    # S2: Buy & Hold equal-weight
    print("[S2] Buy & Hold equal-weight portfolio...")
    bh_eq = run_buy_hold(crypto_data)
    comparison["S2: Buy & Hold EqWeight"] = compute_metrics(bh_eq)

    # S3: V2 Portfolio (long-only, no adaptive leverage)
    print("[S3] V2 Portfolio (long-only baseline)...")
    try:
        v2_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                   enable_long=True, enable_short=False,
                                   enable_adaptive_leverage=False,
                                   leverage_map={i: 1.0 for i in range(6)},
                                   label="V2")
        comparison["S3: V2 Portfolio (long-only)"] = compute_metrics(v2_df["daily_pnl"])
    except Exception as e:
        print(f"  S3 failed: {e}")
        comparison["S3: V2 Portfolio (long-only)"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S4: V3 Long-only (adaptive leverage, no shorts)
    print("[S4] V3 Long-only (adaptive leverage)...")
    try:
        s4_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                   enable_long=True, enable_short=False,
                                   enable_adaptive_leverage=True, label="S4")
        comparison["S4: V3 Long-only + AdaptLev"] = compute_metrics(s4_df["daily_pnl"])
    except Exception as e:
        print(f"  S4 failed: {e}")
        comparison["S4: V3 Long-only + AdaptLev"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S5: V3 Short-only
    print("[S5] V3 Short-only...")
    try:
        s5_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                   enable_long=False, enable_short=True,
                                   enable_adaptive_leverage=False, label="S5")
        comparison["S5: V3 Short-only"] = compute_metrics(s5_df["daily_pnl"])
    except Exception as e:
        print(f"  S5 failed: {e}")
        comparison["S5: V3 Short-only"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S6: V3 Long+Short (flat 1x)
    print("[S6] V3 Long+Short (flat 1x)...")
    try:
        s6_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                   enable_long=True, enable_short=True,
                                   enable_adaptive_leverage=False,
                                   leverage_map={i: 1.0 for i in range(6)}, label="S6")
        comparison["S6: V3 L+S flat 1x"] = compute_metrics(s6_df["daily_pnl"])
    except Exception as e:
        print(f"  S6 failed: {e}")
        comparison["S6: V3 L+S flat 1x"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S7: V3 Full
    print("[S7] V3 Full (long + short + adaptive leverage)...")
    try:
        s7_df, s7_per_asset = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                              enable_long=True, enable_short=True,
                                              enable_adaptive_leverage=True, label="S7")
        comparison["S7: V3 Full"] = compute_metrics(s7_df["daily_pnl"])
    except Exception as e:
        print(f"  S7 failed: {e}")
        s7_df = pd.DataFrame()
        comparison["S7: V3 Full"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S8: V3 Conservative (max leverage 1.5x)
    print("[S8] V3 Full Conservative (max lev 1.5x)...")
    try:
        lev_conservative = {5: 1.5, 4: 1.2, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
        s8_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                   enable_long=True, enable_short=True,
                                   enable_adaptive_leverage=True,
                                   leverage_map=lev_conservative, label="S8")
        comparison["S8: V3 Conservative"] = compute_metrics(s8_df["daily_pnl"])
    except Exception as e:
        print(f"  S8 failed: {e}")
        comparison["S8: V3 Conservative"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S9: V3 Aggressive (max leverage 2.5x)
    print("[S9] V3 Full Aggressive (max lev 2.5x)...")
    try:
        lev_aggressive = {5: 2.5, 4: 2.0, 3: 1.5, 2: 0.8, 1: 0.5, 0: 0.0}
        s9_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                   enable_long=True, enable_short=True,
                                   enable_adaptive_leverage=True,
                                   leverage_map=lev_aggressive, label="S9")
        comparison["S9: V3 Aggressive"] = compute_metrics(s9_df["daily_pnl"])
    except Exception as e:
        print(f"  S9 failed: {e}")
        comparison["S9: V3 Aggressive"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # S10: V3 Full + Vol-adjusted
    print("[S10] V3 Full + Vol-adjusted (halve when vol > 80%)...")
    try:
        s10_df, _ = run_v3_variant(crypto_data, macro_data, cross_asset_data,
                                    enable_long=True, enable_short=True,
                                    enable_adaptive_leverage=True,
                                    vol_ceiling=0.8, label="S10")
        comparison["S10: V3 Vol-Adjusted"] = compute_metrics(s10_df["daily_pnl"])
    except Exception as e:
        print(f"  S10 failed: {e}")
        comparison["S10: V3 Vol-Adjusted"] = {k: 0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate"]}

    # Print comparison table
    print_comparison_table(comparison)

    # Detailed analysis for S7
    if len(s7_df) > 0:
        # Regime breakdown
        reg_data = regime_breakdown(s7_df)
        print_regime_table(reg_data)

        # Long vs Short P&L attribution
        print("\n" + "=" * 60)
        print("LONG vs SHORT P&L ATTRIBUTION — S7")
        print("=" * 60)
        total_long = s7_df["long_pnl"].sum() * 100
        total_short = s7_df["short_pnl"].sum() * 100
        total_funding = s7_df["funding_pnl"].sum() * 100
        print(f"  Long P&L:    {total_long:>+8.1f}%")
        print(f"  Short P&L:   {total_short:>+8.1f}%")
        print(f"  Funding:     {total_funding:>+8.1f}%")
        print(f"  Total:       {(total_long + total_short + total_funding):>+8.1f}%")

        # Leverage distribution
        print("\n" + "=" * 60)
        print("LEVERAGE DISTRIBUTION — S7")
        print("=" * 60)
        lev = s7_df["total_leverage"]
        print(f"  Mean leverage:   {lev.mean():.3f}")
        print(f"  Median leverage: {lev.median():.3f}")
        print(f"  Max leverage:    {lev.max():.3f}")
        print(f"  % time lev>1.5:  {(lev > 1.5).mean() * 100:.1f}%")
        print(f"  % time lev<0.5:  {(lev < 0.5).mean() * 100:.1f}%")
        print(f"  % time lev=0:    {(lev == 0).mean() * 100:.1f}%")

        # Crash analysis
        crash_data = crash_analysis(s7_df)
        print_crash_table(crash_data)

        # Monthly heatmap
        heatmap = monthly_heatmap(s7_df)
        print_monthly_heatmap(heatmap)

        # Rolling 1-year Sharpe
        print("\n" + "=" * 60)
        print("ROLLING 1-YEAR SHARPE — S7")
        print("=" * 60)
        rolling_sharpe = (s7_df["daily_pnl"].rolling(252).mean() /
                          s7_df["daily_pnl"].rolling(252).std()) * np.sqrt(252)
        yearly = rolling_sharpe.resample("YE").last().dropna()
        for dt, val in yearly.items():
            print(f"  {dt.year}: {val:.3f}")

        # Save detailed results
        reg_data_safe = reg_data
    else:
        reg_data_safe = {}
        crash_data = {}
        heatmap = {}

    # Save results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "comparison": comparison,
        "regime_breakdown": reg_data_safe if len(s7_df) > 0 else {},
        "crash_analysis": crash_data,
        "monthly_heatmap": heatmap,
    }

    out_path = RESULTS_DIR / "mega_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "#" * 70)
    print("#  BACKTEST COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
