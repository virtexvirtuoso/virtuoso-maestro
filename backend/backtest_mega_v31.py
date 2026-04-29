"""
Rigorous Backtest + 14-Fold Walk-Forward for MegaStrategyV3.1

Tests:
- Full 10-strategy comparison (matching V3 backtest format)
- 14-fold walk-forward with statistical significance testing
- Regime breakdown, crash analysis, monthly heatmap
- Rolling Sharpe, leverage distribution
- Bootstrap confidence intervals
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from strategies.composite.mega_strategy_v31 import (
    run_full_strategy, LEVERAGE_MAP, WEIGHTS,
)
from strategies.composite.mega_strategy_v3 import (
    run_full_strategy as run_v3_full,
)

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/backtest_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}
CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}


def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    stock = StockDataLoader()
    fred = MacroDataLoader()
    crypto_data = {}
    for name, ticker in CRYPTO_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            if len(df) > 100:
                crypto_data[name] = df
                print(f"  {name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    cross_asset_data = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset_data[col] = df["close"]
            print(f"  {col} ({ticker}): {len(df)} days")
        except Exception as e:
            print(f"  {col}: FAILED - {e}")
    macro_data = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    print(f"  Macro: {len(macro_data)} days, columns: {list(macro_data.columns)}")
    return crypto_data, cross_asset_data, macro_data


def compute_metrics(returns, name=""):
    if len(returns) < 10 or returns.std() == 0:
        return {k: 0.0 for k in ["total_return","cagr","sharpe","sortino","max_dd","calmar","win_rate",
                                   "omega","profit_factor","tail_ratio","ulcer_index","upi"]}
    eq = (1 + returns).cumprod()
    n_yr = len(returns) / 252
    total_ret = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1)
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    monthly = returns.resample("ME").sum()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0

    # Extended metrics
    threshold = 0
    excess = returns - threshold/252
    omega = float(excess[excess > 0].sum() / abs(excess[excess < 0].sum())) if excess[excess < 0].sum() != 0 else 0
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0
    sorted_ret = returns.sort_values()
    n5 = max(1, int(len(sorted_ret) * 0.05))
    tail_ratio = float(abs(sorted_ret.iloc[-n5:].mean() / sorted_ret.iloc[:n5].mean())) if sorted_ret.iloc[:n5].mean() != 0 else 0
    dd_sq = dd ** 2
    ulcer = float(np.sqrt(dd_sq.mean())) * 100
    upi = float(ann_ret / (np.sqrt(dd_sq.mean()))) if dd_sq.mean() > 0 else 0

    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
        "omega": round(omega, 3),
        "profit_factor": round(profit_factor, 3),
        "tail_ratio": round(tail_ratio, 3),
        "ulcer_index": round(ulcer, 3),
        "upi": round(upi, 3),
    }


def walk_forward_14fold(portfolio_pnl, label=""):
    """14-fold expanding walk-forward with statistical testing."""
    ret = portfolio_pnl.dropna()
    n = len(ret)
    
    min_train = 365  # 1 year minimum training
    test_size = 126  # ~6 months test
    
    folds = []
    fold_sharpes = []
    fold_returns = []
    
    for fold in range(14):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size
        if test_end > n:
            break
        
        test_ret = ret.iloc[train_end:test_end]
        if len(test_ret) < 50 or test_ret.std() == 0:
            continue
        
        fold_sharpe = float(test_ret.mean() / test_ret.std() * np.sqrt(252))
        fold_total = float((1 + test_ret).cumprod().iloc[-1] - 1) * 100
        
        start_date = ret.index[train_end].strftime("%Y-%m-%d")
        end_date = ret.index[min(test_end - 1, n - 1)].strftime("%Y-%m-%d")
        
        folds.append({
            "fold": fold + 1,
            "test_start": start_date,
            "test_end": end_date,
            "sharpe": round(fold_sharpe, 3),
            "return_pct": round(fold_total, 1),
            "days": len(test_ret),
        })
        fold_sharpes.append(fold_sharpe)
        fold_returns.append(fold_total)
    
    if len(fold_sharpes) < 3:
        return {"folds": folds, "n_folds": len(folds), "mean_oos_sharpe": 0,
                "p_value": 1, "significant": False}
    
    # Statistical tests
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes, ddof=1)
    t_stat = mean_sharpe / (std_sharpe / np.sqrt(len(fold_sharpes))) if std_sharpe > 0 else 0
    p_value = 1 - stats.t.cdf(t_stat, df=len(fold_sharpes) - 1)
    
    # Count positive folds
    positive_folds = sum(1 for s in fold_sharpes if s > 0)
    
    # Bootstrap 95% CI
    n_boot = 5000
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(fold_sharpes, size=len(fold_sharpes), replace=True)
        boot_means.append(np.mean(sample))
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    return {
        "folds": folds,
        "n_folds": len(folds),
        "n_active": positive_folds,
        "mean_oos_sharpe": round(mean_sharpe, 3),
        "median_oos_sharpe": round(np.median(fold_sharpes), 3),
        "std_oos_sharpe": round(std_sharpe, 3),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "significant_10pct": p_value < 0.10,
        "significant_5pct": p_value < 0.05,
        "ci_95_lower": round(ci_lower, 3),
        "ci_95_upper": round(ci_upper, 3),
        "positive_folds": f"{positive_folds}/{len(fold_sharpes)}",
        "mean_fold_return": round(np.mean(fold_returns), 1),
    }


def crash_analysis(portfolio_df):
    periods = {
        "2018_bear": ("2018-01-01", "2018-12-31"),
        "covid_mar2020": ("2020-02-15", "2020-04-15"),
        "may2021_crash": ("2021-04-15", "2021-07-20"),
        "luna_may2022": ("2022-05-01", "2022-07-01"),
        "ftx_nov2022": ("2022-10-15", "2022-12-31"),
        "2022_full_bear": ("2022-01-01", "2022-12-31"),
        "2023_recovery": ("2023-01-01", "2023-12-31"),
        "2024_bull": ("2024-01-01", "2024-12-31"),
    }
    results = {}
    for name, (start, end) in periods.items():
        mask = (portfolio_df.index >= pd.Timestamp(start)) & (portfolio_df.index <= pd.Timestamp(end))
        if mask.sum() < 5: continue
        pnl = portfolio_df.loc[mask, "daily_pnl"]
        eq = (1 + pnl).cumprod()
        total = float(eq.iloc[-1] - 1) * 100
        dd = float((eq / eq.cummax() - 1).min()) * 100
        mean_lev = float(portfolio_df.loc[mask, "total_leverage"].mean())
        results[name] = {
            "return_pct": round(total, 1),
            "max_dd_pct": round(dd, 1),
            "mean_leverage": round(mean_lev, 3),
        }
    return results


def regime_breakdown(portfolio_df):
    results = {}
    for reg in ["BULL", "MILD_BULL", "NEUTRAL", "BEAR", "ACCUMULATION"]:
        mask = portfolio_df["regime"] == reg
        if mask.sum() < 5:
            results[reg] = {"days": 0}
            continue
        pnl = portfolio_df.loc[mask, "daily_pnl"]
        ann_ret = pnl.mean() * 252 * 100
        ann_vol = pnl.std() * np.sqrt(252) * 100
        sharpe = (pnl.mean() * 252) / (pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0
        mean_lev = float(portfolio_df.loc[mask, "total_leverage"].mean())
        results[reg] = {
            "days": int(mask.sum()),
            "pct_time": round(mask.mean() * 100, 1),
            "ann_return_pct": round(float(ann_ret), 1),
            "ann_vol_pct": round(float(ann_vol), 1),
            "sharpe": round(float(sharpe), 3),
            "mean_leverage": round(mean_lev, 3),
        }
    return results


def main():
    print("\n" + "#" * 70)
    print("#  MegaStrategyV3.1 — Rigorous Backtest + Walk-Forward")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    crypto_data, cross_asset_data, macro_data = load_data()
    if not crypto_data:
        print("ERROR: No crypto data!")
        return

    comparison = {}

    # ── V3 Baselines ──
    print("\n[V3-S4] V3 Long+Adaptive (baseline)...")
    v3_s4, _ = run_v3_full(crypto_data, macro_data, cross_asset_data,
                            enable_long=True, enable_short=False, enable_adaptive_leverage=True)
    comparison["V3-S4: Long+Adaptive"] = compute_metrics(v3_s4["daily_pnl"])

    print("[V3-S7] V3 Full (baseline)...")
    v3_s7, _ = run_v3_full(crypto_data, macro_data, cross_asset_data,
                            enable_long=True, enable_short=True, enable_adaptive_leverage=True)
    comparison["V3-S7: Full"] = compute_metrics(v3_s7["daily_pnl"])

    # ── V3.1 Variants ──
    print("\n[V3.1-H2] Production candidate (vol+bear+trail10)...")
    v31_h2, v31_per_asset = run_full_strategy(crypto_data, macro_data, cross_asset_data)
    comparison["V3.1-H2: Production"] = compute_metrics(v31_h2["daily_pnl"])

    print("[V3.1-NoProtect] Direct sizing, no protections...")
    v31_np, _ = run_full_strategy(crypto_data, macro_data, cross_asset_data,
                                   vol_ceiling=0, bear_filter_days=9999,
                                   portfolio_trail_stop=0)
    comparison["V3.1-NoProt: Raw"] = compute_metrics(v31_np["daily_pnl"])

    print("[V3.1-Conservative] Lower leverage...")
    lev_con = {5: 1.5, 4: 1.0, 3: 1.0, 2: 0.6, 1: 0.2, 0: 0.0}
    v31_con, _ = run_full_strategy(crypto_data, macro_data, cross_asset_data,
                                    leverage_map=lev_con)
    comparison["V3.1-Conserv"] = compute_metrics(v31_con["daily_pnl"])

    print("[V3.1-Aggressive] Higher leverage...")
    lev_agg = {5: 2.5, 4: 1.5, 3: 1.5, 2: 1.0, 1: 0.5, 0: 0.0}
    v31_agg, _ = run_full_strategy(crypto_data, macro_data, cross_asset_data,
                                    leverage_map=lev_agg)
    comparison["V3.1-Aggress"] = compute_metrics(v31_agg["daily_pnl"])

    # Buy & Hold
    print("[B&H] Buy & Hold equal-weight...")
    bh_idx = None
    for a in crypto_data:
        idx = crypto_data[a].index
        bh_idx = idx if bh_idx is None else bh_idx.intersection(idx)
    bh_ret = pd.Series(0.0, index=bh_idx)
    for a in crypto_data:
        bh_ret += crypto_data[a]["close"].reindex(bh_idx).pct_change().fillna(0) / len(crypto_data)
    comparison["Buy & Hold EqWeight"] = compute_metrics(bh_ret)

    # ── Print Comparison ──
    print("\n" + "=" * 130)
    print("STRATEGY COMPARISON — V3 vs V3.1")
    print("=" * 130)
    header = (f"{'Strategy':<30} {'Ret%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MaxDD%':>8} {'Calmar':>7} {'Omega':>6} {'PF':>6} {'UPI':>6} {'WinR%':>6}")
    print(header)
    print("-" * 130)
    for name, m in comparison.items():
        print(f"{name:<30} {m['total_return']:>8.1f} {m['cagr']:>7.1f} {m['sharpe']:>7.3f} "
              f"{m['sortino']:>8.3f} {m['max_dd']:>8.1f} {m['calmar']:>7.3f} "
              f"{m['omega']:>6.2f} {m['profit_factor']:>6.2f} {m['upi']:>6.2f} {m['win_rate']:>6.1f}")

    # ── Regime Breakdown ──
    reg_data = regime_breakdown(v31_h2)
    print("\n" + "=" * 90)
    print("REGIME BREAKDOWN — V3.1-H2")
    print("=" * 90)
    print(f"{'Regime':<18} {'Days':>6} {'%Time':>7} {'AnnRet%':>9} {'AnnVol%':>9} {'Sharpe':>7} {'MeanLev':>8}")
    print("-" * 65)
    for reg, d in reg_data.items():
        if d["days"] == 0: continue
        print(f"{reg:<18} {d['days']:>6} {d['pct_time']:>7.1f} {d.get('ann_return_pct',0):>9.1f} "
              f"{d.get('ann_vol_pct',0):>9.1f} {d.get('sharpe',0):>7.3f} {d.get('mean_leverage',0):>8.3f}")

    # ── Crash Analysis ──
    crash_data = crash_analysis(v31_h2)
    print("\n" + "=" * 70)
    print("CRASH/PERIOD ANALYSIS — V3.1-H2")
    print("=" * 70)
    print(f"{'Period':<22} {'Return%':>9} {'MaxDD%':>8} {'MeanLev':>8}")
    print("-" * 50)
    for period, d in crash_data.items():
        print(f"{period:<22} {d['return_pct']:>+9.1f} {d['max_dd_pct']:>8.1f} {d['mean_leverage']:>8.3f}")

    # ── Leverage Distribution ──
    print("\n" + "=" * 60)
    print("LEVERAGE DISTRIBUTION — V3.1-H2")
    print("=" * 60)
    lev = v31_h2["total_leverage"]
    print(f"  Mean:     {lev.mean():.3f}")
    print(f"  Median:   {lev.median():.3f}")
    print(f"  Max:      {lev.max():.3f}")
    print(f"  % > 1.0:  {(lev > 1.0).mean()*100:.1f}%")
    print(f"  % > 0.5:  {(lev > 0.5).mean()*100:.1f}%")
    print(f"  % flat:   {(lev < 0.01).mean()*100:.1f}%")

    # ── Monthly Heatmap ──
    print("\n" + "=" * 100)
    print("MONTHLY RETURNS HEATMAP — V3.1-H2")
    print("=" * 100)
    monthly = v31_h2["daily_pnl"].resample("ME").sum() * 100
    heatmap = {}
    for dt, val in monthly.items():
        yr = str(dt.year)
        mo = str(dt.month)
        if yr not in heatmap: heatmap[yr] = {}
        heatmap[yr][mo] = round(float(val), 1)

    months = [str(i) for i in range(1, 13)]
    print(f"{'Year':<6}" + "".join(f"{'M'+m:>7}" for m in months) + f"{'Total':>8}")
    print("-" * 100)
    for yr in sorted(heatmap.keys()):
        row = f"{yr:<6}"
        total = 0
        for m in months:
            val = heatmap[yr].get(m, 0)
            total += val
            row += f"{val:>7.1f}" if val != 0 else f"{'--':>7}"
        row += f"{total:>8.1f}"
        print(row)

    # ── Rolling Sharpe ──
    print("\n" + "=" * 60)
    print("ROLLING 1-YEAR SHARPE — V3.1-H2 vs V3-S4")
    print("=" * 60)
    v31_rolling = (v31_h2["daily_pnl"].rolling(252).mean() / v31_h2["daily_pnl"].rolling(252).std()) * np.sqrt(252)
    v3_rolling = (v3_s4["daily_pnl"].rolling(252).mean() / v3_s4["daily_pnl"].rolling(252).std()) * np.sqrt(252)
    v31_yearly = v31_rolling.resample("YE").last().dropna()
    v3_yearly = v3_rolling.resample("YE").last().dropna()
    print(f"{'Year':<6} {'V3.1-H2':>10} {'V3-S4':>10} {'Delta':>10}")
    print("-" * 40)
    for dt in v31_yearly.index:
        v31_val = v31_yearly.loc[dt]
        v3_val = v3_yearly.get(dt, 0) if dt in v3_yearly.index else 0
        print(f"{dt.year:<6} {v31_val:>10.3f} {v3_val:>10.3f} {v31_val - v3_val:>+10.3f}")

    # ── 14-FOLD WALK-FORWARD ──
    print("\n" + "=" * 90)
    print("14-FOLD WALK-FORWARD VALIDATION")
    print("=" * 90)

    wf_v31 = walk_forward_14fold(v31_h2["daily_pnl"], "V3.1-H2")
    wf_v3 = walk_forward_14fold(v3_s4["daily_pnl"], "V3-S4")

    for label, wf in [("V3.1-H2", wf_v31), ("V3-S4", wf_v3)]:
        print(f"\n  {label}:")
        print(f"    Folds: {wf['n_folds']}, Active (Sharpe>0): {wf.get('positive_folds', 'N/A')}")
        print(f"    Mean OOS Sharpe:   {wf['mean_oos_sharpe']:.3f}")
        print(f"    Median OOS Sharpe: {wf.get('median_oos_sharpe', 0):.3f}")
        print(f"    Std OOS Sharpe:    {wf.get('std_oos_sharpe', 0):.3f}")
        print(f"    t-statistic:       {wf.get('t_statistic', 0):.3f}")
        print(f"    p-value:           {wf.get('p_value', 1):.4f}")
        print(f"    Significant (5%):  {wf.get('significant_5pct', False)}")
        print(f"    Significant (10%): {wf.get('significant_10pct', False)}")
        print(f"    95% CI:            [{wf.get('ci_95_lower', 0):.3f}, {wf.get('ci_95_upper', 0):.3f}]")
        print(f"    Mean fold return:  {wf.get('mean_fold_return', 0):.1f}%")

        print(f"\n    Per-fold detail:")
        print(f"    {'Fold':>4} {'Test Period':<25} {'Sharpe':>8} {'Return%':>9}")
        print(f"    {'-'*50}")
        for f in wf["folds"]:
            marker = " ★" if f["sharpe"] > 0 else ""
            print(f"    {f['fold']:>4} {f['test_start']} → {f['test_end']} {f['sharpe']:>8.3f} {f['return_pct']:>+9.1f}{marker}")

    # ── Save ──
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "MegaStrategyV3.1-H2",
        "comparison": comparison,
        "regime_breakdown": reg_data,
        "crash_analysis": crash_data,
        "monthly_heatmap": heatmap,
        "walk_forward_v31": wf_v31,
        "walk_forward_v3": wf_v3,
        "leverage_stats": {
            "mean": round(float(lev.mean()), 3),
            "median": round(float(lev.median()), 3),
            "max": round(float(lev.max()), 3),
            "pct_flat": round(float((lev < 0.01).mean() * 100), 1),
        },
    }
    out_path = RESULTS_DIR / "mega_v31_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "#" * 70)
    print("#  BACKTEST COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
