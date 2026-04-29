"""
Institutional-Grade Quant Analysis — MegaStrategyV3.1
Part 1: Altcoin expansion (10 alts)
Part 2: Risk, Factor, Robustness, Capacity, Benchmarks, Stability
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader
from datasource.factor_loader import FactorDataLoader
from strategies.composite.mega_strategy_v31 import run_full_strategy, LEVERAGE_MAP, WEIGHTS
from strategies.composite.mega_strategy_v3 import compute_confluence, ASSET_CONFIGS, _realized_vol, TX_COST

RESULTS_DIR = Path(os.path.expanduser("~/Desktop/maestro/data/research"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

stock = StockDataLoader()
fred = MacroDataLoader()

CROSS_ASSET_MAP = {"gold": "GLD", "dxy": "UUP", "bonds": "TLT", "hyg": "HYG", "copper": "COPX"}
FRED_SERIES = {"yield_curve": "T10Y2Y", "m2": "M2SL", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "hy_spread": "BAMLH0A0HYM2"}

ALT_TICKERS = {
    "AVAX": "AVAX-USD", "DOGE": "DOGE-USD", "ADA": "ADA-USD", "UNI": "UNI-USD",
    "AAVE": "AAVE-USD", "NEAR": "NEAR-USD", "CRV": "CRV-USD", "ZEC": "ZEC-USD",
    "ARB": "ARB-USD", "OP": "OP-USD",
}
CORE_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "LINK": "LINK-USD"}


def load_common_data():
    """Load macro + cross-asset data once."""
    print("Loading macro & cross-asset data...")
    cross_asset = pd.DataFrame()
    for col, ticker in CROSS_ASSET_MAP.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
            cross_asset[col] = df["close"]
        except: pass
    macro = fred.get_multiple(FRED_SERIES, start_date="2015-01-01")
    return cross_asset, macro


def backtest_single_asset_v31(close_series, macro_data, cross_asset_data, asset_name="BTC", tx_cost=0.001):
    """Run V3.1 signals on a single asset (equal weight=1.0) and return daily returns."""
    cfg = ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"])
    idx = close_series.index

    conf, bd = compute_confluence(
        close_series, macro_data, cross_asset_data,
        sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"]
    )
    conf = conf.reindex(idx).fillna(0)
    bd = bd.reindex(idx).fillna(0)

    # V3.1 leverage map
    lmap = LEVERAGE_MAP
    base_lev = conf.map(lambda c: lmap.get(min(int(c), 5), 0.0))

    # Score 4 smart demotion
    is_s4 = conf == 4
    cm_off = bd["crypto_momentum"] == 0
    base_lev = base_lev.where(~(is_s4 & cm_off), 0.5)
    base_lev = base_lev.shift(1).fillna(0)

    # Vol ceiling
    rvol = close_series.pct_change().rolling(30).std() * np.sqrt(252)
    vol_mask = rvol > 0.80
    base_lev = base_lev.where(~vol_mask.shift(1).fillna(False), base_lev * 0.5)

    # Bear filter (30 consecutive days score <= 1)
    low_score = (conf <= 1).astype(int)
    consec = low_score.groupby((low_score != low_score.shift()).cumsum()).cumcount() + 1
    consec = consec * low_score
    bear_mask = consec >= 30
    base_lev = base_lev.where(~bear_mask.shift(1).fillna(False), 0.0)

    # Returns
    asset_ret = close_series.pct_change().fillna(0)
    lev_change = base_lev.diff().abs().fillna(0)
    strat_ret = base_lev * asset_ret - lev_change * tx_cost

    return strat_ret


def compute_metrics(returns):
    """Core metrics dict."""
    if len(returns) < 20 or returns.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "total_return": 0}
    eq = (1 + returns).cumprod()
    n_yr = len(returns) / 252
    cagr = float(eq.iloc[-1] ** (1/max(n_yr, 0.1)) - 1)
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    sortino_denom = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(returns.mean() * 252 / sortino_denom) if sortino_denom > 0 else 0
    return {
        "sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2), "total_return": round(float(eq.iloc[-1] - 1) * 100, 2),
        "sortino": round(sortino, 3),
    }


# ══════════════════════════════════════════════════════════════════
# PART 1: ALTCOIN UNIVERSE
# ══════════════════════════════════════════════════════════════════
def part1_altcoin_expansion(cross_asset, macro):
    print("\n" + "=" * 70)
    print("PART 1: ALTCOIN UNIVERSE EXPANSION")
    print("=" * 70)

    # BTC B&H benchmark
    btc_df = stock.get_ohlcv("BTC-USD", "1d", start_date="2020-01-01")
    btc_bh_ret = btc_df["close"].pct_change().fillna(0)
    btc_bh = compute_metrics(btc_bh_ret)
    btc_strat = backtest_single_asset_v31(btc_df["close"], macro, cross_asset, "BTC")
    btc_strat_m = compute_metrics(btc_strat)

    results = {}
    print(f"\n{'Asset':<8} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'TotRet%':>9} {'Alpha vs BTC':>13}")
    print("-" * 55)

    for name, ticker in ALT_TICKERS.items():
        try:
            df = stock.get_ohlcv(ticker, "1d", start_date="2020-01-01")
            if len(df) < 200:
                print(f"{name:<8} insufficient data ({len(df)} days)")
                continue
            close = df["close"]
            strat_ret = backtest_single_asset_v31(close, macro, cross_asset, "BTC")  # use BTC config for macro
            m = compute_metrics(strat_ret)

            # BTC B&H over same period
            common_start = close.index[0]
            btc_slice = btc_bh_ret.loc[common_start:]
            btc_m = compute_metrics(btc_slice)
            alpha = m["cagr"] - btc_m["cagr"]

            m["alpha_vs_btc_bh"] = round(alpha, 2)
            m["days"] = len(df)
            m["start"] = str(df.index[0].date())
            results[name] = m
            print(f"{name:<8} {m['sharpe']:>7.3f} {m['cagr']:>7.1f} {m['max_dd']:>8.1f} {m['total_return']:>9.1f} {alpha:>+13.1f}")
        except Exception as e:
            print(f"{name:<8} FAILED: {e}")

    # Rank by alpha
    ranked = sorted(results.items(), key=lambda x: x[1].get("alpha_vs_btc_bh", -999), reverse=True)
    print(f"\nBTC B&H benchmark: Sharpe={btc_bh['sharpe']:.3f}, CAGR={btc_bh['cagr']:.1f}%")
    print(f"BTC V3.1:          Sharpe={btc_strat_m['sharpe']:.3f}, CAGR={btc_strat_m['cagr']:.1f}%")
    print(f"\nTop 5 by Alpha over BTC B&H:")
    for i, (name, m) in enumerate(ranked[:5]):
        print(f"  {i+1}. {name}: Alpha={m['alpha_vs_btc_bh']:+.1f}%, Sharpe={m['sharpe']:.3f}")

    return {"altcoin_results": results, "btc_bh": btc_bh, "btc_v31": btc_strat_m, "ranking": [r[0] for r in ranked]}


# ══════════════════════════════════════════════════════════════════
# PART 2: INSTITUTIONAL ANALYTICS (core portfolio)
# ══════════════════════════════════════════════════════════════════
def load_core_portfolio(cross_asset, macro):
    """Run V3.1 on core BTC/ETH/SOL/LINK and return portfolio returns."""
    crypto_data = {}
    for name, ticker in CORE_TICKERS.items():
        df = stock.get_ohlcv(ticker, "1d", start_date="2017-01-01")
        crypto_data[name] = df
        print(f"  {name}: {len(df)} days")

    portfolio_df, per_asset = run_full_strategy(crypto_data, macro, cross_asset)
    return portfolio_df, per_asset, crypto_data


def part2a_risk(port_ret):
    """CVaR, MaxDD duration, recovery, skew, kurtosis, tail ratio."""
    print("\n" + "=" * 70)
    print("PART 2A: RISK ANALYTICS")
    print("=" * 70)
    r = port_ret.dropna()

    cvar_95 = float(r[r <= r.quantile(0.05)].mean()) * 100
    cvar_99 = float(r[r <= r.quantile(0.01)].mean()) * 100

    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1

    # MaxDD duration & recovery
    in_dd = dd < 0
    dd_groups = (~in_dd).cumsum()
    dd_durations = in_dd.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # Recovery: time from trough to new high
    trough_idx = dd.idxmin()
    post_trough = eq.loc[trough_idx:]
    peak_before = eq.cummax().loc[trough_idx]
    recovery_mask = post_trough >= peak_before
    if recovery_mask.any():
        recovery_date = recovery_mask.idxmax()
        recovery_days = int((recovery_date - trough_idx).days)
    else:
        recovery_days = -1  # hasn't recovered

    skew_val = float(r.skew())
    kurt_val = float(r.kurtosis())

    sorted_r = r.sort_values()
    n5 = max(1, int(len(sorted_r) * 0.05))
    tail_ratio = float(abs(sorted_r.iloc[-n5:].mean() / sorted_r.iloc[:n5].mean())) if sorted_r.iloc[:n5].mean() != 0 else 0

    result = {
        "cvar_95_pct": round(cvar_95, 3),
        "cvar_99_pct": round(cvar_99, 3),
        "max_dd_duration_days": max_dd_duration,
        "recovery_from_max_dd_days": recovery_days,
        "skewness": round(skew_val, 3),
        "excess_kurtosis": round(kurt_val, 3),
        "tail_ratio": round(tail_ratio, 3),
    }
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def part2b_factor_exposure(port_ret):
    """Regress on FF5 + Momentum."""
    print("\n" + "=" * 70)
    print("PART 2B: FACTOR EXPOSURE (FF5 + Momentum)")
    print("=" * 70)
    fl = FactorDataLoader()
    factors = fl.get_all_factors()

    # Convert strategy daily returns to monthly
    monthly_ret = port_ret.resample("ME").sum()
    monthly_ret.index = monthly_ret.index.to_period("M").to_timestamp()

    # Align
    common = monthly_ret.index.intersection(factors.index)
    if len(common) < 12:
        print("  Insufficient overlap for factor regression")
        return {"error": "insufficient data"}

    y = monthly_ret.reindex(common).values
    rf = factors.reindex(common)["RF"].values if "RF" in factors.columns else np.zeros(len(common))
    y_excess = y - rf

    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in factors.columns]
    X = factors.reindex(common)[factor_cols].values
    X = np.column_stack([np.ones(len(X)), X])  # add intercept

    # OLS
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y_excess, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y_excess - y_hat) ** 2)
        ss_tot = np.sum((y_excess - y_excess.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n, p = X.shape
        mse = ss_res / (n - p)
        se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
        t_stats = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

        labels = ["alpha"] + factor_cols
        result = {"r_squared": round(r_squared, 4), "n_months": len(common)}
        print(f"  R² = {r_squared:.4f}, N = {len(common)} months")
        print(f"  {'Factor':<10} {'Beta':>8} {'t-stat':>8} {'p-val':>8}")
        print(f"  {'-'*36}")
        for i, lab in enumerate(labels):
            result[lab] = {"beta": round(float(beta[i]), 4), "t_stat": round(float(t_stats[i]), 3), "p_value": round(float(p_vals[i]), 4)}
            sig = "*" if p_vals[i] < 0.05 else ""
            print(f"  {lab:<10} {beta[i]:>8.4f} {t_stats[i]:>8.3f} {p_vals[i]:>8.4f} {sig}")

        # Annualized alpha
        monthly_alpha = beta[0]
        annual_alpha = ((1 + monthly_alpha) ** 12 - 1) * 100
        result["annualized_alpha_pct"] = round(annual_alpha, 2)
        print(f"\n  Annualized factor-adjusted alpha: {annual_alpha:+.2f}%")
        return result
    except Exception as e:
        print(f"  Regression failed: {e}")
        return {"error": str(e)}


def part2c_robustness(cross_asset, macro, crypto_data):
    """Cost sensitivity, bootstrap Sharpe CI, parameter sensitivity."""
    print("\n" + "=" * 70)
    print("PART 2C: ROBUSTNESS TESTS")
    print("=" * 70)

    # Cost sensitivity
    print("\n  Cost Sensitivity:")
    cost_results = {}
    for cost in [0.0005, 0.001, 0.002, 0.003, 0.005]:
        pf, _ = run_full_strategy(crypto_data, macro, cross_asset, tx_cost=cost)
        m = compute_metrics(pf["daily_pnl"])
        cost_results[str(cost)] = m
        print(f"    Cost={cost*100:.2f}%: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1f}%, MaxDD={m['max_dd']:.1f}%")

    # Bootstrap Sharpe CI (500x)
    print("\n  Bootstrap Sharpe (500 iterations)...")
    pf_base, _ = run_full_strategy(crypto_data, macro, cross_asset)
    ret = pf_base["daily_pnl"].dropna().values
    n = len(ret)
    boot_sharpes = []
    for _ in range(500):
        sample = np.random.choice(ret, size=n, replace=True)
        s = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
        boot_sharpes.append(s)
    ci_low, ci_high = np.percentile(boot_sharpes, [2.5, 97.5])
    boot_result = {
        "mean": round(np.mean(boot_sharpes), 3),
        "median": round(np.median(boot_sharpes), 3),
        "ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "prob_positive": round(float(np.mean(np.array(boot_sharpes) > 0)), 3),
    }
    print(f"    Sharpe: {boot_result['mean']:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"    P(Sharpe > 0) = {boot_result['prob_positive']:.1%}")

    # Parameter sensitivity (±20%)
    print("\n  Parameter Sensitivity (±20%):")
    base_sharpe = compute_metrics(pf_base["daily_pnl"])["sharpe"]
    param_results = {}
    param_tests = {
        "vol_ceiling": [0.64, 0.80, 0.96],
        "bear_filter_days": [24, 30, 36],
        "portfolio_trail_stop": [0.08, 0.10, 0.12],
    }
    for param, values in param_tests.items():
        sharpes = []
        for v in values:
            kwargs = {param: v}
            pf_test, _ = run_full_strategy(crypto_data, macro, cross_asset, **kwargs)
            s = compute_metrics(pf_test["daily_pnl"])["sharpe"]
            sharpes.append(s)
        param_results[param] = {str(v): round(s, 3) for v, s in zip(values, sharpes)}
        spread = max(sharpes) - min(sharpes)
        print(f"    {param}: {[round(s,3) for s in sharpes]} (spread={spread:.3f})")

    return {"cost_sensitivity": cost_results, "bootstrap_sharpe": boot_result, "parameter_sensitivity": param_results}


def part2d_capacity(crypto_data):
    """Volume analysis and market impact estimates."""
    print("\n" + "=" * 70)
    print("PART 2D: CAPACITY ANALYSIS")
    print("=" * 70)
    result = {}
    for name, df in crypto_data.items():
        vol = df["volume"] * df["close"]  # dollar volume
        avg_daily = float(vol.tail(90).mean())
        result[name] = {
            "avg_daily_volume_usd": round(avg_daily, 0),
            "impact_1M_bps": round(1e6 / avg_daily * 10, 1) if avg_daily > 0 else 999,
            "impact_10M_bps": round(1e7 / avg_daily * 10, 1) if avg_daily > 0 else 999,
            "impact_50M_bps": round(5e7 / avg_daily * 10, 1) if avg_daily > 0 else 999,
        }
        print(f"  {name}: AvgVol=${avg_daily/1e9:.2f}B, Impact@$10M={result[name]['impact_10M_bps']:.1f}bps")

    # Total capacity estimate (< 1% of min daily volume)
    min_vol = min(r["avg_daily_volume_usd"] for r in result.values())
    capacity = min_vol * 0.01
    result["estimated_capacity_usd"] = round(capacity, 0)
    print(f"\n  Estimated capacity (1% min daily vol): ${capacity/1e6:.1f}M")
    return result


def part2e_benchmarks(port_ret, cross_asset, macro, crypto_data):
    """Compare vs multiple benchmarks."""
    print("\n" + "=" * 70)
    print("PART 2E: BENCHMARK COMPARISON")
    print("=" * 70)

    benchmarks = {}

    # 1. BTC B&H
    btc_ret = crypto_data["BTC"]["close"].pct_change().fillna(0)
    common = port_ret.index.intersection(btc_ret.index)
    benchmarks["BTC_BH"] = compute_metrics(btc_ret.reindex(common))

    # 2. 60/40 SPY/BND
    try:
        spy = stock.get_ohlcv("SPY", "1d", start_date="2017-01-01")["close"].pct_change().fillna(0)
        bnd = stock.get_ohlcv("BND", "1d", start_date="2017-01-01")["close"].pct_change().fillna(0)
        common60 = spy.index.intersection(bnd.index).intersection(port_ret.index)
        ret_6040 = 0.6 * spy.reindex(common60) + 0.4 * bnd.reindex(common60)
        benchmarks["60_40_SPY_BND"] = compute_metrics(ret_6040)
    except: benchmarks["60_40_SPY_BND"] = {"error": "data unavailable"}

    # 3. SMA(200) on BTC
    btc_close = crypto_data["BTC"]["close"]
    sma200 = btc_close.rolling(200).mean()
    sma_signal = (btc_close > sma200).shift(1).fillna(False).astype(float)
    sma_ret = sma_signal * btc_close.pct_change().fillna(0)
    benchmarks["BTC_SMA200"] = compute_metrics(sma_ret.reindex(common))

    # 4. Equal-weight crypto B&H
    eq_ret = pd.Series(0.0, index=common)
    n_assets = len(crypto_data)
    for name, df in crypto_data.items():
        eq_ret += df["close"].pct_change().fillna(0).reindex(common, fill_value=0) / n_assets
    benchmarks["EqWeight_Crypto_BH"] = compute_metrics(eq_ret)

    # V3.1
    benchmarks["V31_Strategy"] = compute_metrics(port_ret.reindex(common))

    print(f"  {'Benchmark':<22} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8}")
    print(f"  {'-'*46}")
    for name, m in benchmarks.items():
        if "error" in m: continue
        print(f"  {name:<22} {m['sharpe']:>7.3f} {m['cagr']:>7.1f} {m['max_dd']:>8.1f}")

    return benchmarks


def part2f_stability(port_ret):
    """Rolling Sharpe, monthly stats, streaks."""
    print("\n" + "=" * 70)
    print("PART 2F: STABILITY ANALYSIS")
    print("=" * 70)

    # Rolling 252d Sharpe
    rolling_sharpe = (port_ret.rolling(252).mean() / port_ret.rolling(252).std()) * np.sqrt(252)
    rs_stats = {
        "mean": round(float(rolling_sharpe.dropna().mean()), 3),
        "min": round(float(rolling_sharpe.dropna().min()), 3),
        "max": round(float(rolling_sharpe.dropna().max()), 3),
        "pct_negative": round(float((rolling_sharpe.dropna() < 0).mean() * 100), 1),
    }
    print(f"  Rolling 252d Sharpe: mean={rs_stats['mean']}, min={rs_stats['min']}, max={rs_stats['max']}, %neg={rs_stats['pct_negative']}%")

    # Monthly returns
    monthly = port_ret.resample("ME").sum() * 100
    monthly_stats = {
        "mean": round(float(monthly.mean()), 2),
        "median": round(float(monthly.median()), 2),
        "std": round(float(monthly.std()), 2),
        "best": round(float(monthly.max()), 2),
        "worst": round(float(monthly.min()), 2),
        "best_month": str(monthly.idxmax().date()) if len(monthly) > 0 else "",
        "worst_month": str(monthly.idxmin().date()) if len(monthly) > 0 else "",
        "pct_positive": round(float((monthly > 0).mean() * 100), 1),
    }
    print(f"  Monthly: mean={monthly_stats['mean']:.1f}%, worst={monthly_stats['worst']:.1f}%, best={monthly_stats['best']:.1f}%, %pos={monthly_stats['pct_positive']}%")

    # Win/loss streaks
    monthly_sign = (monthly > 0).astype(int)
    streaks = monthly_sign.groupby((monthly_sign != monthly_sign.shift()).cumsum()).cumcount() + 1
    win_streaks = streaks[monthly_sign == 1]
    loss_streaks = streaks[monthly_sign == 0]
    streak_stats = {
        "max_winning_streak": int(win_streaks.max()) if len(win_streaks) > 0 else 0,
        "max_losing_streak": int(loss_streaks.max()) if len(loss_streaks) > 0 else 0,
    }
    print(f"  Max win streak: {streak_stats['max_winning_streak']} months, Max loss streak: {streak_stats['max_losing_streak']} months")

    return {"rolling_sharpe": rs_stats, "monthly_stats": monthly_stats, "streaks": streak_stats}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("#" * 70)
    print("#  INSTITUTIONAL ANALYSIS — MegaStrategyV3.1")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    cross_asset, macro = load_common_data()

    # PART 1
    alt_results = part1_altcoin_expansion(cross_asset, macro)

    # PART 2: Load core portfolio
    print("\n\nLoading core portfolio...")
    portfolio_df, per_asset, crypto_data = load_core_portfolio(cross_asset, macro)
    port_ret = portfolio_df["daily_pnl"]

    risk = part2a_risk(port_ret)
    factors = part2b_factor_exposure(port_ret)
    robustness = part2c_robustness(cross_asset, macro, crypto_data)
    capacity = part2d_capacity(crypto_data)
    benchmarks = part2e_benchmarks(port_ret, cross_asset, macro, crypto_data)
    stability = part2f_stability(port_ret)

    # ── SAVE ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "MegaStrategyV3.1-H2",
        "part1_altcoin_expansion": alt_results,
        "part2_risk": risk,
        "part2_factor_exposure": factors,
        "part2_robustness": robustness,
        "part2_capacity": capacity,
        "part2_benchmarks": benchmarks,
        "part2_stability": stability,
    }
    out_path = RESULTS_DIR / "institutional_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── EXECUTIVE SUMMARY ──
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    # Top 5 altcoins
    ranked = alt_results.get("ranking", [])[:5]
    print(f"\n  TOP 5 ALTCOINS (by alpha over BTC B&H):")
    for i, name in enumerate(ranked):
        m = alt_results["altcoin_results"].get(name, {})
        print(f"    {i+1}. {name}: Alpha={m.get('alpha_vs_btc_bh',0):+.1f}%, Sharpe={m.get('sharpe',0):.3f}, MaxDD={m.get('max_dd',0):.1f}%")

    # Factor alpha
    alpha_pct = factors.get("annualized_alpha_pct", "N/A")
    r2 = factors.get("r_squared", "N/A")
    print(f"\n  FACTOR-ADJUSTED ALPHA: {alpha_pct}% (R²={r2})")

    # Capacity
    cap = capacity.get("estimated_capacity_usd", 0)
    print(f"  CAPACITY LIMIT: ${cap/1e6:.0f}M (1% of min daily volume)")

    # Vulnerabilities
    print(f"\n  VULNERABILITIES:")
    print(f"    - MaxDD duration: {risk['max_dd_duration_days']} days")
    print(f"    - CVaR 99%: {risk['cvar_99_pct']:.2f}% daily")
    print(f"    - Excess kurtosis: {risk['excess_kurtosis']:.2f} (fat tails)")
    print(f"    - Skewness: {risk['skewness']:.2f}")
    print(f"    - Rolling Sharpe goes negative {stability['rolling_sharpe']['pct_negative']}% of time")

    # Robustness score (0-10)
    boot = robustness["bootstrap_sharpe"]
    prob_pos = boot["prob_positive"]
    ci_low = boot["ci_95"][0]
    param_spreads = []
    for p, vals in robustness["parameter_sensitivity"].items():
        sharpes = list(vals.values())
        param_spreads.append(max(sharpes) - min(sharpes))
    avg_spread = np.mean(param_spreads) if param_spreads else 1

    robustness_score = min(10, (
        3 * prob_pos +
        2 * max(0, ci_low) +
        2 * (1 - min(1, avg_spread)) +
        1.5 * (1 - stability["rolling_sharpe"]["pct_negative"] / 100) +
        1.5 * (stability["monthly_stats"]["pct_positive"] / 100)
    ))
    print(f"\n  ROBUSTNESS SCORE: {robustness_score:.1f}/10")
    print(f"    - Bootstrap P(Sharpe>0): {prob_pos:.1%}")
    print(f"    - Sharpe 95% CI: [{boot['ci_95'][0]:.3f}, {boot['ci_95'][1]:.3f}]")
    print(f"    - Avg param sensitivity: {avg_spread:.3f}")

    # Institutional readiness
    ready = "YES" if (prob_pos > 0.9 and ci_low > 0.5 and robustness_score > 6) else \
            "CONDITIONAL" if (prob_pos > 0.8 and robustness_score > 4) else "NO"
    print(f"\n  INSTITUTIONAL READINESS: {ready}")
    print("=" * 70)

    output["executive_summary"] = {
        "top_5_altcoins": ranked,
        "factor_adjusted_alpha_pct": alpha_pct,
        "capacity_limit_usd": cap,
        "robustness_score": round(robustness_score, 1),
        "institutional_readiness": ready,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
