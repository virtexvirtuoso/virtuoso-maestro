#!/usr/bin/env python3
"""
Walk-Forward Validation: Relative Momentum Signals for DOGE, XRP, AVAX

Signals (6 total):
  For each ALT in {DOGE, XRP, AVAX}:
    - 20d variant: Long ALT when ALT_20d_return > BTC_20d_return AND BTC > BTC_SMA50
    - 10d variant: Long ALT when ALT_10d_return > BTC_10d_return AND BTC > BTC_SMA50

Methodology: expanding-window WF with fixed params, 126-day OOS folds,
252-day minimum warmup, permutation test + bootstrap CI.
"""
import sys, os, json, warnings, time
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import duckdb

# -- Config ---------------------------------------------------------------
DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
SAVE_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")

TRAIN_DAYS_MIN = 252   # 1 year minimum warmup
TEST_DAYS = 126        # 6 months OOS per fold
TX_COST = 0.001        # 10 bps per trade
N_PERM = 500
N_BOOT = 1000
ANNUALIZE = np.sqrt(365)

ALT_SYMBOLS = ["DOGE", "XRP", "AVAX"]
LOOKBACKS = [20, 10]

np.random.seed(42)


# -- Data Loading ----------------------------------------------------------
def load_data():
    """Load BTC + ALT prices from DuckDB perps_daily."""
    con = duckdb.connect(DB_PATH, read_only=True)

    needed = ["BTC"] + ALT_SYMBOLS
    syms_str = ", ".join(f"'{s}'" for s in needed)
    perps = con.execute(f"""
        SELECT date, symbol, close
        FROM perps_daily
        WHERE symbol IN ({syms_str})
        ORDER BY symbol, date
    """).fetchdf()

    con.close()

    perps["date"] = pd.to_datetime(perps["date"])
    return perps


def pivot_data(df, value_col):
    """Pivot long-form data to wide: date x symbol."""
    return df.pivot_table(index="date", columns="symbol", values=value_col).sort_index()


# -- Backtest Engine -------------------------------------------------------
def backtest_fold(daily_returns):
    """Compute metrics for a single fold from daily strategy returns (already cost-adjusted)."""
    ret = np.asarray(daily_returns, dtype=float)

    if len(ret) == 0 or np.all(ret == 0):
        return {
            "sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0,
            "trades": 0, "win_rate": 0.0, "daily_returns": [],
        }

    cum = np.cumprod(1 + ret)
    total_ret = cum[-1] - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0.0

    # Sharpe on ALL days (including flat/zero days) -- standard convention
    daily_mean = np.mean(ret)
    daily_std = np.std(ret)
    sharpe = (daily_mean / daily_std) * ANNUALIZE if daily_std > 0 else 0.0

    winning = np.sum(ret > 0)
    total_active = np.sum(ret != 0)
    win_rate = winning / total_active if total_active > 0 else 0.0

    return {
        "sharpe": sharpe, "total_return": total_ret, "max_dd": max_dd,
        "trades": int(total_active), "win_rate": win_rate,
        "daily_returns": ret.tolist(),
    }


def compute_strategy_returns(position, close_prices, tx_cost=TX_COST):
    """
    Given position Series and close Series (aligned index), compute daily strategy returns.
    Position is already shift(1)'d (no lookahead).
    """
    ret = close_prices.pct_change().fillna(0)
    pos_change = position.diff().abs().fillna(0)
    tc = pos_change * tx_cost
    strat_ret = position * ret - tc
    return strat_ret


# -- Statistical Tests -----------------------------------------------------
def permutation_test(oos_returns, n_perm=N_PERM):
    """
    Sign-randomization test: randomly flip signs of daily returns to test
    H0: mean strategy return = 0 (signal has no edge).
    """
    ret = np.asarray(oos_returns, dtype=float)
    if len(ret) < 10:
        return 1.0
    observed_mean = np.mean(ret)
    count = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(ret))
        perm_mean = np.mean(ret * signs)
        if perm_mean >= observed_mean:
            count += 1
    return count / n_perm


def bootstrap_ci(returns, n_boot=N_BOOT, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio (all days including flat)."""
    if len(returns) < 10:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        std = np.std(sample)
        if std > 0:
            sharpes.append(np.mean(sample) / std * ANNUALIZE)
    if not sharpes:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return {
        "mean": round(float(np.mean(sharpes)), 4),
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
    }


# -- Signal Generator ------------------------------------------------------
def signal_relmom_btc_bull(close_wide, alt_symbol, lookback):
    """
    Long ALT when:
      - ALT Nd return > BTC Nd return
      - AND BTC close > BTC SMA50
    shift(1) for no-lookahead.
    """
    alt_ret = close_wide[alt_symbol].pct_change(lookback)
    btc_ret = close_wide["BTC"].pct_change(lookback)
    btc_sma50 = close_wide["BTC"].rolling(50).mean()

    raw_signal = ((alt_ret > btc_ret) & (close_wide["BTC"] > btc_sma50)).astype(float)
    return raw_signal.shift(1).fillna(0)


# -- Walk-Forward Engine ---------------------------------------------------
def run_walkforward(name, daily_returns_fn, n_total, dates):
    """
    Expanding-window walk-forward validation.

    daily_returns_fn(train_end, test_end): returns array of daily strat returns for test period.
    n_total: total number of bars.
    dates: DatetimeIndex of all bars.
    """
    folds = []
    all_oos_returns = []
    fold_num = 0

    # Expanding window: train always starts at 0, grows each fold
    test_start = TRAIN_DAYS_MIN

    while test_start + TEST_DAYS <= n_total:
        fold_num += 1
        test_end = test_start + TEST_DAYS

        # Get daily returns for this test period
        test_returns = daily_returns_fn(test_start, test_end)

        metrics = backtest_fold(test_returns)
        all_oos_returns.extend(metrics.get("daily_returns", []))

        test_start_date = dates[test_start].strftime("%Y-%m-%d")
        test_end_date = dates[test_end - 1].strftime("%Y-%m-%d")

        fold_result = {
            "fold": fold_num,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "sharpe": round(metrics["sharpe"], 4),
            "total_return": round(metrics["total_return"], 4),
            "max_dd": round(metrics["max_dd"], 4),
            "trades": metrics["trades"],
            "win_rate": round(metrics["win_rate"], 4),
        }
        folds.append(fold_result)

        print(f"  Fold {fold_num:2d}: {test_start_date} to {test_end_date} | "
              f"Sharpe={metrics['sharpe']:+7.3f} | "
              f"Ret={metrics['total_return']*100:+7.1f}% | "
              f"MaxDD={metrics['max_dd']*100:-6.1f}% | "
              f"Trades={metrics['trades']:4d} | "
              f"WR={metrics['win_rate']*100:5.1f}%")

        # Advance: expanding window (train grows, next test starts after current test)
        test_start += TEST_DAYS

    # -- Aggregate stats -------------------------------------------
    oos_daily = np.array(all_oos_returns)

    print(f"\n  Permutation test ({N_PERM} iters)...", end=" ", flush=True)
    p_value = permutation_test(oos_daily)
    print(f"p={p_value:.4f}")

    print(f"  Bootstrap CI ({N_BOOT} samples)...", end=" ", flush=True)
    boot_ci = bootstrap_ci(oos_daily)
    print(f"[{boot_ci['lower']:.3f}, {boot_ci['upper']:.3f}]")

    oos_sharpes = [f["sharpe"] for f in folds]
    positive_folds = sum(1 for s in oos_sharpes if s > 0)

    # Concatenated OOS Sharpe (all days, including flat/zero)
    concat_sharpe = 0.0
    if len(oos_daily) > 1 and np.std(oos_daily) > 0:
        concat_sharpe = (np.mean(oos_daily) / np.std(oos_daily)) * ANNUALIZE

    aggregate = {
        "mean_sharpe": round(float(np.mean(oos_sharpes)), 4),
        "median_sharpe": round(float(np.median(oos_sharpes)), 4),
        "std_sharpe": round(float(np.std(oos_sharpes)), 4),
        "concat_sharpe": round(float(concat_sharpe), 4),
        "positive_folds": f"{positive_folds}/{len(folds)}",
        "p_value": round(float(p_value), 4),
        "bootstrap_ci": boot_ci,
        "n_oos_days": len(oos_daily),
        "n_active_days": int(np.sum(oos_daily != 0)),
    }

    return folds, aggregate


# -- Full-Sample Metrics ---------------------------------------------------
def full_sample_metrics(daily_returns):
    """Compute full-sample performance metrics."""
    ret = np.asarray(daily_returns, dtype=float)
    if len(ret) == 0:
        return {}
    cum = np.cumprod(1 + ret)
    total_ret = cum[-1] - 1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(dd.min())

    n_days = len(ret)
    active = ret[ret != 0]
    exposure = len(active) / n_days if n_days > 0 else 0

    # Sharpe on ALL days (standard convention)
    sharpe = 0.0
    if n_days > 1 and np.std(ret) > 0:
        sharpe = (np.mean(ret) / np.std(ret)) * ANNUALIZE

    cagr = (cum[-1] ** (365 / n_days) - 1) if n_days > 0 and cum[-1] > 0 else 0.0

    trades = int(np.sum(np.abs(np.diff(np.sign(ret))) > 0))  # rough trade count
    win_rate = np.sum(ret > 0) / np.sum(ret != 0) if np.sum(ret != 0) > 0 else 0

    return {
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "total_return": round(total_ret, 4),
        "max_dd": round(max_dd, 4),
        "exposure": round(exposure, 4),
        "trades": trades,
        "win_rate": round(win_rate, 4),
        "n_days": n_days,
    }


# -- Main ------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 80)
    print("WALK-FORWARD VALIDATION: Relative Momentum Signals — DOGE, XRP, AVAX")
    print("=" * 80)
    print(f"Config: TRAIN_MIN={TRAIN_DAYS_MIN}d, TEST={TEST_DAYS}d, TX={TX_COST*10000:.0f}bps")
    print(f"        Annualize=sqrt(365), Expanding window, Fixed params (no optimization)")
    print(f"        Lookbacks: {LOOKBACKS}d, SMA filter: BTC > SMA50")
    print()

    # -- Load Data -------------------------------------------------
    print("Loading data from DuckDB...", flush=True)
    perps = load_data()
    close_wide = pivot_data(perps, "close")

    # Drop rows where any needed symbol is NaN
    needed = ["BTC"] + ALT_SYMBOLS
    close_wide = close_wide.dropna(subset=needed)

    n = len(close_wide)
    dates = close_wide.index
    print(f"Data: {n} bars, {dates[0].date()} to {dates[-1].date()}")
    print(f"Symbols: BTC (filter) + {ALT_SYMBOLS} (traded)")
    max_folds = (n - TRAIN_DAYS_MIN) // TEST_DAYS
    print(f"Expected folds: ~{max_folds}")
    print()

    all_results = {}

    # -- Run each ALT x lookback combination -----------------------
    for alt in ALT_SYMBOLS:
        for lb in LOOKBACKS:
            key = f"{alt.lower()}_relmom_{lb}d"
            label = f"{alt} RelMom {lb}d + BTC Bull"
            desc = (f"Long {alt} when {alt}_{lb}d_return > BTC_{lb}d_return "
                    f"AND BTC > BTC_SMA50")

            print("=" * 80)
            print(f"SIGNAL: {label}")
            print(f"  {desc}")
            print("=" * 80)

            # Generate signal and strategy returns
            signal = signal_relmom_btc_bull(close_wide, alt, lb)
            strat_ret = compute_strategy_returns(signal, close_wide[alt])

            # Full sample metrics
            fs = full_sample_metrics(strat_ret.values)
            print(f"  Full-sample: Sharpe={fs['sharpe']:.3f}, "
                  f"CAGR={fs['cagr']*100:.1f}%, "
                  f"MaxDD={fs['max_dd']*100:.1f}%, "
                  f"Exposure={fs['exposure']*100:.1f}%")
            print()

            # Walk-forward
            def fold_returns_fn(test_start, test_end, sr=strat_ret):
                return sr.iloc[test_start:test_end].values

            folds, agg = run_walkforward(label, fold_returns_fn, n, dates)

            all_results[key] = {
                "signal": label,
                "description": desc,
                "asset": alt,
                "lookback": lb,
                "n_folds": len(folds),
                "folds": folds,
                "aggregate_oos": agg,
                "full_sample": fs,
            }
            print()

    # -- Summary Table ---------------------------------------------
    elapsed = time.time() - t0
    print()
    print("=" * 115)
    print("WALK-FORWARD VALIDATION SUMMARY — Relative Momentum Signals")
    print("=" * 115)
    print(f"{'Signal':<30} | {'FS Sharpe':>9} | {'OOS Concat':>10} | "
          f"{'OOS Mean':>9} | {'p-val':>6} | {'Boot CI':>18} | "
          f"{'+Folds':>7} | {'Verdict':>10}")
    print("-" * 115)

    for key, res in all_results.items():
        fs = res["full_sample"]
        a = res["aggregate_oos"]
        ci = a["bootstrap_ci"]

        # Verdict logic
        oos_sharpe = a["concat_sharpe"]
        p_val = a["p_value"]
        pos_folds = a["positive_folds"]
        pos_num = int(pos_folds.split("/")[0])
        tot_num = int(pos_folds.split("/")[1])
        pos_pct = pos_num / tot_num if tot_num > 0 else 0

        if oos_sharpe > 0.5 and p_val < 0.05 and pos_pct >= 0.6:
            verdict = "PASS"
        elif oos_sharpe > 0.3 and p_val < 0.10 and pos_pct >= 0.5:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        print(f"{res['signal']:<30} | {fs['sharpe']:>9.3f} | {a['concat_sharpe']:>+10.3f} | "
              f"{a['mean_sharpe']:>+9.3f} | {a['p_value']:>6.3f} | "
              f"[{ci['lower']:>+7.3f}, {ci['upper']:>+7.3f}] | "
              f"{a['positive_folds']:>7s} | {verdict:>10s}")

    print("-" * 115)
    print(f"\nExpected 30-50% Sharpe degradation OOS vs full-sample.")
    print(f"PASS     = OOS Concat Sharpe > 0.5, p < 0.05, 60%+ positive folds")
    print(f"MARGINAL = OOS Concat Sharpe > 0.3, p < 0.10, 50%+ positive folds")
    print(f"FAIL     = below marginal thresholds")
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- Save Results ----------------------------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "wf_relmom_alts.json")

    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "config": {
                "train_days_min": TRAIN_DAYS_MIN,
                "test_days": TEST_DAYS,
                "tx_cost_bps": TX_COST * 10000,
                "n_permutations": N_PERM,
                "n_bootstrap": N_BOOT,
                "annualization": "sqrt(365)",
                "window_type": "expanding",
                "lookbacks": LOOKBACKS,
            },
            "universe": ALT_SYMBOLS,
            "btc_filter": "BTC > SMA50",
            "data_range": f"{dates[0].date()} to {dates[-1].date()}",
            "n_bars": n,
        },
        "signals": all_results,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
