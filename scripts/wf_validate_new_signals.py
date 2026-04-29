#!/usr/bin/env python3
"""
Walk-Forward Validation for 3 New Signals from 547-Signal Hunt

Signals:
  1. SOL Relative Momentum + BTC Bull (full-sample Sharpe 1.336)
  2. LSR Divergence Pairs — market-neutral long/short (full-sample Sharpe 1.026)
  3. SOL LSR Contrarian Below p20 (full-sample Sharpe 1.125)

Methodology: expanding-window WF with fixed params, 126-day OOS folds,
252-day minimum warmup, permutation test + bootstrap CI.
"""
import sys, os, json, warnings, time
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import duckdb

# ── Config ────────────────────────────────────────────────────
DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
SAVE_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")

TRAIN_DAYS_MIN = 252   # 1 year minimum warmup
TEST_DAYS = 126        # 6 months OOS per fold
TX_COST = 0.001        # 10 bps per trade
N_PERM = 500
N_BOOT = 1000
ANNUALIZE = np.sqrt(365)

# 14 symbols with both perps_daily and cg_lsr_global from 2021
LSR_UNIVERSE = [
    "ADA", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT",
    "ETH", "FIL", "LINK", "NEAR", "SOL", "UNI", "XRP",
]

np.random.seed(42)


# ── Data Loading ──────────────────────────────────────────────
def load_data():
    """Load all needed data from DuckDB."""
    con = duckdb.connect(DB_PATH, read_only=True)

    # Load perps_daily for all LSR universe symbols
    syms_str = ", ".join(f"'{s}'" for s in LSR_UNIVERSE)
    perps = con.execute(f"""
        SELECT date, symbol, close
        FROM perps_daily
        WHERE symbol IN ({syms_str})
        ORDER BY symbol, date
    """).fetchdf()

    # Load LSR global for same universe
    lsr = con.execute(f"""
        SELECT date, symbol, global_account_long_short_ratio as lsr
        FROM cg_lsr_global
        WHERE symbol IN ({syms_str})
        ORDER BY symbol, date
    """).fetchdf()

    con.close()

    # Convert dates
    perps["date"] = pd.to_datetime(perps["date"])
    lsr["date"] = pd.to_datetime(lsr["date"])

    return perps, lsr


def pivot_data(df, value_col):
    """Pivot long-form data to wide: date x symbol."""
    return df.pivot_table(index="date", columns="symbol", values=value_col).sort_index()


# ── Backtest Engine ───────────────────────────────────────────
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


def compute_pairs_returns(positions_long, positions_short, close_wide, tx_cost=TX_COST):
    """
    Compute daily returns for market-neutral long/short pairs strategy.
    positions_long: DataFrame (date x symbol), 1/3 for selected, 0 otherwise
    positions_short: DataFrame (date x symbol), -1/3 for selected, 0 otherwise
    close_wide: DataFrame (date x symbol) of close prices
    """
    # Returns for each asset
    asset_returns = close_wide.pct_change().fillna(0)

    # Combined positions
    positions = positions_long + positions_short  # long is +1/3, short is -1/3

    # Position changes for tx costs
    pos_change = positions.diff().abs().fillna(0)
    tc = pos_change * tx_cost

    # Strategy return per day = sum across assets of (pos * ret - tc)
    daily_strat = (positions * asset_returns - tc).sum(axis=1)

    return daily_strat


# ── Statistical Tests ─────────────────────────────────────────
def permutation_test(oos_returns, n_perm=N_PERM):
    """
    Sign-randomization test: randomly flip signs of daily returns to test
    H0: mean strategy return = 0 (signal has no edge).
    This is the correct test for timing signals with partial exposure.
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


# ── Signal 1: SOL Relative Momentum + BTC Bull ───────────────
def signal_sol_relmom_btc_bull(close_wide):
    """
    Long SOL when:
      - SOL 20d return > BTC 20d return
      - AND BTC close > BTC SMA50
    shift(1) for no-lookahead.
    """
    sol_ret20 = close_wide["SOL"].pct_change(20)
    btc_ret20 = close_wide["BTC"].pct_change(20)
    btc_sma50 = close_wide["BTC"].rolling(50).mean()

    raw_signal = ((sol_ret20 > btc_ret20) & (close_wide["BTC"] > btc_sma50)).astype(float)
    return raw_signal.shift(1).fillna(0)


# ── Signal 2: LSR Divergence Pairs ───────────────────────────
def signal_lsr_divergence_pairs(lsr_wide, close_wide):
    """
    Market-neutral long/short:
      - Rank all 14 assets by current LSR
      - Long bottom 3 (most bearish positioning -- contrarian long)
      - Short top 3 (most bullish positioning -- contrarian short)
      - Equal weight (1/3 each leg). Rebalance daily. shift(1).
    Returns: positions_long, positions_short DataFrames
    """
    # Only use symbols present in both LSR and close data
    common_syms = sorted(set(lsr_wide.columns) & set(close_wide.columns))
    lsr_aligned = lsr_wide[common_syms].reindex(close_wide.index).ffill()
    close_aligned = close_wide[common_syms]

    # Rank each day (ascending: rank 1 = lowest LSR = most bearish)
    ranks = lsr_aligned.rank(axis=1, method="average", ascending=True)

    n_assets = len(common_syms)
    n_long = 3
    n_short = 3

    # Long: bottom 3 by LSR rank (rank <= 3)
    positions_long = (ranks <= n_long).astype(float)
    # Normalize so each long position = 1/3
    long_count = positions_long.sum(axis=1).replace(0, 1)
    positions_long = positions_long.div(long_count, axis=0)

    # Short: top 3 by LSR rank (rank > n_assets - 3)
    positions_short = (ranks > n_assets - n_short).astype(float)
    short_count = positions_short.sum(axis=1).replace(0, 1)
    positions_short = -positions_short.div(short_count, axis=0)

    # shift(1) for no-lookahead
    positions_long = positions_long.shift(1).fillna(0)
    positions_short = positions_short.shift(1).fillna(0)

    return positions_long, positions_short, close_aligned


# ── Signal 3: SOL LSR Contrarian Below p20 ───────────────────
def signal_sol_lsr_contrarian(lsr_wide, close_wide):
    """
    Long SOL when SOL's LSR < rolling 30-day 20th percentile.
    shift(1) for no-lookahead.
    """
    sol_lsr = lsr_wide["SOL"].reindex(close_wide.index).ffill()
    rolling_p20 = sol_lsr.rolling(30).quantile(0.20)
    raw_signal = (sol_lsr < rolling_p20).astype(float)
    return raw_signal.shift(1).fillna(0)


# ── Walk-Forward Engine ──────────────────────────────────────
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
    train_start = 0
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

    # ── Aggregate stats ───────────────────────────────
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


# ── Full-Sample Metrics ──────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 80)
    print("WALK-FORWARD VALIDATION: 3 New Signals from 547-Signal Hunt")
    print("=" * 80)
    print(f"Config: TRAIN_MIN={TRAIN_DAYS_MIN}d, TEST={TEST_DAYS}d, TX={TX_COST*10000:.0f}bps")
    print(f"        Annualize=sqrt(365), Expanding window, Fixed params (no optimization)")
    print()

    # ── Load Data ──────────────────────────────────────
    print("Loading data from DuckDB...", flush=True)
    perps, lsr = load_data()

    close_wide = pivot_data(perps, "close")
    lsr_wide = pivot_data(lsr, "lsr")

    # Forward-fill LSR (often has some gaps)
    lsr_wide = lsr_wide.ffill()

    # Find common date range where we have both
    common_start = max(close_wide.index[0], lsr_wide.index[0])
    common_end = min(close_wide.index[-1], lsr_wide.index[-1])
    close_wide = close_wide.loc[common_start:common_end]
    lsr_wide = lsr_wide.loc[common_start:common_end]

    # Drop dates with too many NaNs (need at least 14 symbols for pairs)
    valid_mask = close_wide[LSR_UNIVERSE].notna().sum(axis=1) >= len(LSR_UNIVERSE)
    first_valid = valid_mask[valid_mask].index[0]
    close_wide = close_wide.loc[first_valid:]
    lsr_wide = lsr_wide.loc[first_valid:]

    n = len(close_wide)
    print(f"Data: {n} bars, {close_wide.index[0].date()} to {close_wide.index[-1].date()}")
    print(f"Symbols: {LSR_UNIVERSE}")
    max_folds = (n - TRAIN_DAYS_MIN) // TEST_DAYS
    print(f"Expected folds: ~{max_folds}")
    print()

    all_results = {}

    # ──────────────────────────────────────────────────────────
    # SIGNAL 1: SOL Relative Momentum + BTC Bull
    # ──────────────────────────────────────────────────────────
    print("=" * 80)
    print("SIGNAL 1: SOL Relative Momentum + BTC Bull")
    print("  Long SOL when SOL_20d_ret > BTC_20d_ret AND BTC > BTC_SMA50")
    print("=" * 80)

    signal1 = signal_sol_relmom_btc_bull(close_wide)
    strat_ret1 = compute_strategy_returns(signal1, close_wide["SOL"])

    # Full sample
    fs1 = full_sample_metrics(strat_ret1.values)
    print(f"  Full-sample: Sharpe={fs1['sharpe']:.3f}, CAGR={fs1['cagr']*100:.1f}%, "
          f"MaxDD={fs1['max_dd']*100:.1f}%, Exposure={fs1['exposure']*100:.1f}%")
    print()

    def sig1_fold_returns(test_start, test_end):
        return strat_ret1.iloc[test_start:test_end].values

    dates = close_wide.index
    folds1, agg1 = run_walkforward("SOL_RelMom_BTCBull", sig1_fold_returns, n, dates)
    all_results["sol_relmom_btc_bull"] = {
        "signal": "SOL Relative Momentum + BTC Bull",
        "description": "Long SOL when SOL 20d return > BTC 20d return AND BTC > SMA50",
        "n_folds": len(folds1),
        "folds": folds1,
        "aggregate_oos": agg1,
        "full_sample": fs1,
    }

    # ──────────────────────────────────────────────────────────
    # SIGNAL 2: LSR Divergence Pairs (Market-Neutral)
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SIGNAL 2: LSR Divergence Pairs (Market-Neutral)")
    print("  Long bottom-3 LSR (contrarian), Short top-3 LSR. Equal weight. Daily rebal.")
    print("=" * 80)

    pos_long, pos_short, close_aligned = signal_lsr_divergence_pairs(lsr_wide, close_wide)
    strat_ret2 = compute_pairs_returns(pos_long, pos_short, close_aligned)

    fs2 = full_sample_metrics(strat_ret2.values)
    print(f"  Full-sample: Sharpe={fs2['sharpe']:.3f}, CAGR={fs2['cagr']*100:.1f}%, "
          f"MaxDD={fs2['max_dd']*100:.1f}%, Exposure={fs2['exposure']*100:.1f}%")
    print()

    def sig2_fold_returns(test_start, test_end):
        return strat_ret2.iloc[test_start:test_end].values

    folds2, agg2 = run_walkforward("LSR_Divergence_Pairs", sig2_fold_returns, n, dates)
    all_results["lsr_divergence_pairs"] = {
        "signal": "LSR Divergence Pairs (Market-Neutral)",
        "description": "Long bottom-3 LSR, short top-3 LSR. Equal weight. Daily rebalance.",
        "n_folds": len(folds2),
        "folds": folds2,
        "aggregate_oos": agg2,
        "full_sample": fs2,
    }

    # ──────────────────────────────────────────────────────────
    # SIGNAL 3: SOL LSR Contrarian Below p20
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SIGNAL 3: SOL LSR Contrarian Below p20")
    print("  Long SOL when SOL LSR < rolling 30d 20th percentile")
    print("=" * 80)

    signal3 = signal_sol_lsr_contrarian(lsr_wide, close_wide)
    strat_ret3 = compute_strategy_returns(signal3, close_wide["SOL"])

    fs3 = full_sample_metrics(strat_ret3.values)
    print(f"  Full-sample: Sharpe={fs3['sharpe']:.3f}, CAGR={fs3['cagr']*100:.1f}%, "
          f"MaxDD={fs3['max_dd']*100:.1f}%, Exposure={fs3['exposure']*100:.1f}%")
    print()

    def sig3_fold_returns(test_start, test_end):
        return strat_ret3.iloc[test_start:test_end].values

    folds3, agg3 = run_walkforward("SOL_LSR_Contrarian", sig3_fold_returns, n, dates)
    all_results["sol_lsr_contrarian_p20"] = {
        "signal": "SOL LSR Contrarian Below p20",
        "description": "Long SOL when LSR < rolling 30d 20th percentile",
        "n_folds": len(folds3),
        "folds": folds3,
        "aggregate_oos": agg3,
        "full_sample": fs3,
    }

    # ──────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 100)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 100)
    print(f"{'Signal':<40} | {'FS Sharpe':>9} | {'OOS Mean':>9} | {'OOS Med':>8} | "
          f"{'Concat':>7} | {'p-val':>6} | {'Boot CI':>18} | {'+Folds':>7} | {'Verdict':>10}")
    print("-" * 100)

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

        print(f"{res['signal']:<40} | {fs['sharpe']:>9.3f} | {a['mean_sharpe']:>+9.3f} | "
              f"{a['median_sharpe']:>+8.3f} | {a['concat_sharpe']:>+7.3f} | "
              f"{a['p_value']:>6.3f} | [{ci['lower']:>+7.3f}, {ci['upper']:>+7.3f}] | "
              f"{a['positive_folds']:>7s} | {verdict:>10s}")

    print("-" * 100)
    print(f"\nExpected 30-50% Sharpe degradation OOS vs full-sample.")
    print(f"PASS = OOS Sharpe > 0.5, p < 0.05, 60%+ positive folds")
    print(f"MARGINAL = OOS Sharpe > 0.3, p < 0.10, 50%+ positive folds")
    print(f"FAIL = below marginal thresholds")
    print(f"\nElapsed: {elapsed:.1f}s")

    # ── Save Results ───────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "wf_new_signals.json")

    # Clean up for JSON serialization
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
            },
            "universe": LSR_UNIVERSE,
            "data_range": f"{close_wide.index[0].date()} to {close_wide.index[-1].date()}",
            "n_bars": n,
        },
        "signals": all_results,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
