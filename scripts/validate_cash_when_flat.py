#!/usr/bin/env python3
"""
Validate Cash-When-Flat vs Hedge-When-Flat — Head-to-Head Walk-Forward Comparison.

Consensus matrix recommended replacing the hedge overlay with cash (7.5/10 avg).
This script runs both variants through identical WF validation and analyzes
regime-conditional performance to determine which mode is optimal.

Variants tested:
  A) HEDGE: Current production — short BTC when flat + bearish (hedge=True)
  B) CASH:  Cash-when-flat — do nothing when signals are flat (hedge=False)
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/maestro"))

import duckdb, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Reuse all signal functions from the main WF script
from scripts.walkforward_full_system import (
    load_data, compute_full_system_pnl, WEIGHTS, TX_COST,
    TRAIN_DAYS_MIN, TEST_DAYS, BLOCK_LEN
)

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
N_PERMS = 10000
N_BOOT = 5000
np.random.seed(42)


def wf_folds(pnl, warmup=50):
    """Generate fold boundaries and extract OOS returns per fold."""
    p = pnl[warmup:]
    n = len(p)
    folds = []
    start = TRAIN_DAYS_MIN
    while start + TEST_DAYS <= n:
        folds.append((start, start + TEST_DAYS))
        start += TEST_DAYS
    return p, folds


def compute_fold_metrics(oos_returns):
    """Compute metrics for a single fold's OOS returns."""
    if len(oos_returns) < 20:
        return None
    mu = np.mean(oos_returns)
    sd = np.std(oos_returns)
    sharpe = mu / sd * np.sqrt(365) if sd > 0 else 0
    equity = np.cumprod(1 + np.clip(oos_returns, -0.99, None))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return {
        'sharpe': round(sharpe, 4),
        'total_return': round(float(equity[-1] - 1), 4),
        'max_dd': round(abs(dd.min()), 4),
        'mean_daily_bps': round(mu * 10000, 2),
        'n_days': len(oos_returns),
    }


def block_perm_pvalue(oos, block_len=BLOCK_LEN, n_perms=N_PERMS):
    """Block sign-permutation p-value."""
    obs_mean = np.mean(oos)
    n_blocks = int(np.ceil(len(oos) / block_len))
    count = 0
    for _ in range(n_perms):
        signs = np.repeat(np.random.choice([-1, 1], size=n_blocks), block_len)[:len(oos)]
        if np.mean(oos * signs) >= obs_mean:
            count += 1
    return (count + 1) / (n_perms + 1)


def bootstrap_sharpe_ci(oos, n_boot=N_BOOT):
    """Bootstrap 95% CI on Sharpe ratio."""
    boots = []
    for _ in range(n_boot):
        b = np.random.choice(oos, size=len(oos), replace=True)
        sd = np.std(b)
        boots.append(np.mean(b) / sd * np.sqrt(365) if sd > 0 else 0)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def full_wf_analysis(daily_pnl, label, warmup=50):
    """Run complete WF analysis on a daily P&L series."""
    p, folds = wf_folds(daily_pnl, warmup)
    if not folds:
        return None

    fold_results = []
    all_oos = []

    for idx, (ts, te) in enumerate(folds):
        test = p[ts:te]
        m = compute_fold_metrics(test)
        if m:
            m['fold'] = idx + 1
            fold_results.append(m)
            all_oos.extend(test.tolist())

    if not fold_results:
        return None

    oos = np.array(all_oos)
    mu = np.mean(oos)
    sd = np.std(oos)
    concat_sharpe = mu / sd * np.sqrt(365) if sd > 0 else 0

    p_val = block_perm_pvalue(oos)
    ci_lo, ci_hi = bootstrap_sharpe_ci(oos)

    f_sharpes = [f['sharpe'] for f in fold_results]
    pos_folds = sum(1 for s in f_sharpes if s > 0)

    # Full-sample
    fs_mu = np.mean(p)
    fs_sd = np.std(p)
    fs_sharpe = fs_mu / fs_sd * np.sqrt(365) if fs_sd > 0 else 0
    fs_eq = np.cumprod(1 + np.clip(p, -0.99, None))
    ny = len(p) / 365
    fs_cagr = fs_eq[-1] ** (1 / ny) - 1 if ny > 0 and fs_eq[-1] > 0 else -1
    fs_peak = np.maximum.accumulate(fs_eq)
    fs_dd = (fs_eq - fs_peak) / np.where(fs_peak > 0, fs_peak, 1)

    return {
        'label': label,
        'full_sample': {
            'sharpe': round(fs_sharpe, 4),
            'cagr': round(fs_cagr, 4),
            'max_dd': round(abs(fs_dd.min()), 4),
            'daily_mean_bps': round(fs_mu * 10000, 2),
            'n_days': len(p),
        },
        'wf_oos': {
            'concat_sharpe': round(concat_sharpe, 4),
            'mean_sharpe': round(np.mean(f_sharpes), 4),
            'median_sharpe': round(np.median(f_sharpes), 4),
            'p_value': round(p_val, 4),
            'ci_lower': round(ci_lo, 2),
            'ci_upper': round(ci_hi, 2),
            'positive_folds': f"{pos_folds}/{len(fold_results)}",
            'n_oos_days': len(oos),
            'n_folds': len(fold_results),
        },
        'folds': fold_results,
        'oos_daily': oos.tolist(),
    }


def regime_analysis(df, pnl_hedge, pnl_cash, warmup=50):
    """Analyze hedge vs cash performance by BTC market regime."""
    close = df['btc_close'].values[warmup:]
    sma50 = pd.Series(df['btc_close'].values).rolling(50, min_periods=30).mean().values[warmup:]
    ret20 = pd.Series(df['btc_close'].values).pct_change(20).values[warmup:]
    vol20 = pd.Series(df['btc_close'].pct_change()).rolling(20).std().values[warmup:]

    ph = pnl_hedge[warmup:]
    pc = pnl_cash[warmup:]
    n = min(len(close), len(ph), len(pc))

    # Classify regime for each day
    regimes = []
    for i in range(n):
        if np.isnan(sma50[i]) or np.isnan(ret20[i]) or np.isnan(vol20[i]):
            regimes.append('unknown')
            continue
        above_sma = close[i] > sma50[i]
        trending_up = ret20[i] > 0.05
        trending_down = ret20[i] < -0.05
        high_vol = vol20[i] > np.nanmedian(vol20[:i+1]) * 1.2 if i > 20 else False

        if trending_up and above_sma:
            regimes.append('bull_trend')
        elif trending_down and not above_sma:
            regimes.append('bear_trend')
        elif high_vol:
            regimes.append('choppy')
        else:
            regimes.append('range')

    results = {}
    for regime in ['bull_trend', 'bear_trend', 'choppy', 'range']:
        mask = np.array([r == regime for r in regimes[:n]])
        if mask.sum() < 20:
            continue
        h_rets = ph[:n][mask]
        c_rets = pc[:n][mask]
        h_mu = np.mean(h_rets)
        c_mu = np.mean(c_rets)
        h_sd = np.std(h_rets)
        c_sd = np.std(c_rets)

        results[regime] = {
            'n_days': int(mask.sum()),
            'pct_of_total': round(mask.sum() / n * 100, 1),
            'hedge_mean_bps': round(h_mu * 10000, 2),
            'cash_mean_bps': round(c_mu * 10000, 2),
            'hedge_sharpe': round(h_mu / h_sd * np.sqrt(365), 2) if h_sd > 0 else 0,
            'cash_sharpe': round(c_mu / c_sd * np.sqrt(365), 2) if c_sd > 0 else 0,
            'winner': 'HEDGE' if h_mu > c_mu else 'CASH',
            'delta_bps': round((h_mu - c_mu) * 10000, 2),
        }

    return results


def paired_fold_test(folds_hedge, folds_cash):
    """Paired t-test on per-fold Sharpe differences."""
    n = min(len(folds_hedge), len(folds_cash))
    diffs = [folds_hedge[i]['sharpe'] - folds_cash[i]['sharpe'] for i in range(n)]
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    t_stat = mean_diff / (sd_diff / np.sqrt(n)) if sd_diff > 0 else 0
    # Approximate two-sided p-value using normal (valid for n >= 10)
    from scipy import stats
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    return {
        'mean_sharpe_diff': round(mean_diff, 4),
        'std_sharpe_diff': round(sd_diff, 4),
        't_stat': round(t_stat, 4),
        'p_value': round(p_val, 4),
        'n_folds': n,
        'hedge_wins': sum(1 for d in diffs if d > 0),
        'cash_wins': sum(1 for d in diffs if d < 0),
        'per_fold_diffs': [round(d, 4) for d in diffs],
    }


if __name__ == '__main__':
    print("=" * 100)
    print("  CASH-WHEN-FLAT vs HEDGE-WHEN-FLAT — Walk-Forward Validation")
    print("  Consensus matrix recommendation: replace hedge with cash (7.5/10)")
    print("=" * 100)

    print("\nLoading data...")
    df, lsr_all, price_all, symbols = load_data()
    warmup = 50
    print(f"Period: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} ({len(df)} days)")

    # ── Compute both variants ────────────────────────────────────────────
    print("\nComputing daily P&L for both variants...")
    pnl_hedge = compute_full_system_pnl(df, lsr_all, price_all, symbols,
                                         hedge=True, vol_mult=2.0)
    pnl_cash = compute_full_system_pnl(df, lsr_all, price_all, symbols,
                                        hedge=False, vol_mult=2.0)

    # ── Walk-Forward Validation ──────────────────────────────────────────
    print("\nRunning WF validation for HEDGE variant (10K block perms)...")
    result_hedge = full_wf_analysis(pnl_hedge, "Hedge-When-Flat", warmup)

    print("Running WF validation for CASH variant (10K block perms)...")
    np.random.seed(42)  # Reset seed for fair comparison
    result_cash = full_wf_analysis(pnl_cash, "Cash-When-Flat", warmup)

    # ── Print Results ────────────────────────────────────────────────────
    for label, r in [("HEDGE-WHEN-FLAT (Current)", result_hedge),
                     ("CASH-WHEN-FLAT (Proposed)", result_cash)]:
        if not r:
            continue
        fs = r['full_sample']
        wf = r['wf_oos']
        sig = "***" if wf['p_value'] < 0.01 else "**" if wf['p_value'] < 0.05 else "*" if wf['p_value'] < 0.10 else ""

        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        print(f"  Full-Sample:  Sharpe {fs['sharpe']:.2f}  |  CAGR {fs['cagr']:+.1%}  |  MaxDD {fs['max_dd']:.1%}")
        print(f"  WF OOS:       Sharpe {wf['concat_sharpe']:.2f} {sig}  |  p={wf['p_value']:.4f}  |  CI [{wf['ci_lower']:.2f}, {wf['ci_upper']:.2f}]")
        print(f"  Folds:        {wf['positive_folds']} positive  |  Mean {wf['mean_sharpe']:.2f}  |  Median {wf['median_sharpe']:.2f}")

        print(f"\n  {'Fold':>4} | {'Sharpe':>7} | {'Return':>8} | {'MaxDD':>7} | {'Bps/day':>8}")
        print(f"  {'-'*45}")
        for f in r['folds']:
            print(f"  {f['fold']:>4} | {f['sharpe']:>7.2f} | {f['total_return']:>+7.1%} | "
                  f"{f['max_dd']:>6.1%} | {f['mean_daily_bps']:>+7.1f}")

    # ── Head-to-Head Comparison ──────────────────────────────────────────
    if result_hedge and result_cash:
        print(f"\n{'='*100}")
        print("  HEAD-TO-HEAD: HEDGE vs CASH-WHEN-FLAT")
        print(f"{'='*100}")

        wh = result_hedge['wf_oos']
        wc = result_cash['wf_oos']
        fh = result_hedge['full_sample']
        fc = result_cash['full_sample']

        print(f"\n  {'Metric':<25} | {'Hedge':>12} | {'Cash':>12} | {'Delta':>10} | {'Winner':>8}")
        print(f"  {'-'*75}")
        print(f"  {'FS Sharpe':<25} | {fh['sharpe']:>12.2f} | {fc['sharpe']:>12.2f} | "
              f"{fh['sharpe'] - fc['sharpe']:>+10.2f} | {'HEDGE' if fh['sharpe'] > fc['sharpe'] else 'CASH':>8}")
        print(f"  {'FS CAGR':<25} | {fh['cagr']:>11.1%} | {fc['cagr']:>11.1%} | "
              f"{(fh['cagr'] - fc['cagr'])*100:>+9.1f}pp | {'HEDGE' if fh['cagr'] > fc['cagr'] else 'CASH':>8}")
        print(f"  {'FS MaxDD':<25} | {fh['max_dd']:>11.1%} | {fc['max_dd']:>11.1%} | "
              f"{(fh['max_dd'] - fc['max_dd'])*100:>+9.1f}pp | {'CASH' if fh['max_dd'] > fc['max_dd'] else 'HEDGE':>8}")
        print(f"  {'WF Concat Sharpe':<25} | {wh['concat_sharpe']:>12.2f} | {wc['concat_sharpe']:>12.2f} | "
              f"{wh['concat_sharpe'] - wc['concat_sharpe']:>+10.2f} | {'HEDGE' if wh['concat_sharpe'] > wc['concat_sharpe'] else 'CASH':>8}")
        print(f"  {'WF p-value':<25} | {wh['p_value']:>12.4f} | {wc['p_value']:>12.4f} | "
              f"{'':>10} | {'HEDGE' if wh['p_value'] < wc['p_value'] else 'CASH':>8}")
        print(f"  {'Positive folds':<25} | {wh['positive_folds']:>12} | {wc['positive_folds']:>12} | "
              f"{'':>10} |")
        print(f"  {'95% CI':<25} | [{wh['ci_lower']:.2f}, {wh['ci_upper']:.2f}]{'':<2} | "
              f"[{wc['ci_lower']:.2f}, {wc['ci_upper']:.2f}]{'':<2} |")

        # Per-fold comparison
        print(f"\n  Per-Fold Sharpe:")
        print(f"  {'Fold':>4} | {'Hedge':>8} | {'Cash':>8} | {'Delta':>8} | {'Winner':>8}")
        print(f"  {'-'*45}")
        for fh_f, fc_f in zip(result_hedge['folds'], result_cash['folds']):
            d = fh_f['sharpe'] - fc_f['sharpe']
            w = "HEDGE" if d > 0 else "CASH"
            print(f"  {fh_f['fold']:>4} | {fh_f['sharpe']:>8.2f} | {fc_f['sharpe']:>8.2f} | {d:>+8.2f} | {w:>8}")

        # Paired test
        print(f"\n  Paired Fold Test (H0: hedge - cash = 0):")
        pt = paired_fold_test(result_hedge['folds'], result_cash['folds'])
        sig = "***" if pt['p_value'] < 0.01 else "**" if pt['p_value'] < 0.05 else "*" if pt['p_value'] < 0.10 else "ns"
        print(f"    Mean Sharpe diff: {pt['mean_sharpe_diff']:+.4f}")
        print(f"    t-stat:           {pt['t_stat']:.4f}")
        print(f"    p-value:          {pt['p_value']:.4f} {sig}")
        print(f"    Hedge wins:       {pt['hedge_wins']}/{pt['n_folds']} folds")
        print(f"    Cash wins:        {pt['cash_wins']}/{pt['n_folds']} folds")

        # ── Regime Analysis ──────────────────────────────────────────────
        print(f"\n{'='*100}")
        print("  REGIME-CONDITIONAL PERFORMANCE")
        print(f"{'='*100}")
        print(f"  When does hedge help vs hurt? Classifying days into regimes.\n")

        regimes = regime_analysis(df, pnl_hedge, pnl_cash, warmup)

        print(f"  {'Regime':<14} | {'Days':>5} | {'%':>5} | {'Hedge bps':>10} | {'Cash bps':>9} | "
              f"{'H Sharpe':>9} | {'C Sharpe':>9} | {'Delta':>7} | {'Winner':>7}")
        print(f"  {'-'*90}")

        for regime in ['bull_trend', 'bear_trend', 'choppy', 'range']:
            if regime not in regimes:
                continue
            r = regimes[regime]
            print(f"  {regime:<14} | {r['n_days']:>5} | {r['pct_of_total']:>4.1f}% | "
                  f"{r['hedge_mean_bps']:>+9.2f} | {r['cash_mean_bps']:>+8.2f} | "
                  f"{r['hedge_sharpe']:>9.2f} | {r['cash_sharpe']:>9.2f} | "
                  f"{r['delta_bps']:>+6.2f} | {r['winner']:>7}")

        # ── Verdict ──────────────────────────────────────────────────────
        print(f"\n{'='*100}")
        print("  VERDICT")
        print(f"{'='*100}")

        hedge_sharpe = wh['concat_sharpe']
        cash_sharpe = wc['concat_sharpe']
        diff = hedge_sharpe - cash_sharpe
        winner = "HEDGE" if diff > 0 else "CASH"

        if abs(pt['p_value']) < 0.05:
            conclusion = f"{winner} is STATISTICALLY SIGNIFICANTLY better (paired p={pt['p_value']:.4f})"
        elif abs(diff) < 0.1:
            conclusion = "NO MEANINGFUL DIFFERENCE — pick based on regime preference"
        else:
            conclusion = f"{winner} has higher Sharpe ({diff:+.2f}) but NOT statistically significant (p={pt['p_value']:.4f})"

        print(f"\n  WF Sharpe:  Hedge {hedge_sharpe:.2f} vs Cash {cash_sharpe:.2f}  (delta {diff:+.2f})")
        print(f"  Conclusion: {conclusion}")

        # Regime-based recommendation
        bear_regime = regimes.get('bear_trend', {})
        choppy_regime = regimes.get('choppy', {})

        if bear_regime.get('winner') == 'HEDGE' and choppy_regime.get('winner') == 'CASH':
            print(f"\n  NUANCE: Hedge helps in bear trends ({bear_regime.get('delta_bps', 0):+.1f} bps/day)")
            print(f"          Cash helps in chop ({choppy_regime.get('delta_bps', 0):+.1f} bps/day)")
            print(f"          Consider: regime-adaptive switching (hedge in bear, cash in chop)")
        elif bear_regime.get('winner') == 'HEDGE':
            print(f"\n  Hedge adds value specifically in bear regimes ({bear_regime.get('delta_bps', 0):+.1f} bps/day)")
        else:
            print(f"\n  Cash is broadly better across regimes — switch to cash-when-flat")

    # ── Save Results ─────────────────────────────────────────────────────
    output = {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}",
        'config': {
            'weights': WEIGHTS,
            'vol_mult': 2.0,
            'tx_cost': TX_COST,
            'train_min': TRAIN_DAYS_MIN,
            'test_days': TEST_DAYS,
            'n_perms': N_PERMS,
            'n_boot': N_BOOT,
            'block_len': BLOCK_LEN,
        },
        'hedge_variant': {
            'full_sample': result_hedge['full_sample'] if result_hedge else None,
            'wf_oos': result_hedge['wf_oos'] if result_hedge else None,
            'folds': result_hedge['folds'] if result_hedge else None,
        },
        'cash_variant': {
            'full_sample': result_cash['full_sample'] if result_cash else None,
            'wf_oos': result_cash['wf_oos'] if result_cash else None,
            'folds': result_cash['folds'] if result_cash else None,
        },
        'paired_test': pt if result_hedge and result_cash else None,
        'regime_analysis': regimes if result_hedge and result_cash else None,
    }

    out_path = os.path.join(RESULTS_DIR, 'cash_when_flat_validation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"{'='*100}")
