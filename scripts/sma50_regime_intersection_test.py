#!/usr/bin/env python3
"""
SMA50 AND Weighted Regime Intersection — Walk-Forward Validation

Hypothesis: Combining SMA50 trend filter (p=0.18) with weighted regime (p=0.002)
should yield higher precision at cost of lower exposure. If precision rises
faster than frequency falls, the intersection beats both components.

Uses IDENTICAL methodology from regime_detector_test.py:
  - 14-fold expanding walk-forward
  - 500 permutation tests
  - sqrt(365) for crypto annualization (fixed from original sqrt(252))
  - NO optimization, NO tuning
  - Standalone test (no Bonferroni — this is ONE strategy, not a battery)
"""

import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
OUTPUT_PATH = os.path.expanduser("~/Desktop/maestro/data/backtest_results/sma50_regime_intersection.json")
N_PERMS = 500
N_FOLDS = 14
TX_COST = 0.001  # 10bps round-trip


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(symbol='BTC'):
    """Load and merge all data sources."""
    con = duckdb.connect(DB_PATH, read_only=True)

    lsr = con.execute(f"SELECT date, global_account_long_short_ratio as lsr FROM cg_lsr_global WHERE symbol='{symbol}' ORDER BY date").df()
    fr = con.execute(f"SELECT date, close as funding_rate FROM cg_funding_rate WHERE symbol='{symbol}' ORDER BY date").df()
    liq = con.execute(f"SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as total_liq FROM cg_liquidations WHERE symbol='{symbol}' ORDER BY date").df()
    taker = con.execute(f"SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as taker_ratio FROM cg_taker_volume WHERE symbol='{symbol}' ORDER BY date").df()
    price = con.execute(f"SELECT date, open, high, low, close FROM perps_daily WHERE symbol='{symbol}' ORDER BY date").df()

    con.close()

    df = price.copy()
    for src in [lsr, fr, liq, taker]:
        df = df.merge(src, on='date', how='inner')

    df = df.sort_values('date').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    df['fwd_returns'] = df['returns'].shift(-1)

    return df


# ── Signal Computation ───────────────────────────────────────────────────────

def compute_weighted_regime(df, window=30):
    """Weighted regime score (from regime_detector_test.py)."""
    lsr_med = df['lsr'].rolling(window, min_periods=10).median()
    lsr_s = (df['lsr'] < lsr_med).astype(float)
    fund_s = (df['funding_rate'] < 0.03).astype(float)
    liq_p80 = df['total_liq'].rolling(window, min_periods=10).quantile(0.8)
    liq_s = (df['total_liq'] < liq_p80).astype(float)
    taker_s = (df['taker_ratio'] > 1.0).astype(float)

    df['weighted_score'] = lsr_s * 0.35 + fund_s * 0.35 + liq_s * 0.15 + taker_s * 0.15
    return df


def compute_signals(df):
    """Compute all signal variants."""
    df['sma50'] = df['close'].rolling(50, min_periods=50).mean()

    # Individual signals — NO shift here because fwd_returns already provides 1-day lag
    # (position[t] uses data at day t, earns return from t→t+1 via fwd_returns)
    df['sig_sma50'] = (df['close'] > df['sma50']).astype(float)
    df['sig_regime'] = (df['weighted_score'] > 0.5).astype(float)

    # INTERSECTION: both must be true
    df['sig_intersection'] = (df['sig_sma50'] * df['sig_regime']).astype(float)

    return df


# ── Backtest Engine ──────────────────────────────────────────────────────────

def compute_metrics(positions, fwd_returns, tx_cost=TX_COST):
    """Full backtest metrics with transaction costs."""
    valid = ~np.isnan(fwd_returns)
    pos = positions[valid]
    fwd = fwd_returns[valid]

    # Transaction costs on position changes
    pos_changes = np.abs(np.diff(pos))
    tc = np.concatenate([[0], pos_changes]) * tx_cost

    strat_ret = pos * fwd - tc

    if len(strat_ret) == 0 or np.all(strat_ret == 0):
        return {'sharpe': 0, 'total_return': 0, 'max_dd': 0, 'calmar': 0,
                'win_rate': 0, 'exposure': 0, 'n_days': 0, 'trades': 0,
                'daily_returns': []}

    cum = (1 + strat_ret).cumprod()
    total_ret = cum[-1] - 1
    n_years = len(strat_ret) / 365
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    active = strat_ret[strat_ret != 0]
    daily_mean = np.mean(active) if len(active) > 0 else 0
    daily_std = np.std(active) if len(active) > 0 else 1
    sharpe = (daily_mean / daily_std) * np.sqrt(365) if daily_std > 0 else 0

    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    winning = np.sum(strat_ret > 0)
    total_active = np.sum(strat_ret != 0)
    win_rate = winning / total_active if total_active > 0 else 0

    exposure = (pos != 0).mean()
    trades = int(np.count_nonzero(np.diff(pos)))

    return {
        'sharpe': round(float(sharpe), 4),
        'total_return': round(float(total_ret * 100), 2),
        'annual_return': round(float(ann_ret * 100), 2),
        'max_dd': round(float(max_dd * 100), 2),
        'calmar': round(float(calmar), 3),
        'win_rate': round(float(win_rate * 100), 1),
        'exposure': round(float(exposure * 100), 1),
        'n_days': int(len(strat_ret)),
        'trades': trades,
        'daily_returns': strat_ret.tolist(),
    }


# ── Walk-Forward + Permutation Test ─────────────────────────────────────────

def expanding_walkforward(positions, fwd_returns, n_folds=14):
    """14-fold expanding walk-forward. Returns OOS daily returns."""
    n = len(positions)
    min_train = max(60, n // (n_folds + 2))
    fold_size = (n - min_train) // n_folds

    oos_rets = []
    fold_details = []

    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = min(oos_start + fold_size, n)
        if oos_start >= n:
            break

        fold_pos = positions[oos_start:oos_end]
        fold_fwd = fwd_returns[oos_start:oos_end]

        # Apply transaction costs
        pos_changes = np.abs(np.diff(fold_pos))
        tc = np.concatenate([[0], pos_changes]) * TX_COST
        fold_ret = fold_pos * fold_fwd - tc

        valid = ~np.isnan(fold_ret)
        fold_ret = fold_ret[valid]

        oos_rets.extend(fold_ret)

        # Per-fold metrics
        if len(fold_ret) > 0 and np.std(fold_ret[fold_ret != 0]) > 0:
            active = fold_ret[fold_ret != 0]
            fold_sharpe = np.mean(active) / np.std(active) * np.sqrt(365) if len(active) > 0 else 0
        else:
            fold_sharpe = 0

        fold_cum = (1 + fold_ret).cumprod() if len(fold_ret) > 0 else np.array([1.0])
        fold_return = fold_cum[-1] - 1 if len(fold_cum) > 0 else 0

        fold_details.append({
            'fold': i + 1,
            'oos_start_idx': int(oos_start),
            'oos_end_idx': int(oos_end),
            'n_days': int(len(fold_ret)),
            'sharpe': round(float(fold_sharpe), 4),
            'return': round(float(fold_return * 100), 2),
            'exposure': round(float((fold_pos[~np.isnan(fold_fwd)] != 0).mean() * 100), 1) if len(fold_pos) > 0 else 0,
        })

    return np.array(oos_rets), fold_details


def permutation_test(positions, fwd_returns, n_perms=500, n_folds=14):
    """Permutation test on walk-forward OOS returns."""
    real_oos, _ = expanding_walkforward(positions, fwd_returns, n_folds)
    real_oos = real_oos[~np.isnan(real_oos)]
    if len(real_oos) == 0:
        return 0.0, 1.0

    active = real_oos[real_oos != 0]
    if len(active) < 2:
        return 0.0, 1.0

    real_sharpe = np.mean(active) / np.std(active) * np.sqrt(365)

    count = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(fwd_returns)
        perm_oos, _ = expanding_walkforward(positions, shuffled, n_folds)
        perm_oos = perm_oos[~np.isnan(perm_oos)]
        perm_active = perm_oos[perm_oos != 0]
        if len(perm_active) > 1:
            perm_sharpe = np.mean(perm_active) / np.std(perm_active) * np.sqrt(365)
            if perm_sharpe >= real_sharpe:
                count += 1

    return real_sharpe, count / n_perms


def bootstrap_ci(returns, n_boot=1000, ci=0.95):
    """Bootstrap CI on Sharpe."""
    active = returns[returns != 0]
    if len(active) < 2:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(active, size=len(active), replace=True)
        std = np.std(sample)
        if std > 0:
            sharpes.append(np.mean(sample) / std * np.sqrt(365))
    if not sharpes:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return {"mean": round(float(np.mean(sharpes)), 4),
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4)}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("SMA50 AND Weighted Regime Intersection Test")
    print("=" * 65)

    print("\nLoading BTC derivatives data from DuckDB...")
    df = load_data('BTC')
    df = compute_weighted_regime(df)
    df = compute_signals(df)
    df = df.dropna(subset=['fwd_returns', 'sma50', 'weighted_score']).reset_index(drop=True)

    print(f"Data: {df['date'].min().date()} to {df['date'].max().date()}, n={len(df)}")

    fwd = df['fwd_returns'].values

    # ── Test all three variants ──────────────────────────────────────────────
    variants = {
        'SMA50_only': df['sig_sma50'].values,
        'Regime_only': df['sig_regime'].values,
        'INTERSECTION': df['sig_intersection'].values,
        'BuyHold': np.ones(len(df)),
    }

    print(f"\n{'Variant':<20} {'TotRet%':>8} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'Exp%':>6} {'Trades':>7} | {'WF_Sharpe':>10} {'p-val':>7}")
    print("-" * 100)

    all_results = {}

    for name, pos in variants.items():
        metrics = compute_metrics(pos, fwd)

        # Walk-forward + permutation test
        print(f"  Running WF + permutation for {name}...", end="", flush=True)
        wf_sharpe, p_val = permutation_test(pos, fwd, N_PERMS, N_FOLDS)
        _, fold_details = expanding_walkforward(pos, fwd, N_FOLDS)
        print(f" done")

        # Bootstrap CI
        oos_rets, _ = expanding_walkforward(pos, fwd, N_FOLDS)
        oos_active = oos_rets[oos_rets != 0] if len(oos_rets) > 0 else oos_rets
        boot = bootstrap_ci(oos_active)

        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''

        result = {
            'full_sample': {k: v for k, v in metrics.items() if k != 'daily_returns'},
            'wf_sharpe': round(float(wf_sharpe), 4),
            'p_value': round(float(p_val), 4),
            'bootstrap_ci': boot,
            'fold_details': fold_details,
            'positive_folds': sum(1 for f in fold_details if f['sharpe'] > 0),
            'total_folds': len(fold_details),
        }
        all_results[name] = result

        print(f"  {name:<20} {metrics['total_return']:>8.1f} {metrics['annual_return']:>8.1f} "
              f"{metrics['sharpe']:>7.3f} {metrics['max_dd']:>7.1f} {metrics['exposure']:>6.1f} "
              f"{metrics['trades']:>7d} | {wf_sharpe:>10.3f} {p_val:>7.4f} {sig}")

    # ── Detailed Fold-by-Fold for Intersection ───────────────────────────────
    print(f"\n{'='*65}")
    print("INTERSECTION FOLD-BY-FOLD DETAIL")
    print(f"{'='*65}")

    inter_folds = all_results['INTERSECTION']['fold_details']
    print(f"{'Fold':>4} | {'Days':>5} | {'Sharpe':>7} | {'Return%':>8} | {'Exposure%':>9}")
    print("-" * 50)
    for f in inter_folds:
        print(f"  {f['fold']:2d} | {f['n_days']:5d} | {f['sharpe']:+7.2f} | {f['return']:+8.1f} | {f['exposure']:8.1f}")

    pos_folds = all_results['INTERSECTION']['positive_folds']
    total_folds = all_results['INTERSECTION']['total_folds']
    fold_sharpes = [f['sharpe'] for f in inter_folds]

    print(f"\nMean fold Sharpe: {np.mean(fold_sharpes):.4f}")
    print(f"Median fold Sharpe: {np.median(fold_sharpes):.4f}")
    print(f"Positive folds: {pos_folds}/{total_folds}")

    # ── Summary Comparison ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY COMPARISON")
    print(f"{'='*65}")
    print(f"\n{'Variant':<20} {'WF Sharpe':>10} {'p-value':>10} {'95% CI':>20} {'Pos Folds':>10}")
    print("-" * 75)
    for name, r in all_results.items():
        ci = r['bootstrap_ci']
        ci_str = f"[{ci['lower']:.2f}, {ci['upper']:.2f}]"
        pf = f"{r['positive_folds']}/{r['total_folds']}"
        print(f"  {name:<20} {r['wf_sharpe']:>8.3f} {r['p_value']:>10.4f} {ci_str:>20} {pf:>10}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'test': 'SMA50 AND Weighted Regime Intersection',
            'symbol': 'BTC',
            'n_permutations': N_PERMS,
            'n_folds': N_FOLDS,
            'tx_cost': TX_COST,
            'annualization': 'sqrt(365)',
            'note': 'Standalone test — no Bonferroni needed (single strategy)',
        },
        'results': {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
    }

    # Remove daily_returns from full_sample to keep JSON manageable
    for v in output['results'].values():
        if 'daily_returns' in v.get('full_sample', {}):
            del v['full_sample']['daily_returns']

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")
