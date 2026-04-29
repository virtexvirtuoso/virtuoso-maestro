"""
V4 Honest System — Maestro Research Platform
==============================================
Based on 2,300+ walk-forward assumption tests.

Layer 1: SMA50 trend filter (long only, no shorts)
Layer 2: Risk management (trailing stops, vol ceiling, DD breaker)
Layer 3: Portfolio construction (BTC/ETH/SOL equal-weight, spot only)

All signals use bar N signal → bar N+1 execution (no look-ahead).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import json
import os
from datetime import datetime


# ─── Configuration ───────────────────────────────────────────────────────────

ASSETS = ['BTC', 'ETH', 'SOL']
WEIGHTS = {a: 1/3 for a in ASSETS}
SMA_WINDOW = 50

# Per-asset trailing stop distances (from V3.2 vol profile research)
TRAILING_STOPS = {'BTC': 0.12, 'ETH': 0.15, 'SOL': 0.08}

# Vol ceiling: halve position when 30d realized vol > 80% annualized
VOL_LOOKBACK = 30
VOL_CEILING = 0.80  # annualized

# Transaction costs (matching V3's 10bps per trade)
TX_COST = 0.001

# Drawdown breaker
MAX_PORTFOLIO_DD = 0.25

# Walk-forward config
N_FOLDS = 14
MIN_TRAIN_DAYS = 365

# Statistical tests
N_PERMUTATIONS = 500
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95


# ─── Core Signal Generation ─────────────────────────────────────────────────

def sma_signal(close: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    """SMA trend filter. 1 = long, 0 = flat. Signal on bar N, NO execution yet."""
    sma = close.rolling(window).mean()
    signal = (close > sma).astype(float)
    signal.iloc[:window] = 0  # no signal during warmup
    return signal


def apply_no_lookahead(signal: pd.Series) -> pd.Series:
    """Shift signal by 1 bar: signal from bar N executes on bar N+1."""
    return signal.shift(1).fillna(0)


# ─── Risk Management Layers ─────────────────────────────────────────────────

def trailing_stop(close: pd.Series, position: pd.Series, stop_pct: float) -> pd.Series:
    """Per-asset trailing stop. Exits when price drops stop_pct from peak while in position."""
    result = position.copy()
    peak = close.iloc[0]
    in_position = False
    stopped_out = False

    for i in range(len(close)):
        if position.iloc[i] > 0 and not stopped_out:
            if not in_position:
                peak = close.iloc[i]
                in_position = True
            else:
                peak = max(peak, close.iloc[i])

            if close.iloc[i] < peak * (1 - stop_pct):
                result.iloc[i] = 0
                stopped_out = True
                in_position = False
        elif position.iloc[i] > 0 and stopped_out:
            # Stay out until position signal goes to 0 then back to 1 (new cross)
            result.iloc[i] = 0
        else:
            in_position = False
            stopped_out = False
            peak = close.iloc[i]

    return result


def vol_ceiling(close: pd.Series, position: pd.Series,
                lookback: int = VOL_LOOKBACK, ceiling: float = VOL_CEILING) -> pd.Series:
    """Halve position when realized vol exceeds ceiling."""
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(lookback).std() * np.sqrt(365)
    result = position.copy()
    high_vol = realized_vol > ceiling
    result[high_vol] = result[high_vol] * 0.5
    return result


def drawdown_breaker(portfolio_equity: pd.Series, positions: Dict[str, pd.Series],
                     max_dd: float = MAX_PORTFOLIO_DD,
                     closes: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
    """Go flat if portfolio drawdown exceeds threshold. Re-entry on new SMA50 cross."""
    running_max = portfolio_equity.expanding().max()
    drawdown = (portfolio_equity - running_max) / running_max
    breaker_active = drawdown < -max_dd

    result = {a: pos.copy() for a, pos in positions.items()}

    # Track breaker state
    breaker_on = False
    prev_signals = {a: 0.0 for a in positions}

    for i in range(len(portfolio_equity)):
        if breaker_active.iloc[i]:
            breaker_on = True
            for a in result:
                result[a].iloc[i] = 0
                prev_signals[a] = 0.0
        elif breaker_on:
            # Need new SMA50 cross to re-enter (signal goes 0→1)
            all_clear = True
            for a in result:
                curr = positions[a].iloc[i]
                if curr > 0 and prev_signals[a] == 0:
                    # New cross detected for this asset
                    pass  # allow re-entry
                elif curr > 0 and prev_signals[a] > 0:
                    # Continuing old signal, block
                    result[a].iloc[i] = 0
                    all_clear = False
                prev_signals[a] = positions[a].iloc[i]

            # Check if drawdown recovered
            if drawdown.iloc[i] > -max_dd * 0.5:  # recovered to half the threshold
                breaker_on = False
        else:
            for a in positions:
                prev_signals[a] = positions[a].iloc[i]

    return result


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def compute_returns(close: pd.Series, position: pd.Series, tx_cost: float = TX_COST) -> pd.Series:
    """Compute strategy returns with transaction costs deducted on position changes."""
    daily_ret = close.pct_change().fillna(0)
    pos_change = position.diff().abs().fillna(0)
    return position.shift(0) * daily_ret - pos_change * tx_cost


def compute_equity(returns: pd.Series) -> pd.Series:
    """Cumulative equity curve from returns."""
    return (1 + returns).cumprod()


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(365)


def cagr(equity: pd.Series, n_days: int) -> float:
    """Compound annual growth rate."""
    if equity.iloc[0] == 0 or n_days == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = n_days / 365.25
    if years == 0:
        return 0.0
    return total_return ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    running_max = equity.expanding().max()
    dd = (equity - running_max) / running_max
    return dd.min()


def calmar_ratio(cagr_val: float, mdd: float) -> float:
    if mdd == 0:
        return 0.0
    return cagr_val / abs(mdd)


# ─── Variant Runners ─────────────────────────────────────────────────────────

def run_variant(variant: str, closes: Dict[str, pd.Series],
                common_index: pd.DatetimeIndex) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Run a strategy variant on aligned data.
    Returns (portfolio_returns, per_asset_returns_dict).
    """
    asset_returns = {}
    asset_positions = {}

    for asset in ASSETS:
        close = closes[asset].reindex(common_index).ffill()
        raw_signal = sma_signal(close)
        position = apply_no_lookahead(raw_signal)

        if variant in ('V4b', 'V4c', 'V4d'):
            position = trailing_stop(close, position, TRAILING_STOPS[asset])

        if variant in ('V4c', 'V4d'):
            position = vol_ceiling(close, position)

        asset_positions[asset] = position
        # Weight
        position = position * WEIGHTS[asset]
        asset_returns[asset] = compute_returns(close, position)

    # Portfolio returns (before DD breaker)
    port_ret = sum(asset_returns[a] for a in ASSETS)

    if variant == 'V4d':
        # Apply drawdown breaker at portfolio level
        port_equity = compute_equity(port_ret)
        adjusted_positions = drawdown_breaker(
            port_equity,
            {a: asset_positions[a] * WEIGHTS[a] for a in ASSETS},
            closes=closes
        )
        # Recompute returns with adjusted positions
        asset_returns = {}
        for asset in ASSETS:
            close = closes[asset].reindex(common_index).ffill()
            asset_returns[asset] = compute_returns(close, adjusted_positions[asset])
        port_ret = sum(asset_returns[a] for a in ASSETS)

    return port_ret, asset_returns


def run_buyhold(closes: Dict[str, pd.Series], common_index: pd.DatetimeIndex) -> pd.Series:
    """Equal-weight buy-and-hold baseline."""
    port_ret = pd.Series(0.0, index=common_index)
    for asset in ASSETS:
        close = closes[asset].reindex(common_index).ffill()
        ret = close.pct_change().fillna(0) * WEIGHTS[asset]
        port_ret += ret
    return port_ret


# ─── Walk-Forward Engine ─────────────────────────────────────────────────────

def walk_forward_folds(common_index: pd.DatetimeIndex, n_folds: int = N_FOLDS,
                       min_train: int = MIN_TRAIN_DAYS) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate expanding-window walk-forward folds."""
    n = len(common_index)
    test_size = (n - min_train) // n_folds
    folds = []

    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)
        if test_start >= n:
            break
        folds.append((
            common_index[:train_end],
            common_index[test_start:test_end]
        ))

    return folds


# ─── Statistical Tests ───────────────────────────────────────────────────────

def permutation_test(returns: pd.Series, n_perms: int = N_PERMUTATIONS) -> float:
    """Permutation test: p-value for Sharpe ratio."""
    observed = sharpe_ratio(returns)
    count = 0
    ret_arr = returns.values.copy()

    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        rng.shuffle(ret_arr)
        if sharpe_ratio(pd.Series(ret_arr)) >= observed:
            count += 1

    return (count + 1) / (n_perms + 1)


def bootstrap_ci(returns: pd.Series, n_boot: int = N_BOOTSTRAP,
                 ci: float = BOOTSTRAP_CI) -> Dict:
    """Bootstrap confidence intervals for Sharpe, CAGR, MaxDD."""
    n = len(returns)
    rng = np.random.RandomState(42)
    sharpes, cagrs, mdds = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_ret = returns.iloc[idx].reset_index(drop=True)
        boot_eq = compute_equity(boot_ret)
        sharpes.append(sharpe_ratio(boot_ret))
        cagrs.append(cagr(boot_eq, n))
        mdds.append(max_drawdown(boot_eq))

    alpha = (1 - ci) / 2
    lo = alpha
    hi = 1 - alpha

    return {
        'sharpe': (float(np.percentile(sharpes, lo*100)), float(np.percentile(sharpes, hi*100))),
        'cagr': (float(np.percentile(cagrs, lo*100)), float(np.percentile(cagrs, hi*100))),
        'max_dd': (float(np.percentile(mdds, lo*100)), float(np.percentile(mdds, hi*100))),
    }


# ─── Main Backtest ───────────────────────────────────────────────────────────

def load_data() -> Dict[str, pd.DataFrame]:
    """Load spot daily CSVs."""
    data = {}
    base = os.path.expanduser('~/Desktop/maestro/data/spot/')
    for asset in ASSETS:
        path = os.path.join(base, f'{asset}_spot_daily.csv')
        df = pd.read_csv(path, index_col='date', parse_dates=True)
        # Handle yfinance multi-level columns if present
        if 'Close' in df.columns:
            df.columns = [c.lower() for c in df.columns]
        data[asset] = df
    return data


def run_full_backtest():
    """Run the complete V4 Honest System backtest."""
    print("=" * 80)
    print("V4 HONEST SYSTEM — FULL BACKTEST")
    print("=" * 80)

    # Load data
    data = load_data()
    closes = {a: data[a]['close'] for a in ASSETS}

    # Common index (intersection of all assets)
    common_index = closes['BTC'].index
    for a in ASSETS[1:]:
        common_index = common_index.intersection(closes[a].index)
    common_index = common_index.sort_values()
    print(f"\nCommon date range: {common_index[0].date()} to {common_index[-1].date()} ({len(common_index)} days)")

    # Walk-forward folds
    folds = walk_forward_folds(common_index)
    print(f"Walk-forward folds: {len(folds)}")

    variants = ['V4a', 'V4b', 'V4c', 'V4d']
    results = {}

    for variant in variants:
        print(f"\n{'─' * 60}")
        print(f"Running {variant}...")

        # Full-sample backtest
        port_ret, asset_rets = run_variant(variant, closes, common_index)
        port_eq = compute_equity(port_ret)

        full_metrics = {
            'sharpe': sharpe_ratio(port_ret),
            'cagr': cagr(port_eq, len(common_index)),
            'max_dd': max_drawdown(port_eq),
            'calmar': calmar_ratio(cagr(port_eq, len(common_index)), max_drawdown(port_eq)),
            'total_return': float(port_eq.iloc[-1] - 1),
        }

        # Per-asset metrics
        per_asset = {}
        for a in ASSETS:
            a_eq = compute_equity(asset_rets[a])
            per_asset[a] = {
                'sharpe': sharpe_ratio(asset_rets[a]),
                'cagr': cagr(a_eq, len(common_index)),
                'max_dd': max_drawdown(a_eq),
                'total_return': float(a_eq.iloc[-1] - 1),
            }

        # Time-period stability analysis
        # NOTE: This is NOT true walk-forward optimization. Parameters (SMA_WINDOW=50,
        # TRAILING_STOPS, VOL_CEILING, etc.) are fixed across all folds — no per-fold
        # optimization occurs. This measures temporal stability of a fixed-parameter
        # system, not out-of-sample generalization from optimized parameters.
        oos_returns = []
        fold_results = []
        for fi, (train_idx, test_idx) in enumerate(folds):
            # Fixed params applied to full dataset, returns sliced by test period
            full_port_ret, _ = run_variant(variant, closes, common_index)
            test_ret = full_port_ret.reindex(test_idx)
            oos_returns.append(test_ret)
            test_eq = compute_equity(test_ret)

            fold_results.append({
                'fold': fi + 1,
                'train_end': str(train_idx[-1].date()),
                'test_start': str(test_idx[0].date()),
                'test_end': str(test_idx[-1].date()),
                'test_days': len(test_idx),
                'sharpe': sharpe_ratio(test_ret),
                'return': float(test_eq.iloc[-1] - 1) if len(test_eq) > 0 else 0,
                'max_dd': max_drawdown(test_eq) if len(test_eq) > 0 else 0,
            })

        # Concatenate OOS returns
        oos_all = pd.concat(oos_returns)
        oos_eq = compute_equity(oos_all)
        oos_metrics = {
            'sharpe': sharpe_ratio(oos_all),
            'cagr': cagr(oos_eq, len(oos_all)),
            'max_dd': max_drawdown(oos_eq),
            'total_return': float(oos_eq.iloc[-1] - 1),
        }

        # Permutation tests
        print(f"  Running permutation tests (portfolio)...")
        perm_p_portfolio = permutation_test(oos_all)
        perm_p_assets = {}
        for a in ASSETS:
            a_oos = pd.concat([full_port_ret.reindex(test_idx) for _, test_idx in folds])
            # Use per-asset returns for per-asset permutation
            full_port_ret_a, full_asset_rets_a = run_variant(variant, closes, common_index)
            a_full_ret = full_asset_rets_a[a]
            a_oos_ret = pd.concat([a_full_ret.reindex(test_idx) for _, test_idx in folds])
            perm_p_assets[a] = permutation_test(a_oos_ret)

        # Bootstrap CIs
        print(f"  Running bootstrap CIs...")
        boot_ci = bootstrap_ci(oos_all)

        results[variant] = {
            'full_sample': full_metrics,
            'per_asset': per_asset,
            'oos': oos_metrics,
            'folds': fold_results,
            'permutation_p': {'portfolio': perm_p_portfolio, **perm_p_assets},
            'bootstrap_ci': boot_ci,
        }

    # Buy-and-hold baseline
    print(f"\n{'─' * 60}")
    print("Running Buy & Hold baseline...")
    bh_ret = run_buyhold(closes, common_index)
    bh_eq = compute_equity(bh_ret)
    bh_metrics = {
        'sharpe': sharpe_ratio(bh_ret),
        'cagr': cagr(bh_eq, len(common_index)),
        'max_dd': max_drawdown(bh_eq),
        'total_return': float(bh_eq.iloc[-1] - 1),
    }
    results['BuyHold'] = {'full_sample': bh_metrics}

    # Year-by-year attribution
    print(f"\n{'─' * 60}")
    print("Computing year-by-year attribution...")
    yearly = {}
    for year in range(common_index[0].year, common_index[-1].year + 1):
        year_mask = common_index.year == year
        year_idx = common_index[year_mask]
        if len(year_idx) < 30:
            continue
        yearly[year] = {}
        for variant in variants:
            full_ret, _ = run_variant(variant, closes, common_index)
            yr = full_ret.reindex(year_idx)
            yr_eq = compute_equity(yr)
            yearly[year][variant] = {
                'return': float(yr_eq.iloc[-1] - 1),
                'sharpe': sharpe_ratio(yr),
                'max_dd': max_drawdown(yr_eq),
            }
        # BH
        bh_yr = bh_ret.reindex(year_idx)
        bh_yr_eq = compute_equity(bh_yr)
        yearly[year]['BuyHold'] = {
            'return': float(bh_yr_eq.iloc[-1] - 1),
            'sharpe': sharpe_ratio(bh_yr),
            'max_dd': max_drawdown(bh_yr_eq),
        }

    results['yearly'] = yearly

    # ─── Print Results ───────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("COMPARISON TABLE: V4a / V4b / V4c / V4d / Buy&Hold")
    print("=" * 80)
    header = f"{'Metric':<25} {'V4a':>10} {'V4b':>10} {'V4c':>10} {'V4d':>10} {'BuyHold':>10}"
    print(header)
    print("─" * len(header))

    for metric_name, metric_key in [('Sharpe (full)', 'sharpe'), ('CAGR (full)', 'cagr'),
                                      ('MaxDD (full)', 'max_dd'), ('Total Return', 'total_return'),
                                      ('Calmar', 'calmar')]:
        row = f"{metric_name:<25}"
        for v in variants:
            val = results[v]['full_sample'].get(metric_key, 0)
            if metric_key in ('cagr', 'max_dd', 'total_return'):
                row += f" {val:>9.1%}"
            else:
                row += f" {val:>10.2f}"
        # BH
        val = results['BuyHold']['full_sample'].get(metric_key, 0)
        if metric_key in ('cagr', 'max_dd', 'total_return'):
            row += f" {val:>9.1%}"
        else:
            row += f" {val:>10.2f}"
        print(row)

    print(f"\n{'OOS Metrics (walk-forward)':<25}")
    print("─" * len(header))
    for metric_name, metric_key in [('OOS Sharpe', 'sharpe'), ('OOS CAGR', 'cagr'),
                                      ('OOS MaxDD', 'max_dd')]:
        row = f"{metric_name:<25}"
        for v in variants:
            val = results[v]['oos'].get(metric_key, 0)
            if metric_key in ('cagr', 'max_dd'):
                row += f" {val:>9.1%}"
            else:
                row += f" {val:>10.2f}"
        row += f" {'N/A':>10}"
        print(row)

    print(f"\n{'Statistical Tests':<25}")
    print("─" * len(header))
    row = f"{'Perm. p-val (port.)':<25}"
    for v in variants:
        row += f" {results[v]['permutation_p']['portfolio']:>10.4f}"
    row += f" {'N/A':>10}"
    print(row)

    for a in ASSETS:
        row = f"{'Perm. p-val (' + a + ')':<25}"
        for v in variants:
            row += f" {results[v]['permutation_p'].get(a, 0):>10.4f}"
        row += f" {'N/A':>10}"
        print(row)

    print(f"\n{'Bootstrap 95% CIs (V4d)':<25}")
    print("─" * 40)
    for metric in ['sharpe', 'cagr', 'max_dd']:
        lo, hi = results['V4d']['bootstrap_ci'][metric]
        if metric in ('cagr', 'max_dd'):
            print(f"  {metric:<15} [{lo:>8.1%}, {hi:>8.1%}]")
        else:
            print(f"  {metric:<15} [{lo:>8.2f}, {hi:>8.2f}]")

    # Per-asset table
    print(f"\n{'=' * 60}")
    print("PER-ASSET RESULTS (Full V4d)")
    print("=" * 60)
    for a in ASSETS:
        m = results['V4d']['per_asset'][a]
        print(f"  {a}: Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.1%}  MaxDD={m['max_dd']:.1%}  Return={m['total_return']:.1%}")

    # Walk-forward fold-by-fold
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD FOLD-BY-FOLD (V4d)")
    print("=" * 80)
    print(f"{'Fold':>4} {'Train→':>12} {'Test Period':>25} {'Days':>5} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8}")
    print("─" * 75)
    for f in results['V4d']['folds']:
        print(f"{f['fold']:>4} {f['train_end']:>12} {f['test_start']+' → '+f['test_end']:>25} {f['test_days']:>5} {f['sharpe']:>8.2f} {f['return']:>7.1%} {f['max_dd']:>7.1%}")

    # Year-by-year
    print(f"\n{'=' * 80}")
    print("YEAR-BY-YEAR RETURNS")
    print("=" * 80)
    print(f"{'Year':>6} {'V4a':>8} {'V4b':>8} {'V4c':>8} {'V4d':>8} {'BH':>8} {'V4d Edge':>10}")
    print("─" * 60)
    for year in sorted(yearly.keys()):
        y = yearly[year]
        edge = y['V4d']['return'] - y['BuyHold']['return']
        print(f"{year:>6} {y['V4a']['return']:>7.1%} {y['V4b']['return']:>7.1%} {y['V4c']['return']:>7.1%} {y['V4d']['return']:>7.1%} {y['BuyHold']['return']:>7.1%} {edge:>9.1%}")

    # Success criteria check
    print(f"\n{'=' * 80}")
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    v4d = results['V4d']
    checks = [
        ('OOS Sharpe >= 1.0', v4d['oos']['sharpe'] >= 1.0, f"{v4d['oos']['sharpe']:.2f}"),
        ('MaxDD <= -30%', v4d['full_sample']['max_dd'] >= -0.30, f"{v4d['full_sample']['max_dd']:.1%}"),
        ('Perm p < 0.05', v4d['permutation_p']['portfolio'] < 0.05, f"p={v4d['permutation_p']['portfolio']:.4f}"),
        ('Bootstrap CI excludes 0 (Sharpe)', v4d['bootstrap_ci']['sharpe'][0] > 0, f"[{v4d['bootstrap_ci']['sharpe'][0]:.2f}, {v4d['bootstrap_ci']['sharpe'][1]:.2f}]"),
        ('V4d Sharpe > V4a Sharpe', v4d['oos']['sharpe'] > results['V4a']['oos']['sharpe'], f"V4d={v4d['oos']['sharpe']:.2f} vs V4a={results['V4a']['oos']['sharpe']:.2f}"),
    ]
    for name, passed, detail in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}: {detail}")

    # Save JSON
    out_path = os.path.expanduser('~/Desktop/maestro/data/backtest_results/v4_honest_system.json')

    # Convert for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        return obj

    save_results = make_serializable(results)
    save_results['metadata'] = {
        'generated': datetime.now().isoformat(),
        'assets': ASSETS,
        'sma_window': SMA_WINDOW,
        'trailing_stops': TRAILING_STOPS,
        'vol_ceiling': VOL_CEILING,
        'max_dd_breaker': MAX_PORTFOLIO_DD,
        'n_folds': N_FOLDS,
        'n_permutations': N_PERMUTATIONS,
        'common_days': len(common_index),
        'date_range': [str(common_index[0].date()), str(common_index[-1].date())],
    }

    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n📁 Results saved to: {out_path}")
    print(f"📁 Strategy saved to: {os.path.abspath(__file__)}")

    return results


if __name__ == '__main__':
    run_full_backtest()
