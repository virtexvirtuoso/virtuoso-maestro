#!/usr/bin/env python3
"""
Quick re-validation of top strategies with full multi-metric reporting.
Runs EWMAC + Dual Momentum on top assets at 4h/1d, outputs full metric suite.
"""
import sys
sys.path.insert(0, '/Users/ffv_macmini/Desktop/maestro/backend/research')

import pandas as pd
import numpy as np
from pathlib import Path
from metrics import compute_metrics, format_metrics_summary, grade_strategy

SPOT_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/spot")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/backend/research/results")
COST_BPS = 10
N_FOLDS = 14
MIN_TRADES = 30

TOP_ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'FTM', 'ARB', 'RENDER', 'TIA', 'SEI', 'DYDX', 'INJ']
TIMEFRAMES = {'4h': 365.25 * 6, '1d': 365.25}

def load_spot(asset, tf):
    path = SPOT_DIR / tf / f"{asset}_spot_{tf}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for c in df.columns:
        cl = c.lower()
        if cl in ('timestamp', 'date', 'datetime'):
            df.rename(columns={c: 'timestamp'}, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def ewmac_signals(df, fast=8, slow=32):
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    diff = ema_fast - ema_slow
    vol = df['close'].pct_change().rolling(32).std()
    normalized = diff / (df['close'] * vol)
    signals = pd.Series(0, index=df.index)
    signals[normalized > 0] = 1
    signals[normalized < 0] = -1
    signals.iloc[:slow] = 0
    return signals

def dual_momentum_signals(df, lookback=63, abs_threshold=0.0):
    ret = df['close'].pct_change(lookback)
    signals = pd.Series(0, index=df.index)
    signals[ret > abs_threshold] = 1
    signals[ret < -abs_threshold] = -1
    signals.iloc[:lookback] = 0
    return signals

def walk_forward_test(df, signals, holding_bars, bars_per_year):
    n = len(df)
    fold_size = n // N_FOLDS
    if fold_size < 20:
        return None
    
    test_start = fold_size * 7
    test_df = df.iloc[test_start:]
    test_sig = signals.iloc[test_start:]
    
    trades = []
    i = 0
    while i < len(test_df) - holding_bars:
        sig = test_sig.iloc[i]
        if sig != 0:
            entry = test_df['close'].iloc[i]
            exit_ = test_df['close'].iloc[i + holding_bars]
            ret = (exit_ / entry - 1) * sig - COST_BPS / 10000
            trades.append(ret)
            i += holding_bars
        else:
            i += 1
    
    if len(trades) < MIN_TRADES:
        return None
    
    returns = np.array(trades)
    test_bars = len(test_df)
    test_years = test_bars / bars_per_year
    tpy = len(trades) / test_years if test_years > 0 else len(trades)
    
    return compute_metrics(returns, bars_per_year=bars_per_year,
                          total_bars=test_bars, trades_per_year=tpy)

STRATEGIES = {
    'EWMAC_8_32': lambda df: ewmac_signals(df, 8, 32),
    'EWMAC_16_64': lambda df: ewmac_signals(df, 16, 64),
    'DualMom_63': lambda df: dual_momentum_signals(df, 63),
    'DualMom_126': lambda df: dual_momentum_signals(df, 126),
}

HOLDINGS = {
    '4h': {'1d': 6, '3d': 18, '1w': 42},
    '1d': {'3d': 3, '1w': 7, '2w': 14},
}

print("=" * 100)
print("MULTI-METRIC STRATEGY REVALIDATION")
print("=" * 100)

all_results = []
for tf, bpy in TIMEFRAMES.items():
    for asset in TOP_ASSETS:
        df = load_spot(asset, tf)
        if df is None:
            continue
        for strat_name, strat_fn in STRATEGIES.items():
            signals = strat_fn(df)
            for hold_name, hold_bars in HOLDINGS[tf].items():
                m = walk_forward_test(df, signals, hold_bars, bpy)
                if m is None:
                    continue
                grade = grade_strategy(m)
                m['asset'] = asset
                m['timeframe'] = tf
                m['strategy'] = strat_name
                m['holding'] = hold_name
                m['grade'] = grade
                all_results.append(m)
                
                if m['sharpe'] > 0.5 or m['sharpe'] < -2:
                    icon = {'A': '🏆', 'B': '⭐', 'C': '📊', 'D': '📉', 'F': '💀'}[grade]
                    print(f"{icon} [{grade}] {asset} {tf} {strat_name} hold={hold_name}: {format_metrics_summary(m)}")

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / 'multimetric_top_strategies.csv', index=False)

print(f"\n{'=' * 100}")
print(f"TOTAL: {len(all_results)} tests")
print(f"\nGrade distribution:")
for g in ['A', 'B', 'C', 'D', 'F']:
    count = len([r for r in all_results if r['grade'] == g])
    pct = count / len(all_results) * 100 if all_results else 0
    print(f"  {g}: {count} ({pct:.1f}%)")

print(f"\n--- TOP 15 BY SHARPE ---")
top = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:15]
print(f"{'Asset':<8} {'TF':<4} {'Strategy':<14} {'Hold':<5} {'Gr':>2} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MDD':>8} {'PF':>6} {'WR':>5} {'Payoff':>7} {'Tail':>6} {'Skew':>6} {'Trades':>6}")
print("-" * 100)
for r in top:
    print(f"{r['asset']:<8} {r['timeframe']:<4} {r['strategy']:<14} {r['holding']:<5} {r['grade']:>2} "
          f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['max_drawdown']:>7.1%} "
          f"{r['profit_factor']:>6.2f} {r['win_rate']:>4.0%} {r['payoff_ratio']:>7.2f} {r['tail_ratio']:>6.2f} "
          f"{r['skewness']:>6.2f} {r['n_trades']:>6}")

print(f"\n--- TOP 15 BY SORTINO ---")
top_sort = sorted(all_results, key=lambda x: x['sortino'], reverse=True)[:15]
for r in top_sort:
    print(f"{r['asset']:<8} {r['timeframe']:<4} {r['strategy']:<14} {r['holding']:<5} [{r['grade']}] "
          f"Sortino={r['sortino']:>7.2f}  Sharpe={r['sharpe']:>6.2f}  Calmar={r['calmar']:>6.2f}  MDD={r['max_drawdown']:>7.1%}")

print(f"\n--- WORST 10 BY MAX DRAWDOWN ---")
worst_dd = sorted(all_results, key=lambda x: x['max_drawdown'])[:10]
for r in worst_dd:
    print(f"{r['asset']:<8} {r['timeframe']:<4} {r['strategy']:<14} {r['holding']:<5} "
          f"MDD={r['max_drawdown']:>7.1%}  Sharpe={r['sharpe']:>6.2f}  Calmar={r['calmar']:>6.2f}  DDdur={r['max_dd_duration']}")

print(f"\n--- STRATEGY AVERAGES ---")
for strat in STRATEGIES:
    subset = [r for r in all_results if r['strategy'] == strat]
    if not subset:
        continue
    avg_sharpe = np.mean([r['sharpe'] for r in subset])
    avg_sortino = np.mean([r['sortino'] for r in subset])
    avg_calmar = np.mean([r['calmar'] for r in subset])
    avg_mdd = np.mean([r['max_drawdown'] for r in subset])
    avg_pf = np.mean([r['profit_factor'] for r in subset])
    avg_wr = np.mean([r['win_rate'] for r in subset])
    grades = [r['grade'] for r in subset]
    print(f"  {strat:<14} Sharpe={avg_sharpe:>6.2f} Sortino={avg_sortino:>7.2f} Calmar={avg_calmar:>6.2f} "
          f"MDD={avg_mdd:>7.1%} PF={avg_pf:>5.2f} WR={avg_wr:>4.0%} "
          f"Grades: A={grades.count('A')} B={grades.count('B')} C={grades.count('C')} D={grades.count('D')} F={grades.count('F')}")

print(f"\nResults saved to: {RESULTS_DIR / 'multimetric_top_strategies.csv'}")
