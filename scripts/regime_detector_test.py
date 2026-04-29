#!/usr/bin/env python3
"""
Multi-Signal Regime Detector — Full Backtest
Combines LSR, Funding, Liquidations, Taker Volume into composite score (0-4).
Tests 6 variants with 14-fold expanding walk-forward + 500 permutation tests.
"""

import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
OUTPUT_PATH = os.path.expanduser("~/Desktop/maestro/data/backtest_results/regime_detector_test.json")
SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB']
N_PERMS = 500
N_FOLDS = 14
BONFERRONI_VARIANTS = 6  # number of variants tested

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(symbol):
    """Load and merge all data sources for a symbol."""
    con = duckdb.connect(DB_PATH, read_only=True)
    
    lsr = con.execute(f"SELECT date, global_account_long_short_ratio as lsr FROM cg_lsr_global WHERE symbol='{symbol}' ORDER BY date").df()
    fr = con.execute(f"SELECT date, close as funding_rate FROM cg_funding_rate WHERE symbol='{symbol}' ORDER BY date").df()
    liq = con.execute(f"SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as total_liq FROM cg_liquidations WHERE symbol='{symbol}' ORDER BY date").df()
    taker = con.execute(f"SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as taker_ratio FROM cg_taker_volume WHERE symbol='{symbol}' ORDER BY date").df()
    price = con.execute(f"SELECT date, open, high, low, close FROM perps_daily WHERE symbol='{symbol}' ORDER BY date").df()
    
    con.close()
    
    # Merge all on date (inner join)
    df = price.copy()
    for src in [lsr, fr, liq, taker]:
        df = df.merge(src, on='date', how='inner')
    
    df = df.sort_values('date').reset_index(drop=True)
    df['returns'] = df['close'].pct_change()
    df['fwd_returns'] = df['returns'].shift(-1)  # next-day return
    
    return df

def compute_regime_score(df, window=30):
    """Compute 0-4 regime score."""
    # LSR: +1 if < 50th percentile (30d rolling)
    lsr_med = df['lsr'].rolling(window, min_periods=10).median()
    df['lsr_score'] = (df['lsr'] < lsr_med).astype(int)
    
    # Funding: +1 if < 0.03%
    df['funding_score'] = (df['funding_rate'] < 0.03).astype(int)
    
    # Liquidations: +1 if < 80th percentile (30d rolling)
    liq_p80 = df['total_liq'].rolling(window, min_periods=10).quantile(0.8)
    df['liq_score'] = (df['total_liq'] < liq_p80).astype(int)
    
    # Taker: +1 if buy/sell ratio > 1.0
    df['taker_score'] = (df['taker_ratio'] > 1.0).astype(int)
    
    df['regime_score'] = df['lsr_score'] + df['funding_score'] + df['liq_score'] + df['taker_score']
    
    return df

def compute_weighted_regime(df, window=30):
    """Weighted regime with LSR/funding weighted higher."""
    lsr_med = df['lsr'].rolling(window, min_periods=10).median()
    lsr_s = (df['lsr'] < lsr_med).astype(float)
    fund_s = (df['funding_rate'] < 0.03).astype(float)
    liq_p80 = df['total_liq'].rolling(window, min_periods=10).quantile(0.8)
    liq_s = (df['total_liq'] < liq_p80).astype(float)
    taker_s = (df['taker_ratio'] > 1.0).astype(float)
    
    # Weights: LSR=0.35, Funding=0.35, Liq=0.15, Taker=0.15
    df['weighted_score'] = lsr_s * 0.35 + fund_s * 0.35 + liq_s * 0.15 + taker_s * 0.15
    return df

def compute_sma50_signal(df):
    """V4 base: SMA50 trend following with trailing stop."""
    df['sma50'] = df['close'].rolling(50, min_periods=50).mean()
    df['v4_signal'] = (df['close'] > df['sma50']).astype(float)
    
    # Simple trailing stop: exit if drawdown from peak > 8%
    peak = df['close'].expanding().max()
    dd = (df['close'] - peak) / peak
    df['trailing_stop'] = (dd > -0.08).astype(float)
    df['v4_signal'] = df['v4_signal'] * df['trailing_stop']
    
    return df

# ── Strategy Variants ─────────────────────────────────────────────────────────

def signal_regime_standalone(df):
    """Long when score>=3, half when 2, flat when <=1."""
    pos = np.where(df['regime_score'] >= 3, 1.0,
          np.where(df['regime_score'] == 2, 0.5, 0.0))
    return pos

def signal_regime_v4_overlay(df):
    """V4 base scaled by regime: 4=100%, 3=75%, 2=50%, 1=25%, 0=flat."""
    scale = df['regime_score'] / 4.0
    return df['v4_signal'].values * scale.values

def signal_regime_v4_riskoff(df):
    """V4 runs normally, override to flat only when score=0."""
    pos = df['v4_signal'].values.copy()
    pos[df['regime_score'].values == 0] = 0.0
    return pos

def signal_binary_regime(df):
    """Score>=2 = risk-on (full), <2 = flat."""
    return (df['regime_score'] >= 2).astype(float).values

def signal_weighted_regime(df):
    """Weighted score > 0.5 = risk-on."""
    return (df['weighted_score'] > 0.5).astype(float).values

def signal_v4_baseline(df):
    return df['v4_signal'].values.copy()

def signal_sma50(df):
    return (df['close'] > df['sma50']).astype(float).values

# ── Walk-Forward + Permutation Testing ────────────────────────────────────────

def expanding_walkforward(positions, fwd_returns, n_folds=14):
    """14-fold expanding walk-forward. Returns OOS returns for each fold."""
    n = len(positions)
    min_train = max(60, n // (n_folds + 2))  # minimum training window
    fold_size = (n - min_train) // n_folds
    
    oos_rets = []
    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = min(oos_start + fold_size, n)
        if oos_start >= n:
            break
        fold_ret = positions[oos_start:oos_end] * fwd_returns[oos_start:oos_end]
        oos_rets.extend(fold_ret)
    
    return np.array(oos_rets)

def permutation_test(positions, fwd_returns, n_perms=500, n_folds=14):
    """Permutation test on walk-forward returns."""
    real_oos = expanding_walkforward(positions, fwd_returns, n_folds)
    real_oos = real_oos[~np.isnan(real_oos)]
    if len(real_oos) == 0:
        return 0.0, 0.0, 1.0
    
    real_sharpe = np.mean(real_oos) / (np.std(real_oos) + 1e-10) * np.sqrt(252)
    
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(fwd_returns)
        perm_oos = expanding_walkforward(positions, shuffled, n_folds)
        perm_oos = perm_oos[~np.isnan(perm_oos)]
        if len(perm_oos) > 0:
            s = np.mean(perm_oos) / (np.std(perm_oos) + 1e-10) * np.sqrt(252)
            perm_sharpes.append(s)
    
    if len(perm_sharpes) == 0:
        return real_sharpe, 0.0, 1.0
    
    p_value = np.mean(np.array(perm_sharpes) >= real_sharpe)
    return real_sharpe, np.mean(perm_sharpes), p_value

def compute_metrics(positions, fwd_returns):
    """Compute standard backtest metrics."""
    strat_ret = positions * fwd_returns
    valid = ~np.isnan(strat_ret)
    strat_ret = strat_ret[valid]
    
    if len(strat_ret) == 0:
        return {'total_return': 0, 'annual_return': 0, 'sharpe': 0, 'max_dd': 0, 
                'win_rate': 0, 'n_trades': 0, 'exposure': 0}
    
    cum = (1 + strat_ret).cumprod()
    total_ret = cum[-1] - 1
    n_years = len(strat_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
    
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    trades = strat_ret[strat_ret != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    exposure = (positions[valid] != 0).mean()
    
    return {
        'total_return': round(float(total_ret * 100), 2),
        'annual_return': round(float(ann_ret * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd * 100), 2),
        'win_rate': round(float(win_rate * 100), 1),
        'exposure': round(float(exposure * 100), 1),
        'n_days': int(len(strat_ret))
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    results = {}
    all_regime_data = {}
    
    # First load BTC for cross-asset regime
    btc_df = load_data('BTC')
    btc_df = compute_regime_score(btc_df)
    btc_regime = btc_df[['date', 'regime_score']].rename(columns={'regime_score': 'btc_regime_score'})
    
    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")
        
        try:
            df = load_data(symbol)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        
        df = compute_regime_score(df)
        df = compute_weighted_regime(df)
        df = compute_sma50_signal(df)
        
        # Merge BTC regime for cross-asset
        df = df.merge(btc_regime, on='date', how='left')
        df['btc_regime_score'] = df['btc_regime_score'].fillna(2)  # neutral default
        
        # Drop rows without enough data
        df = df.dropna(subset=['fwd_returns', 'regime_score', 'sma50']).reset_index(drop=True)
        
        print(f"  Data: {df['date'].min().date()} to {df['date'].max().date()}, n={len(df)}")
        
        # ── Regime Score Distribution ──
        score_dist = df['regime_score'].value_counts(normalize=True).sort_index()
        print(f"\n  Regime Score Distribution:")
        for score in range(5):
            pct = score_dist.get(score, 0) * 100
            print(f"    Score {score}: {pct:5.1f}%")
        
        # ── Average Returns per Score (Event Study) ──
        print(f"\n  Avg Next-Day Return by Regime Score:")
        for score in range(5):
            mask = df['regime_score'] == score
            if mask.sum() > 0:
                avg_ret = df.loc[mask, 'fwd_returns'].mean() * 100
                n = mask.sum()
                print(f"    Score {score}: {avg_ret:+.4f}% (n={n})")
        
        # ── Define variants ──
        fwd = df['fwd_returns'].values
        
        variants = {
            '1_regime_standalone': signal_regime_standalone(df),
            '2_regime_v4_overlay': signal_regime_v4_overlay(df),
            '3_regime_v4_riskoff': signal_regime_v4_riskoff(df),
            '4_binary_regime': signal_binary_regime(df),
            '5_weighted_regime': signal_weighted_regime(df),
            '6_cross_asset_regime': (df['btc_regime_score'] >= 2).astype(float).values if symbol != 'BTC' else signal_binary_regime(df),
        }
        
        baselines = {
            'V4_baseline': signal_v4_baseline(df),
            'SMA50': signal_sma50(df),
            'BuyHold': np.ones(len(df)),
        }
        
        # ── Run all ──
        symbol_results = {}
        
        print(f"\n  {'Variant':<28} {'TotRet%':>8} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'Exp%':>5} | {'WF_Sharpe':>9} {'p-val':>7} {'Sig':>4}")
        print(f"  {'-'*110}")
        
        all_variants = {**baselines, **variants}
        
        for name, pos in all_variants.items():
            metrics = compute_metrics(pos, fwd)
            
            # Walk-forward + permutation
            wf_sharpe, _, p_val = permutation_test(pos, fwd, N_PERMS, N_FOLDS)
            
            # Bonferroni correction (only for test variants, not baselines)
            if name in variants:
                adj_p = min(p_val * BONFERRONI_VARIANTS, 1.0)
            else:
                adj_p = p_val
            
            sig = '***' if adj_p < 0.01 else '**' if adj_p < 0.05 else '*' if adj_p < 0.1 else ''
            
            metrics['wf_sharpe'] = round(float(wf_sharpe), 3)
            metrics['p_value'] = round(float(p_val), 4)
            metrics['p_value_bonferroni'] = round(float(adj_p), 4)
            
            symbol_results[name] = metrics
            
            print(f"  {name:<28} {metrics['total_return']:>8.1f} {metrics['annual_return']:>8.1f} {metrics['sharpe']:>7.3f} {metrics['max_dd']:>7.1f} {metrics['win_rate']:>6.1f} {metrics['exposure']:>5.1f} | {wf_sharpe:>9.3f} {adj_p:>7.4f} {sig:>4}")
        
        results[symbol] = symbol_results
        
        # Save regime data for analysis
        all_regime_data[symbol] = {
            'score_distribution': {int(k): round(float(v), 4) for k, v in score_dist.items()},
            'avg_return_by_score': {
                int(score): round(float(df.loc[df['regime_score'] == score, 'fwd_returns'].mean() * 100), 4)
                for score in range(5) if (df['regime_score'] == score).sum() > 0
            }
        }
    
    # ── Summary Comparison ──
    print(f"\n\n{'='*80}")
    print(f"  CROSS-ASSET SUMMARY: Variant vs V4 Baseline (Sharpe Improvement)")
    print(f"{'='*80}")
    
    print(f"\n  {'Variant':<28}", end='')
    for s in SYMBOLS:
        print(f" {s:>10}", end='')
    print(f" {'Avg':>10}")
    print(f"  {'-'*78}")
    
    variant_names = list(list(results.values())[0].keys()) if results else []
    for vname in variant_names:
        if vname in ['V4_baseline', 'SMA50', 'BuyHold']:
            continue
        print(f"  {vname:<28}", end='')
        diffs = []
        for s in SYMBOLS:
            if s in results and vname in results[s] and 'V4_baseline' in results[s]:
                diff = results[s][vname]['sharpe'] - results[s]['V4_baseline']['sharpe']
                diffs.append(diff)
                print(f" {diff:>+10.3f}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        if diffs:
            print(f" {np.mean(diffs):>+10.3f}")
        else:
            print()
    
    # MaxDD improvement
    print(f"\n  {'Variant':<28}", end='')
    for s in SYMBOLS:
        print(f" {s:>10}", end='')
    print(f" {'Avg':>10}")
    print(f"  {'-'*78}")
    print(f"  (MaxDD improvement vs V4, positive = better)")
    
    for vname in variant_names:
        if vname in ['V4_baseline', 'SMA50', 'BuyHold']:
            continue
        print(f"  {vname:<28}", end='')
        diffs = []
        for s in SYMBOLS:
            if s in results and vname in results[s] and 'V4_baseline' in results[s]:
                # MaxDD is negative, so less negative = better = positive improvement
                diff = results[s][vname]['max_dd'] - results[s]['V4_baseline']['max_dd']
                diffs.append(diff)
                print(f" {diff:>+10.1f}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        if diffs:
            print(f" {np.mean(diffs):>+10.1f}")
        else:
            print()
    
    # ── Save Results ──
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'symbols': SYMBOLS,
            'n_permutations': N_PERMS,
            'n_folds': N_FOLDS,
            'bonferroni_variants': BONFERRONI_VARIANTS,
        },
        'regime_analysis': all_regime_data,
        'results': results,
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    run_all()
