#!/usr/bin/env python3
"""
Walk-Forward Validation of ALL 11 Derivatives Strategies
=========================================================
Tests across 13 tokens with 10 expanding folds.
Merges real derivatives data where available, falls back to proxies.
"""

import sys
import os
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Setup paths
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / 'strategies'))

DATA_DIR = BACKEND_DIR.parent / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
DERIV_DIR = DATA_DIR / "derivatives"
RESULTS_DIR = DATA_DIR / "backtest_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = ['arb', 'avax', 'btc', 'eth', 'fet', 'inj', 'link', 'op', 'render', 'sol', 'sui', 'tao', 'tia']
N_FOLDS = 10
MIN_OOS_BARS = 30  # minimum OOS bars needed
MIN_TOTAL_BARS = 100  # minimum total bars for walk-forward

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(token: str) -> pd.DataFrame:
    """Load daily OHLCV data."""
    path = OHLCV_DIR / f"binance_{token}_usdt_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df


def load_funding(token: str) -> pd.Series:
    """Load funding rate data, resample to daily."""
    path = DERIV_DIR / f"{token}_funding.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[0]
    daily = df[col].resample('1D').mean()
    return daily


def load_taker(token: str) -> pd.DataFrame:
    """Load taker buy/sell ratio, resample to daily."""
    path = DERIV_DIR / f"{token}_taker.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    daily = df.select_dtypes(include='number').resample('1D').mean()
    return daily


def load_lsr(token: str) -> pd.DataFrame:
    """Load LSR data, resample to daily."""
    path = DERIV_DIR / f"{token}_lsr_global.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    daily = df.select_dtypes(include='number').resample('1D').mean()
    return daily


def load_oi(token: str) -> pd.Series:
    """Load OI data (daily)."""
    path = DERIV_DIR / f"{token}_oi_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    col = 'sumOpenInterestValue' if 'sumOpenInterestValue' in df.columns else None
    if col is None:
        numeric_cols = df.select_dtypes(include='number').columns
        col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
    return pd.to_numeric(df[col], errors='coerce')


def build_merged_df(token: str) -> pd.DataFrame:
    """Build merged OHLCV + derivatives DataFrame."""
    df = load_ohlcv(token)
    if df is None:
        return None

    # Merge funding rate
    funding = load_funding(token)
    if funding is not None and len(funding) > 10:
        df['funding_rate'] = funding.reindex(df.index, method='ffill')

    # Merge OI
    oi = load_oi(token)
    if oi is not None and len(oi) > 5:
        df['oi'] = oi.reindex(df.index, method='ffill')

    # Merge LSR
    lsr = load_lsr(token)
    if lsr is not None and len(lsr) > 5:
        if 'longAccount' in lsr.columns:
            df['long_ratio'] = lsr['longAccount'].reindex(df.index, method='ffill')

    # Merge taker
    taker = load_taker(token)
    if taker is not None and len(taker) > 5:
        if 'buySellRatio' in taker.columns:
            df['taker_ratio'] = taker['buySellRatio'].reindex(df.index, method='ffill')
        if 'buyVol' in taker.columns and 'sellVol' in taker.columns:
            df['buy_vol'] = taker['buyVol'].reindex(df.index, method='ffill')
            df['sell_vol'] = taker['sellVol'].reindex(df.index, method='ffill')

    return df


# ============================================================================
# STRATEGY LOADING
# ============================================================================

def load_all_strategies():
    """Load all 11 derivatives strategies, return dict of name -> callable."""
    strategies = {}

    # Function-based strategies (simple: generate_signals(df, ...))
    from strategies.derivatives.cvd_scalp import generate_signals as cvd_signals
    strategies['CVDScalp'] = {'fn': cvd_signals, 'type': 'function', 'needs': ['ohlcv']}

    from strategies.derivatives.delta_flow import generate_signals as delta_signals
    strategies['DeltaFlow'] = {'fn': delta_signals, 'type': 'function', 'needs': ['ohlcv']}

    from strategies.derivatives.funding_rate import generate_signals as fr_signals
    strategies['FundingRate'] = {'fn': fr_signals, 'type': 'function', 'needs': ['funding']}

    from strategies.derivatives.liquidation_scalp import generate_signals as liqscalp_signals
    strategies['LiquidationScalp'] = {'fn': liqscalp_signals, 'type': 'function', 'needs': ['ohlcv']}

    from strategies.derivatives.micro_basis import generate_signals as micro_signals
    strategies['MicroBasis'] = {'fn': micro_signals, 'type': 'function', 'needs': ['funding']}

    # Class-based strategies (use DerivativesDataMixin)
    from strategies.derivatives.real_funding_rate import RealFundingRateStrategy
    strategies['RealFundingRate'] = {'cls': RealFundingRateStrategy, 'type': 'class', 'needs': ['funding']}

    from strategies.derivatives.real_oi_momentum import RealOIMomentumStrategy
    strategies['RealOIMomentum'] = {'cls': RealOIMomentumStrategy, 'type': 'class', 'needs': ['oi']}

    from strategies.derivatives.spot_perp_basis import SpotPerpBasisStrategy
    strategies['SpotPerpBasis'] = {'cls': SpotPerpBasisStrategy, 'type': 'class', 'needs': ['ohlcv']}

    from strategies.derivatives.liquidation_cascade import LiquidationCascadeStrategy
    strategies['LiquidationCascade'] = {'cls': LiquidationCascadeStrategy, 'type': 'class', 'needs': ['ohlcv']}

    from strategies.derivatives.cross_exchange_oi import CrossExchangeOIDivergenceStrategy
    strategies['CrossExchangeOI'] = {'cls': CrossExchangeOIDivergenceStrategy, 'type': 'class', 'needs': ['oi']}

    from strategies.derivatives.vol_regime_funding import VolRegimeFundingOverlayStrategy
    strategies['VolRegimeFunding'] = {'cls': VolRegimeFundingOverlayStrategy, 'type': 'class', 'needs': ['funding']}

    return strategies


def run_strategy(strat_info, df, token):
    """Run a strategy and return signals Series."""
    symbol = token.upper()
    if strat_info['type'] == 'function':
        return strat_info['fn'](df)
    else:
        obj = strat_info['cls']()
        return obj.generate_signals(df, symbol=symbol)


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def walk_forward_test(signals: pd.Series, prices: pd.Series, n_folds: int = 10):
    """
    Expanding-window walk-forward test.
    
    Returns dict with OOS metrics or None if insufficient data.
    """
    returns = prices.pct_change().fillna(0)
    
    # Align
    common_idx = signals.index.intersection(returns.index)
    signals = signals.reindex(common_idx).fillna(0)
    returns = returns.reindex(common_idx).fillna(0)
    
    n = len(common_idx)
    if n < MIN_TOTAL_BARS:
        return None
    
    # Strategy returns: signal * next-day return (shift signals by 1 to avoid lookahead)
    strat_returns = signals.shift(1).fillna(0) * returns
    
    min_train = max(n // 3, 50)  # At least 50 bars or 1/3 of data for initial training
    fold_size = (n - min_train) // n_folds
    
    if fold_size < 5:
        return None
    
    oos_returns_all = []
    fold_results = []
    
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        
        if test_end <= train_end:
            break
        
        oos_ret = strat_returns.iloc[train_end:test_end]
        oos_returns_all.extend(oos_ret.values)
        
        fold_sharpe = (oos_ret.mean() / oos_ret.std() * np.sqrt(252)) if oos_ret.std() > 0 else 0
        fold_results.append({
            'fold': fold,
            'train_end': train_end,
            'test_bars': test_end - train_end,
            'sharpe': round(float(fold_sharpe), 4),
            'return': round(float(oos_ret.sum()), 6),
        })
    
    oos_arr = np.array(oos_returns_all)
    
    if len(oos_arr) < MIN_OOS_BARS or np.std(oos_arr) < 1e-10:
        return None
    
    oos_sharpe = float(np.mean(oos_arr) / np.std(oos_arr) * np.sqrt(252))
    
    # t-test: is mean OOS return significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(oos_arr, 0)
    
    # Trade stats
    trade_signals = signals.shift(1).fillna(0)
    # Count position changes as trades
    position_changes = (trade_signals.diff().fillna(0) != 0).sum()
    
    # Win rate on bars where we have a position
    positioned = oos_arr[oos_arr != 0] if len(oos_arr[oos_arr != 0]) > 0 else oos_arr
    wins = (positioned > 0).sum()
    total_positioned = len(positioned)
    win_rate = float(wins / total_positioned) if total_positioned > 0 else 0
    
    total_return = float(np.sum(oos_arr))
    max_dd = float(np.min(np.minimum.accumulate(np.cumsum(oos_arr)) - np.cumsum(oos_arr)))
    
    return {
        'oos_sharpe': round(oos_sharpe, 4),
        'p_value': round(float(p_value), 6),
        't_stat': round(float(t_stat), 4),
        'total_return': round(total_return, 6),
        'max_drawdown': round(max_dd, 6),
        'oos_bars': len(oos_arr),
        'trades': int(position_changes),
        'win_rate': round(win_rate, 4),
        'folds': fold_results,
    }


def permutation_test(signals: pd.Series, prices: pd.Series, n_perms: int = 200):
    """Run permutation test - shuffle signal-return alignment."""
    returns = prices.pct_change().fillna(0)
    common_idx = signals.index.intersection(returns.index)
    signals_aligned = signals.reindex(common_idx).fillna(0)
    returns_aligned = returns.reindex(common_idx).fillna(0)
    
    strat_returns = (signals_aligned.shift(1).fillna(0) * returns_aligned).values
    real_sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-10) * np.sqrt(252)
    
    perm_sharpes = []
    ret_vals = returns_aligned.values
    for _ in range(n_perms):
        shuffled = np.random.permutation(ret_vals)
        sr = signals_aligned.shift(1).fillna(0).values * shuffled
        perm_sharpe = np.mean(sr) / (np.std(sr) + 1e-10) * np.sqrt(252)
        perm_sharpes.append(perm_sharpe)
    
    perm_p = float(np.mean(np.array(perm_sharpes) >= real_sharpe))
    return {
        'real_sharpe': round(float(real_sharpe), 4),
        'perm_p_value': round(perm_p, 4),
        'perm_mean_sharpe': round(float(np.mean(perm_sharpes)), 4),
        'perm_std_sharpe': round(float(np.std(perm_sharpes)), 4),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DERIVATIVES STRATEGIES WALK-FORWARD VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Tokens: {len(TOKENS)} | Folds: {N_FOLDS}")
    print("=" * 80)
    
    # Load strategies
    print("\nLoading strategies...")
    strategies = load_all_strategies()
    print(f"Loaded {len(strategies)} strategies: {list(strategies.keys())}")
    
    # Load data for all tokens
    print("\nLoading data...")
    token_data = {}
    for token in TOKENS:
        df = build_merged_df(token)
        if df is not None:
            deriv_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
            print(f"  {token.upper():6s}: {len(df)} bars, derivatives cols: {deriv_cols}")
            token_data[token] = df
        else:
            print(f"  {token.upper():6s}: NO DATA")
    
    # Run walk-forward for all combos
    results = []
    failures = []
    
    print("\n" + "=" * 80)
    print("RUNNING WALK-FORWARD TESTS...")
    print("=" * 80)
    
    for strat_name, strat_info in strategies.items():
        print(f"\n--- {strat_name} ---")
        for token in TOKENS:
            if token not in token_data:
                failures.append({'strategy': strat_name, 'token': token, 'reason': 'No OHLCV data'})
                continue
            
            df = token_data[token]
            
            try:
                signals = run_strategy(strat_info, df, token)
                
                n_long = int((signals == 1).sum())
                n_short = int((signals == -1).sum())
                
                if n_long + n_short < 5:
                    failures.append({
                        'strategy': strat_name, 'token': token,
                        'reason': f'Too few signals: {n_long}L/{n_short}S'
                    })
                    print(f"  {token.upper():6s}: SKIP (only {n_long}L/{n_short}S signals)")
                    continue
                
                wf = walk_forward_test(signals, df['close'], n_folds=N_FOLDS)
                
                if wf is None:
                    failures.append({
                        'strategy': strat_name, 'token': token,
                        'reason': 'Insufficient data for walk-forward'
                    })
                    print(f"  {token.upper():6s}: SKIP (insufficient data for WF)")
                    continue
                
                result = {
                    'strategy': strat_name,
                    'token': token.upper(),
                    'signals_long': n_long,
                    'signals_short': n_short,
                    **wf,
                }
                
                # Permutation test if significant
                if wf['p_value'] < 0.05:
                    perm = permutation_test(signals, df['close'], n_perms=200)
                    result['permutation'] = perm
                    print(f"  {token.upper():6s}: Sharpe={wf['oos_sharpe']:+.3f} p={wf['p_value']:.4f} ** perm_p={perm['perm_p_value']:.3f}")
                else:
                    print(f"  {token.upper():6s}: Sharpe={wf['oos_sharpe']:+.3f} p={wf['p_value']:.4f} trades={wf['trades']}")
                
                results.append(result)
                
            except Exception as e:
                failures.append({
                    'strategy': strat_name, 'token': token,
                    'reason': f'Error: {str(e)}'
                })
                print(f"  {token.upper():6s}: ERROR - {e}")
                traceback.print_exc()
    
    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    print("\n" + "=" * 100)
    print("GRAND RESULTS TABLE")
    print("=" * 100)
    print(f"{'Strategy':<22} {'Token':<8} {'OOS Sharpe':>11} {'p-value':>9} {'Trades':>7} {'Win Rate':>9} {'Return':>10} {'Perm p':>8}")
    print("-" * 100)
    
    # Sort by OOS Sharpe descending
    results_sorted = sorted(results, key=lambda x: x['oos_sharpe'], reverse=True)
    
    for r in results_sorted:
        perm_p = r.get('permutation', {}).get('perm_p_value', '')
        perm_str = f"{perm_p:.3f}" if perm_p != '' else ''
        sig_marker = '**' if r['p_value'] < 0.05 else '  '
        print(f"{r['strategy']:<22} {r['token']:<8} {r['oos_sharpe']:>+10.3f} {r['p_value']:>9.4f}{sig_marker} {r['trades']:>6} {r['win_rate']:>8.1%} {r['total_return']:>+10.4f} {perm_str:>8}")
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    significant = [r for r in results if r['p_value'] < 0.05]
    perm_significant = [r for r in results if r.get('permutation', {}).get('perm_p_value', 1.0) < 0.05]
    positive_sharpe = [r for r in results if r['oos_sharpe'] > 0]
    
    print(f"Total combos tested: {len(results)}")
    print(f"Combos failed/skipped: {len(failures)}")
    print(f"Positive OOS Sharpe: {len(positive_sharpe)}/{len(results)} ({100*len(positive_sharpe)/max(len(results),1):.0f}%)")
    print(f"Significant (p<0.05): {len(significant)}/{len(results)}")
    print(f"Survives permutation (perm_p<0.05): {len(perm_significant)}/{len(significant)}")
    
    if significant:
        print(f"\nTop significant results:")
        for r in sorted(significant, key=lambda x: x['oos_sharpe'], reverse=True)[:10]:
            perm_p = r.get('permutation', {}).get('perm_p_value', 'N/A')
            print(f"  {r['strategy']:<22} {r['token']:<6} Sharpe={r['oos_sharpe']:+.3f} p={r['p_value']:.4f} perm_p={perm_p}")
    
    # By strategy
    print(f"\nBy Strategy (avg OOS Sharpe):")
    strat_groups = {}
    for r in results:
        strat_groups.setdefault(r['strategy'], []).append(r['oos_sharpe'])
    for name, sharpes in sorted(strat_groups.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  {name:<22} avg={np.mean(sharpes):+.3f} med={np.median(sharpes):+.3f} n={len(sharpes)}")
    
    # Failures
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:20]:
            print(f"  {f['strategy']:<22} {f['token']:<6} {f['reason']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures)-20} more")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {'tokens': TOKENS, 'n_folds': N_FOLDS, 'min_oos_bars': MIN_OOS_BARS},
        'results': results_sorted,
        'failures': failures,
        'summary': {
            'total_tested': len(results),
            'total_failed': len(failures),
            'positive_sharpe': len(positive_sharpe),
            'significant_p05': len(significant),
            'survives_permutation': len(perm_significant),
        }
    }
    
    out_path = RESULTS_DIR / "derivatives_all_wf.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
