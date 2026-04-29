"""
Walk-forward backtest for FRAPS signal.
10-fold expanding window, permutation testing for significant results.
"""
import sys
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from strategies.derivatives.fraps_signal import backtest_fraps, compute_fraps_features

DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = DATA_DIR / 'backtest_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(token: str):
    """Load multi-exchange funding + price data."""
    funding_path = DATA_DIR / 'derivatives' / 'multi_exchange' / f'{token.lower()}_multi_exchange_funding.csv'
    price_path = DATA_DIR / 'ohlcv' / f'binance_{token.lower()}_usdt_1d.csv'
    
    funding = pd.read_csv(funding_path)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'])
    
    price = pd.read_csv(price_path)
    price['timestamp'] = pd.to_datetime(price['timestamp'])
    price = price.set_index('timestamp')
    
    return funding, price


def walk_forward_test(funding, price, params, n_folds=10):
    """Expanding window walk-forward test."""
    # Get overlapping date range
    fund_dates = funding['timestamp'].sort_values().unique()
    price_dates = price.index.sort_values().unique()
    
    min_date = max(fund_dates.min(), price_dates.min())
    max_date = min(fund_dates.max(), price_dates.max())
    
    funding = funding[(funding['timestamp'] >= min_date) & (funding['timestamp'] <= max_date)]
    price = price[(price.index >= min_date) & (price.index <= max_date)]
    
    total_days = (max_date - min_date).days
    fold_size = total_days // n_folds
    
    oos_returns = []
    fold_results = []
    
    for fold in range(2, n_folds + 1):  # start from fold 2 (need IS data)
        # IS: all data up to fold boundary
        # OOS: data in this fold
        is_end = min_date + pd.Timedelta(days=fold * fold_size)
        oos_start = is_end
        oos_end = min_date + pd.Timedelta(days=(fold + 1) * fold_size) if fold < n_folds else max_date
        
        if oos_start >= max_date:
            break
        
        oos_funding = funding[(funding['timestamp'] >= min_date) & (funding['timestamp'] <= oos_end)]
        oos_price = price[(price.index >= min_date) & (price.index <= oos_end)]
        
        result = backtest_fraps(
            oos_funding, oos_price,
            **params, commission_bps=20
        )
        
        # Extract only OOS period returns
        if 'result_df' in result and result['result_df'] is not None:
            rdf = result['result_df']
            oos_slice = rdf[(rdf.index >= oos_start) & (rdf.index <= oos_end)]
            if len(oos_slice) > 0:
                fold_ret = float((1 + oos_slice['strategy_return']).prod() - 1)
                oos_returns.append(fold_ret)
                fold_results.append({
                    'fold': fold,
                    'oos_start': str(oos_start.date()) if hasattr(oos_start, 'date') else str(oos_start)[:10],
                    'oos_end': str(oos_end.date()) if hasattr(oos_end, 'date') else str(oos_end)[:10],
                    'oos_return': round(fold_ret * 100, 2),
                    'n_trades': int(oos_slice['signal_change'].gt(0).sum()),
                })
    
    if not oos_returns:
        return None
    
    oos_returns = np.array(oos_returns)
    mean_ret = float(oos_returns.mean())
    
    # t-test: is mean return > 0?
    from scipy import stats
    if len(oos_returns) > 1 and oos_returns.std() > 0:
        t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
        p_value = p_value / 2  # one-tailed
    else:
        t_stat, p_value = 0, 1.0
    
    return {
        'mean_oos_return': round(mean_ret * 100, 4),
        'std_oos_return': round(float(oos_returns.std()) * 100, 4),
        't_stat': round(float(t_stat), 4),
        'p_value': round(float(p_value), 6),
        'n_folds': len(oos_returns),
        'folds': fold_results,
        'total_oos_return': round(float((1 + oos_returns).prod() - 1) * 100, 2),
    }


def permutation_test(funding, price, params, n_perms=200):
    """Random permutation test to verify signal isn't noise."""
    # Get actual result
    actual = backtest_fraps(funding, price, **params, commission_bps=20)
    actual_sharpe = actual['sharpe']
    actual_return = actual['total_return']
    
    # Permute: shuffle the signal-price alignment
    perm_sharpes = []
    perm_returns = []
    
    result_df = actual.get('result_df')
    if result_df is None or len(result_df) == 0:
        return None
    
    price_returns = result_df['price_return'].values
    signals = result_df['signal'].values
    
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        # Circular shift of signals by random amount
        shift = rng.randint(30, len(signals) - 30)
        perm_signals = np.roll(signals, shift)
        
        perm_strat = perm_signals[:-1] * price_returns[1:]
        if len(perm_strat) > 0 and perm_strat.std() > 0:
            perm_sharpes.append(float(perm_strat.mean() / perm_strat.std() * np.sqrt(365)))
            perm_returns.append(float((1 + perm_strat).prod() - 1) * 100)
    
    if not perm_sharpes:
        return None
    
    perm_sharpes = np.array(perm_sharpes)
    p_sharpe = float((perm_sharpes >= actual_sharpe).sum() / len(perm_sharpes))
    
    return {
        'actual_sharpe': actual_sharpe,
        'actual_return': actual_return,
        'perm_mean_sharpe': round(float(perm_sharpes.mean()), 4),
        'perm_std_sharpe': round(float(perm_sharpes.std()), 4),
        'p_value_permutation': round(p_sharpe, 4),
        'n_permutations': n_perms,
        'perm_95th_sharpe': round(float(np.percentile(perm_sharpes, 95)), 4),
    }


def main():
    tokens = ['BTC', 'ETH']
    
    # Parameter grid
    param_grid = {
        'dispersion_zscore_threshold': [1.0, 1.5, 2.0],
        'consensus_threshold': [0.7, 0.8, 0.9],
        'lookback': [30, 60, 90],
        'hold_period': [1, 3, 7],
    }
    
    # Generate all combos
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"Testing {len(combos)} parameter combinations per token")
    
    all_results = {}
    
    for token in tokens:
        print(f"\n{'='*60}")
        print(f"  {token}")
        print(f"{'='*60}")
        
        try:
            funding, price = load_data(token)
        except FileNotFoundError as e:
            print(f"  Missing data for {token}: {e}")
            continue
        
        # First, show data summary
        features = compute_fraps_features(funding)
        print(f"  Funding data: {len(funding)} rows, {funding['exchange'].nunique()} exchanges")
        print(f"  Price data: {len(price)} rows")
        print(f"  Feature dates: {features.index.min()} to {features.index.max()}")
        print(f"  Avg funding dispersion: {features['funding_std'].mean():.6f}")
        print(f"  Avg consensus: {features['consensus'].mean():.3f}")
        
        token_results = []
        best_sharpe = -999
        best_params = None
        
        for combo in combos:
            params = dict(zip(keys, combo))
            
            # Walk-forward test
            wf = walk_forward_test(funding, price, params)
            if wf is None:
                continue
            
            result_entry = {
                'params': params,
                'walk_forward': wf,
            }
            token_results.append(result_entry)
            
            if wf['mean_oos_return'] > 0 and (wf.get('t_stat', 0) > best_sharpe):
                best_sharpe = wf.get('t_stat', 0)
                best_params = params
        
        # Sort by t-stat
        token_results.sort(key=lambda x: x['walk_forward'].get('t_stat', 0), reverse=True)
        
        # Print top 5
        print(f"\n  Top 5 parameter sets by t-stat:")
        for i, r in enumerate(token_results[:5]):
            wf = r['walk_forward']
            p = r['params']
            print(f"  {i+1}. t={wf['t_stat']:.3f} p={wf['p_value']:.4f} "
                  f"ret={wf['mean_oos_return']:.2f}% "
                  f"[z={p['dispersion_zscore_threshold']} c={p['consensus_threshold']} "
                  f"lb={p['lookback']} hp={p['hold_period']}]")
        
        # Run permutation test on significant results (p < 0.05)
        significant = [r for r in token_results if r['walk_forward']['p_value'] < 0.05]
        print(f"\n  Significant results (p<0.05): {len(significant)} / {len(token_results)}")
        
        for r in significant[:3]:  # top 3 significant
            print(f"  Running permutation test for {r['params']}...")
            perm = permutation_test(funding, price, r['params'], n_perms=200)
            r['permutation_test'] = perm
            if perm:
                print(f"    Perm p-value: {perm['p_value_permutation']:.4f} "
                      f"(actual Sharpe: {perm['actual_sharpe']:.3f}, "
                      f"perm 95th: {perm['perm_95th_sharpe']:.3f})")
        
        all_results[token] = {
            'n_combos_tested': len(combos),
            'n_with_results': len(token_results),
            'n_significant': len(significant),
            'top_results': token_results[:10],  # save top 10
            'best_params': best_params,
        }
    
    # Also run the proxy approach using single-exchange funding volatility
    print(f"\n{'='*60}")
    print("  PROXY: Funding Rate Volatility (single exchange)")
    print(f"{'='*60}")
    
    for token in tokens:
        try:
            funding, price = load_data(token)
        except FileNotFoundError:
            continue
        
        # Use only binance funding and simulate dispersion via rolling std of funding changes
        binance_funding = funding[funding['exchange'] == 'binance'].copy()
        if len(binance_funding) == 0:
            continue
        
        binance_funding = binance_funding.sort_values('timestamp')
        binance_funding['funding_change'] = binance_funding['funding_close'].diff()
        
        # Create a "pseudo multi-exchange" df using lagged funding as different "exchanges"
        pseudo_dfs = []
        for lag in [0, 1, 2, 3]:
            tmp = binance_funding[['timestamp', 'funding_close']].copy()
            tmp['funding_close'] = tmp['funding_close'].shift(lag)
            tmp['exchange'] = f'lag_{lag}'
            tmp['token'] = token
            pseudo_dfs.append(tmp)
        
        pseudo_funding = pd.concat(pseudo_dfs).dropna()
        
        # Test with default params
        params = {'dispersion_zscore_threshold': 1.5, 'consensus_threshold': 0.8,
                  'lookback': 60, 'hold_period': 3}
        wf = walk_forward_test(pseudo_funding, price, params)
        if wf:
            print(f"  {token} proxy: t={wf['t_stat']:.3f} p={wf['p_value']:.4f} ret={wf['mean_oos_return']:.2f}%")
    
    # Save results
    # Remove non-serializable result_df
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != 'result_df'}
        if isinstance(obj, list):
            return [clean_for_json(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    out_path = RESULTS_DIR / 'fraps_results.json'
    with open(out_path, 'w') as f:
        json.dump(clean_for_json(all_results), f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("  FRAPS SUMMARY")
    print(f"{'='*60}")
    for token, res in all_results.items():
        print(f"\n{token}:")
        print(f"  Combos tested: {res['n_combos_tested']}")
        print(f"  With results: {res['n_with_results']}")
        print(f"  Significant (p<0.05): {res['n_significant']}")
        if res['best_params']:
            print(f"  Best params: {res['best_params']}")
        if res['top_results']:
            top = res['top_results'][0]
            wf = top['walk_forward']
            print(f"  Best t-stat: {wf['t_stat']:.3f} (p={wf['p_value']:.4f})")
            print(f"  Best OOS return: {wf['mean_oos_return']:.2f}%")
            if 'permutation_test' in top and top['permutation_test']:
                pt = top['permutation_test']
                print(f"  Permutation p-value: {pt['p_value_permutation']:.4f}")


if __name__ == '__main__':
    main()
