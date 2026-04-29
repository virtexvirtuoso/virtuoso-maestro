"""
Walk-Forward Backtest: 6 Price Structure Strategies — Multi-Asset
Tests on: ETH, SOL, SUI, LINK, RENDER, AVAX, INJ, OP
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))

from strategies.technical.market_structure import generate_signals as market_structure_signals
from strategies.technical.range_sfp import generate_signals as range_sfp_signals
from strategies.technical.fair_value_gaps import generate_signals as fvg_signals
from strategies.technical.order_blocks import generate_signals as order_blocks_signals
from strategies.technical.volume_profile import generate_signals as volume_profile_signals
from strategies.technical.sr_levels import generate_signals as sr_levels_signals

DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv/')
ASSETS = ['eth', 'sol', 'sui', 'link', 'render', 'avax', 'inj', 'op']
N_FOLDS = 14
COMMISSION = 0.001

STRATEGIES = {
    'MarketStructure': market_structure_signals,
    'RangeSFP': range_sfp_signals,
    'FairValueGaps': fvg_signals,
    'OrderBlocks': order_blocks_signals,
    'VolumeProfile': volume_profile_signals,
    'SRLevels': sr_levels_signals,
}


def load_asset(asset):
    path = os.path.join(DATA_DIR, f'binance_{asset}_usdt_1d.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def compute_metrics(returns):
    if len(returns) == 0 or returns.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'total_return': 0}
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 365
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(365)
    cum = (1 + returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd * 100, 1), 'total_return': round(total_ret * 100, 1)}


def backtest_strategy(df, signal_func):
    try:
        signals = signal_func(df)
    except:
        return pd.Series(0, index=df.index)
    daily_returns = df['close'].pct_change().shift(-1)
    position_changes = signals.diff().abs().fillna(0)
    return (signals * daily_returns - position_changes * COMMISSION).fillna(0)


def walk_forward(df, signal_func):
    n = len(df)
    min_is = 100
    fold_size = (n - min_is) // N_FOLDS
    oos_sharpes = []
    
    for fold in range(N_FOLDS):
        is_end = min_is + (fold + 1) * fold_size
        oos_end = min(is_end + fold_size, n)
        if is_end >= n or oos_end <= is_end:
            break
        oos_returns = backtest_strategy(df.iloc[is_end:oos_end], signal_func)
        m = compute_metrics(oos_returns)
        oos_sharpes.append(m['sharpe'])
    
    oos_arr = np.array(oos_sharpes)
    pos = (oos_arr > 0).sum()
    mean_sh = oos_arr.mean() if len(oos_arr) > 0 else 0
    if len(oos_arr) > 1 and oos_arr.std() > 0:
        _, p = stats.ttest_1samp(oos_arr, 0)
        p = p / 2
    else:
        p = 1.0
    
    full_ret = backtest_strategy(df, signal_func)
    full = compute_metrics(full_ret)
    bh_ret = df['close'].pct_change().fillna(0)
    bh = compute_metrics(bh_ret)
    
    return {
        'full': full,
        'bh': bh,
        'oos_sharpe': round(mean_sh, 3),
        'pos_folds': f"{pos}/{len(oos_sharpes)}",
        'p_value': round(p, 4),
        'sig': p < 0.05,
    }


def main():
    print("=" * 80)
    print("  PRICE STRUCTURE STRATEGIES — Multi-Asset Walk-Forward")
    print("=" * 80)
    
    all_results = {}
    
    for asset in ASSETS:
        df = load_asset(asset)
        if df is None:
            print(f"\n⚠️  {asset.upper()}: no data")
            continue
        
        ticker = asset.upper()
        print(f"\n{'='*80}")
        print(f"  {ticker} — {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
        print(f"{'='*80}")
        
        bh_printed = False
        asset_results = {}
        
        for sname, sfunc in STRATEGIES.items():
            r = walk_forward(df, sfunc)
            asset_results[sname] = r
            
            if not bh_printed:
                print(f"  Buy & Hold: Sharpe {r['bh']['sharpe']}, CAGR {r['bh']['cagr']}%, MaxDD {r['bh']['max_dd']}%")
                print(f"  {'Strategy':<18} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'OOS Sh':>7} {'Folds':>7} {'p-val':>7} {'Sig':>4}")
                print(f"  {'-'*65}")
                bh_printed = True
            
            f = r['full']
            sig = '✅' if r['sig'] else '❌'
            print(f"  {sname:<18} {f['sharpe']:>7.3f} {f['cagr']:>6.1f}% {f['max_dd']:>6.1f}% {r['oos_sharpe']:>7.3f} {r['pos_folds']:>7} {r['p_value']:>7.4f} {sig:>4}")
        
        all_results[ticker] = asset_results
    
    # Cross-asset summary: best strategy per asset
    print(f"\n{'='*80}")
    print(f"  CROSS-ASSET SUMMARY — Best Strategy Per Asset")
    print(f"{'='*80}")
    print(f"  {'Asset':<8} {'Best Strategy':<18} {'OOS Sharpe':>10} {'p-value':>8} {'Sig':>4}")
    print(f"  {'-'*52}")
    
    # Also track: best asset per strategy
    strat_scores = {s: [] for s in STRATEGIES}
    
    for asset, sresults in all_results.items():
        best = max(sresults.items(), key=lambda x: x[1]['oos_sharpe'])
        sig = '✅' if best[1]['sig'] else '❌'
        print(f"  {asset:<8} {best[0]:<18} {best[1]['oos_sharpe']:>10.3f} {best[1]['p_value']:>8.4f} {sig:>4}")
        for s, r in sresults.items():
            strat_scores[s].append(r['oos_sharpe'])
    
    print(f"\n  {'Strategy':<18} {'Mean OOS':>8} {'Median':>8} {'Assets>0':>9}")
    print(f"  {'-'*47}")
    for s, scores in strat_scores.items():
        arr = np.array(scores)
        pos = (arr > 0).sum()
        print(f"  {s:<18} {arr.mean():>8.3f} {np.median(arr):>8.3f} {pos}/{len(arr):>7}")
    
    # Save
    output = os.path.expanduser('~/Desktop/maestro/data/backtest_results/price_structure_multi_wf.json')
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output}")


if __name__ == '__main__':
    main()
