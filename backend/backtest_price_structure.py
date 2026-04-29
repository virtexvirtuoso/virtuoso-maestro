"""
Walk-Forward Backtest: 6 Price Structure Strategies on BTC Daily
14-fold walk-forward with 70% IS / 30% OOS splits.
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

DATA_PATH = os.path.expanduser('~/Desktop/maestro/data/ohlcv/binance_btc_usdt_1d.csv')
N_FOLDS = 14
IS_RATIO = 0.7
INITIAL_CASH = 100_000
COMMISSION = 0.001  # 10bps round trip

STRATEGIES = {
    'MarketStructure': market_structure_signals,
    'RangeSFP': range_sfp_signals,
    'FairValueGaps': fvg_signals,
    'OrderBlocks': order_blocks_signals,
    'VolumeProfile': volume_profile_signals,
    'SRLevels': sr_levels_signals,
}


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def compute_metrics(returns):
    """Compute strategy metrics from daily returns."""
    if len(returns) == 0 or returns.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'win_rate': 0, 'trades': 0, 'total_return': 0}
    
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 365
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    # Win rate (on days with positions)
    active = returns[returns != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    
    return {
        'sharpe': round(sharpe, 3),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'win_rate': round(win_rate * 100, 1),
        'total_return': round(total_ret * 100, 1),
    }


def backtest_strategy(df, signal_func, name):
    """Run strategy and return daily returns."""
    try:
        signals = signal_func(df)
    except Exception as e:
        print(f"  ⚠️  {name} signal generation failed: {e}")
        return pd.Series(0, index=df.index)
    
    # Forward returns (next day's return based on today's signal)
    daily_returns = df['close'].pct_change().shift(-1)
    
    # Apply signals with commission on position changes
    position_changes = signals.diff().abs().fillna(0)
    commission_cost = position_changes * COMMISSION
    
    strategy_returns = (signals * daily_returns - commission_cost).fillna(0)
    return strategy_returns


def walk_forward(df, signal_func, name):
    """14-fold expanding walk-forward validation."""
    n = len(df)
    min_is = 100  # minimum in-sample bars
    fold_size = (n - min_is) // N_FOLDS
    
    is_results = []
    oos_results = []
    fold_details = []
    
    for fold in range(N_FOLDS):
        # Expanding window IS, fixed-size OOS
        is_end = min_is + (fold + 1) * fold_size
        oos_end = min(is_end + fold_size, n)
        
        if is_end >= n or oos_end <= is_end:
            break
        
        is_df = df.iloc[:is_end]
        oos_df = df.iloc[is_end:oos_end]
        
        if len(oos_df) < 10:
            break
        
        is_returns = backtest_strategy(is_df, signal_func, name)
        oos_returns = backtest_strategy(oos_df, signal_func, name)
        
        is_metrics = compute_metrics(is_returns)
        oos_metrics = compute_metrics(oos_returns)
        
        is_results.append(is_metrics['sharpe'])
        oos_results.append(oos_metrics['sharpe'])
        
        fold_details.append({
            'fold': fold + 1,
            'is_period': f"{is_df.index[0].strftime('%Y-%m-%d')} → {is_df.index[-1].strftime('%Y-%m-%d')}",
            'oos_period': f"{oos_df.index[0].strftime('%Y-%m-%d')} → {oos_df.index[-1].strftime('%Y-%m-%d')}",
            'is_sharpe': is_metrics['sharpe'],
            'oos_sharpe': oos_metrics['sharpe'],
            'oos_return': oos_metrics['total_return'],
        })
    
    # Full sample backtest
    full_returns = backtest_strategy(df, signal_func, name)
    full_metrics = compute_metrics(full_returns)
    
    # Buy and hold comparison
    bh_returns = df['close'].pct_change().fillna(0)
    bh_metrics = compute_metrics(bh_returns)
    
    # Signal stats
    try:
        sigs = signal_func(df)
        n_long = (sigs == 1).sum()
        n_short = (sigs == -1).sum()
        n_flat = (sigs == 0).sum()
        pct_invested = round((1 - n_flat / len(sigs)) * 100, 1)
    except:
        n_long = n_short = n_flat = 0
        pct_invested = 0
    
    # OOS significance test
    oos_sharpes = np.array(oos_results)
    positive_folds = (oos_sharpes > 0).sum()
    mean_oos_sharpe = oos_sharpes.mean() if len(oos_sharpes) > 0 else 0
    
    if len(oos_sharpes) > 1 and oos_sharpes.std() > 0:
        t_stat, p_value = stats.ttest_1samp(oos_sharpes, 0)
        p_value = p_value / 2  # one-tailed
    else:
        t_stat, p_value = 0, 1
    
    return {
        'name': name,
        'full_sample': full_metrics,
        'buy_hold': bh_metrics,
        'oos_mean_sharpe': round(mean_oos_sharpe, 3),
        'oos_positive_folds': f"{positive_folds}/{len(oos_results)}",
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'signal_stats': {
            'long_days': int(n_long),
            'short_days': int(n_short),
            'flat_days': int(n_flat),
            'pct_invested': pct_invested,
        },
        'fold_details': fold_details,
    }


def main():
    print("=" * 70)
    print("  PRICE STRUCTURE STRATEGIES — BTC Walk-Forward Backtest")
    print("=" * 70)
    
    df = load_data()
    print(f"\nData: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} days)")
    print(f"Walk-Forward: {N_FOLDS} folds, expanding IS window\n")
    
    results = {}
    
    for name, func in STRATEGIES.items():
        print(f"Running {name}...")
        result = walk_forward(df, func, name)
        results[name] = result
    
    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    # Buy & hold reference
    bh = list(results.values())[0]['buy_hold']
    print(f"\n📊 Buy & Hold: Sharpe {bh['sharpe']}, CAGR {bh['cagr']}%, MaxDD {bh['max_dd']}%\n")
    
    print(f"{'Strategy':<18} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WinR':>6} {'OOS Sh':>7} {'Pos Folds':>10} {'p-val':>7} {'Sig':>4}")
    print("-" * 80)
    
    for name, r in results.items():
        f = r['full_sample']
        sig = '✅' if r['significant'] else '❌'
        print(f"{name:<18} {f['sharpe']:>7.3f} {f['cagr']:>6.1f}% {f['max_dd']:>6.1f}% {f['win_rate']:>5.1f}% {r['oos_mean_sharpe']:>7.3f} {r['oos_positive_folds']:>10} {r['p_value']:>7.4f} {sig:>4}")
    
    # Signal distribution
    print(f"\n{'Strategy':<18} {'Long':>6} {'Short':>6} {'Flat':>6} {'Invested':>9}")
    print("-" * 50)
    for name, r in results.items():
        s = r['signal_stats']
        print(f"{name:<18} {s['long_days']:>6} {s['short_days']:>6} {s['flat_days']:>6} {s['pct_invested']:>8.1f}%")
    
    # Per-fold details for top strategy
    best = max(results.items(), key=lambda x: x[1]['oos_mean_sharpe'])
    print(f"\n📈 Best OOS: {best[0]} (mean OOS Sharpe: {best[1]['oos_mean_sharpe']})")
    print(f"\nFold details for {best[0]}:")
    print(f"{'Fold':>4} {'OOS Period':<30} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Ret':>8}")
    print("-" * 68)
    for fd in best[1]['fold_details']:
        print(f"{fd['fold']:>4} {fd['oos_period']:<30} {fd['is_sharpe']:>10.3f} {fd['oos_sharpe']:>11.3f} {fd['oos_return']:>7.1f}%")
    
    # Save results
    output_path = os.path.expanduser('~/Desktop/maestro/data/backtest_results/price_structure_btc_wf.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
