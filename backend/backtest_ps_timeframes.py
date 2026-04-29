#!/usr/bin/env python3
"""Walk-forward backtest of 6 price structure strategies across 4 timeframes on BTC."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))

import numpy as np
import pandas as pd
from scipy import stats

STRATEGIES = {
    'market_structure': 'strategies.technical.market_structure',
    'range_sfp': 'strategies.technical.range_sfp',
    'fair_value_gaps': 'strategies.technical.fair_value_gaps',
    'order_blocks': 'strategies.technical.order_blocks',
    'volume_profile': 'strategies.technical.volume_profile',
    'sr_levels': 'strategies.technical.sr_levels',
}

TIMEFRAMES = {
    '1d':  {'file': 'binance_btc_usdt_1d.csv',  'bpy': 365,   'max_bars': None},
    '4h':  {'file': 'binance_btc_usdt_4h.csv',  'bpy': 2190,  'max_bars': None},
    '1h':  {'file': 'binance_btc_usdt_1h.csv',  'bpy': 8760,  'max_bars': 10000},
    '15m': {'file': 'binance_btc_usdt_15m.csv', 'bpy': 35040, 'max_bars': 10000},
}

DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
OUT_DIR = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(tf_info):
    path = os.path.join(DATA_DIR, tf_info['file'])
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    if tf_info['max_bars']:
        df = df.iloc[-tf_info['max_bars']:]
    return df

def backtest(df, signal_func, **params):
    signals = signal_func(df, **params)
    daily_returns = df['close'].pct_change().shift(-1)
    position_changes = signals.diff().abs().fillna(0)
    strategy_returns = (signals * daily_returns - position_changes * 0.001).fillna(0)
    return strategy_returns

def sharpe(returns, bars_per_year):
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

def walk_forward(df, signal_func, bpy, n_folds=10, min_is=200):
    n = len(df)
    if n < min_is + 50:
        return {'oos_sharpes': [], 'mean_oos': 0, 'full_sharpe': 0, 'pos_folds': 0, 'pval': 1.0}
    
    # Full sample
    try:
        full_ret = backtest(df, signal_func)
        full_sh = sharpe(full_ret, bpy)
    except:
        full_sh = 0.0
    
    # Expanding walk-forward
    fold_size = (n - min_is) // n_folds
    if fold_size < 10:
        return {'oos_sharpes': [], 'mean_oos': 0, 'full_sharpe': full_sh, 'pos_folds': 0, 'pval': 1.0}
    
    oos_sharpes = []
    for i in range(n_folds):
        is_end = min_is + i * fold_size
        oos_end = is_end + fold_size
        if oos_end > n:
            oos_end = n
        if is_end >= n:
            break
        try:
            oos_df = df.iloc[is_end:oos_end]
            ret = backtest(df.iloc[:oos_end], signal_func)
            oos_ret = ret.iloc[is_end:oos_end]
            oos_sharpes.append(sharpe(oos_ret, bpy))
        except:
            oos_sharpes.append(0.0)
    
    mean_oos = np.mean(oos_sharpes) if oos_sharpes else 0.0
    pos_folds = sum(1 for s in oos_sharpes if s > 0)
    
    if len(oos_sharpes) >= 2 and np.std(oos_sharpes) > 0:
        t_stat, p_two = stats.ttest_1samp(oos_sharpes, 0)
        pval = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        pval = 1.0
    
    return {
        'oos_sharpes': oos_sharpes,
        'mean_oos': round(mean_oos, 3),
        'full_sharpe': round(full_sh, 3),
        'pos_folds': pos_folds,
        'pval': round(pval, 4),
    }

def load_strategy(module_path):
    import importlib
    mod = importlib.import_module(module_path)
    return mod.generate_signals

def main():
    all_results = {}
    
    for tf_name, tf_info in TIMEFRAMES.items():
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {tf_name}")
        print(f"{'='*60}")
        
        df = load_data(tf_info)
        bpy = tf_info['bpy']
        print(f"  Loaded {len(df)} bars")
        
        # Buy and hold
        bh_ret = df['close'].pct_change().shift(-1).fillna(0)
        bh_sharpe = sharpe(bh_ret, bpy)
        print(f"  Buy & Hold Sharpe: {bh_sharpe:.3f}")
        
        tf_results = {'buy_hold_sharpe': round(bh_sharpe, 3)}
        
        for strat_name, module_path in STRATEGIES.items():
            print(f"  Running {strat_name}...", end=' ', flush=True)
            try:
                sig_func = load_strategy(module_path)
                result = walk_forward(df, sig_func, bpy)
                tf_results[strat_name] = result
                sig = '***' if result['pval'] < 0.01 else '**' if result['pval'] < 0.05 else '*' if result['pval'] < 0.1 else ''
                print(f"OOS Sharpe: {result['mean_oos']:.3f}  Full: {result['full_sharpe']:.3f}  "
                      f"Pos: {result['pos_folds']}/10  p={result['pval']:.4f} {sig}")
            except Exception as e:
                print(f"FAILED: {e}")
                tf_results[strat_name] = {'mean_oos': 0, 'full_sharpe': 0, 'pos_folds': 0, 'pval': 1.0, 'oos_sharpes': []}
        
        all_results[tf_name] = tf_results
    
    # GRAND SUMMARY 1
    print(f"\n\n{'='*100}")
    print("GRAND SUMMARY — Best Timeframe Per Strategy")
    print(f"{'='*100}")
    header = f"{'Strategy':<20} | {'1D OOS':>8} | {'4H OOS':>8} | {'1H OOS':>8} | {'15m OOS':>8} | {'Best TF':>8} | {'Best OOS':>10} | {'p-value':>8}"
    print(header)
    print('-' * len(header))
    
    for strat_name in STRATEGIES:
        row = {}
        for tf in TIMEFRAMES:
            r = all_results[tf].get(strat_name, {})
            row[tf] = r.get('mean_oos', 0)
        best_tf = max(row, key=row.get)
        best_oos = row[best_tf]
        best_pval = all_results[best_tf].get(strat_name, {}).get('pval', 1.0)
        print(f"{strat_name:<20} | {row['1d']:>8.3f} | {row['4h']:>8.3f} | {row['1h']:>8.3f} | {row['15m']:>8.3f} | {best_tf:>8} | {best_oos:>10.3f} | {best_pval:>8.4f}")
    
    # GRAND SUMMARY 2
    print(f"\n{'='*80}")
    print("GRAND SUMMARY — Best Strategy Per Timeframe")
    print(f"{'='*80}")
    header2 = f"{'Timeframe':<12} | {'Best Strategy':<20} | {'OOS Sharpe':>10} | {'p-value':>8} | {'Sig':>5}"
    print(header2)
    print('-' * len(header2))
    
    for tf in TIMEFRAMES:
        best_strat = None
        best_oos = -999
        for strat_name in STRATEGIES:
            r = all_results[tf].get(strat_name, {})
            oos = r.get('mean_oos', 0)
            if oos > best_oos:
                best_oos = oos
                best_strat = strat_name
        pval = all_results[tf].get(best_strat, {}).get('pval', 1.0)
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        bh = all_results[tf].get('buy_hold_sharpe', 0)
        print(f"{tf:<12} | {best_strat:<20} | {best_oos:>10.3f} | {pval:>8.4f} | {sig:>5}  (B&H: {bh:.3f})")
    
    # Save JSON
    # Convert for JSON serialization
    save_results = {}
    for tf, data in all_results.items():
        save_results[tf] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                save_results[tf][k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else 
                                             [float(x) for x in vv] if isinstance(vv, list) else vv)
                                        for kk, vv in v.items()}
            else:
                save_results[tf][k] = float(v) if isinstance(v, (np.floating, float)) else v
    
    out_path = os.path.join(OUT_DIR, 'price_structure_timeframes.json')
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
