#!/usr/bin/env python3
"""
Walk-forward backtest for CAS and CTUS derivatives signals.

- 10-fold expanding walk-forward
- Multiple parameter combos
- 200 random permutation test for best config
- Commission: 20bps round-trip
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import product
from datetime import datetime

from strategies.derivatives.cas_signal import generate_signals as cas_signals
from strategies.derivatives.ctus_signal import generate_signals as ctus_signals

DATA_DIR = Path(os.path.expanduser('~/Desktop/maestro/data'))
DERIV_DIR = DATA_DIR / 'derivatives'
OHLCV_DIR = DATA_DIR / 'ohlcv'
RESULTS_DIR = DATA_DIR / 'backtest_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = ['btc', 'eth', 'sol', 'arb', 'op', 'sui', 'tia', 'inj', 'link', 'avax', 'fet', 'tao', 'render']
COMMISSION_BPS = 20  # round-trip
COMMISSION = COMMISSION_BPS / 10000

# Parameter grids
CAS_PARAMS = [
    {'oi_drop_z': -1.5, 'funding_extreme_z': 1.5, 'lsr_extreme_z': 1.0, 'lookback': 20, 'absorption_bars': 2, 'hold_bars': 3},
    {'oi_drop_z': -2.0, 'funding_extreme_z': 1.5, 'lsr_extreme_z': 1.0, 'lookback': 30, 'absorption_bars': 2, 'hold_bars': 5},
    {'oi_drop_z': -1.0, 'funding_extreme_z': 1.0, 'lsr_extreme_z': 0.8, 'lookback': 20, 'absorption_bars': 1, 'hold_bars': 3},
    {'oi_drop_z': -1.5, 'funding_extreme_z': 2.0, 'lsr_extreme_z': 1.5, 'lookback': 40, 'absorption_bars': 3, 'hold_bars': 5},
    {'oi_drop_z': -2.0, 'funding_extreme_z': 2.0, 'lsr_extreme_z': 1.5, 'lookback': 30, 'absorption_bars': 2, 'hold_bars': 7},
    {'oi_drop_z': -1.2, 'funding_extreme_z': 1.2, 'lsr_extreme_z': 1.0, 'lookback': 25, 'absorption_bars': 2, 'hold_bars': 4},
]

CTUS_PARAMS = [
    {'oi_z_thresh': 1.5, 'funding_z_thresh': 1.5, 'lsr_z_thresh': 1.0, 'lookback': 30, 'hold_bars': 3},
    {'oi_z_thresh': 1.0, 'funding_z_thresh': 1.0, 'lsr_z_thresh': 0.8, 'lookback': 20, 'hold_bars': 3},
    {'oi_z_thresh': 2.0, 'funding_z_thresh': 2.0, 'lsr_z_thresh': 1.5, 'lookback': 30, 'hold_bars': 5},
    {'oi_z_thresh': 1.5, 'funding_z_thresh': 1.5, 'lsr_z_thresh': 1.0, 'lookback': 40, 'hold_bars': 5},
    {'oi_z_thresh': 1.0, 'funding_z_thresh': 1.5, 'lsr_z_thresh': 1.0, 'lookback': 25, 'hold_bars': 4},
    {'oi_z_thresh': 1.2, 'funding_z_thresh': 1.2, 'lsr_z_thresh': 0.8, 'lookback': 20, 'hold_bars': 5},
]


def load_token_data(token: str) -> pd.DataFrame:
    """Load and merge OHLCV + derivatives data for a token."""
    # Price data
    price_file = OHLCV_DIR / f'binance_{token}_usdt_1d.csv'
    if not price_file.exists():
        return pd.DataFrame()
    price = pd.read_csv(price_file, parse_dates=['timestamp'])
    price = price[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # OI
    oi_file = DERIV_DIR / f'{token}_oi_daily_full.csv'
    if not oi_file.exists():
        return pd.DataFrame()
    oi = pd.read_csv(oi_file, parse_dates=['timestamp'])
    oi = oi.rename(columns={'o': 'oi_open', 'h': 'oi_high', 'l': 'oi_low', 'c': 'oi_close'})

    # Funding
    funding_file = DERIV_DIR / f'{token}_funding_full.csv'
    if funding_file.exists():
        funding = pd.read_csv(funding_file, parse_dates=['timestamp'])
    else:
        funding = pd.DataFrame()

    # LSR
    lsr_file = DERIV_DIR / f'{token}_lsr_daily_full.csv'
    if lsr_file.exists():
        lsr = pd.read_csv(lsr_file, parse_dates=['timestamp'])
    else:
        lsr = pd.DataFrame()

    # Merge all on date
    price['date'] = price['timestamp'].dt.date
    oi['date'] = oi['timestamp'].dt.date

    df = price.merge(oi[['date', 'oi_open', 'oi_high', 'oi_low', 'oi_close']], on='date', how='inner')

    if not funding.empty:
        funding['date'] = funding['timestamp'].dt.date
        df = df.merge(funding[['date', 'funding_rate']], on='date', how='left')
    else:
        df['funding_rate'] = 0.0

    if not lsr.empty:
        lsr['date'] = lsr['timestamp'].dt.date
        df = df.merge(lsr[['date', 'long_ratio', 'short_ratio']], on='date', how='left')
    else:
        df['long_ratio'] = 50.0
        df['short_ratio'] = 50.0

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['funding_rate'] = df['funding_rate'].fillna(0)
    df['long_ratio'] = df['long_ratio'].fillna(50)
    df['short_ratio'] = df['short_ratio'].fillna(50)

    return df


def backtest_signals(df: pd.DataFrame, signals: pd.Series) -> dict:
    """Simple vectorized backtest with commission."""
    returns = df['close'].pct_change().fillna(0)
    
    # Position changes for commission
    pos_changes = signals.diff().abs().fillna(0)
    # Each change costs commission/2 (entry or exit), full round-trip on signal flip
    commissions = pos_changes * COMMISSION / 2
    
    strategy_returns = signals.shift(1).fillna(0) * returns - commissions
    
    cum_ret = (1 + strategy_returns).cumprod()
    total_return = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    # Stats
    n_days = len(strategy_returns)
    ann_factor = np.sqrt(365)
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = mean_ret / std_ret * ann_factor if std_ret > 0 else 0
    
    # Trade count
    trades = (signals.diff().abs() > 0).sum()
    
    # Win rate
    active = strategy_returns[signals.shift(1) != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    
    # Max drawdown
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return {
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'trades': int(trades),
        'win_rate': float(win_rate),
        'max_drawdown': float(max_dd),
        'n_days': int(n_days),
        'trades_per_month': float(trades / max(n_days / 30, 1)),
    }


def walk_forward_test(df: pd.DataFrame, signal_fn, params: dict, n_folds: int = 10) -> dict:
    """Expanding walk-forward with n_folds."""
    n = len(df)
    min_train = max(params.get('lookback', 30) * 3, 60)
    
    if n < min_train + 30:
        return {'oos_sharpe': 0, 'oos_trades': 0, 'oos_win_rate': 0, 'n_folds_used': 0}
    
    fold_size = (n - min_train) // n_folds
    if fold_size < 10:
        n_folds = max(1, (n - min_train) // 10)
        fold_size = (n - min_train) // n_folds
    
    oos_returns = []
    oos_signals_all = []
    
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if train_end >= n or test_end <= train_end:
            break
        
        # Generate signals on full data up to test_end (expanding window)
        test_df = df.iloc[:test_end].copy()
        sigs = signal_fn(test_df, **params)
        
        # Only evaluate OOS portion
        oos_sigs = sigs.iloc[train_end:test_end]
        oos_rets = df['close'].pct_change().fillna(0).iloc[train_end:test_end]
        
        pos_changes = oos_sigs.diff().abs().fillna(0)
        commissions = pos_changes * COMMISSION / 2
        strat_rets = oos_sigs.shift(1).fillna(0) * oos_rets - commissions
        
        oos_returns.extend(strat_rets.values)
        oos_signals_all.extend(oos_sigs.values)
    
    if not oos_returns:
        return {'oos_sharpe': 0, 'oos_trades': 0, 'oos_win_rate': 0, 'n_folds_used': 0}
    
    oos_returns = np.array(oos_returns)
    oos_signals = np.array(oos_signals_all)
    
    mean_r = oos_returns.mean()
    std_r = oos_returns.std()
    sharpe = mean_r / std_r * np.sqrt(365) if std_r > 0 else 0
    
    trades = (np.abs(np.diff(np.concatenate([[0], oos_signals]))) > 0).sum()
    active = oos_returns[np.roll(oos_signals, 1) != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    total_ret = float(np.prod(1 + oos_returns) - 1)
    
    return {
        'oos_sharpe': float(sharpe),
        'oos_total_return': total_ret,
        'oos_trades': int(trades),
        'oos_win_rate': float(win_rate),
        'oos_trades_per_month': float(trades / max(len(oos_returns) / 30, 1)),
        'n_folds_used': n_folds,
        'oos_days': len(oos_returns),
    }


def permutation_test(df: pd.DataFrame, signal_fn, params: dict, n_perms: int = 200) -> float:
    """Shuffle signal dates, compute p-value of observed Sharpe."""
    # Get observed signals and Sharpe
    sigs = signal_fn(df, **params)
    observed = backtest_signals(df, sigs)
    obs_sharpe = observed['sharpe']
    
    if observed['trades'] < 5:
        return 1.0  # Not enough trades
    
    count_better = 0
    returns = df['close'].pct_change().fillna(0)
    
    for _ in range(n_perms):
        # Shuffle signal assignments
        shuffled = sigs.copy()
        non_zero_mask = sigs != 0
        non_zero_vals = sigs[non_zero_mask].values.copy()
        np.random.shuffle(non_zero_vals)
        shuffled[non_zero_mask] = non_zero_vals
        
        # Also randomly shift positions
        shift = np.random.randint(1, len(df))
        shuffled = pd.Series(np.roll(shuffled.values, shift), index=sigs.index)
        
        perm_result = backtest_signals(df, shuffled)
        if perm_result['sharpe'] >= obs_sharpe:
            count_better += 1
    
    return (count_better + 1) / (n_perms + 1)


def main():
    print("=" * 70)
    print("DERIVATIVES SIGNALS WALK-FORWARD BACKTEST")
    print("=" * 70)
    print(f"Tokens: {len(TOKENS)}")
    print(f"CAS param combos: {len(CAS_PARAMS)}")
    print(f"CTUS param combos: {len(CTUS_PARAMS)}")
    print(f"Commission: {COMMISSION_BPS}bps round-trip")
    print()

    all_results = {'cas': {}, 'ctus': {}, 'summary': {}}

    for token in TOKENS:
        print(f"\n{'='*60}")
        print(f"TOKEN: {token.upper()}")
        print('=' * 60)

        df = load_token_data(token)
        if df.empty or len(df) < 100:
            print(f"  Insufficient data ({len(df)} rows), skipping")
            continue

        print(f"  Data: {len(df)} days ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

        # --- CAS ---
        print(f"\n  --- CAS ({len(CAS_PARAMS)} configs) ---")
        best_cas_sharpe = -999
        best_cas_params = None
        best_cas_result = None

        for i, params in enumerate(CAS_PARAMS):
            result = walk_forward_test(df, cas_signals, params, n_folds=10)
            tag = f"cas_{i}"
            
            # Full backtest for reference
            full_sigs = cas_signals(df, **params)
            full_result = backtest_signals(df, full_sigs)
            
            print(f"    Config {i}: OOS Sharpe={result['oos_sharpe']:.3f}, "
                  f"Trades={result['oos_trades']}, "
                  f"WR={result['oos_win_rate']:.1%}, "
                  f"Ret={result.get('oos_total_return', 0):.1%}, "
                  f"Full Sharpe={full_result['sharpe']:.3f}")
            
            all_results['cas'][f'{token}_{tag}'] = {
                'params': params,
                'oos': result,
                'full': full_result,
            }
            
            if result['oos_sharpe'] > best_cas_sharpe and result['oos_trades'] >= 3:
                best_cas_sharpe = result['oos_sharpe']
                best_cas_params = params
                best_cas_result = result

        # --- CTUS ---
        print(f"\n  --- CTUS ({len(CTUS_PARAMS)} configs) ---")
        best_ctus_sharpe = -999
        best_ctus_params = None
        best_ctus_result = None

        for i, params in enumerate(CTUS_PARAMS):
            result = walk_forward_test(df, ctus_signals, params, n_folds=10)
            tag = f"ctus_{i}"
            
            full_sigs = ctus_signals(df, **params)
            full_result = backtest_signals(df, full_sigs)
            
            print(f"    Config {i}: OOS Sharpe={result['oos_sharpe']:.3f}, "
                  f"Trades={result['oos_trades']}, "
                  f"WR={result['oos_win_rate']:.1%}, "
                  f"Ret={result.get('oos_total_return', 0):.1%}, "
                  f"Full Sharpe={full_result['sharpe']:.3f}")
            
            all_results['ctus'][f'{token}_{tag}'] = {
                'params': params,
                'oos': result,
                'full': full_result,
            }
            
            if result['oos_sharpe'] > best_ctus_sharpe and result['oos_trades'] >= 3:
                best_ctus_sharpe = result['oos_sharpe']
                best_ctus_params = params
                best_ctus_result = result

        # Permutation test for best configs
        print(f"\n  Best CAS: Sharpe={best_cas_sharpe:.3f}")
        if best_cas_params and best_cas_sharpe > 0:
            print(f"    Running 200 permutations...")
            p_val = permutation_test(df, cas_signals, best_cas_params, 200)
            print(f"    p-value: {p_val:.4f}")
            all_results['summary'][f'{token}_cas'] = {
                'best_sharpe': best_cas_sharpe,
                'best_params': best_cas_params,
                'p_value': p_val,
                'result': best_cas_result,
            }
        
        print(f"  Best CTUS: Sharpe={best_ctus_sharpe:.3f}")
        if best_ctus_params and best_ctus_sharpe > 0:
            print(f"    Running 200 permutations...")
            p_val = permutation_test(df, ctus_signals, best_ctus_params, 200)
            print(f"    p-value: {p_val:.4f}")
            all_results['summary'][f'{token}_ctus'] = {
                'best_sharpe': best_ctus_sharpe,
                'best_params': best_ctus_params,
                'p_value': p_val,
                'result': best_ctus_result,
            }

    # Save results
    output_file = RESULTS_DIR / 'derivatives_signals.json'
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print('=' * 70)
    
    for key, val in sorted(all_results['summary'].items()):
        pv = val.get('p_value', 'N/A')
        res = val.get('result', {})
        print(f"  {key}: Sharpe={val['best_sharpe']:.3f}, "
              f"p={pv:.4f}, "
              f"Trades={res.get('oos_trades', 0)}, "
              f"WR={res.get('oos_win_rate', 0):.1%}, "
              f"Ret={res.get('oos_total_return', 0):.1%}")
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
