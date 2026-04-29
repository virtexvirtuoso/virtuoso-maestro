#!/usr/bin/env python3
"""Optuna hyperparameter optimization for 6 price structure strategies across 4 timeframes."""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/Desktop/maestro/backend'))

import numpy as np
import pandas as pd
import optuna
from scipy import stats
optuna.logging.set_verbosity(optuna.logging.WARNING)

from strategies.technical.market_structure import generate_signals as market_structure_signals
from strategies.technical.range_sfp import generate_signals as range_sfp_signals
from strategies.technical.fair_value_gaps import generate_signals as fair_value_gaps_signals
from strategies.technical.order_blocks import generate_signals as order_blocks_signals
from strategies.technical.volume_profile import generate_signals as _volume_profile_signals_orig
from strategies.technical.sr_levels import generate_signals as sr_levels_signals


def volume_profile_signals_fast(df, bins=50, value_area_pct=0.70, lookback=30):
    """Vectorized volume profile - much faster than the original O(n*lookback*bins) version."""
    signals = pd.Series(0, index=df.index)
    n = len(df)
    if n < lookback:
        return signals

    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    for i in range(lookback, n):
        start = i - lookback
        w_high = high[start:i]
        w_low = low[start:i]
        w_vol = volume[start:i]

        price_min = w_low.min()
        price_max = w_high.max()
        if price_max <= price_min:
            continue

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_width = (price_max - price_min) / bins
        vol_profile = np.zeros(bins)

        # Vectorized: for each bar, find which bins it spans
        bar_lo_bins = np.clip(((w_low - price_min) / bin_width).astype(int), 0, bins - 1)
        bar_hi_bins = np.clip(((w_high - price_min) / bin_width).astype(int), 0, bins - 1)
        n_bins_per_bar = bar_hi_bins - bar_lo_bins + 1
        vol_per_bin = w_vol / np.maximum(n_bins_per_bar, 1)

        for j in range(lookback):
            vol_profile[bar_lo_bins[j]:bar_hi_bins[j]+1] += vol_per_bin[j]

        total_vol = vol_profile.sum()
        if total_vol == 0:
            continue

        poc_bin = np.argmax(vol_profile)
        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2

        # Value Area
        va_vol = vol_profile[poc_bin]
        lo_idx = poc_bin - 1
        hi_idx = poc_bin + 1
        target_vol = total_vol * value_area_pct

        while va_vol < target_vol and (lo_idx >= 0 or hi_idx < bins):
            add_lo = vol_profile[lo_idx] if lo_idx >= 0 else 0
            add_hi = vol_profile[hi_idx] if hi_idx < bins else 0
            if add_lo >= add_hi and lo_idx >= 0:
                va_vol += add_lo
                lo_idx -= 1
            elif hi_idx < bins:
                va_vol += add_hi
                hi_idx += 1
            else:
                va_vol += add_lo
                lo_idx -= 1

        val = bin_edges[max(lo_idx + 1, 0)]
        vah = bin_edges[min(hi_idx, bins)]

        if close[i] < val:
            signals.iloc[i] = 1
        elif close[i] > vah:
            signals.iloc[i] = -1

    return signals


STRATEGIES = {
    'market_structure': market_structure_signals,
    'range_sfp': range_sfp_signals,
    'fair_value_gaps': fair_value_gaps_signals,
    'order_blocks': order_blocks_signals,
    'volume_profile': volume_profile_signals_fast,
    'sr_levels': sr_levels_signals,
}

SPACES = {
    'market_structure': {'swing_window': ('int', 3, 20), 'min_swings': ('int', 2, 6)},
    'range_sfp': {'lookback': ('int', 15, 120), 'sfp_threshold': ('float', 0.001, 0.03), 'atr_period': ('int', 8, 25)},
    'fair_value_gaps': {'lookback': ('int', 15, 120), 'proximity_pct': ('float', 0.003, 0.05), 'max_gap_age': ('int', 5, 60)},
    'order_blocks': {'body_threshold': ('float', 0.001, 0.02), 'vol_threshold': ('float', 1.0, 3.0), 'expansion_factor': ('float', 1.0, 3.0), 'max_blocks': ('int', 2, 15)},
    'volume_profile': {'bins': ('int', 15, 120), 'value_area_pct': ('float', 0.4, 0.9), 'lookback': ('int', 10, 80)},
    'sr_levels': {'swing_window': ('int', 3, 25), 'group_threshold': ('float', 0.001, 0.03), 'proximity_pct': ('float', 0.003, 0.05), 'min_touches': ('int', 1, 5)},
}

# Strategies known to be slow on large datasets - cap bars
SLOW_STRATEGIES = {'volume_profile': 3000, 'fair_value_gaps': 5000, 'order_blocks': 5000}

DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
TIMEFRAMES = {
    '1d':  ('binance_btc_usdt_1d.csv',  365,   None),
    '4h':  ('binance_btc_usdt_4h.csv',  2190,  None),
    '1h':  ('binance_btc_usdt_1h.csv',  8760,  10000),
    '15m': ('binance_btc_usdt_15m.csv', 35040, 10000),
}

def load_data(filename, max_bars):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    if max_bars and len(df) > max_bars:
        df = df.iloc[-max_bars:].reset_index(drop=True)
    return df

def compute_returns(df, signal_func, **params):
    try:
        signals = signal_func(df, **params)
    except:
        return pd.Series(0.0, index=df.index)
    daily_ret = df['close'].pct_change().shift(-1)
    pos_changes = signals.diff().abs().fillna(0)
    return (signals * daily_ret - pos_changes * 0.001).fillna(0)

def sharpe(returns, bars_per_year):
    if len(returns) == 0 or returns.std() == 0:
        return -999.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

def cagr(returns, bars_per_year):
    cum = (1 + returns).prod()
    n_years = len(returns) / bars_per_year
    if n_years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1 / n_years) - 1)

def max_dd(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()) if len(dd) > 0 else 0.0

def win_rate(returns):
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return float((trades > 0).sum() / len(trades))

def suggest_params(trial, strategy_name):
    params = {}
    for name, (typ, lo, hi) in SPACES[strategy_name].items():
        if typ == 'int':
            params[name] = trial.suggest_int(name, lo, hi)
        else:
            params[name] = trial.suggest_float(name, lo, hi)
    return params

def optimize(df, signal_func, strategy_name, bars_per_year, n_trials=150, seed=42, timeout=300):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    def objective(trial):
        params = suggest_params(trial, strategy_name)
        rets = compute_returns(df, signal_func, **params)
        return sharpe(rets, bars_per_year)
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    if len(study.trials) == 0 or study.best_trial is None:
        # Return default params
        defaults = {}
        for name, (typ, lo, hi) in SPACES[strategy_name].items():
            defaults[name] = (lo + hi) // 2 if typ == 'int' else (lo + hi) / 2
        return defaults, -999.0
    return study.best_params, study.best_value

def walk_forward(df, signal_func, strategy_name, bars_per_year, n_folds=10, trials_per_fold=50):
    n = len(df)
    min_is = n // (n_folds + 1)
    oos_sharpes = []
    
    for fold in range(n_folds):
        is_end = min_is + fold * (n - min_is) // n_folds
        oos_end = min_is + (fold + 1) * (n - min_is) // n_folds
        if oos_end > n:
            oos_end = n
        
        df_is = df.iloc[:is_end].reset_index(drop=True)
        df_oos = df.iloc[is_end:oos_end].reset_index(drop=True)
        
        if len(df_oos) < 10:
            continue
        
        best_params, _ = optimize(df_is, signal_func, strategy_name, bars_per_year, 
                                   n_trials=trials_per_fold, seed=42+fold, timeout=120)
        oos_rets = compute_returns(df_oos, signal_func, **best_params)
        oos_sharpes.append(sharpe(oos_rets, bars_per_year))
    
    if len(oos_sharpes) < 3:
        return 0.0, 1.0, oos_sharpes
    
    mean_sharpe = np.mean(oos_sharpes)
    t_stat, p_val = stats.ttest_1samp(oos_sharpes, 0)
    p_val = float(p_val) if not np.isnan(p_val) else 1.0
    return float(mean_sharpe), p_val, oos_sharpes

# === MAIN ===
all_results = []
total_start = time.time()

for tf_name, (filename, bpy, max_bars) in TIMEFRAMES.items():
    print(f"\n{'='*60}")
    print(f"=== {tf_name.upper()} TIMEFRAME ===")
    print(f"{'='*60}")
    
    df_full = load_data(filename, max_bars)
    print(f"Loaded {len(df_full)} bars")
    
    tf_results = []
    
    for strat_name, signal_func in STRATEGIES.items():
        t0 = time.time()
        print(f"  Optimizing {strat_name} on {tf_name}...", end=' ', flush=True)
        
        # Cap data for slow strategies
        df = df_full
        cap = SLOW_STRATEGIES.get(strat_name)
        if cap and len(df) > cap:
            df = df.iloc[-cap:].reset_index(drop=True)
        
        split = int(len(df) * 0.7)
        df_is = df.iloc[:split].reset_index(drop=True)
        df_oos = df.iloc[split:].reset_index(drop=True)
        
        # Step 2: Optuna on IS
        best_params, is_sharpe = optimize(df_is, signal_func, strat_name, bpy, n_trials=150, timeout=300)
        
        # Step 3: Evaluate on OOS
        oos_rets = compute_returns(df_oos, signal_func, **best_params)
        oos_sharpe = sharpe(oos_rets, bpy)
        oos_cagr = cagr(oos_rets, bpy)
        oos_maxdd = max_dd(oos_rets)
        oos_total = float((1 + oos_rets).prod() - 1)
        oos_wr = win_rate(oos_rets)
        
        # Step 4: Walk-forward
        wf_mean, wf_pval, wf_sharpes = walk_forward(df, signal_func, strat_name, bpy)
        
        elapsed = time.time() - t0
        sig = "✓" if wf_pval < 0.05 else "✗"
        print(f"done (IS: {is_sharpe:.2f}, OOS: {oos_sharpe:.2f}, WF: {wf_mean:.2f}, p={wf_pval:.3f} {sig}) [{elapsed:.0f}s]")
        
        result = {
            'strategy': strat_name,
            'timeframe': tf_name,
            'is_sharpe': round(is_sharpe, 3),
            'oos_sharpe': round(oos_sharpe, 3),
            'oos_cagr': round(oos_cagr, 4),
            'oos_maxdd': round(oos_maxdd, 4),
            'oos_total_return': round(oos_total, 4),
            'oos_win_rate': round(oos_wr, 4),
            'wf_mean_sharpe': round(wf_mean, 3),
            'wf_pval': round(wf_pval, 4),
            'significant': wf_pval < 0.05,
            'best_params': best_params,
            'wf_fold_sharpes': [round(s, 3) for s in wf_sharpes],
        }
        tf_results.append(result)
        all_results.append(result)
    
    # Print per-timeframe table
    print(f"\n{'Strategy':<20} | {'IS Sharpe':>9} | {'OOS Sharpe':>10} | {'OOS CAGR':>8} | {'OOS MaxDD':>9} | {'WF Mean':>7} | {'WF p-val':>8} | {'Sig':>3} | Best Params")
    print('-' * 150)
    for r in tf_results:
        params_str = ', '.join(f"{k}={v}" for k, v in r['best_params'].items())
        print(f"{r['strategy']:<20} | {r['is_sharpe']:>9.3f} | {r['oos_sharpe']:>10.3f} | {r['oos_cagr']:>7.2%} | {r['oos_maxdd']:>8.2%} | {r['wf_mean_sharpe']:>7.3f} | {r['wf_pval']:>8.4f} | {'✓' if r['significant'] else '✗':>3} | {params_str}")

# Grand summary
print(f"\n{'='*80}")
print("GRAND SUMMARY — All Combos Ranked by OOS Sharpe")
print(f"{'='*80}")
ranked = sorted(all_results, key=lambda x: x['oos_sharpe'], reverse=True)

print(f"{'Rank':>4} | {'Strategy + TF':<30} | {'IS Sharpe':>9} | {'OOS Sharpe':>10} | {'WF Mean':>7} | {'p-value':>8} | {'Sig':>3}")
print('-' * 85)

show_indices = list(range(min(10, len(ranked))))
if len(ranked) > 15:
    show_indices += list(range(len(ranked)-5, len(ranked)))
else:
    show_indices = list(range(len(ranked)))

for idx, i in enumerate(sorted(set(show_indices))):
    if i >= len(ranked):
        continue
    r = ranked[i]
    label = f"{r['strategy']}_{r['timeframe']}"
    if i == min(10, len(ranked)) and len(ranked) > 15:
        print(f"{'...':>4} | {'...':^30} | {'...':>9} | {'...':>10} | {'...':>7} | {'...':>8} | {'...':>3}")
    print(f"{i+1:>4} | {label:<30} | {r['is_sharpe']:>9.3f} | {r['oos_sharpe']:>10.3f} | {r['wf_mean_sharpe']:>7.3f} | {r['wf_pval']:>8.4f} | {'✓' if r['significant'] else '✗':>3}")

# Significant results
sig_results = [r for r in ranked if r['significant']]
print(f"\n{'='*80}")
print(f"SIGNIFICANT RESULTS (p < 0.05): {len(sig_results)} of {len(all_results)}")
print(f"{'='*80}")
if sig_results:
    print(f"{'Strategy + TF':<30} | {'OOS Sharpe':>10} | {'WF Mean':>7} | {'p-value':>8} | Best Params")
    print('-' * 110)
    for r in sig_results:
        label = f"{r['strategy']}_{r['timeframe']}"
        params_str = ', '.join(f"{k}={v}" for k, v in r['best_params'].items())
        print(f"{label:<30} | {r['oos_sharpe']:>10.3f} | {r['wf_mean_sharpe']:>7.3f} | {r['wf_pval']:>8.4f} | {params_str}")
else:
    print("None — no strategy achieved statistical significance.")

# Save JSON
out_path = os.path.expanduser('~/Desktop/maestro/data/optimization/ps_optuna_timeframes.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print(f"Total time: {time.time()-total_start:.0f}s")
