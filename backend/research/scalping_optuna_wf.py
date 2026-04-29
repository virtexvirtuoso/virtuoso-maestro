#!/usr/bin/env python3
"""
Optuna Walk-Forward Scalping Optimization
==========================================
For each (asset, strategy_type, session, hold, cost_tier):
  - 5-fold expanding window
  - Each IS fold: Optuna optimizes continuous params (50 trials)
  - Each OOS fold: test best params, collect trades
  - Final metrics computed on ALL OOS trades

This is the gold standard: Bayesian optimization + proper walk-forward.
"""

import pandas as pd
import numpy as np
import optuna
import os, sys, gc, time, warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_metrics, grade_strategy

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/spot/5m/")
RESULTS_DIR = os.path.expanduser("~/Desktop/maestro/results/")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "scalping_optuna_wf.csv")

# Focus on assets that showed promise
ASSETS = ['ETH', 'BTC', 'NEAR', 'DOGE', 'AVAX', 'SOL', 'SUI', 'XRP']

# Realistic cost tiers (round-trip)
COST_TIERS = {
    'limit_both_4bps': 0.0004,
    'realistic_7bps': 0.0007,
}

HOLD_BARS = {'30m': 6, '1h': 12, '2h': 24, '4h': 48}
SESSIONS = ['all', 'us', 'asian']
STRAT_TYPES = ['trend', 'mr', 'kelt_stoch']

N_FOLDS = 5
MIN_IS_BARS = 10000  # need enough data for Optuna to optimize meaningfully
OPTUNA_TRIALS = 30   # per IS fold (TPE converges fast)

# ============================================================
# SIGNAL GENERATORS (numpy, backward-looking only)
# ============================================================

def ema(close_arr, span):
    """Exponential moving average — vectorized via pandas"""
    return pd.Series(close_arr).ewm(span=span, adjust=False).mean().values

def rolling_mean(arr, period):
    """Simple rolling mean — vectorized"""
    s = pd.Series(arr)
    return s.rolling(period).mean().values

def rolling_std(arr, period):
    """Rolling standard deviation — vectorized"""
    s = pd.Series(arr)
    return s.rolling(period).std().values

def compute_rsi(close_arr, period):
    delta = np.diff(close_arr, prepend=close_arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rolling_mean(gain, period)
    avg_loss = rolling_mean(loss, period)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
        rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return rolling_mean(tr, period)


# ============================================================
# STRATEGY SIGNAL GENERATORS WITH CONTINUOUS PARAMS
# ============================================================

def gen_trend_signal(close, high, low, volume, open_arr,
                     ema_fast, ema_slow, mom_period, trend_period):
    """Trend: EMA cross + momentum, filtered by trend SMA"""
    ef = ema(close, int(ema_fast))
    es = ema(close, int(ema_slow))
    
    # Momentum
    mom_p = int(mom_period)
    mom = np.zeros(len(close))
    mom[mom_p:] = close[mom_p:] / close[:-mom_p] - 1
    
    # EMA cross signal
    s1 = np.zeros(len(close), dtype=np.int8)
    s1[ef > es] = 1
    s1[ef < es] = -1
    
    # Momentum signal
    s2 = np.zeros(len(close), dtype=np.int8)
    s2[mom > 0] = 1
    s2[mom < 0] = -1
    
    # Combined: both must agree
    combined = np.zeros(len(close), dtype=np.int8)
    combined[(s1 == 1) & (s2 == 1)] = 1
    combined[(s1 == -1) & (s2 == -1)] = -1
    
    # Trend filter
    tp = int(trend_period)
    sma = rolling_mean(close, tp)
    combined[(combined == 1) & (close < sma)] = 0
    combined[(combined == -1) & (close > sma)] = 0
    
    return combined


def gen_mr_signal(close, high, low, volume, open_arr,
                  boll_period, boll_std, rsi_period, rsi_ob, rsi_os,
                  vol_filter_period):
    """Mean reversion: Bollinger + RSI confluence, vol filtered"""
    bp = int(boll_period)
    sma = rolling_mean(close, bp)
    std = rolling_std(close, bp)
    
    s1 = np.zeros(len(close), dtype=np.int8)
    with np.errstate(invalid='ignore'):
        lower = sma - boll_std * std
        upper = sma + boll_std * std
    s1[close < lower] = 1
    s1[close > upper] = -1
    
    rsi = compute_rsi(close, int(rsi_period))
    s2 = np.zeros(len(close), dtype=np.int8)
    s2[rsi < rsi_os] = 1
    s2[rsi > rsi_ob] = -1
    
    combined = np.zeros(len(close), dtype=np.int8)
    combined[(s1 == 1) & (s2 == 1)] = 1
    combined[(s1 == -1) & (s2 == -1)] = -1
    
    # Vol filter
    vfp = int(vol_filter_period)
    ret_vol = rolling_std(np.diff(close, prepend=close[0]) / np.maximum(close, 1e-10), vfp)
    # Simple percentile filter: skip if vol too extreme
    vol_ma = rolling_mean(ret_vol, min(500, len(close)//4))
    vol_std = rolling_std(ret_vol, min(500, len(close)//4))
    with np.errstate(invalid='ignore'):
        vol_z = (ret_vol - vol_ma) / np.where(vol_std == 0, np.nan, vol_std)
    combined[np.abs(vol_z) > 1.5] = 0
    
    return combined


def gen_kelt_stoch_signal(close, high, low, volume, open_arr,
                          kelt_ema_period, kelt_atr_period, kelt_mult,
                          rsi_period, stoch_period, k_ob, k_os):
    """Keltner + StochRSI confluence"""
    ke = ema(close, int(kelt_ema_period))
    atr = compute_atr(high, low, close, int(kelt_atr_period))
    
    s1 = np.zeros(len(close), dtype=np.int8)
    with np.errstate(invalid='ignore'):
        s1[close < (ke - kelt_mult * atr)] = 1
        s1[close > (ke + kelt_mult * atr)] = -1
    
    # StochRSI — vectorized
    rsi = compute_rsi(close, int(rsi_period))
    sp = int(stoch_period)
    rsi_s = pd.Series(rsi)
    rmin = rsi_s.rolling(sp).min().values
    rmax = rsi_s.rolling(sp).max().values
    with np.errstate(divide='ignore', invalid='ignore'):
        stoch_k = np.where(rmax > rmin, (rsi - rmin) / (rmax - rmin) * 100, np.nan)
    
    s2 = np.zeros(len(close), dtype=np.int8)
    s2[stoch_k < k_os] = 1
    s2[stoch_k > k_ob] = -1
    
    combined = np.zeros(len(close), dtype=np.int8)
    combined[(s1 == 1) & (s2 == 1)] = 1
    combined[(s1 == -1) & (s2 == -1)] = -1
    
    return combined


# ============================================================
# NON-OVERLAPPING TRADE EXECUTION
# ============================================================

def execute_trades(sig, close, hold_bars, cost_rt, start_idx, end_idx):
    """Non-overlapping trades in [start_idx, end_idx), returns list of returns"""
    trades = []
    i = start_idx
    while i < end_idx:
        if sig[i] != 0 and i + hold_bars < len(close):
            raw = sig[i] * (close[i + hold_bars] / close[i] - 1)
            trades.append(raw - cost_rt)
            i += hold_bars
        else:
            i += 1
    return trades


def sharpe_from_trades(trades, hold_bars):
    """Annualized Sharpe from trade returns"""
    if len(trades) < 10:
        return -999.0
    arr = np.array(trades)
    trades_per_year = 365.25 * 24 * 12 / hold_bars
    return arr.mean() / (arr.std() + 1e-12) * np.sqrt(trades_per_year)


# ============================================================
# SESSION MASKS
# ============================================================

def make_session_mask(hours, session):
    if session == 'all':
        return np.ones(len(hours), dtype=bool)
    elif session == 'us':
        return (hours >= 13) & (hours < 21)
    elif session == 'asian':
        return (hours >= 0) & (hours < 8)
    return np.ones(len(hours), dtype=bool)


# ============================================================
# OPTUNA OBJECTIVE FACTORY
# ============================================================

def make_objective(strat_type, close, high, low, volume, open_arr,
                   sess_mask, hold_bars, cost_rt, is_end):
    """Returns an Optuna objective function for IS optimization"""
    
    def objective(trial):
        if strat_type == 'trend':
            params = {
                'ema_fast': trial.suggest_int('ema_fast', 3, 15),
                'ema_slow': trial.suggest_int('ema_slow', 15, 50),
                'mom_period': trial.suggest_int('mom_period', 3, 30),
                'trend_period': trial.suggest_int('trend_period', 30, 250),
            }
            sig = gen_trend_signal(close[:is_end], high[:is_end], low[:is_end],
                                   volume[:is_end], open_arr[:is_end], **params)
        
        elif strat_type == 'mr':
            params = {
                'boll_period': trial.suggest_int('boll_period', 10, 40),
                'boll_std': trial.suggest_float('boll_std', 1.2, 3.0),
                'rsi_period': trial.suggest_int('rsi_period', 5, 21),
                'rsi_ob': trial.suggest_float('rsi_ob', 65, 85),
                'rsi_os': trial.suggest_float('rsi_os', 15, 35),
                'vol_filter_period': trial.suggest_int('vol_filter_period', 20, 100),
            }
            sig = gen_mr_signal(close[:is_end], high[:is_end], low[:is_end],
                                volume[:is_end], open_arr[:is_end], **params)
        
        elif strat_type == 'kelt_stoch':
            params = {
                'kelt_ema_period': trial.suggest_int('kelt_ema_period', 10, 40),
                'kelt_atr_period': trial.suggest_int('kelt_atr_period', 7, 25),
                'kelt_mult': trial.suggest_float('kelt_mult', 1.2, 3.5),
                'rsi_period': trial.suggest_int('rsi_period', 7, 21),
                'stoch_period': trial.suggest_int('stoch_period', 7, 21),
                'k_ob': trial.suggest_float('k_ob', 70, 90),
                'k_os': trial.suggest_float('k_os', 10, 30),
            }
            sig = gen_kelt_stoch_signal(close[:is_end], high[:is_end], low[:is_end],
                                         volume[:is_end], open_arr[:is_end], **params)
        
        # Apply session mask
        sig[~sess_mask[:is_end]] = 0
        
        trades = execute_trades(sig, close[:is_end], hold_bars, cost_rt, 0, is_end)
        return sharpe_from_trades(trades, hold_bars)
    
    return objective


# ============================================================
# WALK-FORWARD WITH OPTUNA
# ============================================================

def walk_forward_optuna(strat_type, close, high, low, volume, open_arr, idx_array,
                         sess_mask, hold_bars, cost_rt):
    """
    Expanding-window walk-forward with Optuna IS optimization.
    Returns (daily_returns, n_trades, best_params_per_fold) or None.
    """
    n = len(close)
    fold_size = n // (N_FOLDS + 1)
    if fold_size < 1000:
        return None
    
    all_oos_trades = []
    fold_params = []
    
    for fold_idx in range(1, N_FOLDS + 1):
        is_end = fold_idx * fold_size
        oos_start = is_end
        oos_end = min(oos_start + fold_size, n)
        
        if is_end < MIN_IS_BARS:
            continue
        
        # === OPTUNA IS OPTIMIZATION ===
        objective = make_objective(strat_type, close, high, low, volume, open_arr,
                                   sess_mask, hold_bars, cost_rt, is_end)
        
        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42 + fold_idx))
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        
        best_params = study.best_params
        best_is_sharpe = study.best_value
        fold_params.append({'fold': fold_idx, 'is_sharpe': best_is_sharpe, **best_params})
        del study
        gc.collect()
        
        # === OOS TEST WITH BEST PARAMS ===
        # Generate signal on data up to oos_end using best params
        if strat_type == 'trend':
            sig = gen_trend_signal(close[:oos_end], high[:oos_end], low[:oos_end],
                                   volume[:oos_end], open_arr[:oos_end], **best_params)
        elif strat_type == 'mr':
            sig = gen_mr_signal(close[:oos_end], high[:oos_end], low[:oos_end],
                                volume[:oos_end], open_arr[:oos_end], **best_params)
        elif strat_type == 'kelt_stoch':
            sig = gen_kelt_stoch_signal(close[:oos_end], high[:oos_end], low[:oos_end],
                                         volume[:oos_end], open_arr[:oos_end], **best_params)
        
        sig[~sess_mask[:oos_end]] = 0
        sig[:oos_start] = 0  # only trade in OOS window
        
        # Collect OOS trades with timestamps
        i = oos_start
        while i < oos_end:
            if sig[i] != 0 and i + hold_bars < len(close):
                raw = sig[i] * (close[i + hold_bars] / close[i] - 1)
                all_oos_trades.append((idx_array[i], raw - cost_rt))
                i += hold_bars
            else:
                i += 1
    
    if len(all_oos_trades) < 20:
        return None
    
    # Convert to daily returns
    times = pd.to_datetime([t[0] for t in all_oos_trades])
    rets = np.array([t[1] for t in all_oos_trades])
    df_trades = pd.DataFrame({'date': times.date, 'ret': rets})
    daily = df_trades.groupby('date')['ret'].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(full_range, fill_value=0.0)
    
    return daily, len(all_oos_trades), fold_params


# ============================================================
# MAIN
# ============================================================

def load_data(asset):
    path = os.path.join(DATA_DIR, f"{asset}_spot_5m.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def main():
    print("=" * 80)
    print("OPTUNA WALK-FORWARD SCALPING — Gold Standard Validation")
    print("=" * 80)
    
    total_est = len(ASSETS) * len(STRAT_TYPES) * len(SESSIONS) * len(HOLD_BARS) * len(COST_TIERS)
    print(f"Assets: {len(ASSETS)} | Strategies: {STRAT_TYPES}")
    print(f"Sessions: {SESSIONS} | Holds: {list(HOLD_BARS.keys())}")
    print(f"Cost tiers: {list(COST_TIERS.keys())}")
    print(f"Estimated tests: {total_est}")
    print(f"Optuna trials per IS fold: {OPTUNA_TRIALS}")
    print(f"Walk-forward: {N_FOLDS}-fold expanding, min IS: {MIN_IS_BARS} bars")
    print("=" * 80)
    
    all_results = []
    t0 = time.time()
    test_num = 0
    
    for asset in ASSETS:
        print(f"\n{'='*60}")
        print(f"Loading {asset}...")
        df = load_data(asset)
        if df is None:
            print(f"  SKIP - no data")
            continue
        print(f"  {len(df):,} rows, {df.index[0]} to {df.index[-1]}")
        
        close = df['Close'].values.astype(np.float64)
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        open_arr = df['Open'].values.astype(np.float64)
        idx_array = df.index.values
        hours = df.index.hour
        
        for strat_type in STRAT_TYPES:
            for sess in SESSIONS:
                smask = make_session_mask(hours, sess)
                
                for hold_name, hold_bars in HOLD_BARS.items():
                    for cost_name, cost_rt in COST_TIERS.items():
                        test_num += 1
                        t1 = time.time()
                        
                        result = walk_forward_optuna(
                            strat_type, close, high, low, volume, open_arr,
                            idx_array, smask, hold_bars, cost_rt
                        )
                        
                        elapsed_test = time.time() - t1
                        
                        if result is None:
                            continue
                        
                        daily_rets, n_trades, fold_params = result
                        metrics = compute_metrics(daily_rets)
                        grade = grade_strategy(metrics)
                        
                        # Average IS Sharpe across folds (for overfitting detection)
                        avg_is_sharpe = np.mean([fp['is_sharpe'] for fp in fold_params])
                        oos_sharpe = metrics.get('sharpe', -99)
                        overfit_ratio = oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else -99
                        
                        row = {
                            'asset': asset,
                            'strategy_type': strat_type,
                            'session': sess,
                            'hold': hold_name,
                            'cost_tier': cost_name,
                            'cost_bps': cost_rt * 10000,
                            'n_trades': n_trades,
                            'avg_is_sharpe': round(avg_is_sharpe, 3),
                            'oos_sharpe': round(oos_sharpe, 3),
                            'overfit_ratio': round(overfit_ratio, 3),
                            'fold_params': str(fold_params),
                            'grade': grade,
                        }
                        row.update(metrics)
                        all_results.append(row)
                        
                        marker = ""
                        if oos_sharpe > 1.0:
                            marker = "⭐⭐"
                        elif oos_sharpe > 0.5:
                            marker = "⭐"
                        
                        if oos_sharpe > 0.5:
                            print(f"  {marker} {strat_type}|{sess}|{hold_name}|{cost_name} "
                                  f"OOS_Sharpe={oos_sharpe:.2f} IS_Sharpe={avg_is_sharpe:.2f} "
                                  f"Overfit={overfit_ratio:.2f} "
                                  f"Sortino={metrics.get('sortino',0):.2f} "
                                  f"MDD={metrics.get('max_drawdown',0):.1%} "
                                  f"WR={metrics.get('win_rate',0):.1%} "
                                  f"Grade={grade} Trades={n_trades} "
                                  f"({elapsed_test:.1f}s)")
                        
                        if test_num % 10 == 0:
                            elapsed = time.time() - t0
                            print(f"  [{test_num}/{total_est}] {elapsed:.0f}s, "
                                  f"~{elapsed/test_num*(total_est-test_num)/60:.0f}min remaining")
        
        # Incremental save
        if all_results:
            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
            print(f"  💾 Saved {len(all_results)} results after {asset}")
        gc.collect()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — OPTUNA WALK-FORWARD OOS")
    print("=" * 80)
    
    if not all_results:
        print("NO RESULTS!")
        return
    
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(RESULTS_FILE, index=False)
    
    total = len(rdf)
    print(f"Total completed: {total}")
    for thresh in [0, 0.5, 1.0, 1.5, 2.0]:
        n = (rdf['oos_sharpe'] > thresh).sum()
        print(f"  OOS Sharpe > {thresh}: {n} ({n/total:.1%})")
    
    print(f"\nGrade distribution:")
    for g in ['A', 'B', 'C', 'D', 'F']:
        n = (rdf['grade'] == g).sum()
        if n: print(f"  {g}: {n} ({n/total:.1%})")
    
    # Overfitting analysis
    print(f"\nOverfitting analysis (OOS/IS Sharpe ratio):")
    valid = rdf[rdf['avg_is_sharpe'] > 0]
    if len(valid):
        print(f"  Avg overfit ratio: {valid['overfit_ratio'].mean():.2f}")
        print(f"  Median overfit ratio: {valid['overfit_ratio'].median():.2f}")
        print(f"  % where OOS > 50% of IS: {(valid['overfit_ratio'] > 0.5).mean():.1%}")
    
    for dim, col in [('cost tier', 'cost_tier'), ('hold', 'hold'),
                     ('session', 'session'), ('strategy', 'strategy_type')]:
        print(f"\nBy {dim}:")
        for val in sorted(rdf[col].unique()):
            sub = rdf[rdf[col] == val]
            print(f"  {val}: avg OOS Sharpe={sub['oos_sharpe'].mean():.3f} | "
                  f"{sub['oos_sharpe'].gt(0).mean():.1%} positive")
    
    print(f"\nTop 20 by OOS Sharpe:")
    for _, r in rdf.nlargest(20, 'oos_sharpe').iterrows():
        print(f"  {r['asset']}|{r['strategy_type']}|{r['session']}|{r['hold']}|{r['cost_tier']} "
              f"OOS={r['oos_sharpe']:.2f} IS={r['avg_is_sharpe']:.2f} "
              f"Overfit={r['overfit_ratio']:.2f} "
              f"Sortino={r.get('sortino',0):.2f} MDD={r.get('max_drawdown',0):.1%} "
              f"WR={r.get('win_rate',0):.1%} Grade={r['grade']} Trades={r['n_trades']}")
    
    print(f"\nCompleted in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
