#!/usr/bin/env python3
"""
Ultimate Scalping Validation — Mr. V's Challenge
==================================================
PROPER walk-forward: expanding window, optimize params IS, test OOS.
Pre-computes all signals once per asset for speed.
"""

import pandas as pd
import numpy as np
import os, sys, gc, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_metrics, grade_strategy

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/spot/5m/")
RESULTS_DIR = os.path.expanduser("~/Desktop/maestro/results/")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "scalping_ultimate.csv")

ASSETS = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'SUI', 'AVAX', 'NEAR', 'PEPE', 'WIF']

COST_TIERS = {
    'maker_2bps': 0.0002,
    'taker_4bps': 0.0004,
    'conservative_7bps': 0.0007,
    'pessimistic_10bps': 0.0010,
}

HOLD_BARS = {'15m': 3, '30m': 6, '1h': 12, '2h': 24, '4h': 48}
SESSIONS = ['all', 'us', 'london', 'asian']
N_FOLDS = 5
MIN_IS_BARS = 5000

# ============================================================
# SIGNAL GENERATORS (all backward-looking only)
# ============================================================

def sig_ema_cross(close, fast, slow):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    s = np.zeros(len(close), dtype=np.int8)
    s[ef.values > es.values] = 1
    s[ef.values < es.values] = -1
    return s

def sig_rsi(close, period, ob, os_):
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rs = g / l.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).values
    s = np.zeros(len(close), dtype=np.int8)
    s[rsi < os_] = 1
    s[rsi > ob] = -1
    return s

def sig_bollinger(close, period, std_mult):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    cv = close.values
    s = np.zeros(len(close), dtype=np.int8)
    s[cv < (sma - std_mult * std).values] = 1
    s[cv > (sma + std_mult * std).values] = -1
    return s

def sig_keltner(df, ema_p, atr_p, mult):
    ema = df['Close'].ewm(span=ema_p, adjust=False).mean()
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_p).mean()
    cv = df['Close'].values
    s = np.zeros(len(df), dtype=np.int8)
    s[cv < (ema - mult * atr).values] = 1
    s[cv > (ema + mult * atr).values] = -1
    return s

def sig_momentum(close, period):
    ret = close.pct_change(period).values
    s = np.zeros(len(close), dtype=np.int8)
    s[ret > 0] = 1
    s[ret < 0] = -1
    return s

def sig_macd(close, fast=12, slow=26, sig_p=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    hist = (ef - es) - (ef - es).ewm(span=sig_p, adjust=False).mean()
    s = np.zeros(len(close), dtype=np.int8)
    s[hist.values > 0] = 1
    s[hist.values < 0] = -1
    return s

def sig_stoch_rsi(close, rsi_p=14, stoch_p=14, k_sm=3):
    d = close.diff()
    g = d.clip(lower=0).rolling(rsi_p).mean()
    l = (-d.clip(upper=0)).rolling(rsi_p).mean()
    rsi = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    rmin = rsi.rolling(stoch_p).min()
    rmax = rsi.rolling(stoch_p).max()
    k = ((rsi - rmin) / (rmax - rmin).replace(0, np.nan) * 100).rolling(k_sm).mean()
    s = np.zeros(len(close), dtype=np.int8)
    s[k.values < 20] = 1
    s[k.values > 80] = -1
    return s

def sig_vol_spike(df, period=20, thresh=2.0):
    vma = df['Volume'].rolling(period).mean()
    spike = df['Volume'].values > (vma.values * thresh)
    green = df['Close'].values > df['Open'].values
    s = np.zeros(len(df), dtype=np.int8)
    s[spike & green] = 1
    s[spike & (~green)] = -1
    return s

# ============================================================
# FILTERS (numpy arrays)
# ============================================================

def make_session_masks(df):
    """Pre-compute all session masks once"""
    hour = df.index.hour
    return {
        'all': np.ones(len(df), dtype=bool),
        'asian': (hour >= 0) & (hour < 8),
        'london': (hour >= 8) & (hour < 16),
        'us': (hour >= 13) & (hour < 21),
    }

def make_trend_filter(close, period):
    """1 = uptrend, -1 = downtrend"""
    sma = close.rolling(period).mean().values
    cv = close.values
    f = np.zeros(len(close), dtype=np.int8)
    f[cv > sma] = 1
    f[cv < sma] = -1
    return f

def make_vol_filter(close, vol_period=48, rank_window=500):
    """True = medium volatility (20th-80th percentile)"""
    ret_vol = close.pct_change().abs().rolling(vol_period).std()
    rw = min(rank_window, len(close) // 3)
    if rw < 50:
        return np.ones(len(close), dtype=bool)
    vol_rank = ret_vol.rolling(rw).rank(pct=True).values
    return (vol_rank >= 0.20) & (vol_rank <= 0.80)

# ============================================================
# STRATEGY CONFIGS — each is a dict with:
#   name, type ('trend'/'mr'/'vol'), param_variants (list of signal arrays to try)
# We pre-compute ALL signal arrays per asset, then walk-forward selects best IS
# ============================================================

def precompute_strategies(df):
    """
    Returns dict: { (strat_type, param_idx): numpy_signal_array }
    Each signal array is the COMBINED multi-signal confluence.
    """
    close = df['Close']
    n = len(df)
    strategies = {}
    
    # Trend filters
    tf50 = make_trend_filter(close, 50)
    tf100 = make_trend_filter(close, 100)
    tf200 = make_trend_filter(close, 200)
    
    # Vol filter
    vf = make_vol_filter(close)
    
    # --- TREND STRATEGIES ---
    trend_params = [
        # (ema_fast, ema_slow, mom_period, use_macd, trend_filter)
        (5, 13, 6, False, tf50),
        (5, 13, 12, False, tf100),
        (8, 21, 6, False, tf50),
        (8, 21, 12, False, tf100),
        (8, 21, 12, True, tf100),
        (13, 34, 12, False, tf200),
        (13, 34, 24, True, tf200),
    ]
    
    macd_sig = sig_macd(close)
    
    for pi, (ef, es, mp, use_macd, tf) in enumerate(trend_params):
        s1 = sig_ema_cross(close, ef, es)
        s2 = sig_momentum(close, mp)
        
        if use_macd:
            vote = s1.astype(np.int16) + s2.astype(np.int16) + macd_sig.astype(np.int16)
            combined = np.zeros(n, dtype=np.int8)
            combined[vote >= 2] = 1
            combined[vote <= -2] = -1
        else:
            combined = np.zeros(n, dtype=np.int8)
            combined[(s1 == 1) & (s2 == 1)] = 1
            combined[(s1 == -1) & (s2 == -1)] = -1
        
        # Apply trend filter: only longs in uptrend, shorts in downtrend
        combined[(combined == 1) & (tf != 1)] = 0
        combined[(combined == -1) & (tf != -1)] = 0
        
        strategies[('trend', pi)] = combined
    
    # --- MEAN REVERSION STRATEGIES ---
    mr_params = [
        # (boll_p, boll_std, rsi_p, rsi_ob, rsi_os, use_kelt, kelt_mult)
        (14, 1.5, 7, 75, 25, False, 1.5),
        (14, 2.0, 7, 75, 25, False, 2.0),
        (20, 2.0, 14, 70, 30, False, 2.0),
        (20, 2.0, 14, 70, 30, True, 2.0),
        (20, 2.5, 14, 70, 30, False, 2.0),
        (20, 2.5, 14, 70, 30, True, 2.5),
        (30, 2.0, 14, 65, 35, True, 2.0),
    ]
    
    for pi, (bp, bs, rp, rob, ros, use_k, km) in enumerate(mr_params):
        s1 = sig_bollinger(close, bp, bs)
        s2 = sig_rsi(close, rp, rob, ros)
        
        if use_k:
            s3 = sig_keltner(df, bp, 14, km)
            vote = s1.astype(np.int16) + s2.astype(np.int16) + s3.astype(np.int16)
            combined = np.zeros(n, dtype=np.int8)
            combined[vote >= 2] = 1
            combined[vote <= -2] = -1
        else:
            combined = np.zeros(n, dtype=np.int8)
            combined[(s1 == 1) & (s2 == 1)] = 1
            combined[(s1 == -1) & (s2 == -1)] = -1
        
        # Vol filter
        combined[~vf] = 0
        strategies[('mr', pi)] = combined
    
    # --- VOLUME-CONFIRMED STRATEGIES ---
    vol_params = [
        (5, 13, 12, 1.5),
        (8, 21, 20, 2.0),
        (8, 21, 20, 2.5),
        (13, 34, 30, 2.0),
    ]
    
    for pi, (ef, es, vp, vt) in enumerate(vol_params):
        s1 = sig_ema_cross(close, ef, es)
        s2 = sig_vol_spike(df, vp, vt)
        combined = np.zeros(n, dtype=np.int8)
        combined[(s1 == 1) & (s2 == 1)] = 1
        combined[(s1 == -1) & (s2 == -1)] = -1
        strategies[('vol', pi)] = combined
    
    # --- EXTRA: StochRSI + Keltner (the best from previous tests) ---
    for pi, km in enumerate([1.5, 2.0, 2.5]):
        s1 = sig_keltner(df, 20, 14, km)
        s2 = sig_stoch_rsi(close)
        combined = np.zeros(n, dtype=np.int8)
        combined[(s1 == 1) & (s2 == 1)] = 1
        combined[(s1 == -1) & (s2 == -1)] = -1
        combined[~vf] = 0
        strategies[('kelt_stoch', pi)] = combined
    
    return strategies


# ============================================================
# WALK-FORWARD (expanding window, proper IS/OOS)
# ============================================================

def non_overlapping_sharpe(sig_arr, close_arr, hold_bars, cost_rt, start_idx, end_idx):
    """Fast: compute Sharpe from non-overlapping trades in [start_idx, end_idx)"""
    trades = []
    i = start_idx
    while i < end_idx:
        if sig_arr[i] != 0 and i + hold_bars < len(close_arr):
            raw = sig_arr[i] * (close_arr[i + hold_bars] / close_arr[i] - 1)
            trades.append(raw - cost_rt)
            i += hold_bars
        else:
            i += 1
    if len(trades) < 5:
        return -999.0, 0
    arr = np.array(trades)
    # Annualize: each trade spans hold_bars * 5min. Trades per year ≈ 365.25*24*12/hold_bars
    trades_per_year = 365.25 * 24 * 12 / hold_bars
    sharpe = arr.mean() / (arr.std() + 1e-12) * np.sqrt(trades_per_year)
    return sharpe, len(trades)


def non_overlapping_trades_full(sig_arr, close_arr, idx_array, hold_bars, cost_rt, start_idx, end_idx):
    """Return list of (timestamp, return) for OOS trades"""
    trades = []
    i = start_idx
    while i < end_idx:
        if sig_arr[i] != 0 and i + hold_bars < len(close_arr):
            raw = sig_arr[i] * (close_arr[i + hold_bars] / close_arr[i] - 1)
            trades.append((idx_array[i], raw - cost_rt))
            i += hold_bars
        else:
            i += 1
    return trades


def walk_forward_precomputed(strat_signals, close_arr, idx_array, session_mask_arr,
                              hold_bars, cost_rt):
    """
    Proper expanding-window walk-forward:
    - strat_signals: dict { param_idx: numpy_signal_array } for ONE strategy type
    - For each OOS fold: pick best param_idx on IS data, test on OOS
    - Returns (daily_returns_series, n_trades, winning_param_indices)
    """
    n = len(close_arr)
    fold_size = n // (N_FOLDS + 1)
    if fold_size < 500:
        return None
    
    all_oos_trades = []
    winning_params = []
    
    for fold_idx in range(1, N_FOLDS + 1):
        is_end = fold_idx * fold_size
        oos_start = is_end
        oos_end = min(oos_start + fold_size, n)
        
        if is_end < MIN_IS_BARS:
            continue
        
        # === IS: find best param_idx ===
        best_sharpe = -999
        best_pi = list(strat_signals.keys())[0]
        
        for pi, sig_full in strat_signals.items():
            # Apply session mask
            sig_is = sig_full[:is_end].copy()
            sig_is[~session_mask_arr[:is_end]] = 0
            
            sh, nt = non_overlapping_sharpe(sig_is, close_arr, hold_bars, cost_rt, 0, is_end)
            if sh > best_sharpe:
                best_sharpe = sh
                best_pi = pi
        
        # === OOS: test best param on OOS window ===
        sig_oos = strat_signals[best_pi][:oos_end].copy()
        sig_oos[~session_mask_arr[:oos_end]] = 0
        sig_oos[:oos_start] = 0  # only trade in OOS window
        
        trades = non_overlapping_trades_full(sig_oos, close_arr, idx_array, hold_bars, cost_rt,
                                              oos_start, oos_end)
        all_oos_trades.extend(trades)
        winning_params.append(best_pi)
    
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
    
    return daily, len(all_oos_trades), winning_params


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
    print("ULTIMATE SCALPING VALIDATION — Proper Walk-Forward")
    print("=" * 80)
    print(f"Assets: {len(ASSETS)} | Sessions: {len(SESSIONS)}")
    print(f"Holds: {list(HOLD_BARS.keys())} | Cost tiers: {list(COST_TIERS.keys())}")
    print(f"Walk-forward: {N_FOLDS}-fold expanding window, params optimized IS, tested OOS")
    print(f"Min IS bars: {MIN_IS_BARS} | Min OOS trades: 20")
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
        
        # Pre-compute ALL signals for ALL param variants
        t1 = time.time()
        strategies = precompute_strategies(df)
        print(f"  Pre-computed {len(strategies)} signal variants in {time.time()-t1:.1f}s")
        
        # Group by strategy type
        strat_types = {}
        for (stype, pidx), sig in strategies.items():
            strat_types.setdefault(stype, {})[pidx] = sig
        
        close_arr = df['Close'].values
        idx_array = df.index.values
        sess_masks = make_session_masks(df)
        
        for stype, param_sigs in strat_types.items():
            for sess in SESSIONS:
                smask = sess_masks[sess]
                for hold_name, hold_bars in HOLD_BARS.items():
                    for cost_name, cost_rt in COST_TIERS.items():
                        test_num += 1
                        
                        result = walk_forward_precomputed(
                            param_sigs, close_arr, idx_array, smask,
                            hold_bars, cost_rt
                        )
                        
                        if result is None:
                            continue
                        
                        daily_rets, n_trades, win_params = result
                        metrics = compute_metrics(daily_rets)
                        grade = grade_strategy(metrics)
                        
                        row = {
                            'asset': asset,
                            'strategy_type': stype,
                            'session': sess,
                            'hold': hold_name,
                            'cost_tier': cost_name,
                            'cost_bps': cost_rt * 10000,
                            'n_trades': n_trades,
                            'winning_params': str(win_params),
                            'grade': grade,
                        }
                        row.update(metrics)
                        all_results.append(row)
                        
                        sharpe = metrics.get('sharpe', -99)
                        if sharpe > 0.75:
                            print(f"  ⭐ {stype}|{sess}|{hold_name}|{cost_name} "
                                  f"Sharpe={sharpe:.2f} Sortino={metrics.get('sortino',0):.2f} "
                                  f"MDD={metrics.get('max_drawdown',0):.1%} WR={metrics.get('win_rate',0):.1%} "
                                  f"Grade={grade} Trades={n_trades}")
            
            if test_num % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{test_num}] {elapsed:.0f}s, {test_num/elapsed:.1f} tests/sec")
        
        # Incremental save
        if all_results:
            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
            print(f"  Saved {len(all_results)} results after {asset} ({time.time()-t0:.0f}s)")
        gc.collect()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — OOS WALK-FORWARD RESULTS")
    print("=" * 80)
    
    if not all_results:
        print("NO RESULTS!")
        return
    
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(RESULTS_FILE, index=False)
    
    total = len(rdf)
    print(f"Total completed: {total}")
    for thresh in [0, 0.5, 1.0, 1.5]:
        n = (rdf['sharpe'] > thresh).sum()
        print(f"  Sharpe > {thresh}: {n} ({n/total:.1%})")
    
    print(f"\nGrade distribution:")
    for g in ['A', 'B', 'C', 'D', 'F']:
        n = (rdf['grade'] == g).sum()
        if n: print(f"  {g}: {n} ({n/total:.1%})")
    
    for dim, col in [('cost tier', 'cost_tier'), ('hold', 'hold'), ('session', 'session'), ('strategy', 'strategy_type')]:
        print(f"\nBy {dim}:")
        for val in rdf[col].unique():
            sub = rdf[rdf[col] == val]
            print(f"  {val}: avg Sharpe={sub['sharpe'].mean():.3f} | {sub['sharpe'].gt(0).mean():.1%} positive")
    
    print(f"\nTop 20 by Sharpe (ALL OOS):")
    for _, r in rdf.nlargest(20, 'sharpe').iterrows():
        print(f"  {r['asset']}|{r['strategy_type']}|{r['session']}|{r['hold']}|{r['cost_tier']} "
              f"Sharpe={r['sharpe']:.2f} Sortino={r.get('sortino',0):.2f} "
              f"MDD={r.get('max_drawdown',0):.1%} WR={r.get('win_rate',0):.1%} "
              f"Grade={r['grade']} Trades={r['n_trades']}")
    
    print(f"\nCompleted in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
