#!/usr/bin/env python3
"""
Volatility Regime Trading Strategies - Walk-Forward Backtest
4 strategies × 10 tokens × multiple param combos × 10-fold expanding window
"""

import pandas as pd
import numpy as np
import json, os, time, warnings
from pathlib import Path
from itertools import product
from datetime import datetime

warnings.filterwarnings('ignore')

DATA_DIR = Path.home() / "Desktop/maestro/data/ohlcv"
OUT_DIR = Path.home() / "Desktop/maestro/data/backtest_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = ["arb", "avax", "btc", "eth", "fet", "inj", "link", "op", "sol", "sui"]
COMMISSION = 0.002  # 20bps per trade (applied on entry+exit = 40bps round trip)

# ─── Data Loading ───────────────────────────────────────────────────────────

def load_token(token: str) -> pd.DataFrame:
    fp = DATA_DIR / f"binance_{token}_usdt_1d.csv"
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def load_all() -> dict[str, pd.DataFrame]:
    data = {}
    for t in TOKENS:
        try:
            data[t] = load_token(t)
        except Exception as e:
            print(f"  Skip {t}: {e}")
    return data

# ─── Indicators ─────────────────────────────────────────────────────────────

def atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    cp = c.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def norm_atr(df, period=14):
    return atr(df, period) / df['close']

def realized_vol(df, period=20):
    return df['close'].pct_change().rolling(period).std() * np.sqrt(365)

def bollinger(df, period=20, std_mult=2.0):
    ma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return ma, ma + std_mult * std, ma - std_mult * std

def momentum(df, period=20):
    return df['close'].pct_change(period)

# ─── Strategy 1: Volatility Compression Breakout ───────────────────────────

def strat_vol_compression(df, atr_period=14, bb_period=20, bb_std=2.0,
                          lookback=60, pct_thresh=20, hold_days=10):
    n = len(df)
    natr = norm_atr(df, atr_period)
    _, bb_upper, bb_lower = bollinger(df, bb_period, bb_std)
    close = df['close'].values
    signals = np.zeros(n)
    
    natr_vals = natr.values
    bb_up = bb_upper.values
    bb_lo = bb_lower.values
    
    position = 0
    days_held = 0
    
    for i in range(lookback, n):
        if position != 0:
            days_held += 1
            if days_held >= hold_days:
                position = 0
                days_held = 0
        
        window = natr_vals[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            signals[i] = position
            continue
        
        threshold = np.percentile(valid, pct_thresh)
        
        if not np.isnan(natr_vals[i]) and natr_vals[i] <= threshold:
            if close[i] > bb_up[i] and not np.isnan(bb_up[i]):
                position = 1
                days_held = 0
            elif close[i] < bb_lo[i] and not np.isnan(bb_lo[i]):
                position = -1
                days_held = 0
        
        signals[i] = position
    
    return signals

# ─── Strategy 2: Vol Mean-Reversion ────────────────────────────────────────

def strat_vol_meanrev(df, vol_period=20, lookback=60, high_std=2.0, low_std=1.0,
                      mom_period=10):
    n = len(df)
    rv = realized_vol(df, vol_period).values
    close = df['close'].values
    signals = np.zeros(n)
    
    for i in range(lookback + vol_period, n):
        window = rv[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        
        med = np.median(valid)
        std = np.std(valid)
        
        if np.isnan(rv[i]):
            continue
        
        # Recent price direction
        if i >= mom_period:
            price_dir = 1 if close[i] > close[i - mom_period] else -1
        else:
            price_dir = 0
        
        if rv[i] > med + high_std * std:
            # High vol → fade the move
            signals[i] = -price_dir
        elif rv[i] < med - low_std * std:
            # Low vol → buy breakout direction
            signals[i] = price_dir
        else:
            signals[i] = 0
    
    return signals

# ─── Strategy 3: Vol Regime Switch ──────────────────────────────────────────

def strat_vol_regime(df, vol_period=20, lookback=60, low_pct=25, high_pct=75,
                     mom_period=20, mean_rev_period=10,
                     low_size=0.3, normal_size=1.0, high_size=0.7):
    n = len(df)
    rv = realized_vol(df, vol_period).values
    close = df['close'].values
    signals = np.zeros(n)
    
    for i in range(lookback + vol_period, n):
        window = rv[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) < 20 or np.isnan(rv[i]):
            continue
        
        low_th = np.percentile(valid, low_pct)
        high_th = np.percentile(valid, high_pct)
        
        mom = (close[i] / close[max(0, i - mom_period)] - 1) if i >= mom_period else 0
        
        if rv[i] < low_th:
            # Low vol: small breakout position
            if abs(mom) > 0.02:
                signals[i] = np.sign(mom) * low_size
            else:
                signals[i] = 0
        elif rv[i] > high_th:
            # High vol: mean-revert
            short_ret = (close[i] / close[max(0, i - mean_rev_period)] - 1) if i >= mean_rev_period else 0
            signals[i] = -np.sign(short_ret) * high_size
        else:
            # Normal: trend-follow
            signals[i] = np.sign(mom) * normal_size
    
    return signals

# ─── Strategy 4: Cross-Asset Vol Dispersion ─────────────────────────────────

def compute_vol_dispersion(all_data: dict, vol_period=20) -> pd.DataFrame:
    """Compute cross-asset vol dispersion aligned to common dates."""
    vols = {}
    for token, df in all_data.items():
        rv = realized_vol(df, vol_period)
        s = rv.copy()
        s.index = df['timestamp']
        vols[token] = s
    
    vol_df = pd.DataFrame(vols)
    vol_df = vol_df.dropna(how='all')
    
    dispersion = vol_df.std(axis=1)
    median_disp = dispersion.rolling(60).median()
    
    return vol_df, dispersion, median_disp

def strat_vol_dispersion(df, token, dispersion_series, median_disp_series,
                         mom_period=20, mean_rev_period=10, disp_thresh=1.0):
    """Per-token signals based on cross-asset vol dispersion."""
    n = len(df)
    signals = np.zeros(n)
    close = df['close'].values
    timestamps = df['timestamp'].values
    
    disp_dict = dict(zip(dispersion_series.index, dispersion_series.values))
    med_dict = dict(zip(median_disp_series.index, median_disp_series.values))
    
    for i in range(max(mom_period, mean_rev_period) + 60, n):
        ts = pd.Timestamp(timestamps[i])
        d = disp_dict.get(ts, np.nan)
        m = med_dict.get(ts, np.nan)
        
        if np.isnan(d) or np.isnan(m) or m == 0:
            continue
        
        mom = (close[i] / close[max(0, i - mom_period)] - 1)
        short_ret = (close[i] / close[max(0, i - mean_rev_period)] - 1)
        
        if d < m * (1 - disp_thresh * 0.3):
            # Low dispersion → momentum
            signals[i] = np.sign(mom)
        elif d > m * (1 + disp_thresh * 0.3):
            # High dispersion → mean-revert
            signals[i] = -np.sign(short_ret)
        else:
            signals[i] = 0
    
    return signals

# ─── Backtest Engine ────────────────────────────────────────────────────────

def backtest(signals, close, commission=COMMISSION):
    """Simple vectorized backtest with commission."""
    n = len(signals)
    returns = np.diff(close) / close[:-1]
    
    pos = signals[:-1]  # position at start of each bar
    strat_ret = pos * returns
    
    # Commission on trades
    trades = np.diff(signals)
    trade_cost = np.abs(trades) * commission
    strat_ret[1:] -= trade_cost[:-1] if len(trade_cost) > 1 else 0
    
    equity = np.cumprod(1 + strat_ret)
    total_ret = equity[-1] - 1 if len(equity) > 0 else 0
    
    # Metrics
    n_days = len(strat_ret)
    if n_days < 10:
        return {"total_return": 0, "sharpe": 0, "max_dd": 0, "n_trades": 0, "n_days": n_days}
    
    ann_factor = np.sqrt(365)
    mean_ret = np.mean(strat_ret)
    std_ret = np.std(strat_ret)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0
    
    # Max drawdown
    cum = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))
    
    n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))
    
    return {
        "total_return": round(float(total_ret), 4),
        "sharpe": round(float(sharpe), 4),
        "max_dd": round(float(max_dd), 4),
        "n_trades": n_trades,
        "n_days": n_days
    }

# ─── Walk-Forward ───────────────────────────────────────────────────────────

def walk_forward(df, signal_func, n_folds=10, **kwargs):
    """Expanding window walk-forward test."""
    n = len(df)
    min_train = max(120, n // (n_folds + 2))
    fold_size = (n - min_train) // n_folds
    
    if fold_size < 20:
        return None
    
    oos_results = []
    close = df['close'].values
    
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        
        if test_end <= train_end + 10:
            continue
        
        # Generate signals on full data up to test_end (expanding)
        signals = signal_func(df.iloc[:test_end], **kwargs)
        
        # Only evaluate OOS portion
        oos_signals = signals[train_end:test_end]
        oos_close = close[train_end:test_end]
        
        if len(oos_close) < 10:
            continue
        
        result = backtest(oos_signals, oos_close)
        result['fold'] = fold
        oos_results.append(result)
    
    if not oos_results:
        return None
    
    # Aggregate
    returns = [r['total_return'] for r in oos_results]
    sharpes = [r['sharpe'] for r in oos_results]
    
    agg = {
        "mean_return": round(float(np.mean(returns)), 4),
        "median_return": round(float(np.median(returns)), 4),
        "mean_sharpe": round(float(np.mean(sharpes)), 4),
        "median_sharpe": round(float(np.median(sharpes)), 4),
        "worst_fold_return": round(float(np.min(returns)), 4),
        "best_fold_return": round(float(np.max(returns)), 4),
        "pct_positive_folds": round(float(np.mean([1 if r > 0 else 0 for r in returns])), 4),
        "n_folds": len(oos_results),
        "total_trades": sum(r['n_trades'] for r in oos_results),
        "folds": oos_results
    }
    
    # Simple t-test for significance
    if len(returns) > 2 and np.std(returns) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        agg['t_stat'] = round(float(t_stat), 4)
        agg['p_value'] = round(float(p_value), 4)
    else:
        agg['t_stat'] = 0
        agg['p_value'] = 1.0
    
    return agg

# ─── Permutation Test ───────────────────────────────────────────────────────

def permutation_test(df, signal_func, observed_sharpe, n_perms=200, **kwargs):
    """Random permutation test: shuffle returns, re-run signals."""
    n = len(df)
    close = df['close'].values
    signals = signal_func(df, **kwargs)
    returns = np.diff(close) / close[:-1]
    
    count_better = 0
    for _ in range(n_perms):
        perm_ret = np.random.permutation(returns)
        perm_close = np.zeros(n)
        perm_close[0] = close[0]
        for i in range(1, n):
            perm_close[i] = perm_close[i - 1] * (1 + perm_ret[i - 1] if i - 1 < len(perm_ret) else 0)
        
        perm_df = df.copy()
        perm_df['close'] = perm_close
        perm_df['high'] = perm_close * (1 + np.random.uniform(0, 0.03, n))
        perm_df['low'] = perm_close * (1 - np.random.uniform(0, 0.03, n))
        
        perm_signals = signal_func(perm_df, **kwargs)
        perm_result = backtest(perm_signals, perm_close)
        
        if perm_result['sharpe'] >= observed_sharpe:
            count_better += 1
    
    return round(count_better / n_perms, 4)

# ─── Walk-Forward for Dispersion Strategy ───────────────────────────────────

def walk_forward_dispersion(df, token, all_data, n_folds=10, **kwargs):
    """Walk-forward for cross-asset dispersion strategy."""
    vol_period = kwargs.get('vol_period', 20)
    vol_df, dispersion, median_disp = compute_vol_dispersion(all_data, vol_period)
    
    n = len(df)
    min_train = max(120, n // (n_folds + 2))
    fold_size = (n - min_train) // n_folds
    
    if fold_size < 20:
        return None
    
    oos_results = []
    close = df['close'].values
    
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        
        if test_end <= train_end + 10:
            continue
        
        sub_df = df.iloc[:test_end].copy()
        signals = strat_vol_dispersion(sub_df, token, dispersion, median_disp,
                                        mom_period=kwargs.get('mom_period', 20),
                                        mean_rev_period=kwargs.get('mean_rev_period', 10),
                                        disp_thresh=kwargs.get('disp_thresh', 1.0))
        
        oos_signals = signals[train_end:test_end]
        oos_close = close[train_end:test_end]
        
        if len(oos_close) < 10:
            continue
        
        result = backtest(oos_signals, oos_close)
        result['fold'] = fold
        oos_results.append(result)
    
    if not oos_results:
        return None
    
    returns = [r['total_return'] for r in oos_results]
    sharpes = [r['sharpe'] for r in oos_results]
    
    agg = {
        "mean_return": round(float(np.mean(returns)), 4),
        "median_return": round(float(np.median(returns)), 4),
        "mean_sharpe": round(float(np.mean(sharpes)), 4),
        "median_sharpe": round(float(np.median(sharpes)), 4),
        "worst_fold_return": round(float(np.min(returns)), 4),
        "best_fold_return": round(float(np.max(returns)), 4),
        "pct_positive_folds": round(float(np.mean([1 if r > 0 else 0 for r in returns])), 4),
        "n_folds": len(oos_results),
        "total_trades": sum(r['n_trades'] for r in oos_results),
        "folds": oos_results
    }
    
    if len(returns) > 2 and np.std(returns) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        agg['t_stat'] = round(float(t_stat), 4)
        agg['p_value'] = round(float(p_value), 4)
    else:
        agg['t_stat'] = 0
        agg['p_value'] = 1.0
    
    return agg

# ─── Parameter Grids ────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "vol_compression": [
        {"atr_period": 14, "bb_period": 20, "bb_std": 2.0, "lookback": 60, "pct_thresh": 20, "hold_days": 10},
        {"atr_period": 14, "bb_period": 20, "bb_std": 2.5, "lookback": 60, "pct_thresh": 15, "hold_days": 15},
        {"atr_period": 10, "bb_period": 20, "bb_std": 2.0, "lookback": 40, "pct_thresh": 20, "hold_days": 7},
        {"atr_period": 14, "bb_period": 30, "bb_std": 2.0, "lookback": 90, "pct_thresh": 25, "hold_days": 10},
        {"atr_period": 20, "bb_period": 20, "bb_std": 1.5, "lookback": 60, "pct_thresh": 10, "hold_days": 20},
    ],
    "vol_meanrev": [
        {"vol_period": 20, "lookback": 60, "high_std": 2.0, "low_std": 1.0, "mom_period": 10},
        {"vol_period": 15, "lookback": 45, "high_std": 1.5, "low_std": 0.5, "mom_period": 10},
        {"vol_period": 20, "lookback": 90, "high_std": 2.5, "low_std": 1.5, "mom_period": 15},
        {"vol_period": 30, "lookback": 60, "high_std": 2.0, "low_std": 1.0, "mom_period": 20},
        {"vol_period": 10, "lookback": 40, "high_std": 1.5, "low_std": 0.8, "mom_period": 5},
    ],
    "vol_regime": [
        {"vol_period": 20, "lookback": 60, "low_pct": 25, "high_pct": 75, "mom_period": 20, "mean_rev_period": 10, "low_size": 0.3, "normal_size": 1.0, "high_size": 0.7},
        {"vol_period": 15, "lookback": 45, "low_pct": 20, "high_pct": 80, "mom_period": 15, "mean_rev_period": 7, "low_size": 0.5, "normal_size": 1.0, "high_size": 0.5},
        {"vol_period": 20, "lookback": 90, "low_pct": 30, "high_pct": 70, "mom_period": 30, "mean_rev_period": 14, "low_size": 0.3, "normal_size": 0.8, "high_size": 1.0},
        {"vol_period": 30, "lookback": 60, "low_pct": 25, "high_pct": 75, "mom_period": 20, "mean_rev_period": 10, "low_size": 0.2, "normal_size": 1.0, "high_size": 0.8},
        {"vol_period": 10, "lookback": 40, "low_pct": 20, "high_pct": 80, "mom_period": 10, "mean_rev_period": 5, "low_size": 0.4, "normal_size": 1.0, "high_size": 0.6},
    ],
    "vol_dispersion": [
        {"vol_period": 20, "mom_period": 20, "mean_rev_period": 10, "disp_thresh": 1.0},
        {"vol_period": 15, "mom_period": 15, "mean_rev_period": 7, "disp_thresh": 0.5},
        {"vol_period": 20, "mom_period": 30, "mean_rev_period": 14, "disp_thresh": 1.5},
        {"vol_period": 30, "mom_period": 20, "mean_rev_period": 10, "disp_thresh": 0.8},
        {"vol_period": 10, "mom_period": 10, "mean_rev_period": 5, "disp_thresh": 1.2},
    ],
}

STRAT_FUNCS = {
    "vol_compression": strat_vol_compression,
    "vol_meanrev": strat_vol_meanrev,
    "vol_regime": strat_vol_regime,
}

# ─── Main ───────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 70)
    print("VOLATILITY REGIME TRADING - Walk-Forward Backtest")
    print("=" * 70)
    
    t0 = time.time()
    all_data = load_all()
    print(f"Loaded {len(all_data)} tokens")
    
    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "tokens": list(all_data.keys()),
            "commission_bps": 20,
            "n_folds": 10,
            "strategies": list(PARAM_GRIDS.keys()),
        },
        "strategies": {}
    }
    
    # ── Strategies 1-3: per-token ──
    for strat_name in ["vol_compression", "vol_meanrev", "vol_regime"]:
        print(f"\n{'─' * 60}")
        print(f"Strategy: {strat_name}")
        print(f"{'─' * 60}")
        
        strat_results = {"per_token": {}, "portfolio": {}}
        func = STRAT_FUNCS[strat_name]
        
        for pi, params in enumerate(PARAM_GRIDS[strat_name]):
            param_key = f"params_{pi}"
            print(f"  Param set {pi}: {params}")
            
            token_results = {}
            all_fold_returns = []
            
            for token in all_data:
                df = all_data[token]
                wf = walk_forward(df, func, n_folds=10, **params)
                
                if wf is None:
                    print(f"    {token}: insufficient data")
                    continue
                
                # Permutation test if p < 0.05
                perm_p = None
                if wf['p_value'] < 0.05 and wf['mean_sharpe'] != 0:
                    print(f"    {token}: p={wf['p_value']:.4f} → running 200 permutations...")
                    signals_full = func(df, **params)
                    full_bt = backtest(signals_full, df['close'].values)
                    perm_p = permutation_test(df, func, full_bt['sharpe'], n_perms=200, **params)
                    wf['permutation_p'] = perm_p
                    print(f"    {token}: perm_p={perm_p:.4f}")
                
                token_results[token] = wf
                all_fold_returns.extend([r['total_return'] for r in wf['folds']])
                
                sym = "✓" if wf['mean_sharpe'] > 0 else "✗"
                pp = f" perm_p={perm_p:.3f}" if perm_p is not None else ""
                print(f"    {sym} {token}: sharpe={wf['mean_sharpe']:.2f} ret={wf['mean_return']:.2%} p={wf['p_value']:.3f}{pp}")
            
            strat_results["per_token"][param_key] = {
                "params": params,
                "tokens": token_results
            }
            
            # Portfolio aggregate
            if token_results:
                sharpes = [v['mean_sharpe'] for v in token_results.values()]
                rets = [v['mean_return'] for v in token_results.values()]
                strat_results["portfolio"][param_key] = {
                    "params": params,
                    "mean_sharpe": round(float(np.mean(sharpes)), 4),
                    "median_sharpe": round(float(np.median(sharpes)), 4),
                    "mean_return": round(float(np.mean(rets)), 4),
                    "pct_tokens_positive_sharpe": round(float(np.mean([1 if s > 0 else 0 for s in sharpes])), 4),
                    "best_token": max(token_results, key=lambda t: token_results[t]['mean_sharpe']),
                    "worst_token": min(token_results, key=lambda t: token_results[t]['mean_sharpe']),
                }
        
        results["strategies"][strat_name] = strat_results
    
    # ── Strategy 4: Cross-Asset Vol Dispersion ──
    print(f"\n{'─' * 60}")
    print("Strategy: vol_dispersion (cross-asset)")
    print(f"{'─' * 60}")
    
    strat_results = {"per_token": {}, "portfolio": {}}
    
    for pi, params in enumerate(PARAM_GRIDS["vol_dispersion"]):
        param_key = f"params_{pi}"
        print(f"  Param set {pi}: {params}")
        
        token_results = {}
        
        for token in all_data:
            df = all_data[token]
            wf = walk_forward_dispersion(df, token, all_data, n_folds=10, **params)
            
            if wf is None:
                print(f"    {token}: insufficient data")
                continue
            
            # Permutation test if significant
            perm_p = None
            if wf['p_value'] < 0.05 and wf['mean_sharpe'] != 0:
                print(f"    {token}: p={wf['p_value']:.4f} → running 200 permutations...")
                vol_period = params.get('vol_period', 20)
                _, disp, med_disp = compute_vol_dispersion(all_data, vol_period)
                signals_full = strat_vol_dispersion(df, token, disp, med_disp,
                                                     mom_period=params.get('mom_period', 20),
                                                     mean_rev_period=params.get('mean_rev_period', 10),
                                                     disp_thresh=params.get('disp_thresh', 1.0))
                full_bt = backtest(signals_full, df['close'].values)
                
                # Simplified permutation for dispersion (shuffle returns)
                count_better = 0
                close = df['close'].values
                returns = np.diff(close) / close[:-1]
                for _ in range(200):
                    perm_ret = np.random.permutation(returns)
                    perm_close = np.zeros(len(df))
                    perm_close[0] = close[0]
                    for j in range(1, len(df)):
                        perm_close[j] = perm_close[j-1] * (1 + (perm_ret[j-1] if j-1 < len(perm_ret) else 0))
                    perm_bt = backtest(signals_full, perm_close)
                    if perm_bt['sharpe'] >= full_bt['sharpe']:
                        count_better += 1
                perm_p = round(count_better / 200, 4)
                wf['permutation_p'] = perm_p
                print(f"    {token}: perm_p={perm_p:.4f}")
            
            token_results[token] = wf
            sym = "✓" if wf['mean_sharpe'] > 0 else "✗"
            pp = f" perm_p={perm_p:.3f}" if perm_p is not None else ""
            print(f"    {sym} {token}: sharpe={wf['mean_sharpe']:.2f} ret={wf['mean_return']:.2%} p={wf['p_value']:.3f}{pp}")
        
        strat_results["per_token"][param_key] = {
            "params": params,
            "tokens": token_results
        }
        
        if token_results:
            sharpes = [v['mean_sharpe'] for v in token_results.values()]
            rets = [v['mean_return'] for v in token_results.values()]
            strat_results["portfolio"][param_key] = {
                "params": params,
                "mean_sharpe": round(float(np.mean(sharpes)), 4),
                "median_sharpe": round(float(np.median(sharpes)), 4),
                "mean_return": round(float(np.mean(rets)), 4),
                "pct_tokens_positive_sharpe": round(float(np.mean([1 if s > 0 else 0 for s in sharpes])), 4),
                "best_token": max(token_results, key=lambda t: token_results[t]['mean_sharpe']),
                "worst_token": min(token_results, key=lambda t: token_results[t]['mean_sharpe']),
            }
    
    results["strategies"]["vol_dispersion"] = strat_results
    
    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"SUMMARY (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    
    summary = {}
    for sname, sdata in results["strategies"].items():
        best_params = None
        best_sharpe = -999
        for pk, pdata in sdata["portfolio"].items():
            if pdata["mean_sharpe"] > best_sharpe:
                best_sharpe = pdata["mean_sharpe"]
                best_params = pk
        
        if best_params:
            bp = sdata["portfolio"][best_params]
            summary[sname] = {
                "best_param_set": best_params,
                "portfolio_sharpe": bp["mean_sharpe"],
                "portfolio_return": bp["mean_return"],
                "pct_positive": bp["pct_tokens_positive_sharpe"],
                "best_token": bp["best_token"],
            }
            print(f"  {sname}: sharpe={bp['mean_sharpe']:.2f} ret={bp['mean_return']:.2%} "
                  f"pos={bp['pct_tokens_positive_sharpe']:.0%} best={bp['best_token']}")
    
    results["summary"] = summary
    results["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    
    # Save
    out_path = OUT_DIR / "vol_regime_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    return results

if __name__ == "__main__":
    run_all()
