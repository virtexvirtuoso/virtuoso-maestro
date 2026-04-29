"""
Lookback Scheme Exploration + Hyperoptimization
Tests multiple lookback architectures for the M2 proxy signal,
then Optuna-optimizes the best scheme.

Schemes:
1. Fixed (baseline) - single lookback window
2. Dual-speed - fast + slow lookback, consensus
3. Adaptive volatility - lookback scales with realized vol
4. Expanding window - growing lookback with floor
5. Multi-scale voting - 3+ windows vote
6. Regime-aware - different lookbacks per vol regime
7. Exponential decay - weighted lookback with half-life
8. Fractal - nested lookbacks (short confirms long)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasource.yfinance_loader import StockDataLoader
from datasource.fred_loader import MacroDataLoader

# ── Data Loading ──────────────────────────────────────────────────
def load_data():
    stock = StockDataLoader()
    
    btc = stock.get_ohlcv('BTC-USD', start_date='2015-01-01')
    df = pd.DataFrame(index=btc.index)
    df['btc_close'] = btc['close']
    
    for sym, col in [('DX-Y.NYB', 'dxy'), ('GLD', 'gold'), ('TLT', 'tlt'), ('HYG', 'hyg')]:
        tmp = stock.get_ohlcv(sym, start_date='2015-01-01')
        df[col] = tmp['close']
    
    df = df.ffill().dropna()
    return df

# ── Proxy Signal Generators ──────────────────────────────────────

def scheme_fixed(df, lookback=180, threshold=2):
    """Fixed lookback with N-of-4 consensus."""
    dxy_down = df['dxy'].pct_change(lookback) < 0
    gold_up = df['gold'].pct_change(lookback) > 0
    tlt_up = df['tlt'].pct_change(lookback) > 0
    hyg_up = df['hyg'].pct_change(lookback) > 0
    
    score = dxy_down.astype(int) + gold_up.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
    return (score >= threshold).astype(int).shift(1)

def scheme_dual_speed(df, fast=60, slow=180, threshold=2):
    """Fast and slow lookbacks — signal when both agree."""
    def proxy(lb):
        dxy_down = df['dxy'].pct_change(lb) < 0
        gold_up = df['gold'].pct_change(lb) > 0
        tlt_up = df['tlt'].pct_change(lb) > 0
        hyg_up = df['hyg'].pct_change(lb) > 0
        return dxy_down.astype(int) + gold_up.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
    
    fast_score = proxy(fast)
    slow_score = proxy(slow)
    combined = ((fast_score >= threshold) & (slow_score >= threshold)).astype(int)
    return combined.shift(1)

def scheme_adaptive_vol(df, base_lb=120, vol_window=60, min_lb=40, max_lb=300):
    """Lookback adapts to BTC realized volatility."""
    ret = df['btc_close'].pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(365)
    vol_median = vol.expanding().median()
    
    # High vol → shorter lookback (react faster), low vol → longer
    ratio = vol / vol_median
    adaptive_lb = (base_lb / ratio).clip(min_lb, max_lb).fillna(base_lb).astype(int)
    
    signal = pd.Series(0, index=df.index)
    for i in range(max_lb, len(df)):
        lb = adaptive_lb.iloc[i]
        dxy_down = df['dxy'].iloc[i] < df['dxy'].iloc[i - lb]
        gold_up = df['gold'].iloc[i] > df['gold'].iloc[i - lb]
        tlt_up = df['tlt'].iloc[i] > df['tlt'].iloc[i - lb]
        hyg_up = df['hyg'].iloc[i] > df['hyg'].iloc[i - lb]
        score = int(dxy_down) + int(gold_up) + int(tlt_up) + int(hyg_up)
        signal.iloc[i] = 1 if score >= 2 else 0
    
    return signal.shift(1)

def scheme_expanding(df, min_lb=90, threshold=2):
    """Expanding window — uses all available history with floor."""
    signal = pd.Series(0, index=df.index)
    for i in range(min_lb, len(df)):
        lb = max(min_lb, i // 2)  # Use half of available history, min floor
        lb = min(lb, 500)  # Cap at 500
        dxy_down = df['dxy'].iloc[i] < df['dxy'].iloc[i - lb]
        gold_up = df['gold'].iloc[i] > df['gold'].iloc[i - lb]
        tlt_up = df['tlt'].iloc[i] > df['tlt'].iloc[i - lb]
        hyg_up = df['hyg'].iloc[i] > df['hyg'].iloc[i - lb]
        score = int(dxy_down) + int(gold_up) + int(tlt_up) + int(hyg_up)
        signal.iloc[i] = 1 if score >= threshold else 0
    return signal.shift(1)

def scheme_multiscale_vote(df, windows=[60, 120, 180, 252], min_votes=2, threshold=2):
    """Multiple windows vote — signal when enough timeframes agree."""
    votes = pd.Series(0.0, index=df.index)
    for w in windows:
        dxy_down = df['dxy'].pct_change(w) < 0
        gold_up = df['gold'].pct_change(w) > 0
        tlt_up = df['tlt'].pct_change(w) > 0
        hyg_up = df['hyg'].pct_change(w) > 0
        score = dxy_down.astype(int) + gold_up.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
        votes += (score >= threshold).astype(int)
    
    return (votes >= min_votes).astype(int).shift(1)

def scheme_regime_aware(df, vol_window=60, low_vol_lb=252, high_vol_lb=60, threshold=2):
    """Different lookbacks for different vol regimes."""
    ret = df['btc_close'].pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(365)
    vol_median = vol.expanding().median()
    
    high_vol = vol > vol_median
    
    signal = pd.Series(0, index=df.index)
    max_lb = max(low_vol_lb, high_vol_lb)
    for i in range(max_lb, len(df)):
        lb = high_vol_lb if high_vol.iloc[i] else low_vol_lb
        dxy_down = df['dxy'].iloc[i] < df['dxy'].iloc[i - lb]
        gold_up = df['gold'].iloc[i] > df['gold'].iloc[i - lb]
        tlt_up = df['tlt'].iloc[i] > df['tlt'].iloc[i - lb]
        hyg_up = df['hyg'].iloc[i] > df['hyg'].iloc[i - lb]
        score = int(dxy_down) + int(gold_up) + int(tlt_up) + int(hyg_up)
        signal.iloc[i] = 1 if score >= threshold else 0
    return signal.shift(1)

def scheme_exp_decay(df, half_life=90, threshold=2):
    """Exponentially-weighted lookback — recent data weighted more."""
    span = int(half_life * 2)
    alpha = np.log(2) / half_life
    
    def ewm_direction(series):
        ret = series.pct_change()
        weights = np.exp(-alpha * np.arange(span)[::-1])
        weighted = ret.rolling(span).apply(lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]), raw=True)
        return weighted > 0
    
    dxy_down = ~ewm_direction(df['dxy'])  # DXY down = good
    gold_up = ewm_direction(df['gold'])
    tlt_up = ewm_direction(df['tlt'])
    hyg_up = ewm_direction(df['hyg'])
    
    score = dxy_down.astype(int) + gold_up.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
    return (score >= threshold).astype(int).shift(1)

def scheme_fractal(df, short=40, medium=120, long=252, threshold=2):
    """Nested: long confirms trend, medium times entry, short confirms."""
    def proxy(lb):
        dxy_down = df['dxy'].pct_change(lb) < 0
        gold_up = df['gold'].pct_change(lb) > 0
        tlt_up = df['tlt'].pct_change(lb) > 0
        hyg_up = df['hyg'].pct_change(lb) > 0
        return dxy_down.astype(int) + gold_up.astype(int) + tlt_up.astype(int) + hyg_up.astype(int)
    
    long_ok = proxy(long) >= threshold      # Trend confirmed
    medium_ok = proxy(medium) >= threshold   # Timing
    short_ok = proxy(short) >= max(1, threshold - 1)  # Entry confirm (looser)
    
    return (long_ok & medium_ok & short_ok).astype(int).shift(1)


# ── Backtest Engine ───────────────────────────────────────────────

def backtest_signal(df, signal, cost=0.001):
    """Simple long/flat backtest with transaction costs."""
    ret = df['btc_close'].pct_change()
    
    # Align
    signal = signal.reindex(ret.index).fillna(0)
    
    # Transaction costs on position changes
    trades = signal.diff().abs().fillna(0)
    costs = trades * cost
    
    strat_ret = signal * ret - costs
    
    # Trim to valid
    valid = strat_ret.dropna()
    if len(valid) < 252:
        return {'sharpe': -1, 'cagr': 0, 'maxdd': -1, 'trades': 0, 'exposure': 0}
    
    cum = (1 + valid).cumprod()
    years = len(valid) / 252
    cagr = (cum.iloc[-1] ** (1 / years) - 1) * 100
    maxdd = ((cum / cum.cummax()) - 1).min() * 100
    sharpe = valid.mean() / valid.std() * np.sqrt(252) if valid.std() > 0 else 0
    trades_count = int(trades.sum())
    exposure = signal.mean() * 100
    
    return {
        'sharpe': round(sharpe, 3),
        'cagr': round(cagr, 1),
        'maxdd': round(maxdd, 1),
        'trades': trades_count,
        'exposure': round(exposure, 1)
    }

def walk_forward_test(df, signal_func, n_folds=7, train_days=730, test_days=180):
    """Walk-forward OOS test."""
    ret = df['btc_close'].pct_change()
    oos_returns = []
    
    start_idx = 400  # Skip warmup
    total_days = len(df) - start_idx
    fold_size = train_days + test_days
    
    for fold in range(n_folds):
        fold_start = start_idx + fold * test_days
        train_end = fold_start + train_days
        test_end = train_end + test_days
        
        if test_end > len(df):
            break
        
        # Generate signal on full data (no lookahead since signal is shifted)
        signal = signal_func(df)
        
        # Collect OOS returns
        test_signal = signal.iloc[train_end:test_end]
        test_ret = ret.iloc[train_end:test_end]
        trades = test_signal.diff().abs().fillna(0)
        oos = test_signal * test_ret - trades * 0.001
        oos_returns.append(oos)
    
    if not oos_returns:
        return -1
    
    all_oos = pd.concat(oos_returns).dropna()
    if len(all_oos) < 100:
        return -1
    return round(all_oos.mean() / all_oos.std() * np.sqrt(252), 3) if all_oos.std() > 0 else 0


# ── Main ──────────────────────────────────────────────────────────

def run_exploration():
    print("=" * 70)
    print("LOOKBACK SCHEME EXPLORATION")
    print("=" * 70)
    
    print("\nLoading data...")
    df = load_data()
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} days)")
    
    schemes = {
        'Fixed-180d': lambda d: scheme_fixed(d, lookback=180, threshold=2),
        'Fixed-120d': lambda d: scheme_fixed(d, lookback=120, threshold=2),
        'Fixed-252d': lambda d: scheme_fixed(d, lookback=252, threshold=2),
        'Dual-Speed(60/180)': lambda d: scheme_dual_speed(d, fast=60, slow=180, threshold=2),
        'Dual-Speed(90/252)': lambda d: scheme_dual_speed(d, fast=90, slow=252, threshold=2),
        'Adaptive-Vol': lambda d: scheme_adaptive_vol(d, base_lb=120, vol_window=60),
        'Expanding-90': lambda d: scheme_expanding(d, min_lb=90, threshold=2),
        'MultiScale-Vote': lambda d: scheme_multiscale_vote(d, windows=[60, 120, 180, 252], min_votes=2),
        'MultiScale-3of4': lambda d: scheme_multiscale_vote(d, windows=[60, 120, 180, 252], min_votes=3),
        'Regime-Aware': lambda d: scheme_regime_aware(d, vol_window=60, low_vol_lb=252, high_vol_lb=60),
        'Exp-Decay-90': lambda d: scheme_exp_decay(d, half_life=90, threshold=2),
        'Exp-Decay-60': lambda d: scheme_exp_decay(d, half_life=60, threshold=2),
        'Fractal(40/120/252)': lambda d: scheme_fractal(d, short=40, medium=120, long=252),
        'Fractal(60/180/365)': lambda d: scheme_fractal(d, short=60, medium=180, long=365),
    }
    
    results = {}
    print(f"\nTesting {len(schemes)} schemes...\n")
    print(f"{'Scheme':<25} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'Trades':>7} {'Exp%':>6} {'OOS Sharpe':>11}")
    print("-" * 75)
    
    for name, func in schemes.items():
        signal = func(df)
        bt = backtest_signal(df, signal)
        oos_sharpe = walk_forward_test(df, func)
        
        results[name] = {**bt, 'oos_sharpe': oos_sharpe}
        
        print(f"{name:<25} {bt['sharpe']:>7.3f} {bt['cagr']:>7.1f} {bt['maxdd']:>8.1f} {bt['trades']:>7} {bt['exposure']:>6.1f} {oos_sharpe:>11.3f}")
    
    # Rank by OOS Sharpe
    ranked = sorted(results.items(), key=lambda x: x[1]['oos_sharpe'], reverse=True)
    
    print("\n" + "=" * 70)
    print("RANKING BY OOS SHARPE")
    print("=" * 70)
    for i, (name, r) in enumerate(ranked, 1):
        marker = " ★" if i <= 3 else ""
        print(f"  {i}. {name:<25} OOS: {r['oos_sharpe']:>7.3f}  IS: {r['sharpe']:>7.3f}  CAGR: {r['cagr']:>7.1f}%{marker}")
    
    # Save
    outpath = os.path.expanduser('~/Desktop/maestro/data/research/lookback_schemes_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")
    
    # Return best scheme name for Optuna phase
    best_name = ranked[0][0]
    print(f"\n{'=' * 70}")
    print(f"BEST SCHEME: {best_name} (OOS Sharpe {ranked[0][1]['oos_sharpe']:.3f})")
    print(f"{'=' * 70}")
    
    return best_name, df

def run_optuna_optimization(best_scheme, df):
    """Optuna-optimize the winning scheme."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\n⚠ Optuna not installed. pip install optuna")
        return
    
    print(f"\n{'=' * 70}")
    print(f"OPTUNA HYPEROPTIMIZATION: {best_scheme}")
    print(f"{'=' * 70}")
    
    def objective(trial):
        if 'Fixed' in best_scheme:
            lb = trial.suggest_int('lookback', 60, 365, step=10)
            thr = trial.suggest_int('threshold', 1, 3)
            signal = scheme_fixed(df, lookback=lb, threshold=thr)
        elif 'Dual' in best_scheme:
            fast = trial.suggest_int('fast', 20, 120, step=10)
            slow = trial.suggest_int('slow', 120, 365, step=10)
            thr = trial.suggest_int('threshold', 1, 3)
            if fast >= slow:
                return -10
            signal = scheme_dual_speed(df, fast=fast, slow=slow, threshold=thr)
        elif 'MultiScale' in best_scheme:
            w1 = trial.suggest_int('w1', 20, 80, step=10)
            w2 = trial.suggest_int('w2', 80, 160, step=10)
            w3 = trial.suggest_int('w3', 140, 252, step=10)
            w4 = trial.suggest_int('w4', 200, 400, step=10)
            min_v = trial.suggest_int('min_votes', 2, 4)
            thr = trial.suggest_int('threshold', 1, 3)
            signal = scheme_multiscale_vote(df, windows=[w1, w2, w3, w4], min_votes=min_v, threshold=thr)
        elif 'Exp-Decay' in best_scheme:
            hl = trial.suggest_int('half_life', 30, 180, step=10)
            thr = trial.suggest_int('threshold', 1, 3)
            signal = scheme_exp_decay(df, half_life=hl, threshold=thr)
        elif 'Fractal' in best_scheme:
            short = trial.suggest_int('short', 20, 80, step=10)
            medium = trial.suggest_int('medium', 60, 200, step=10)
            long = trial.suggest_int('long', 150, 400, step=10)
            thr = trial.suggest_int('threshold', 1, 3)
            if short >= medium or medium >= long:
                return -10
            signal = scheme_fractal(df, short=short, medium=medium, long=long, threshold=thr)
        elif 'Regime' in best_scheme:
            vol_w = trial.suggest_int('vol_window', 30, 120, step=10)
            low_lb = trial.suggest_int('low_vol_lb', 150, 365, step=10)
            high_lb = trial.suggest_int('high_vol_lb', 30, 150, step=10)
            thr = trial.suggest_int('threshold', 1, 3)
            signal = scheme_regime_aware(df, vol_window=vol_w, low_vol_lb=low_lb, high_vol_lb=high_lb, threshold=thr)
        elif 'Adaptive' in best_scheme:
            base = trial.suggest_int('base_lb', 60, 252, step=10)
            vol_w = trial.suggest_int('vol_window', 30, 120, step=10)
            signal = scheme_adaptive_vol(df, base_lb=base, vol_window=vol_w)
        elif 'Expanding' in best_scheme:
            min_lb = trial.suggest_int('min_lb', 40, 180, step=10)
            thr = trial.suggest_int('threshold', 1, 3)
            signal = scheme_expanding(df, min_lb=min_lb, threshold=thr)
        else:
            return -10
        
        bt = backtest_signal(df, signal)
        
        # Penalize low exposure or too many trades
        if bt['exposure'] < 10 or bt['exposure'] > 90:
            return bt['sharpe'] * 0.5
        
        return bt['sharpe']
    
    study = optuna.create_study(direction='maximize', study_name=f'lookback_{best_scheme}')
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    
    print(f"\nBest Sharpe: {study.best_value:.3f}")
    print(f"Best Params: {study.best_params}")
    
    # Run walk-forward on best params
    bp = study.best_params
    print("\nWalk-forward validation of optimized params...")
    
    # Reconstruct signal func with best params
    if 'Fixed' in best_scheme:
        opt_func = lambda d: scheme_fixed(d, lookback=bp['lookback'], threshold=bp['threshold'])
    elif 'Dual' in best_scheme:
        opt_func = lambda d: scheme_dual_speed(d, fast=bp['fast'], slow=bp['slow'], threshold=bp['threshold'])
    elif 'MultiScale' in best_scheme:
        opt_func = lambda d: scheme_multiscale_vote(d, windows=[bp['w1'], bp['w2'], bp['w3'], bp['w4']], min_votes=bp['min_votes'], threshold=bp['threshold'])
    elif 'Exp-Decay' in best_scheme:
        opt_func = lambda d: scheme_exp_decay(d, half_life=bp['half_life'], threshold=bp['threshold'])
    elif 'Fractal' in best_scheme:
        opt_func = lambda d: scheme_fractal(d, short=bp['short'], medium=bp['medium'], long=bp['long'], threshold=bp['threshold'])
    elif 'Regime' in best_scheme:
        opt_func = lambda d: scheme_regime_aware(d, vol_window=bp['vol_window'], low_vol_lb=bp['low_vol_lb'], high_vol_lb=bp['high_vol_lb'], threshold=bp['threshold'])
    elif 'Adaptive' in best_scheme:
        opt_func = lambda d: scheme_adaptive_vol(d, base_lb=bp['base_lb'], vol_window=bp['vol_window'])
    elif 'Expanding' in best_scheme:
        opt_func = lambda d: scheme_expanding(d, min_lb=bp['min_lb'], threshold=bp['threshold'])
    
    opt_signal = opt_func(df)
    opt_bt = backtest_signal(df, opt_signal)
    oos_sharpe = walk_forward_test(df, opt_func)
    
    print(f"\nOPTIMIZED RESULTS:")
    print(f"  IS Sharpe:  {opt_bt['sharpe']:.3f}")
    print(f"  OOS Sharpe: {oos_sharpe:.3f}")
    print(f"  CAGR:       {opt_bt['cagr']:.1f}%")
    print(f"  MaxDD:      {opt_bt['maxdd']:.1f}%")
    print(f"  Trades:     {opt_bt['trades']}")
    print(f"  Exposure:   {opt_bt['exposure']:.1f}%")
    
    # Save
    outpath = os.path.expanduser('~/Desktop/maestro/data/research/lookback_optuna_results.json')
    result = {
        'scheme': best_scheme,
        'best_params': bp,
        'is_sharpe': opt_bt['sharpe'],
        'oos_sharpe': oos_sharpe,
        'cagr': opt_bt['cagr'],
        'maxdd': opt_bt['maxdd'],
        'trades': opt_bt['trades'],
        'exposure': opt_bt['exposure'],
        'n_trials': 200
    }
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {outpath}")
    
    # Compare to baseline
    print(f"\n{'=' * 70}")
    print("COMPARISON: Baseline (Fixed-180d) vs Optimized")
    print(f"{'=' * 70}")
    baseline_signal = scheme_fixed(df, lookback=180, threshold=2)
    baseline_bt = backtest_signal(df, baseline_signal)
    baseline_oos = walk_forward_test(df, lambda d: scheme_fixed(d, lookback=180, threshold=2))
    
    print(f"  {'Metric':<15} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    print(f"  {'IS Sharpe':<15} {baseline_bt['sharpe']:>10.3f} {opt_bt['sharpe']:>10.3f} {opt_bt['sharpe']-baseline_bt['sharpe']:>+10.3f}")
    print(f"  {'OOS Sharpe':<15} {baseline_oos:>10.3f} {oos_sharpe:>10.3f} {oos_sharpe-baseline_oos:>+10.3f}")
    print(f"  {'CAGR%':<15} {baseline_bt['cagr']:>10.1f} {opt_bt['cagr']:>10.1f} {opt_bt['cagr']-baseline_bt['cagr']:>+10.1f}")
    print(f"  {'MaxDD%':<15} {baseline_bt['maxdd']:>10.1f} {opt_bt['maxdd']:>10.1f} {opt_bt['maxdd']-baseline_bt['maxdd']:>+10.1f}")

if __name__ == '__main__':
    best_name, df = run_exploration()
    run_optuna_optimization(best_name, df)
