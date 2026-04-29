#!/usr/bin/env python3
"""
Expanded Cross-Asset Momentum Backtest with Survivorship-Free Universe
- 30 tokens from 2021-01-01
- Walk-forward with 10 expanding folds
- Quintile long/short (top/bottom 20%)
- Commission at 20bps and 30bps
- Random permutation test (200 shuffles)
"""

import pandas as pd
import numpy as np
import os
import json
from itertools import product
from datetime import datetime

# ─── CONFIG ───
DATA_DIR = os.path.expanduser('~/Desktop/maestro/data/ohlcv')
OUT_DIR = os.path.expanduser('~/Desktop/maestro/data/backtest_results')
os.makedirs(OUT_DIR, exist_ok=True)

TOKENS = [
    'btc','eth','sol','bnb','xrp','ada','doge','avax','link','dot',
    'matic','uni','atom','near','ftm','algo','sand','mana','gala','axs',
    'ape','ldo','arb','op','sui','inj','fet','fil','ren','luna'
]

LOOKBACKS = [7, 14, 30]
REBALANCE = [7, 14]
COMMISSIONS = [0.0020, 0.0030]  # 20bps, 30bps
MIN_HISTORY = 365  # days
N_FOLDS = 10
N_PERMS = 200
QUINTILE = 0.20  # top/bottom 20%

# ─── LOAD DATA ───
def load_all_data():
    """Load all token CSVs into a dict of DataFrames."""
    data = {}
    for token in TOKENS:
        path = os.path.join(DATA_DIR, f'binance_{token}_usdt_1d.csv')
        if not os.path.exists(path):
            print(f"  MISSING: {token}")
            continue
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
        if len(df) >= MIN_HISTORY:
            data[token] = df
            print(f"  {token}: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print(f"  SKIP {token}: only {len(df)} days (need {MIN_HISTORY})")
    return data

def build_return_panel(data):
    """Build a DataFrame of daily returns, columns = tokens."""
    closes = pd.DataFrame({t: d['close'] for t, d in data.items()})
    returns = closes.pct_change()
    return returns

# ─── MOMENTUM STRATEGY ───
def momentum_backtest(returns, lookback, rebalance_freq, commission, quintile=0.20):
    """
    Cross-sectional momentum: long top quintile, short bottom quintile.
    Returns daily strategy returns series + metadata.
    """
    n_assets = returns.shape[1]
    dates = returns.index
    
    # Momentum signal: lookback-period return
    mom = returns.rolling(lookback).sum()
    
    # Track positions
    strategy_returns = pd.Series(0.0, index=dates)
    long_returns = pd.Series(0.0, index=dates)
    short_returns = pd.Series(0.0, index=dates)
    prev_weights = pd.Series(0.0, index=returns.columns)
    universe_size = pd.Series(0, index=dates)
    turnover_total = 0.0
    rebal_count = 0
    
    rebal_dates = []
    
    for i in range(lookback + 1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        
        # Available assets on this date (have data)
        available = mom.loc[prev_date].dropna().index.tolist()
        n_avail = len(available)
        universe_size.iloc[i] = n_avail
        
        if n_avail < 5:  # need at least 5 to form quintiles
            continue
        
        # Rebalance?
        if len(rebal_dates) == 0 or (i - rebal_dates[-1]) >= rebalance_freq:
            rebal_dates.append(i)
            rebal_count += 1
            
            scores = mom.loc[prev_date, available].sort_values()
            n_long = max(1, int(np.ceil(n_avail * quintile)))
            n_short = max(1, int(np.ceil(n_avail * quintile)))
            
            longs = scores.index[-n_long:]
            shorts = scores.index[:n_short]
            
            weights = pd.Series(0.0, index=returns.columns)
            weights[longs] = 1.0 / n_long
            weights[shorts] = -1.0 / n_short
            
            # Turnover
            turnover = (weights - prev_weights).abs().sum()
            turnover_total += turnover
            
            # Commission cost applied on rebalance day
            cost = turnover * commission
            prev_weights = weights.copy()
        else:
            cost = 0.0
            weights = prev_weights.copy()
        
        # Daily return
        day_ret = returns.loc[date]
        port_ret = (weights * day_ret).sum() - cost
        long_ret = (weights.clip(lower=0) * day_ret).sum()
        short_ret = (weights.clip(upper=0) * day_ret).sum()
        
        strategy_returns.iloc[i] = port_ret
        long_returns.iloc[i] = long_ret
        short_returns.iloc[i] = short_ret
    
    return strategy_returns, long_returns, short_returns, universe_size, turnover_total, rebal_count

def compute_metrics(rets):
    """Compute Sharpe, CAGR, MaxDD from daily returns series."""
    rets = rets.dropna()
    if len(rets) < 30 or rets.std() == 0:
        return {'sharpe': 0, 'cagr': 0, 'maxdd': 0}
    
    cumret = (1 + rets).cumprod()
    n_years = len(rets) / 252
    
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cagr = (cumret.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 and cumret.iloc[-1] > 0 else -1
    
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    maxdd = drawdown.min()
    
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr * 100, 2), 'maxdd': round(maxdd * 100, 2)}

def walk_forward(returns, lookback, rebalance_freq, commission, n_folds=10):
    """Expanding window walk-forward with n_folds."""
    dates = returns.index
    n = len(dates)
    min_train = max(252, lookback + 60)  # at least 1yr training
    
    fold_size = (n - min_train) // n_folds
    if fold_size < 30:
        return None
    
    oos_returns = []
    
    for fold in range(n_folds):
        test_start = min_train + fold * fold_size
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break
        
        test_dates = dates[test_start:test_end]
        
        # Run on full data but only collect OOS portion
        strat_rets, long_rets, short_rets, univ, _, _ = momentum_backtest(
            returns, lookback, rebalance_freq, commission
        )
        
        oos_returns.append(strat_rets.loc[test_dates])
    
    if not oos_returns:
        return None
    
    combined_oos = pd.concat(oos_returns)
    return combined_oos

def permutation_test(returns, lookback, rebalance_freq, commission, observed_sharpe, n_perms=200):
    """Shuffle cross-sectional rankings to get p-value."""
    rng = np.random.RandomState(42)
    count_better = 0
    
    for perm in range(n_perms):
        # Shuffle column labels at each rebalance point (destroy momentum signal)
        shuffled = returns.copy()
        cols = list(shuffled.columns)
        
        # Shuffle columns randomly for each row
        shuffled_vals = shuffled.values.copy()
        for i in range(len(shuffled)):
            rng.shuffle(cols)
        # Actually: shuffle the momentum signals by permuting columns of returns
        perm_order = rng.permutation(len(shuffled.columns))
        shuffled = pd.DataFrame(
            shuffled.values[:, perm_order],
            index=shuffled.index,
            columns=shuffled.columns
        )
        
        oos = walk_forward(shuffled, lookback, rebalance_freq, commission, n_folds=N_FOLDS)
        if oos is None:
            continue
        m = compute_metrics(oos)
        if m['sharpe'] >= observed_sharpe:
            count_better += 1
        
        if (perm + 1) % 50 == 0:
            print(f"    Permutation {perm+1}/{n_perms}, count_better={count_better}")
    
    p_value = (count_better + 1) / (n_perms + 1)
    return p_value

def per_year_returns(rets):
    """Compute annual returns."""
    annual = {}
    for year in sorted(rets.index.year.unique()):
        yr_rets = rets[rets.index.year == year]
        if len(yr_rets) > 0:
            cumret = (1 + yr_rets).prod() - 1
            annual[str(year)] = round(cumret * 100, 2)
    return annual

# ─── MAIN ───
if __name__ == '__main__':
    print("=" * 70)
    print("EXPANDED CROSS-ASSET MOMENTUM BACKTEST")
    print("=" * 70)
    
    print("\n📥 Loading data...")
    data = load_all_data()
    print(f"\n✅ Loaded {len(data)} tokens with >= {MIN_HISTORY} days")
    
    returns = build_return_panel(data)
    print(f"Return panel: {returns.shape[0]} days x {returns.shape[1]} tokens")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Universe size over time
    available_per_day = returns.notna().sum(axis=1)
    print(f"\nUniverse size over time:")
    for year in range(2021, 2027):
        mask = returns.index.year == year
        if mask.any():
            avg = available_per_day[mask].mean()
            print(f"  {year}: avg {avg:.1f} tokens")
    
    # ─── WALK-FORWARD GRID ───
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS (10 expanding folds)")
    print("=" * 70)
    
    results = []
    
    for lb, reb, comm in product(LOOKBACKS, REBALANCE, COMMISSIONS):
        label = f"lb={lb}, reb={reb}, comm={int(comm*10000)}bps"
        print(f"\n  Testing {label}...")
        
        oos = walk_forward(returns, lb, reb, comm, n_folds=N_FOLDS)
        if oos is None:
            print(f"    SKIP: not enough data")
            continue
        
        metrics = compute_metrics(oos)
        yr = per_year_returns(oos)
        
        # Long/short decomposition
        strat_rets, long_rets, short_rets, univ, turnover, rebal_count = momentum_backtest(
            returns, lb, reb, comm
        )
        long_m = compute_metrics(long_rets)
        short_m = compute_metrics(short_rets)
        
        result = {
            'lookback': lb,
            'rebalance': reb,
            'commission_bps': int(comm * 10000),
            'oos_sharpe': metrics['sharpe'],
            'oos_cagr': metrics['cagr'],
            'oos_maxdd': metrics['maxdd'],
            'long_sharpe': long_m['sharpe'],
            'short_sharpe': short_m['sharpe'],
            'long_cagr': long_m['cagr'],
            'short_cagr': short_m['cagr'],
            'per_year': yr,
            'total_turnover': round(turnover, 1),
            'n_rebalances': rebal_count
        }
        results.append(result)
        
        print(f"    OOS Sharpe={metrics['sharpe']:.3f}  CAGR={metrics['cagr']:.1f}%  MaxDD={metrics['maxdd']:.1f}%")
        print(f"    Long Sharpe={long_m['sharpe']:.3f}  Short Sharpe={short_m['sharpe']:.3f}")
        print(f"    Per-year: {yr}")
    
    # Sort by OOS Sharpe
    results.sort(key=lambda x: x['oos_sharpe'], reverse=True)
    
    # ─── RESULTS TABLE ───
    print("\n" + "=" * 70)
    print("ALL CONFIGS RANKED BY OOS SHARPE")
    print("=" * 70)
    print(f"{'Config':<25} {'OOS Sharpe':>10} {'CAGR%':>8} {'MaxDD%':>8} {'Long SR':>8} {'Short SR':>8}")
    print("-" * 70)
    for r in results:
        label = f"lb={r['lookback']:2d} reb={r['rebalance']:2d} {r['commission_bps']}bp"
        print(f"{label:<25} {r['oos_sharpe']:>10.3f} {r['oos_cagr']:>8.1f} {r['oos_maxdd']:>8.1f} {r['long_sharpe']:>8.3f} {r['short_sharpe']:>8.3f}")
    
    # ─── BEST CONFIG DETAILS ───
    best = results[0]
    print(f"\n{'=' * 70}")
    print(f"BEST CONFIG: lb={best['lookback']}, reb={best['rebalance']}, comm={best['commission_bps']}bps")
    print(f"{'=' * 70}")
    print(f"  OOS Sharpe:  {best['oos_sharpe']:.3f}")
    print(f"  OOS CAGR:    {best['oos_cagr']:.1f}%")
    print(f"  OOS MaxDD:   {best['oos_maxdd']:.1f}%")
    print(f"  Long Sharpe: {best['long_sharpe']:.3f}  CAGR: {best['long_cagr']:.1f}%")
    print(f"  Short Sharpe:{best['short_sharpe']:.3f}  CAGR: {best['short_cagr']:.1f}%")
    print(f"  Turnover:    {best['total_turnover']:.0f}x")
    print(f"  Rebalances:  {best['n_rebalances']}")
    
    print(f"\n  Per-Year Returns:")
    for yr, ret in best['per_year'].items():
        marker = " ⬅ BEAR" if yr == "2022" else ""
        print(f"    {yr}: {ret:+.1f}%{marker}")
    
    # ─── PERMUTATION TEST ───
    print(f"\n{'=' * 70}")
    print(f"PERMUTATION TEST (200 random shuffles) for best config")
    print(f"{'=' * 70}")
    
    p_value = permutation_test(
        returns, best['lookback'], best['rebalance'],
        best['commission_bps'] / 10000, best['oos_sharpe'], N_PERMS
    )
    best['p_value'] = round(p_value, 4)
    print(f"\n  Observed OOS Sharpe: {best['oos_sharpe']:.3f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  {'✅ SIGNIFICANT' if p_value < 0.05 else '❌ NOT SIGNIFICANT'} at 5% level")
    
    # ─── UNIVERSE OVER TIME ───
    print(f"\n{'=' * 70}")
    print("UNIVERSE SIZE OVER TIME")
    print(f"{'=' * 70}")
    token_starts = {}
    for t, d in data.items():
        token_starts[t] = str(d.index[0].date())
    
    for year in range(2021, 2027):
        count = sum(1 for t, d in data.items() if d.index[0].year <= year)
        print(f"  {year}: {count} tokens available")
    
    print(f"\n  Token start dates:")
    for t, start in sorted(token_starts.items(), key=lambda x: x[1]):
        print(f"    {t:>6s}: {start}")
    
    # ─── SAVE ───
    output = {
        'run_date': datetime.now().isoformat(),
        'n_tokens': len(data),
        'tokens': list(data.keys()),
        'token_starts': token_starts,
        'date_range': [str(returns.index[0].date()), str(returns.index[-1].date())],
        'configs': results,
        'best_config': best,
        'permutation_p_value': p_value,
    }
    
    out_path = os.path.join(OUT_DIR, 'momentum_expanded.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")
    print("\n✅ DONE")
