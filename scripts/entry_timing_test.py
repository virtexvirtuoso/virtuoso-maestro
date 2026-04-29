#!/usr/bin/env python3
"""Entry Timing Optimization for V4d SMA50 Strategy"""

import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data")
SYMBOLS = ['BTC', 'ETH', 'SOL']
TRAILING_STOPS = {'BTC': 0.12, 'ETH': 0.15, 'SOL': 0.08}
N_FOLDS = 14

def load_data(symbol):
    df = pd.read_csv(f"{DATA_DIR}/spot/{symbol}_spot_daily.csv", parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.rename(columns={'Date':'date','Close':'close','High':'high','Low':'low','Open':'open','Volume':'volume'}, inplace=True)
    
    # Load taker data
    try:
        import duckdb
        con = duckdb.connect(f"{DATA_DIR}/maestro.duckdb", read_only=True)
        taker = con.execute(f"SELECT * FROM cg_taker_volume WHERE symbol='{symbol}' ORDER BY date").df()
        taker['date'] = pd.to_datetime(taker['date'])
        df = df.merge(taker[['date','taker_buy_volume_usd','taker_sell_volume_usd']], on='date', how='left')
    except:
        df['taker_buy_volume_usd'] = np.nan
        df['taker_sell_volume_usd'] = np.nan
    
    # Indicators
    df['sma50'] = df['close'].rolling(50).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['taker_ratio'] = df['taker_buy_volume_usd'] / df['taker_sell_volume_usd']
    df['prev_close'] = df['close'].shift(1)
    
    # SMA50 cross signal: close crosses above sma50
    df['above_sma'] = (df['close'] > df['sma50']).astype(int)
    df['sma_cross'] = (df['above_sma'] == 1) & (df['above_sma'].shift(1) == 0)
    
    return df.dropna(subset=['sma50']).reset_index(drop=True)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def simulate_variant(df, variant, trail_pct):
    """Simulate a variant. Returns list of trade dicts."""
    trades = []
    i = 0
    n = len(df)
    
    while i < n:
        # Find next SMA50 cross
        while i < n and not df.loc[i, 'sma_cross']:
            i += 1
        if i >= n:
            break
        
        cross_idx = i
        cross_price = df.loc[i, 'close']
        sma_at_cross = df.loc[i, 'sma50']
        
        # Determine entry based on variant
        entry_idx, entry_price, entry_weights = get_entry(df, cross_idx, variant)
        
        if entry_idx is None:
            i = cross_idx + 1
            continue
        
        # Now simulate the trade with trailing stop until close < sma50
        peak = entry_price
        exit_idx = None
        exit_price = None
        
        for j in range(entry_idx + 1, n):
            peak = max(peak, df.loc[j, 'high'])
            # Trailing stop hit
            if df.loc[j, 'low'] <= peak * (1 - trail_pct):
                exit_price = peak * (1 - trail_pct)
                exit_idx = j
                break
            # Exit if close < sma50
            if df.loc[j, 'close'] < df.loc[j, 'sma50']:
                exit_price = df.loc[j + 1, 'open'] if j + 1 < n else df.loc[j, 'close']
                exit_idx = j + 1 if j + 1 < n else j
                break
        
        if exit_idx is None:
            exit_idx = n - 1
            exit_price = df.loc[exit_idx, 'close']
        
        # For scale-in, compute weighted avg entry
        if isinstance(entry_weights, list):
            avg_entry = sum(w * p for w, p in entry_weights) / sum(w for w, _ in entry_weights)
        else:
            avg_entry = entry_price
        
        trades.append({
            'cross_date': str(df.loc[cross_idx, 'date'].date()),
            'entry_date': str(df.loc[entry_idx, 'date'].date()),
            'exit_date': str(df.loc[exit_idx, 'date'].date()),
            'cross_price': cross_price,
            'entry_price': avg_entry,
            'exit_price': exit_price,
            'return': exit_price / avg_entry - 1,
            'days_waited': entry_idx - cross_idx,
            'missed': False
        })
        
        i = exit_idx + 1
    
    return trades

def get_entry(df, cross_idx, variant):
    n = len(df)
    ci = cross_idx
    
    # Entry is always on bar AFTER signal (N+1)
    base_entry = ci + 1
    if base_entry >= n:
        return None, None, None
    
    if variant == 'immediate':
        idx = base_entry
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'first_pullback':
        for j in range(base_entry, min(base_entry + 10, n)):
            if df.loc[j, 'close'] < df.loc[j, 'prev_close']:
                entry = j + 1 if j + 1 < n else j
                return entry, df.loc[entry, 'open'], None
        # Max wait exceeded, enter anyway
        idx = min(base_entry + 10, n - 1)
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'rsi_dip':
        for j in range(base_entry, min(base_entry + 15, n)):
            if df.loc[j, 'rsi'] < 40:
                entry = j + 1 if j + 1 < n else j
                return entry, df.loc[entry, 'open'], None
        idx = min(base_entry + 15, n - 1)
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'pullback_to_sma':
        for j in range(base_entry, min(base_entry + 20, n)):
            sma_val = df.loc[j, 'sma50']
            if df.loc[j, 'close'] <= sma_val * 1.02:
                entry = j + 1 if j + 1 < n else j
                return entry, df.loc[entry, 'open'], None
        idx = min(base_entry + 20, n - 1)
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'volume_confirmed':
        for j in range(ci, min(ci + 20, n)):
            if j >= 0 and df.loc[j, 'volume'] > 1.5 * df.loc[j, 'vol_ma20']:
                entry = j + 1 if j + 1 < n else j
                return entry, df.loc[entry, 'open'], None
        idx = min(base_entry + 20, n - 1)
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'taker_confirmed':
        for j in range(ci, min(ci + 20, n)):
            ratio = df.loc[j, 'taker_ratio']
            if not np.isnan(ratio) and ratio > 1.0:
                entry = j + 1 if j + 1 < n else j
                return entry, df.loc[entry, 'open'], None
        idx = min(base_entry + 20, n - 1)
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'limit_at_sma':
        sma_val = df.loc[ci, 'sma50']
        for j in range(base_entry, min(base_entry + 10, n)):
            if df.loc[j, 'low'] <= sma_val:
                return j, sma_val, None
        idx = min(base_entry + 10, n - 1)
        return idx, df.loc[idx, 'open'], None
    
    elif variant == 'scale_in':
        # 1/3 immediately
        idx = base_entry
        entry_price = df.loc[idx, 'open']
        weights = [(1/3, entry_price)]
        
        got_3pct = False
        got_5pct = False
        for j in range(idx + 1, min(idx + 20, n)):
            if not got_3pct and df.loc[j, 'low'] <= entry_price * 0.97:
                weights.append((1/3, entry_price * 0.97))
                got_3pct = True
            if not got_5pct and df.loc[j, 'low'] <= entry_price * 0.95:
                weights.append((1/3, entry_price * 0.95))
                got_5pct = True
                break
        
        # After 20 days, fill remaining
        remaining = 1.0 - sum(w for w, _ in weights)
        if remaining > 0.01:
            fill_idx = min(idx + 20, n - 1)
            weights.append((remaining, df.loc[fill_idx, 'open']))
        
        return idx, entry_price, weights
    
    return None, None, None

def walk_forward_backtest(df, variant, trail_pct, n_folds=14):
    """Expanding window walk-forward."""
    n = len(df)
    min_train = max(200, n // (n_folds + 1))
    fold_size = (n - min_train) // n_folds
    
    all_trades = []
    for fold in range(n_folds):
        test_start = min_train + fold * fold_size
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break
        
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)
        # Recalc indicators on full data up to test_end, then slice
        # For simplicity, use pre-computed indicators but only trade in test window
        sub_df = df.iloc[:test_end].reset_index(drop=True)
        
        trades = simulate_variant(sub_df, variant, trail_pct)
        # Filter to trades that entered during test window
        test_start_date = df.loc[test_start, 'date']
        test_end_date = df.loc[min(test_end - 1, n - 1), 'date']
        
        for t in trades:
            edate = pd.Timestamp(t['entry_date'])
            if test_start_date <= edate <= test_end_date:
                t['fold'] = fold
                all_trades.append(t)
    
    return all_trades

def compute_metrics(trades, df):
    if not trades:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'n_trades': 0, 'avg_return': 0, 'win_rate': 0}
    
    returns = [t['return'] for t in trades]
    n_trades = len(trades)
    avg_ret = np.mean(returns)
    win_rate = sum(1 for r in returns if r > 0) / n_trades
    
    # Approximate annualized
    total_days = (pd.Timestamp(trades[-1]['exit_date']) - pd.Timestamp(trades[0]['entry_date'])).days
    total_days = max(total_days, 1)
    years = total_days / 365.25
    
    cumret = np.prod([1 + r for r in returns])
    cagr = cumret ** (1 / max(years, 0.01)) - 1
    
    # Max drawdown from equity curve
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()
    
    # Sharpe (rough: avg trade return / std, annualized by sqrt(trades/year))
    if len(returns) > 1 and np.std(returns) > 0:
        trades_per_year = n_trades / max(years, 0.01)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    avg_days_waited = np.mean([t['days_waited'] for t in trades])
    
    return {
        'sharpe': round(sharpe, 3),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'avg_return': round(avg_ret * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'avg_days_waited': round(avg_days_waited, 1),
        'total_return': round((np.prod([1+r for r in returns]) - 1) * 100, 2)
    }

# ============ MAIN ============
VARIANTS = ['immediate', 'first_pullback', 'rsi_dip', 'pullback_to_sma', 
            'volume_confirmed', 'taker_confirmed', 'limit_at_sma', 'scale_in']

VARIANT_NAMES = {
    'immediate': '1. Immediate (Baseline)',
    'first_pullback': '2. First Pullback',
    'rsi_dip': '3. RSI Dip (<40)',
    'pullback_to_sma': '4. Pullback to SMA',
    'volume_confirmed': '5. Volume Confirmed',
    'taker_confirmed': '6. Taker Confirmed',
    'limit_at_sma': '7. Limit at SMA50',
    'scale_in': '8. Scale-In (3 tranche)'
}

results = {}
all_data = {}

for sym in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"  {sym} - Entry Timing Comparison")
    print(f"{'='*60}")
    
    df = load_data(sym)
    all_data[sym] = df
    results[sym] = {}
    
    # Count total SMA50 crosses
    total_crosses = df['sma_cross'].sum()
    print(f"Total SMA50 bullish crosses: {total_crosses}")
    print(f"Data: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} ({len(df)} days)")
    
    baseline_entries = {}
    
    for var in VARIANTS:
        trades = walk_forward_backtest(df, var, TRAILING_STOPS[sym])
        metrics = compute_metrics(trades, df)
        results[sym][var] = metrics
        
        if var == 'immediate':
            baseline_entries = {t['cross_date']: t['entry_price'] for t in trades}
            metrics['price_improvement'] = 0.0
            metrics['signals_missed'] = 0
        else:
            # Price improvement vs baseline
            improvements = []
            matched = 0
            for t in trades:
                base_price = baseline_entries.get(t['cross_date'])
                if base_price and base_price > 0:
                    imp = (base_price - t['entry_price']) / base_price * 100
                    improvements.append(imp)
                    matched += 1
            
            metrics['price_improvement'] = round(np.mean(improvements), 2) if improvements else 0
            metrics['signals_missed'] = max(0, results[sym]['immediate']['n_trades'] - len(trades))
    
    # Print table
    print(f"\n{'Variant':<28} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>7} {'Trades':>7} {'WinR%':>6} {'AvgWait':>8} {'PriceImp%':>10} {'Missed':>7}")
    print("-" * 100)
    for var in VARIANTS:
        m = results[sym][var]
        print(f"{VARIANT_NAMES[var]:<28} {m['sharpe']:>7.2f} {m['cagr']:>7.1f} {m['max_dd']:>7.1f} {m['n_trades']:>7} {m['win_rate']:>6.1f} {m['avg_days_waited']:>8.1f} {m.get('price_improvement',0):>10.2f} {m.get('signals_missed',0):>7}")

# ============ PORTFOLIO SUMMARY ============
print(f"\n{'='*60}")
print(f"  PORTFOLIO SUMMARY (Equal Weight BTC/ETH/SOL)")
print(f"{'='*60}")

print(f"\n{'Variant':<28} {'AvgSharpe':>10} {'AvgCAGR%':>9} {'AvgMaxDD%':>10} {'AvgPriceImp%':>13} {'TotalMissed':>12}")
print("-" * 90)

for var in VARIANTS:
    sharpes = [results[s][var]['sharpe'] for s in SYMBOLS]
    cagrs = [results[s][var]['cagr'] for s in SYMBOLS]
    dds = [results[s][var]['max_dd'] for s in SYMBOLS]
    pimps = [results[s][var].get('price_improvement', 0) for s in SYMBOLS]
    missed = sum(results[s][var].get('signals_missed', 0) for s in SYMBOLS)
    
    print(f"{VARIANT_NAMES[var]:<28} {np.mean(sharpes):>10.2f} {np.mean(cagrs):>9.1f} {np.mean(dds):>10.1f} {np.mean(pimps):>13.2f} {missed:>12}")

# ============ VERDICT ============
print(f"\n{'='*60}")
print(f"  VERDICT")
print(f"{'='*60}")

# Find best variant by avg Sharpe across portfolio
best_var = None
best_sharpe = -999
for var in VARIANTS:
    avg_s = np.mean([results[s][var]['sharpe'] for s in SYMBOLS])
    if avg_s > best_sharpe:
        best_sharpe = avg_s
        best_var = var

baseline_sharpe = np.mean([results[s]['immediate']['sharpe'] for s in SYMBOLS])
best_pimp = np.mean([results[s][best_var].get('price_improvement', 0) for s in SYMBOLS])
best_missed = sum(results[s][best_var].get('signals_missed', 0) for s in SYMBOLS)

print(f"\nBest variant: {VARIANT_NAMES[best_var]}")
print(f"  Avg Sharpe: {best_sharpe:.3f} vs Baseline: {baseline_sharpe:.3f} (delta: {best_sharpe - baseline_sharpe:+.3f})")
print(f"  Avg price improvement: {best_pimp:+.2f}%")
print(f"  Total signals missed: {best_missed}")

if best_var == 'immediate':
    print(f"\n✅ VERDICT: Immediate entry is OPTIMAL. Waiting does NOT improve risk-adjusted returns.")
    print(f"   The cost of missed/delayed entries outweighs any price improvement from waiting.")
elif best_sharpe > baseline_sharpe * 1.1:
    print(f"\n✅ VERDICT: {VARIANT_NAMES[best_var]} is SIGNIFICANTLY better than immediate entry.")
    print(f"   Sharpe improvement of {((best_sharpe/baseline_sharpe)-1)*100:.1f}% justifies the wait.")
else:
    print(f"\n⚠️ VERDICT: {VARIANT_NAMES[best_var]} is marginally better but improvement is modest.")
    print(f"   Consider sticking with immediate entry for simplicity unless price improvement is meaningful.")

# For each symbol, which is best?
print(f"\nPer-asset best:")
for sym in SYMBOLS:
    best_v = max(VARIANTS, key=lambda v: results[sym][v]['sharpe'])
    m = results[sym][best_v]
    print(f"  {sym}: {VARIANT_NAMES[best_v]} (Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1f}%)")

# ============ SAVE ============
os.makedirs(f"{DATA_DIR}/backtest_results", exist_ok=True)
output = {
    'test': 'entry_timing_optimization',
    'date': '2026-02-15',
    'symbols': SYMBOLS,
    'trailing_stops': TRAILING_STOPS,
    'n_folds': N_FOLDS,
    'results': results,
    'best_variant': best_var,
    'best_variant_name': VARIANT_NAMES[best_var],
    'baseline_sharpe': baseline_sharpe,
    'best_sharpe': best_sharpe,
}

with open(f"{DATA_DIR}/backtest_results/entry_timing_test.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to {DATA_DIR}/backtest_results/entry_timing_test.json")
