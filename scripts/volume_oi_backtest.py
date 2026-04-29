#!/usr/bin/env python3
"""Volume Profile Breakout & Liquidation+OI Divergence Strategies Backtest"""

import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data")
DB_PATH = os.path.join(DATA_DIR, "maestro.duckdb")
RESULTS_DIR = os.path.join(DATA_DIR, "backtest_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX']
OI_SYMBOLS = ['BTC', 'ETH', 'SOL', 'AVAX']  # BNB may not have OI

con = duckdb.connect(DB_PATH, read_only=True)

# ─── Data Loading ───
def load_price(symbol):
    df = con.execute(f"SELECT date, open, high, low, close, volume FROM perps_daily WHERE symbol='{symbol}' ORDER BY date").df()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def load_taker(symbol):
    try:
        df = con.execute(f"SELECT date, taker_buy_volume_usd, taker_sell_volume_usd FROM cg_taker_volume WHERE symbol='{symbol}' ORDER BY date").df()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['taker_ratio'] = df['taker_buy_volume_usd'] / df['taker_sell_volume_usd'].replace(0, np.nan)
        return df
    except:
        return None

def load_liq(symbol):
    try:
        df = con.execute(f"SELECT date, aggregated_long_liquidation_usd, aggregated_short_liquidation_usd FROM cg_liquidations WHERE symbol='{symbol}' ORDER BY date").df()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['total_liq'] = df['aggregated_long_liquidation_usd'] + df['aggregated_short_liquidation_usd']
        return df
    except:
        return None

def load_oi(symbol):
    path = os.path.join(DATA_DIR, f"derivatives/{symbol.lower()}_oi_daily_full.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['timestamp'])
    df.set_index('date', inplace=True)
    df.rename(columns={'c': 'oi_close', 'o': 'oi_open', 'h': 'oi_high', 'l': 'oi_low'}, inplace=True)
    return df[['oi_close']]

# ─── Strategy Signal Generators ───
def sma50_cross(price):
    """Returns 1 on day price crosses above SMA50"""
    sma50 = price['close'].rolling(50).mean()
    cross = (price['close'] > sma50) & (price['close'].shift(1) <= sma50.shift(1))
    return cross.astype(int)

def s1_volume_confirmed_sma50(price):
    sma50 = price['close'].rolling(50).mean()
    vol_avg = price['volume'].rolling(20).mean()
    cross = (price['close'] > sma50) & (price['close'].shift(1) <= sma50.shift(1))
    vol_confirm = price['volume'] > 1.5 * vol_avg
    return (cross & vol_confirm).astype(int)

def s2_volume_surge_breakout(price):
    high20 = price['high'].rolling(20).max().shift(1)
    vol_avg = price['volume'].rolling(20).mean()
    breakout = price['close'] > high20
    vol_surge = price['volume'] > 2 * vol_avg
    return (breakout & vol_surge).astype(int)

def s3_volume_dryup_breakout(price):
    vol_avg = price['volume'].rolling(20).mean()
    # Check if volume has been < 50% of avg for 10+ days
    low_vol = price['volume'] < 0.5 * vol_avg
    low_vol_streak = low_vol.rolling(10).sum() >= 10
    vol_spike = price['volume'] > 1.5 * vol_avg
    signal = low_vol_streak.shift(1) & vol_spike
    # Direction: go with the day's move
    direction = np.sign(price['close'] - price['open'])
    return (signal.astype(int) * direction).fillna(0).astype(int)

def s4_taker_confirmed_breakout(price, taker):
    if taker is None:
        return pd.Series(0, index=price.index)
    merged = price.join(taker[['taker_ratio']], how='left')
    merged['taker_ratio'] = merged['taker_ratio'].ffill()
    high20 = price['high'].rolling(20).max().shift(1)
    breakout = price['close'] > high20
    taker_confirm = merged['taker_ratio'] > 1.1
    return (breakout & taker_confirm).astype(int)

def s5_v4_volume_confirmed(price):
    """V4 = SMA50 cross with volume > 1.2x avg"""
    sma50 = price['close'].rolling(50).mean()
    vol_avg = price['volume'].rolling(20).mean()
    cross = (price['close'] > sma50) & (price['close'].shift(1) <= sma50.shift(1))
    vol_confirm = price['volume'] > 1.2 * vol_avg
    return (cross & vol_confirm).astype(int)

def s6_liq_oi_continuation(price, liq, oi):
    if liq is None or oi is None:
        return pd.Series(0, index=price.index)
    merged = price.join(liq[['total_liq']], how='left').join(oi[['oi_close']], how='left')
    merged = merged.ffill()
    liq_high = merged['total_liq'] > merged['total_liq'].rolling(20).quantile(0.8)
    oi_rising = merged['oi_close'] > merged['oi_close'].shift(1)
    trend = np.sign(price['close'] - price['close'].rolling(20).mean())
    signal = liq_high & oi_rising
    return (signal.astype(int) * trend).fillna(0).astype(int)

def s7_liq_oi_exhaustion(price, liq, oi):
    if liq is None or oi is None:
        return pd.Series(0, index=price.index)
    merged = price.join(liq[['total_liq']], how='left').join(oi[['oi_close']], how='left')
    merged = merged.ffill()
    liq_high = merged['total_liq'] > merged['total_liq'].rolling(20).quantile(0.8)
    oi_falling = merged['oi_close'] < merged['oi_close'].shift(1)
    # Contrarian after 3-day cooling
    exhaust = liq_high & oi_falling
    # Use shift(3) for cooling period
    trend = np.sign(price['close'] - price['close'].rolling(20).mean())
    signal = exhaust.shift(3).fillna(False)
    # Contrarian = opposite of trend
    return (signal.astype(int) * (-trend)).fillna(0).astype(int)

def s8_oi_buildup_breakout(price, oi):
    if oi is None:
        return pd.Series(0, index=price.index)
    merged = price.join(oi[['oi_close']], how='left').ffill()
    # OI rising for 10+ days
    oi_rising = merged['oi_close'] > merged['oi_close'].shift(1)
    oi_streak = oi_rising.rolling(10).sum() >= 8  # 8 of 10 days rising
    # Price range-bound (< 5% range over 20d)
    price_range = (price['high'].rolling(20).max() - price['low'].rolling(20).min()) / price['close'].rolling(20).mean()
    range_bound = price_range < 0.10
    # Breakout above 20d high
    high20 = price['high'].rolling(20).max().shift(1)
    breakout_up = price['close'] > high20
    signal = oi_streak & range_bound & breakout_up
    return signal.astype(int)

def s9_oi_divergence(price, oi):
    """Price new highs but OI declining = weak, go flat/short"""
    if oi is None:
        return pd.Series(0, index=price.index)
    merged = price.join(oi[['oi_close']], how='left').ffill()
    new_high = price['close'] >= price['close'].rolling(20).max()
    oi_declining = merged['oi_close'].rolling(5).mean() < merged['oi_close'].rolling(20).mean()
    signal = new_high & oi_declining
    return (-signal.astype(int))  # Short/reduce signal

def s10_combined_score(price, taker, liq, oi):
    vol_avg = price['volume'].rolling(20).mean()
    vol_confirmed = (price['volume'] > 1.5 * vol_avg).astype(int)
    
    if taker is not None:
        merged_t = price.join(taker[['taker_ratio']], how='left').ffill()
        taker_confirmed = (merged_t['taker_ratio'] > 1.1).astype(int)
    else:
        taker_confirmed = pd.Series(0, index=price.index)
    
    if oi is not None:
        merged_oi = price.join(oi[['oi_close']], how='left').ffill()
        oi_rising_flag = (merged_oi['oi_close'] > merged_oi['oi_close'].shift(1)).astype(int)
    else:
        oi_rising_flag = pd.Series(0, index=price.index)
    
    if liq is not None:
        merged_l = price.join(liq[['total_liq']], how='left').ffill()
        low_liq = (merged_l['total_liq'] < merged_l['total_liq'].rolling(20).quantile(0.5)).astype(int)
    else:
        low_liq = pd.Series(0, index=price.index)
    
    score = vol_confirmed + taker_confirmed + oi_rising_flag + low_liq
    return (score >= 3).astype(int)

# ─── Baselines ───
def baseline_sma50(price):
    sma50 = price['close'].rolling(50).mean()
    return (price['close'] > sma50).astype(int)

def baseline_bnh(price):
    return pd.Series(1, index=price.index)

# ─── Backtest Engine ───
def compute_returns(price, signals, hold_days=10):
    """Signal on bar N, enter bar N+1, hold for hold_days or until signal flips"""
    daily_ret = price['close'].pct_change()
    # Shift signal by 1 to trade next day
    pos = signals.shift(1).fillna(0)
    # For long-only signals (0/1), hold position
    strat_ret = pos * daily_ret
    strat_ret = strat_ret.fillna(0)
    cum = (1 + strat_ret).cumprod()
    return strat_ret, cum

def calc_metrics(strat_ret, cum, price):
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    n_years = len(strat_ret) / 365.25
    ann_ret = (1 + total_ret) ** (1/max(n_years, 0.01)) - 1
    vol = strat_ret.std() * np.sqrt(365)
    sharpe = ann_ret / vol if vol > 0 else 0
    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    # Win rate
    trades = strat_ret[strat_ret != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0
    # Buy and hold
    bnh_ret = price['close'].iloc[-1] / price['close'].iloc[0] - 1 if len(price) > 1 else 0
    return {
        'total_return': round(float(total_ret * 100), 2),
        'ann_return': round(float(ann_ret * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd * 100), 2),
        'win_rate': round(float(win_rate * 100), 1),
        'n_signals': int((strat_ret != 0).sum()),
        'n_years': round(n_years, 1),
        'bnh_return': round(float(bnh_ret * 100), 2)
    }

def expanding_walkforward(price, signal_func, signal_args, n_folds=14):
    """Expanding window walk-forward"""
    n = len(price)
    min_train = max(100, n // (n_folds + 1))
    fold_size = (n - min_train) // n_folds
    
    if fold_size < 20:
        # Not enough data, use all
        signals = signal_func(*signal_args)
        signals = signals.reindex(price.index).fillna(0)
        strat_ret, cum = compute_returns(price, signals)
        return calc_metrics(strat_ret, cum, price), strat_ret
    
    all_oos_ret = pd.Series(0.0, index=price.index)
    
    actual_folds = min(n_folds, (n - min_train) // fold_size)
    for fold in range(actual_folds):
        oos_start = min_train + fold * fold_size
        oos_end = min(oos_start + fold_size, n)
        # Generate signals on all data up to oos_end (expanding)
        sub_price = price.iloc[:oos_end]
        sub_args = []
        for a in signal_args:
            if isinstance(a, pd.DataFrame) and a is not None:
                sub_args.append(a.reindex(sub_price.index))
            else:
                sub_args.append(a)
        
        signals = signal_func(*sub_args)
        signals = signals.reindex(sub_price.index).fillna(0)
        daily_ret = price['close'].pct_change()
        pos = signals.shift(1).fillna(0)
        fold_ret = pos * daily_ret
        all_oos_ret.iloc[oos_start:oos_end] = fold_ret.iloc[oos_start:oos_end]
    
    cum = (1 + all_oos_ret).cumprod()
    return calc_metrics(all_oos_ret, cum, price), all_oos_ret

def permutation_test(price, signal_func, signal_args, n_perms=500):
    """Permutation test for statistical significance"""
    # Get actual performance
    signals = signal_func(*signal_args)
    signals = signals.reindex(price.index).fillna(0)
    strat_ret, cum = compute_returns(price, signals)
    actual_sharpe = calc_metrics(strat_ret, cum, price)['sharpe']
    
    # Permute
    count_better = 0
    daily_ret = price['close'].pct_change().fillna(0)
    sig_values = signals.values.copy()
    
    for _ in range(n_perms):
        np.random.shuffle(sig_values)
        perm_sig = pd.Series(sig_values, index=signals.index)
        pos = perm_sig.shift(1).fillna(0)
        perm_ret = pos * daily_ret
        perm_cum = (1 + perm_ret).cumprod()
        perm_sharpe = calc_metrics(perm_ret, perm_cum, price)['sharpe']
        if perm_sharpe >= actual_sharpe:
            count_better += 1
    
    p_value = (count_better + 1) / (n_perms + 1)
    return p_value

# ─── Event Study for Volume Surges ───
def event_study_volume_surge(price):
    vol_avg = price['volume'].rolling(20).mean()
    surge = price['volume'] > 2 * vol_avg
    surge_dates = price.index[surge]
    
    horizons = [1, 3, 5, 10, 20]
    results = {}
    for h in horizons:
        fwd_rets = []
        for d in surge_dates:
            idx = price.index.get_loc(d)
            if idx + h < len(price):
                fwd_ret = price['close'].iloc[idx + h] / price['close'].iloc[idx] - 1
                fwd_rets.append(fwd_ret)
        if fwd_rets:
            results[f'{h}d'] = {
                'mean': round(np.mean(fwd_rets) * 100, 2),
                'median': round(np.median(fwd_rets) * 100, 2),
                'win_rate': round((np.array(fwd_rets) > 0).mean() * 100, 1),
                'n_events': len(fwd_rets)
            }
    return results

# ─── Main Execution ───
print("=" * 80)
print("VOLUME PROFILE BREAKOUT & LIQUIDATION+OI DIVERGENCE BACKTEST")
print("=" * 80)

all_results = {}

# OI availability summary
print("\n📊 OI DATA AVAILABILITY:")
print("-" * 50)
oi_avail = {}
for sym in SYMBOLS:
    oi = load_oi(sym)
    if oi is not None:
        oi_avail[sym] = f"{oi.index.min().date()} to {oi.index.max().date()} ({len(oi)} days)"
        print(f"  {sym}: {oi_avail[sym]}")
    else:
        oi_avail[sym] = "NOT AVAILABLE"
        print(f"  {sym}: NOT AVAILABLE")

strategies = {
    'S1_VolConfirmed_SMA50': ('price_only', s1_volume_confirmed_sma50),
    'S2_VolSurge_Breakout': ('price_only', s2_volume_surge_breakout),
    'S3_VolDryup_Breakout': ('price_only', s3_volume_dryup_breakout),
    'S4_Taker_Breakout': ('price_taker', s4_taker_confirmed_breakout),
    'S5_V4_VolConfirmed': ('price_only', s5_v4_volume_confirmed),
    'S6_Liq_OI_Continuation': ('price_liq_oi', s6_liq_oi_continuation),
    'S7_Liq_OI_Exhaustion': ('price_liq_oi', s7_liq_oi_exhaustion),
    'S8_OI_Buildup_Breakout': ('price_oi', s8_oi_buildup_breakout),
    'S9_OI_Divergence': ('price_oi', s9_oi_divergence),
    'S10_Combined_Score': ('all', s10_combined_score),
}

baselines = {
    'BL_SMA50': baseline_sma50,
    'BL_BuyHold': baseline_bnh,
}

for sym in SYMBOLS:
    print(f"\n{'='*80}")
    print(f"  {sym}")
    print(f"{'='*80}")
    
    price = load_price(sym)
    taker = load_taker(sym)
    liq = load_liq(sym)
    oi = load_oi(sym)
    
    print(f"  Price: {price.index.min().date()} to {price.index.max().date()} ({len(price)} days)")
    if taker is not None: print(f"  Taker: {len(taker)} days")
    if liq is not None: print(f"  Liq: {len(liq)} days")
    if oi is not None: print(f"  OI: {len(oi)} days")
    
    sym_results = {}
    
    # Baselines
    for bl_name, bl_func in baselines.items():
        signals = bl_func(price)
        strat_ret, cum = compute_returns(price, signals)
        metrics = calc_metrics(strat_ret, cum, price)
        sym_results[bl_name] = metrics
    
    # Strategies
    for strat_name, (data_type, strat_func) in strategies.items():
        if data_type == 'price_only':
            args = (price,)
        elif data_type == 'price_taker':
            args = (price, taker)
        elif data_type == 'price_liq_oi':
            if oi is None:
                sym_results[strat_name] = {'skip': True, 'reason': 'no OI data'}
                continue
            args = (price, liq, oi)
        elif data_type == 'price_oi':
            if oi is None:
                sym_results[strat_name] = {'skip': True, 'reason': 'no OI data'}
                continue
            args = (price, oi)
        elif data_type == 'all':
            args = (price, taker, liq, oi)
        
        try:
            metrics, oos_ret = expanding_walkforward(price, strat_func, args, n_folds=14)
            # Permutation test
            p_val = permutation_test(price, strat_func, args, n_perms=500)
            metrics['p_value'] = round(p_val, 4)
            metrics['significant_bonf'] = p_val < (0.05 / 10)  # Bonferroni for 10 strategies
            sym_results[strat_name] = metrics
        except Exception as e:
            sym_results[strat_name] = {'skip': True, 'reason': str(e)[:100]}
    
    all_results[sym] = sym_results
    
    # Print table
    print(f"\n  {'Strategy':<30} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'WinR%':>7} {'Signals':>8} {'p-val':>7} {'Sig?':>5}")
    print(f"  {'-'*85}")
    for name, m in sym_results.items():
        if m.get('skip'):
            print(f"  {name:<30} {'SKIPPED - ' + m.get('reason','')}")
            continue
        sig = '✅' if m.get('significant_bonf', False) else ('⚠️' if m.get('p_value', 1) < 0.05 else '❌')
        print(f"  {name:<30} {m['total_return']:>8.1f} {m['sharpe']:>7.3f} {m['max_dd']:>8.1f} {m['win_rate']:>7.1f} {m['n_signals']:>8} {m.get('p_value','n/a'):>7} {sig:>5}")

# Event study
print(f"\n{'='*80}")
print("📈 EVENT STUDY: Volume Surge (>2x avg) Forward Returns")
print(f"{'='*80}")
for sym in SYMBOLS:
    price = load_price(sym)
    es = event_study_volume_surge(price)
    print(f"\n  {sym}:")
    print(f"  {'Horizon':<10} {'Mean%':>8} {'Median%':>8} {'WinRate%':>9} {'N':>5}")
    for h, v in es.items():
        print(f"  {h:<10} {v['mean']:>8.2f} {v['median']:>8.2f} {v['win_rate']:>9.1f} {v['n_events']:>5}")

# Save results
all_results['metadata'] = {
    'run_date': datetime.now().isoformat(),
    'symbols': SYMBOLS,
    'oi_availability': oi_avail,
    'methodology': {
        'signal_bar': 'N, trade bar N+1',
        'walk_forward': '14-fold expanding',
        'permutation_tests': 500,
        'bonferroni_correction': '0.05/10 = 0.005',
    }
}

with open(os.path.join(RESULTS_DIR, 'volume_oi_test.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n✅ Results saved to {RESULTS_DIR}/volume_oi_test.json")
con.close()
