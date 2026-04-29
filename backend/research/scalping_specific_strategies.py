#!/usr/bin/env python3
"""
Specific Scalping Strategy Validation
======================================
Tests the exact strategies described by the user:

1. Range Trading — detect consolidation zones, buy support, sell resistance
2. Breakout Trading — break S/R with volume confirmation + pullback retest variant
   (enhanced beyond our previous simple range_breakout)

Also tests variants and combinations:
- Range with different lookback periods for S/R detection
- Breakout with and without volume filter
- Breakout with pullback retest (wait for price to return to broken level)
- Combined: trade range until breakout, then switch to breakout mode

Tested across 5m/15m/1h with granular holding periods.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time
import sys
import gc

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)
from metrics import compute_metrics, format_metrics_summary, grade_strategy

warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/spot")
RESULTS_DIR = Path("/Users/ffv_macmini/Desktop/maestro/backend/research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 7  # realistic: limit open + market close
MIN_TRADES = 30
N_FOLDS = 14

ASSETS = [
    'BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'AVAX', 'DOT',
    'LINK', 'UNI', 'ATOM', 'FTM', 'NEAR', 'OP', 'ARB', 'SUI',
    'DOGE', 'XRP', 'RENDER', 'FET', 'TIA', 'SEI', 'DYDX', 'INJ'
]

TF_CONFIG = {
    '5m': {
        'holdings': {
            '10m': 2, '15m': 3, '30m': 6, '1h': 12, '2h': 24, '4h': 48, '8h': 96, '12h': 144,
        },
        'bpy': 365.25 * 24 * 12,
    },
    '15m': {
        'holdings': {
            '30m': 2, '1h': 4, '2h': 8, '4h': 16, '8h': 32, '12h': 48, '24h': 96,
        },
        'bpy': 365.25 * 24 * 4,
    },
    '1h': {
        'holdings': {
            '2h': 2, '4h': 4, '8h': 8, '12h': 12, '1d': 24,
        },
        'bpy': 365.25 * 24,
    },
}

# ============================================================
# INDICATORS
# ============================================================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def atr(high, low, close, p=14):
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def adx(high, low, close, p=14):
    pdm = high.diff(); mdm = -low.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100 * pdm.ewm(alpha=1/p, min_periods=p).mean() / a
    mdi = 100 * mdm.ewm(alpha=1/p, min_periods=p).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.ewm(alpha=1/p, min_periods=p).mean()

# ============================================================
# STRATEGY 1: RANGE TRADING (multiple variants)
# ============================================================

def range_trading_v1(df, lookback=48, atr_mult=0.5, adx_thresh=25):
    """
    Classic range trading: identify consolidation, buy at support, sell at resistance.
    
    1. Detect range: ADX < threshold (not trending)
    2. Support = rolling low, Resistance = rolling high
    3. Buy when price touches support zone (within atr_mult * ATR)
    4. Sell when price touches resistance zone
    """
    n = len(df)
    support = df['low'].rolling(lookback).min()
    resistance = df['high'].rolling(lookback).max()
    a = atr(df['high'], df['low'], df['close'])
    ax = adx(df['high'], df['low'], df['close'])
    
    # Range width relative to ATR — only trade tight ranges
    range_width = (resistance - support) / (a + 1e-10)
    
    signals = pd.Series(0, index=df.index)
    
    # Only trade when: ADX low (ranging) AND range is reasonable (not too wide)
    ranging = (ax < adx_thresh) & (range_width < 10) & (range_width > 2)
    
    # Buy near support
    near_support = (df['close'] - support) < (atr_mult * a)
    # Price recovering (not still falling)
    recovering = df['close'] > df['close'].shift(1)
    signals[ranging & near_support & recovering] = 1
    
    # Sell near resistance
    near_resistance = (resistance - df['close']) < (atr_mult * a)
    declining = df['close'] < df['close'].shift(1)
    signals[ranging & near_resistance & declining] = -1
    
    return signals


def range_trading_v2(df, lookback=48, percentile_band=0.1):
    """
    Percentile-based range trading.
    Buy in bottom 10% of recent range, sell in top 10%.
    Only when range is tight (low volatility).
    """
    rolling_min = df['low'].rolling(lookback).min()
    rolling_max = df['high'].rolling(lookback).max()
    position_in_range = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    # Volatility filter: only trade when range is contracting
    a = atr(df['high'], df['low'], df['close'])
    avg_atr = a.rolling(lookback).mean()
    low_vol = a < avg_atr  # current ATR below average
    
    signals = pd.Series(0, index=df.index)
    signals[(position_in_range < percentile_band) & low_vol & (df['close'] > df['close'].shift(1))] = 1
    signals[(position_in_range > (1 - percentile_band)) & low_vol & (df['close'] < df['close'].shift(1))] = -1
    return signals


def range_trading_v3(df, lookback=48, atr_mult=0.3):
    """
    Pivot-based range trading.
    Uses swing highs/lows as support/resistance.
    """
    # Detect swing highs and lows (local extremes over 5 bars)
    swing_period = 5
    is_swing_high = (df['high'] == df['high'].rolling(2*swing_period+1, center=True).max())
    is_swing_low = (df['low'] == df['low'].rolling(2*swing_period+1, center=True).min())
    
    # Use most recent swing high as resistance, swing low as support
    resistance = df['high'].where(is_swing_high).ffill()
    support = df['low'].where(is_swing_low).ffill()
    a = atr(df['high'], df['low'], df['close'])
    
    # ADX filter for ranging
    ax = adx(df['high'], df['low'], df['close'])
    ranging = ax < 25
    
    signals = pd.Series(0, index=df.index)
    near_support = (df['close'] - support).abs() < (atr_mult * a)
    near_resistance = (resistance - df['close']).abs() < (atr_mult * a)
    recovering = df['close'] > df['close'].shift(1)
    declining = df['close'] < df['close'].shift(1)
    
    signals[ranging & near_support & recovering] = 1
    signals[ranging & near_resistance & declining] = -1
    return signals


def range_trading_tight(df, lookback=24, max_range_pct=0.03):
    """
    Only trades in very tight ranges (< 3% width).
    Higher frequency but tighter stops.
    """
    rolling_min = df['low'].rolling(lookback).min()
    rolling_max = df['high'].rolling(lookback).max()
    range_pct = (rolling_max - rolling_min) / (rolling_min + 1e-10)
    
    position = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    tight = range_pct < max_range_pct
    
    signals = pd.Series(0, index=df.index)
    signals[tight & (position < 0.15) & (df['close'] > df['close'].shift(1))] = 1
    signals[tight & (position > 0.85) & (df['close'] < df['close'].shift(1))] = -1
    return signals


# ============================================================
# STRATEGY 2: BREAKOUT TRADING (multiple variants)
# ============================================================

def breakout_volume_confirmed(df, lookback=24, vol_mult=1.5):
    """
    Breakout with volume confirmation.
    Only enter when volume is > vol_mult * average.
    """
    hi = df['high'].rolling(lookback).max().shift(1)
    lo = df['low'].rolling(lookback).min().shift(1)
    avg_vol = df['volume'].rolling(lookback).mean()
    high_vol = df['volume'] > (vol_mult * avg_vol)
    
    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > hi) & high_vol] = 1
    signals[(df['close'] < lo) & high_vol] = -1
    return signals


def breakout_pullback_retest(df, lookback=24, retest_bars=6, atr_mult=0.5):
    """
    Breakout with pullback retest.
    1. Price breaks above resistance
    2. Wait for pullback to broken level (now support)
    3. Enter when price bounces from the retest
    """
    hi = df['high'].rolling(lookback).max().shift(1)
    lo = df['low'].rolling(lookback).min().shift(1)
    a = atr(df['high'], df['low'], df['close'])
    
    signals = pd.Series(0, index=df.index)
    
    # Track breakout events
    broke_up = df['close'] > hi
    broke_down = df['close'] < lo
    
    # For each bar, check if there was a breakout in the last retest_bars
    # and price has pulled back near the broken level
    for i in range(retest_bars, len(df)):
        # Check for upside retest
        if any(broke_up.iloc[max(0,i-retest_bars):i]):
            broken_level = hi.iloc[i-1]  # the level that was broken
            near_level = abs(df['close'].iloc[i] - broken_level) < (atr_mult * a.iloc[i])
            bouncing = df['close'].iloc[i] > df['close'].iloc[i-1]
            if near_level and bouncing and not broke_up.iloc[i]:
                signals.iloc[i] = 1
        
        # Check for downside retest
        if any(broke_down.iloc[max(0,i-retest_bars):i]):
            broken_level = lo.iloc[i-1]
            near_level = abs(df['close'].iloc[i] - broken_level) < (atr_mult * a.iloc[i])
            declining = df['close'].iloc[i] < df['close'].iloc[i-1]
            if near_level and declining and not broke_down.iloc[i]:
                signals.iloc[i] = -1
    
    return signals


def breakout_pullback_vectorized(df, lookback=24, retest_bars=6, atr_mult=0.5):
    """
    Faster vectorized version of pullback retest.
    Uses rolling window to detect recent breakouts + current retest.
    """
    hi = df['high'].rolling(lookback).max().shift(1)
    lo = df['low'].rolling(lookback).min().shift(1)
    a = atr(df['high'], df['low'], df['close'])
    
    broke_up = (df['close'] > hi).astype(float)
    broke_down = (df['close'] < lo).astype(float)
    
    # Was there a breakout in the last N bars?
    recent_break_up = broke_up.rolling(retest_bars).sum().shift(1) > 0
    recent_break_down = broke_down.rolling(retest_bars).sum().shift(1) > 0
    
    # Currently near the broken level (not currently breaking out)
    near_hi = (df['close'] - hi).abs() < (atr_mult * a)
    near_lo = (df['close'] - lo).abs() < (atr_mult * a)
    
    bouncing = df['close'] > df['close'].shift(1)
    declining = df['close'] < df['close'].shift(1)
    
    # Not currently in breakout (retest, not continuation)
    not_breaking = ~(df['close'] > hi) & ~(df['close'] < lo)
    
    signals = pd.Series(0, index=df.index)
    signals[recent_break_up & near_hi & bouncing & not_breaking] = 1
    signals[recent_break_down & near_lo & declining & not_breaking] = -1
    return signals


def breakout_strong_candle(df, lookback=24, body_mult=1.5):
    """
    Breakout confirmed by a strong (large body) candle.
    Body must be > body_mult * average body size.
    """
    hi = df['high'].rolling(lookback).max().shift(1)
    lo = df['low'].rolling(lookback).min().shift(1)
    
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(lookback).mean()
    strong = body > (body_mult * avg_body)
    
    signals = pd.Series(0, index=df.index)
    # Bullish strong candle breaking resistance
    signals[(df['close'] > hi) & strong & (df['close'] > df['open'])] = 1
    # Bearish strong candle breaking support
    signals[(df['close'] < lo) & strong & (df['close'] < df['open'])] = -1
    return signals


def breakout_atr_filter(df, lookback=24, atr_expansion=1.2):
    """
    Breakout only when volatility is expanding (ATR increasing).
    Filters out low-conviction breakouts in dying volatility.
    """
    hi = df['high'].rolling(lookback).max().shift(1)
    lo = df['low'].rolling(lookback).min().shift(1)
    a = atr(df['high'], df['low'], df['close'])
    avg_atr = a.rolling(lookback).mean()
    expanding = a > (atr_expansion * avg_atr)
    
    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > hi) & expanding] = 1
    signals[(df['close'] < lo) & expanding] = -1
    return signals


def range_then_breakout(df, lookback=48, adx_break_thresh=30):
    """
    Combined: range trade when ADX low, switch to breakout when ADX spikes.
    """
    hi = df['high'].rolling(lookback).max().shift(1)
    lo = df['low'].rolling(lookback).min().shift(1)
    ax = adx(df['high'], df['low'], df['close'])
    a = atr(df['high'], df['low'], df['close'])
    
    position = (df['close'] - lo) / (hi - lo + 1e-10)
    
    signals = pd.Series(0, index=df.index)
    
    # Range mode: ADX < 20
    ranging = ax < 20
    signals[ranging & (position < 0.15) & (df['close'] > df['close'].shift(1))] = 1
    signals[ranging & (position > 0.85) & (df['close'] < df['close'].shift(1))] = -1
    
    # Breakout mode: ADX > threshold and breaking out
    trending = ax > adx_break_thresh
    signals[trending & (df['close'] > hi)] = 1
    signals[trending & (df['close'] < lo)] = -1
    
    return signals


# ============================================================
# ALL STRATEGIES
# ============================================================

# TF-specific parameters
TF_PARAMS = {
    '5m':  {'range_lb': [24, 48, 96], 'brk_lb': [12, 24, 48], 'retest_bars': 6},
    '15m': {'range_lb': [16, 32, 64], 'brk_lb': [8, 16, 32],  'retest_bars': 4},
    '1h':  {'range_lb': [12, 24, 48], 'brk_lb': [6, 12, 24],  'retest_bars': 3},
}


def get_all_strategies(tf):
    """Generate all strategy variants for a given TF."""
    p = TF_PARAMS[tf]
    strategies = {}
    
    for lb in p['range_lb']:
        strategies[f'range_v1_lb{lb}'] = ('range', lambda df, lb=lb: range_trading_v1(df, lookback=lb))
        strategies[f'range_v2_lb{lb}'] = ('range', lambda df, lb=lb: range_trading_v2(df, lookback=lb))
        strategies[f'range_v3_lb{lb}'] = ('range', lambda df, lb=lb: range_trading_v3(df, lookback=lb))
        strategies[f'range_tight_lb{lb}'] = ('range', lambda df, lb=lb: range_trading_tight(df, lookback=lb))
    
    for lb in p['brk_lb']:
        strategies[f'brk_vol_lb{lb}'] = ('breakout', lambda df, lb=lb: breakout_volume_confirmed(df, lookback=lb))
        strategies[f'brk_pullback_lb{lb}'] = ('breakout', lambda df, lb=lb: breakout_pullback_vectorized(df, lookback=lb, retest_bars=p['retest_bars']))
        strategies[f'brk_strong_lb{lb}'] = ('breakout', lambda df, lb=lb: breakout_strong_candle(df, lookback=lb))
        strategies[f'brk_atr_lb{lb}'] = ('breakout', lambda df, lb=lb: breakout_atr_filter(df, lookback=lb))
    
    for lb in p['range_lb']:
        strategies[f'range_then_brk_lb{lb}'] = ('combined', lambda df, lb=lb: range_then_breakout(df, lookback=lb))
    
    return strategies


# ============================================================
# DATA + ENGINE
# ============================================================

def load_spot(asset, tf):
    path = DATA_DIR / tf / f"{asset}_spot_{tf}.csv"
    if not path.exists(): return None
    df = pd.read_csv(path); df.columns = [c.lower() for c in df.columns]
    renames = {c: 'timestamp' for c in df.columns if c in ('date', 'time', 'datetime')}
    df.rename(columns=renames, inplace=True)
    if 'timestamp' not in df.columns: return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df.sort_values('timestamp').reset_index(drop=True)


def walk_forward_test(df, signals, holding_bars, bpy):
    n = len(df)
    if n < 100: return None
    fold_size = n // N_FOLDS
    if fold_size < 20: return None
    test_start = fold_size * 7
    test_df = df.iloc[test_start:]; test_sig = signals.iloc[test_start:]
    trades = []
    i = 0
    while i < len(test_df) - holding_bars:
        sig = test_sig.iloc[i]
        if sig != 0:
            entry = test_df['close'].iloc[i]; exit_ = test_df['close'].iloc[i + holding_bars]
            ret = (exit_ / entry - 1) * sig - COST_BPS / 10000
            trades.append({'direction': sig, 'return': ret}); i += holding_bars
        else: i += 1
    if len(trades) < MIN_TRADES: return None
    returns = np.array([t['return'] for t in trades])
    test_bars = len(test_df)
    test_years = test_bars / bpy if bpy > 0 else 1
    tpy = len(trades) / test_years if test_years > 0 else len(trades)
    return compute_metrics(returns, bars_per_year=bpy, total_bars=test_bars, trades_per_year=tpy)


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    
    # Count total strategies across TFs
    total_strats = sum(len(get_all_strategies(tf)) for tf in TF_CONFIG)
    total_holds = sum(len(TF_CONFIG[tf]['holdings']) for tf in TF_CONFIG)
    est_tests = len(ASSETS) * total_strats * total_holds // len(TF_CONFIG)
    
    print("=" * 100)
    print("SPECIFIC SCALPING STRATEGIES: Range Trading + Breakout Variants")
    print("=" * 100)
    
    for tf in TF_CONFIG:
        strats = get_all_strategies(tf)
        print(f"\n  {tf}: {len(strats)} strategies × {len(TF_CONFIG[tf]['holdings'])} holdings × {len(ASSETS)} assets")
        for sn in sorted(strats.keys()):
            print(f"    - {sn} ({strats[sn][0]})")
    
    print(f"\nTotal est tests: ~{est_tests}")
    bonf_alpha = 0.05 / max(est_tests, 1)
    print(f"Bonferroni α: {bonf_alpha:.2e}")
    print(f"Costs: {COST_BPS}bps | Min trades: {MIN_TRADES}\n")

    all_results = []
    
    for tf, cfg in TF_CONFIG.items():
        bpy = cfg['bpy']
        strategies = get_all_strategies(tf)
        print(f"\n{'='*80}\nTIMEFRAME: {tf} — {len(strategies)} strategies\n{'='*80}", flush=True)
        
        for ai, asset in enumerate(ASSETS):
            df = load_spot(asset, tf)
            if df is None: continue
            print(f"  [{ai+1}/{len(ASSETS)}] {asset} ({len(df)} bars)...", end='', flush=True)
            hits = 0
            
            for sname, (stype, sfn) in strategies.items():
                try: signals = sfn(df)
                except Exception as e: continue
                
                for hname, hbars in cfg['holdings'].items():
                    m = walk_forward_test(df, signals, hbars, bpy)
                    if m is None: continue
                    grade = grade_strategy(m)
                    m.update({
                        'asset': asset, 'timeframe': tf, 'strategy': sname, 'type': stype,
                        'holding': hname, 'grade': grade,
                        'pass_raw': m['p_value'] < 0.05 and m['sharpe'] > 0,
                        'pass_bonf': m['p_value'] < bonf_alpha and m['sharpe'] > 0,
                    })
                    all_results.append(m)
                    if m['sharpe'] > 1.0: hits += 1
            
            print(f" {hits} hits", flush=True)
            del df; gc.collect()
        
        # Save incrementally after each TF
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'scalping_range_breakout.csv', index=False)
        print(f"  [saved {len(all_results)} results so far]", flush=True)

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)

    # ============================================================
    # REPORT
    # ============================================================
    print(f"\n{'='*100}")
    print(f"RESULTS: {len(all_results)} valid tests in {elapsed:.1f}s")
    print(f"{'='*100}")
    if not all_results:
        print("No valid results!"); return

    raw = sum(1 for r in all_results if r['pass_raw'])
    bonf = sum(1 for r in all_results if r['pass_bonf'])
    pos = sum(1 for r in all_results if r['sharpe'] > 0)
    print(f"Positive Sharpe: {pos} ({pos/len(all_results)*100:.1f}%)")
    print(f"Raw passes (p<0.05): {raw}")
    print(f"Bonferroni passes: {bonf}")
    
    print(f"\n--- GRADES ---")
    for g in 'ABCDF':
        c = sum(1 for r in all_results if r['grade'] == g)
        print(f"  {g}: {c} ({c/len(all_results)*100:.1f}%)")

    # By strategy type
    print(f"\n--- BY STRATEGY TYPE ---")
    for stype in ['range', 'breakout', 'combined']:
        sub = [r for r in all_results if r['type'] == stype]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        print(f"  {stype:<10} N={n:>5} %Pos={p/n*100:>5.1f}% AvgSharpe={np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"AvgSortino={np.mean([r['sortino'] for r in sub]):>7.2f} AvgMDD={np.mean([r['max_drawdown'] for r in sub]):>7.1%}")

    # By TF
    print(f"\n--- BY TIMEFRAME ---")
    for tf in TF_CONFIG:
        sub = [r for r in all_results if r['timeframe'] == tf]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        print(f"  {tf:<4} N={n:>5} %Pos={p/n*100:>5.1f}% AvgSharpe={np.mean([r['sharpe'] for r in sub]):>7.2f}")

    # By strategy (individual)
    print(f"\n--- BY STRATEGY ---")
    print(f"{'Strategy':<25} {'Type':<8} {'N':>4} {'%Pos':>6} {'AvgSh':>7} {'AvgSort':>8} {'BestSh':>7} {'BestAsset':<7}")
    all_strat_names = sorted(set(r['strategy'] for r in all_results))
    for sn in all_strat_names:
        sub = [r for r in all_results if r['strategy'] == sn]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        best = max(sub, key=lambda x: x['sharpe'])
        print(f"  {sn:<25} {sub[0]['type']:<8} {n:>4} {p/n*100:>5.1f}% {np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"{np.mean([r['sortino'] for r in sub]):>8.2f} {best['sharpe']:>7.2f} {best['asset']:<7}")

    # By holding period
    print(f"\n--- BY HOLDING PERIOD ---")
    for tf in TF_CONFIG:
        print(f"\n  {tf}:")
        for hname in TF_CONFIG[tf]['holdings']:
            sub = [r for r in all_results if r['timeframe'] == tf and r['holding'] == hname]
            if not sub: continue
            n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
            print(f"    {hname:<6} N={n:>4} %Pos={p/n*100:>5.1f}% AvgSh={np.mean([r['sharpe'] for r in sub]):>7.2f} "
                  f"AvgSort={np.mean([r['sortino'] for r in sub]):>7.2f} AvgMDD={np.mean([r['max_drawdown'] for r in sub]):>7.1%}")

    # Top 25 overall
    print(f"\n--- TOP 25 BY SHARPE ---")
    top = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:25]
    print(f"{'Asset':<7} {'TF':<4} {'Strategy':<25} {'Hold':<6} {'Gr':>2} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MDD':>8} {'PF':>6} {'WR':>5} {'N':>5} {'p':>10}")
    for r in top:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<25} {r['holding']:<6} {r['grade']:>2} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['max_drawdown']:>7.1%} "
              f"{r['profit_factor']:>6.2f} {r['win_rate']:>4.0%} {r['n_trades']:>5} {r['p_value']:>10.6f}")

    # Top composite
    print(f"\n--- TOP 15 COMPOSITE ---")
    for r in all_results:
        r['composite'] = r['sharpe'] + r['sortino']/3 + r['calmar']/2 - abs(r['max_drawdown'])
    top_c = sorted(all_results, key=lambda x: x['composite'], reverse=True)[:15]
    for r in top_c:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<25} hold={r['holding']:<5} [{r['grade']}] "
              f"Comp={r['composite']:>6.2f} Sh={r['sharpe']:>5.2f} So={r['sortino']:>5.2f} "
              f"Ca={r['calmar']:>5.2f} MDD={r['max_drawdown']:>7.1%} n={r['n_trades']:>4}")

    # Range vs Breakout head-to-head by TF and holding
    print(f"\n--- RANGE vs BREAKOUT HEAD-TO-HEAD ---")
    for tf in TF_CONFIG:
        print(f"\n  {tf}:")
        for hname in TF_CONFIG[tf]['holdings']:
            range_sub = [r for r in all_results if r['timeframe'] == tf and r['holding'] == hname and r['type'] == 'range']
            brk_sub = [r for r in all_results if r['timeframe'] == tf and r['holding'] == hname and r['type'] == 'breakout']
            if not range_sub or not brk_sub: continue
            r_sh = np.mean([r['sharpe'] for r in range_sub])
            b_sh = np.mean([r['sharpe'] for r in brk_sub])
            winner = "RANGE" if r_sh > b_sh else "BREAKOUT"
            print(f"    {hname:<6} Range={r_sh:>6.2f} vs Breakout={b_sh:>6.2f} → {winner}")

    print(f"\nResults saved to: {RESULTS_DIR / 'scalping_range_breakout.csv'}")

if __name__ == '__main__':
    main()
