#!/usr/bin/env python3
"""
Strategy #5 (Indicator Combos) + #6 (Price Action Patterns) Validation
=======================================================================

#5 Indicator Combos (not just individual indicators — COMBINED):
- RSI + Bollinger Bands (buy RSI<30 at lower band)
- EMA 3/8/13 ribbon + volume filter
- MACD + RSI dual confirmation
- EMA crossover + VWAP deviation
- Stoch + Bollinger combo
- RSI divergence (price makes lower low, RSI makes higher low)

#6 Price Action Patterns:
- Bull/Bear Flag (sharp move + tight consolidation + breakout)
- Ascending/Descending Triangle (flat top/bottom + converging trendline)
- Channel Trading (parallel trendlines — buy bottom, sell top)
- Inside Bar Breakout (bar contained within prior bar)
- Engulfing Candle (bullish/bearish engulfing pattern)
- Pin Bar / Hammer (long wick rejection at S/R)

Tested at 15m/1h/4h with granular holding periods.
"""

import pandas as pd
import numpy as np
from pathlib import Path
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

COST_BPS = 10
MIN_TRADES = 30
N_FOLDS = 14

ASSETS = [
    'BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'AVAX', 'DOT',
    'LINK', 'UNI', 'ATOM', 'FTM', 'NEAR', 'OP', 'ARB', 'SUI',
    'DOGE', 'XRP', 'RENDER', 'FET', 'TIA', 'SEI', 'DYDX', 'INJ'
]

TF_CONFIG = {
    '15m': {
        'holdings': {'30m': 2, '1h': 4, '2h': 8, '4h': 16, '8h': 32, '12h': 48, '24h': 96},
        'bpy': 365.25 * 24 * 4,
    },
    '1h': {
        'holdings': {'2h': 2, '4h': 4, '8h': 8, '12h': 12, '1d': 24},
        'bpy': 365.25 * 24,
    },
    '4h': {
        'holdings': {'12h': 3, '1d': 6, '2d': 12, '3d': 18, '1w': 42},
        'bpy': 365.25 * 6,
    },
}

# ============================================================
# INDICATORS
# ============================================================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

def rsi(close, p=14):
    d = close.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    return 100 - 100 / (1 + g.ewm(alpha=1/p, min_periods=p).mean() / l.ewm(alpha=1/p, min_periods=p).mean())

def bbands(close, p=20, mult=2.0):
    m = close.rolling(p).mean(); s = close.rolling(p).std()
    return m + mult * s, m, m - mult * s

def macd(close, f=12, s=26, sig=9):
    ml = ema(close, f) - ema(close, s); return ml, ema(ml, sig), ml - ema(ml, sig)

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

def rolling_vwap(high, low, close, vol, p=48):
    tp = (high + low + close) / 3
    return (tp * vol).rolling(p).sum() / (vol.rolling(p).sum() + 1e-10)

def stochastic(high, low, close, p=14):
    lo = low.rolling(p).min(); hi = high.rolling(p).max()
    raw = (close - lo) / (hi - lo + 1e-10) * 100
    return raw.rolling(3).mean(), raw.rolling(3).mean().rolling(3).mean()

# ============================================================
# STRATEGY #5: INDICATOR COMBOS
# ============================================================

def combo_rsi_bollinger(df, rsi_p=14, bb_p=20, bb_std=2.0):
    """RSI oversold/overbought + Bollinger Band touch. The classic combo."""
    r = rsi(df['close'], rsi_p)
    upper, mid, lower = bbands(df['close'], bb_p, bb_std)
    s = pd.Series(0, index=df.index)
    # Long: RSI < 30 AND close at/below lower band AND recovering
    s[(r < 30) & (df['close'] <= lower) & (df['close'] > df['close'].shift(1))] = 1
    # Short: RSI > 70 AND close at/above upper band AND declining
    s[(r > 70) & (df['close'] >= upper) & (df['close'] < df['close'].shift(1))] = -1
    return s

def combo_ema_ribbon_vol(df, e1=3, e2=8, e3=13, vol_mult=1.5):
    """EMA 3/8/13 ribbon with volume confirmation."""
    ema1 = ema(df['close'], e1); ema2 = ema(df['close'], e2); ema3 = ema(df['close'], e3)
    avg_vol = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > (vol_mult * avg_vol)
    bull = (ema1 > ema2) & (ema2 > ema3)
    bear = (ema1 < ema2) & (ema2 < ema3)
    s = pd.Series(0, index=df.index)
    s[bull & ~bull.shift(1).fillna(False) & high_vol] = 1
    s[bear & ~bear.shift(1).fillna(False) & high_vol] = -1
    return s

def combo_macd_rsi(df, rsi_p=14, macd_f=12, macd_s=26, macd_sig=9):
    """MACD cross confirmed by RSI direction."""
    r = rsi(df['close'], rsi_p)
    ml, sl, hist = macd(df['close'], macd_f, macd_s, macd_sig)
    s = pd.Series(0, index=df.index)
    macd_cross_up = (ml > sl) & (ml.shift(1) <= sl.shift(1))
    macd_cross_down = (ml < sl) & (ml.shift(1) >= sl.shift(1))
    # Confirm with RSI: RSI rising for long, falling for short
    rsi_rising = r > r.shift(1)
    rsi_falling = r < r.shift(1)
    s[macd_cross_up & rsi_rising & (r < 70)] = 1
    s[macd_cross_down & rsi_falling & (r > 30)] = -1
    return s

def combo_ema_vwap(df, ema_f=8, ema_s=21, vwap_p=48):
    """EMA crossover only when price is on the right side of VWAP."""
    ef = ema(df['close'], ema_f); es = ema(df['close'], ema_s)
    vw = rolling_vwap(df['high'], df['low'], df['close'], df['volume'], vwap_p)
    s = pd.Series(0, index=df.index)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_down = (ef < es) & (ef.shift(1) >= es.shift(1))
    # Only long if price above VWAP, short if below
    s[cross_up & (df['close'] > vw)] = 1
    s[cross_down & (df['close'] < vw)] = -1
    return s

def combo_stoch_bollinger(df, stoch_p=14, bb_p=20):
    """Stochastic oversold + Bollinger lower band = double confirmation."""
    k, d = stochastic(df['high'], df['low'], df['close'], stoch_p)
    upper, mid, lower = bbands(df['close'], bb_p)
    s = pd.Series(0, index=df.index)
    # Long: stoch K < 20, crossing up, at lower band
    s[(k < 20) & (k > d) & (k.shift(1) <= d.shift(1)) & (df['close'] <= lower * 1.005)] = 1
    # Short: stoch K > 80, crossing down, at upper band
    s[(k > 80) & (k < d) & (k.shift(1) >= d.shift(1)) & (df['close'] >= upper * 0.995)] = -1
    return s

def combo_rsi_divergence(df, rsi_p=14, lookback=20):
    """RSI bullish/bearish divergence — price vs RSI disagreement."""
    r = rsi(df['close'], rsi_p)
    s = pd.Series(0, index=df.index)
    
    # Bullish divergence: price makes lower low but RSI makes higher low
    price_lower_low = df['close'] < df['close'].rolling(lookback).min().shift(1)
    rsi_higher_low = r > r.rolling(lookback).min().shift(1)
    recovering = df['close'] > df['close'].shift(1)
    s[price_lower_low & rsi_higher_low & recovering & (r < 40)] = 1
    
    # Bearish divergence: price makes higher high but RSI makes lower high
    price_higher_high = df['close'] > df['close'].rolling(lookback).max().shift(1)
    rsi_lower_high = r < r.rolling(lookback).max().shift(1)
    declining = df['close'] < df['close'].shift(1)
    s[price_higher_high & rsi_lower_high & declining & (r > 60)] = -1
    return s


# ============================================================
# STRATEGY #6: PRICE ACTION PATTERNS
# ============================================================

def pattern_flag(df, impulse_bars=5, consol_bars=10, impulse_mult=2.0, consol_pct=0.3):
    """
    Bull/Bear Flag: sharp move (impulse) + tight consolidation + breakout.
    - Impulse: move > impulse_mult * ATR in impulse_bars
    - Consolidation: range < consol_pct * impulse in next consol_bars
    - Entry: breakout from consolidation in impulse direction
    """
    a = atr(df['high'], df['low'], df['close'])
    s = pd.Series(0, index=df.index)
    
    for i in range(impulse_bars + consol_bars, len(df)):
        # Check for bull flag
        impulse_start = i - impulse_bars - consol_bars
        impulse_end = i - consol_bars
        impulse_move = df['close'].iloc[impulse_end] - df['close'].iloc[impulse_start]
        
        if abs(impulse_move) > impulse_mult * a.iloc[impulse_end]:
            # Check consolidation tightness
            consol_range = df['high'].iloc[impulse_end:i].max() - df['low'].iloc[impulse_end:i].min()
            if consol_range < consol_pct * abs(impulse_move):
                # Breakout
                consol_high = df['high'].iloc[impulse_end:i].max()
                consol_low = df['low'].iloc[impulse_end:i].min()
                if impulse_move > 0 and df['close'].iloc[i] > consol_high:
                    s.iloc[i] = 1
                elif impulse_move < 0 and df['close'].iloc[i] < consol_low:
                    s.iloc[i] = -1
    return s


def pattern_flag_vectorized(df, impulse_bars=5, consol_bars=10, impulse_mult=2.0):
    """Faster vectorized flag detection (approximate)."""
    a = atr(df['high'], df['low'], df['close'])
    
    # Impulse: big move over impulse_bars
    impulse = df['close'].diff(impulse_bars).shift(consol_bars)
    
    # Consolidation: tight range in recent consol_bars
    consol_range = df['high'].rolling(consol_bars).max() - df['low'].rolling(consol_bars).min()
    tight_consol = consol_range < 0.3 * impulse.abs()
    
    # Breakout from consolidation
    consol_high = df['high'].rolling(consol_bars).max().shift(1)
    consol_low = df['low'].rolling(consol_bars).min().shift(1)
    
    big_impulse = impulse.abs() > impulse_mult * a
    
    s = pd.Series(0, index=df.index)
    # Bull flag: up impulse + tight consol + break above
    s[(impulse > 0) & big_impulse & tight_consol & (df['close'] > consol_high)] = 1
    # Bear flag: down impulse + tight consol + break below
    s[(impulse < 0) & big_impulse & tight_consol & (df['close'] < consol_low)] = -1
    return s


def pattern_triangle(df, lookback=20, min_touches=3):
    """
    Ascending/Descending Triangle (vectorized approximation).
    - Ascending: flat resistance + rising support → bullish breakout
    - Descending: flat support + falling resistance → bearish breakout
    """
    # Use rolling max/min as approximate flat levels
    hi = df['high'].rolling(lookback).max()
    lo = df['low'].rolling(lookback).min()
    
    # Check if highs are relatively flat (range of highs < 30% of total range)
    rolling_high_range = df['high'].rolling(lookback).max() - df['high'].rolling(lookback).min()
    rolling_low_range = df['low'].rolling(lookback).max() - df['low'].rolling(lookback).min()
    total_range = hi - lo + 1e-10
    
    flat_top = rolling_high_range / total_range < 0.3  # resistance is flat
    flat_bottom = rolling_low_range / total_range < 0.3  # support is flat
    
    # Rising lows (ascending) or falling highs (descending)
    rising_lows = df['low'].rolling(lookback // 2).min() > df['low'].rolling(lookback).min().shift(lookback // 2)
    falling_highs = df['high'].rolling(lookback // 2).max() < df['high'].rolling(lookback).max().shift(lookback // 2)
    
    # Narrowing range = triangle
    narrowing = total_range < total_range.shift(lookback // 2)
    
    s = pd.Series(0, index=df.index)
    # Ascending triangle breakout: flat top + rising bottom → break above
    s[flat_top & rising_lows & narrowing & (df['close'] > hi.shift(1))] = 1
    # Descending triangle breakdown: flat bottom + falling top → break below
    s[flat_bottom & falling_highs & narrowing & (df['close'] < lo.shift(1))] = -1
    return s


def pattern_channel(df, lookback=30, atr_mult=0.3):
    """
    Channel Trading: buy at bottom of channel, sell at top.
    Channel defined by linear regression of highs and lows.
    Simplified: use rolling quantiles as channel bounds.
    """
    upper = df['high'].rolling(lookback).quantile(0.95)
    lower = df['low'].rolling(lookback).quantile(0.05)
    mid = (upper + lower) / 2
    a = atr(df['high'], df['low'], df['close'])
    
    # Only trade when channel is established (consistent width)
    width = upper - lower
    avg_width = width.rolling(lookback).mean()
    stable = (width / (avg_width + 1e-10)).between(0.5, 1.5)
    
    # ADX filter — channels exist in ranging markets
    ax = adx(df['high'], df['low'], df['close'])
    ranging = ax < 30
    
    s = pd.Series(0, index=df.index)
    near_lower = (df['close'] - lower) < (atr_mult * a)
    near_upper = (upper - df['close']) < (atr_mult * a)
    recovering = df['close'] > df['close'].shift(1)
    declining = df['close'] < df['close'].shift(1)
    
    s[stable & ranging & near_lower & recovering] = 1
    s[stable & ranging & near_upper & declining] = -1
    return s


def pattern_inside_bar(df, vol_confirm=True):
    """
    Inside Bar Breakout: bar completely inside prior bar.
    Enter on breakout of the inside bar range.
    """
    inside = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    
    s = pd.Series(0, index=df.index)
    # Previous bar was inside bar, current bar breaks out
    prev_inside = inside.shift(1).fillna(False)
    mother_high = df['high'].shift(2)  # the bar before the inside bar
    mother_low = df['low'].shift(2)
    
    breakout_up = prev_inside & (df['close'] > df['high'].shift(1))
    breakout_down = prev_inside & (df['close'] < df['low'].shift(1))
    
    if vol_confirm:
        avg_vol = df['volume'].rolling(20).mean()
        high_vol = df['volume'] > avg_vol
        breakout_up = breakout_up & high_vol
        breakout_down = breakout_down & high_vol
    
    s[breakout_up] = 1
    s[breakout_down] = -1
    return s


def pattern_engulfing(df, min_body_ratio=1.5):
    """
    Bullish/Bearish Engulfing: current candle engulfs prior candle body.
    """
    body_curr = (df['close'] - df['open']).abs()
    body_prev = (df['close'].shift(1) - df['open'].shift(1)).abs()
    
    bull_engulf = (df['close'] > df['open']) & (df['open'].shift(1) > df['close'].shift(1)) & \
                  (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & \
                  (body_curr > min_body_ratio * body_prev)
    
    bear_engulf = (df['close'] < df['open']) & (df['open'].shift(1) < df['close'].shift(1)) & \
                  (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & \
                  (body_curr > min_body_ratio * body_prev)
    
    s = pd.Series(0, index=df.index)
    s[bull_engulf] = 1
    s[bear_engulf] = -1
    return s


def pattern_pinbar(df, wick_ratio=2.0):
    """
    Pin Bar / Hammer: long wick rejection at support/resistance.
    Bullish pin: long lower wick (>2x body), small upper wick
    Bearish pin: long upper wick (>2x body), small lower wick
    """
    body = (df['close'] - df['open']).abs()
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']
    
    # Bullish pin: long lower wick, small upper wick, body at top
    bull_pin = (lower_wick > wick_ratio * body) & (upper_wick < body) & (body > 0)
    # Bearish pin: long upper wick, small lower wick, body at bottom
    bear_pin = (upper_wick > wick_ratio * body) & (lower_wick < body) & (body > 0)
    
    s = pd.Series(0, index=df.index)
    s[bull_pin] = 1
    s[bear_pin] = -1
    return s


# ============================================================
# ALL STRATEGIES
# ============================================================

STRATEGIES = {
    # #5 Indicator Combos
    'rsi_bb_combo':        ('combo', lambda df: combo_rsi_bollinger(df)),
    'rsi_bb_tight':        ('combo', lambda df: combo_rsi_bollinger(df, rsi_p=10, bb_p=16, bb_std=1.8)),
    'ema_3_8_13_vol':      ('combo', lambda df: combo_ema_ribbon_vol(df, 3, 8, 13)),
    'ema_5_13_34_vol':     ('combo', lambda df: combo_ema_ribbon_vol(df, 5, 13, 34)),
    'macd_rsi_combo':      ('combo', lambda df: combo_macd_rsi(df)),
    'macd_rsi_fast':       ('combo', lambda df: combo_macd_rsi(df, rsi_p=10, macd_f=8, macd_s=21, macd_sig=5)),
    'ema_vwap_combo':      ('combo', lambda df: combo_ema_vwap(df)),
    'stoch_bb_combo':      ('combo', lambda df: combo_stoch_bollinger(df)),
    'rsi_divergence':      ('combo', lambda df: combo_rsi_divergence(df)),
    # #6 Price Action
    'flag_pattern':        ('priceaction', lambda df: pattern_flag_vectorized(df)),
    'flag_tight':          ('priceaction', lambda df: pattern_flag_vectorized(df, impulse_bars=3, consol_bars=8, impulse_mult=1.5)),
    'triangle':            ('priceaction', lambda df: pattern_triangle(df)),
    'triangle_short':      ('priceaction', lambda df: pattern_triangle(df, lookback=12)),
    'channel':             ('priceaction', lambda df: pattern_channel(df)),
    'channel_short':       ('priceaction', lambda df: pattern_channel(df, lookback=16)),
    'inside_bar':          ('priceaction', lambda df: pattern_inside_bar(df)),
    'inside_bar_novol':    ('priceaction', lambda df: pattern_inside_bar(df, vol_confirm=False)),
    'engulfing':           ('priceaction', lambda df: pattern_engulfing(df)),
    'engulfing_strong':    ('priceaction', lambda df: pattern_engulfing(df, min_body_ratio=2.0)),
    'pinbar':              ('priceaction', lambda df: pattern_pinbar(df)),
    'pinbar_strict':       ('priceaction', lambda df: pattern_pinbar(df, wick_ratio=3.0)),
}


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


def main():
    t0 = time.time()
    est = len(ASSETS) * len(STRATEGIES) * sum(len(c['holdings']) for c in TF_CONFIG.values())
    bonf_alpha = 0.05 / max(est, 1)

    print("=" * 100)
    print("STRATEGY #5 (INDICATOR COMBOS) + #6 (PRICE ACTION PATTERNS)")
    print("=" * 100)
    print(f"Strategies: {len(STRATEGIES)} | Assets: {len(ASSETS)} | TFs: {list(TF_CONFIG.keys())}")
    print(f"Est tests: ~{est} | Bonferroni α: {bonf_alpha:.2e}")
    
    print(f"\n--- #5 Indicator Combos ---")
    for sn, (st, _) in STRATEGIES.items():
        if st == 'combo': print(f"  {sn}")
    print(f"\n--- #6 Price Action ---")
    for sn, (st, _) in STRATEGIES.items():
        if st == 'priceaction': print(f"  {sn}")
    print()

    all_results = []
    for tf, cfg in TF_CONFIG.items():
        bpy = cfg['bpy']
        print(f"\n{'='*80}\nTIMEFRAME: {tf}\n{'='*80}", flush=True)
        for ai, asset in enumerate(ASSETS):
            df = load_spot(asset, tf)
            if df is None: continue
            print(f"  [{ai+1}/{len(ASSETS)}] {asset} ({len(df)} bars)...", end='', flush=True)
            hits = 0
            for sname, (stype, sfn) in STRATEGIES.items():
                try: signals = sfn(df)
                except: continue
                for hname, hbars in cfg['holdings'].items():
                    m = walk_forward_test(df, signals, hbars, bpy)
                    if m is None: continue
                    grade = grade_strategy(m)
                    m.update({'asset': asset, 'timeframe': tf, 'strategy': sname, 'type': stype,
                              'holding': hname, 'grade': grade,
                              'pass_raw': m['p_value'] < 0.05 and m['sharpe'] > 0,
                              'pass_bonf': m['p_value'] < bonf_alpha and m['sharpe'] > 0})
                    all_results.append(m)
                    if m['sharpe'] > 1.0: hits += 1
            print(f" {hits} hits", flush=True)
            del df; gc.collect()
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'scalping_combos_priceaction.csv', index=False)
        print(f"  [saved {len(all_results)} results]", flush=True)

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*100}")
    print(f"RESULTS: {len(all_results)} valid tests in {elapsed:.1f}s")
    print(f"{'='*100}")
    if not all_results: return

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

    # #5 vs #6
    print(f"\n--- #5 COMBOS vs #6 PRICE ACTION ---")
    for stype in ['combo', 'priceaction']:
        sub = [r for r in all_results if r['type'] == stype]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        label = '#5 Indicator Combos' if stype == 'combo' else '#6 Price Action'
        print(f"  {label:<25} N={n:>5} %Pos={p/n*100:>5.1f}% AvgSharpe={np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"AvgSortino={np.mean([r['sortino'] for r in sub]):>7.2f}")

    # By TF
    print(f"\n--- BY TIMEFRAME ---")
    for tf in TF_CONFIG:
        sub = [r for r in all_results if r['timeframe'] == tf]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        print(f"  {tf:<4} N={n:>5} %Pos={p/n*100:>5.1f}% AvgSharpe={np.mean([r['sharpe'] for r in sub]):>7.2f}")

    # By strategy
    print(f"\n--- BY STRATEGY ---")
    print(f"{'Strategy':<22} {'Type':<6} {'N':>4} {'%Pos':>6} {'AvgSh':>7} {'AvgSort':>8} {'BestSh':>7} {'Best@':<12}")
    for sn in sorted(STRATEGIES.keys()):
        sub = [r for r in all_results if r['strategy'] == sn]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        best = max(sub, key=lambda x: x['sharpe'])
        print(f"  {sn:<22} {sub[0]['type']:<6} {n:>4} {p/n*100:>5.1f}% {np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"{np.mean([r['sortino'] for r in sub]):>8.2f} {best['sharpe']:>7.2f} {best['asset']+'_'+best['timeframe']+'_'+best['holding']:<12}")

    # By holding
    print(f"\n--- BY HOLDING ---")
    for tf in TF_CONFIG:
        for hname in TF_CONFIG[tf]['holdings']:
            sub = [r for r in all_results if r['timeframe'] == tf and r['holding'] == hname]
            if not sub: continue
            n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
            print(f"  {tf} {hname:<5} N={n:>4} %Pos={p/n*100:>5.1f}% AvgSh={np.mean([r['sharpe'] for r in sub]):>7.2f}")

    # Top 25
    print(f"\n--- TOP 25 BY SHARPE ---")
    top = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:25]
    print(f"{'Asset':<7} {'TF':<4} {'Strategy':<22} {'Hold':<5} {'Gr':>2} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MDD':>8} {'PF':>6} {'WR':>5} {'N':>5} {'p':>10}")
    for r in top:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<22} {r['holding']:<5} {r['grade']:>2} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['max_drawdown']:>7.1%} "
              f"{r['profit_factor']:>6.2f} {r['win_rate']:>4.0%} {r['n_trades']:>5} {r['p_value']:>10.6f}")

    # Top composite
    print(f"\n--- TOP 15 COMPOSITE ---")
    for r in all_results:
        r['composite'] = r['sharpe'] + r['sortino']/3 + r['calmar']/2 - abs(r['max_drawdown'])
    top_c = sorted(all_results, key=lambda x: x['composite'], reverse=True)[:15]
    for r in top_c:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<22} hold={r['holding']:<5} [{r['grade']}] "
              f"Comp={r['composite']:>6.2f} Sh={r['sharpe']:>5.2f} So={r['sortino']:>5.2f} "
              f"Ca={r['calmar']:>5.2f} MDD={r['max_drawdown']:>7.1%} n={r['n_trades']:>4}")

    print(f"\nResults: {RESULTS_DIR / 'scalping_combos_priceaction.csv'}")

if __name__ == '__main__':
    main()
