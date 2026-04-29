#!/usr/bin/env python3
"""
All-Timeframe Scalping/Short-Term Strategy Validation
======================================================
Tests 12 scalping strategies across ALL available timeframes (5m/15m/1h/4h/1d)
with full multi-metric reporting (Sharpe, Sortino, Calmar, MDD, etc.)

Uses the centralized metrics.py module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics, format_metrics_summary, grade_strategy

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

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

# Timeframes → holding periods (in bars) + bars_per_year
TF_CONFIG = {
    '5m':  {'holdings': {'15m': 3, '1h': 12, '4h': 48},           'bpy': 365.25 * 24 * 12},
    '15m': {'holdings': {'1h': 4, '4h': 16, '12h': 48},           'bpy': 365.25 * 24 * 4},
    '1h':  {'holdings': {'4h': 4, '12h': 12, '1d': 24},           'bpy': 365.25 * 24},
    '4h':  {'holdings': {'1d': 6, '3d': 18, '1w': 42},            'bpy': 365.25 * 6},
    '1d':  {'holdings': {'3d': 3, '1w': 7, '2w': 14},             'bpy': 365.25},
}

# ============================================================
# TF-AWARE PARAMETERS
# ============================================================

def get_tf_params(tf):
    base = {
        '5m':  dict(rsi=7,  stoch=8,  bb=12, ema_f=5,  ema_m=13, ema_s=34, macd_f=8,  macd_s=21, macd_sig=5,  kelt=12, atr=10, adx=10, vwap=48,  brk=24, vol_lb=20, vol_th=2.0, rsi_os=20, rsi_ob=80, stoch_os=15, stoch_ob=85, bb_std=2.0, kelt_m=1.5, adx_th=20, mom_n=3,  mom_th=0.003),
        '15m': dict(rsi=10, stoch=10, bb=16, ema_f=5,  ema_m=13, ema_s=34, macd_f=8,  macd_s=21, macd_sig=5,  kelt=16, atr=12, adx=12, vwap=32,  brk=16, vol_lb=16, vol_th=2.0, rsi_os=25, rsi_ob=75, stoch_os=20, stoch_ob=80, bb_std=2.0, kelt_m=1.5, adx_th=22, mom_n=3,  mom_th=0.005),
        '1h':  dict(rsi=14, stoch=14, bb=20, ema_f=8,  ema_m=21, ema_s=55, macd_f=12, macd_s=26, macd_sig=9,  kelt=20, atr=14, adx=14, vwap=24,  brk=24, vol_lb=24, vol_th=2.0, rsi_os=30, rsi_ob=70, stoch_os=20, stoch_ob=80, bb_std=2.0, kelt_m=2.0, adx_th=25, mom_n=4,  mom_th=0.008),
        '4h':  dict(rsi=14, stoch=14, bb=20, ema_f=8,  ema_m=21, ema_s=55, macd_f=12, macd_s=26, macd_sig=9,  kelt=20, atr=14, adx=14, vwap=30,  brk=20, vol_lb=20, vol_th=2.0, rsi_os=30, rsi_ob=70, stoch_os=20, stoch_ob=80, bb_std=2.0, kelt_m=2.0, adx_th=25, mom_n=3,  mom_th=0.015),
        '1d':  dict(rsi=14, stoch=14, bb=20, ema_f=10, ema_m=21, ema_s=50, macd_f=12, macd_s=26, macd_sig=9,  kelt=20, atr=14, adx=14, vwap=20,  brk=20, vol_lb=20, vol_th=2.0, rsi_os=30, rsi_ob=70, stoch_os=20, stoch_ob=80, bb_std=2.0, kelt_m=2.0, adx_th=25, mom_n=3,  mom_th=0.025),
    }
    return base[tf]

# ============================================================
# INDICATORS
# ============================================================

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

def rsi(close, p=14):
    d = close.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al)

def stoch_rsi(close, rsi_p, stoch_p, k_sm=3, d_sm=3):
    r = rsi(close, rsi_p)
    lo = r.rolling(stoch_p).min()
    hi = r.rolling(stoch_p).max()
    sr = (r - lo) / (hi - lo + 1e-10)
    k = sr.rolling(k_sm).mean()
    d = k.rolling(d_sm).mean()
    return k, d

def stochastic(high, low, close, p=14, k_sm=3, d_sm=3):
    lo = low.rolling(p).min()
    hi = high.rolling(p).max()
    raw = (close - lo) / (hi - lo + 1e-10) * 100
    k = raw.rolling(k_sm).mean()
    d = k.rolling(d_sm).mean()
    return k, d

def bbands(close, p=20, mult=2.0):
    m = close.rolling(p).mean()
    s = close.rolling(p).std()
    return m + mult * s, m, m - mult * s

def keltner(high, low, close, ep=20, atr_p=14, mult=2.0):
    e = ema(close, ep)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.rolling(atr_p).mean()
    return e + mult * a, e, e - mult * a

def macd(close, f=12, s=26, sig=9):
    ml = ema(close, f) - ema(close, s)
    sl = ema(ml, sig)
    return ml, sl, ml - sl

def atr(high, low, close, p=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def adx(high, low, close, p=14):
    pdm = high.diff(); mdm = -low.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100 * pdm.ewm(alpha=1/p, min_periods=p).mean() / a
    mdi = 100 * mdm.ewm(alpha=1/p, min_periods=p).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.ewm(alpha=1/p, min_periods=p).mean()

def rolling_vwap(high, low, close, vol, p=48):
    tp = (high + low + close) / 3
    return (tp * vol).rolling(p).sum() / (vol.rolling(p).sum() + 1e-10)

# ============================================================
# 12 STRATEGIES
# ============================================================

def sig_vwap_mr(df, p):
    vw = rolling_vwap(df['high'], df['low'], df['close'], df['volume'], p['vwap'])
    a = atr(df['high'], df['low'], df['close'], p['atr'])
    dev = (df['close'] - vw) / (a + 1e-10)
    s = pd.Series(0, index=df.index)
    s[dev < -1.5] = 1; s[dev > 1.5] = -1
    return s

def sig_stoch_rsi(df, p):
    k, d = stoch_rsi(df['close'], p['rsi'], p['stoch'])
    s = pd.Series(0, index=df.index)
    s[(k > d) & (k.shift(1) <= d.shift(1)) & (k < 0.3)] = 1
    s[(k < d) & (k.shift(1) >= d.shift(1)) & (k > 0.7)] = -1
    return s

def sig_ema_ribbon(df, p):
    ef = ema(df['close'], p['ema_f']); em = ema(df['close'], p['ema_m']); es = ema(df['close'], p['ema_s'])
    bull = (ef > em) & (em > es); bear = (ef < em) & (em < es)
    s = pd.Series(0, index=df.index)
    s[bull & ~bull.shift(1).fillna(False)] = 1
    s[bear & ~bear.shift(1).fillna(False)] = -1
    return s

def sig_bollinger_mr(df, p):
    u, m, l = bbands(df['close'], p['bb'], p['bb_std'])
    s = pd.Series(0, index=df.index)
    s[(df['close'] <= l) & (df['close'] > df['close'].shift(1))] = 1
    s[(df['close'] >= u) & (df['close'] < df['close'].shift(1))] = -1
    return s

def sig_rsi_extreme(df, p):
    r = rsi(df['close'], p['rsi'])
    s = pd.Series(0, index=df.index)
    s[(r > p['rsi_os']) & (r.shift(1) <= p['rsi_os'])] = 1
    s[(r < p['rsi_ob']) & (r.shift(1) >= p['rsi_ob'])] = -1
    return s

def sig_momentum_burst(df, p):
    pc = df['close'].pct_change(p['mom_n'])
    s = pd.Series(0, index=df.index)
    s[pc > p['mom_th']] = 1; s[pc < -p['mom_th']] = -1
    return s

def sig_range_breakout(df, p):
    hi = df['high'].rolling(p['brk']).max().shift(1)
    lo = df['low'].rolling(p['brk']).min().shift(1)
    s = pd.Series(0, index=df.index)
    s[df['close'] > hi] = 1; s[df['close'] < lo] = -1
    return s

def sig_macd_scalp(df, p):
    ml, sl, _ = macd(df['close'], p['macd_f'], p['macd_s'], p['macd_sig'])
    s = pd.Series(0, index=df.index)
    s[(ml > sl) & (ml.shift(1) <= sl.shift(1))] = 1
    s[(ml < sl) & (ml.shift(1) >= sl.shift(1))] = -1
    return s

def sig_keltner_mr(df, p):
    u, m, l = keltner(df['high'], df['low'], df['close'], p['kelt'], p['atr'], p['kelt_m'])
    s = pd.Series(0, index=df.index)
    s[(df['close'] < l) & (df['close'] > df['close'].shift(1))] = 1
    s[(df['close'] > u) & (df['close'] < df['close'].shift(1))] = -1
    return s

def sig_dual_ema_adx(df, p):
    ef = ema(df['close'], p['ema_f']); es = ema(df['close'], p['ema_m'])
    ax = adx(df['high'], df['low'], df['close'], p['adx'])
    s = pd.Series(0, index=df.index)
    t = ax > p['adx_th']
    s[(ef > es) & (ef.shift(1) <= es.shift(1)) & t] = 1
    s[(ef < es) & (ef.shift(1) >= es.shift(1)) & t] = -1
    return s

def sig_vol_spike_breakout(df, p):
    avg_v = df['volume'].rolling(p['vol_lb']).mean()
    vr = df['volume'] / (avg_v + 1e-10)
    hi = df['high'].rolling(p['brk']).max().shift(1)
    lo = df['low'].rolling(p['brk']).min().shift(1)
    hv = vr > p['vol_th']
    s = pd.Series(0, index=df.index)
    s[(df['close'] > hi) & hv] = 1; s[(df['close'] < lo) & hv] = -1
    return s

def sig_stochastic_cross(df, p):
    k, d = stochastic(df['high'], df['low'], df['close'], p['stoch'], 3, 3)
    s = pd.Series(0, index=df.index)
    s[(k > d) & (k.shift(1) <= d.shift(1)) & (k < p['stoch_os'])] = 1
    s[(k < d) & (k.shift(1) >= d.shift(1)) & (k > p['stoch_ob'])] = -1
    return s

STRATEGIES = {
    'vwap_mr':       ('mean_reversion', sig_vwap_mr),
    'stoch_rsi':     ('mean_reversion', sig_stoch_rsi),
    'ema_ribbon':    ('trend',          sig_ema_ribbon),
    'bollinger_mr':  ('mean_reversion', sig_bollinger_mr),
    'rsi_extreme':   ('mean_reversion', sig_rsi_extreme),
    'momentum':      ('trend',          sig_momentum_burst),
    'range_breakout':('trend',          sig_range_breakout),
    'macd_scalp':    ('trend',          sig_macd_scalp),
    'keltner_mr':    ('mean_reversion', sig_keltner_mr),
    'dual_ema_adx':  ('trend',          sig_dual_ema_adx),
    'vol_spike_brk': ('trend',          sig_vol_spike_breakout),
    'stoch_cross':   ('mean_reversion', sig_stochastic_cross),
}

# ============================================================
# DATA LOADING
# ============================================================

def load_spot(asset, tf):
    path = DATA_DIR / tf / f"{asset}_spot_{tf}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    renames = {}
    for c in df.columns:
        if c in ('date', 'time', 'datetime'): renames[c] = 'timestamp'
    df.rename(columns=renames, inplace=True)
    if 'timestamp' not in df.columns:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def walk_forward_test(df, signals, holding_bars, bpy):
    n = len(df)
    if n < 100:
        return None
    fold_size = n // N_FOLDS
    if fold_size < 20:
        return None

    test_start = fold_size * 7
    test_df = df.iloc[test_start:]
    test_sig = signals.iloc[test_start:]

    trades = []
    i = 0
    while i < len(test_df) - holding_bars:
        sig = test_sig.iloc[i]
        if sig != 0:
            entry = test_df['close'].iloc[i]
            exit_ = test_df['close'].iloc[i + holding_bars]
            ret = (exit_ / entry - 1) * sig - COST_BPS / 10000
            trades.append({'direction': sig, 'return': ret})
            i += holding_bars
        else:
            i += 1

    if len(trades) < MIN_TRADES:
        return None

    returns = np.array([t['return'] for t in trades])
    test_bars = len(test_df)
    test_years = test_bars / bpy if bpy > 0 else 1
    tpy = len(trades) / test_years if test_years > 0 else len(trades)

    m = compute_metrics(returns, bars_per_year=bpy, total_bars=test_bars, trades_per_year=tpy)

    longs = [t['return'] for t in trades if t['direction'] == 1]
    shorts = [t['return'] for t in trades if t['direction'] == -1]
    m['n_long'] = len(longs)
    m['n_short'] = len(shorts)
    m['mean_long'] = np.mean(longs) if longs else 0
    m['mean_short'] = np.mean(shorts) if shorts else 0
    return m

# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    tfs = list(TF_CONFIG.keys())
    
    # Estimate total tests
    est = len(ASSETS) * len(STRATEGIES) * sum(len(TF_CONFIG[tf]['holdings']) for tf in tfs)
    bonf_alpha = 0.05 / est
    
    print("=" * 100)
    print("ALL-TIMEFRAME SCALPING/SHORT-TERM STRATEGY VALIDATION — Multi-Metric")
    print("=" * 100)
    print(f"Assets: {len(ASSETS)} | Strategies: {len(STRATEGIES)} | TFs: {tfs}")
    print(f"Est. tests: ~{est} | Bonferroni α: {bonf_alpha:.2e}")
    print(f"Costs: {COST_BPS}bps | Min trades: {MIN_TRADES} | Folds: {N_FOLDS}")
    print()

    all_results = []
    valid = 0

    for tf in tfs:
        cfg = TF_CONFIG[tf]
        bpy = cfg['bpy']
        params = get_tf_params(tf)
        print(f"\n{'='*80}")
        print(f"TIMEFRAME: {tf} | Holdings: {list(cfg['holdings'].keys())}")
        print(f"{'='*80}")

        for asset in ASSETS:
            df = load_spot(asset, tf)
            if df is None:
                continue

            for sname, (stype, sfn) in STRATEGIES.items():
                try:
                    signals = sfn(df, params)
                except Exception:
                    continue

                for hname, hbars in cfg['holdings'].items():
                    m = walk_forward_test(df, signals, hbars, bpy)
                    if m is None:
                        continue
                    valid += 1
                    grade = grade_strategy(m)
                    m['asset'] = asset
                    m['timeframe'] = tf
                    m['strategy'] = sname
                    m['type'] = stype
                    m['holding'] = hname
                    m['grade'] = grade
                    m['pass_raw'] = (m['p_value'] < 0.05 and m['sharpe'] > 0)
                    m['pass_bonf'] = (m['p_value'] < bonf_alpha and m['sharpe'] > 0)
                    all_results.append(m)

                    if m['sharpe'] > 1.0 or m['sharpe'] < -3:
                        icon = {'A': '🏆', 'B': '⭐', 'C': '📊', 'D': '📉', 'F': '💀'}[grade]
                        print(f"  {icon} [{grade}] {asset:>6} {sname:<16} hold={hname:<4}: {format_metrics_summary(m)}")

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'scalping_all_tf_multimetric.csv', index=False)

    # ============================================================
    # REPORT
    # ============================================================
    print(f"\n{'='*100}")
    print(f"RESULTS: {valid} valid tests in {elapsed:.1f}s")
    print(f"{'='*100}")

    if len(all_results) == 0:
        print("No valid results!")
        return

    raw_passes = sum(1 for r in all_results if r['pass_raw'])
    bonf_passes = sum(1 for r in all_results if r['pass_bonf'])
    print(f"Raw passes (p<0.05, Sharpe>0): {raw_passes} ({raw_passes/valid*100:.1f}%)")
    print(f"Bonferroni passes: {bonf_passes}")

    print(f"\n--- GRADE DISTRIBUTION ---")
    for g in ['A', 'B', 'C', 'D', 'F']:
        c = sum(1 for r in all_results if r['grade'] == g)
        print(f"  {g}: {c} ({c/valid*100:.1f}%)")

    # By timeframe
    print(f"\n--- BY TIMEFRAME ---")
    print(f"{'TF':<5} {'N':>5} {'AvgSh':>7} {'AvgSort':>8} {'AvgCal':>7} {'AvgMDD':>8} {'AvgPF':>6} {'Raw%':>6} {'Bonf':>5}")
    for tf in tfs:
        sub = [r for r in all_results if r['timeframe'] == tf]
        if not sub: continue
        n = len(sub)
        print(f"{tf:<5} {n:>5} {np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"{np.mean([r['sortino'] for r in sub]):>8.2f} "
              f"{np.mean([r['calmar'] for r in sub]):>7.2f} "
              f"{np.mean([r['max_drawdown'] for r in sub]):>7.1%} "
              f"{np.mean([r['profit_factor'] for r in sub]):>6.2f} "
              f"{sum(1 for r in sub if r['pass_raw'])/n*100:>5.1f}% "
              f"{sum(1 for r in sub if r['pass_bonf']):>5}")

    # By strategy
    print(f"\n--- BY STRATEGY ---")
    print(f"{'Strategy':<18} {'Type':<5} {'N':>4} {'AvgSh':>7} {'AvgSort':>8} {'AvgMDD':>8} {'AvgPF':>6} {'%Pos':>6} {'Raw':>4} {'Bonf':>4}")
    for sn in sorted(STRATEGIES.keys()):
        sub = [r for r in all_results if r['strategy'] == sn]
        if not sub: continue
        n = len(sub)
        st = STRATEGIES[sn][0][:4]
        print(f"{sn:<18} {st:<5} {n:>4} {np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"{np.mean([r['sortino'] for r in sub]):>8.2f} "
              f"{np.mean([r['max_drawdown'] for r in sub]):>7.1%} "
              f"{np.mean([r['profit_factor'] for r in sub]):>6.2f} "
              f"{sum(1 for r in sub if r['sharpe'] > 0)/n*100:>5.1f}% "
              f"{sum(1 for r in sub if r['pass_raw']):>4} "
              f"{sum(1 for r in sub if r['pass_bonf']):>4}")

    # By strategy type
    print(f"\n--- BY TYPE ---")
    for stype in ['trend', 'mean_reversion']:
        sub = [r for r in all_results if r['type'] == stype]
        if not sub: continue
        n = len(sub)
        print(f"  {stype:<16} N={n:>4} AvgSharpe={np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"AvgSortino={np.mean([r['sortino'] for r in sub]):>7.2f} "
              f"AvgMDD={np.mean([r['max_drawdown'] for r in sub]):>7.1%} "
              f"%Positive={sum(1 for r in sub if r['sharpe'] > 0)/n*100:.1f}%")

    # By holding period
    print(f"\n--- BY HOLDING PERIOD ---")
    all_holds = sorted(set(r['holding'] for r in all_results), 
                       key=lambda h: {'15m': 0.25, '1h': 1, '4h': 4, '12h': 12, '1d': 24, '3d': 72, '1w': 168, '2w': 336}.get(h, 0))
    for h in all_holds:
        sub = [r for r in all_results if r['holding'] == h]
        if not sub: continue
        n = len(sub)
        print(f"  {h:<5} N={n:>4} AvgSharpe={np.mean([r['sharpe'] for r in sub]):>7.2f} "
              f"AvgSortino={np.mean([r['sortino'] for r in sub]):>7.2f} "
              f"AvgMDD={np.mean([r['max_drawdown'] for r in sub]):>7.1%}")

    # Top 20 by Sharpe
    print(f"\n--- TOP 20 BY SHARPE ---")
    top = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:20]
    print(f"{'Asset':<7} {'TF':<4} {'Strategy':<16} {'Hold':<5} {'Gr':>2} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MDD':>8} {'PF':>6} {'WR':>5} {'Tail':>6} {'Skew':>6} {'N':>5} {'p':>10}")
    for r in top:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<16} {r['holding']:<5} {r['grade']:>2} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['max_drawdown']:>7.1%} "
              f"{r['profit_factor']:>6.2f} {r['win_rate']:>4.0%} {r['tail_ratio']:>6.2f} "
              f"{r['skewness']:>6.2f} {r['n_trades']:>5} {r['p_value']:>10.6f}")

    # Top 20 by Sortino
    print(f"\n--- TOP 20 BY SORTINO ---")
    top_s = sorted(all_results, key=lambda x: x['sortino'], reverse=True)[:20]
    for r in top_s:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<16} {r['holding']:<5} [{r['grade']}] "
              f"Sortino={r['sortino']:>7.2f} Sharpe={r['sharpe']:>6.2f} Calmar={r['calmar']:>6.2f} MDD={r['max_drawdown']:>7.1%}")

    # Top 20 by Calmar
    print(f"\n--- TOP 20 BY CALMAR ---")
    top_c = sorted(all_results, key=lambda x: x['calmar'], reverse=True)[:20]
    for r in top_c:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<16} {r['holding']:<5} [{r['grade']}] "
              f"Calmar={r['calmar']:>7.2f} Sharpe={r['sharpe']:>6.2f} MDD={r['max_drawdown']:>7.1%} TotRet={r['total_return']:>7.1%}")

    # Best by multi-metric score (composite: normalized Sharpe + Sortino + Calmar - MDD penalty)
    print(f"\n--- TOP 20 BY COMPOSITE SCORE (Sharpe + Sortino/3 + Calmar/2 - |MDD|) ---")
    for r in all_results:
        r['composite'] = r['sharpe'] + r['sortino']/3 + r['calmar']/2 - abs(r['max_drawdown'])
    top_comp = sorted(all_results, key=lambda x: x['composite'], reverse=True)[:20]
    for r in top_comp:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<16} {r['holding']:<5} [{r['grade']}] "
              f"Comp={r['composite']:>6.2f} Sh={r['sharpe']:>5.2f} So={r['sortino']:>5.2f} "
              f"Ca={r['calmar']:>5.2f} MDD={r['max_drawdown']:>7.1%} p={r['p_value']:.4f}")

    print(f"\nResults saved to: {RESULTS_DIR / 'scalping_all_tf_multimetric.csv'}")

if __name__ == '__main__':
    main()
