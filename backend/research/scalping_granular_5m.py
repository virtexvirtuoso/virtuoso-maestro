#!/usr/bin/env python3
"""
Granular 5m/15m Scalping Validation
====================================
Tests every reasonable holding period for 5m and 15m signals.

5m holdings:  5m(1), 10m(2), 15m(3), 30m(6), 45m(9), 1h(12), 1.5h(18), 2h(24), 3h(36), 4h(48), 6h(72), 8h(96), 12h(144)
15m holdings: 15m(1), 30m(2), 45m(3), 1h(4), 1.5h(6), 2h(8), 3h(12), 4h(16), 6h(24), 8h(32), 12h(48), 16h(64), 24h(96)

This answers: is there ANY hold duration where 5m signals produce edge?
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

COST_BPS = 10
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
            '5m': 1, '10m': 2, '15m': 3, '30m': 6, '45m': 9,
            '1h': 12, '1.5h': 18, '2h': 24, '3h': 36, '4h': 48,
            '6h': 72, '8h': 96, '12h': 144,
        },
        'bpy': 365.25 * 24 * 12,
        'params': dict(rsi=7, stoch=8, bb=12, ema_f=5, ema_m=13, ema_s=34,
                       macd_f=8, macd_s=21, macd_sig=5, kelt=12, atr=10, adx=10,
                       vwap=48, brk=24, vol_lb=20, vol_th=2.0, rsi_os=20, rsi_ob=80,
                       stoch_os=15, stoch_ob=85, bb_std=2.0, kelt_m=1.5, adx_th=20,
                       mom_n=3, mom_th=0.003),
    },
    '15m': {
        'holdings': {
            '15m': 1, '30m': 2, '45m': 3, '1h': 4, '1.5h': 6,
            '2h': 8, '3h': 12, '4h': 16, '6h': 24, '8h': 32,
            '12h': 48, '16h': 64, '24h': 96,
        },
        'bpy': 365.25 * 24 * 4,
        'params': dict(rsi=10, stoch=10, bb=16, ema_f=5, ema_m=13, ema_s=34,
                       macd_f=8, macd_s=21, macd_sig=5, kelt=16, atr=12, adx=12,
                       vwap=32, brk=16, vol_lb=16, vol_th=2.0, rsi_os=25, rsi_ob=75,
                       stoch_os=20, stoch_ob=80, bb_std=2.0, kelt_m=1.5, adx_th=22,
                       mom_n=3, mom_th=0.005),
    },
}

# ============================================================
# INDICATORS (compact)
# ============================================================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rsi(close, p=14):
    d = close.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    return 100 - 100 / (1 + g.ewm(alpha=1/p, min_periods=p).mean() / l.ewm(alpha=1/p, min_periods=p).mean())
def stoch_rsi(close, rsi_p, stoch_p):
    r = rsi(close, rsi_p); lo = r.rolling(stoch_p).min(); hi = r.rolling(stoch_p).max()
    sr = (r - lo) / (hi - lo + 1e-10); return sr.rolling(3).mean(), sr.rolling(3).mean().rolling(3).mean()
def stochastic(high, low, close, p=14):
    lo = low.rolling(p).min(); hi = high.rolling(p).max()
    raw = (close - lo) / (hi - lo + 1e-10) * 100; k = raw.rolling(3).mean(); return k, k.rolling(3).mean()
def bbands(close, p=20, mult=2.0):
    m = close.rolling(p).mean(); s = close.rolling(p).std(); return m + mult * s, m, m - mult * s
def keltner(high, low, close, ep, atr_p, mult):
    e = ema(close, ep)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.rolling(atr_p).mean(); return e + mult * a, e, e - mult * a
def macd(close, f, s, sig):
    ml = ema(close, f) - ema(close, s); return ml, ema(ml, sig), ml - ema(ml, sig)
def atr(high, low, close, p):
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def adx(high, low, close, p):
    pdm = high.diff(); mdm = -low.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100 * pdm.ewm(alpha=1/p, min_periods=p).mean() / a
    mdi = 100 * mdm.ewm(alpha=1/p, min_periods=p).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.ewm(alpha=1/p, min_periods=p).mean()
def rolling_vwap(high, low, close, vol, p):
    tp = (high + low + close) / 3; return (tp * vol).rolling(p).sum() / (vol.rolling(p).sum() + 1e-10)

# ============================================================
# STRATEGIES
# ============================================================
def sig_vwap_mr(df, p):
    vw = rolling_vwap(df['high'], df['low'], df['close'], df['volume'], p['vwap'])
    a = atr(df['high'], df['low'], df['close'], p['atr'])
    dev = (df['close'] - vw) / (a + 1e-10); s = pd.Series(0, index=df.index)
    s[dev < -1.5] = 1; s[dev > 1.5] = -1; return s

def sig_stoch_rsi(df, p):
    k, d = stoch_rsi(df['close'], p['rsi'], p['stoch']); s = pd.Series(0, index=df.index)
    s[(k > d) & (k.shift(1) <= d.shift(1)) & (k < 0.3)] = 1
    s[(k < d) & (k.shift(1) >= d.shift(1)) & (k > 0.7)] = -1; return s

def sig_ema_ribbon(df, p):
    ef = ema(df['close'], p['ema_f']); em = ema(df['close'], p['ema_m']); es = ema(df['close'], p['ema_s'])
    bull = (ef > em) & (em > es); bear = (ef < em) & (em < es); s = pd.Series(0, index=df.index)
    s[bull & ~bull.shift(1).fillna(False)] = 1; s[bear & ~bear.shift(1).fillna(False)] = -1; return s

def sig_bollinger_mr(df, p):
    u, m, l = bbands(df['close'], p['bb'], p['bb_std']); s = pd.Series(0, index=df.index)
    s[(df['close'] <= l) & (df['close'] > df['close'].shift(1))] = 1
    s[(df['close'] >= u) & (df['close'] < df['close'].shift(1))] = -1; return s

def sig_rsi_extreme(df, p):
    r = rsi(df['close'], p['rsi']); s = pd.Series(0, index=df.index)
    s[(r > p['rsi_os']) & (r.shift(1) <= p['rsi_os'])] = 1
    s[(r < p['rsi_ob']) & (r.shift(1) >= p['rsi_ob'])] = -1; return s

def sig_momentum(df, p):
    pc = df['close'].pct_change(p['mom_n']); s = pd.Series(0, index=df.index)
    s[pc > p['mom_th']] = 1; s[pc < -p['mom_th']] = -1; return s

def sig_range_breakout(df, p):
    hi = df['high'].rolling(p['brk']).max().shift(1); lo = df['low'].rolling(p['brk']).min().shift(1)
    s = pd.Series(0, index=df.index); s[df['close'] > hi] = 1; s[df['close'] < lo] = -1; return s

def sig_macd_scalp(df, p):
    ml, sl, _ = macd(df['close'], p['macd_f'], p['macd_s'], p['macd_sig']); s = pd.Series(0, index=df.index)
    s[(ml > sl) & (ml.shift(1) <= sl.shift(1))] = 1; s[(ml < sl) & (ml.shift(1) >= sl.shift(1))] = -1; return s

def sig_keltner_mr(df, p):
    u, m, l = keltner(df['high'], df['low'], df['close'], p['kelt'], p['atr'], p['kelt_m'])
    s = pd.Series(0, index=df.index)
    s[(df['close'] < l) & (df['close'] > df['close'].shift(1))] = 1
    s[(df['close'] > u) & (df['close'] < df['close'].shift(1))] = -1; return s

def sig_dual_ema_adx(df, p):
    ef = ema(df['close'], p['ema_f']); es = ema(df['close'], p['ema_m'])
    ax = adx(df['high'], df['low'], df['close'], p['adx']); s = pd.Series(0, index=df.index)
    t = ax > p['adx_th']
    s[(ef > es) & (ef.shift(1) <= es.shift(1)) & t] = 1
    s[(ef < es) & (ef.shift(1) >= es.shift(1)) & t] = -1; return s

def sig_vol_spike_brk(df, p):
    avg_v = df['volume'].rolling(p['vol_lb']).mean(); vr = df['volume'] / (avg_v + 1e-10)
    hi = df['high'].rolling(p['brk']).max().shift(1); lo = df['low'].rolling(p['brk']).min().shift(1)
    hv = vr > p['vol_th']; s = pd.Series(0, index=df.index)
    s[(df['close'] > hi) & hv] = 1; s[(df['close'] < lo) & hv] = -1; return s

def sig_stoch_cross(df, p):
    k, d = stochastic(df['high'], df['low'], df['close'], p['stoch']); s = pd.Series(0, index=df.index)
    s[(k > d) & (k.shift(1) <= d.shift(1)) & (k < p['stoch_os'])] = 1
    s[(k < d) & (k.shift(1) >= d.shift(1)) & (k > p['stoch_ob'])] = -1; return s

STRATEGIES = {
    'vwap_mr': ('mr', sig_vwap_mr), 'stoch_rsi': ('mr', sig_stoch_rsi),
    'ema_ribbon': ('tr', sig_ema_ribbon), 'bollinger_mr': ('mr', sig_bollinger_mr),
    'rsi_extreme': ('mr', sig_rsi_extreme), 'momentum': ('tr', sig_momentum),
    'range_breakout': ('tr', sig_range_breakout), 'macd_scalp': ('tr', sig_macd_scalp),
    'keltner_mr': ('mr', sig_keltner_mr), 'dual_ema_adx': ('tr', sig_dual_ema_adx),
    'vol_spike_brk': ('tr', sig_vol_spike_brk), 'stoch_cross': ('mr', sig_stoch_cross),
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
    total_est = sum(len(ASSETS) * len(STRATEGIES) * len(TF_CONFIG[tf]['holdings']) for tf in TF_CONFIG)
    bonf_alpha = 0.05 / total_est

    print("=" * 100)
    print("GRANULAR 5m/15m SCALPING — Every Holding Period")
    print("=" * 100)
    print(f"5m holdings: {list(TF_CONFIG['5m']['holdings'].keys())}")
    print(f"15m holdings: {list(TF_CONFIG['15m']['holdings'].keys())}")
    print(f"Assets: {len(ASSETS)} | Strategies: {len(STRATEGIES)} | Est tests: ~{total_est}")
    print(f"Bonferroni α: {bonf_alpha:.2e}\n")

    all_results = []
    for tf, cfg in TF_CONFIG.items():
        bpy = cfg['bpy']; params = cfg['params']
        print(f"\n{'='*80}\nTIMEFRAME: {tf}\n{'='*80}", flush=True)
        for ai, asset in enumerate(ASSETS):
            df = load_spot(asset, tf)
            if df is None: continue
            print(f"  [{ai+1}/{len(ASSETS)}] {asset} ({len(df)} bars)...", end='', flush=True)
            asset_hits = 0
            for sname, (stype, sfn) in STRATEGIES.items():
                try: signals = sfn(df, params)
                except: continue
                for hname, hbars in cfg['holdings'].items():
                    m = walk_forward_test(df, signals, hbars, bpy)
                    if m is None: continue
                    grade = grade_strategy(m)
                    m.update({'asset': asset, 'timeframe': tf, 'strategy': sname, 'type': stype,
                              'holding': hname, 'holding_bars': hbars, 'grade': grade,
                              'pass_raw': m['p_value'] < 0.05 and m['sharpe'] > 0,
                              'pass_bonf': m['p_value'] < bonf_alpha and m['sharpe'] > 0})
                    all_results.append(m)
                    if m['sharpe'] > 1.0: asset_hits += 1
            print(f" {asset_hits} hits (Sharpe>1)", flush=True)
            del df; gc.collect()

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'scalping_granular_5m15m.csv', index=False)

    print(f"\n{'='*100}")
    print(f"RESULTS: {len(all_results)} valid tests in {elapsed:.1f}s")
    print(f"{'='*100}")
    if not all_results: return

    raw = sum(1 for r in all_results if r['pass_raw'])
    bonf = sum(1 for r in all_results if r['pass_bonf'])
    pos = sum(1 for r in all_results if r['sharpe'] > 0)
    print(f"Positive Sharpe: {pos} ({pos/len(all_results)*100:.1f}%)")
    print(f"Raw passes (p<0.05): {raw} ({raw/len(all_results)*100:.1f}%)")
    print(f"Bonferroni passes: {bonf}")

    # Grades
    print(f"\n--- GRADES ---")
    for g in 'ABCDF':
        c = sum(1 for r in all_results if r['grade'] == g)
        print(f"  {g}: {c} ({c/len(all_results)*100:.1f}%)")

    # By TF
    print(f"\n--- BY TIMEFRAME ---")
    for tf in ['5m', '15m']:
        sub = [r for r in all_results if r['timeframe'] == tf]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        print(f"  {tf}: {n} tests, {p} positive ({p/n*100:.1f}%), avg Sharpe={np.mean([r['sharpe'] for r in sub]):.2f}")

    # By holding period — THE KEY TABLE
    print(f"\n--- BY HOLDING PERIOD (averaged across all strategies/assets) ---")
    print(f"{'Hold':<6} {'TF':<4} {'Bars':>4} {'N':>5} {'%Pos':>6} {'AvgSh':>7} {'AvgSort':>8} {'AvgMDD':>8} {'AvgPF':>6} {'Raw':>4}")
    hold_order = {'5m': 0.083, '10m': 0.167, '15m': 0.25, '30m': 0.5, '45m': 0.75,
                  '1h': 1, '1.5h': 1.5, '2h': 2, '3h': 3, '4h': 4, '6h': 6, '8h': 8,
                  '12h': 12, '16h': 16, '24h': 24}
    for tf in ['5m', '15m']:
        holds = TF_CONFIG[tf]['holdings']
        for hname in holds:
            sub = [r for r in all_results if r['timeframe'] == tf and r['holding'] == hname]
            if not sub: continue
            n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
            rw = sum(1 for r in sub if r['pass_raw'])
            print(f"{hname:<6} {tf:<4} {holds[hname]:>4} {n:>5} {p/n*100:>5.1f}% "
                  f"{np.mean([r['sharpe'] for r in sub]):>7.2f} "
                  f"{np.mean([r['sortino'] for r in sub]):>8.2f} "
                  f"{np.mean([r['max_drawdown'] for r in sub]):>7.1%} "
                  f"{np.mean([r['profit_factor'] for r in sub]):>6.2f} {rw:>4}")

    # By strategy
    print(f"\n--- BY STRATEGY ---")
    print(f"{'Strategy':<16} {'Type':<3} {'N':>5} {'%Pos':>6} {'AvgSh':>7} {'Best':>6} {'BestAsset':<8}")
    for sn in sorted(STRATEGIES.keys()):
        sub = [r for r in all_results if r['strategy'] == sn]
        if not sub: continue
        n = len(sub); p = sum(1 for r in sub if r['sharpe'] > 0)
        best = max(sub, key=lambda x: x['sharpe'])
        print(f"{sn:<16} {STRATEGIES[sn][0]:<3} {n:>5} {p/n*100:>5.1f}% "
              f"{np.mean([r['sharpe'] for r in sub]):>7.2f} {best['sharpe']:>6.2f} {best['asset']:<8}")

    # Top 25 overall
    print(f"\n--- TOP 25 BY SHARPE ---")
    top = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:25]
    print(f"{'Asset':<7} {'TF':<4} {'Strategy':<16} {'Hold':<6} {'Gr':>2} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MDD':>8} {'PF':>6} {'WR':>5} {'Tail':>6} {'N':>5} {'p':>10}")
    for r in top:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<16} {r['holding']:<6} {r['grade']:>2} "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['max_drawdown']:>7.1%} "
              f"{r['profit_factor']:>6.2f} {r['win_rate']:>4.0%} {r['tail_ratio']:>6.2f} "
              f"{r['n_trades']:>5} {r['p_value']:>10.6f}")

    # Top composite
    print(f"\n--- TOP 15 BY COMPOSITE (Sharpe + Sortino/3 + Calmar/2 - |MDD|) ---")
    for r in all_results:
        r['composite'] = r['sharpe'] + r['sortino']/3 + r['calmar']/2 - abs(r['max_drawdown'])
    top_c = sorted(all_results, key=lambda x: x['composite'], reverse=True)[:15]
    for r in top_c:
        print(f"{r['asset']:<7} {r['timeframe']:<4} {r['strategy']:<16} hold={r['holding']:<6} [{r['grade']}] "
              f"Comp={r['composite']:>6.2f} Sh={r['sharpe']:>5.2f} So={r['sortino']:>5.2f} "
              f"Ca={r['calmar']:>5.2f} MDD={r['max_drawdown']:>7.1%} n={r['n_trades']:>4}")

    # THE CRITICAL CHART: Sharpe vs holding period (hours)
    print(f"\n--- SHARPE CURVE BY HOLD DURATION (hours) ---")
    for tf in ['5m', '15m']:
        print(f"\n  {tf}:")
        holds = TF_CONFIG[tf]['holdings']
        bars_per_hour = 12 if tf == '5m' else 4
        for hname, hbars in holds.items():
            hours = hbars / bars_per_hour
            sub = [r for r in all_results if r['timeframe'] == tf and r['holding'] == hname]
            if not sub: continue
            avg_sh = np.mean([r['sharpe'] for r in sub])
            pos_pct = sum(1 for r in sub if r['sharpe'] > 0) / len(sub) * 100
            bar = '█' * max(0, int((avg_sh + 20) / 2)) if avg_sh > -20 else ''
            print(f"    {hours:>6.1f}h ({hname:>5}): avg_sharpe={avg_sh:>7.2f} pos={pos_pct:>5.1f}% {bar}")

    print(f"\nResults: {RESULTS_DIR / 'scalping_granular_5m15m.csv'}")

if __name__ == '__main__':
    main()
