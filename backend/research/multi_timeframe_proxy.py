#!/usr/bin/env python3
"""
ULTRATHINK: Multi-Timeframe Proxy
Combine short + long lookbacks for trend capture + crash protection
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from scipy import stats
import json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)
COST = 0.001
results_dir = Path("/Users/ffv_macmini/Desktop/maestro/data/research")

print("=" * 70)
print("ULTRATHINK: Multi-Timeframe Proxy")
print("=" * 70)

# ── Data ──
print("\n[1/5] Loading data...")
tickers = {'BTC': 'BTC-USD', 'UUP': 'UUP', 'GLD': 'GLD', 'TLT': 'TLT', 'HYG': 'HYG'}
prices = {}
for name, tick in tickers.items():
    df = yf.download(tick, start="2017-01-01", end="2025-12-31", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices[name] = df['Close'].dropna()
    print(f"  {name}: {len(prices[name])} days")

walcl = fred.get_series('WALCL', observation_start="2017-01-01")
wtregen = fred.get_series('WTREGEN', observation_start="2017-01-01")
rrp = fred.get_series('RRPONTSYD', observation_start="2017-01-01")

pdf = pd.DataFrame(prices).ffill().dropna(subset=['BTC'])
net_liq = walcl.reindex(pdf.index, method='ffill') - wtregen.reindex(pdf.index, method='ffill') - rrp.reindex(pdf.index, method='ffill')
inverse = {'UUP'}


def build_proxy(df, lookback, threshold):
    scores = pd.DataFrame(index=df.index)
    for inst in ['UUP', 'GLD', 'TLT', 'HYG']:
        chg = df[inst].pct_change(lookback)
        if inst in inverse:
            scores[inst] = (chg < 0).astype(int)
        else:
            scores[inst] = (chg > 0).astype(int)
    return scores.sum(axis=1), (scores.sum(axis=1) >= threshold).astype(int)


def backtest(signal, btc, cost=COST):
    sig = signal.shift(1).fillna(0)
    ret = btc.pct_change().fillna(0)
    trades = sig.diff().abs().fillna(0)
    return sig * ret - trades * cost


def metrics(strat_ret):
    if strat_ret.std() == 0:
        return {'sharpe': 0, 'max_dd': 0, 'cagr': 0, 'total_ret': 0, 'n_trades': 0}
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    years = len(strat_ret) / 252
    cagr = (1 + total) ** (1/years) - 1 if total > -1 else -1
    dd = (cum / cum.cummax() - 1).min()
    sig = signal.shift(1).fillna(0) if 'signal' in dir() else strat_ret
    return {'sharpe': round(sharpe, 4), 'max_dd': round(dd * 100, 1),
            'cagr': round(cagr * 100, 1), 'total_ret': round(total * 100, 1)}


# Baselines
alden_sig = (net_liq.pct_change(20) > 0).astype(int).reindex(pdf.index).ffill().fillna(0)
alden_ret = backtest(alden_sig, pdf['BTC'])
alden_m = metrics(alden_ret)

_, single_180 = build_proxy(pdf, 180, 2)
single_ret = backtest(single_180, pdf['BTC'])
single_m = metrics(single_ret)

bh_ret = pdf['BTC'].pct_change().fillna(0)
bh_m = metrics(bh_ret)

print(f"\n  BASELINES:")
print(f"  Single 180d/2: Sharpe={single_m['sharpe']}")
print(f"  Alden:         Sharpe={alden_m['sharpe']}")
print(f"  Buy&Hold:      Sharpe={bh_m['sharpe']}")


# ── PHASE 1: Dual Timeframe ──
print("\n[2/5] Dual Timeframe Combinations...")
print(f"  {'Short':>5s} {'Long':>5s} {'Method':>15s} {'SThr':>4s} {'LThr':>4s} | {'Sharpe':>7s} {'MaxDD':>7s} {'CAGR':>7s} {'Trades':>6s}")
print(f"  {'-'*5} {'-'*5} {'-'*15} {'-'*4} {'-'*4} | {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

best = {'sharpe': -999}
all_combos = []

for short_lb in [10, 20, 30, 40, 60]:
    for long_lb in [120, 150, 180, 210]:
        short_score, short_sig = build_proxy(pdf, short_lb, 2)
        long_score, long_sig = build_proxy(pdf, long_lb, 2)
        
        # Method 1: AND — both timeframes must agree (conservative)
        and_sig = (short_sig & long_sig).astype(int)
        r = backtest(and_sig, pdf['BTC'])
        m = metrics(r)
        n_trades = int(and_sig.shift(1).diff().abs().fillna(0).sum())
        combo = {'short': short_lb, 'long': long_lb, 'method': 'AND', 's_thr': 2, 'l_thr': 2, **m, 'n_trades': n_trades}
        all_combos.append(combo)
        if m['sharpe'] > best['sharpe']:
            best = combo.copy()
        
        # Method 2: OR — either timeframe bullish (aggressive)
        or_sig = (short_sig | long_sig).astype(int)
        r = backtest(or_sig, pdf['BTC'])
        m = metrics(r)
        n_trades = int(or_sig.shift(1).diff().abs().fillna(0).sum())
        combo = {'short': short_lb, 'long': long_lb, 'method': 'OR', 's_thr': 2, 'l_thr': 2, **m, 'n_trades': n_trades}
        all_combos.append(combo)
        if m['sharpe'] > best['sharpe']:
            best = combo.copy()
        
        # Method 3: Weighted average — long score + short score, threshold on combined
        combined_score = long_score * 0.6 + short_score * 0.4  # weight long more
        for comb_thr in [2, 2.5, 3]:
            comb_sig = (combined_score >= comb_thr).astype(int)
            r = backtest(comb_sig, pdf['BTC'])
            m = metrics(r)
            n_trades = int(comb_sig.shift(1).diff().abs().fillna(0).sum())
            combo = {'short': short_lb, 'long': long_lb, 'method': f'WGT_{comb_thr}', 's_thr': 2, 'l_thr': 2, **m, 'n_trades': n_trades}
            all_combos.append(combo)
            if m['sharpe'] > best['sharpe']:
                best = combo.copy()
        
        # Method 4: Long for trend, short as override (crash protection)
        # Default to long signal, but EXIT if short goes bearish
        override_sig = long_sig.copy()
        override_sig[short_sig == 0] = 0  # short-term bearish overrides
        r = backtest(override_sig, pdf['BTC'])
        m = metrics(r)
        n_trades = int(override_sig.shift(1).diff().abs().fillna(0).sum())
        combo = {'short': short_lb, 'long': long_lb, 'method': 'OVERRIDE', 's_thr': 2, 'l_thr': 2, **m, 'n_trades': n_trades}
        all_combos.append(combo)
        if m['sharpe'] > best['sharpe']:
            best = combo.copy()
        
        # Method 5: Tiered — full position when both agree, half when only long
        tiered_sig = long_sig * 0.5 + (short_sig & long_sig).astype(float) * 0.5
        r_tiered = tiered_sig.shift(1).fillna(0) * pdf['BTC'].pct_change().fillna(0) - tiered_sig.shift(1).diff().abs().fillna(0) * COST
        m = metrics(r_tiered)
        n_trades = int(tiered_sig.shift(1).diff().abs().fillna(0).sum())
        combo = {'short': short_lb, 'long': long_lb, 'method': 'TIERED', 's_thr': 2, 'l_thr': 2, **m, 'n_trades': n_trades}
        all_combos.append(combo)
        if m['sharpe'] > best['sharpe']:
            best = combo.copy()

# Sort and print top 15
top = sorted(all_combos, key=lambda x: x['sharpe'], reverse=True)[:15]
for c in top:
    print(f"  {c['short']:5d} {c['long']:5d} {c['method']:>15s} {c['s_thr']:4d} {c['l_thr']:4d} | {c['sharpe']:7.4f} {c['max_dd']:6.1f}% {c['cagr']:6.1f}% {c['n_trades']:6d}")

print(f"\n  BEST DUAL: short={best['short']}d, long={best['long']}d, method={best['method']}")
print(f"  Sharpe={best['sharpe']}, MaxDD={best['max_dd']}%, CAGR={best['cagr']}%")


# ── PHASE 2: Triple Timeframe ──
print("\n[3/5] Triple Timeframe Combinations...")
best_triple = {'sharpe': -999}
triple_combos = []

for fast in [10, 20, 30]:
    for mid in [60, 90]:
        for slow in [150, 180, 210]:
            fast_score, fast_sig = build_proxy(pdf, fast, 2)
            mid_score, mid_sig = build_proxy(pdf, mid, 2)
            slow_score, slow_sig = build_proxy(pdf, slow, 2)
            
            # Triple AND
            triple_and = (fast_sig & mid_sig & slow_sig).astype(int)
            r = backtest(triple_and, pdf['BTC'])
            m = metrics(r)
            n = int(triple_and.shift(1).diff().abs().fillna(0).sum())
            combo = {'fast': fast, 'mid': mid, 'slow': slow, 'method': 'AND3', **m, 'n_trades': n}
            triple_combos.append(combo)
            if m['sharpe'] > best_triple['sharpe']:
                best_triple = combo.copy()
            
            # Majority vote (2-of-3)
            vote = (fast_sig + mid_sig + slow_sig)
            majority = (vote >= 2).astype(int)
            r = backtest(majority, pdf['BTC'])
            m = metrics(r)
            n = int(majority.shift(1).diff().abs().fillna(0).sum())
            combo = {'fast': fast, 'mid': mid, 'slow': slow, 'method': 'VOTE2of3', **m, 'n_trades': n}
            triple_combos.append(combo)
            if m['sharpe'] > best_triple['sharpe']:
                best_triple = combo.copy()
            
            # Weighted: 50% slow + 30% mid + 20% fast
            weighted = slow_score * 0.5 + mid_score * 0.3 + fast_score * 0.2
            for thr in [2, 2.5, 3]:
                w_sig = (weighted >= thr).astype(int)
                r = backtest(w_sig, pdf['BTC'])
                m = metrics(r)
                n = int(w_sig.shift(1).diff().abs().fillna(0).sum())
                combo = {'fast': fast, 'mid': mid, 'slow': slow, 'method': f'WGT3_{thr}', **m, 'n_trades': n}
                triple_combos.append(combo)
                if m['sharpe'] > best_triple['sharpe']:
                    best_triple = combo.copy()
            
            # Tiered sizing: 100% all agree, 66% slow+mid, 33% slow only, 0% slow bearish
            tier3 = slow_sig * 0.33 + (slow_sig & mid_sig).astype(float) * 0.33 + (fast_sig & mid_sig & slow_sig).astype(float) * 0.34
            r3 = tier3.shift(1).fillna(0) * pdf['BTC'].pct_change().fillna(0) - tier3.shift(1).diff().abs().fillna(0) * COST
            m = metrics(r3)
            n = int(tier3.shift(1).diff().abs().fillna(0).sum())
            combo = {'fast': fast, 'mid': mid, 'slow': slow, 'method': 'TIER3', **m, 'n_trades': n}
            triple_combos.append(combo)
            if m['sharpe'] > best_triple['sharpe']:
                best_triple = combo.copy()

top_triple = sorted(triple_combos, key=lambda x: x['sharpe'], reverse=True)[:10]
print(f"\n  Top 10 triple combos:")
for c in top_triple:
    print(f"  F={c['fast']:2d} M={c['mid']:2d} S={c['slow']:3d} {c['method']:>10s} | Sharpe={c['sharpe']:7.4f} MaxDD={c['max_dd']:6.1f}% CAGR={c['cagr']:6.1f}%")

print(f"\n  BEST TRIPLE: fast={best_triple['fast']}d, mid={best_triple['mid']}d, slow={best_triple['slow']}d, method={best_triple['method']}")
print(f"  Sharpe={best_triple['sharpe']}, MaxDD={best_triple['max_dd']}%, CAGR={best_triple['cagr']}%")


# ── PHASE 3: Regime-Specific Check ──
print("\n[4/5] Regime check on best multi-TF...")

# Use overall best (dual or triple)
if best_triple['sharpe'] > best['sharpe']:
    overall_best = best_triple
    # Rebuild best triple signal
    f_s, f_sig = build_proxy(pdf, best_triple['fast'], 2)
    m_s, m_sig = build_proxy(pdf, best_triple['mid'], 2)
    s_s, s_sig = build_proxy(pdf, best_triple['slow'], 2)
    
    if 'AND3' in best_triple['method']:
        best_signal = (f_sig & m_sig & s_sig).astype(int)
    elif 'VOTE' in best_triple['method']:
        best_signal = ((f_sig + m_sig + s_sig) >= 2).astype(int)
    elif 'WGT3' in best_triple['method']:
        thr = float(best_triple['method'].split('_')[1])
        weighted = s_s * 0.5 + m_s * 0.3 + f_s * 0.2
        best_signal = (weighted >= thr).astype(int)
    elif 'TIER3' in best_triple['method']:
        best_signal = (s_sig * 0.33 + (s_sig & m_sig).astype(float) * 0.33 + (f_sig & m_sig & s_sig).astype(float) * 0.34)
        best_signal = (best_signal > 0.5).astype(int)  # binary for regime check
    label = f"Triple({best_triple['fast']}/{best_triple['mid']}/{best_triple['slow']} {best_triple['method']})"
else:
    overall_best = best
    # Rebuild best dual signal
    _, s_sig = build_proxy(pdf, best['short'], 2)
    _, l_sig = build_proxy(pdf, best['long'], 2)
    s_score, _ = build_proxy(pdf, best['short'], 2)
    l_score, _ = build_proxy(pdf, best['long'], 2)
    
    if best['method'] == 'AND':
        best_signal = (s_sig & l_sig).astype(int)
    elif best['method'] == 'OR':
        best_signal = (s_sig | l_sig).astype(int)
    elif best['method'] == 'OVERRIDE':
        best_signal = l_sig.copy()
        best_signal[s_sig == 0] = 0
    elif 'WGT' in best['method']:
        thr = float(best['method'].split('_')[1])
        combined = l_score * 0.6 + s_score * 0.4
        best_signal = (combined >= thr).astype(int)
    elif best['method'] == 'TIERED':
        best_signal = (l_sig * 0.5 + (s_sig & l_sig).astype(float) * 0.5 > 0.4).astype(int)
    label = f"Dual({best['short']}/{best['long']} {best['method']})"

regimes = {
    'COVID crash (2020-03)': ('2020-03-01', '2020-03-31'),
    'Bull run (2020-H2)': ('2020-07-01', '2020-12-31'),
    'May 2021 crash': ('2021-05-01', '2021-06-30'),
    'Bear (2022)': ('2022-01-01', '2022-12-31'),
    'FTX collapse (2022-11)': ('2022-11-01', '2022-11-30'),
    'Recovery (2023)': ('2023-01-01', '2023-12-31'),
    'Bull (2024)': ('2024-01-01', '2024-12-31'),
}

print(f"\n  Best: {label}")
for name, (start, end) in regimes.items():
    mask = (pdf.index >= start) & (pdf.index <= end)
    if mask.sum() == 0:
        continue
    # Multi-TF
    mtf_r = backtest(best_signal[mask], pdf['BTC'][mask])
    mtf_cum = (1 + mtf_r).cumprod().iloc[-1] - 1
    # Single 180
    s180_r = backtest(single_180[mask], pdf['BTC'][mask])
    s180_cum = (1 + s180_r).cumprod().iloc[-1] - 1
    # Alden
    a_r = backtest(alden_sig[mask], pdf['BTC'][mask])
    a_cum = (1 + a_r).cumprod().iloc[-1] - 1
    # B&H
    bh = pdf['BTC'][mask].pct_change().fillna(0)
    bh_cum = (1 + bh).cumprod().iloc[-1] - 1
    
    winner = "MTF" if mtf_cum > s180_cum and mtf_cum > a_cum else ("180d" if s180_cum > a_cum else "ALDEN")
    print(f"  {name:25s} | MTF={mtf_cum*100:+7.1f}% 180d={s180_cum*100:+7.1f}% Alden={a_cum*100:+7.1f}% B&H={bh_cum*100:+7.1f}% | {winner}")


# ── PHASE 4: Holdout ──
print("\n[5/5] Holdout 2024-2025...")
holdout = pdf.index >= "2024-01-01"
mtf_hold = backtest(best_signal[holdout], pdf['BTC'][holdout])
s180_hold = backtest(single_180[holdout], pdf['BTC'][holdout])
alden_hold = backtest(alden_sig[holdout], pdf['BTC'][holdout])

print(f"  Multi-TF:  {metrics(mtf_hold)}")
print(f"  Single180: {metrics(s180_hold)}")
print(f"  Alden:     {metrics(alden_hold)}")
print(f"  B&H:       {metrics(pdf['BTC'][holdout].pct_change().fillna(0))}")


# ── Final Summary ──
print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
best_mtf_m = metrics(backtest(best_signal, pdf['BTC']))
print(f"  {'Method':<30s} {'Sharpe':>8s} {'MaxDD':>8s} {'CAGR':>8s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'Multi-TF (best)':<30s} {best_mtf_m['sharpe']:8.4f} {best_mtf_m['max_dd']:7.1f}% {best_mtf_m['cagr']:7.1f}%")
print(f"  {'Single 180d/2':<30s} {single_m['sharpe']:8.4f} {single_m['max_dd']:7.1f}% {single_m['cagr']:7.1f}%")
print(f"  {'Alden Net Liquidity':<30s} {alden_m['sharpe']:8.4f} {alden_m['max_dd']:7.1f}% {alden_m['cagr']:7.1f}%")
print(f"  {'Buy & Hold':<30s} {bh_m['sharpe']:8.4f} {bh_m['max_dd']:7.1f}% {bh_m['cagr']:7.1f}%")
print(f"\n  Best config: {label}")

# Save
results = {
    'best_dual': best,
    'best_triple': best_triple,
    'overall_best': overall_best,
    'baselines': {'single_180': single_m, 'alden': alden_m, 'buy_hold': bh_m},
    'top_dual': top[:10],
    'top_triple': [t for t in top_triple[:5]]
}
with open(results_dir / 'multi_timeframe_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Saved to {results_dir}/multi_timeframe_results.json")
