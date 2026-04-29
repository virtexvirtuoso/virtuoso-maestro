#!/usr/bin/env python3
"""
ULTRATHINK: Deep Analysis of 356 Scalp Signals

Questions to answer:
1. Is the 66.7% win rate real or misleading? (time_stop != real win)
2. What's the ACTUAL expectancy after fees?
3. Which filters would turn this from break-even to profitable?
4. Does edge score predict outcomes?
5. Does confidence/base_score predict outcomes?
6. Time-of-day patterns? (Asian/EU/US)
7. Trigger type analysis (vol spike, funding window, prime hours)
8. Symbol-level alpha — who's generating vs destroying value?
9. Does range_position predict anything? (overbought/oversold)
10. Consecutive win/loss patterns — is there momentum?
11. What's the optimal signal to STOP sending? (expected value filter)
12. Would VPIN thresholds have filtered bad trades?
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load signals
signals = []
with open('/tmp/scalp_signals.jsonl') as f:
    for line in f:
        signals.append(json.loads(line))

df = pd.DataFrame(signals)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek
df['date'] = df['timestamp'].dt.date

# Exclude pending
resolved = df[df['outcome'] != 'pending'].copy()
print(f"Total: {len(df)}, Resolved: {len(resolved)}, Pending: {len(df)-len(resolved)}")
print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

###############################################################################
# 1. WIN RATE REALITY CHECK
###############################################################################
print("\n" + "="*70)
print("1. WIN RATE REALITY CHECK")
print("="*70)

outcome_counts = resolved['outcome'].value_counts()
print(f"\nOutcome distribution:")
for o, c in outcome_counts.items():
    pct = c / len(resolved) * 100
    avg_pnl = resolved[resolved['outcome']==o]['actual_pnl_pct'].mean()
    print(f"  {o:>12}: {c:>4} ({pct:.1f}%)  avg P&L: {avg_pnl:+.4f}%")

# Real win = positive PnL, not just TP hit
has_pnl = resolved[resolved['actual_pnl_pct'].notna()]
real_wins = (has_pnl['actual_pnl_pct'] > 0).sum()
real_losses = (has_pnl['actual_pnl_pct'] <= 0).sum()
print(f"\nReal win rate (PnL > 0): {real_wins}/{len(has_pnl)} = {real_wins/len(has_pnl)*100:.1f}%")
print(f"Avg winning trade: {has_pnl[has_pnl['actual_pnl_pct']>0]['actual_pnl_pct'].mean():+.4f}%")
print(f"Avg losing trade:  {has_pnl[has_pnl['actual_pnl_pct']<=0]['actual_pnl_pct'].mean():+.4f}%")

# Profit factor
gross_win = has_pnl[has_pnl['actual_pnl_pct'] > 0]['actual_pnl_pct'].sum()
gross_loss = abs(has_pnl[has_pnl['actual_pnl_pct'] <= 0]['actual_pnl_pct'].sum())
pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
print(f"Profit factor (gross): {pf:.2f}")

###############################################################################
# 2. AFTER FEES
###############################################################################
print("\n" + "="*70)
print("2. AFTER FEES (realistic Bybit VIP0)")
print("="*70)

for fee_label, fee_bps in [('0bps (gross)', 0), ('4bps (limit+limit)', 4), 
                             ('7.5bps (limit+market)', 7.5), ('11bps (market+market)', 11)]:
    fee_pct = fee_bps / 100
    net_pnls = has_pnl['actual_pnl_pct'] - fee_pct
    total = net_pnls.sum()
    avg = net_pnls.mean()
    wr = (net_pnls > 0).mean() * 100
    sharpe = net_pnls.mean() / (net_pnls.std() + 1e-10) * np.sqrt(356/30*365)  # annualized
    print(f"  {fee_label:>25}: total={total:+.2f}%, avg={avg:+.4f}%, WR={wr:.1f}%, Sharpe={sharpe:.2f}")

###############################################################################
# 3. EDGE SCORE ANALYSIS
###############################################################################
print("\n" + "="*70)
print("3. EDGE SCORE AS PREDICTOR")
print("="*70)

# Edge = abs(long_score - short_score)
for col, label in [('edge', 'Edge (L-S gap)'), ('base_score', 'Base Score'), ('confidence', 'Confidence')]:
    if col not in has_pnl.columns or has_pnl[col].isna().all():
        continue
    
    # Quintile analysis
    try:
        has_pnl_copy = has_pnl.copy()
        has_pnl_copy['q'] = pd.qcut(has_pnl_copy[col], 5, labels=False, duplicates='drop')
        print(f"\n{label} quintiles:")
        for q in sorted(has_pnl_copy['q'].unique()):
            qdf = has_pnl_copy[has_pnl_copy['q'] == q]
            wr = (qdf['actual_pnl_pct'] > 0).mean() * 100
            avg = qdf['actual_pnl_pct'].mean()
            range_str = f"{qdf[col].min():.1f}-{qdf[col].max():.1f}"
            print(f"  Q{q} ({range_str:>12}): {len(qdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%")
    except Exception as e:
        print(f"  {label}: insufficient variation ({e})")

###############################################################################
# 4. TIME-OF-DAY ANALYSIS
###############################################################################
print("\n" + "="*70)
print("4. TIME-OF-DAY ANALYSIS (UTC)")
print("="*70)

sessions = {
    'Asian (0-8)': range(0, 8),
    'EU (8-14)': range(8, 14),
    'US (14-22)': range(14, 22),
    'Late (22-24)': range(22, 24),
}

for sname, hours in sessions.items():
    sdf = has_pnl[has_pnl['hour'].isin(hours)]
    if len(sdf) == 0:
        continue
    wr = (sdf['actual_pnl_pct'] > 0).mean() * 100
    avg = sdf['actual_pnl_pct'].mean()
    total = sdf['actual_pnl_pct'].sum()
    print(f"  {sname:>15}: {len(sdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%, total={total:+.3f}%")

# Hour-by-hour
print("\n  Hour-by-hour (best/worst):")
hourly = has_pnl.groupby('hour').agg(
    count=('actual_pnl_pct', 'count'),
    wr=('actual_pnl_pct', lambda x: (x>0).mean()*100),
    avg_pnl=('actual_pnl_pct', 'mean'),
    total_pnl=('actual_pnl_pct', 'sum')
).sort_values('total_pnl', ascending=False)

for h in hourly.head(5).index:
    r = hourly.loc[h]
    print(f"    {h:02d}:00 UTC: {r['count']:.0f} signals, WR={r['wr']:.0f}%, total={r['total_pnl']:+.3f}%")
print("    ...")
for h in hourly.tail(3).index:
    r = hourly.loc[h]
    print(f"    {h:02d}:00 UTC: {r['count']:.0f} signals, WR={r['wr']:.0f}%, total={r['total_pnl']:+.3f}%")

###############################################################################
# 5. TRIGGER TYPE ANALYSIS
###############################################################################
print("\n" + "="*70)
print("5. TRIGGER TYPE ANALYSIS")
print("="*70)

# Parse trigger types
def classify_trigger(reason):
    if not reason:
        return 'UNKNOWN'
    reason = reason.upper()
    if 'VOLAT' in reason:
        return 'VOLATILITY'
    elif 'FUND' in reason:
        return 'FUNDING'
    elif 'VOLUME' in reason or 'SPIKE' in reason:
        return 'VOLUME_SPIKE'
    elif 'PRIME' in reason or 'HOUR' in reason:
        return 'PRIME_HOURS'
    elif 'CORRELA' in reason:
        return 'CORRELATION'
    else:
        return 'OTHER'

has_pnl_t = has_pnl.copy()
has_pnl_t['trigger_type'] = has_pnl_t['trigger_reason'].apply(classify_trigger)

for tt in has_pnl_t['trigger_type'].value_counts().index:
    tdf = has_pnl_t[has_pnl_t['trigger_type'] == tt]
    wr = (tdf['actual_pnl_pct'] > 0).mean() * 100
    avg = tdf['actual_pnl_pct'].mean()
    total = tdf['actual_pnl_pct'].sum()
    print(f"  {tt:>15}: {len(tdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%, total={total:+.3f}%")

###############################################################################
# 6. SYMBOL-LEVEL DEEP DIVE
###############################################################################
print("\n" + "="*70)
print("6. SYMBOL ALPHA (sorted by total P&L)")
print("="*70)

sym_stats = has_pnl.groupby('symbol').agg(
    count=('actual_pnl_pct', 'count'),
    wr=('actual_pnl_pct', lambda x: (x>0).mean()*100),
    avg_pnl=('actual_pnl_pct', 'mean'),
    total_pnl=('actual_pnl_pct', 'sum'),
    avg_edge=('edge', 'mean'),
    avg_score=('base_score', 'mean'),
    avg_time=('time_to_outcome_minutes', 'mean'),
).sort_values('total_pnl', ascending=False)

print(f"{'Symbol':<20} {'N':>4} {'WR':>5} {'Avg PnL':>9} {'Total':>8} {'AvgEdge':>8} {'AvgTime':>8}")
print("-"*70)
for sym in sym_stats.index:
    r = sym_stats.loc[sym]
    sym_short = sym.replace('/USDT:USDT', '')
    print(f"{sym_short:<20} {r['count']:>4.0f} {r['wr']:>4.0f}% {r['avg_pnl']:>+8.4f}% {r['total_pnl']:>+7.3f}% {r['avg_edge']:>7.1f} {r['avg_time']:>7.1f}m")

# Value destroyers vs creators
creators = sym_stats[sym_stats['total_pnl'] > 0]
destroyers = sym_stats[sym_stats['total_pnl'] <= 0]
print(f"\nValue creators: {len(creators)} symbols, total={creators['total_pnl'].sum():+.3f}%")
print(f"Value destroyers: {len(destroyers)} symbols, total={destroyers['total_pnl'].sum():+.3f}%")

###############################################################################
# 7. DIRECTION ANALYSIS
###############################################################################
print("\n" + "="*70)
print("7. DIRECTION ANALYSIS")
print("="*70)

for direction in ['LONG', 'SHORT']:
    ddf = has_pnl[has_pnl['direction'] == direction]
    wr = (ddf['actual_pnl_pct'] > 0).mean() * 100
    avg = ddf['actual_pnl_pct'].mean()
    total = ddf['actual_pnl_pct'].sum()
    print(f"  {direction:>6}: {len(ddf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%, total={total:+.3f}%")

# LONG vs SHORT by session
print("\n  Direction × Session:")
for direction in ['LONG', 'SHORT']:
    for sname, hours in sessions.items():
        sdf = has_pnl[(has_pnl['direction']==direction) & (has_pnl['hour'].isin(hours))]
        if len(sdf) < 5:
            continue
        wr = (sdf['actual_pnl_pct'] > 0).mean() * 100
        avg = sdf['actual_pnl_pct'].mean()
        print(f"    {direction} {sname:>15}: {len(sdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%")

###############################################################################
# 8. RANGE POSITION ANALYSIS
###############################################################################
print("\n" + "="*70)
print("8. RANGE POSITION (overbought/oversold at signal)")
print("="*70)

rp = has_pnl[has_pnl['range_position_pct'].notna()].copy()
if len(rp) > 20:
    rp['rp_q'] = pd.qcut(rp['range_position_pct'], 4, labels=['Low', 'MidLow', 'MidHigh', 'High'], duplicates='drop')
    for q in ['Low', 'MidLow', 'MidHigh', 'High']:
        qdf = rp[rp['rp_q'] == q]
        if len(qdf) == 0:
            continue
        wr = (qdf['actual_pnl_pct'] > 0).mean() * 100
        avg = qdf['actual_pnl_pct'].mean()
        rp_range = f"{qdf['range_position_pct'].min():.2f}-{qdf['range_position_pct'].max():.2f}"
        print(f"  {q:>8} ({rp_range}): {len(qdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%")

    # Contrarian check: LONG at low range_position, SHORT at high
    rp_long_low = rp[(rp['direction']=='LONG') & (rp['range_position_pct'] < 0.3)]
    rp_short_high = rp[(rp['direction']=='SHORT') & (rp['range_position_pct'] > 0.7)]
    rp_long_high = rp[(rp['direction']=='LONG') & (rp['range_position_pct'] > 0.7)]
    rp_short_low = rp[(rp['direction']=='SHORT') & (rp['range_position_pct'] < 0.3)]
    
    print(f"\n  Contrarian signals:")
    for label, subset in [('LONG at low (contrarian)', rp_long_low), ('SHORT at high (contrarian)', rp_short_high),
                           ('LONG at high (momentum)', rp_long_high), ('SHORT at low (momentum)', rp_short_low)]:
        if len(subset) < 3:
            continue
        wr = (subset['actual_pnl_pct'] > 0).mean() * 100
        avg = subset['actual_pnl_pct'].mean()
        print(f"    {label}: {len(subset)} signals, WR={wr:.0f}%, avg={avg:+.4f}%")

###############################################################################
# 9. VOLATILITY & SPREAD FILTERS
###############################################################################
print("\n" + "="*70)
print("9. VOLATILITY & SPREAD FILTERING")
print("="*70)

for vol_max in [2.0, 3.0, 5.0, 10.0, 20.0]:
    fdf = has_pnl[has_pnl['volatility_pct'] <= vol_max]
    if len(fdf) < 10:
        continue
    wr = (fdf['actual_pnl_pct'] > 0).mean() * 100
    avg = fdf['actual_pnl_pct'].mean()
    total = fdf['actual_pnl_pct'].sum()
    print(f"  Vol <= {vol_max:>5.1f}%: {len(fdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%, total={total:+.3f}%")

for spread_max in [0.01, 0.02, 0.03, 0.05, 0.10]:
    fdf = has_pnl[has_pnl['spread_pct'] <= spread_max]
    if len(fdf) < 10:
        continue
    wr = (fdf['actual_pnl_pct'] > 0).mean() * 100
    avg = fdf['actual_pnl_pct'].mean()
    total = fdf['actual_pnl_pct'].sum()
    print(f"  Spread <= {spread_max:>5.2f}%: {len(fdf):>3} signals, WR={wr:.0f}%, avg={avg:+.4f}%, total={total:+.3f}%")

###############################################################################
# 10. CONSECUTIVE PATTERNS & STREAKS
###############################################################################
print("\n" + "="*70)
print("10. STREAK ANALYSIS")
print("="*70)

wins = (has_pnl['actual_pnl_pct'] > 0).values
max_win_streak = max_loss_streak = cur_win = cur_loss = 0
for w in wins:
    if w:
        cur_win += 1; cur_loss = 0
        max_win_streak = max(max_win_streak, cur_win)
    else:
        cur_loss += 1; cur_win = 0
        max_loss_streak = max(max_loss_streak, cur_loss)

print(f"  Max win streak: {max_win_streak}")
print(f"  Max loss streak: {max_loss_streak}")

# After a loss, does next trade improve?
pnls = has_pnl['actual_pnl_pct'].values
after_win = [pnls[i] for i in range(1, len(pnls)) if pnls[i-1] > 0]
after_loss = [pnls[i] for i in range(1, len(pnls)) if pnls[i-1] <= 0]
print(f"  After win:  avg next = {np.mean(after_win):+.4f}% ({len(after_win)} trades)")
print(f"  After loss: avg next = {np.mean(after_loss):+.4f}% ({len(after_loss)} trades)")

###############################################################################
# 11. DAILY P&L CURVE
###############################################################################
print("\n" + "="*70)
print("11. DAILY P&L CURVE")
print("="*70)

daily = has_pnl.groupby('date').agg(
    count=('actual_pnl_pct', 'count'),
    total_pnl=('actual_pnl_pct', 'sum'),
    wr=('actual_pnl_pct', lambda x: (x>0).mean()*100),
).sort_index()

cum_pnl = daily['total_pnl'].cumsum()
pos_days = (daily['total_pnl'] > 0).sum()
neg_days = (daily['total_pnl'] <= 0).sum()
print(f"  Green days: {pos_days}, Red days: {neg_days} ({pos_days/(pos_days+neg_days)*100:.0f}% green)")
print(f"  Best day: {daily['total_pnl'].max():+.3f}%")
print(f"  Worst day: {daily['total_pnl'].min():+.3f}%")
print(f"  Avg signals/day: {daily['count'].mean():.1f}")

# Max drawdown
peak = cum_pnl.cummax()
dd = cum_pnl - peak
max_dd = dd.min()
print(f"  Max drawdown: {max_dd:.3f}%")
print(f"  Final cumulative: {cum_pnl.iloc[-1]:+.3f}%")

# Weekly
weekly = has_pnl.copy()
weekly['week'] = weekly['timestamp'].dt.strftime('%G-W%V')
wk_pnl = weekly.groupby('week')['actual_pnl_pct'].agg(['count', 'sum', lambda x: (x>0).mean()*100])
wk_pnl.columns = ['count', 'total_pnl', 'wr']
print(f"\n  Weekly P&L:")
for wk in wk_pnl.index:
    r = wk_pnl.loc[wk]
    print(f"    {wk}: {r['count']:.0f} signals, P&L={r['total_pnl']:+.3f}%, WR={r['wr']:.0f}%")

###############################################################################
# 12. OPTIMAL FILTERS (what to cut)
###############################################################################
print("\n" + "="*70)
print("12. OPTIMAL FILTERS — WHAT TO CUT")
print("="*70)

# Test various filter combinations
filters = {
    'No filter': has_pnl,
    'Drop XRP': has_pnl[~has_pnl['symbol'].str.contains('XRP')],
    'Drop XRP+DOGE': has_pnl[~has_pnl['symbol'].str.contains('XRP|DOGE')],
    'Only top 5 symbols': has_pnl[has_pnl['symbol'].isin(sym_stats.head(5).index)],
    'Edge > 3': has_pnl[has_pnl['edge'] > 3],
    'Edge > 5': has_pnl[has_pnl['edge'] > 5],
    'Score > 75': has_pnl[has_pnl['base_score'] > 75],
    'Score > 80': has_pnl[has_pnl['base_score'] > 80],
    'Not Asian session': has_pnl[~has_pnl['hour'].isin(range(0,8))],
    'EU+US only': has_pnl[has_pnl['hour'].isin(range(8,22))],
    'Vol < 5%': has_pnl[has_pnl['volatility_pct'] < 5],
    'Vol 2-10%': has_pnl[(has_pnl['volatility_pct'] >= 2) & (has_pnl['volatility_pct'] <= 10)],
    'Spread < 0.03': has_pnl[has_pnl['spread_pct'] < 0.03],
    'LONG only': has_pnl[has_pnl['direction'] == 'LONG'],
    'SHORT only': has_pnl[has_pnl['direction'] == 'SHORT'],
}

# Add combo filters
combo1 = has_pnl[(has_pnl['edge'] > 3) & (~has_pnl['symbol'].str.contains('XRP'))]
combo2 = has_pnl[(has_pnl['base_score'] > 75) & (has_pnl['spread_pct'] < 0.03)]
combo3 = has_pnl[(has_pnl['edge'] > 3) & (~has_pnl['symbol'].str.contains('XRP')) & (has_pnl['hour'].isin(range(8,22)))]
combo4 = has_pnl[(has_pnl['base_score'] > 75) & (~has_pnl['symbol'].str.contains('XRP|DOGE')) & (has_pnl['spread_pct'] < 0.05)]
filters['Combo: Edge>3 + no XRP'] = combo1
filters['Combo: Score>75 + spread<0.03'] = combo2
filters['Combo: Edge>3 + no XRP + EU/US'] = combo3
filters['Combo: Score>75 + no XRP/DOGE + spread<0.05'] = combo4

fee_pct = 0.075  # 7.5bps realistic

print(f"\n{'Filter':<45} {'N':>4} {'WR':>5} {'Gross':>7} {'Net':>7} {'NetAvg':>8}")
print("-"*80)
for name, fdf in sorted(filters.items(), key=lambda x: (x[1]['actual_pnl_pct'] - fee_pct).sum() if len(x[1])>0 else -999, reverse=True):
    if len(fdf) < 5:
        continue
    wr = (fdf['actual_pnl_pct'] > 0).mean() * 100
    gross = fdf['actual_pnl_pct'].sum()
    net = (fdf['actual_pnl_pct'] - fee_pct).sum()
    net_avg = (fdf['actual_pnl_pct'] - fee_pct).mean()
    print(f"{name:<45} {len(fdf):>4} {wr:>4.0f}% {gross:>+6.2f}% {net:>+6.2f}% {net_avg:>+7.4f}%")

###############################################################################
# 13. VERSION ANALYSIS (v5 vs older if any)
###############################################################################
print("\n" + "="*70)
print("13. VERSION & ATR ANALYSIS")
print("="*70)

if 'version' in has_pnl.columns:
    for v in has_pnl['version'].unique():
        vdf = has_pnl[has_pnl['version'] == v]
        wr = (vdf['actual_pnl_pct'] > 0).mean() * 100
        avg = vdf['actual_pnl_pct'].mean()
        print(f"  {v}: {len(vdf)} signals, WR={wr:.0f}%, avg={avg:+.4f}%")

if 'atr_sl_pct' in has_pnl.columns:
    atr_data = has_pnl[has_pnl['atr_sl_pct'].notna()]
    if len(atr_data) > 0:
        print(f"\n  ATR-based stops:")
        print(f"    Avg SL%: {atr_data['atr_sl_pct'].mean():.3f}%")
        print(f"    Avg TP%: {atr_data['atr_tp_pct'].mean():.3f}%")
        print(f"    Avg R:R: {(atr_data['atr_tp_pct']/atr_data['atr_sl_pct']).mean():.2f}")
        
        # Does wider ATR help?
        for q_label, q_range in [('Tight ATR', (0, 0.5)), ('Wide ATR', (0.5, 1.0))]:
            qdf = atr_data[(atr_data['atr_sl_pct'] >= atr_data['atr_sl_pct'].quantile(q_range[0])) & 
                           (atr_data['atr_sl_pct'] <= atr_data['atr_sl_pct'].quantile(q_range[1]))]
            if len(qdf) > 5:
                wr = (qdf['actual_pnl_pct'] > 0).mean() * 100
                avg = qdf['actual_pnl_pct'].mean()
                print(f"    {q_label}: {len(qdf)} signals, WR={wr:.0f}%, avg={avg:+.4f}%")

###############################################################################
# 14. TIME-TO-OUTCOME ANALYSIS
###############################################################################
print("\n" + "="*70)
print("14. TIME-TO-OUTCOME ANALYSIS")
print("="*70)

tto = has_pnl[has_pnl['time_to_outcome_minutes'].notna()]
for outcome in ['hit_tp', 'hit_sl', 'time_stop', 'expired']:
    odf = tto[tto['outcome'] == outcome]
    if len(odf) == 0:
        continue
    avg_t = odf['time_to_outcome_minutes'].mean()
    med_t = odf['time_to_outcome_minutes'].median()
    print(f"  {outcome:>10}: avg={avg_t:.1f}min, median={med_t:.1f}min")

###############################################################################
# SUMMARY
###############################################################################
print("\n" + "="*70)
print("ULTRATHINK SUMMARY")
print("="*70)
print(f"""
Key findings from {len(has_pnl)} resolved signals over {(df['timestamp'].max() - df['timestamp'].min()).days} days:

1. GROSS: +{has_pnl['actual_pnl_pct'].sum():.2f}% | NET (7.5bps): {(has_pnl['actual_pnl_pct'] - 0.075).sum():+.2f}%
2. Real expectancy per trade: {has_pnl['actual_pnl_pct'].mean():+.4f}% gross, {(has_pnl['actual_pnl_pct'].mean() - 0.075):+.4f}% net
3. Best filter found: see ranking above
""")
