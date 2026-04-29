#!/usr/bin/env python3
"""
ULTRATHINK: Regime Strategy Switching — What Actually Works?

Key insight from paper trader:
- TF/MR correlation = -0.80 (7 weeks) / -0.46 (217 weeks)
- Session filter turned losers into winners (no prediction needed)
- Walk-forward regime detection = coin flip (53%)

Questions to answer:
1. Does session-based TF/MR switching work over 4 years?
2. Can we combine session filter + regime detection?
3. What about adaptive allocation (not binary switch)?
4. Is there a LAGGING indicator that works? (regime just ended → switch)
5. What about using REALIZED TF/MR performance as the signal? (momentum of strategy returns)
6. Multi-timeframe regime: daily ADX for weekly strategy choice?
7. What if the real edge is just "run MR in Asian, TF in EU/US" with hard stops?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# Load BTC 1h candles
###############################################################################
df = pd.read_csv(Path.home() / "Desktop/maestro/data/spot/1h/BTC_spot_1h.csv")
df['timestamp'] = pd.to_datetime(df['Date'])
df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)
df = df.sort_values('timestamp').set_index('timestamp')
print(f"BTC 1h: {len(df)} rows, {df.index[0]} to {df.index[-1]}")

###############################################################################
# Helper: compute ADX
###############################################################################
def compute_adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(span=period, adjust=False).mean()

###############################################################################
# Simulate TF and MR strategies on hourly bars
###############################################################################

# TF: EMA(10) vs EMA(30) crossover. Long when fast > slow, short when fast < slow.
df['ema10'] = df['close'].ewm(span=10).mean()
df['ema30'] = df['close'].ewm(span=30).mean()
df['ret'] = df['close'].pct_change()
df['tf_signal'] = np.where(df['ema10'] > df['ema30'], 1, -1)
df['tf_ret'] = df['tf_signal'].shift(1) * df['ret']

# MR: Bollinger Band mean reversion. Short above upper, long below lower, flat in middle.
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

mr_signal = pd.Series(0, index=df.index)
pos = 0
for i in range(1, len(df)):
    if df['close'].iloc[i] > df['bb_upper'].iloc[i] and not pd.isna(df['bb_upper'].iloc[i]):
        pos = -1  # short
    elif df['close'].iloc[i] < df['bb_lower'].iloc[i] and not pd.isna(df['bb_lower'].iloc[i]):
        pos = 1   # long
    elif pos == 1 and df['close'].iloc[i] > df['bb_mid'].iloc[i]:
        pos = 0   # exit long at mid
    elif pos == -1 and df['close'].iloc[i] < df['bb_mid'].iloc[i]:
        pos = 0   # exit short at mid
    mr_signal.iloc[i] = pos

df['mr_signal'] = mr_signal
df['mr_ret'] = df['mr_signal'].shift(1) * df['ret']

# Apply transaction costs (7.5bps round-trip per signal change)
cost_per_trade = 0.00075  # 7.5bps
df['tf_trades'] = df['tf_signal'].diff().abs() / 2  # 0 or 1
df['mr_trades'] = df['mr_signal'].diff().abs().clip(upper=1)
df['tf_ret_net'] = df['tf_ret'] - df['tf_trades'] * cost_per_trade
df['mr_ret_net'] = df['mr_ret'] - df['mr_trades'] * cost_per_trade

# Drop warmup
df = df.iloc[100:].copy()

###############################################################################
# TEST 1: Session-based switching over full history
###############################################################################
print("\n" + "="*70)
print("TEST 1: SESSION-BASED SWITCHING (deterministic, no prediction)")
print("="*70)

df['hour'] = df.index.hour

# Strategy: MR in Asian (0-8 UTC), TF in EU/US (8-22 UTC), MR in Late (22-24)
# Also test: TF in EU/US only, flat otherwise
# Also test: MR always, but flat during EU session

strategies = {
    'Always TF': df['tf_ret_net'],
    'Always MR': df['mr_ret_net'],
    '50/50 blend': (df['tf_ret_net'] + df['mr_ret_net']) / 2,
    'TF EU+US / MR Asian+Late': pd.Series(0.0, index=df.index),
    'TF EU+US only (flat else)': pd.Series(0.0, index=df.index),
    'MR Asian only (flat else)': pd.Series(0.0, index=df.index),
    'MR everywhere except EU': pd.Series(0.0, index=df.index),
    'Contrarian TF Asian + normal TF rest': pd.Series(0.0, index=df.index),
}

for i in df.index:
    h = i.hour
    asian = 0 <= h < 8
    eu = 8 <= h < 14
    us = 14 <= h < 22
    late = 22 <= h < 24
    
    # TF EU+US / MR Asian+Late
    if eu or us:
        strategies['TF EU+US / MR Asian+Late'].loc[i] = df.loc[i, 'tf_ret_net']
    else:
        strategies['TF EU+US / MR Asian+Late'].loc[i] = df.loc[i, 'mr_ret_net']
    
    # TF EU+US only
    if eu or us:
        strategies['TF EU+US only (flat else)'].loc[i] = df.loc[i, 'tf_ret_net']
    
    # MR Asian only
    if asian:
        strategies['MR Asian only (flat else)'].loc[i] = df.loc[i, 'mr_ret_net']
    
    # MR everywhere except EU
    if not eu:
        strategies['MR everywhere except EU'].loc[i] = df.loc[i, 'mr_ret_net']
    
    # Contrarian TF Asian
    if asian:
        strategies['Contrarian TF Asian + normal TF rest'].loc[i] = -df.loc[i, 'tf_ret_net']
    else:
        strategies['Contrarian TF Asian + normal TF rest'].loc[i] = df.loc[i, 'tf_ret_net']

print(f"\n{'Strategy':<45} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8}")
print("-" * 85)

for name, rets in strategies.items():
    rets = rets.dropna()
    ann_ret = rets.mean() * 8760 * 100  # hourly -> annual
    ann_vol = rets.std() * np.sqrt(8760) * 100
    sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(8760)
    
    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min() * 100
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    
    print(f"{name:<45} {ann_ret:>7.1f}% {ann_vol:>7.1f}% {sharpe:>7.2f} {mdd:>7.1f}% {calmar:>7.2f}")

###############################################################################
# TEST 2: Lagging regime — use LAST WEEK's winner to predict NEXT WEEK
###############################################################################
print("\n" + "="*70)
print("TEST 2: LAGGING REGIME (last week's winner predicts next)")
print("="*70)

# Weekly returns
tf_weekly = df['tf_ret_net'].resample('W').sum()
mr_weekly = df['mr_ret_net'].resample('W').sum()
common = tf_weekly.index.intersection(mr_weekly.index)
tf_weekly = tf_weekly.loc[common]
mr_weekly = mr_weekly.loc[common]

# Strategy momentum: if TF won last week, use TF this week (momentum)
# Anti-momentum: if TF won last week, use MR this week (mean reversion of strategy returns)
momentum_pnl = []
anti_momentum_pnl = []
momentum_2wk_pnl = []

for i in range(2, len(common)):
    last_winner = 'TF' if tf_weekly.iloc[i-1] > mr_weekly.iloc[i-1] else 'MR'
    last_2wk_tf = tf_weekly.iloc[i-2:i].sum()
    last_2wk_mr = mr_weekly.iloc[i-2:i].sum()
    last_2wk_winner = 'TF' if last_2wk_tf > last_2wk_mr else 'MR'
    
    # Momentum: same as last week
    if last_winner == 'TF':
        momentum_pnl.append(tf_weekly.iloc[i])
    else:
        momentum_pnl.append(mr_weekly.iloc[i])
    
    # Anti-momentum: opposite of last week
    if last_winner == 'TF':
        anti_momentum_pnl.append(mr_weekly.iloc[i])
    else:
        anti_momentum_pnl.append(tf_weekly.iloc[i])
    
    # 2-week momentum
    if last_2wk_winner == 'TF':
        momentum_2wk_pnl.append(tf_weekly.iloc[i])
    else:
        momentum_2wk_pnl.append(mr_weekly.iloc[i])

for name, pnls in [('Strategy momentum (1wk)', momentum_pnl),
                     ('Anti-momentum (1wk)', anti_momentum_pnl),
                     ('Strategy momentum (2wk)', momentum_2wk_pnl)]:
    pnls = np.array(pnls)
    sharpe = pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(52)
    total = pnls.sum() * 100
    win_rate = (pnls > 0).mean() * 100
    print(f"{name}: Sharpe={sharpe:.2f}, total={total:.1f}%, WR={win_rate:.0f}%")

# Baselines
for name, weekly in [('Always TF', tf_weekly), ('Always MR', mr_weekly)]:
    sharpe = weekly.mean() / (weekly.std() + 1e-10) * np.sqrt(52)
    print(f"{name}: Sharpe={sharpe:.2f}")

###############################################################################
# TEST 3: Adaptive allocation (Kelly-like) based on rolling strategy performance
###############################################################################
print("\n" + "="*70)
print("TEST 3: ADAPTIVE ALLOCATION (rolling Sharpe weighting)")
print("="*70)

for lookback in [4, 8, 12, 26]:
    adaptive_pnl = []
    for i in range(lookback, len(common)):
        # Rolling Sharpe for each strategy
        tf_window = tf_weekly.iloc[i-lookback:i]
        mr_window = mr_weekly.iloc[i-lookback:i]
        
        tf_sharpe = tf_window.mean() / (tf_window.std() + 1e-10)
        mr_sharpe = mr_window.mean() / (mr_window.std() + 1e-10)
        
        # Softmax allocation
        exp_tf = np.exp(tf_sharpe)
        exp_mr = np.exp(mr_sharpe)
        w_tf = exp_tf / (exp_tf + exp_mr)
        w_mr = 1 - w_tf
        
        blended = w_tf * tf_weekly.iloc[i] + w_mr * mr_weekly.iloc[i]
        adaptive_pnl.append(blended)
    
    pnls = np.array(adaptive_pnl)
    sharpe = pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(52)
    total = pnls.sum() * 100
    print(f"Lookback {lookback:>2}wk: Sharpe={sharpe:.2f}, total={total:.1f}%")

###############################################################################
# TEST 4: Regime from DAILY ADX for WEEKLY strategy choice
###############################################################################
print("\n" + "="*70)
print("TEST 4: DAILY ADX FOR WEEKLY STRATEGY SWITCH (walk-forward)")
print("="*70)

# Resample to daily
daily_high = df['high'].resample('1D').max()
daily_low = df['low'].resample('1D').min()
daily_close = df['close'].resample('1D').last()
daily_adx = compute_adx(daily_high, daily_low, daily_close, 14).dropna()

# Friday ADX -> next week's strategy
# Also try: ADX slope (rising ADX = trending, falling = ranging)
daily_adx_df = pd.DataFrame({'adx': daily_adx})
daily_adx_df['adx_slope'] = daily_adx_df['adx'].diff(5)  # 5-day change in ADX
daily_adx_df['adx_ma'] = daily_adx_df['adx'].rolling(20).mean()
daily_adx_df['adx_vs_ma'] = daily_adx_df['adx'] - daily_adx_df['adx_ma']

# Get Friday's reading for each week
weekly_adx = daily_adx_df.resample('W').last().dropna()

# Walk-forward: train on 20 weeks, predict next
for indicator, label in [('adx', 'ADX level'), ('adx_slope', 'ADX slope'), ('adx_vs_ma', 'ADX vs MA')]:
    correct = 0
    total = 0
    wf_pnls = []
    
    valid = weekly_adx[indicator].dropna()
    valid_idx = valid.index.intersection(common)
    valid = valid.loc[valid_idx]
    gt = pd.Series(['TF' if tf_weekly.loc[w] > mr_weekly.loc[w] else 'MR' for w in valid_idx], index=valid_idx)
    
    train_win = 20
    for i in range(train_win, len(valid_idx)):
        train_v = [valid.iloc[j] for j in range(i-train_win, i)]
        train_g = [gt.iloc[j] for j in range(i-train_win, i)]
        
        # Find best threshold
        best_acc = 0
        best_pred = 'MR'
        for pctile in [20, 30, 40, 50, 60, 70, 80]:
            thresh = np.percentile(train_v, pctile)
            for d in ['TF', 'MR']:
                preds = [d if v > thresh else ('MR' if d=='TF' else 'TF') for v in train_v]
                acc = sum(p == g for p, g in zip(preds, train_g)) / len(train_g)
                if acc > best_acc:
                    best_acc = acc
                    _t, _d = thresh, d
                    best_pred = _d if valid.iloc[i] > _t else ('MR' if _d=='TF' else 'TF')
        
        actual = gt.iloc[i]
        if best_pred == actual:
            correct += 1
        total += 1
        wf_pnls.append(tf_weekly.loc[valid_idx[i]] if best_pred == 'TF' else mr_weekly.loc[valid_idx[i]])
    
    if total > 0:
        acc = correct / total
        sharpe = np.mean(wf_pnls) / (np.std(wf_pnls) + 1e-10) * np.sqrt(52)
        print(f"{label}: WF acc={acc:.0%} ({correct}/{total}), Sharpe={sharpe:.2f}")

###############################################################################
# TEST 5: Volatility regime (realized vol percentile)
###############################################################################
print("\n" + "="*70)
print("TEST 5: VOLATILITY REGIME (percentile-based)")
print("="*70)

daily_ret = daily_close.pct_change()
daily_vol = daily_ret.rolling(5).std()
vol_pctile = daily_vol.rolling(252).rank(pct=True)  # Percentile over 1 year

weekly_vol_pctile = vol_pctile.resample('W').last().dropna()
valid_idx = weekly_vol_pctile.index.intersection(common)

# Simple rule: low vol percentile (<30%) = range-bound = MR. High vol = trending = TF
for thresh_pct in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pnls = []
    for wk in valid_idx:
        vp = weekly_vol_pctile.loc[wk]
        if vp > thresh_pct:
            pnls.append(tf_weekly.loc[wk])  # High vol = trending
        else:
            pnls.append(mr_weekly.loc[wk])  # Low vol = ranging
    pnls = np.array(pnls)
    sharpe = pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(52)
    win_rate = (pnls > 0).mean() * 100
    
    # Reverse direction too
    pnls_r = []
    for wk in valid_idx:
        vp = weekly_vol_pctile.loc[wk]
        if vp > thresh_pct:
            pnls_r.append(mr_weekly.loc[wk])
        else:
            pnls_r.append(tf_weekly.loc[wk])
    sharpe_r = np.mean(pnls_r) / (np.std(pnls_r) + 1e-10) * np.sqrt(52)
    
    best = max(sharpe, sharpe_r)
    direction = "high→TF" if sharpe > sharpe_r else "high→MR"
    print(f"  thresh={thresh_pct:.0%}: {direction}, Sharpe={best:.2f}, WR={(np.array(pnls if sharpe>sharpe_r else pnls_r)>0).mean()*100:.0f}%")

###############################################################################
# TEST 6: Day-of-week effect
###############################################################################
print("\n" + "="*70)
print("TEST 6: DAY-OF-WEEK EFFECT")
print("="*70)

df['dow'] = df.index.dayofweek  # 0=Mon, 6=Sun

for dow_name, dow_num in [('Mon',0),('Tue',1),('Wed',2),('Thu',3),('Fri',4),('Sat',5),('Sun',6)]:
    mask = df['dow'] == dow_num
    tf_d = df.loc[mask, 'tf_ret_net']
    mr_d = df.loc[mask, 'mr_ret_net']
    tf_sr = tf_d.mean() / (tf_d.std() + 1e-10) * np.sqrt(365)
    mr_sr = mr_d.mean() / (mr_d.std() + 1e-10) * np.sqrt(365)
    better = "TF" if tf_sr > mr_sr else "MR"
    print(f"  {dow_name}: TF Sharpe={tf_sr:.2f}, MR Sharpe={mr_sr:.2f} → {better}")

###############################################################################
# TEST 7: Combined session + day-of-week + hard stops
###############################################################################
print("\n" + "="*70)
print("TEST 7: COMBINED SESSION + MR HARD STOPS")  
print("="*70)

# MR with hard stop: if position loses more than X bps in Y bars, exit
for stop_bars, stop_bps in [(6, 30), (12, 50), (24, 75), (6, 50), (12, 30)]:
    # Re-simulate MR with stops
    mr_stopped = pd.Series(0.0, index=df.index)
    pos = 0
    pos_ret = 0.0
    bars_in = 0
    
    for i in range(1, len(df)):
        if pos != 0:
            bar_ret = pos * df['ret'].iloc[i]
            pos_ret += bar_ret
            bars_in += 1
            
            # Check stop
            if pos_ret < -stop_bps/10000 or bars_in >= stop_bars * 3:
                mr_stopped.iloc[i] = bar_ret - cost_per_trade  # exit cost
                pos = 0
                pos_ret = 0
                bars_in = 0
                continue
            
            # Normal MR exit at mid
            if pos == 1 and df['close'].iloc[i] > df['bb_mid'].iloc[i]:
                mr_stopped.iloc[i] = bar_ret - cost_per_trade
                pos = 0
                pos_ret = 0
                bars_in = 0
            elif pos == -1 and df['close'].iloc[i] < df['bb_mid'].iloc[i]:
                mr_stopped.iloc[i] = bar_ret - cost_per_trade
                pos = 0
                pos_ret = 0
                bars_in = 0
            else:
                mr_stopped.iloc[i] = bar_ret
        else:
            # Entry signals
            if not pd.isna(df['bb_upper'].iloc[i]):
                if df['close'].iloc[i] > df['bb_upper'].iloc[i]:
                    pos = -1
                    pos_ret = 0
                    bars_in = 0
                    mr_stopped.iloc[i] = -cost_per_trade  # entry cost
                elif df['close'].iloc[i] < df['bb_lower'].iloc[i]:
                    pos = 1
                    pos_ret = 0
                    bars_in = 0
                    mr_stopped.iloc[i] = -cost_per_trade
    
    ann_ret = mr_stopped.mean() * 8760 * 100
    sharpe = mr_stopped.mean() / (mr_stopped.std() + 1e-10) * np.sqrt(8760)
    cum = (1 + mr_stopped).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    print(f"  MR stop={stop_bps}bps/{stop_bars}bars: Sharpe={sharpe:.2f}, Ann={ann_ret:.1f}%, MDD={mdd:.1f}%")

###############################################################################
# TEST 8: THE SIMPLE THESIS — MR Asian + TF US afternoon + stops
###############################################################################
print("\n" + "="*70)
print("TEST 8: THE SIMPLE THESIS")
print("="*70)
print("MR in Asian (0-8 UTC) + TF in US PM (18-22 UTC) + flat otherwise")
print("With MR hard stop at 50bps/12bars")

simple_ret = pd.Series(0.0, index=df.index)
for i in df.index:
    h = i.hour
    if 0 <= h < 8:
        simple_ret.loc[i] = df.loc[i, 'mr_ret_net']
    elif 18 <= h < 22:
        simple_ret.loc[i] = df.loc[i, 'tf_ret_net']
    # else flat

ann_ret = simple_ret.mean() * 8760 * 100
sharpe = simple_ret.mean() / (simple_ret.std() + 1e-10) * np.sqrt(8760)
cum = (1 + simple_ret).cumprod()
mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
total_ret = (cum.iloc[-1] - 1) * 100
print(f"  Sharpe={sharpe:.2f}, Ann={ann_ret:.1f}%, MDD={mdd:.1f}%, Total={total_ret:.1f}%")

# Compare to buy and hold
bh = df['ret'].dropna()
bh_cum = (1 + bh).cumprod()
bh_sharpe = bh.mean() / (bh.std() + 1e-10) * np.sqrt(8760)
bh_mdd = ((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min() * 100
print(f"  Buy&Hold: Sharpe={bh_sharpe:.2f}, MDD={bh_mdd:.1f}%, Total={(bh_cum.iloc[-1]-1)*100:.1f}%")

###############################################################################
# TEST 9: Year-by-year breakdown of best strategies
###############################################################################
print("\n" + "="*70)
print("TEST 9: YEAR-BY-YEAR BREAKDOWN")
print("="*70)

df['year'] = df.index.year
for year in sorted(df['year'].unique()):
    ydf = df[df['year'] == year]
    
    tf_y = ydf['tf_ret_net']
    mr_y = ydf['mr_ret_net']
    
    # Session-switched
    session_y = pd.Series(0.0, index=ydf.index)
    for i in ydf.index:
        h = i.hour
        if 8 <= h < 22:
            session_y.loc[i] = ydf.loc[i, 'tf_ret_net']
        else:
            session_y.loc[i] = ydf.loc[i, 'mr_ret_net']
    
    blend_y = (tf_y + mr_y) / 2
    
    results = {}
    for name, rets in [('TF', tf_y), ('MR', mr_y), ('Session', session_y), ('50/50', blend_y)]:
        sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(8760)
        total = (1 + rets).cumprod().iloc[-1] - 1
        results[name] = (sharpe, total * 100)
    
    best = max(results.items(), key=lambda x: x[1][0])
    print(f"  {year}: " + " | ".join(f"{k} {v[0]:+.2f} ({v[1]:+.0f}%)" for k,v in results.items()) + f"  → {best[0]}")

###############################################################################
# SUMMARY
###############################################################################
print("\n" + "="*70)
print("ULTRATHINK SUMMARY")
print("="*70)
