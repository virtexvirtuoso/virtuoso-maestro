#!/usr/bin/env python3
"""
Deep dive on the findings:
1. Contrarian TF Asian = ONLY positive Sharpe strategy (0.27, +13.8%)
2. Vol percentile regime at 70% → Sharpe 1.11 (but is this walk-forward?)
3. Paper trader vs full backtest disconnect — WHY?
4. TF dominates 2022-2024, MR dominates 2025 — regime shifted?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(Path.home() / "Desktop/maestro/data/spot/1h/BTC_spot_1h.csv")
df['timestamp'] = pd.to_datetime(df['Date'])
df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)
df = df.sort_values('timestamp').set_index('timestamp')

# Strategies
df['ema10'] = df['close'].ewm(span=10).mean()
df['ema30'] = df['close'].ewm(span=30).mean()
df['ret'] = df['close'].pct_change()
df['tf_signal'] = np.where(df['ema10'] > df['ema30'], 1, -1)
df['tf_ret'] = df['tf_signal'].shift(1) * df['ret']

df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
mr_signal = pd.Series(0, index=df.index)
pos = 0
for i in range(1, len(df)):
    if df['close'].iloc[i] > df['bb_upper'].iloc[i] and not pd.isna(df['bb_upper'].iloc[i]):
        pos = -1
    elif df['close'].iloc[i] < df['bb_lower'].iloc[i] and not pd.isna(df['bb_lower'].iloc[i]):
        pos = 1
    elif pos == 1 and df['close'].iloc[i] > df['bb_mid'].iloc[i]:
        pos = 0
    elif pos == -1 and df['close'].iloc[i] < df['bb_mid'].iloc[i]:
        pos = 0
    mr_signal.iloc[i] = pos
df['mr_signal'] = mr_signal
df['mr_ret'] = df['mr_signal'].shift(1) * df['ret']

cost = 0.00075
df['tf_trades'] = df['tf_signal'].diff().abs() / 2
df['mr_trades'] = df['mr_signal'].diff().abs().clip(upper=1)
df['tf_ret_net'] = df['tf_ret'] - df['tf_trades'] * cost
df['mr_ret_net'] = df['mr_ret'] - df['mr_trades'] * cost
df = df.iloc[100:].copy()
df['hour'] = df.index.hour

###############################################################################
# DEEP DIVE 1: Walk-forward vol percentile regime
###############################################################################
print("="*70)
print("WALK-FORWARD VOL PERCENTILE REGIME SWITCHING")
print("="*70)

daily_close = df['close'].resample('1D').last()
daily_ret = daily_close.pct_change()
daily_vol = daily_ret.rolling(5).std()

tf_weekly = df['tf_ret_net'].resample('W').sum()
mr_weekly = df['mr_ret_net'].resample('W').sum()
common = tf_weekly.index.intersection(mr_weekly.index)
tf_weekly = tf_weekly.loc[common]
mr_weekly = mr_weekly.loc[common]

# Walk-forward: use ONLY past vol percentile (rolling 252-day window, no future)
vol_pctile = daily_vol.rolling(252, min_periods=60).rank(pct=True)
weekly_vol = vol_pctile.resample('W').last().dropna()
valid = weekly_vol.index.intersection(common)

# Train on 26 weeks, predict next
train_win = 26
correct = 0
total = 0
wf_pnls = []
always_tf_pnls = []
always_mr_pnls = []

for i in range(train_win, len(valid)):
    wk = valid[i]
    gt = 'TF' if tf_weekly.loc[wk] > mr_weekly.loc[wk] else 'MR'
    
    # Find best threshold from training window
    train_wks = valid[i-train_win:i]
    train_vols = weekly_vol.loc[train_wks].values
    train_gts = ['TF' if tf_weekly.loc[w] > mr_weekly.loc[w] else 'MR' for w in train_wks]
    
    best_acc = 0
    best_pred = 'MR'
    for pctile in range(20, 81, 10):
        thresh = np.percentile(train_vols, pctile)
        for d in ['TF', 'MR']:
            preds = [d if v > thresh else ('MR' if d=='TF' else 'TF') for v in train_vols]
            acc = sum(p == g for p, g in zip(preds, train_gts)) / len(train_gts)
            if acc > best_acc:
                best_acc = acc
                _t, _d = thresh, d
                best_pred = _d if weekly_vol.loc[wk] > _t else ('MR' if _d=='TF' else 'TF')
    
    if best_pred == gt:
        correct += 1
    total += 1
    wf_pnls.append(tf_weekly.loc[wk] if best_pred == 'TF' else mr_weekly.loc[wk])
    always_tf_pnls.append(tf_weekly.loc[wk])
    always_mr_pnls.append(mr_weekly.loc[wk])

acc = correct / total
sharpe = np.mean(wf_pnls) / (np.std(wf_pnls) + 1e-10) * np.sqrt(52)
sharpe_tf = np.mean(always_tf_pnls) / (np.std(always_tf_pnls) + 1e-10) * np.sqrt(52)
sharpe_mr = np.mean(always_mr_pnls) / (np.std(always_mr_pnls) + 1e-10) * np.sqrt(52)
print(f"WF Vol Regime: acc={acc:.0%} ({correct}/{total}), Sharpe={sharpe:.2f}")
print(f"  vs Always TF: Sharpe={sharpe_tf:.2f}")
print(f"  vs Always MR: Sharpe={sharpe_mr:.2f}")
print(f"  vs 50/50: Sharpe={(np.mean(always_tf_pnls)+np.mean(always_mr_pnls))/2 / (np.std([(a+b)/2 for a,b in zip(always_tf_pnls, always_mr_pnls)])+1e-10) * np.sqrt(52):.2f}")

# Year-by-year WF vol regime
print("\n  Year-by-year WF vol regime:")
years = {}
for i, wk in enumerate(valid[train_win:]):
    yr = wk.year
    if yr not in years:
        years[yr] = {'correct': 0, 'total': 0, 'pnl': []}
    gt = 'TF' if tf_weekly.loc[wk] > mr_weekly.loc[wk] else 'MR'
    pred_correct = (wf_pnls[i] == tf_weekly.loc[wk] and gt == 'TF') or (wf_pnls[i] == mr_weekly.loc[wk] and gt == 'MR')
    years[yr]['total'] += 1
    years[yr]['pnl'].append(wf_pnls[i])

for yr in sorted(years.keys()):
    pnls_yr = years[yr]['pnl']
    sr = np.mean(pnls_yr) / (np.std(pnls_yr) + 1e-10) * np.sqrt(52)
    tot = sum(pnls_yr) * 100
    print(f"    {yr}: Sharpe={sr:.2f}, return={tot:+.1f}%")

###############################################################################
# DEEP DIVE 2: Paper trader used 5-min polling — does TF timeframe matter?
###############################################################################
print("\n" + "="*70)
print("DEEP DIVE 2: EMA PARAMETERS MATTER")
print("="*70)

for fast, slow in [(5, 15), (10, 30), (20, 50), (50, 200), (10, 50), (20, 100)]:
    df[f'ema_{fast}'] = df['close'].ewm(span=fast).mean()
    df[f'ema_{slow}'] = df['close'].ewm(span=slow).mean()
    sig = np.where(df[f'ema_{fast}'] > df[f'ema_{slow}'], 1, -1)
    trades_count = pd.Series(sig).diff().abs().sum() / 2
    ret = pd.Series(sig, index=df.index).shift(1) * df['ret']
    trade_cost = pd.Series(sig, index=df.index).diff().abs() / 2 * cost
    ret_net = ret - trade_cost
    sharpe = ret_net.mean() / (ret_net.std() + 1e-10) * np.sqrt(8760)
    ann_ret = ret_net.mean() * 8760 * 100
    print(f"  EMA({fast},{slow}): Sharpe={sharpe:.2f}, Ann={ann_ret:.1f}%, trades={trades_count:.0f}")

###############################################################################
# DEEP DIVE 3: Why does Contrarian TF Asian work?
###############################################################################
print("\n" + "="*70)
print("DEEP DIVE 3: CONTRARIAN TF ASIAN — WHY?")
print("="*70)

# Asian session characteristics
asian = df[df['hour'].isin(range(0, 8))]
non_asian = df[~df['hour'].isin(range(0, 8))]

# TF signal accuracy by session
for session, sdf in [('Asian', asian), ('Non-Asian', non_asian)]:
    # When TF says long, does price go up?
    long_mask = sdf['tf_signal'].shift(1) == 1
    short_mask = sdf['tf_signal'].shift(1) == -1
    
    long_ret = sdf.loc[long_mask, 'ret'].mean() * 10000  # bps
    short_ret = sdf.loc[short_mask, 'ret'].mean() * 10000
    
    # TF says long, price goes UP = correct. Goes DOWN = wrong.
    long_correct = (sdf.loc[long_mask, 'ret'] > 0).mean() * 100
    short_correct = (sdf.loc[short_mask, 'ret'] < 0).mean() * 100
    
    print(f"\n  {session}:")
    print(f"    When TF=LONG:  avg ret={long_ret:+.2f}bps, correct {long_correct:.1f}%")
    print(f"    When TF=SHORT: avg ret={short_ret:+.2f}bps, correct {short_correct:.1f}%")
    
    # Mean reversion: what happens after big moves?
    big_up = sdf[sdf['ret'] > sdf['ret'].quantile(0.9)]
    big_down = sdf[sdf['ret'] < sdf['ret'].quantile(0.1)]
    
    # Next-bar return after big moves
    big_up_next = sdf['ret'].shift(-1).loc[big_up.index].mean() * 10000
    big_down_next = sdf['ret'].shift(-1).loc[big_down.index].mean() * 10000
    print(f"    After big up: next bar {big_up_next:+.2f}bps (MR={big_up_next<0})")
    print(f"    After big down: next bar {big_down_next:+.2f}bps (MR={big_down_next>0})")

###############################################################################
# DEEP DIVE 4: Contrarian TF Asian — walk-forward stability
###############################################################################
print("\n" + "="*70)
print("DEEP DIVE 4: CONTRARIAN TF ASIAN — STABILITY BY QUARTER")
print("="*70)

df['quarter'] = df.index.to_period('Q')
for q in sorted(df['quarter'].unique()):
    qdf = df[df['quarter'] == q]
    asian_q = qdf[qdf['hour'].isin(range(0, 8))]
    
    # Contrarian TF in Asian
    contra_ret = -asian_q['tf_ret_net']
    sharpe = contra_ret.mean() / (contra_ret.std() + 1e-10) * np.sqrt(8760/3)
    total = contra_ret.sum() * 100
    print(f"  {q}: Sharpe={sharpe:+.2f}, return={total:+.1f}%")

###############################################################################
# DEEP DIVE 5: Does adding VPIN improve regime detection?
# (Placeholder — needs VPS data, but test concept with vol as proxy)
###############################################################################
print("\n" + "="*70)
print("DEEP DIVE 5: MULTI-INDICATOR REGIME (vol + ADX + efficiency)")
print("="*70)

def compute_adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0
    tr = pd.DataFrame({
        'hl': high - low, 'hc': (high - close.shift(1)).abs(), 'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(span=period, adjust=False).mean()

daily_h = df['high'].resample('1D').max()
daily_l = df['low'].resample('1D').min()
daily_c = df['close'].resample('1D').last()
daily_adx = compute_adx(daily_h, daily_l, daily_c).dropna()

daily_ret2 = daily_c.pct_change()
daily_vol2 = daily_ret2.rolling(5).std()
daily_eff = abs(daily_c.diff(10)) / (daily_c.diff().abs().rolling(10).sum() + 1e-10)

# Composite score: normalize each to 0-1 via rolling percentile, average
score_adx = daily_adx.rolling(252, min_periods=60).rank(pct=True)
score_vol = daily_vol2.rolling(252, min_periods=60).rank(pct=True)
score_eff = daily_eff.rolling(252, min_periods=60).rank(pct=True)

composite = ((score_adx.fillna(0.5) + score_vol.fillna(0.5) + score_eff.fillna(0.5)) / 3)
weekly_composite = composite.resample('W').last().dropna()
valid_c = weekly_composite.index.intersection(common)

# Walk-forward
train_win = 26
wf_pnls = []
for i in range(train_win, len(valid_c)):
    wk = valid_c[i]
    train_wks = valid_c[i-train_win:i]
    train_scores = weekly_composite.loc[train_wks].values
    train_gts = ['TF' if tf_weekly.loc[w] > mr_weekly.loc[w] else 'MR' for w in train_wks]
    
    best_acc = 0
    best_pred = 'MR'
    for pctile in range(20, 81, 10):
        thresh = np.percentile(train_scores, pctile)
        for d in ['TF', 'MR']:
            preds = [d if v > thresh else ('MR' if d=='TF' else 'TF') for v in train_scores]
            acc = sum(p == g for p, g in zip(preds, train_gts)) / len(train_gts)
            if acc > best_acc:
                best_acc = acc
                _t, _d = thresh, d
                best_pred = _d if weekly_composite.loc[wk] > _t else ('MR' if _d=='TF' else 'TF')
    
    wf_pnls.append(tf_weekly.loc[wk] if best_pred == 'TF' else mr_weekly.loc[wk])

sharpe = np.mean(wf_pnls) / (np.std(wf_pnls) + 1e-10) * np.sqrt(52)
total = sum(wf_pnls) * 100
print(f"Composite WF regime: Sharpe={sharpe:.2f}, total={total:+.1f}%")

###############################################################################
# DEEP DIVE 6: The REAL question — longer EMA TF + session filter
###############################################################################
print("\n" + "="*70)
print("DEEP DIVE 6: SLOW TF (EMA 50/200) + SESSION FILTER")
print("="*70)

for fast, slow in [(20, 50), (50, 200)]:
    sig = np.where(df['close'].ewm(span=fast).mean() > df['close'].ewm(span=slow).mean(), 1, -1)
    sig = pd.Series(sig, index=df.index)
    
    for session_name, hours in [('All hours', range(24)), ('EU+US only (8-22)', range(8,22)),
                                  ('US PM only (14-22)', range(14,22)), ('Asian+US PM', list(range(0,8))+list(range(14,22)))]:
        mask = df['hour'].isin(hours)
        ret = sig.shift(1) * df['ret']
        trade_cost = sig.diff().abs() / 2 * cost
        ret_net = (ret - trade_cost) * mask
        
        sharpe = ret_net.mean() / (ret_net.std() + 1e-10) * np.sqrt(8760)
        ann = ret_net.mean() * 8760 * 100
        cum = (1 + ret_net).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
        
        print(f"  EMA({fast},{slow}) {session_name:<25}: Sharpe={sharpe:+.2f}, Ann={ann:+.1f}%, MDD={mdd:.1f}%")
    print()

print("\n" + "="*70)
print("DONE")
print("="*70)
