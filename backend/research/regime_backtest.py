#!/usr/bin/env python3
"""
Regime Detector Backtest: Which indicator best predicts TF vs MR weekly winners?

Ground truth: paper_trading_log.jsonl (40 days, 7 weeks)
Then extend to full BTC history with simulated TF/MR returns.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

###############################################################################
# PART 1: Load paper trader ground truth
###############################################################################

log_path = Path.home() / "Desktop/maestro/data/paper_trader/paper_trading_log.jsonl"
trades = []
with open(log_path) as f:
    for line in f:
        trades.append(json.loads(line))

# Weekly P&L by strategy
tf_weekly = defaultdict(float)
mr_weekly = defaultdict(float)
for t in trades:
    if t.get('type') != 'TRADE' or 'pnl' not in t:
        continue
    dt = datetime.fromisoformat(t['timestamp'])
    wk = dt.strftime('%G-W%V')  # ISO week
    if t['strategy'] == 'Trend-Following':
        tf_weekly[wk] += t['pnl']
    elif t['strategy'] == 'Mean-Reversion':
        mr_weekly[wk] += t['pnl']

weeks = sorted(set(tf_weekly.keys()) | set(mr_weekly.keys()))
ground_truth = {}  # week -> 'TF' or 'MR'
for wk in weeks:
    tf = tf_weekly.get(wk, 0)
    mr = mr_weekly.get(wk, 0)
    ground_truth[wk] = 'TF' if tf > mr else 'MR'
    print(f"{wk}: TF=${tf:+.0f}, MR=${mr:+.0f} -> {ground_truth[wk]}")

print(f"\nGround truth: {len(ground_truth)} weeks")
print(f"  TF weeks: {sum(1 for v in ground_truth.values() if v=='TF')}")
print(f"  MR weeks: {sum(1 for v in ground_truth.values() if v=='MR')}")

###############################################################################
# PART 2: Load BTC candles covering the paper trader period
###############################################################################

# Try multiple sources
candle_paths = [
    Path.home() / "Desktop/maestro/data/spot/1h/BTC_spot_1h.csv",
    Path.home() / "Desktop/maestro/data/spot/4h/BTC_spot_4h.csv",
]

df_1h = None
for p in candle_paths:
    if p.exists():
        df = pd.read_csv(p)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        elif 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        # Normalize column names
        col_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=col_map, inplace=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        if '1h' in str(p):
            df_1h = df
            print(f"\nLoaded {p.name}: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
        break

if df_1h is None:
    # Try 4h
    p = candle_paths[1]
    if p.exists():
        df_1h = pd.read_csv(p)
        if 'timestamp' in df_1h.columns:
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        elif 'open_time' in df_1h.columns:
            df_1h['timestamp'] = pd.to_datetime(df_1h['open_time'], unit='ms')
        df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
        print(f"\nLoaded {p.name}: {len(df_1h)} rows")

if df_1h is None:
    print("ERROR: No BTC candle data found!")
    exit(1)

###############################################################################
# PART 3: Compute regime indicators
###############################################################################

df = df_1h.copy()
df.set_index('timestamp', inplace=True)

# 1. ADX (14-period on hourly, resampled to get weekly reading)
def compute_adx(high, low, close, period=14):
    """Manual ADX since we might not have talib."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    # Where plus_dm > minus_dm, minus_dm = 0 and vice versa
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0
    
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx

df['adx'] = compute_adx(df['high'], df['low'], df['close'], 14)

# 2. Bollinger Band Width
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_width'] = (df['bb_std'] * 2) / (df['bb_mid'] + 1e-10)

# 3. Realized Volatility (rolling 24h std of returns)
df['ret'] = df['close'].pct_change()
df['rvol_24h'] = df['ret'].rolling(24).std()
df['rvol_168h'] = df['ret'].rolling(168).std()  # 1 week
df['vol_ratio'] = df['rvol_24h'] / (df['rvol_168h'] + 1e-10)

# 4. ATR / Price (normalized ATR)
tr = pd.DataFrame({
    'hl': df['high'] - df['low'],
    'hc': abs(df['high'] - df['close'].shift(1)),
    'lc': abs(df['low'] - df['close'].shift(1))
}).max(axis=1)
df['atr_14'] = tr.rolling(14).mean()
df['atr_pct'] = df['atr_14'] / df['close']

# 5. Price range as % of price (weekly)
# 6. EMA slope (trend strength)
df['ema_20'] = df['close'].ewm(span=20).mean()
df['ema_slope'] = (df['ema_20'] - df['ema_20'].shift(5)) / df['ema_20'].shift(5) * 100

# 7. Directional movement ratio
df['abs_ret_sum'] = df['ret'].abs().rolling(24).sum()
df['net_ret'] = df['ret'].rolling(24).sum()
df['efficiency'] = abs(df['net_ret']) / (df['abs_ret_sum'] + 1e-10)

###############################################################################
# PART 4: Weekly indicator values at week start
###############################################################################

# Get Monday values for each indicator
df['iso_week'] = df.index.strftime('%G-W%V')

weekly_indicators = {}
for wk in weeks:
    # Get data for this week
    wk_data = df[df['iso_week'] == wk]
    if len(wk_data) == 0:
        continue
    # Use first available reading (start of week)
    first = wk_data.iloc[0]
    # Also use prior week's average for some
    prior_idx = df.index < wk_data.index[0]
    prior_week = df[prior_idx].tail(168)  # last 168 hours = 1 week
    
    weekly_indicators[wk] = {
        'adx_start': first['adx'] if not pd.isna(first['adx']) else 25,
        'adx_prior_avg': prior_week['adx'].mean() if len(prior_week) > 0 else 25,
        'bb_width_start': first['bb_width'] if not pd.isna(first['bb_width']) else 0.02,
        'bb_width_prior_avg': prior_week['bb_width'].mean() if len(prior_week) > 0 else 0.02,
        'rvol_24h': first['rvol_24h'] if not pd.isna(first['rvol_24h']) else 0.01,
        'vol_ratio': first['vol_ratio'] if not pd.isna(first['vol_ratio']) else 1.0,
        'atr_pct': first['atr_pct'] if not pd.isna(first['atr_pct']) else 0.01,
        'ema_slope': first['ema_slope'] if not pd.isna(first['ema_slope']) else 0,
        'efficiency': first['efficiency'] if not pd.isna(first['efficiency']) else 0.5,
    }

###############################################################################
# PART 5: Score each indicator as regime classifier
###############################################################################

print("\n" + "="*70)
print("REGIME INDICATOR ACCURACY (vs Paper Trader Ground Truth)")
print("="*70)

# For each indicator, find optimal threshold that separates TF weeks from MR weeks
indicators = ['adx_start', 'adx_prior_avg', 'bb_width_start', 'bb_width_prior_avg',
              'rvol_24h', 'vol_ratio', 'atr_pct', 'ema_slope', 'efficiency']

# Direction: for some, high = trending (TF), for others high = ranging (MR)
# ADX high -> trending -> TF
# BB width high -> volatile -> could be either
# Rvol high -> volatile -> trending?
# Vol ratio high -> expanding vol -> trending
# ATR pct high -> wide range -> trending
# EMA slope high (abs) -> trending
# Efficiency high -> trending (directional)

for ind in indicators:
    vals = []
    labels = []
    for wk in weeks:
        if wk in weekly_indicators and wk in ground_truth:
            vals.append(weekly_indicators[wk][ind])
            labels.append(ground_truth[wk])
    
    if len(vals) < 4:
        print(f"\n{ind}: insufficient data ({len(vals)} weeks)")
        continue
    
    # Try both directions: high->TF or high->MR
    best_acc = 0
    best_thresh = None
    best_dir = None
    
    sorted_vals = sorted(set(vals))
    for thresh in sorted_vals:
        # Direction 1: above thresh -> TF
        pred1 = ['TF' if v > thresh else 'MR' for v in vals]
        acc1 = sum(p == l for p, l in zip(pred1, labels)) / len(labels)
        # Direction 2: above thresh -> MR
        pred2 = ['MR' if v > thresh else 'TF' for v in vals]
        acc2 = sum(p == l for p, l in zip(pred2, labels)) / len(labels)
        
        if acc1 > best_acc:
            best_acc = acc1
            best_thresh = thresh
            best_dir = f">{thresh:.4f} -> TF"
        if acc2 > best_acc:
            best_acc = acc2
            best_thresh = thresh
            best_dir = f">{thresh:.4f} -> MR"
    
    # Show values per week
    print(f"\n{ind}: best accuracy {best_acc:.0%} ({best_dir})")
    for wk in weeks:
        if wk in weekly_indicators:
            v = weekly_indicators[wk][ind]
            gt = ground_truth[wk]
            print(f"  {wk}: {v:>8.4f}  actual={gt}")

###############################################################################
# PART 6: Extended backtest — simulate TF vs MR on full BTC history
###############################################################################

print("\n" + "="*70)
print("EXTENDED BACKTEST: Simulated TF vs MR on full BTC history")
print("="*70)

# Simple TF: EMA(10) > EMA(30) -> long, else short. P&L = direction * return
# Simple MR: RSI > 70 -> short, RSI < 30 -> long, else flat. P&L on BB bounce.

# Actually, let's do it properly with daily resampling
daily = df['close'].resample('1D').last().dropna()
daily_ret = daily.pct_change().dropna()

# TF proxy: sign of 20-day momentum * daily return
mom_20 = daily.pct_change(20)
tf_ret = (np.sign(mom_20.shift(1)) * daily_ret).dropna()

# MR proxy: contrarian on 5-day return
mom_5 = daily.pct_change(5)
mr_ret = (-np.sign(mom_5.shift(1)) * daily_ret).dropna()

# Align
common = tf_ret.index.intersection(mr_ret.index)
tf_ret = tf_ret.loc[common]
mr_ret = mr_ret.loc[common]

# Weekly aggregation
tf_weekly_ext = tf_ret.resample('W').sum()
mr_weekly_ext = mr_ret.resample('W').sum()

common_weeks_ext = tf_weekly_ext.index.intersection(mr_weekly_ext.index)
tf_weekly_ext = tf_weekly_ext.loc[common_weeks_ext]
mr_weekly_ext = mr_weekly_ext.loc[common_weeks_ext]

n_weeks = len(common_weeks_ext)
print(f"\nExtended period: {n_weeks} weeks ({common_weeks_ext[0].date()} to {common_weeks_ext[-1].date()})")

# Ground truth extended
gt_ext = pd.Series(['TF' if t > m else 'MR' for t, m in zip(tf_weekly_ext, mr_weekly_ext)],
                   index=common_weeks_ext)
print(f"TF wins: {(gt_ext=='TF').sum()} weeks ({(gt_ext=='TF').mean():.0%})")
print(f"MR wins: {(gt_ext=='MR').sum()} weeks ({(gt_ext=='MR').mean():.0%})")

# Correlation
corr = tf_weekly_ext.corr(mr_weekly_ext)
print(f"TF vs MR weekly correlation: {corr:.3f}")

# Now test regime indicators on extended period
daily_df = pd.DataFrame({'close': daily})
daily_df['adx'] = compute_adx(
    df['high'].resample('1D').max(),
    df['low'].resample('1D').min(),
    df['close'].resample('1D').last(),
    14
)
daily_df['bb_mid'] = daily_df['close'].rolling(20).mean()
daily_df['bb_std'] = daily_df['close'].rolling(20).std()
daily_df['bb_width'] = (daily_df['bb_std'] * 2) / (daily_df['bb_mid'] + 1e-10)

daily_df['ret'] = daily.pct_change()
daily_df['rvol_5d'] = daily_df['ret'].rolling(5).std()
daily_df['rvol_20d'] = daily_df['ret'].rolling(20).std()
daily_df['vol_ratio'] = daily_df['rvol_5d'] / (daily_df['rvol_20d'] + 1e-10)

# Efficiency ratio (Kaufman)
daily_df['direction'] = abs(daily.diff(10))
daily_df['volatility'] = daily.diff().abs().rolling(10).sum()
daily_df['er'] = daily_df['direction'] / (daily_df['volatility'] + 1e-10)

# Weekly start values
weekly_start = daily_df.resample('W').first()

# Test each indicator
print(f"\n{'Indicator':<20} {'Accuracy':<10} {'Rule':<30} {'P&L if used':<15}")
print("-" * 75)

indicators_ext = {
    'adx': ('ADX at week start', True),  # high -> TF
    'bb_width': ('BB Width', True),       # high -> trending?
    'vol_ratio': ('Vol Ratio 5d/20d', True),  # high -> expanding vol -> TF
    'er': ('Efficiency Ratio', True),     # high -> trending -> TF
    'rvol_5d': ('5-day RVol', True),      # high -> volatile
}

for col, (name, _) in indicators_ext.items():
    if col not in weekly_start.columns:
        continue
    
    vals = weekly_start[col].reindex(common_weeks_ext).dropna()
    valid_weeks = vals.index.intersection(gt_ext.index)
    if len(valid_weeks) < 10:
        continue
    
    vals = vals.loc[valid_weeks]
    gt_v = gt_ext.loc[valid_weeks]
    
    # Optimal threshold (in-sample, so this is cheating — but shows ceiling)
    best_acc = 0
    best_rule = ""
    best_preds = None
    
    for pctile in range(10, 91, 5):
        thresh = np.percentile(vals, pctile)
        # High -> TF
        preds1 = pd.Series(['TF' if v > thresh else 'MR' for v in vals], index=valid_weeks)
        acc1 = (preds1 == gt_v).mean()
        # High -> MR
        preds2 = pd.Series(['MR' if v > thresh else 'TF' for v in vals], index=valid_weeks)
        acc2 = (preds2 == gt_v).mean()
        
        if acc1 > best_acc:
            best_acc = acc1
            best_rule = f">{thresh:.4f} -> TF (p{pctile})"
            best_preds = preds1
        if acc2 > best_acc:
            best_acc = acc2
            best_rule = f">{thresh:.4f} -> MR (p{pctile})"
            best_preds = preds2
    
    # P&L if we followed this indicator
    if best_preds is not None:
        pnl = 0
        for wk in valid_weeks:
            if best_preds[wk] == 'TF':
                pnl += tf_weekly_ext.get(wk, 0)
            else:
                pnl += mr_weekly_ext.get(wk, 0)
        ann_sharpe = 0
        weekly_pnls = []
        for wk in valid_weeks:
            if best_preds[wk] == 'TF':
                weekly_pnls.append(tf_weekly_ext.get(wk, 0))
            else:
                weekly_pnls.append(mr_weekly_ext.get(wk, 0))
        if len(weekly_pnls) > 1:
            ann_sharpe = np.mean(weekly_pnls) / (np.std(weekly_pnls) + 1e-10) * np.sqrt(52)
        
        print(f"{name:<20} {best_acc:<10.0%} {best_rule:<30} Sharpe={ann_sharpe:.2f}")

# Baselines
print(f"\n{'--- BASELINES ---':<20}")
# Always TF
always_tf = tf_weekly_ext.loc[common_weeks_ext]
sharpe_tf = always_tf.mean() / (always_tf.std() + 1e-10) * np.sqrt(52)
print(f"{'Always TF':<20} {'—':<10} {'—':<30} Sharpe={sharpe_tf:.2f}")

# Always MR
always_mr = mr_weekly_ext.loc[common_weeks_ext]
sharpe_mr = always_mr.mean() / (always_mr.std() + 1e-10) * np.sqrt(52)
print(f"{'Always MR':<20} {'—':<10} {'—':<30} Sharpe={sharpe_mr:.2f}")

# 50/50
fifty = (tf_weekly_ext + mr_weekly_ext) / 2
sharpe_50 = fifty.mean() / (fifty.std() + 1e-10) * np.sqrt(52)
print(f"{'50/50 blend':<20} {'—':<10} {'—':<30} Sharpe={sharpe_50:.2f}")

# Oracle
oracle = pd.Series([max(t, m) for t, m in zip(tf_weekly_ext, mr_weekly_ext)], index=common_weeks_ext)
sharpe_oracle = oracle.mean() / (oracle.std() + 1e-10) * np.sqrt(52)
print(f"{'Oracle (perfect)':<20} {'100%':<10} {'—':<30} Sharpe={sharpe_oracle:.2f}")

###############################################################################
# PART 7: Walk-forward regime detection (no lookahead)
###############################################################################

print("\n" + "="*70)
print("WALK-FORWARD REGIME DETECTION (NO LOOKAHEAD)")
print("="*70)
print("Train on past 12 weeks, predict next week. Rolling window.")

# Use efficiency ratio as primary candidate
for col, name in [('adx', 'ADX'), ('er', 'Efficiency Ratio'), ('vol_ratio', 'Vol Ratio'), ('bb_width', 'BB Width')]:
    if col not in weekly_start.columns:
        continue
    
    vals = weekly_start[col].reindex(common_weeks_ext).dropna()
    valid_weeks = sorted(vals.index.intersection(gt_ext.index))
    
    if len(valid_weeks) < 20:
        continue
    
    train_window = 12
    correct = 0
    total = 0
    wf_pnls = []
    
    for i in range(train_window, len(valid_weeks)):
        # Train: find best threshold on past 12 weeks
        train_vals = [vals[valid_weeks[j]] for j in range(i-train_window, i)]
        train_gt = [gt_ext[valid_weeks[j]] for j in range(i-train_window, i)]
        
        best_acc = 0
        best_pred_fn = lambda v: 'TF'  # default
        
        for pctile in [25, 50, 75]:
            thresh = np.percentile(train_vals, pctile)
            for direction in ['TF', 'MR']:
                preds = [direction if v > thresh else ('MR' if direction=='TF' else 'TF') for v in train_vals]
                acc = sum(p == g for p, g in zip(preds, train_gt)) / len(train_gt)
                if acc > best_acc:
                    best_acc = acc
                    _thresh = thresh
                    _dir = direction
                    best_pred_fn = lambda v, t=_thresh, d=_dir: d if v > t else ('MR' if d=='TF' else 'TF')
        
        # Predict this week
        test_wk = valid_weeks[i]
        pred = best_pred_fn(vals[test_wk])
        actual = gt_ext[test_wk]
        
        if pred == actual:
            correct += 1
        total += 1
        
        if pred == 'TF':
            wf_pnls.append(tf_weekly_ext.get(test_wk, 0))
        else:
            wf_pnls.append(mr_weekly_ext.get(test_wk, 0))
    
    wf_acc = correct / total if total > 0 else 0
    wf_sharpe = np.mean(wf_pnls) / (np.std(wf_pnls) + 1e-10) * np.sqrt(52) if len(wf_pnls) > 1 else 0
    print(f"\n{name}: WF accuracy {wf_acc:.0%} ({correct}/{total}), Sharpe={wf_sharpe:.2f}")

# Composite: use multiple indicators
print("\n--- COMPOSITE (majority vote of ADX + ER + Vol Ratio) ---")
composite_correct = 0
composite_total = 0
composite_pnls = []

for i in range(train_window, len(valid_weeks)):
    votes = []
    for col in ['adx', 'er', 'vol_ratio']:
        if col not in weekly_start.columns:
            continue
        vals_c = weekly_start[col].reindex(common_weeks_ext).dropna()
        if valid_weeks[i] not in vals_c.index:
            continue
        
        train_vals = [vals_c[valid_weeks[j]] for j in range(i-train_window, i) if valid_weeks[j] in vals_c.index]
        train_gt_c = [gt_ext[valid_weeks[j]] for j in range(i-train_window, i) if valid_weeks[j] in vals_c.index]
        
        if len(train_vals) < 6:
            continue
        
        best_acc = 0
        best_pred = 'TF'
        for pctile in [25, 50, 75]:
            thresh = np.percentile(train_vals, pctile)
            for direction in ['TF', 'MR']:
                preds = [direction if v > thresh else ('MR' if direction=='TF' else 'TF') for v in train_vals]
                acc = sum(p == g for p, g in zip(preds, train_gt_c)) / len(train_gt_c)
                if acc > best_acc:
                    best_acc = acc
                    _t = thresh
                    _d = direction
                    best_pred = _d if vals_c[valid_weeks[i]] > _t else ('MR' if _d=='TF' else 'TF')
        
        votes.append(best_pred)
    
    if not votes:
        continue
    
    # Majority vote
    pred = 'TF' if votes.count('TF') > votes.count('MR') else 'MR'
    actual = gt_ext[valid_weeks[i]]
    
    if pred == actual:
        composite_correct += 1
    composite_total += 1
    
    if pred == 'TF':
        composite_pnls.append(tf_weekly_ext.get(valid_weeks[i], 0))
    else:
        composite_pnls.append(mr_weekly_ext.get(valid_weeks[i], 0))

comp_acc = composite_correct / composite_total if composite_total > 0 else 0
comp_sharpe = np.mean(composite_pnls) / (np.std(composite_pnls) + 1e-10) * np.sqrt(52) if len(composite_pnls) > 1 else 0
print(f"Composite WF: accuracy {comp_acc:.0%} ({composite_correct}/{composite_total}), Sharpe={comp_sharpe:.2f}")

print("\n" + "="*70)
print("DONE")
print("="*70)
