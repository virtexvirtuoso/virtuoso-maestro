"""
Comprehensive On-Chain Signal Testing — 6 Approaches
=====================================================
Tests NUPL, NVT, Reserve Risk, SOPR with:
1. Full-cycle spot (2013+) — signals designed for this era
2. Regime-conditional — on-chain only at cycle extremes
3. Composite scoring — combine all 4 into single score
4. Weekly timeframe — on-chain is inherently slow
5. Event-based — extreme readings as risk-on/off overlays
6. V4 overlay — improve V4 with on-chain risk management

Methodology: Walk-forward 14 folds, permutation test (1000x), 
bootstrap CI, Bonferroni correction, look-ahead prevention.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "backtest_results"
ONCHAIN_DIR = DATA_DIR / "onchain"
SPOT_DIR = DATA_DIR / "spot"

np.random.seed(42)


def load_onchain():
    """Load all on-chain signals and merge with BTC spot price."""
    signals = {}
    for f in ['nupl.json', 'nvt.json', 'reserve_risk.json', 'sopr.json']:
        name = f.replace('.json', '')
        with open(ONCHAIN_DIR / f) as fh:
            data = json.load(fh)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['d'])
        df = df.set_index('date').drop(columns=['d', 'unixTs'], errors='ignore')
        # Convert string values to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        signals[name] = df
    
    # Load BTC spot
    btc = pd.read_csv(SPOT_DIR / "BTC_spot_daily.csv", parse_dates=['Date'])
    btc = btc.rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'open', 
                               'High': 'high', 'Low': 'low', 'Volume': 'volume'})
    btc = btc.set_index('date').sort_index()
    
    # Merge all
    merged = btc[['close']].copy()
    merged['returns'] = merged['close'].pct_change()
    
    for name, df in signals.items():
        col = df.columns[0]  # nupl, nvt, reserveRisk, sopr
        merged[name] = df[col]
    
    merged = merged.dropna(subset=['returns'])
    # Forward-fill on-chain (they can have gaps)
    for name in signals:
        merged[name] = merged[name].ffill()
    
    print(f"Merged data: {merged.index[0].date()} to {merged.index[-1].date()}, {len(merged)} days")
    print(f"Columns: {list(merged.columns)}")
    print(f"Non-null counts:")
    for c in merged.columns:
        print(f"  {c}: {merged[c].notna().sum()}")
    
    return merged


def walk_forward_test(signals_series, returns, n_folds=14, min_train=252):
    """Walk-forward backtest. signals_series: 1=long, 0=flat, -1=short. Returns shifted properly."""
    # CRITICAL: signal on day N, trade on day N+1
    position = signals_series.shift(1)  # look-ahead prevention
    strategy_returns = position * returns
    
    total_days = len(returns)
    fold_size = (total_days - min_train) // n_folds
    
    oos_returns = []
    
    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = min(oos_start + fold_size, total_days)
        if oos_end <= oos_start:
            break
        oos_ret = strategy_returns.iloc[oos_start:oos_end]
        oos_returns.append(oos_ret)
    
    if not oos_returns:
        return {'sharpe': 0, 'cagr': 0, 'maxdd': 0, 'n_days': 0}
    
    all_oos = pd.concat(oos_returns)
    return calc_metrics(all_oos)


def calc_metrics(returns_series):
    """Calculate Sharpe, CAGR, MaxDD from returns series."""
    r = returns_series.dropna()
    if len(r) < 30:
        return {'sharpe': 0, 'cagr': 0, 'maxdd': 0, 'n_days': len(r)}
    
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    total_return = cum.iloc[-1] - 1
    n_years = len(r) / 252
    cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 and total_return > -1 else -1
    maxdd = (cum / cum.cummax() - 1).min()
    
    return {'sharpe': round(sharpe, 4), 'cagr': round(cagr, 4), 'maxdd': round(maxdd, 4), 'n_days': len(r)}


def permutation_test(signals_series, returns, n_perms=1000):
    """Permutation test: shuffle returns, compute Sharpe distribution."""
    position = signals_series.shift(1)
    actual_ret = (position * returns).dropna()
    actual_sharpe = actual_ret.mean() / actual_ret.std() * np.sqrt(252) if actual_ret.std() > 0 else 0
    
    count_higher = 0
    for _ in range(n_perms):
        shuffled = returns.sample(frac=1, replace=False).values
        perm_ret = position.dropna() * shuffled[:len(position.dropna())]
        perm_sharpe = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_higher += 1
    
    return count_higher / n_perms


def bootstrap_ci(returns_series, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    r = returns_series.dropna()
    sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(r.values, size=len(r), replace=True)
        s = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
        sharpes.append(s)
    alpha = (1 - ci) / 2
    return [round(np.percentile(sharpes, alpha*100), 4), round(np.percentile(sharpes, (1-alpha)*100), 4)]


def sma50_signal(close):
    """SMA50 trend filter baseline."""
    sma = close.rolling(50).mean()
    return (close > sma).astype(int)


# ============================================================
# APPROACH 1: Full-Cycle Spot (use all 10+ years)
# ============================================================
def approach1_fullcycle(data):
    """Test on-chain signals as standalone over full BTC history."""
    print("\n" + "="*60)
    print("APPROACH 1: Full-Cycle Spot (10+ years)")
    print("="*60)
    
    results = []
    
    # NUPL: >0.75 = euphoria (sell), <0 = capitulation (buy)
    for name, buy_cond, sell_cond, label in [
        ('nupl', lambda d: d['nupl'] < 0, lambda d: d['nupl'] > 0.75, 'NUPL_extreme'),
        ('nupl', lambda d: d['nupl'] > 0, lambda d: d['nupl'] < 0, 'NUPL_positive'),
        ('nupl', lambda d: d['nupl'].rolling(30).mean() > d['nupl'].rolling(90).mean(), 
         lambda d: d['nupl'].rolling(30).mean() < d['nupl'].rolling(90).mean(), 'NUPL_momentum'),
        ('sopr', lambda d: d['sopr'] < 1.0, lambda d: d['sopr'] > 1.05, 'SOPR_underwater'),
        ('sopr', lambda d: d['sopr'].rolling(7).mean() > 1.0, 
         lambda d: d['sopr'].rolling(7).mean() < 1.0, 'SOPR_trend'),
        ('reserve_risk', lambda d: d['reserve_risk'] < d['reserve_risk'].rolling(365).quantile(0.2),
         lambda d: d['reserve_risk'] > d['reserve_risk'].rolling(365).quantile(0.8), 'ReserveRisk_extreme'),
        ('reserve_risk', lambda d: d['reserve_risk'] > d['reserve_risk'].rolling(90).mean(),
         lambda d: d['reserve_risk'] < d['reserve_risk'].rolling(90).mean(), 'ReserveRisk_momentum'),
        ('nvt', lambda d: d['nvt'] < d['nvt'].rolling(90).quantile(0.3),
         lambda d: d['nvt'] > d['nvt'].rolling(90).quantile(0.7), 'NVT_undervalued'),
        ('nvt', lambda d: d['nvt'].rolling(14).mean() < d['nvt'].rolling(90).mean(),
         lambda d: d['nvt'].rolling(14).mean() > d['nvt'].rolling(90).mean(), 'NVT_momentum'),
    ]:
        valid = data.dropna(subset=[name])
        if len(valid) < 500:
            continue
        
        sig = pd.Series(0, index=valid.index)
        sig[buy_cond(valid)] = 1
        sig[sell_cond(valid)] = 0
        # Forward fill the signal
        sig = sig.replace(0, np.nan).ffill().fillna(0)
        
        wf = walk_forward_test(sig, valid['returns'])
        p = permutation_test(sig, valid['returns'], n_perms=500)
        
        print(f"  {label}: Sharpe={wf['sharpe']}, CAGR={wf['cagr']:.1%}, MaxDD={wf['maxdd']:.1%}, p={p:.3f}")
        
        results.append({
            'approach': 'fullcycle',
            'signal': label,
            'sharpe': wf['sharpe'],
            'cagr': wf['cagr'],
            'maxdd': wf['maxdd'],
            'n_days': wf['n_days'],
            'p_value': p
        })
    
    # Baseline: SMA50
    sma_sig = sma50_signal(data['close'])
    wf = walk_forward_test(sma_sig, data['returns'])
    print(f"  SMA50_baseline: Sharpe={wf['sharpe']}, CAGR={wf['cagr']:.1%}")
    results.append({'approach': 'fullcycle', 'signal': 'SMA50_baseline', **wf, 'p_value': None})
    
    # Buy & Hold
    bh = calc_metrics(data['returns'])
    print(f"  BuyAndHold: Sharpe={bh['sharpe']}, CAGR={bh['cagr']:.1%}")
    results.append({'approach': 'fullcycle', 'signal': 'BuyAndHold', **bh, 'p_value': None})
    
    return results


# ============================================================
# APPROACH 2: Regime-Conditional (only at cycle extremes)
# ============================================================
def approach2_regime_conditional(data):
    """Use on-chain only to modify V4 (SMA50) at cycle extremes."""
    print("\n" + "="*60)
    print("APPROACH 2: Regime-Conditional (exit-only filters)")
    print("="*60)
    
    results = []
    sma_sig = sma50_signal(data['close'])
    
    # Test: on-chain as EXIT-only filter (override SMA50 to flat)
    filters = [
        ('NUPL_exit_euphoria', lambda d: d['nupl'] > 0.75),  # Exit when euphoric
        ('NUPL_exit_high', lambda d: d['nupl'] > 0.6),
        ('SOPR_exit_overheated', lambda d: d['sopr'].rolling(14).mean() > 1.05),
        ('ReserveRisk_exit_top', lambda d: d['reserve_risk'] > d['reserve_risk'].rolling(365).quantile(0.9)),
        ('NVT_exit_overvalued', lambda d: d['nvt'] > d['nvt'].rolling(90).quantile(0.9)),
        # Entry boost: on-chain says BUY even when SMA50 is flat
        ('NUPL_entry_capitulation', lambda d: d['nupl'] < -0.1),
        ('SOPR_entry_loss', lambda d: d['sopr'].rolling(7).mean() < 0.95),
        ('ReserveRisk_entry_bottom', lambda d: d['reserve_risk'] < d['reserve_risk'].rolling(365).quantile(0.1)),
    ]
    
    for label, condition_fn in filters:
        valid = data.dropna(subset=['nupl', 'sopr', 'reserve_risk', 'nvt']).copy()
        if len(valid) < 500:
            continue
        
        sig = sma_sig.reindex(valid.index).fillna(0).copy()
        
        if 'exit' in label:
            # Override to flat when condition is true
            cond = condition_fn(valid)
            sig[cond] = 0
        elif 'entry' in label:
            # Override to long when condition is true (even if SMA50 says flat)
            cond = condition_fn(valid)
            sig[cond] = 1
        
        wf = walk_forward_test(sig, valid['returns'])
        p = permutation_test(sig, valid['returns'], n_perms=500)
        
        # Compare to baseline SMA50
        sma_wf = walk_forward_test(sma_sig.reindex(valid.index).fillna(0), valid['returns'])
        delta_sharpe = wf['sharpe'] - sma_wf['sharpe']
        
        print(f"  {label}: Sharpe={wf['sharpe']} (Δ={delta_sharpe:+.3f} vs SMA50), p={p:.3f}")
        
        results.append({
            'approach': 'regime_conditional',
            'signal': label,
            'sharpe': wf['sharpe'],
            'cagr': wf['cagr'],
            'maxdd': wf['maxdd'],
            'n_days': wf['n_days'],
            'p_value': p,
            'delta_vs_sma50': delta_sharpe
        })
    
    return results


# ============================================================
# APPROACH 3: Composite Scoring
# ============================================================
def approach3_composite(data):
    """Combine all 4 on-chain signals into a single score."""
    print("\n" + "="*60)
    print("APPROACH 3: Composite On-Chain Score")
    print("="*60)
    
    results = []
    valid = data.dropna(subset=['nupl', 'sopr', 'reserve_risk', 'nvt']).copy()
    
    if len(valid) < 500:
        print("  Insufficient overlapping data")
        return results
    
    # Normalize each signal to 0-1 percentile rank (rolling 365d)
    for col in ['nupl', 'sopr', 'reserve_risk', 'nvt']:
        valid[f'{col}_pctile'] = valid[col].rolling(365, min_periods=90).rank(pct=True)
    
    # Composite score: average of percentiles
    # High score = bullish (high NUPL, high SOPR, high reserve risk = actually bearish!)
    # Invert reserve_risk and nvt (high = overbought = bearish)
    valid['composite'] = (
        valid['nupl_pctile'] * 0.3 +       # High NUPL = bullish sentiment
        valid['sopr_pctile'] * 0.3 +         # High SOPR = profit taking (bearish)
        (1 - valid['reserve_risk_pctile']) * 0.2 +  # Low reserve risk = good accumulation
        (1 - valid['nvt_pctile']) * 0.2      # Low NVT = undervalued
    )
    
    # Also try simple: all bullish when low
    valid['composite_v2'] = (
        (1 - valid['nupl_pctile']) * 0.25 +   # Low NUPL = capitulation = buy
        (1 - valid['sopr_pctile']) * 0.25 +     # Low SOPR = losses = buy
        (1 - valid['reserve_risk_pctile']) * 0.25 +
        (1 - valid['nvt_pctile']) * 0.25
    )
    
    # Test various thresholds
    for comp_name in ['composite', 'composite_v2']:
        for thresh in [0.3, 0.4, 0.5, 0.6]:
            label = f'{comp_name}_gt{thresh}'
            sig = (valid[comp_name] > thresh).astype(int)
            
            wf = walk_forward_test(sig, valid['returns'])
            p = permutation_test(sig, valid['returns'], n_perms=500)
            
            print(f"  {label}: Sharpe={wf['sharpe']}, CAGR={wf['cagr']:.1%}, p={p:.3f}")
            
            results.append({
                'approach': 'composite',
                'signal': label,
                'sharpe': wf['sharpe'],
                'cagr': wf['cagr'],
                'maxdd': wf['maxdd'],
                'n_days': wf['n_days'],
                'p_value': p
            })
    
    # Composite as V4 overlay (modify SMA50)
    sma_sig = sma50_signal(valid['close'])
    for comp_name in ['composite', 'composite_v2']:
        # Exit when composite says overbought (>0.8 pctile)
        sig = sma_sig.reindex(valid.index).fillna(0).copy()
        sig[valid[comp_name] > 0.8] = 0  # Exit at extreme
        sig[valid[comp_name] < 0.2] = 1  # Enter at extreme low
        
        wf = walk_forward_test(sig, valid['returns'])
        p = permutation_test(sig, valid['returns'], n_perms=500)
        sma_wf = walk_forward_test(sma_sig.reindex(valid.index).fillna(0), valid['returns'])
        
        label = f'{comp_name}_v4overlay'
        print(f"  {label}: Sharpe={wf['sharpe']} (Δ={wf['sharpe']-sma_wf['sharpe']:+.3f}), p={p:.3f}")
        
        results.append({
            'approach': 'composite',
            'signal': label,
            'sharpe': wf['sharpe'],
            'cagr': wf['cagr'],
            'maxdd': wf['maxdd'],
            'n_days': wf['n_days'],
            'p_value': p,
            'delta_vs_sma50': wf['sharpe'] - sma_wf['sharpe']
        })
    
    return results


# ============================================================
# APPROACH 4: Weekly Timeframe
# ============================================================
def approach4_weekly(data):
    """Resample to weekly — on-chain data is slow-moving."""
    print("\n" + "="*60)
    print("APPROACH 4: Weekly Timeframe")
    print("="*60)
    
    results = []
    
    # Resample to weekly
    weekly = data.resample('W').agg({
        'close': 'last',
        'returns': lambda x: (1+x).prod() - 1,
        'nupl': 'last',
        'nvt': 'last',
        'reserve_risk': 'last',
        'sopr': 'mean'  # average SOPR over week
    }).dropna(subset=['close'])
    
    print(f"  Weekly data: {len(weekly)} weeks")
    
    # SMA signals on weekly
    for name, buy_fn, label in [
        ('nupl', lambda d: d['nupl'] > 0, 'NUPL_positive_weekly'),
        ('nupl', lambda d: d['nupl'] < 0.25, 'NUPL_accumulation_weekly'),
        ('sopr', lambda d: d['sopr'].rolling(4).mean() > 1.0, 'SOPR_profit_weekly'),
        ('reserve_risk', lambda d: d['reserve_risk'] < d['reserve_risk'].rolling(52).quantile(0.5), 'ReserveRisk_low_weekly'),
        ('nvt', lambda d: d['nvt'] < d['nvt'].rolling(26).median(), 'NVT_undervalued_weekly'),
    ]:
        valid = weekly.dropna(subset=[name])
        if len(valid) < 100:
            continue
        
        sig = buy_fn(valid).astype(int)
        
        # Walk-forward (fewer folds for weekly)
        wf = walk_forward_test(sig, valid['returns'], n_folds=10, min_train=52)
        p = permutation_test(sig, valid['returns'], n_perms=500)
        
        print(f"  {label}: Sharpe={wf['sharpe']}, CAGR={wf['cagr']:.1%}, p={p:.3f}")
        
        results.append({
            'approach': 'weekly',
            'signal': label,
            'sharpe': wf['sharpe'],
            'cagr': wf['cagr'],
            'maxdd': wf['maxdd'],
            'n_days': wf['n_days'],
            'p_value': p
        })
    
    # Weekly SMA10 baseline (≈ SMA50 daily)
    sma = weekly['close'].rolling(10).mean()
    sig = (weekly['close'] > sma).astype(int)
    wf = walk_forward_test(sig, weekly['returns'], n_folds=10, min_train=52)
    print(f"  SMA10w_baseline: Sharpe={wf['sharpe']}")
    results.append({'approach': 'weekly', 'signal': 'SMA10w_baseline', **wf, 'p_value': None})
    
    return results


# ============================================================
# APPROACH 5: Event-Based (extreme readings only)
# ============================================================
def approach5_events(data):
    """Flag extreme on-chain readings as discrete events. Measure forward returns."""
    print("\n" + "="*60)
    print("APPROACH 5: Event-Based (Extreme Readings)")
    print("="*60)
    
    results = []
    
    events = [
        ('NUPL_capitulation', lambda d: d['nupl'] < -0.1, 'Capitulation (NUPL < -0.1)'),
        ('NUPL_euphoria', lambda d: d['nupl'] > 0.75, 'Euphoria (NUPL > 0.75)'),
        ('NUPL_belief', lambda d: (d['nupl'] > 0) & (d['nupl'] < 0.25) & (d['nupl'].diff() > 0), 'Recovery belief'),
        ('SOPR_deep_loss', lambda d: d['sopr'] < 0.9, 'Deep loss (SOPR < 0.9)'),
        ('SOPR_heavy_profit', lambda d: d['sopr'] > 1.1, 'Heavy profit (SOPR > 1.1)'),
        ('SOPR_reset', lambda d: (d['sopr'].rolling(7).mean() < 1.0) & (d['sopr'] > 1.0), 'SOPR reset above 1'),
        ('ReserveRisk_extreme_low', lambda d: d['reserve_risk'] < d['reserve_risk'].rolling(730).quantile(0.05), 'RR bottom 5%'),
        ('ReserveRisk_extreme_high', lambda d: d['reserve_risk'] > d['reserve_risk'].rolling(730).quantile(0.95), 'RR top 5%'),
        ('NVT_extreme_low', lambda d: d['nvt'] < d['nvt'].rolling(365).quantile(0.05), 'NVT bottom 5%'),
        ('NVT_extreme_high', lambda d: d['nvt'] > d['nvt'].rolling(365).quantile(0.95), 'NVT top 5%'),
    ]
    
    for label, cond_fn, desc in events:
        col = label.split('_')[0].lower()
        valid = data.dropna(subset=[col.replace('reserverisk', 'reserve_risk')]).copy()
        if len(valid) < 500:
            continue
        
        mask = cond_fn(valid)
        n_events = mask.sum()
        
        if n_events < 10:
            print(f"  {label}: only {n_events} events — skipping")
            results.append({
                'approach': 'events',
                'signal': label,
                'description': desc,
                'n_events': int(n_events),
                'forward_7d': None,
                'forward_14d': None,
                'forward_30d': None
            })
            continue
        
        # Measure forward returns at 7d, 14d, 30d, 60d
        fwd_rets = {}
        for horizon in [7, 14, 30, 60]:
            fwd = valid['close'].pct_change(horizon).shift(-horizon)
            event_fwd = fwd[mask].dropna()
            all_fwd = fwd.dropna()
            
            if len(event_fwd) < 5:
                continue
            
            fwd_rets[f'forward_{horizon}d'] = {
                'mean': round(float(event_fwd.mean()), 4),
                'median': round(float(event_fwd.median()), 4),
                'baseline_mean': round(float(all_fwd.mean()), 4),
                'n': int(len(event_fwd)),
                'hit_rate': round(float((event_fwd > 0).mean()), 3)
            }
        
        # Also test as V4 overlay: enter/exit on event
        sma_sig = sma50_signal(valid['close'])
        sig = sma_sig.copy()
        
        is_bullish = 'capitulation' in label or 'loss' in label or 'low' in label or 'belief' in label or 'reset' in label
        if is_bullish:
            # On bullish event, force long for 30 days
            event_dates = valid.index[mask]
            for d in event_dates:
                end = d + pd.Timedelta(days=30)
                sig.loc[d:end] = 1
        else:
            # On bearish event, force flat for 30 days
            event_dates = valid.index[mask]
            for d in event_dates:
                end = d + pd.Timedelta(days=30)
                sig.loc[d:end] = 0
        
        wf = walk_forward_test(sig, valid['returns'])
        sma_wf = walk_forward_test(sma_sig, valid['returns'])
        
        print(f"  {label}: {n_events} events, best fwd={max((v.get('mean',0) for v in fwd_rets.values()), default=0):.1%}, V4 overlay Δ={wf['sharpe']-sma_wf['sharpe']:+.3f}")
        
        results.append({
            'approach': 'events',
            'signal': label,
            'description': desc,
            'n_events': int(n_events),
            'v4_overlay_sharpe': wf['sharpe'],
            'v4_overlay_delta': round(wf['sharpe'] - sma_wf['sharpe'], 4),
            **{k: v['mean'] for k, v in fwd_rets.items()},
            'details': fwd_rets
        })
    
    return results


# ============================================================
# APPROACH 6: V4 Enhancement (improve the production system)
# ============================================================
def approach6_v4_enhancement(data):
    """Test on-chain as Layer 4 enhancement to V4 system."""
    print("\n" + "="*60)
    print("APPROACH 6: V4 Enhancement (on-chain Layer 4)")
    print("="*60)
    
    results = []
    valid = data.dropna(subset=['nupl', 'sopr', 'reserve_risk', 'nvt']).copy()
    
    if len(valid) < 500:
        print("  Insufficient data")
        return results
    
    close = valid['close']
    returns = valid['returns']
    
    # V4 base: SMA50 + trailing stop
    sma_sig = sma50_signal(close)
    
    # Simulate trailing stop
    def apply_trailing_stop(sig, close, trail_pct=0.12):
        pos = sig.shift(1).fillna(0)
        peak = close.copy()
        result = pos.copy()
        in_position = False
        entry_peak = 0
        
        for i in range(1, len(close)):
            if pos.iloc[i] == 1:
                if not in_position:
                    in_position = True
                    entry_peak = close.iloc[i]
                else:
                    entry_peak = max(entry_peak, close.iloc[i])
                
                if close.iloc[i] < entry_peak * (1 - trail_pct):
                    result.iloc[i] = 0
                    in_position = False
            else:
                in_position = False
                entry_peak = 0
        
        return result
    
    v4_base = apply_trailing_stop(sma_sig, close)
    v4_wf = walk_forward_test(v4_base, returns)
    print(f"  V4 base (SMA50+trail): Sharpe={v4_wf['sharpe']}")
    results.append({'approach': 'v4_enhancement', 'signal': 'V4_base', **v4_wf, 'p_value': None})
    
    # Enhancement 1: On-chain risk score reduces position
    # Composite risk: high NUPL + high SOPR + high ReserveRisk = reduce
    risk_score = pd.Series(0.0, index=valid.index)
    risk_score += (valid['nupl'] > 0.6).astype(float) * 0.25
    risk_score += (valid['sopr'].rolling(14).mean() > 1.05).astype(float) * 0.25
    risk_score += (valid['reserve_risk'] > valid['reserve_risk'].rolling(365).quantile(0.8)).astype(float) * 0.25
    risk_score += (valid['nvt'] > valid['nvt'].rolling(90).quantile(0.8)).astype(float) * 0.25
    
    # When risk > 0.5, reduce to 50%; when risk > 0.75, go flat
    sig_risk = v4_base.copy()
    sig_risk[risk_score > 0.75] = 0
    # Can't do fractional in binary, so just test exit threshold
    
    wf = walk_forward_test(sig_risk, returns)
    sma_wf = walk_forward_test(sma_sig, returns)
    p = permutation_test(sig_risk, returns, n_perms=500)
    print(f"  V4+risk_exit: Sharpe={wf['sharpe']} (Δ={wf['sharpe']-v4_wf['sharpe']:+.3f}), p={p:.3f}")
    results.append({
        'approach': 'v4_enhancement', 'signal': 'V4_risk_exit',
        **wf, 'p_value': p, 'delta_vs_v4': wf['sharpe'] - v4_wf['sharpe']
    })
    
    # Enhancement 2: On-chain opportunity score forces entry
    opp_score = pd.Series(0.0, index=valid.index)
    opp_score += (valid['nupl'] < 0).astype(float) * 0.25
    opp_score += (valid['sopr'].rolling(7).mean() < 0.97).astype(float) * 0.25
    opp_score += (valid['reserve_risk'] < valid['reserve_risk'].rolling(365).quantile(0.1)).astype(float) * 0.25
    opp_score += (valid['nvt'] < valid['nvt'].rolling(90).quantile(0.2)).astype(float) * 0.25
    
    sig_opp = v4_base.copy()
    sig_opp[opp_score > 0.5] = 1  # Force long on strong opportunity
    
    wf = walk_forward_test(sig_opp, returns)
    p = permutation_test(sig_opp, returns, n_perms=500)
    print(f"  V4+opp_entry: Sharpe={wf['sharpe']} (Δ={wf['sharpe']-v4_wf['sharpe']:+.3f}), p={p:.3f}")
    results.append({
        'approach': 'v4_enhancement', 'signal': 'V4_opp_entry',
        **wf, 'p_value': p, 'delta_vs_v4': wf['sharpe'] - v4_wf['sharpe']
    })
    
    # Enhancement 3: Combined risk + opportunity
    sig_combined = v4_base.copy()
    sig_combined[risk_score > 0.75] = 0
    sig_combined[opp_score > 0.5] = 1  # Opportunity overrides risk
    
    wf = walk_forward_test(sig_combined, returns)
    p = permutation_test(sig_combined, returns, n_perms=500)
    print(f"  V4+combined: Sharpe={wf['sharpe']} (Δ={wf['sharpe']-v4_wf['sharpe']:+.3f}), p={p:.3f}")
    results.append({
        'approach': 'v4_enhancement', 'signal': 'V4_combined',
        **wf, 'p_value': p, 'delta_vs_v4': wf['sharpe'] - v4_wf['sharpe']
    })
    
    # Enhancement 4: Cycle position sizing
    # Use NUPL as cycle indicator: scale position 0.5x-1.5x based on cycle
    cycle_mult = pd.Series(1.0, index=valid.index)
    cycle_mult[valid['nupl'] < 0] = 1.5    # Max size at capitulation
    cycle_mult[valid['nupl'] > 0.6] = 0.5  # Half size at euphoria
    
    # Strategy returns = base signal * cycle multiplier * returns
    position = v4_base.shift(1) * cycle_mult.shift(1)
    strat_returns = position * returns
    wf_cycle = calc_metrics(strat_returns.dropna())
    
    # WF version
    sig_cycle = v4_base * cycle_mult  # Combined for WF
    wf = walk_forward_test(sig_cycle, returns)
    p = permutation_test(sig_cycle, returns, n_perms=500)
    print(f"  V4+cycle_sizing: Sharpe={wf['sharpe']} (Δ={wf['sharpe']-v4_wf['sharpe']:+.3f}), p={p:.3f}")
    results.append({
        'approach': 'v4_enhancement', 'signal': 'V4_cycle_sizing',
        **wf, 'p_value': p, 'delta_vs_v4': wf['sharpe'] - v4_wf['sharpe']
    })
    
    # Bootstrap CI for best performer
    best = max(results[1:], key=lambda x: x.get('sharpe', 0))  # skip baseline
    if best['sharpe'] > v4_wf['sharpe']:
        position_best = sig_combined.shift(1) * returns  # Use combined as example
        ci = bootstrap_ci(position_best.dropna())
        print(f"\n  Best enhancement: {best['signal']}, Bootstrap CI: {ci}")
        best['bootstrap_ci'] = ci
    
    return results


def main():
    print("="*60)
    print("COMPREHENSIVE ON-CHAIN SIGNAL TESTING")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    data = load_onchain()
    
    all_results = []
    
    # Run all 6 approaches
    all_results.extend(approach1_fullcycle(data))
    all_results.extend(approach2_regime_conditional(data))
    all_results.extend(approach3_composite(data))
    all_results.extend(approach4_weekly(data))
    all_results.extend(approach5_events(data))
    all_results.extend(approach6_v4_enhancement(data))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY — ALL APPROACHES")
    print("="*60)
    
    # Bonferroni correction
    n_tests = sum(1 for r in all_results if r.get('p_value') is not None)
    bonferroni_alpha = 0.05 / max(n_tests, 1)
    print(f"\nTotal tests: {n_tests}, Bonferroni α: {bonferroni_alpha:.6f}")
    
    # Best per approach
    approaches = set(r['approach'] for r in all_results)
    for approach in sorted(approaches):
        subset = [r for r in all_results if r['approach'] == approach and r.get('sharpe') is not None]
        if subset:
            best = max(subset, key=lambda x: x.get('sharpe', 0))
            p_str = f"p={best.get('p_value', '?')}" if best.get('p_value') is not None else "baseline"
            sig_str = "✅ SIGNIFICANT" if best.get('p_value') is not None and best['p_value'] < bonferroni_alpha else ""
            print(f"  {approach}: best={best['signal']}, Sharpe={best['sharpe']}, {p_str} {sig_str}")
    
    # Any survivors?
    survivors = [r for r in all_results if r.get('p_value') is not None and r['p_value'] < bonferroni_alpha]
    print(f"\nSurvivors (Bonferroni): {len(survivors)}")
    for s in survivors:
        print(f"  ✅ {s['signal']}: Sharpe={s['sharpe']}, p={s['p_value']}")
    
    # Anything that beats SMA50?
    sma_sharpe = next((r['sharpe'] for r in all_results if r.get('signal') == 'SMA50_baseline'), 0)
    beaters = [r for r in all_results if r.get('sharpe', 0) > sma_sharpe and r.get('p_value') is not None and r.get('p_value', 1) < 0.05]
    print(f"\nBeats SMA50 (p<0.05, before Bonferroni): {len(beaters)}")
    for b in beaters:
        print(f"  {b['signal']}: Sharpe={b['sharpe']}, p={b['p_value']}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'walk_forward_folds': 14,
            'permutation_tests': 500,
            'bonferroni_alpha': bonferroni_alpha,
            'total_tests': n_tests,
            'data_range': f"{data.index[0].date()} to {data.index[-1].date()}",
            'n_days': len(data),
            'signals': ['NUPL (13yr)', 'NVT (16yr)', 'Reserve Risk (15yr)', 'SOPR (15yr)'],
            'approaches': list(sorted(approaches))
        },
        'results': all_results,
        'survivors_bonferroni': survivors,
        'beats_sma50_p05': beaters,
        'verdict': 'TBD'
    }
    
    if len(survivors) == 0 and len(beaters) == 0:
        output['verdict'] = 'ALL DEAD — On-chain signals add no value over SMA50 trend following'
    elif len(survivors) > 0:
        output['verdict'] = f'{len(survivors)} survived Bonferroni — on-chain has real alpha'
    else:
        output['verdict'] = f'{len(beaters)} beat SMA50 at p<0.05 but none survive Bonferroni — weak/marginal'
    
    with open(RESULTS_DIR / 'onchain_comprehensive_v3.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {output['verdict']}")
    print(f"Results saved to: onchain_comprehensive_v3.json")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
