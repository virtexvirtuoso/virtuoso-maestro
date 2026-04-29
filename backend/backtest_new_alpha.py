#!/usr/bin/env python3
"""Three new alpha research strategies with walk-forward validation."""

import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')
p = print

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
DERIV_DIR = os.path.expanduser("~/Desktop/maestro/data/derivatives")
OUT_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

TOKENS = ['arb','avax','btc','eth','fet','inj','link','op','sol','sui']
N_FOLDS = 10

# ── Load Data ──────────────────────────────────────────────────────────
def load_ohlcv():
    frames = {}
    for t in TOKENS:
        df = pd.read_csv(f"{DATA_DIR}/binance_{t}_usdt_1d.csv", parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        frames[t] = df
    return frames

def load_funding():
    frames = {}
    for t in TOKENS:
        fp = f"{DERIV_DIR}/{t}_funding.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=['timestamp'])
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date')['fundingRate'].mean().reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.set_index('date').sort_index()
        frames[t] = daily
    return frames

# ── Helpers ────────────────────────────────────────────────────────────
def calc_sharpe(returns):
    if len(returns) < 30 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(365)

def calc_cagr(returns):
    cum = (1 + returns).prod()
    n_years = len(returns) / 365
    if n_years <= 0 or cum <= 0:
        return 0.0
    return cum ** (1/n_years) - 1

def calc_maxdd(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def walk_forward_expanding(returns, n_folds=N_FOLDS):
    """Expanding window WF. Returns list of OOS fold returns."""
    n = len(returns)
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos_returns = []
    for i in range(n_folds):
        oos_start = min_train + i * fold_size
        oos_end = min(oos_start + fold_size, n)
        if oos_start >= n:
            break
        oos_returns.append(returns.iloc[oos_start:oos_end])
    return oos_returns

def wf_stats(returns):
    """Get OOS Sharpe and p-value from walk-forward."""
    folds = walk_forward_expanding(returns)
    if not folds:
        return 0.0, 1.0
    fold_sharpes = [calc_sharpe(f) for f in folds]
    oos_all = pd.concat(folds)
    oos_sharpe = calc_sharpe(oos_all)
    # t-test: are fold sharpes > 0?
    if len(fold_sharpes) >= 3:
        t_stat, p_val = stats.ttest_1samp(fold_sharpes, 0)
        p_val = p_val / 2 if t_stat > 0 else 1.0  # one-sided
    else:
        p_val = 1.0
    return oos_sharpe, p_val

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ── Build aligned returns matrix ───────────────────────────────────────
def build_returns_matrix(ohlcv):
    closes = pd.DataFrame({t: ohlcv[t]['close'] for t in TOKENS})
    closes = closes.sort_index().dropna(how='all')
    returns = closes.pct_change()
    return closes, returns

# ══════════════════════════════════════════════════════════════════════
# STRATEGY 1: Cross-Asset Relative Momentum
# ══════════════════════════════════════════════════════════════════════
def strategy1_momentum(closes, returns):
    p("\n" + "="*70, flush=True)
    p("=== STRATEGY 1: Cross-Asset Relative Momentum (Long/Short Quintiles) ===", flush=True)
    p("="*70, flush=True)
    
    lookbacks = [7, 14, 30, 60, 90]
    rebalances = [1, 7, 14]
    results = []
    
    p(f"{'Params':<20} | {'Full Sharpe':>11} | {'OOS Sharpe':>10} | {'CAGR':>8} | {'MaxDD':>8} | {'WF p-val':>8} | Sig", flush=True)
    p("-"*95, flush=True)
    
    n_tokens = len(TOKENS)
    q_size = max(n_tokens // 5, 1)  # quintile = 20%
    
    for lb in lookbacks:
        # trailing returns
        trail_ret = closes.pct_change(lb)
        
        for reb in rebalances:
            # For each day, rank tokens, long top quintile, short bottom
            strat_rets = []
            dates = trail_ret.dropna(how='all').index[lb:]
            
            last_reb = -999
            long_tokens = []
            short_tokens = []
            
            for i, dt in enumerate(dates):
                row = trail_ret.loc[dt].dropna()
                daily_ret = returns.loc[dt].dropna()
                
                if len(row) < 5:
                    strat_rets.append(0.0)
                    continue
                
                # Rebalance check
                if i - last_reb >= reb:
                    ranked = row.sort_values(ascending=False)
                    long_tokens = ranked.index[:q_size].tolist()
                    short_tokens = ranked.index[-q_size:].tolist()
                    last_reb = i
                    commission = 0.002  # 20bps
                else:
                    commission = 0.0
                
                # Calculate spread return
                long_ret = daily_ret.reindex(long_tokens).mean() if long_tokens else 0.0
                short_ret = daily_ret.reindex(short_tokens).mean() if short_tokens else 0.0
                spread = long_ret - short_ret - commission
                strat_rets.append(spread if not np.isnan(spread) else 0.0)
            
            sr = pd.Series(strat_rets, index=dates)
            full_sharpe = calc_sharpe(sr)
            cagr = calc_cagr(sr)
            maxdd = calc_maxdd(sr)
            oos_sharpe, p_val = wf_stats(sr)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            
            label = f"lb={lb},reb={reb}"
            p(f"{label:<20} | {full_sharpe:>11.3f} | {oos_sharpe:>10.3f} | {cagr:>7.1%} | {maxdd:>7.1%} | {p_val:>8.4f} | {sig}", flush=True)
            results.append({
                'strategy': 'CrossAssetMomentum', 'params': label,
                'lookback': lb, 'rebalance': reb,
                'full_sharpe': round(full_sharpe,4), 'oos_sharpe': round(oos_sharpe,4),
                'cagr': round(cagr,4), 'maxdd': round(maxdd,4), 'p_value': round(p_val,4)
            })
    return results

# ══════════════════════════════════════════════════════════════════════
# STRATEGY 2: Funding Rate Carry
# ══════════════════════════════════════════════════════════════════════
def strategy2_funding(ohlcv, funding):
    p("\n" + "="*70, flush=True)
    p("=== STRATEGY 2: Funding Rate Carry ===", flush=True)
    p("="*70, flush=True)
    
    # Build aligned daily returns and funding
    closes = pd.DataFrame({t: ohlcv[t]['close'] for t in TOKENS})
    returns = closes.pct_change()
    fund_df = pd.DataFrame({t: funding[t]['fundingRate'] for t in TOKENS if t in funding})
    
    # Align indices
    common_idx = returns.index.intersection(fund_df.index)
    returns = returns.loc[common_idx]
    fund_df = fund_df.loc[common_idx]
    
    thresholds = [0.0001, 0.0003, 0.0005, 0.001]
    results = []
    
    p(f"{'Params':<20} | {'Full Sharpe':>11} | {'OOS Sharpe':>10} | {'CAGR':>8} | {'MaxDD':>8} | {'WF p-val':>8} | Sig", flush=True)
    p("-"*95, flush=True)
    
    for thresh in thresholds:
        strat_rets = []
        for dt in common_idx:
            fr = fund_df.loc[dt].dropna()
            dr = returns.loc[dt].dropna()
            
            # Signals: short when funding high (collect), long when funding negative
            short_tokens = fr[fr > thresh].index
            long_tokens = fr[fr < -thresh].index
            
            n_pos = len(short_tokens) + len(long_tokens)
            if n_pos == 0:
                strat_rets.append(0.0)
                continue
            
            w = 1.0 / n_pos
            ret = 0.0
            # Short positions: -price_return + collect funding
            for t in short_tokens:
                if t in dr.index:
                    ret += w * (-dr[t] + fr[t])  # short P&L + funding income
            # Long positions: +price_return + collect (negative) funding
            for t in long_tokens:
                if t in dr.index:
                    ret += w * (dr[t] - fr[t])  # long P&L + funding income
            
            ret -= 0.001 * (n_pos > 0)  # 10bps commission approx
            strat_rets.append(ret if not np.isnan(ret) else 0.0)
        
        sr = pd.Series(strat_rets, index=common_idx)
        full_sharpe = calc_sharpe(sr)
        cagr = calc_cagr(sr)
        maxdd = calc_maxdd(sr)
        oos_sharpe, p_val = wf_stats(sr)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        
        label = f"thresh={thresh}"
        p(f"{label:<20} | {full_sharpe:>11.3f} | {oos_sharpe:>10.3f} | {cagr:>7.1%} | {maxdd:>7.1%} | {p_val:>8.4f} | {sig}", flush=True)
        results.append({
            'strategy': 'FundingCarry', 'params': label,
            'threshold': thresh,
            'full_sharpe': round(full_sharpe,4), 'oos_sharpe': round(oos_sharpe,4),
            'cagr': round(cagr,4), 'maxdd': round(maxdd,4), 'p_value': round(p_val,4)
        })
    return results

# ══════════════════════════════════════════════════════════════════════
# STRATEGY 3: Mean-Reversion RSI Cross-Sectional
# ══════════════════════════════════════════════════════════════════════
def strategy3_rsi(closes, returns):
    p("\n" + "="*70, flush=True)
    p("=== STRATEGY 3: Mean-Reversion RSI Cross-Sectional ===", flush=True)
    p("="*70, flush=True)
    
    rsi_periods = [7, 14, 21]
    results = []
    
    p(f"{'Params':<20} | {'Full Sharpe':>11} | {'OOS Sharpe':>10} | {'CAGR':>8} | {'MaxDD':>8} | {'WF p-val':>8} | Sig", flush=True)
    p("-"*95, flush=True)
    
    for rsi_p in rsi_periods:
        # Calculate RSI for all tokens
        rsi_df = pd.DataFrame({t: calc_rsi(closes[t], rsi_p) for t in TOKENS})
        
        strat_rets = []
        valid_dates = rsi_df.dropna(how='all').index[rsi_p+5:]
        
        for dt in valid_dates:
            rsi_row = rsi_df.loc[dt].dropna()
            daily_ret = returns.loc[dt].dropna()
            
            long_tokens = rsi_row[rsi_row < 30].index
            short_tokens = rsi_row[rsi_row > 70].index
            
            n_pos = len(long_tokens) + len(short_tokens)
            if n_pos == 0:
                strat_rets.append(0.0)
                continue
            
            w = 1.0 / n_pos
            ret = 0.0
            for t in long_tokens:
                if t in daily_ret.index:
                    ret += w * daily_ret[t]
            for t in short_tokens:
                if t in daily_ret.index:
                    ret += w * (-daily_ret[t])
            
            ret -= 0.002  # 20bps daily commission (daily rebalance)
            strat_rets.append(ret if not np.isnan(ret) else 0.0)
        
        sr = pd.Series(strat_rets, index=valid_dates)
        full_sharpe = calc_sharpe(sr)
        cagr = calc_cagr(sr)
        maxdd = calc_maxdd(sr)
        oos_sharpe, p_val = wf_stats(sr)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        
        label = f"RSI({rsi_p})"
        p(f"{label:<20} | {full_sharpe:>11.3f} | {oos_sharpe:>10.3f} | {cagr:>7.1%} | {maxdd:>7.1%} | {p_val:>8.4f} | {sig}", flush=True)
        results.append({
            'strategy': 'RSIMeanReversion', 'params': label,
            'rsi_period': rsi_p,
            'full_sharpe': round(full_sharpe,4), 'oos_sharpe': round(oos_sharpe,4),
            'cagr': round(cagr,4), 'maxdd': round(maxdd,4), 'p_value': round(p_val,4)
        })
    return results

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    p("Loading data...", flush=True)
    ohlcv = load_ohlcv()
    funding = load_funding()
    closes, returns = build_returns_matrix(ohlcv)
    p(f"Loaded {len(TOKENS)} tokens, {len(closes)} days ({closes.index[0].date()} to {closes.index[-1].date()})", flush=True)
    p(f"Funding data for: {list(funding.keys())}", flush=True)
    
    all_results = []
    
    r1 = strategy1_momentum(closes, returns)
    all_results.extend(r1)
    
    r2 = strategy2_funding(ohlcv, funding)
    all_results.extend(r2)
    
    r3 = strategy3_rsi(closes, returns)
    all_results.extend(r3)
    
    # Grand Summary
    p("\n" + "="*90, flush=True)
    p("GRAND SUMMARY: BEST RESULTS ACROSS ALL THREE STRATEGIES", flush=True)
    p("="*90, flush=True)
    
    df_res = pd.DataFrame(all_results)
    df_res = df_res.sort_values('oos_sharpe', ascending=False)
    
    p(f"{'Strategy':<25} | {'Params':<20} | {'OOS Sharpe':>10} | {'p-value':>8} | {'CAGR':>8} | {'MaxDD':>8}", flush=True)
    p("-"*95, flush=True)
    for _, row in df_res.head(10).iterrows():
        sig = " ⭐" if row['p_value'] < 0.05 else ""
        p(f"{row['strategy']:<25} | {row['params']:<20} | {row['oos_sharpe']:>10.3f} | {row['p_value']:>8.4f} | {row['cagr']:>7.1%} | {row['maxdd']:>7.1%}{sig}", flush=True)
    
    sig_results = df_res[df_res['p_value'] < 0.05]
    if len(sig_results) > 0:
        p(f"\n🔥 {len(sig_results)} strategies with p < 0.05!", flush=True)
    else:
        p("\n⚠️  No strategies reached p < 0.05 significance.", flush=True)
    
    # Save
    out_path = os.path.join(OUT_DIR, "new_alpha_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    p(f"\nResults saved to {out_path}", flush=True)
