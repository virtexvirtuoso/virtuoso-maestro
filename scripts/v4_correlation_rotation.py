#!/usr/bin/env python3
"""V4 Top 5 Correlation Analysis & Adaptive Rotation Backtest"""

import os, json, glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats

warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
OUT_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
os.makedirs(OUT_DIR, exist_ok=True)

V4_STATIC = ['sol', 'ftm', 'avax', 'bnb', 'sui']

# ============================================================
# DATA LOADING
# ============================================================
def load_all_tokens(min_days=730):
    """Load all tokens, return dict of DataFrames with >=min_days data."""
    tokens = {}
    for f in glob.glob(os.path.join(DATA_DIR, "binance_*_usdt_1d.csv")):
        basename = os.path.basename(f)
        if '_usdt_1d.csv' not in basename:
            continue
        name = basename.replace("binance_", "").replace("_usdt_1d.csv", "")
        df = pd.read_csv(f, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        if len(df) >= min_days:
            tokens[name] = df
    # Also load non-binance
    for f in glob.glob(os.path.join(DATA_DIR, "1000*_1d.csv")):
        name = os.path.basename(f).replace("_1d.csv", "").lower()
        df = pd.read_csv(f, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        if len(df) >= min_days:
            tokens[name] = df
    return tokens

def build_returns(tokens):
    """Build aligned daily returns DataFrame."""
    closes = {}
    for name, df in tokens.items():
        closes[name] = df['close']
    prices = pd.DataFrame(closes)
    returns = prices.pct_change().dropna(how='all')
    return prices, returns

# ============================================================
# PART 1: CORRELATION ANALYSIS
# ============================================================
def part1_correlation(tokens):
    print("=" * 70)
    print("PART 1: CORRELATION ANALYSIS — V4 Top 5")
    print("=" * 70)
    
    # Get V4 tokens + BTC
    v4_tokens = {k: v for k, v in tokens.items() if k in V4_STATIC}
    v4_tokens['btc'] = tokens['btc']
    
    prices, returns = build_returns(v4_tokens)
    v4_ret = returns[V4_STATIC].dropna()
    
    # 1. Pairwise correlation matrix
    print("\n📊 1. PAIRWISE RETURN CORRELATION MATRIX (full period)")
    corr = v4_ret.corr()
    print(corr.round(3).to_string())
    avg_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().mean()
    print(f"\n  Average pairwise correlation: {avg_corr:.3f}")
    
    # 2. Rolling 60d correlation
    print("\n📊 2. ROLLING 60d CORRELATION STATS")
    rolling_corrs = {}
    pairs = []
    for i, a in enumerate(V4_STATIC):
        for b in V4_STATIC[i+1:]:
            rc = v4_ret[a].rolling(60).corr(v4_ret[b])
            pair = f"{a}/{b}"
            rolling_corrs[pair] = rc
            pairs.append(pair)
    
    rc_df = pd.DataFrame(rolling_corrs)
    avg_rolling = rc_df.mean(axis=1).dropna()
    
    # Check if correlation spikes during drawdowns
    # Use BTC as market proxy, find drawdown periods
    btc_ret = returns['btc'].reindex(v4_ret.index)
    btc_cum = (1 + btc_ret).cumprod()
    btc_dd = btc_cum / btc_cum.cummax() - 1
    
    dd_threshold = -0.10  # 10% drawdown
    in_dd = btc_dd < dd_threshold
    
    corr_in_dd = avg_rolling[in_dd].mean()
    corr_no_dd = avg_rolling[~in_dd].mean()
    
    print(f"  Mean rolling correlation (normal):    {corr_no_dd:.3f}")
    print(f"  Mean rolling correlation (BTC DD>10%): {corr_in_dd:.3f}")
    print(f"  Correlation INCREASE during stress:    {corr_in_dd - corr_no_dd:+.3f}")
    if corr_in_dd > corr_no_dd:
        print("  ⚠️  Diversification FAILS when needed most!")
    
    print(f"\n  Rolling corr stats:")
    print(f"    Min:  {avg_rolling.min():.3f}")
    print(f"    Mean: {avg_rolling.mean():.3f}")
    print(f"    Max:  {avg_rolling.max():.3f}")
    print(f"    Std:  {avg_rolling.std():.3f}")
    
    # 3. Correlation during SMA50-long periods only
    print("\n📊 3. CORRELATION DURING SMA50-LONG PERIODS")
    # Compute SMA50 filter for each V4 token
    v4_prices = pd.DataFrame({k: tokens[k]['close'] for k in V4_STATIC})
    sma50 = v4_prices.rolling(50).mean()
    above_sma = (v4_prices > sma50)
    
    # All above SMA50 simultaneously
    all_long = above_sma.all(axis=1).reindex(v4_ret.index).fillna(False)
    any_long = above_sma.any(axis=1).reindex(v4_ret.index).fillna(False)
    
    if all_long.sum() > 60:
        corr_long = v4_ret[all_long].corr()
        avg_corr_long = corr_long.where(np.triu(np.ones(corr_long.shape), k=1).astype(bool)).stack().mean()
        print(f"  When ALL 5 above SMA50 ({all_long.sum()} days):")
        print(f"    Average pairwise correlation: {avg_corr_long:.3f}")
    
    if any_long.sum() > 60:
        corr_any = v4_ret[any_long].corr()
        avg_corr_any = corr_any.where(np.triu(np.ones(corr_any.shape), k=1).astype(bool)).stack().mean()
        print(f"  When ANY above SMA50 ({any_long.sum()} days):")
        print(f"    Average pairwise correlation: {avg_corr_any:.3f}")
    
    # 4. Beta to BTC
    print("\n📊 4. BETA TO BTC")
    btc_aligned = returns['btc'].reindex(v4_ret.index).dropna()
    common_idx = v4_ret.index.intersection(btc_aligned.index)
    
    betas = {}
    print(f"  {'Asset':<8} {'Beta':>8} {'R²':>8} {'Alpha(ann)':>12}")
    print(f"  {'-'*40}")
    for tok in V4_STATIC:
        y = v4_ret[tok].loc[common_idx].dropna()
        x = btc_aligned.loc[y.index]
        mask = x.notna() & y.notna()
        slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
        betas[tok] = {'beta': round(slope, 3), 'r2': round(r**2, 3), 'alpha_ann': round(intercept * 365, 3)}
        print(f"  {tok.upper():<8} {slope:>8.3f} {r**2:>8.3f} {intercept*365:>12.1%}")
    
    avg_beta = np.mean([b['beta'] for b in betas.values()])
    print(f"\n  Average beta: {avg_beta:.3f}")
    if avg_beta > 1.5:
        print("  ⚠️  Portfolio is essentially LEVERAGED BTC")
    elif avg_beta > 1.0:
        print("  ⚠️  Portfolio has HIGH beta to BTC")
    
    # 5. PCA
    print("\n📊 5. PRINCIPAL COMPONENT ANALYSIS")
    clean = v4_ret.dropna()
    pca = PCA()
    pca.fit(clean)
    var_explained = pca.explained_variance_ratio_
    
    for i, v in enumerate(var_explained):
        print(f"  PC{i+1}: {v:.1%} variance explained (cumulative: {var_explained[:i+1].sum():.1%})")
    
    print(f"\n  PC1 alone explains {var_explained[0]:.1%} — ", end="")
    if var_explained[0] > 0.7:
        print("portfolio is essentially ONE FACTOR (market)")
    elif var_explained[0] > 0.5:
        print("strong single-factor dominance")
    else:
        print("some genuine diversification exists")
    
    return {
        'correlation_matrix': corr.round(4).to_dict(),
        'avg_pairwise_corr': round(avg_corr, 4),
        'rolling_corr_normal': round(corr_no_dd, 4),
        'rolling_corr_stress': round(corr_in_dd, 4),
        'betas': betas,
        'avg_beta': round(avg_beta, 3),
        'pca_variance_explained': [round(v, 4) for v in var_explained.tolist()],
        'pc1_pct': round(var_explained[0], 4)
    }

# ============================================================
# PART 2: ADAPTIVE ROTATION
# ============================================================
def compute_sma(series, window):
    return series.rolling(window).mean()

def compute_atr(df, period=20):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def v4d_backtest(prices_dict, selected_tokens_by_period, rebal_dates, tokens_data):
    """
    Run V4d-style backtest with trailing stops, vol ceiling, DD breaker.
    prices_dict: {token: Series of close prices}
    selected_tokens_by_period: list of (date, [tokens]) for each rebalance
    """
    # Build a combined timeline
    all_dates = sorted(set().union(*[set(s.index) for s in prices_dict.values()]))
    all_dates = pd.DatetimeIndex(all_dates).sort_values()
    
    equity = pd.Series(1.0, index=all_dates)
    current_tokens = []
    positions = {}  # token -> {'entry': price, 'trail_stop': price, 'peak': price}
    daily_returns = pd.Series(0.0, index=all_dates)
    turnover_count = 0
    
    rebal_dict = {}
    for date, toks in selected_tokens_by_period:
        rebal_dict[date] = toks
    
    # For vol ceiling: use 20d realized vol, cap at 60% annualized
    VOL_CEIL = 0.60  # 60% annualized
    # DD breaker: if portfolio DD > 25%, go to cash
    DD_BREAK = -0.25
    
    prev_equity = 1.0
    peak_equity = 1.0
    in_cash = False
    cash_reentry_date = None
    
    for i, date in enumerate(all_dates):
        if i == 0:
            continue
        
        # Check rebalance
        if date in rebal_dict:
            new_tokens = rebal_dict[date]
            old_set = set(current_tokens)
            new_set = set(new_tokens)
            turnover_count += len(old_set.symmetric_difference(new_set))
            current_tokens = new_tokens
            # Reset positions
            positions = {}
            for tok in current_tokens:
                if tok in prices_dict and date in prices_dict[tok].index:
                    p = prices_dict[tok].loc[date]
                    atr_val = 0.05 * p  # default 5% if can't compute
                    if tok in tokens_data:
                        atr_s = compute_atr(tokens_data[tok])
                        if date in atr_s.index and not np.isnan(atr_s.loc[date]):
                            atr_val = atr_s.loc[date]
                    stop_dist = max(0.05 * p, min(0.20 * p, 2 * atr_val))
                    positions[tok] = {'entry': p, 'peak': p, 'trail_stop': p - stop_dist}
        
        if in_cash:
            # Re-enter after 21 days
            if cash_reentry_date and date >= cash_reentry_date:
                in_cash = False
            else:
                daily_returns.iloc[i] = 0
                equity.iloc[i] = prev_equity
                continue
        
        # Compute daily portfolio return
        if not current_tokens or not positions:
            daily_returns.iloc[i] = 0
            equity.iloc[i] = prev_equity
            continue
        
        weight = 1.0 / len(current_tokens) if current_tokens else 0
        port_ret = 0.0
        stopped_out = []
        
        for tok in list(positions.keys()):
            if tok not in prices_dict:
                continue
            prev_date = all_dates[i-1]
            if prev_date not in prices_dict[tok].index or date not in prices_dict[tok].index:
                continue
            
            p_prev = prices_dict[tok].loc[prev_date]
            p_now = prices_dict[tok].loc[date]
            
            # Check SMA50 filter
            if tok in prices_dict:
                sma = prices_dict[tok].rolling(50).mean()
                if date in sma.index and not np.isnan(sma.loc[date]):
                    if p_now < sma.loc[date]:
                        stopped_out.append(tok)
                        continue
            
            ret = (p_now / p_prev) - 1 if p_prev > 0 else 0
            
            # Vol ceiling: scale down if vol too high
            if tok in prices_dict:
                recent_ret = prices_dict[tok].pct_change().loc[:date].tail(20)
                if len(recent_ret) >= 10:
                    ann_vol = recent_ret.std() * np.sqrt(365)
                    if ann_vol > VOL_CEIL:
                        ret *= VOL_CEIL / ann_vol
            
            port_ret += weight * ret
            
            # Update trailing stop
            pos = positions[tok]
            if p_now > pos['peak']:
                pos['peak'] = p_now
                atr_val = 0.05 * p_now
                if tok in tokens_data:
                    atr_s = compute_atr(tokens_data[tok])
                    if date in atr_s.index and not np.isnan(atr_s.loc[date]):
                        atr_val = atr_s.loc[date]
                stop_dist = max(0.05 * p_now, min(0.20 * p_now, 2 * atr_val))
                pos['trail_stop'] = max(pos['trail_stop'], p_now - stop_dist)
            
            if p_now <= pos['trail_stop']:
                stopped_out.append(tok)
        
        for tok in stopped_out:
            if tok in positions:
                del positions[tok]
        
        daily_returns.iloc[i] = port_ret
        new_equity = prev_equity * (1 + port_ret)
        equity.iloc[i] = new_equity
        
        # DD breaker
        peak_equity = max(peak_equity, new_equity)
        dd = (new_equity / peak_equity) - 1
        if dd < DD_BREAK:
            in_cash = True
            cash_reentry_date = date + pd.Timedelta(days=21)
        
        prev_equity = new_equity
    
    return equity, daily_returns, turnover_count

def compute_metrics(equity, daily_returns):
    """Compute Sharpe, CAGR, MaxDD from equity curve."""
    total_days = (equity.index[-1] - equity.index[0]).days
    if total_days <= 0:
        return {'sharpe': 0, 'cagr': 0, 'maxdd': 0}
    
    final = equity.iloc[-1] / equity.iloc[0]
    years = total_days / 365.25
    cagr = final ** (1/years) - 1 if years > 0 else 0
    
    cum_max = equity.cummax()
    dd = (equity / cum_max) - 1
    maxdd = dd.min()
    
    dr = daily_returns.replace([np.inf, -np.inf], 0).fillna(0)
    if dr.std() > 0:
        sharpe = (dr.mean() / dr.std()) * np.sqrt(365)
    else:
        sharpe = 0
    
    return {'sharpe': round(sharpe, 3), 'cagr': round(cagr, 4), 'maxdd': round(maxdd, 4)}

def part2_rotation(tokens):
    print("\n" + "=" * 70)
    print("PART 2: ADAPTIVE ROTATION")
    print("=" * 70)
    
    # Build prices dict
    prices_dict = {}
    for name, df in tokens.items():
        prices_dict[name] = df['close']
    
    # Find common start date where we have enough tokens
    all_starts = {k: v.index[0] for k, v in tokens.items()}
    
    # Rotation parameters to test
    rotation_periods = [21, 63, 126]  # monthly, quarterly, semi-annual
    top_ns = [3, 5, 10]
    momentum_lookbacks = [60, 90, 180]
    
    results = {}
    
    # Static baseline first
    print("\n🔒 STATIC BASELINE (SOL/FTM/AVAX/BNB/SUI)")
    static_tokens_available = [t for t in V4_STATIC if t in tokens]
    
    # Find common date range for static
    static_start = max(tokens[t].index[0] for t in static_tokens_available)
    static_end = min(tokens[t].index[-1] for t in static_tokens_available)
    
    # Build rebalance schedule for static (just one "rebalance" at start)
    static_rebal = [(static_start + pd.Timedelta(days=90), static_tokens_available)]  # after warmup
    
    # Actually, for static we rebalance once and hold
    all_dates = pd.date_range(static_start, static_end, freq='D')
    first_valid = static_start + pd.Timedelta(days=90)
    static_rebal = [(d, static_tokens_available) for d in all_dates if d >= first_valid][::63]  # restate quarterly
    if not static_rebal:
        static_rebal = [(first_valid, static_tokens_available)]
    
    eq_static, dr_static, to_static = v4d_backtest(prices_dict, static_rebal, [], tokens)
    m_static = compute_metrics(eq_static, dr_static)
    m_static['turnover'] = to_static
    results['static_top5'] = m_static
    print(f"  Sharpe: {m_static['sharpe']:.3f}  CAGR: {m_static['cagr']:.1%}  MaxDD: {m_static['maxdd']:.1%}  Turnover: {to_static}")
    
    # Dynamic rotation variants
    print("\n🔄 DYNAMIC ROTATION VARIANTS")
    print(f"  {'Variant':<35} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Turn':>6}")
    print(f"  {'-'*70}")
    
    available_tokens = list(tokens.keys())
    # Remove BTC from rotation candidates (it's the benchmark)
    rotation_candidates = [t for t in available_tokens if t != 'btc']
    
    best_sharpe = -999
    best_variant = None
    
    for rot_period in rotation_periods:
        for top_n in top_ns:
            for mom_lb in momentum_lookbacks:
                variant_name = f"rot{rot_period}d_top{top_n}_mom{mom_lb}d"
                
                # Determine common date range
                common_start = max(tokens[t].index[0] for t in rotation_candidates if t in tokens)
                common_end = min(tokens[t].index[-1] for t in rotation_candidates if t in tokens)
                warmup = max(mom_lb, 50) + 10
                
                trade_dates = pd.date_range(common_start, common_end, freq='D')
                if len(trade_dates) < warmup + rot_period:
                    continue
                
                # Build rebalance schedule
                rebal_schedule = []
                rebal_start = trade_dates[warmup]
                rebal_dates = trade_dates[trade_dates >= rebal_start][::rot_period]
                
                for rd in rebal_dates:
                    # Rank by momentum
                    ranked = []
                    for tok in rotation_candidates:
                        if tok not in prices_dict:
                            continue
                        p = prices_dict[tok]
                        if rd not in p.index:
                            continue
                        lookback_start = rd - pd.Timedelta(days=mom_lb + 10)
                        p_window = p.loc[lookback_start:rd]
                        if len(p_window) < mom_lb * 0.8:
                            continue
                        
                        # Momentum = return over lookback
                        mom = (p_window.iloc[-1] / p_window.iloc[0]) - 1 if p_window.iloc[0] > 0 else 0
                        
                        # SMA50 filter
                        sma50 = p.loc[:rd].tail(60).mean() if len(p.loc[:rd]) >= 50 else None
                        if sma50 is None or p.loc[rd] < sma50:
                            continue
                        
                        ranked.append((tok, mom))
                    
                    ranked.sort(key=lambda x: x[1], reverse=True)
                    selected = [t for t, _ in ranked[:top_n]]
                    if selected:
                        rebal_schedule.append((rd, selected))
                
                if len(rebal_schedule) < 3:
                    continue
                
                eq, dr, to = v4d_backtest(prices_dict, rebal_schedule, [], tokens)
                m = compute_metrics(eq, dr)
                m['turnover'] = to
                results[variant_name] = m
                
                print(f"  {variant_name:<35} {m['sharpe']:>8.3f} {m['cagr']:>8.1%} {m['maxdd']:>8.1%} {to:>6}")
                
                if m['sharpe'] > best_sharpe:
                    best_sharpe = m['sharpe']
                    best_variant = variant_name
    
    # Bootstrap CI on best variant
    print(f"\n🏆 BEST VARIANT: {best_variant}")
    print(f"   Sharpe: {results[best_variant]['sharpe']:.3f}")
    print(f"   CAGR:   {results[best_variant]['cagr']:.1%}")
    print(f"   MaxDD:  {results[best_variant]['maxdd']:.1%}")
    
    # Compare to static
    print(f"\n📊 STATIC vs BEST DYNAMIC:")
    print(f"  {'Metric':<12} {'Static':>10} {'Dynamic':>10} {'Diff':>10}")
    for metric in ['sharpe', 'cagr', 'maxdd']:
        s = m_static[metric]
        d = results[best_variant][metric]
        diff = d - s
        if metric in ['cagr', 'maxdd']:
            print(f"  {metric:<12} {s:>10.1%} {d:>10.1%} {diff:>+10.1%}")
        else:
            print(f"  {metric:<12} {s:>10.3f} {d:>10.3f} {diff:>+10.3f}")
    
    return results, best_variant

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading data...")
    tokens = load_all_tokens(min_days=730)
    print(f"Loaded {len(tokens)} tokens with ≥730 days: {sorted(tokens.keys())}")
    
    # Part 1
    p1_results = part1_correlation(tokens)
    
    # Part 2
    p2_results, best_var = part2_rotation(tokens)
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    avg_corr = p1_results['avg_pairwise_corr']
    avg_beta = p1_results['avg_beta']
    pc1 = p1_results['pc1_pct']
    stress_spike = p1_results['rolling_corr_stress'] - p1_results['rolling_corr_normal']
    
    print(f"\n🔍 DIVERSIFICATION:")
    if avg_corr > 0.7:
        print(f"  ❌ FAKE — avg correlation {avg_corr:.3f} means these move together")
    elif avg_corr > 0.5:
        print(f"  ⚠️  WEAK — avg correlation {avg_corr:.3f}, moderate co-movement")
    else:
        print(f"  ✅ REAL — avg correlation {avg_corr:.3f}, genuine diversification")
    
    if avg_beta > 1.5:
        print(f"  ❌ Portfolio is leveraged BTC (avg beta {avg_beta:.2f})")
    elif avg_beta > 1.0:
        print(f"  ⚠️  High-beta BTC exposure (avg beta {avg_beta:.2f})")
    
    if pc1 > 0.7:
        print(f"  ❌ PC1 explains {pc1:.0%} — single factor dominates")
    elif pc1 > 0.5:
        print(f"  ⚠️  PC1 explains {pc1:.0%} — significant market factor")
    
    if stress_spike > 0.05:
        print(f"  ❌ Correlation rises +{stress_spike:.3f} during stress — diversification fails when needed")
    
    static_sharpe = p2_results.get('static_top5', {}).get('sharpe', 0)
    best_sharpe = p2_results.get(best_var, {}).get('sharpe', 0)
    
    print(f"\n🔄 ROTATION:")
    if best_sharpe > static_sharpe + 0.2:
        print(f"  ✅ Dynamic rotation HELPS — best Sharpe {best_sharpe:.3f} vs static {static_sharpe:.3f}")
    elif best_sharpe > static_sharpe:
        print(f"  ⚠️  Marginal improvement — best {best_sharpe:.3f} vs static {static_sharpe:.3f}")
    else:
        print(f"  ❌ Rotation does NOT help — best {best_sharpe:.3f} vs static {static_sharpe:.3f}")
    
    # Save results
    output = {
        'part1_correlation': p1_results,
        'part2_rotation': p2_results,
        'best_dynamic_variant': best_var,
        'verdict': {
            'diversification_real': avg_corr < 0.5,
            'avg_correlation': avg_corr,
            'avg_beta': avg_beta,
            'pc1_variance': pc1,
            'stress_correlation_spike': stress_spike,
            'rotation_helps': best_sharpe > static_sharpe + 0.2,
            'best_dynamic_sharpe': best_sharpe,
            'static_sharpe': static_sharpe
        }
    }
    
    out_path = os.path.join(OUT_DIR, "v4_correlation_rotation.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Results saved to {out_path}")
