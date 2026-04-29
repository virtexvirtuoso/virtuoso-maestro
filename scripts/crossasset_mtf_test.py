#!/usr/bin/env python3
"""Cross-Asset Divergence & Multi-Timeframe Backtest"""

import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── Data Loading ───
DB = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
con = duckdb.connect(DB, read_only=True)

ALTS = ['ETH', 'SOL', 'BNB', 'AVAX', 'DOGE']
ALL_SYMS = ['BTC'] + ALTS
MTF_SYMS = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX']

def load_table(table, symbols=None):
    q = f"SELECT * FROM {table} ORDER BY date"
    df = con.execute(q).df()
    if symbols:
        df = df[df['symbol'].isin(symbols)]
    return df

# Load all data
print("Loading data...")
prices = load_table('perps_daily', ALL_SYMS)
lsr = load_table('cg_lsr_global', ALL_SYMS)
funding = load_table('cg_funding_rate', ALL_SYMS)
taker = load_table('cg_taker_volume', ALL_SYMS)
liqs = load_table('cg_liquidations', ALL_SYMS)

# Pivot to wide format
def pivot_field(df, field, sym_col='symbol'):
    return df.pivot_table(index='date', columns=sym_col, values=field)

price_close = pivot_field(prices, 'close')
lsr_ratio = pivot_field(lsr, 'global_account_long_short_ratio')
fr_close = pivot_field(funding, 'close')  # funding rate close

# Taker buy ratio
taker['buy_ratio'] = taker['taker_buy_volume_usd'] / (taker['taker_buy_volume_usd'] + taker['taker_sell_volume_usd'])
taker_ratio = pivot_field(taker, 'buy_ratio')

# Total liquidations
liqs['total_liq'] = liqs['aggregated_long_liquidation_usd'] + liqs['aggregated_short_liquidation_usd']
liq_total = pivot_field(liqs, 'total_liq')

# Align all to common dates
common_dates = price_close.dropna(subset=['BTC']).index
for df in [lsr_ratio, fr_close, taker_ratio, liq_total]:
    common_dates = common_dates.intersection(df.dropna(subset=['BTC']).index)

price_close = price_close.loc[common_dates]
lsr_ratio = lsr_ratio.reindex(common_dates)
fr_close = fr_close.reindex(common_dates)
taker_ratio = taker_ratio.reindex(common_dates)
liq_total = liq_total.reindex(common_dates)

# Forward returns for alts (next-day)
returns = price_close.pct_change().shift(-1)  # forward returns

print(f"Data: {len(common_dates)} common dates, {common_dates.min()} to {common_dates.max()}")

# ─── Walk-Forward + Permutation Framework ───
N_FOLDS = 14
N_PERMS = 500
BONFERRONI = 9  # total strategies tested

def expanding_wf_folds(dates, n_folds=14):
    """Expanding walk-forward: each fold adds ~1/14 of data to IS, tests on next chunk."""
    n = len(dates)
    chunk = n // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        is_end = chunk * (i + 2)
        oos_start = is_end
        oos_end = min(is_end + chunk, n)
        if oos_end <= oos_start:
            break
        folds.append((0, is_end, oos_start, oos_end))
    return folds

def evaluate_strategy(signal_func, dates, alts=ALTS):
    """
    signal_func(dates_idx, is_end) -> Series of positions for OOS period
    Returns OOS daily returns across all folds.
    """
    folds = expanding_wf_folds(dates)
    all_oos_rets = []
    
    for is_start, is_end, oos_start, oos_end in folds:
        oos_dates = dates[oos_start:oos_end]
        # Get signals (computed using data up to is_end)
        for alt in alts:
            if alt not in price_close.columns:
                continue
            signals = signal_func(dates, is_end, alt)
            if signals is None:
                continue
            # Only OOS signals
            oos_signals = signals.reindex(oos_dates).fillna(0)
            # Signal on bar N, trade on bar N+1 (returns already shifted)
            oos_ret = oos_signals * returns[alt].reindex(oos_dates)
            all_oos_rets.append(oos_ret.dropna())
    
    if not all_oos_rets:
        return pd.Series(dtype=float)
    return pd.concat(all_oos_rets)

def permutation_test(oos_returns, n_perms=500):
    """Permutation test: shuffle signs of returns."""
    if len(oos_returns) < 20:
        return 1.0
    observed = oos_returns.mean()
    rng = np.random.RandomState(42)
    count = 0
    vals = oos_returns.values
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(vals))
        if (vals * signs).mean() >= observed:
            count += 1
    return count / n_perms

def calc_metrics(oos_returns):
    if len(oos_returns) < 20:
        return {'total_ret': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'n_trades': 0, 'p_value': 1.0}
    
    total = (1 + oos_returns).prod() - 1
    sharpe = oos_returns.mean() / (oos_returns.std() + 1e-10) * np.sqrt(365)
    cumret = (1 + oos_returns).cumprod()
    max_dd = (cumret / cumret.cummax() - 1).min()
    win_rate = (oos_returns > 0).mean()
    p_val = permutation_test(oos_returns)
    
    return {
        'total_ret': round(total * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'n_trades': int((oos_returns != 0).sum()),
        'p_value': round(p_val, 4),
        'p_bonf': round(min(p_val * BONFERRONI, 1.0), 4)
    }

dates = common_dates.values

# ─── SMA50 Baseline ───
print("\n=== SMA50 Daily Baseline ===")
def sma50_signal(dates_arr, is_end, alt):
    px = price_close[alt].reindex(dates_arr[:is_end+len(dates_arr[is_end:])])
    sma = px.rolling(50).mean()
    sig = (px > sma).astype(float)
    return sig

baseline_rets = evaluate_strategy(sma50_signal, dates)
baseline_metrics = calc_metrics(baseline_rets)
print(f"SMA50 Baseline: {baseline_metrics}")

# ─── Part 1: Cross-Asset Divergence ───
print("\n" + "="*60)
print("PART 1: CROSS-ASSET DIVERGENCE SIGNALS")
print("="*60)

results = {}

# Strategy 1: BTC LSR vs Alt LSR Divergence
print("\n--- S1: BTC LSR vs Alt LSR Divergence ---")
def s1_lsr_divergence(dates_arr, is_end, alt):
    all_dates = dates_arr[:is_end + len(dates_arr[is_end:])]
    btc_lsr_vals = lsr_ratio['BTC'].reindex(all_dates)
    alt_lsr_vals = lsr_ratio[alt].reindex(all_dates) if alt in lsr_ratio.columns else None
    if alt_lsr_vals is None:
        return None
    # Use expanding percentiles up to is_end for thresholds
    is_dates = dates_arr[:is_end]
    btc_50 = btc_lsr_vals.reindex(is_dates).quantile(0.5)
    alt_50 = alt_lsr_vals.reindex(is_dates).quantile(0.5)
    # Signal: BTC LSR < 50th pctl (bearish crowd) AND alt LSR > 50th pctl
    sig = ((btc_lsr_vals < btc_50) & (alt_lsr_vals > alt_50)).astype(float)
    return sig

s1_rets = evaluate_strategy(s1_lsr_divergence, dates)
results['S1_LSR_Divergence'] = calc_metrics(s1_rets)
print(f"  {results['S1_LSR_Divergence']}")

# Strategy 2: BTC Funding vs Alt Funding Spread
print("\n--- S2: BTC Funding vs Alt Funding Spread ---")
def s2_funding_spread(dates_arr, is_end, alt):
    all_dates = dates_arr[:is_end + len(dates_arr[is_end:])]
    btc_fr = fr_close['BTC'].reindex(all_dates)
    alt_fr = fr_close[alt].reindex(all_dates) if alt in fr_close.columns else None
    if alt_fr is None:
        return None
    is_dates = dates_arr[:is_end]
    btc_75 = btc_fr.reindex(is_dates).quantile(0.75)
    alt_25 = alt_fr.reindex(is_dates).quantile(0.25)
    # BTC funding high, alt funding low → long alt
    sig = ((btc_fr > btc_75) & (alt_fr < alt_25)).astype(float)
    return sig

s2_rets = evaluate_strategy(s2_funding_spread, dates)
results['S2_Funding_Spread'] = calc_metrics(s2_rets)
print(f"  {results['S2_Funding_Spread']}")

# Strategy 3: Taker Volume Rotation
print("\n--- S3: Taker Volume Rotation ---")
def s3_taker_rotation(dates_arr, is_end, alt):
    all_dates = dates_arr[:is_end + len(dates_arr[is_end:])]
    btc_tk = taker_ratio['BTC'].reindex(all_dates)
    alt_tk = taker_ratio[alt].reindex(all_dates) if alt in taker_ratio.columns else None
    if alt_tk is None:
        return None
    # BTC taker declining (5d change < 0), alt rising (5d change > 0)
    btc_chg = btc_tk.rolling(5).mean().diff(5)
    alt_chg = alt_tk.rolling(5).mean().diff(5)
    sig = ((btc_chg < 0) & (alt_chg > 0)).astype(float)
    return sig

s3_rets = evaluate_strategy(s3_taker_rotation, dates)
results['S3_Taker_Rotation'] = calc_metrics(s3_rets)
print(f"  {results['S3_Taker_Rotation']}")

# Strategy 4: Liquidation Divergence
print("\n--- S4: Liquidation Divergence ---")
def s4_liq_divergence(dates_arr, is_end, alt):
    all_dates = dates_arr[:is_end + len(dates_arr[is_end:])]
    btc_lq = liq_total['BTC'].reindex(all_dates)
    alt_lq = liq_total[alt].reindex(all_dates) if alt in liq_total.columns else None
    if alt_lq is None:
        return None
    is_dates = dates_arr[:is_end]
    btc_90 = btc_lq.reindex(is_dates).quantile(0.90)
    alt_50 = alt_lq.reindex(is_dates).quantile(0.50)
    # Major BTC liquidation but alt is calm → long alt
    sig = ((btc_lq > btc_90) & (alt_lq < alt_50)).astype(float)
    return sig

s4_rets = evaluate_strategy(s4_liq_divergence, dates)
results['S4_Liq_Divergence'] = calc_metrics(s4_rets)
print(f"  {results['S4_Liq_Divergence']}")

# Strategy 5: Relative Regime Score
print("\n--- S5: Relative Regime Score ---")
def compute_regime_score(sym, dates_arr, is_end):
    """Composite regime score from LSR + FR + Taker + Liq."""
    all_dates = dates_arr[:is_end + len(dates_arr[is_end:])]
    is_dates = dates_arr[:is_end]
    
    score = pd.Series(0.0, index=all_dates)
    
    # LSR component: high LSR = bullish crowd = contrarian bearish → negative
    if sym in lsr_ratio.columns:
        v = lsr_ratio[sym].reindex(all_dates)
        med = v.reindex(is_dates).median()
        score += np.where(v > med, -1, 1)
    
    # Funding: high = overcrowded long → negative
    if sym in fr_close.columns:
        v = fr_close[sym].reindex(all_dates)
        med = v.reindex(is_dates).median()
        score += np.where(v > med, -1, 1)
    
    # Taker buy ratio: high = bullish flow → positive  
    if sym in taker_ratio.columns:
        v = taker_ratio[sym].reindex(all_dates)
        med = v.reindex(is_dates).median()
        score += np.where(v > med, 1, -1)
    
    # Liq: low = calm → positive
    if sym in liq_total.columns:
        v = liq_total[sym].reindex(all_dates)
        med = v.reindex(is_dates).median()
        score += np.where(v < med, 1, -1)
    
    return pd.Series(score, index=all_dates)

def s5_regime_score(dates_arr, is_end, alt):
    btc_score = compute_regime_score('BTC', dates_arr, is_end)
    alt_score = compute_regime_score(alt, dates_arr, is_end)
    # Long alt when its regime score > BTC's
    sig = (alt_score > btc_score).astype(float)
    return sig

s5_rets = evaluate_strategy(s5_regime_score, dates)
results['S5_Regime_Score'] = calc_metrics(s5_rets)
print(f"  {results['S5_Regime_Score']}")

# ─── Part 2: Multi-Timeframe ───
print("\n" + "="*60)
print("PART 2: MULTI-TIMEFRAME SIGNALS")
print("="*60)

# Load 4H data
sym_4h_map = {
    'BTC': 'binance_btc_usdt_4h.csv',
    'ETH': 'binance_eth_usdt_4h.csv',
    'SOL': 'binance_sol_usdt_4h.csv',
    'AVAX': 'binance_avax_usdt_4h.csv',
}
# BNB 4h? Check
bnb_4h = os.path.expanduser('~/Desktop/maestro/data/ohlcv/binance_bnb_usdt_4h.csv')
if os.path.exists(bnb_4h):
    sym_4h_map['BNB'] = 'binance_bnb_usdt_4h.csv'

data_4h = {}
for sym, fn in sym_4h_map.items():
    fp = os.path.expanduser(f'~/Desktop/maestro/data/ohlcv/{fn}')
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        data_4h[sym] = df
        print(f"  4H {sym}: {len(df)} bars, {df.index.min()} to {df.index.max()}")

# Get end-of-day 4H bar (last bar of day, typically 20:00 UTC)
def get_eod_4h(sym):
    if sym not in data_4h:
        return None
    df = data_4h[sym].copy()
    df['date'] = df.index.date
    # Last 4H bar of each day
    eod = df.groupby('date').last()
    eod.index = pd.to_datetime(eod.index)
    return eod

# 4H RSI
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

# Get daily price for MTF symbols
daily_price = {}
for sym in MTF_SYMS:
    fp = os.path.expanduser(f'~/Desktop/maestro/data/ohlcv/binance_{sym.lower()}_usdt_1d.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=['timestamp']).sort_values('timestamp').set_index('timestamp')
        daily_price[sym] = df
    else:
        # Use perps_daily
        df = prices[prices['symbol'] == sym][['date', 'close', 'volume']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        daily_price[sym] = df

# Strategy 6: 4H RSI Dip Buying on Daily Trend
print("\n--- S6: 4H RSI Dip on Daily SMA50 ---")
def test_mtf_strategy(strat_name, signal_func_mtf):
    all_rets_mtf = []
    all_rets_baseline = []
    per_sym = {}
    
    for sym in MTF_SYMS:
        if sym not in daily_price or sym not in data_4h:
            continue
        dp = daily_price[sym]
        eod_4h = get_eod_4h(sym)
        if eod_4h is None:
            continue
        
        px = dp['close']
        sma50 = px.rolling(50).mean()
        daily_trend = (px > sma50).astype(float)
        
        # Align dates
        common = px.index.intersection(eod_4h.index)
        if len(common) < 100:
            continue
        
        fwd_ret = px.pct_change().shift(-1)
        
        # Expanding walk-forward
        folds = expanding_wf_folds(common.values)
        sym_rets_mtf = []
        sym_rets_base = []
        
        for is_s, is_e, oos_s, oos_e in folds:
            oos_dates = common[oos_s:oos_e]
            
            # Baseline: just daily SMA50
            base_sig = daily_trend.reindex(oos_dates).fillna(0)
            base_ret = (base_sig * fwd_ret.reindex(oos_dates)).dropna()
            sym_rets_base.append(base_ret)
            
            # MTF signal
            mtf_sig = signal_func_mtf(sym, common, is_e, daily_trend, eod_4h, dp)
            mtf_sig = mtf_sig.reindex(oos_dates).fillna(0)
            mtf_ret = (mtf_sig * fwd_ret.reindex(oos_dates)).dropna()
            sym_rets_mtf.append(mtf_ret)
        
        if sym_rets_mtf:
            mtf_all = pd.concat(sym_rets_mtf)
            base_all = pd.concat(sym_rets_base)
            all_rets_mtf.append(mtf_all)
            all_rets_baseline.append(base_all)
            per_sym[sym] = {
                'mtf': calc_metrics(mtf_all),
                'baseline': calc_metrics(base_all)
            }
    
    if all_rets_mtf:
        combined_mtf = pd.concat(all_rets_mtf)
        combined_base = pd.concat(all_rets_baseline)
        return calc_metrics(combined_mtf), calc_metrics(combined_base), per_sym
    return None, None, {}

def s6_rsi_dip(sym, common, is_end, daily_trend, eod_4h, dp):
    """Daily SMA50 long + 4H RSI < 30 = entry."""
    rsi_4h = calc_rsi(eod_4h['close'], 14)
    all_dates = common[:is_end + len(common[is_end:])]
    trend = daily_trend.reindex(all_dates)
    rsi = rsi_4h.reindex(all_dates)
    # Enter on RSI dip, stay in until RSI > 50
    sig = pd.Series(0.0, index=all_dates)
    in_pos = False
    for d in all_dates:
        t = trend.get(d, 0)
        r = rsi.get(d, 50)
        if t > 0 and r < 30 and not in_pos:
            in_pos = True
        elif in_pos and (t <= 0 or r > 70):
            in_pos = False
        sig[d] = 1.0 if in_pos else (t if t > 0 else 0)
    return sig

# Vectorized version for speed
def s6_rsi_dip_v(sym, common, is_end, daily_trend, eod_4h, dp):
    """Daily SMA50 long + 4H RSI < 30 refinement."""
    rsi_4h = calc_rsi(eod_4h['close'], 14)
    all_dates = common[:is_end + len(common[is_end:])]
    trend = daily_trend.reindex(all_dates).fillna(0)
    rsi = rsi_4h.reindex(all_dates).fillna(50)
    # Simple: in daily uptrend, use RSI < 30 as entry trigger (hold until RSI > 60)
    sig = trend.copy()  # baseline is daily trend
    # Override: only enter fresh positions when RSI < 30
    # This is complex to vectorize properly; use simpler filter
    # When in uptrend: if RSI < 30, signal = 1 (dip buy); else still 1 (hold)
    # Net effect: same as baseline. Instead: only go long when RSI was recently < 30
    rsi_dip = (rsi < 30).rolling(5).max().fillna(0)  # RSI was < 30 in last 5 days
    sig = trend * rsi_dip
    return sig

s6_mtf, s6_base, s6_per = test_mtf_strategy('S6', s6_rsi_dip_v)
results['S6_4H_RSI_Dip'] = s6_mtf if s6_mtf else {'total_ret': 0, 'sharpe': 0}
print(f"  MTF: {s6_mtf}")
print(f"  Base: {s6_base}")

# Strategy 7: 4H Volume Confirmation
print("\n--- S7: 4H Volume Confirmation ---")
def s7_vol_confirm(sym, common, is_end, daily_trend, eod_4h, dp):
    all_dates = common[:is_end + len(common[is_end:])]
    trend = daily_trend.reindex(all_dates).fillna(0)
    trend_chg = trend.diff()  # 1 = new long, -1 = exit
    
    # 4H volume (end of day bar)
    vol_4h = eod_4h['volume'].reindex(all_dates)
    vol_avg = vol_4h.rolling(20).mean()
    vol_spike = (vol_4h > 2 * vol_avg).astype(float)
    
    # Only enter on SMA cross if volume confirms
    sig = trend.copy()
    # On cross days without volume confirmation, stay flat
    cross_days = trend_chg.abs() > 0
    sig[cross_days & (vol_spike == 0)] = 0
    # Forward fill the filtered signals
    return sig

s7_mtf, s7_base, s7_per = test_mtf_strategy('S7', s7_vol_confirm)
results['S7_4H_Vol_Confirm'] = s7_mtf if s7_mtf else {'total_ret': 0, 'sharpe': 0}
print(f"  MTF: {s7_mtf}")
print(f"  Base: {s7_base}")

# Strategy 8: 4H EMA Ribbon
print("\n--- S8: 4H EMA Ribbon ---")
def s8_ema_ribbon(sym, common, is_end, daily_trend, eod_4h, dp):
    all_dates = common[:is_end + len(common[is_end:])]
    trend = daily_trend.reindex(all_dates).fillna(0)
    
    c = eod_4h['close']
    ema8 = c.ewm(span=8).mean()
    ema21 = c.ewm(span=21).mean()
    ema55 = c.ewm(span=55).mean()
    
    fully_bullish = ((ema8 > ema21) & (ema21 > ema55)).reindex(all_dates).fillna(False)
    
    # Full position when fully bullish, half when inverted
    sig = trend.copy()
    sig[trend > 0] = np.where(fully_bullish[trend > 0], 1.0, 0.5)
    return sig

s8_mtf, s8_base, s8_per = test_mtf_strategy('S8', s8_ema_ribbon)
results['S8_EMA_Ribbon'] = s8_mtf if s8_mtf else {'total_ret': 0, 'sharpe': 0}
print(f"  MTF: {s8_mtf}")
print(f"  Base: {s8_base}")

# Strategy 9: 4H Trailing Stop
print("\n--- S9: 4H Trailing Stop ---")
def s9_trailing_4h(sym, common, is_end, daily_trend, eod_4h, dp):
    all_dates = common[:is_end + len(common[is_end:])]
    trend = daily_trend.reindex(all_dates).fillna(0)
    
    # Use all 4H bars for trailing stop (not just EOD)
    df_4h = data_4h[sym]
    
    # For each day, check if 4H trailing stop was hit
    # ATR-based trail: 2x ATR on 4H
    h = eod_4h['high'].reindex(all_dates)
    l = eod_4h['low'].reindex(all_dates)
    c = eod_4h['close'].reindex(all_dates)
    
    atr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    
    # Track trailing stop
    sig = pd.Series(0.0, index=all_dates)
    peak = 0
    in_pos = False
    
    for i, d in enumerate(all_dates):
        t = trend.get(d, 0)
        cv = c.get(d, np.nan)
        av = atr.get(d, np.nan)
        
        if np.isnan(cv) or np.isnan(av):
            sig.iloc[i] = t
            continue
            
        if t > 0 and not in_pos:
            in_pos = True
            peak = cv
        elif in_pos:
            peak = max(peak, cv)
            if cv < peak - 2 * av:  # Trailing stop hit
                in_pos = False
                peak = 0
        
        sig.iloc[i] = 1.0 if in_pos else 0.0
    
    return sig

s9_mtf, s9_base, s9_per = test_mtf_strategy('S9', s9_trailing_4h)
results['S9_4H_Trail'] = s9_mtf if s9_mtf else {'total_ret': 0, 'sharpe': 0}
print(f"  MTF: {s9_mtf}")
print(f"  Base: {s9_base}")

# ─── Summary Tables ───
print("\n" + "="*60)
print("CROSS-ASSET DIVERGENCE RESULTS")
print("="*60)
print(f"{'Strategy':<25} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'Trades':>6} {'p-val':>7} {'p-bonf':>7}")
print("-" * 80)
for s in ['S1_LSR_Divergence', 'S2_Funding_Spread', 'S3_Taker_Rotation', 'S4_Liq_Divergence', 'S5_Regime_Score']:
    m = results[s]
    sig = "***" if m.get('p_bonf', 1) < 0.05 else ("**" if m.get('p_bonf', 1) < 0.1 else ("*" if m.get('p_value', 1) < 0.05 else ""))
    print(f"{s:<25} {m.get('total_ret',0):>7.1f}% {m.get('sharpe',0):>7.3f} {m.get('max_dd',0):>7.1f} {m.get('win_rate',0):>5.1f}% {m.get('n_trades',0):>6} {m.get('p_value',1):>7.4f} {m.get('p_bonf',1):>7.4f} {sig}")

print(f"\nSMA50 Baseline: {baseline_metrics}")

print("\n" + "="*60)
print("MULTI-TIMEFRAME RESULTS")
print("="*60)
print(f"{'Strategy':<25} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'Trades':>6} {'p-val':>7} {'p-bonf':>7}")
print("-" * 80)
for s in ['S6_4H_RSI_Dip', 'S7_4H_Vol_Confirm', 'S8_EMA_Ribbon', 'S9_4H_Trail']:
    m = results[s]
    sig = "***" if m.get('p_bonf', 1) < 0.05 else ("**" if m.get('p_bonf', 1) < 0.1 else ("*" if m.get('p_value', 1) < 0.05 else ""))
    print(f"{s:<25} {m.get('total_ret',0):>7.1f}% {m.get('sharpe',0):>7.3f} {m.get('max_dd',0):>7.1f} {m.get('win_rate',0):>5.1f}% {m.get('n_trades',0):>6} {m.get('p_value',1):>7.4f} {m.get('p_bonf',1):>7.4f} {sig}")

# ─── Verdict ───
print("\n" + "="*60)
print("DOES 4H DATA ADD ANYTHING OVER DAILY-ONLY?")
print("="*60)
for s, label in [('S6_4H_RSI_Dip', 'RSI Dip'), ('S7_4H_Vol_Confirm', 'Vol Confirm'), 
                  ('S8_EMA_Ribbon', 'EMA Ribbon'), ('S9_4H_Trail', '4H Trail')]:
    m = results[s]
    base_sharpe = s6_base['sharpe'] if s6_base else 0  # all baselines same
    mtf_sharpe = m.get('sharpe', 0)
    improvement = mtf_sharpe - (s6_base['sharpe'] if s6_base else 0)
    print(f"  {label}: Sharpe {mtf_sharpe:.3f} vs baseline {base_sharpe:.3f} (Δ {improvement:+.3f})")

# ─── Save Results ───
output = {
    'timestamp': datetime.now().isoformat(),
    'methodology': {
        'folds': N_FOLDS,
        'permutations': N_PERMS,
        'bonferroni_factor': BONFERRONI,
        'signal_lag': 'N+1',
        'walk_forward': 'expanding'
    },
    'baseline_sma50': baseline_metrics,
    'cross_asset_divergence': {k: v for k, v in results.items() if k.startswith('S') and int(k[1]) <= 5},
    'multi_timeframe': {k: v for k, v in results.items() if k.startswith('S') and int(k[1]) >= 6},
    'mtf_per_symbol': {
        'S6': s6_per if s6_per else {},
        'S7': s7_per if s7_per else {},
        'S8': s8_per if s8_per else {},
        'S9': s9_per if s9_per else {},
    }
}

os.makedirs(os.path.expanduser('~/Desktop/maestro/data/backtest_results'), exist_ok=True)
with open(os.path.expanduser('~/Desktop/maestro/data/backtest_results/crossasset_mtf_test.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("\n✅ Results saved to ~/Desktop/maestro/data/backtest_results/crossasset_mtf_test.json")
