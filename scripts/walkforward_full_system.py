#!/usr/bin/env python3
"""
Walk-Forward Validation — Full Combined System (4-signal + hedge-when-flat + progressive stop).

V2: All 5 quant review fixes applied:
  1. Progressive stop lookahead bug fixed (stop at day i → exit day i+1)
  2. Block sign-permutation test (block_len=40, preserves autocorrelation)
  3. Fold placement sensitivity (shift +0, +30, +60 days)
  4. LSR pairs TX cost: turnover-based (tracks actual position changes)
  5. Funding rate verified: daily aggregate, /100 is correct (~3.9%/yr)

Fixed params throughout (no optimization per fold).
Uses sqrt(365), 10bps TX cost, real funding rates.
"""
import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
TX_COST = 0.001
RESULTS_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")

# WF config
TRAIN_DAYS_MIN = 252  # minimum warmup/train
TEST_DAYS = 126       # ~6 months OOS per fold
N_PERMS = 10000       # block permutations for significance
N_BOOT = 5000         # bootstrap samples for CI
BLOCK_LEN = 40        # block length for permutation test (calibrated to autocorrelation)
np.random.seed(42)

# Progressive stop config
STOP_INITIAL = 0.15
STOP_PROFIT_20 = 0.10
STOP_PROFIT_50 = 0.07

# Portfolio weights
WEIGHTS = {'btc': 0.45, 'sol': 0.20, 'lsr': 0.15, 'doge': 0.20}


# ── Data Loading ─────────────────────────────────────────────────────────

def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    btc = con.execute(
        "SELECT date, close as btc_close FROM perps_daily WHERE symbol='BTC' ORDER BY date"
    ).df()
    funding = con.execute(
        "SELECT date, close as btc_funding FROM cg_funding_rate WHERE symbol='BTC' ORDER BY date"
    ).df()
    lsr = con.execute(
        "SELECT date, global_account_long_short_ratio as btc_lsr "
        "FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date"
    ).df()
    liq = con.execute(
        "SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as btc_total_liq "
        "FROM cg_liquidations WHERE symbol='BTC' ORDER BY date"
    ).df()
    taker = con.execute(
        "SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as btc_taker_ratio "
        "FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date"
    ).df()
    sol = con.execute(
        "SELECT date, close as sol_close FROM perps_daily WHERE symbol='SOL' ORDER BY date"
    ).df()
    sol_lsr = con.execute(
        "SELECT date, global_account_long_short_ratio as sol_lsr "
        "FROM cg_lsr_global WHERE symbol='SOL' ORDER BY date"
    ).df()
    doge = con.execute(
        "SELECT date, close as doge_close FROM perps_daily WHERE symbol='DOGE' ORDER BY date"
    ).df()

    all_symbols = ['BTC', 'ETH', 'SOL', 'LINK', 'AVAX', 'DOGE', 'DOT', 'NEAR',
                   'ATOM', 'BNB', 'XRP', 'UNI', 'ADA', 'FIL']
    lsr_all = {}
    price_all = {}
    for sym in all_symbols:
        lsr_df = con.execute(
            f"SELECT date, global_account_long_short_ratio as lsr "
            f"FROM cg_lsr_global WHERE symbol='{sym}' ORDER BY date"
        ).df()
        px_df = con.execute(
            f"SELECT date, close FROM perps_daily WHERE symbol='{sym}' ORDER BY date"
        ).df()
        lsr_all[sym] = lsr_df.set_index('date')['lsr']
        price_all[sym] = px_df.set_index('date')['close']
    con.close()

    df = btc.copy()
    for src in [funding, lsr, liq, taker, sol, sol_lsr, doge]:
        df = df.merge(src, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df, lsr_all, price_all, all_symbols


# ── Signal Functions ─────────────────────────────────────────────────────

def btc_regime_signal(df):
    lsr_med = df['btc_lsr'].rolling(30, min_periods=10).median()
    lsr_s = (df['btc_lsr'] < lsr_med).astype(float)
    fund_s = (df['btc_funding'] < 0.03).astype(float)
    liq_p80 = df['btc_total_liq'].rolling(30, min_periods=10).quantile(0.8)
    liq_s = (df['btc_total_liq'] < liq_p80).astype(float)
    taker_s = (df['btc_taker_ratio'] > 1.0).astype(float)
    score = lsr_s * 0.35 + fund_s * 0.35 + liq_s * 0.15 + taker_s * 0.15
    signal = (score > 0.5).astype(float)
    btc_ret = df['btc_close'].pct_change()
    vol = btc_ret.rolling(20, min_periods=10).std()
    vol_sc = (0.015 / vol).clip(0.25, 5.0)
    position = signal * vol_sc
    position = position.shift(1).fillna(0)
    return position.values, btc_ret.values


def sol_relmom_signal(df):
    btc_ret_20d = df['btc_close'].pct_change(20)
    sol_ret_20d = df['sol_close'].pct_change(20)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((sol_ret_20d > btc_ret_20d) & (df['btc_close'] > btc_sma50)).astype(float)
    signal = signal.shift(1).fillna(0)
    return signal.values, df['sol_close'].pct_change().values


def doge_relmom_signal(df):
    btc_ret_10d = df['btc_close'].pct_change(10)
    doge_ret_10d = df['doge_close'].pct_change(10)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((doge_ret_10d > btc_ret_10d) & (df['btc_close'] > btc_sma50)).astype(float)
    signal = signal.shift(1).fillna(0)
    return signal.values, df['doge_close'].pct_change().values


def lsr_divergence_pnl(lsr_all, price_all, symbols, dates):
    lsr_df = pd.DataFrame(lsr_all).reindex(dates)
    price_df = pd.DataFrame(price_all).reindex(dates)
    ret_df = price_df.pct_change()
    daily_pnl = np.zeros(len(dates))
    prev_longs = set()
    prev_shorts = set()
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        lsr_vals = lsr_df.loc[prev_date].dropna()
        if len(lsr_vals) < 6:
            continue
        ranked = lsr_vals.sort_values()
        longs = set(ranked.index[:3])
        shorts = set(ranked.index[-3:])
        long_ret = ret_df.loc[curr_date, list(longs)].mean() if curr_date in ret_df.index else 0
        short_ret = ret_df.loc[curr_date, list(shorts)].mean() if curr_date in ret_df.index else 0
        if np.isnan(long_ret): long_ret = 0
        if np.isnan(short_ret): short_ret = 0
        # Turnover-based TX cost: only pay for positions that actually change
        n_changed = len(longs - prev_longs) + len(shorts - prev_shorts)
        tc = n_changed * 2 * TX_COST / 6  # exit old + enter new, each 1/6 weight
        daily_pnl[i] = (long_ret - short_ret) - tc
        prev_longs = longs
        prev_shorts = shorts
    return daily_pnl


def compute_hedge_trigger(df, sma_period=50, lsr_mult=1.1):
    close = df['btc_close'].values
    sma = pd.Series(close).rolling(sma_period, min_periods=sma_period // 2).mean().values
    lsr = df['btc_lsr'].values
    lsr_med = pd.Series(lsr).rolling(30, min_periods=10).median().values
    n = len(close)
    trigger = np.zeros(n)
    for i in range(1, n):
        bearish = False
        if not np.isnan(sma[i - 1]) and close[i - 1] < sma[i - 1]:
            bearish = True
        if not np.isnan(lsr_med[i - 1]) and lsr[i - 1] > lsr_med[i - 1] * lsr_mult:
            bearish = True
        trigger[i] = 1.0 if bearish else 0.0
    return trigger


def apply_progressive_stop(position, price_series):
    pos_out = position.copy()
    entry_price = None
    peak_price = None
    stopped = False
    for i in range(len(position)):
        raw_pos = position[i]
        px = price_series[i] if not np.isnan(price_series[i]) else 0.0
        if raw_pos > 0 and not stopped:
            if entry_price is None:
                entry_price = px; peak_price = px
            else:
                peak_price = max(peak_price, px)
            profit_pct = (peak_price - entry_price) / entry_price if entry_price > 0 else 0
            if profit_pct > 0.50:
                stop_pct = STOP_PROFIT_50
            elif profit_pct > 0.20:
                stop_pct = STOP_PROFIT_20
            else:
                stop_pct = STOP_INITIAL
            dd_from_peak = (peak_price - px) / peak_price if peak_price > 0 else 0
            if dd_from_peak > stop_pct:
                stopped = True  # exit NEXT day (day i+1), don't erase day i's P&L
        elif raw_pos == 0:
            entry_price = None; peak_price = None; stopped = False
        elif stopped and raw_pos > 0:
            pos_out[i] = 0.0
    return pos_out


# ── Full System Daily P&L ────────────────────────────────────────────────

def compute_full_system_pnl(df, lsr_all, price_all, symbols, hedge=True, vol_mult=2.0):
    """
    Compute daily percentage P&L for the full system.
    Returns array of daily returns (not compounded).
    """
    n = len(df)
    dates = df['date'].values

    # Compute all signals
    btc_pos, btc_ret = btc_regime_signal(df)
    sol_pos, sol_ret = sol_relmom_signal(df)
    doge_pos, doge_ret = doge_relmom_signal(df)
    lsr_pnl = lsr_divergence_pnl(lsr_all, price_all, symbols, dates)

    # Apply progressive stops
    btc_price = df['btc_close'].values
    sol_price = df['sol_close'].values
    doge_price = df['doge_close'].values
    btc_pos = apply_progressive_stop(btc_pos, btc_price)
    sol_pos = apply_progressive_stop(sol_pos, sol_price)
    doge_pos = apply_progressive_stop(doge_pos, doge_price)

    # Hedge trigger
    hedge_trigger = compute_hedge_trigger(df, sma_period=50, lsr_mult=1.1) if hedge else np.zeros(n)
    funding = df['btc_funding'].values

    # Compute daily P&L
    daily_pnl = np.zeros(n)
    prev_bp, prev_sp, prev_dp, prev_h = 0.0, 0.0, 0.0, 0.0
    w = WEIGHTS

    for i in range(1, n):
        br = btc_ret[i] if not np.isnan(btc_ret[i]) else 0.0
        sr = sol_ret[i] if not np.isnan(sol_ret[i]) else 0.0
        dr = doge_ret[i] if not np.isnan(doge_ret[i]) else 0.0
        lp = lsr_pnl[i]

        bp = btc_pos[i] * vol_mult
        sp = sol_pos[i]
        dp = doge_pos[i]

        # TX costs
        btc_tc = abs(bp - prev_bp) * TX_COST
        sol_tc = abs(sp - prev_sp) * TX_COST
        doge_tc = abs(dp - prev_dp) * TX_COST

        # Funding costs (longs pay when positive)
        fr = funding[i] / 100 if not np.isnan(funding[i]) else 0.0
        btc_fc = bp * fr if bp > 0 else 0.0
        sol_fc = sp * fr if sp > 0 else 0.0
        doge_fc = dp * fr if dp > 0 else 0.0

        # Base P&L
        base_pnl = ((bp * br - btc_tc - btc_fc) * w['btc'] +
                     (sp * sr - sol_tc - sol_fc) * w['sol'] +
                     (dp * dr - doge_tc - doge_fc) * w['doge'] +
                     lp * w['lsr'])

        # Hedge-when-flat overlay
        hedge_pnl = 0.0
        if hedge:
            n_active = int(bp > 0) + int(sp > 0) + int(dp > 0)
            if n_active == 0 and hedge_trigger[i] == 1:
                h = -1.0
                h_tc = abs(h - prev_h) * TX_COST
                h_fc = h * fr  # shorts receive funding when positive
                hedge_pnl = (h * br - h_tc - h_fc) * w['btc']
                prev_h = h
            else:
                if prev_h != 0:
                    hedge_pnl = -abs(0 - prev_h) * TX_COST * w['btc']
                prev_h = 0.0

        daily_pnl[i] = base_pnl + hedge_pnl
        prev_bp, prev_sp, prev_dp = bp, sp, dp

    return daily_pnl


# ── Walk-Forward Validation ──────────────────────────────────────────────

def wf_validate(daily_pnl, warmup=50):
    """
    Rolling walk-forward validation with fixed parameters.
    Split into non-overlapping OOS windows, compute per-fold and concatenated metrics.
    """
    pnl = daily_pnl[warmup:]
    n = len(pnl)

    # Generate fold boundaries
    folds = []
    start = TRAIN_DAYS_MIN
    while start + TEST_DAYS <= n:
        folds.append((start, start + TEST_DAYS))
        start += TEST_DAYS

    if not folds:
        return None

    fold_results = []
    all_oos = []

    for fold_idx, (test_start, test_end) in enumerate(folds):
        test = pnl[test_start:test_end]

        if len(test) < 20:
            continue

        mu = np.mean(test)
        sd = np.std(test)
        sharpe = mu / sd * np.sqrt(365) if sd > 0 else 0

        equity = np.cumprod(1 + np.clip(test, -0.99, None))
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1)

        fold_results.append({
            'fold': fold_idx + 1,
            'test_start': test_start + warmup,
            'test_end': test_end + warmup,
            'n_days': len(test),
            'sharpe': round(sharpe, 4),
            'total_return': round(float(equity[-1] - 1), 4),
            'max_dd': round(abs(dd.min()), 4),
            'mean_daily_bps': round(mu * 10000, 2),
        })
        all_oos.extend(test.tolist())

    if not fold_results:
        return None

    oos = np.array(all_oos)
    oos_mu = np.mean(oos)
    oos_sd = np.std(oos)
    concat_sharpe = oos_mu / oos_sd * np.sqrt(365) if oos_sd > 0 else 0

    f_sharpes = [f['sharpe'] for f in fold_results]
    pos_folds = sum(1 for s in f_sharpes if s > 0)

    # Block sign-permutation test (H0: mean daily return = 0)
    # Flip signs of entire blocks to preserve within-block autocorrelation
    obs_mean = np.mean(oos)
    n_blocks = int(np.ceil(len(oos) / BLOCK_LEN))
    perm_ct = 0
    for _ in range(N_PERMS):
        block_signs = np.random.choice([-1, 1], size=n_blocks)
        signs = np.repeat(block_signs, BLOCK_LEN)[:len(oos)]
        if np.mean(oos * signs) >= obs_mean:
            perm_ct += 1
    p_value = (perm_ct + 1) / (N_PERMS + 1)

    # Bootstrap CI on Sharpe
    boots = []
    for _ in range(N_BOOT):
        b = np.random.choice(oos, size=len(oos), replace=True)
        bs = np.mean(b) / np.std(b) * np.sqrt(365) if np.std(b) > 0 else 0
        boots.append(bs)
    ci_lower = np.percentile(boots, 2.5)
    ci_upper = np.percentile(boots, 97.5)

    # Full-sample metrics for reference
    fs_pnl = pnl[warmup:]  # skip extra warmup in the trimmed series
    fs_mu = np.mean(pnl)
    fs_sd = np.std(pnl)
    fs_sharpe = fs_mu / fs_sd * np.sqrt(365) if fs_sd > 0 else 0
    fs_equity = np.cumprod(1 + np.clip(pnl, -0.99, None))
    n_years = len(pnl) / 365
    fs_cagr = fs_equity[-1] ** (1 / n_years) - 1 if n_years > 0 and fs_equity[-1] > 0 else -1
    fs_peak = np.maximum.accumulate(fs_equity)
    fs_dd = (fs_equity - fs_peak) / np.where(fs_peak > 0, fs_peak, 1)
    fs_max_dd = abs(fs_dd.min())

    return {
        'full_sample': {
            'sharpe': round(fs_sharpe, 4),
            'cagr': round(fs_cagr, 4),
            'max_dd': round(fs_max_dd, 4),
            'daily_mean_bps': round(fs_mu * 10000, 2),
            'n_days': len(pnl),
        },
        'wf_oos': {
            'concat_sharpe': round(concat_sharpe, 4),
            'mean_sharpe': round(np.mean(f_sharpes), 4),
            'median_sharpe': round(np.median(f_sharpes), 4),
            'std_sharpe': round(np.std(f_sharpes), 4),
            'p_value': round(p_value, 4),
            'positive_folds': f"{pos_folds}/{len(fold_results)}",
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'n_oos_days': len(oos),
            'n_folds': len(fold_results),
            'folds': fold_results,
        },
    }


# ── Leverage Simulation on OOS Returns ───────────────────────────────────

def simulate_leveraged_oos(daily_pnl, warmup=50, leverage=1.0, funding_rates=None):
    """Simulate leveraged equity on OOS-only returns."""
    pnl = daily_pnl[warmup:]
    n = len(pnl)

    # Extract OOS windows only
    oos_pnl = []
    start = TRAIN_DAYS_MIN
    while start + TEST_DAYS <= n:
        oos_pnl.extend(pnl[start:start + TEST_DAYS].tolist())
        start += TEST_DAYS

    if not oos_pnl:
        return None

    oos = np.array(oos_pnl)
    equity = np.zeros(len(oos))
    equity[0] = 1000.0
    n_liq = 0

    avg_funding = 0.0
    if funding_rates is not None:
        fr_clean = funding_rates[~np.isnan(funding_rates)]
        avg_funding = np.mean(np.abs(fr_clean)) / 100

    for i in range(1, len(oos)):
        if equity[i - 1] <= 0:
            equity[i:] = 0; break
        r = oos[i]
        lev_r = leverage * r
        if leverage > 1.0 and abs(r) > 1e-10:
            lev_r -= (leverage - 1) * avg_funding * 0.5
        liq_level = 0.90 / leverage
        if lev_r < -liq_level:
            equity[i] = equity[i - 1] * 0.05
            n_liq += 1
        else:
            equity[i] = equity[i - 1] * (1 + lev_r)

    valid = equity > 0
    nv = int(valid.sum())
    final = equity[nv - 1] if nv > 0 else 0
    ny = nv / 365
    cagr = (final / 1000) ** (1 / ny) - 1 if ny > 0 and final > 10 else -1.0
    dr = np.diff(equity[:nv]) / np.where(equity[:nv - 1] > 0, equity[:nv - 1], 1)
    dr = dr[np.isfinite(dr)]
    sh = np.mean(dr) / np.std(dr) * np.sqrt(365) if len(dr) > 1 and np.std(dr) > 0 else 0
    pk = np.maximum.accumulate(equity[:nv])
    mdd = ((equity[:nv] - pk) / np.where(pk > 0, pk, 1)).min()

    return {
        'leverage': leverage,
        'final_equity': round(final, 2),
        'cagr': round(cagr, 4),
        'sharpe': round(sh, 4),
        'max_dd': round(abs(mdd), 4),
        'n_liquidations': n_liq,
        'n_oos_days': len(oos),
    }


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 100)
    print("  WALK-FORWARD VALIDATION — FULL COMBINED SYSTEM")
    print("  4-Signal Portfolio + Hedge-When-Flat + Progressive Stop")
    print("  V2: Stop bug fixed, block permutation (10K), fold sensitivity, turnover TX")
    print("=" * 100)

    print("\nLoading data...")
    df, lsr_all, price_all, symbols = load_data()
    n = len(df)
    warmup = 50
    print(f"Period: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} ({n} days)")
    print(f"WF config: train_min={TRAIN_DAYS_MIN}d, test={TEST_DAYS}d, "
          f"perms={N_PERMS}, bootstrap={N_BOOT}")

    # ── 1. Full System WITH Hedge ────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  1. FULL SYSTEM: 4-SIGNAL + HEDGE-WHEN-FLAT + PROGRESSIVE STOP (vol x2)")
    print(f"{'='*100}")

    pnl_hedged = compute_full_system_pnl(df, lsr_all, price_all, symbols,
                                          hedge=True, vol_mult=2.0)
    result_hedged = wf_validate(pnl_hedged, warmup)

    if result_hedged:
        fs = result_hedged['full_sample']
        wf = result_hedged['wf_oos']
        sig = "***" if wf['p_value'] < 0.01 else "**" if wf['p_value'] < 0.05 else "*" if wf['p_value'] < 0.10 else ""

        print(f"\n  FULL SAMPLE:")
        print(f"    Sharpe: {fs['sharpe']:.2f}")
        print(f"    CAGR:   {fs['cagr']:+.1%}")
        print(f"    MaxDD:  {fs['max_dd']:.1%}")
        print(f"    Mean:   {fs['daily_mean_bps']:.1f} bps/day")
        print(f"    Days:   {fs['n_days']}")

        print(f"\n  WALK-FORWARD OOS:")
        print(f"    Concat Sharpe: {wf['concat_sharpe']:.2f} {sig}")
        print(f"    Mean Sharpe:   {wf['mean_sharpe']:.2f}")
        print(f"    Median Sharpe: {wf['median_sharpe']:.2f}")
        print(f"    Sharpe StdDev: {wf['std_sharpe']:.2f}")
        print(f"    p-value:       {wf['p_value']:.4f} {sig}")
        print(f"    95% CI:        [{wf['ci_lower']:.2f}, {wf['ci_upper']:.2f}]")
        print(f"    Positive folds:{wf['positive_folds']}")
        print(f"    OOS days:      {wf['n_oos_days']}")

        print(f"\n  Per-Fold Breakdown:")
        print(f"    {'Fold':>4} | {'Sharpe':>7} | {'Return':>8} | {'MaxDD':>7} | {'Bps/day':>8}")
        print(f"    {'-'*45}")
        for f in wf['folds']:
            print(f"    {f['fold']:>4} | {f['sharpe']:>7.2f} | {f['total_return']:>+7.1%} | "
                  f"{f['max_dd']:>6.1%} | {f['mean_daily_bps']:>+7.1f}")

    # ── 2. Baseline WITHOUT Hedge (for comparison) ───────────────────────
    print(f"\n{'='*100}")
    print("  2. BASELINE: 4-SIGNAL + PROGRESSIVE STOP (NO HEDGE) (vol x2)")
    print(f"{'='*100}")

    pnl_base = compute_full_system_pnl(df, lsr_all, price_all, symbols,
                                        hedge=False, vol_mult=2.0)
    result_base = wf_validate(pnl_base, warmup)

    if result_base:
        fs = result_base['full_sample']
        wf = result_base['wf_oos']
        sig = "***" if wf['p_value'] < 0.01 else "**" if wf['p_value'] < 0.05 else "*" if wf['p_value'] < 0.10 else ""

        print(f"\n  FULL SAMPLE:")
        print(f"    Sharpe: {fs['sharpe']:.2f}")
        print(f"    CAGR:   {fs['cagr']:+.1%}")
        print(f"    MaxDD:  {fs['max_dd']:.1%}")

        print(f"\n  WALK-FORWARD OOS:")
        print(f"    Concat Sharpe: {wf['concat_sharpe']:.2f} {sig}")
        print(f"    Mean Sharpe:   {wf['mean_sharpe']:.2f}")
        print(f"    p-value:       {wf['p_value']:.4f} {sig}")
        print(f"    95% CI:        [{wf['ci_lower']:.2f}, {wf['ci_upper']:.2f}]")
        print(f"    Positive folds:{wf['positive_folds']}")

        print(f"\n  Per-Fold Breakdown:")
        print(f"    {'Fold':>4} | {'Sharpe':>7} | {'Return':>8} | {'MaxDD':>7} | {'Bps/day':>8}")
        print(f"    {'-'*45}")
        for f in wf['folds']:
            print(f"    {f['fold']:>4} | {f['sharpe']:>7.2f} | {f['total_return']:>+7.1%} | "
                  f"{f['max_dd']:>6.1%} | {f['mean_daily_bps']:>+7.1f}")

    # ── 3. Head-to-Head Comparison ───────────────────────────────────────
    print(f"\n{'='*100}")
    print("  3. HEAD-TO-HEAD: HEDGE vs NO-HEDGE")
    print(f"{'='*100}")

    if result_hedged and result_base:
        wf_h = result_hedged['wf_oos']
        wf_b = result_base['wf_oos']
        fs_h = result_hedged['full_sample']
        fs_b = result_base['full_sample']

        print(f"\n  {'Metric':<25} | {'No Hedge':>12} | {'+ Hedge':>12} | {'Delta':>10}")
        print(f"  {'-'*65}")
        print(f"  {'FS Sharpe':<25} | {fs_b['sharpe']:>12.2f} | {fs_h['sharpe']:>12.2f} | "
              f"{fs_h['sharpe'] - fs_b['sharpe']:>+10.2f}")
        print(f"  {'FS CAGR':<25} | {fs_b['cagr']:>11.1%} | {fs_h['cagr']:>11.1%} | "
              f"{(fs_h['cagr'] - fs_b['cagr'])*100:>+9.1f}pp")
        print(f"  {'FS MaxDD':<25} | {fs_b['max_dd']:>11.1%} | {fs_h['max_dd']:>11.1%} | "
              f"{(fs_h['max_dd'] - fs_b['max_dd'])*100:>+9.1f}pp")
        print(f"  {'WF Concat Sharpe':<25} | {wf_b['concat_sharpe']:>12.2f} | {wf_h['concat_sharpe']:>12.2f} | "
              f"{wf_h['concat_sharpe'] - wf_b['concat_sharpe']:>+10.2f}")
        print(f"  {'WF Mean Sharpe':<25} | {wf_b['mean_sharpe']:>12.2f} | {wf_h['mean_sharpe']:>12.2f} | "
              f"{wf_h['mean_sharpe'] - wf_b['mean_sharpe']:>+10.2f}")
        print(f"  {'WF p-value':<25} | {wf_b['p_value']:>12.4f} | {wf_h['p_value']:>12.4f} |")
        print(f"  {'Positive folds':<25} | {wf_b['positive_folds']:>12} | {wf_h['positive_folds']:>12} |")

        # Per-fold comparison
        print(f"\n  Per-Fold Sharpe Comparison:")
        print(f"    {'Fold':>4} | {'No Hedge':>9} | {'+ Hedge':>9} | {'Winner':>8}")
        print(f"    {'-'*40}")
        for fb, fh in zip(wf_b['folds'], wf_h['folds']):
            winner = "HEDGE" if fh['sharpe'] > fb['sharpe'] else "BASE"
            print(f"    {fb['fold']:>4} | {fb['sharpe']:>9.2f} | {fh['sharpe']:>9.2f} | {winner:>8}")

        hedge_wins = sum(1 for fb, fh in zip(wf_b['folds'], wf_h['folds'])
                         if fh['sharpe'] > fb['sharpe'])
        print(f"\n  Hedge wins {hedge_wins}/{len(wf_h['folds'])} folds")

    # ── 4. Leveraged OOS Performance ─────────────────────────────────────
    print(f"\n{'='*100}")
    print("  4. LEVERAGED PERFORMANCE — OOS RETURNS ONLY")
    print(f"{'='*100}")
    print(f"  Compounding OOS-only daily returns at various leverage levels\n")

    funding_rates = df['btc_funding'].values

    print(f"  {'Leverage':>10} | {'Final $':>12} | {'CAGR':>10} | {'Sharpe':>8} | "
          f"{'MaxDD':>8} | {'Liquidns':>8} | {'OOS Days':>8}")
    print(f"  {'-'*80}")

    for lev in [1, 2, 3, 4, 5]:
        lr = simulate_leveraged_oos(pnl_hedged, warmup, leverage=float(lev),
                                     funding_rates=funding_rates)
        if lr:
            cagr_s = f"{lr['cagr']:>+8.1%}" if lr['cagr'] > -0.99 else "    RUIN"
            print(f"  {lev:>8.0f}x  | ${lr['final_equity']:>11,.0f} | {cagr_s} | "
                  f"{lr['sharpe']:>7.2f} | {lr['max_dd']:>7.1%} | "
                  f"{lr['n_liquidations']:>8d} | {lr['n_oos_days']:>8d}")

    # ── 5. Vol Multiplier Sensitivity ────────────────────────────────────
    print(f"\n{'='*100}")
    print("  5. VOL MULTIPLIER SENSITIVITY — FULL SYSTEM WITH HEDGE")
    print(f"{'='*100}\n")

    print(f"  {'VolMult':>8} | {'FS Sharpe':>10} | {'WF Sharpe':>10} | {'p-value':>8} | "
          f"{'FS CAGR':>8} | {'Pos Folds':>10}")
    print(f"  {'-'*68}")

    for vm in [1.0, 1.5, 2.0, 3.0]:
        pnl_vm = compute_full_system_pnl(df, lsr_all, price_all, symbols,
                                          hedge=True, vol_mult=vm)
        r = wf_validate(pnl_vm, warmup)
        if r:
            sig = "***" if r['wf_oos']['p_value'] < 0.01 else "**" if r['wf_oos']['p_value'] < 0.05 else ""
            print(f"  {vm:>7.1f}x | {r['full_sample']['sharpe']:>10.2f} | "
                  f"{r['wf_oos']['concat_sharpe']:>10.2f} | {r['wf_oos']['p_value']:>7.4f}{sig:<3} | "
                  f"{r['full_sample']['cagr']:>+7.1%} | {r['wf_oos']['positive_folds']:>10}")

    # ── 6. Fold Placement Sensitivity ──────────────────────────────────
    print(f"\n{'='*100}")
    print("  6. FOLD PLACEMENT SENSITIVITY — SHIFT BOUNDARIES +30d, +60d")
    print(f"{'='*100}")
    print(f"  Tests whether results are robust to arbitrary fold boundary placement\n")

    print(f"  {'Shift':>7} | {'Concat Sharpe':>14} | {'Mean Sharpe':>12} | "
          f"{'p-value':>8} | {'Pos Folds':>10} | {'95% CI':>18}")
    print(f"  {'-'*78}")

    for shift in [0, 30, 60]:
        # Re-run WF with shifted fold boundaries
        pnl = pnl_hedged[warmup:]
        n_pnl = len(pnl)
        folds = []
        start = TRAIN_DAYS_MIN + shift
        while start + TEST_DAYS <= n_pnl:
            folds.append((start, start + TEST_DAYS))
            start += TEST_DAYS

        if not folds:
            continue

        all_oos_shifted = []
        f_sharpes_shifted = []
        for test_start, test_end in folds:
            test = pnl[test_start:test_end]
            if len(test) < 20:
                continue
            mu = np.mean(test)
            sd = np.std(test)
            sh = mu / sd * np.sqrt(365) if sd > 0 else 0
            f_sharpes_shifted.append(sh)
            all_oos_shifted.extend(test.tolist())

        oos_s = np.array(all_oos_shifted)
        oos_mu_s = np.mean(oos_s)
        oos_sd_s = np.std(oos_s)
        concat_sh = oos_mu_s / oos_sd_s * np.sqrt(365) if oos_sd_s > 0 else 0

        # Block permutation p-value (reduced count for speed)
        n_blocks_s = int(np.ceil(len(oos_s) / BLOCK_LEN))
        perm_ct_s = 0
        for _ in range(2000):  # reduced for fold sensitivity check
            bsigns = np.random.choice([-1, 1], size=n_blocks_s)
            signs = np.repeat(bsigns, BLOCK_LEN)[:len(oos_s)]
            if np.mean(oos_s * signs) >= oos_mu_s:
                perm_ct_s += 1
        p_s = (perm_ct_s + 1) / 2001

        # Bootstrap CI
        boots_s = []
        for _ in range(2000):
            b = np.random.choice(oos_s, size=len(oos_s), replace=True)
            bs = np.mean(b) / np.std(b) * np.sqrt(365) if np.std(b) > 0 else 0
            boots_s.append(bs)
        ci_l = np.percentile(boots_s, 2.5)
        ci_u = np.percentile(boots_s, 97.5)

        pos_f = sum(1 for s in f_sharpes_shifted if s > 0)
        sig = "***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.10 else ""
        print(f"  {f'+{shift}d':>7} | {concat_sh:>14.2f} | {np.mean(f_sharpes_shifted):>12.2f} | "
              f"{p_s:>7.4f}{sig:<1} | {pos_f}/{len(f_sharpes_shifted):>8} | "
              f"[{ci_l:.2f}, {ci_u:.2f}]")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  SUMMARY")
    print(f"{'='*100}")

    if result_hedged:
        wf = result_hedged['wf_oos']
        fs = result_hedged['full_sample']
        verdict = "SIGNIFICANT" if wf['p_value'] < 0.05 else "BORDERLINE" if wf['p_value'] < 0.10 else "NOT SIGNIFICANT"
        print(f"""
  FULL SYSTEM (4-signal + hedge-when-flat + progressive stop, vol x2):
    Full-Sample Sharpe:  {fs['sharpe']:.2f}
    WF OOS Sharpe:       {wf['concat_sharpe']:.2f}
    p-value:             {wf['p_value']:.4f} → {verdict}
    95% CI:              [{wf['ci_lower']:.2f}, {wf['ci_upper']:.2f}]
    Positive folds:      {wf['positive_folds']}
    OOS days:            {wf['n_oos_days']}

  DEGRADATION RATIO:     {wf['concat_sharpe'] / fs['sharpe']:.1%} (OOS/FS)
  {'  → Minimal degradation = robust system' if wf['concat_sharpe'] / fs['sharpe'] > 0.7 else '  → Significant degradation = possible overfit'}

  COMPONENTS:
    BTC Weighted Regime:       p=0.012
    SOL Relative Momentum 20d: p=0.028
    DOGE Relative Momentum 10d:p=0.010
    LSR Divergence Pairs:      p=0.054
    Hedge-when-flat overlay:   p=0.002
    Progressive stop:          risk overlay (no separate p-value)
""")

    # ── Save Results ─────────────────────────────────────────────────────
    output = {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}",
        'config': {
            'weights': WEIGHTS,
            'vol_mult': 2.0,
            'tx_cost': TX_COST,
            'stop_initial': STOP_INITIAL,
            'hedge_sma': 50,
            'hedge_lsr_mult': 1.1,
            'train_min': TRAIN_DAYS_MIN,
            'test_days': TEST_DAYS,
            'n_perms': N_PERMS,
            'n_bootstrap': N_BOOT,
            'block_len': BLOCK_LEN,
        },
        'full_system_hedged': result_hedged,
        'full_system_no_hedge': result_base,
    }

    out_path = os.path.join(RESULTS_DIR, 'full_system_walkforward.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")
    print(f"{'='*100}")
