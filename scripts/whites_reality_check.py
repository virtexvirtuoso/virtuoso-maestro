#!/usr/bin/env python3
"""
White's Reality Check & Hansen's Superior Predictive Ability (SPA) Test.

Tests whether the best strategy's performance is significant AFTER
accounting for the full graveyard of strategies tested during research.

We tested ~10 strategy variants and kept 4 survivors + 1 hedge.
A naive p=0.001 on the combined system doesn't account for this
multiple testing / data snooping. This test does.

Method:
  1. Reconstruct daily returns for ALL strategies tested (survivors + failures)
  2. Center returns under H0 (no strategy works)
  3. Block-bootstrap the full k×T return matrix (preserving cross-correlation)
  4. For each bootstrap: compute max(mean return) across all k strategies
  5. p-value = fraction of bootstrap max means >= observed max mean

References:
  - White (2000): "A Reality Check for Data Snooping"
  - Hansen (2005): "A Test for Superior Predictive Ability"
"""
import duckdb, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
TX_COST = 0.001
RESULTS_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
BLOCK_LEN = 40
N_BOOTSTRAP = 10000
np.random.seed(42)

STOP_INITIAL = 0.15
STOP_PROFIT_20 = 0.10
STOP_PROFIT_50 = 0.07
WEIGHTS = {'btc': 0.45, 'sol': 0.20, 'lsr': 0.15, 'doge': 0.20}


# ── Data Loading (reused from walkforward_full_system.py) ───────────────

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
    eth = con.execute(
        "SELECT date, close as eth_close FROM perps_daily WHERE symbol='ETH' ORDER BY date"
    ).df()

    all_symbols = ['BTC', 'ETH', 'SOL', 'LINK', 'AVAX', 'DOGE', 'DOT', 'NEAR',
                   'ATOM', 'BNB', 'XRP', 'UNI', 'ADA', 'FIL']
    lsr_all, price_all = {}, {}
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
    for src in [funding, lsr, liq, taker, sol, sol_lsr, doge, eth]:
        df = df.merge(src, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df, lsr_all, price_all, all_symbols


# ── Signal Functions (survivors — exact copies) ─────────────────────────

def apply_progressive_stop(position, price_series):
    pos_out = position.copy()
    entry_price = peak_price = None
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
            stop_pct = STOP_PROFIT_50 if profit_pct > 0.50 else (STOP_PROFIT_20 if profit_pct > 0.20 else STOP_INITIAL)
            dd = (peak_price - px) / peak_price if peak_price > 0 else 0
            if dd > stop_pct:
                stopped = True
        elif raw_pos == 0:
            entry_price = peak_price = None; stopped = False
        elif stopped and raw_pos > 0:
            pos_out[i] = 0.0
    return pos_out


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
    return position.shift(1).fillna(0).values, btc_ret.values


def sol_relmom_signal(df):
    btc_ret_20d = df['btc_close'].pct_change(20)
    sol_ret_20d = df['sol_close'].pct_change(20)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((sol_ret_20d > btc_ret_20d) & (df['btc_close'] > btc_sma50)).astype(float)
    return signal.shift(1).fillna(0).values, df['sol_close'].pct_change().values


def doge_relmom_signal(df):
    btc_ret_10d = df['btc_close'].pct_change(10)
    doge_ret_10d = df['doge_close'].pct_change(10)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((doge_ret_10d > btc_ret_10d) & (df['btc_close'] > btc_sma50)).astype(float)
    return signal.shift(1).fillna(0).values, df['doge_close'].pct_change().values


def lsr_divergence_pnl(lsr_all, price_all, symbols, dates):
    lsr_df = pd.DataFrame(lsr_all).reindex(dates)
    price_df = pd.DataFrame(price_all).reindex(dates)
    ret_df = price_df.pct_change()
    daily_pnl = np.zeros(len(dates))
    prev_longs, prev_shorts = set(), set()
    for i in range(1, len(dates)):
        prev_date, curr_date = dates[i - 1], dates[i]
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
        n_changed = len(longs - prev_longs) + len(shorts - prev_shorts)
        tc = n_changed * 2 * TX_COST / 6
        daily_pnl[i] = (long_ret - short_ret) - tc
        prev_longs, prev_shorts = longs, shorts
    return daily_pnl


def compute_hedge_trigger(df, sma_period=50, lsr_mult=1.1):
    close = df['btc_close'].values
    sma = pd.Series(close).rolling(sma_period, min_periods=sma_period // 2).mean().values
    lsr_v = df['btc_lsr'].values
    lsr_med = pd.Series(lsr_v).rolling(30, min_periods=10).median().values
    n = len(close)
    trigger = np.zeros(n)
    for i in range(1, n):
        bearish = False
        if not np.isnan(sma[i - 1]) and close[i - 1] < sma[i - 1]:
            bearish = True
        if not np.isnan(lsr_med[i - 1]) and lsr_v[i - 1] > lsr_med[i - 1] * lsr_mult:
            bearish = True
        trigger[i] = 1.0 if bearish else 0.0
    return trigger


# ── Failed Strategy Reconstructions ─────────────────────────────────────

def macro_momentum_signal(df):
    """Macro Momentum (failed) — SMA crossover + trailing stop on BTC."""
    close = df['btc_close']
    sma200 = close.rolling(200).mean()
    sma50 = close.rolling(50).mean()
    roc30 = close.pct_change(30)
    entry = (close > sma200) & (sma50 > sma200) & (roc30 > 0)
    # Build position with trailing stop
    n = len(close)
    pos = np.zeros(n)
    in_pos = False
    peak = 0.0
    for i in range(200, n):
        if not in_pos:
            if entry.iloc[i]:
                in_pos = True
                peak = close.iloc[i]
                pos[i] = 1.0
        else:
            peak = max(peak, close.iloc[i])
            if close.iloc[i] < peak * (1 - 0.15):
                in_pos = False
                pos[i] = 0.0
            else:
                pos[i] = 1.0
    pos = pd.Series(pos).shift(1).fillna(0).values
    btc_ret = close.pct_change().values
    return pos, btc_ret


def confluence_v31_signal(df):
    """V3.1 5-Signal Confluence (failed) — score-based entry+sizing on BTC."""
    lsr_med = df['btc_lsr'].rolling(30, min_periods=10).median()
    lsr_s = (df['btc_lsr'] < lsr_med).astype(float)
    fund_s = (df['btc_funding'] < 0.03).astype(float)
    liq_p80 = df['btc_total_liq'].rolling(30, min_periods=10).quantile(0.8)
    liq_s = (df['btc_total_liq'] < liq_p80).astype(float)
    taker_s = (df['btc_taker_ratio'] > 1.0).astype(float)
    # Add SMA trend as 5th signal (V3.1 style)
    btc_sma = df['btc_close'].rolling(200, min_periods=100).mean()
    trend_s = (df['btc_close'] > btc_sma).astype(float)
    score = (lsr_s + fund_s + liq_s + taker_s + trend_s) / 5.0
    # V3.1 uses score for BOTH entry and sizing (the failure mode)
    position = score * (score > 0.6).astype(float)
    position = position.shift(1).fillna(0).values
    btc_ret = df['btc_close'].pct_change().values
    return position, btc_ret


def funding_rate_signal(df):
    """Funding Rate Arb (failed) — long when funding negative, short when high."""
    fr = df['btc_funding']
    fr_ma = fr.rolling(7, min_periods=3).mean()
    # Long when funding deeply negative (shorts paying longs)
    pos = np.where(fr_ma < -0.01, 1.0, np.where(fr_ma > 0.05, -1.0, 0.0))
    pos = pd.Series(pos).shift(1).fillna(0).values
    btc_ret = df['btc_close'].pct_change().values
    return pos, btc_ret


def sma_crossover_signal(df):
    """Simple SMA Crossover (failed) — naive trend following baseline."""
    close = df['btc_close']
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    signal = (sma50 > sma200).astype(float)
    signal = signal.shift(1).fillna(0).values
    btc_ret = close.pct_change().values
    return signal, btc_ret


def eth_relmom_signal(df):
    """ETH Relative Momentum (tested, not selected) — same pattern as SOL/DOGE."""
    btc_ret_20d = df['btc_close'].pct_change(20)
    eth_ret_20d = df['eth_close'].pct_change(20)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((eth_ret_20d > btc_ret_20d) & (df['btc_close'] > btc_sma50)).astype(float)
    return signal.shift(1).fillna(0).values, df['eth_close'].pct_change().values


def buyhold_btc(df):
    """Buy & Hold BTC (benchmark)."""
    pos = np.ones(len(df))
    pos = pd.Series(pos).shift(1).fillna(0).values
    btc_ret = df['btc_close'].pct_change().values
    return pos, btc_ret


# ── Full Combined System PnL (survivor — the "best" strategy) ──────────

def compute_full_system_pnl(df, lsr_all, price_all, symbols, hedge=True, vol_mult=2.0):
    n = len(df)
    dates = df['date'].values
    btc_pos, btc_ret = btc_regime_signal(df)
    sol_pos, sol_ret = sol_relmom_signal(df)
    doge_pos, doge_ret = doge_relmom_signal(df)
    lsr_pnl = lsr_divergence_pnl(lsr_all, price_all, symbols, dates)
    btc_pos = apply_progressive_stop(btc_pos, df['btc_close'].values)
    sol_pos = apply_progressive_stop(sol_pos, df['sol_close'].values)
    doge_pos = apply_progressive_stop(doge_pos, df['doge_close'].values)
    hedge_trigger = compute_hedge_trigger(df) if hedge else np.zeros(n)
    funding = df['btc_funding'].values
    daily_pnl = np.zeros(n)
    prev_bp = prev_sp = prev_dp = prev_h = 0.0
    w = WEIGHTS
    for i in range(1, n):
        br = btc_ret[i] if not np.isnan(btc_ret[i]) else 0.0
        sr = sol_ret[i] if not np.isnan(sol_ret[i]) else 0.0
        dr = doge_ret[i] if not np.isnan(doge_ret[i]) else 0.0
        lp = lsr_pnl[i]
        bp = btc_pos[i] * vol_mult
        sp = sol_pos[i]
        dp = doge_pos[i]
        btc_tc = abs(bp - prev_bp) * TX_COST
        sol_tc = abs(sp - prev_sp) * TX_COST
        doge_tc = abs(dp - prev_dp) * TX_COST
        fr = funding[i] / 100 if not np.isnan(funding[i]) else 0.0
        btc_fc = bp * fr if bp > 0 else 0.0
        sol_fc = sp * fr if sp > 0 else 0.0
        doge_fc = dp * fr if dp > 0 else 0.0
        base_pnl = ((bp * br - btc_tc - btc_fc) * w['btc'] +
                     (sp * sr - sol_tc - sol_fc) * w['sol'] +
                     (dp * dr - doge_tc - doge_fc) * w['doge'] +
                     lp * w['lsr'])
        hedge_pnl = 0.0
        if hedge:
            n_active = int(bp > 0) + int(sp > 0) + int(dp > 0)
            if n_active == 0 and hedge_trigger[i] == 1:
                h = -1.0
                h_tc = abs(h - prev_h) * TX_COST
                h_fc = h * fr
                hedge_pnl = (h * br - h_tc - h_fc) * w['btc']
                prev_h = h
            else:
                if prev_h != 0:
                    hedge_pnl = -abs(0 - prev_h) * TX_COST * w['btc']
                prev_h = 0.0
        daily_pnl[i] = base_pnl + hedge_pnl
        prev_bp, prev_sp, prev_dp = bp, sp, dp
    return daily_pnl


# ── White's Reality Check + Hansen's SPA ────────────────────────────────

def compute_strategy_returns(pos, ret):
    """Compute daily returns for a single-asset strategy with TX costs."""
    n = len(pos)
    pnl = np.zeros(n)
    for i in range(1, n):
        r = ret[i] if not np.isnan(ret[i]) else 0.0
        tc = abs(pos[i] - pos[i - 1]) * TX_COST
        pnl[i] = pos[i] * r - tc
    return pnl


def block_bootstrap_index(n, block_len, rng):
    """Generate block bootstrap sample indices."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) % n for s in starts])
    return idx[:n]


def whites_reality_check(return_matrix, n_bootstrap=N_BOOTSTRAP, block_len=BLOCK_LEN):
    """
    White's Reality Check for data snooping.

    return_matrix: (T, k) array of daily returns for k strategies
    Returns: p-value for the best strategy after accounting for snooping
    """
    T, k = return_matrix.shape
    rng = np.random.default_rng(42)

    # Observed test statistic: max mean return across strategies
    mean_returns = np.mean(return_matrix, axis=0)
    best_idx = np.argmax(mean_returns)
    obs_max = mean_returns[best_idx]

    # Center returns under H0 (no strategy has positive mean)
    centered = return_matrix - mean_returns[np.newaxis, :]

    # Block bootstrap
    boot_max = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = block_bootstrap_index(T, block_len, rng)
        boot_returns = centered[idx, :]
        boot_means = np.mean(boot_returns, axis=0)
        boot_max[b] = np.max(boot_means)

    # p-value: fraction of bootstrap max means >= observed
    p_value = (np.sum(boot_max >= obs_max) + 1) / (n_bootstrap + 1)
    return p_value, best_idx, obs_max, boot_max


def hansens_spa(return_matrix, n_bootstrap=N_BOOTSTRAP, block_len=BLOCK_LEN):
    """
    Hansen's Superior Predictive Ability test (studentized version).
    More powerful than White's RC.
    """
    T, k = return_matrix.shape
    rng = np.random.default_rng(42)

    mean_returns = np.mean(return_matrix, axis=0)
    std_returns = np.std(return_matrix, axis=0, ddof=1)
    std_returns = np.maximum(std_returns, 1e-10)  # prevent div by zero

    # Studentized statistic
    t_stats = mean_returns / (std_returns / np.sqrt(T))
    obs_max_t = np.max(t_stats)
    best_idx = np.argmax(t_stats)

    # Center under H0
    centered = return_matrix - mean_returns[np.newaxis, :]

    boot_max_t = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = block_bootstrap_index(T, block_len, rng)
        boot_returns = centered[idx, :]
        boot_means = np.mean(boot_returns, axis=0)
        boot_stds = np.std(boot_returns, axis=0, ddof=1)
        boot_stds = np.maximum(boot_stds, 1e-10)
        boot_t = boot_means / (boot_stds / np.sqrt(T))
        boot_max_t[b] = np.max(boot_t)

    p_value = (np.sum(boot_max_t >= obs_max_t) + 1) / (n_bootstrap + 1)
    return p_value, best_idx, obs_max_t, boot_max_t


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 72)
    print("  WHITE'S REALITY CHECK & HANSEN'S SPA TEST")
    print("  Accounting for data snooping across ALL strategies tested")
    print("=" * 72)
    print()

    # Load data
    print("[1/4] Loading data from DuckDB...")
    df, lsr_all, price_all, all_symbols = load_data()
    dates = df['date'].values
    n = len(df)
    warmup = 200  # skip first 200 days for SMA warmup
    print(f"  Data: {n} days, using [{warmup}:] = {n - warmup} days")
    print()

    # Generate all strategy returns
    print("[2/4] Generating daily returns for ALL strategies tested...")
    strategies = {}

    # Survivors
    btc_pos, btc_ret = btc_regime_signal(df)
    btc_pos = apply_progressive_stop(btc_pos, df['btc_close'].values)
    strategies['BTC_Regime'] = compute_strategy_returns(btc_pos * 2.0, btc_ret)

    sol_pos, sol_ret = sol_relmom_signal(df)
    sol_pos = apply_progressive_stop(sol_pos, df['sol_close'].values)
    strategies['SOL_RelMom'] = compute_strategy_returns(sol_pos, sol_ret)

    doge_pos, doge_ret = doge_relmom_signal(df)
    doge_pos = apply_progressive_stop(doge_pos, df['doge_close'].values)
    strategies['DOGE_RelMom'] = compute_strategy_returns(doge_pos, doge_ret)

    strategies['LSR_Pairs'] = lsr_divergence_pnl(lsr_all, price_all, all_symbols, dates)

    # Combined systems
    strategies['Combined_Hedged'] = compute_full_system_pnl(df, lsr_all, price_all, all_symbols, hedge=True)
    strategies['Combined_NoHedge'] = compute_full_system_pnl(df, lsr_all, price_all, all_symbols, hedge=False)

    # Failed strategies
    mm_pos, mm_ret = macro_momentum_signal(df)
    strategies['MacroMomentum'] = compute_strategy_returns(mm_pos, mm_ret)

    v31_pos, v31_ret = confluence_v31_signal(df)
    strategies['V31_Confluence'] = compute_strategy_returns(v31_pos, v31_ret)

    fr_pos, fr_ret = funding_rate_signal(df)
    strategies['FundingRate'] = compute_strategy_returns(fr_pos, fr_ret)

    sma_pos, sma_ret = sma_crossover_signal(df)
    strategies['SMA_Crossover'] = compute_strategy_returns(sma_pos, sma_ret)

    eth_pos, eth_ret = eth_relmom_signal(df)
    eth_pos = apply_progressive_stop(eth_pos, df['eth_close'].values)
    strategies['ETH_RelMom'] = compute_strategy_returns(eth_pos, eth_ret)

    bh_pos, bh_ret = buyhold_btc(df)
    strategies['BuyHold_BTC'] = compute_strategy_returns(bh_pos, bh_ret)

    # Trim warmup and build matrix
    strat_names = list(strategies.keys())
    k = len(strat_names)
    return_matrix = np.column_stack([strategies[s][warmup:] for s in strat_names])
    T = return_matrix.shape[0]

    print(f"  {k} strategies, {T} days each")
    print()

    # Print individual strategy stats
    print("  Strategy Summary (post-warmup):")
    print(f"  {'Strategy':<22} {'Mean bps':>10} {'Sharpe':>8} {'StdDev':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8}")
    for j, name in enumerate(strat_names):
        rets = return_matrix[:, j]
        mean_bps = np.mean(rets) * 10000
        std = np.std(rets, ddof=1)
        sharpe = np.mean(rets) / std * np.sqrt(365) if std > 0 else 0
        print(f"  {name:<22} {mean_bps:>+10.2f} {sharpe:>8.2f} {std*100:>7.3f}%")
    print()

    # Run White's Reality Check
    print(f"[3/4] Running White's Reality Check ({N_BOOTSTRAP:,} block-bootstrap samples, block={BLOCK_LEN}d)...")
    wrc_p, wrc_best, wrc_obs, wrc_dist = whites_reality_check(return_matrix)
    print(f"  Best strategy: {strat_names[wrc_best]}")
    print(f"  Observed max mean return: {wrc_obs*10000:.2f} bps/day")
    print(f"  White's RC p-value: {wrc_p:.4f}", end="")
    if wrc_p < 0.01:
        print("  ***")
    elif wrc_p < 0.05:
        print("  **")
    elif wrc_p < 0.10:
        print("  *")
    else:
        print("  (not significant)")
    print()

    # Run Hansen's SPA
    print(f"[4/4] Running Hansen's SPA test ({N_BOOTSTRAP:,} samples)...")
    spa_p, spa_best, spa_obs, spa_dist = hansens_spa(return_matrix)
    print(f"  Best strategy (studentized): {strat_names[spa_best]}")
    print(f"  Observed max t-stat: {spa_obs:.3f}")
    print(f"  Hansen's SPA p-value: {spa_p:.4f}", end="")
    if spa_p < 0.01:
        print("  ***")
    elif spa_p < 0.05:
        print("  **")
    elif spa_p < 0.10:
        print("  *")
    else:
        print("  (not significant)")
    print()

    # Summary
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if wrc_p < 0.05 and spa_p < 0.05:
        print("  PASS: Edge survives data-snooping correction.")
        print(f"  The best strategy ({strat_names[wrc_best]}) is significant even")
        print(f"  after accounting for testing {k} strategies on the same data.")
    elif wrc_p < 0.10 or spa_p < 0.10:
        print("  MARGINAL: Borderline significance after snooping correction.")
        print("  Proceed with caution; forward test is critical.")
    else:
        print("  FAIL: Edge does not survive data-snooping correction.")
        print("  The observed performance could be explained by testing many strategies.")
    print()
    print(f"  White's RC:  p = {wrc_p:.4f} (unstudentized)")
    print(f"  Hansen's SPA: p = {spa_p:.4f} (studentized, more powerful)")
    print(f"  Strategies tested: {k}")
    print(f"  Days analyzed: {T}")
    print(f"  Block length: {BLOCK_LEN}")
    print(f"  Bootstrap samples: {N_BOOTSTRAP:,}")
    print("=" * 72)

    # Save results
    results = {
        'run_date': datetime.now().isoformat(),
        'n_strategies': k,
        'n_days': int(T),
        'block_len': BLOCK_LEN,
        'n_bootstrap': N_BOOTSTRAP,
        'strategy_names': strat_names,
        'strategy_sharpes': {
            name: float(np.mean(return_matrix[:, j]) / np.std(return_matrix[:, j], ddof=1) * np.sqrt(365))
            for j, name in enumerate(strat_names)
        },
        'strategy_mean_bps': {
            name: float(np.mean(return_matrix[:, j]) * 10000)
            for j, name in enumerate(strat_names)
        },
        'whites_rc': {
            'p_value': float(wrc_p),
            'best_strategy': strat_names[wrc_best],
            'obs_max_mean_bps': float(wrc_obs * 10000),
            'boot_5th_pctile_bps': float(np.percentile(wrc_dist, 5) * 10000),
            'boot_95th_pctile_bps': float(np.percentile(wrc_dist, 95) * 10000),
        },
        'hansens_spa': {
            'p_value': float(spa_p),
            'best_strategy': strat_names[spa_best],
            'obs_max_t': float(spa_obs),
            'boot_5th_pctile': float(np.percentile(spa_dist, 5)),
            'boot_95th_pctile': float(np.percentile(spa_dist, 95)),
        },
    }
    out_path = os.path.join(RESULTS_DIR, 'whites_reality_check.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
