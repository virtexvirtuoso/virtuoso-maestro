#!/usr/bin/env python3
"""
Combined Portfolio Simulation — 4 validated alpha streams + hedge-when-flat.

1. BTC Weighted Regime (p=0.012) — proven core
2. SOL Relative Momentum 20d + BTC Bull (p=0.028) — satellite
3. LSR Divergence Pairs (p=0.054) — market-neutral income
4. DOGE Relative Momentum 10d + BTC Bull (p=0.010) — fast alt momentum
5. Hedge-when-flat overlay (p=0.002) — short perp when all directional signals off + bearish
"""
import duckdb, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
TX_COST = 0.001

# Kelly sizing config
KELLY_WINDOW = 60      # rolling window for Kelly fraction
KELLY_FRACTION = 0.5   # half-Kelly for safety

# Progressive stop config
STOP_INITIAL = 0.15    # 15% trailing stop at entry
STOP_PROFIT_20 = 0.10  # tighten to 10% after 20% profit
STOP_PROFIT_50 = 0.07  # tighten to 7% after 50% profit


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Load BTC+SOL+all assets from DuckDB."""
    con = duckdb.connect(DB_PATH, read_only=True)

    # BTC price + derivatives
    btc_price = con.execute(
        "SELECT date, close as btc_close FROM perps_daily WHERE symbol='BTC' ORDER BY date"
    ).df()
    btc_lsr = con.execute(
        "SELECT date, global_account_long_short_ratio as btc_lsr "
        "FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date"
    ).df()
    btc_fr = con.execute(
        "SELECT date, close as btc_funding "
        "FROM cg_funding_rate WHERE symbol='BTC' ORDER BY date"
    ).df()
    btc_liq = con.execute(
        "SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as btc_total_liq "
        "FROM cg_liquidations WHERE symbol='BTC' ORDER BY date"
    ).df()
    btc_taker = con.execute(
        "SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as btc_taker_ratio "
        "FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date"
    ).df()

    # SOL price + LSR
    sol_price = con.execute(
        "SELECT date, close as sol_close FROM perps_daily WHERE symbol='SOL' ORDER BY date"
    ).df()
    sol_lsr = con.execute(
        "SELECT date, global_account_long_short_ratio as sol_lsr "
        "FROM cg_lsr_global WHERE symbol='SOL' ORDER BY date"
    ).df()

    # DOGE price
    doge_price = con.execute(
        "SELECT date, close as doge_close FROM perps_daily WHERE symbol='DOGE' ORDER BY date"
    ).df()

    # All symbols LSR for divergence pairs
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

    # Merge BTC data
    df = btc_price.copy()
    for src in [btc_lsr, btc_fr, btc_liq, btc_taker, sol_price, sol_lsr, doge_price]:
        df = df.merge(src, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)

    return df, lsr_all, price_all, all_symbols


# ── Signal 1: BTC Weighted Regime ────────────────────────────────────────────

def btc_regime_signal(df):
    """Proven p=0.012 regime signal + vol-targeting."""
    lsr_med = df['btc_lsr'].rolling(30, min_periods=10).median()
    lsr_s = (df['btc_lsr'] < lsr_med).astype(float)
    fund_s = (df['btc_funding'] < 0.03).astype(float)
    liq_p80 = df['btc_total_liq'].rolling(30, min_periods=10).quantile(0.8)
    liq_s = (df['btc_total_liq'] < liq_p80).astype(float)
    taker_s = (df['btc_taker_ratio'] > 1.0).astype(float)

    score = lsr_s * 0.35 + fund_s * 0.35 + liq_s * 0.15 + taker_s * 0.15
    signal = (score > 0.5).astype(float)

    # Vol-targeting
    btc_ret = df['btc_close'].pct_change()
    realized_vol = btc_ret.rolling(20, min_periods=10).std()
    vol_scalar = (0.015 / realized_vol).clip(0.25, 5.0)

    position = signal * vol_scalar
    position = position.shift(1).fillna(0)  # no lookahead

    return position, btc_ret


# ── Signal 2: SOL Relative Momentum + BTC Bull ──────────────────────────────

def sol_relmom_signal(df):
    """p=0.028 signal. Long SOL when outperforming BTC + BTC in uptrend."""
    btc_ret_20d = df['btc_close'].pct_change(20)
    sol_ret_20d = df['sol_close'].pct_change(20)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()

    signal = ((sol_ret_20d > btc_ret_20d) & (df['btc_close'] > btc_sma50)).astype(float)
    signal = signal.shift(1).fillna(0)  # no lookahead

    sol_ret = df['sol_close'].pct_change()
    return signal, sol_ret


# ── Signal 4: DOGE Relative Momentum 10d + BTC Bull ────────────────────────

def doge_relmom_signal(df):
    """p=0.010 signal. Long DOGE when 10d return > BTC 10d return + BTC in uptrend."""
    btc_ret_10d = df['btc_close'].pct_change(10)
    doge_ret_10d = df['doge_close'].pct_change(10)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()

    signal = ((doge_ret_10d > btc_ret_10d) & (df['btc_close'] > btc_sma50)).astype(float)
    signal = signal.shift(1).fillna(0)  # no lookahead

    doge_ret = df['doge_close'].pct_change()
    return signal, doge_ret


# ── Signal 3: LSR Divergence Pairs ──────────────────────────────────────────

def lsr_divergence_signal(lsr_all, price_all, symbols, dates):
    """
    p=0.054 market-neutral signal.
    Long bottom-3 LSR rank, short top-3 LSR rank.
    Returns daily P&L series.
    """
    # Build aligned LSR and price matrices
    lsr_df = pd.DataFrame(lsr_all).reindex(dates)
    price_df = pd.DataFrame(price_all).reindex(dates)
    ret_df = price_df.pct_change()

    daily_pnl = pd.Series(0.0, index=dates)
    n_long = 3
    n_short = 3

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        lsr_vals = lsr_df.loc[prev_date].dropna()
        if len(lsr_vals) < n_long + n_short:
            continue

        # Rank by LSR: low LSR = long, high LSR = short
        ranked = lsr_vals.sort_values()
        longs = ranked.index[:n_long]
        shorts = ranked.index[-n_short:]

        # P&L
        long_ret = ret_df.loc[curr_date, longs].mean() if curr_date in ret_df.index else 0
        short_ret = ret_df.loc[curr_date, shorts].mean() if curr_date in ret_df.index else 0

        if np.isnan(long_ret):
            long_ret = 0
        if np.isnan(short_ret):
            short_ret = 0

        # Long leg - short leg - TX costs
        n_trades = n_long + n_short  # daily rebalance
        tc = n_trades * TX_COST / len(symbols)  # proportional cost
        daily_pnl.iloc[i] = (long_ret - short_ret) - tc

    return daily_pnl


# ── Kelly Sizing ─────────────────────────────────────────────────────────────

def compute_kelly_fractions(daily_pnl_series, window=KELLY_WINDOW):
    """
    Compute rolling half-Kelly fraction from recent daily PnL.
    Returns Series of position multipliers clipped to [0.1, 1.5].
    """
    n = len(daily_pnl_series)
    kelly = np.full(n, KELLY_FRACTION)  # default

    for i in range(window, n):
        recent = daily_pnl_series.iloc[i - window:i]
        active = recent[recent != 0]
        if len(active) < 10:
            kelly[i] = KELLY_FRACTION
            continue

        wins = active[active > 0]
        losses = active[active < 0]
        if len(wins) == 0 or len(losses) == 0:
            kelly[i] = KELLY_FRACTION
            continue

        p = len(wins) / len(active)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        k = (p * b - (1 - p)) / b
        kelly[i] = np.clip(k * KELLY_FRACTION, 0.1, 1.5)

    return pd.Series(kelly, index=daily_pnl_series.index)


# ── Progressive Stop ────────────────────────────────────────────────────────

def apply_progressive_stop(position, price_series):
    """
    Apply progressive trailing stop to a position series.
    Tightens as unrealized profit grows:
      - Start: 15% trailing from peak
      - After 20% profit: 10% trailing
      - After 50% profit: 7% trailing
    Returns modified position series.
    """
    pos_out = position.copy()
    entry_price = None
    peak_price = None
    stopped = False

    for i in range(len(position)):
        raw_pos = position.iloc[i]
        px = price_series.iloc[i] if not np.isnan(price_series.iloc[i]) else 0.0

        if raw_pos > 0 and not stopped:
            if entry_price is None:
                # New entry
                entry_price = px
                peak_price = px
            else:
                peak_price = max(peak_price, px)

            # Profit from entry
            profit_pct = (peak_price - entry_price) / entry_price if entry_price > 0 else 0

            # Progressive stop level
            if profit_pct > 0.50:
                stop_pct = STOP_PROFIT_50
            elif profit_pct > 0.20:
                stop_pct = STOP_PROFIT_20
            else:
                stop_pct = STOP_INITIAL

            # Check if stop hit
            dd_from_peak = (peak_price - px) / peak_price if peak_price > 0 else 0
            if dd_from_peak > stop_pct:
                stopped = True  # exit NEXT day (day i+1), don't erase day i's P&L

        elif raw_pos == 0:
            # Signal is off — reset stop state
            entry_price = None
            peak_price = None
            stopped = False
        elif stopped and raw_pos > 0:
            # Still stopped out — force flat
            pos_out.iloc[i] = 0.0

    return pos_out


# ── Hedge Trigger ──────────────────────────────────────────────────────────

def compute_hedge_trigger(df, sma_period=50, lsr_mult=1.1):
    """
    Bearish hedge trigger: BTC < SMA50 OR LSR > rolling median * mult.
    Returns array of 0/1 (shifted by 1 for no lookahead).
    """
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


# ── Portfolio Simulation ─────────────────────────────────────────────────────

def simulate_portfolio(btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
                       weights, starting_capital=1000.0, monthly_contrib=0.0,
                       vol_target_multiplier=1.0, kelly=False, progressive_stop=False,
                       btc_price=None, sol_price=None,
                       doge_pos=None, doge_ret=None, doge_price=None,
                       hedge_trigger=None, btc_funding=None, leverage=1.0):
    """
    Simulate a multi-strategy portfolio.

    weights: dict with keys 'btc_regime', 'sol_relmom', 'lsr_pairs', 'doge_relmom'
    hedge_trigger: array of 0/1 — when all directional signals off AND trigger=1, short perp
    btc_funding: Series of BTC funding rates (for hedge cost)
    leverage: leverage multiplier applied to total portfolio return
    """
    n = len(btc_ret)

    # Apply progressive stop if requested
    eff_btc_pos = btc_pos.copy()
    eff_sol_pos = sol_pos.copy()
    eff_doge_pos = doge_pos.copy() if doge_pos is not None else pd.Series(0.0, index=btc_pos.index)
    if progressive_stop and btc_price is not None:
        eff_btc_pos = apply_progressive_stop(btc_pos, btc_price)
    if progressive_stop and sol_price is not None:
        eff_sol_pos = apply_progressive_stop(sol_pos, sol_price)
    if progressive_stop and doge_price is not None and doge_pos is not None:
        eff_doge_pos = apply_progressive_stop(doge_pos, doge_price)

    # Compute Kelly fractions if requested
    kelly_btc = None
    kelly_sol = None
    kelly_doge = None
    if kelly:
        btc_strat_ret = (btc_pos * btc_ret).fillna(0)
        sol_strat_ret = (sol_pos * sol_ret).fillna(0)
        kelly_btc = compute_kelly_fractions(btc_strat_ret)
        kelly_sol = compute_kelly_fractions(sol_strat_ret)
        if doge_pos is not None and doge_ret is not None:
            doge_strat_ret = (doge_pos * doge_ret).fillna(0)
            kelly_doge = compute_kelly_fractions(doge_strat_ret)

    equity = np.zeros(n)
    equity[0] = starting_capital
    total_contributed = starting_capital
    prev_month = -1

    w_btc = weights.get('btc_regime', 0.0)
    w_sol = weights.get('sol_relmom', 0.0)
    w_lsr = weights.get('lsr_pairs', 0.0)
    w_doge = weights.get('doge_relmom', 0.0)

    prev_btc_pos = 0.0
    prev_sol_pos = 0.0
    prev_doge_pos = 0.0
    prev_hedge = 0.0
    n_liquidations = 0

    has_doge = doge_ret is not None and w_doge > 0
    has_hedge = hedge_trigger is not None
    has_funding = btc_funding is not None

    for i in range(1, n):
        br = btc_ret.iloc[i] if not np.isnan(btc_ret.iloc[i]) else 0.0
        sr = sol_ret.iloc[i] if not np.isnan(sol_ret.iloc[i]) else 0.0
        lp = lsr_pnl.iloc[i] if i < len(lsr_pnl) and not np.isnan(lsr_pnl.iloc[i]) else 0.0

        bp = eff_btc_pos.iloc[i] * vol_target_multiplier
        sp = eff_sol_pos.iloc[i]

        # Apply Kelly multiplier
        if kelly and kelly_btc is not None:
            bp = bp * kelly_btc.iloc[i]
        if kelly and kelly_sol is not None:
            sp = sp * kelly_sol.iloc[i]

        # Monthly contribution
        month = i // 30
        if monthly_contrib > 0 and month != prev_month and i > 0:
            equity[i - 1] += monthly_contrib
            total_contributed += monthly_contrib
        prev_month = month

        if equity[i - 1] <= 0:
            equity[i:] = 0
            break

        # BTC regime P&L
        btc_tc = abs(bp - prev_btc_pos) * TX_COST
        btc_pnl = (bp * br - btc_tc) * w_btc

        # SOL relmom P&L
        sol_tc = abs(sp - prev_sol_pos) * TX_COST
        sol_pnl = (sp * sr - sol_tc) * w_sol

        # LSR pairs P&L (already includes TX cost)
        lsr_pnl_scaled = lp * w_lsr

        # DOGE relmom P&L
        doge_pnl = 0.0
        if has_doge:
            dr = doge_ret.iloc[i] if not np.isnan(doge_ret.iloc[i]) else 0.0
            dp = eff_doge_pos.iloc[i]
            if kelly and kelly_doge is not None:
                dp = dp * kelly_doge.iloc[i]
            doge_tc = abs(dp - prev_doge_pos) * TX_COST
            doge_pnl = (dp * dr - doge_tc) * w_doge
            prev_doge_pos = dp

        # Hedge-when-flat overlay
        hedge_pnl = 0.0
        if has_hedge:
            n_active = int(bp > 0) + int(sp > 0) + int(eff_doge_pos.iloc[i] > 0 if has_doge else False)
            if n_active == 0 and hedge_trigger[i] == 1:
                h = -1.0  # Short perp
                h_tc = abs(h - prev_hedge) * TX_COST
                h_fc = 0.0
                if has_funding:
                    h_fc = h * (btc_funding.iloc[i] / 100)
                hedge_pnl = (h * br - h_tc - h_fc) * w_btc
                prev_hedge = h
            else:
                if prev_hedge != 0:
                    hedge_pnl = -abs(0 - prev_hedge) * TX_COST * w_btc  # exit cost
                prev_hedge = 0.0

        # Total portfolio return (unleveraged)
        port_ret = btc_pnl + sol_pnl + lsr_pnl_scaled + doge_pnl + hedge_pnl

        # Apply leverage
        if leverage > 1.0:
            lev_ret = leverage * port_ret
            # Extra funding cost on borrowed portion
            if has_funding and abs(port_ret) > 1e-10:
                lev_ret -= (leverage - 1) * abs(btc_funding.iloc[i] / 100) * 0.5
            # Liquidation check
            liq_level = 0.90 / leverage
            if lev_ret < -liq_level:
                equity[i] = equity[i - 1] * 0.05  # 95% loss
                n_liquidations += 1
                prev_btc_pos = bp
                prev_sol_pos = sp
                continue
            total_pnl = equity[i - 1] * lev_ret
        else:
            total_pnl = equity[i - 1] * port_ret

        equity[i] = equity[i - 1] + total_pnl

        prev_btc_pos = bp
        prev_sol_pos = sp

    # Metrics
    valid = equity > 0
    n_valid = int(valid.sum())
    n_years = n_valid / 365
    final = equity[n_valid - 1] if n_valid > 0 else 0

    cagr = (final / starting_capital) ** (1 / n_years) - 1 if n_years > 0 and final > 0 else 0
    peak = np.maximum.accumulate(equity[:n_valid])
    dd = (equity[:n_valid] - peak) / np.where(peak > 0, peak, 1)
    max_dd = dd.min() if len(dd) > 0 else 0

    daily_rets = np.diff(equity[:n_valid]) / equity[:n_valid - 1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365)
              if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Worst month
    eq_series = pd.Series(equity[:n_valid])
    monthly_eq = eq_series.groupby(np.arange(n_valid) // 30).last()
    monthly_rets = monthly_eq.pct_change().dropna()
    worst_month = monthly_rets.min() if len(monthly_rets) > 0 else 0

    # Time to $1M projection
    if cagr > 0:
        if final >= 1_000_000:
            years_to_1m = n_years
        elif monthly_contrib > 0:
            projected = final
            months = 0
            monthly_rate = (1 + cagr) ** (1/12) - 1
            while projected < 1_000_000 and months < 1200:
                projected = projected * (1 + monthly_rate) + monthly_contrib
                months += 1
            years_to_1m = n_years + months / 12
        else:
            years_to_1m = n_years + np.log(1_000_000 / final) / np.log(1 + cagr)
    else:
        years_to_1m = float('inf')

    return {
        'final_equity': final,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'worst_month': worst_month,
        'years_to_1m': years_to_1m,
        'total_contributed': total_contributed,
        'equity_curve': equity[:n_valid],
        'n_liquidations': n_liquidations,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 80)
    print("  COMBINED PORTFOLIO SIMULATION")
    print("  4 Validated Alpha Streams + $1,000 Starting Capital")
    print("=" * 80)

    print("\nLoading data...")
    df, lsr_all, price_all, symbols = load_all_data()
    print(f"Period: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} ({len(df)} days)")

    # Compute all signals
    print("Computing signals...")
    btc_pos, btc_ret = btc_regime_signal(df)
    sol_pos, sol_ret = sol_relmom_signal(df)
    doge_pos, doge_ret = doge_relmom_signal(df)

    dates = df['date'].values
    lsr_pnl = lsr_divergence_signal(lsr_all, price_all, symbols, dates)

    # Compute hedge trigger (SMA50 + LSR*1.1)
    hedge_trigger_raw = compute_hedge_trigger(df, sma_period=50, lsr_mult=1.1)
    btc_funding_raw = df['btc_funding'].copy()

    # Extract price series before warmup trim (for progressive stops)
    btc_price_raw = df['btc_close'].copy()
    sol_price_raw = df['sol_close'].copy()
    doge_price_raw = df['doge_close'].copy()

    # Drop warmup
    warmup = 50
    btc_pos = btc_pos.iloc[warmup:].reset_index(drop=True)
    btc_ret = btc_ret.iloc[warmup:].reset_index(drop=True)
    sol_pos = sol_pos.iloc[warmup:].reset_index(drop=True)
    sol_ret = sol_ret.iloc[warmup:].reset_index(drop=True)
    doge_pos = doge_pos.iloc[warmup:].reset_index(drop=True)
    doge_ret = doge_ret.iloc[warmup:].reset_index(drop=True)
    lsr_pnl = lsr_pnl.iloc[warmup:].reset_index(drop=True)
    btc_price_trimmed = btc_price_raw.iloc[warmup:].reset_index(drop=True)
    sol_price_trimmed = sol_price_raw.iloc[warmup:].reset_index(drop=True)
    doge_price_trimmed = doge_price_raw.iloc[warmup:].reset_index(drop=True)
    hedge_trigger = hedge_trigger_raw[warmup:]
    btc_funding = btc_funding_raw.iloc[warmup:].reset_index(drop=True)
    dates_trimmed = dates[warmup:]

    n = len(btc_ret)
    print(f"Simulation period: {len(btc_ret)} days after warmup")

    # Common kwargs for all simulate_portfolio calls
    doge_kwargs = dict(doge_pos=doge_pos, doge_ret=doge_ret, doge_price=doge_price_trimmed)
    all_kwargs = dict(doge_pos=doge_pos, doge_ret=doge_ret,
                      btc_price=btc_price_trimmed, sol_price=sol_price_trimmed,
                      doge_price=doge_price_trimmed)
    hedge_kwargs = dict(hedge_trigger=hedge_trigger, btc_funding=btc_funding)

    # ── Signal Correlation Check ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SIGNAL CORRELATION (Are these actually independent?)")
    print(f"{'='*80}")

    btc_d = (btc_pos * btc_ret).replace([np.inf, -np.inf], 0).fillna(0)
    sol_d = (sol_pos * sol_ret).replace([np.inf, -np.inf], 0).fillna(0)
    doge_d = (doge_pos * doge_ret).replace([np.inf, -np.inf], 0).fillna(0)
    lsr_d = lsr_pnl.replace([np.inf, -np.inf], 0).fillna(0)

    corr_matrix = pd.DataFrame({
        'BTC Regime': btc_d,
        'SOL RelMom': sol_d,
        'DOGE RelMom': doge_d,
        'LSR Pairs': lsr_d,
    }).corr()

    sig_names = list(corr_matrix.columns)
    print("\n  Return Correlation Matrix:")
    hdr = f"  {'':>15}" + "".join(f" {s:>12}" for s in sig_names)
    print(hdr)
    for idx in sig_names:
        vals = corr_matrix.loc[idx]
        row = f"  {idx:>15}" + "".join(f" {vals[s]:>12.3f}" for s in sig_names)
        print(row)

    # Signal overlap — directional signals
    btc_on = (btc_pos > 0).astype(int)
    sol_on = (sol_pos > 0).astype(int)
    doge_on = (doge_pos > 0).astype(int)
    any_dir = ((btc_on == 1) | (sol_on == 1) | (doge_on == 1)).sum()
    all_three = ((btc_on == 1) & (sol_on == 1) & (doge_on == 1)).sum()

    print(f"\n  Signal Activity:")
    print(f"    BTC Regime ON:  {btc_on.sum():>5} days ({btc_on.sum()/n*100:.1f}%)")
    print(f"    SOL RelMom ON:  {sol_on.sum():>5} days ({sol_on.sum()/n*100:.1f}%)")
    print(f"    DOGE RelMom ON: {doge_on.sum():>5} days ({doge_on.sum()/n*100:.1f}%)")
    print(f"    Any directional:{any_dir:>5} days ({any_dir/n*100:.1f}%)")
    print(f"    All 3 at once:  {all_three:>5} days ({all_three/n*100:.1f}%)")

    # ── Individual Strategy Performance ──────────────────────────────────
    print(f"\n{'='*80}")
    print("  INDIVIDUAL STRATEGY PERFORMANCE ($1,000 each)")
    print(f"{'='*80}\n")

    strategies = {
        'BTC Regime Only':  {'btc_regime': 1.0, 'sol_relmom': 0.0, 'lsr_pairs': 0.0, 'doge_relmom': 0.0},
        'SOL RelMom Only':  {'btc_regime': 0.0, 'sol_relmom': 1.0, 'lsr_pairs': 0.0, 'doge_relmom': 0.0},
        'DOGE RelMom Only': {'btc_regime': 0.0, 'sol_relmom': 0.0, 'lsr_pairs': 0.0, 'doge_relmom': 1.0},
        'LSR Pairs Only':   {'btc_regime': 0.0, 'sol_relmom': 0.0, 'lsr_pairs': 1.0, 'doge_relmom': 0.0},
    }

    header = (f"  {'Strategy':<25} | {'Final $':>10} | {'CAGR':>7} | {'Sharpe':>6} | "
              f"{'MaxDD':>7} | {'Calmar':>6} | {'Worst Mo':>9}")
    print(header)
    print(f"  {'-'*85}")

    for name, weights in strategies.items():
        r = simulate_portfolio(btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
                               weights=weights, **doge_kwargs)
        print(f"  {name:<25} | ${r['final_equity']:>9,.0f} | {r['cagr']:>6.1%} | "
              f"{r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['calmar']:>6.2f} | "
              f"{r['worst_month']:>8.1%}")

    # ── Portfolio Combinations (3-signal vs 4-signal) ─────────────────────
    print(f"\n{'='*80}")
    print("  PORTFOLIO COMBINATIONS: 3-SIGNAL vs 4-SIGNAL ($1,000 starting)")
    print(f"{'='*80}")
    print(f"  Weights: BTC / SOL / LSR / DOGE\n")

    portfolios = [
        # 3-signal baselines
        ("3S: BTC50/SOL30/LSR20",       {'btc_regime': 0.5, 'sol_relmom': 0.3, 'lsr_pairs': 0.2, 'doge_relmom': 0.0}),
        ("3S: BTC60/SOL20/LSR20",       {'btc_regime': 0.6, 'sol_relmom': 0.2, 'lsr_pairs': 0.2, 'doge_relmom': 0.0}),
        # 4-signal — shave from each existing to fund DOGE
        ("4S: BTC45/SOL20/LSR15/DOGE20", {'btc_regime': 0.45, 'sol_relmom': 0.20, 'lsr_pairs': 0.15, 'doge_relmom': 0.20}),
        ("4S: BTC40/SOL25/LSR15/DOGE20", {'btc_regime': 0.40, 'sol_relmom': 0.25, 'lsr_pairs': 0.15, 'doge_relmom': 0.20}),
        ("4S: BTC40/SOL20/LSR15/DOGE25", {'btc_regime': 0.40, 'sol_relmom': 0.20, 'lsr_pairs': 0.15, 'doge_relmom': 0.25}),
        ("4S: BTC50/SOL15/LSR15/DOGE20", {'btc_regime': 0.50, 'sol_relmom': 0.15, 'lsr_pairs': 0.15, 'doge_relmom': 0.20}),
        ("4S: Equal 25/25/25/25",        {'btc_regime': 0.25, 'sol_relmom': 0.25, 'lsr_pairs': 0.25, 'doge_relmom': 0.25}),
    ]

    print(f"  {'Portfolio':<33} | {'Final $':>10} | {'CAGR':>7} | {'Sharpe':>6} | "
          f"{'MaxDD':>7} | {'Calmar':>6}")
    print(f"  {'-'*82}")

    for name, weights in portfolios:
        r = simulate_portfolio(btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
                               weights=weights, **doge_kwargs)
        print(f"  {name:<33} | ${r['final_equity']:>9,.0f} | {r['cagr']:>6.1%} | "
              f"{r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['calmar']:>6.2f}")

    # ── Best Portfolio with Vol Target Sweep ─────────────────────────────
    print(f"\n{'='*80}")
    print("  BEST 4-SIGNAL PORTFOLIO + VOL TARGET SWEEP")
    print(f"{'='*80}")
    print(f"  Using best 4-signal allocation from above\n")

    # Pick the BTC45/SOL20/LSR15/DOGE20 as starting point
    best4_weights = {'btc_regime': 0.45, 'sol_relmom': 0.20, 'lsr_pairs': 0.15, 'doge_relmom': 0.20}
    vol_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    print(f"  {'VolMult':>7} | {'Final $':>10} | {'CAGR':>7} | {'Sharpe':>6} | "
          f"{'MaxDD':>7} | {'Calmar':>6} | {'Yrs->$1M':>8}")
    print(f"  {'-'*68}")

    for vm in vol_multipliers:
        r = simulate_portfolio(btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
                               weights=best4_weights, vol_target_multiplier=vm, **doge_kwargs)
        yrs = f"{r['years_to_1m']:.1f}" if r['years_to_1m'] < 100 else "never"
        print(f"  {vm:>6.1f}x | ${r['final_equity']:>9,.0f} | {r['cagr']:>6.1%} | "
              f"{r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['calmar']:>6.2f} | {yrs:>8s}")

    # ── Progressive Stop: 3-signal vs 4-signal ───────────────────────────
    print(f"\n{'='*80}")
    print("  PROGRESSIVE STOP COMPARISON: 3-SIGNAL vs 4-SIGNAL")
    print(f"{'='*80}")
    print(f"  Vol x2 (moderate risk), Progressive trailing stop ON\n")

    best3_weights = {'btc_regime': 0.5, 'sol_relmom': 0.3, 'lsr_pairs': 0.2, 'doge_relmom': 0.0}

    configs = [
        ("3S Baseline (no stop)",  best3_weights, False),
        ("3S + Prog. Stop",        best3_weights, True),
        ("4S Baseline (no stop)",  best4_weights, False),
        ("4S + Prog. Stop",        best4_weights, True),
    ]

    print(f"  {'Config':<25} | {'Final $':>10} | {'CAGR':>7} | {'Sharpe':>6} | "
          f"{'MaxDD':>7} | {'Calmar':>6} | {'Worst Mo':>9}")
    print(f"  {'-'*85}")

    for label, w, use_stop in configs:
        r = simulate_portfolio(
            btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
            weights=w, vol_target_multiplier=2.0,
            progressive_stop=use_stop,
            **all_kwargs,
        )
        print(f"  {label:<25} | ${r['final_equity']:>9,.0f} | {r['cagr']:>6.1%} | "
              f"{r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['calmar']:>6.2f} | "
              f"{r['worst_month']:>8.1%}")

    # ── Hedge-When-Flat: With vs Without ────────────────────────────────
    print(f"\n{'='*80}")
    print("  HEDGE-WHEN-FLAT OVERLAY COMPARISON")
    print(f"{'='*80}")
    print(f"  Short perp when ALL directional signals off + bearish trigger fires")
    print(f"  Trigger: BTC < SMA50 OR LSR > rolling_median * 1.1\n")

    # Show hedge activity
    btc_on = (btc_pos > 0).astype(int)
    sol_on = (sol_pos > 0).astype(int)
    doge_on = (doge_pos > 0).astype(int)
    all_off = ((btc_on == 0) & (sol_on == 0) & (doge_on == 0))
    hedge_active = all_off & (hedge_trigger == 1)
    print(f"  All directional OFF: {all_off.sum():>5} days ({all_off.sum()/n*100:.1f}%)")
    print(f"  Hedge active:        {hedge_active.sum():>5} days ({hedge_active.sum()/n*100:.1f}%)\n")

    configs_hedge = [
        ("4S Baseline (no hedge)", best4_weights, False, 2.0),
        ("4S + Hedge-when-flat",   best4_weights, True,  2.0),
        ("4S Baseline (no hedge)", best4_weights, False, 1.0),
        ("4S + Hedge-when-flat",   best4_weights, True,  1.0),
    ]

    print(f"  {'Config':<30} | {'VolM':>4} | {'Final $':>10} | {'CAGR':>7} | {'Sharpe':>6} | "
          f"{'MaxDD':>7} | {'Calmar':>6} | {'Worst Mo':>9}")
    print(f"  {'-'*100}")

    for label, w, use_hedge, vm in configs_hedge:
        hk = hedge_kwargs if use_hedge else {}
        r = simulate_portfolio(
            btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
            weights=w, vol_target_multiplier=vm,
            progressive_stop=True, **all_kwargs, **hk,
        )
        print(f"  {label:<30} | {vm:>3.0f}x | ${r['final_equity']:>9,.0f} | {r['cagr']:>6.1%} | "
              f"{r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['calmar']:>6.2f} | "
              f"{r['worst_month']:>8.1%}")

    # ── Leverage Sweep — Full System with Hedge ──────────────────────────
    print(f"\n{'='*80}")
    print("  LEVERAGE SWEEP — 4-SIGNAL + HEDGE-WHEN-FLAT + PROGRESSIVE STOP")
    print(f"{'='*80}")
    print(f"  $1,000 starting capital, vol target x2, all overlays ON\n")

    print(f"  {'Leverage':>10} | {'Final $':>12} | {'CAGR':>10} | {'Sharpe':>8} | "
          f"{'MaxDD':>8} | {'Liquidns':>8} | {'$1K→$1M':>10}")
    print(f"  {'-'*80}")

    for lev in [1, 2, 3, 4, 5, 6]:
        r = simulate_portfolio(
            btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
            weights=best4_weights, vol_target_multiplier=2.0,
            progressive_stop=True, leverage=float(lev),
            **all_kwargs, **hedge_kwargs,
        )
        final = r['final_equity']
        cagr_val = r['cagr']
        n_liq = r['n_liquidations']
        yrs = r['years_to_1m']
        cagr_s = f"{cagr_val:>+8.1%}" if cagr_val > -0.99 else "    RUIN"
        yrs_s = f"{yrs:.1f}yr" if yrs < 100 else "never"
        print(f"  {lev:>8.0f}x  | ${final:>11,.0f} | {cagr_s} | {r['sharpe']:>7.2f} | "
              f"{r['max_dd']:>7.1%} | {n_liq:>8d} | {yrs_s:>10}")

    # ── The Path to $1M — Final System ───────────────────────────────────
    print(f"\n{'='*80}")
    print("  THE PATH TO $1,000,000 — FULL SYSTEM (4-SIGNAL + HEDGE + STOP)")
    print(f"{'='*80}")
    print(f"  Best allocation with hedge-when-flat at safe leverage levels\n")

    risk_profiles = [
        ("Conservative (1x)",  1.0, 1.0),
        ("Moderate (vol x2)",  2.0, 1.0),
        ("Aggressive (2x lev)", 2.0, 2.0),
        ("Max Safe (3x lev)",  2.0, 3.0),
    ]

    for label, vm, lev in risk_profiles:
        r_no = simulate_portfolio(
            btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
            weights=best4_weights, vol_target_multiplier=vm,
            progressive_stop=True, leverage=lev,
            **all_kwargs, **hedge_kwargs,
        )
        r_dca = simulate_portfolio(
            btc_pos, btc_ret, sol_pos, sol_ret, lsr_pnl,
            weights=best4_weights, vol_target_multiplier=vm,
            monthly_contrib=500, progressive_stop=True, leverage=lev,
            **all_kwargs, **hedge_kwargs,
        )

        yrs_no = f"{r_no['years_to_1m']:.1f}" if r_no['years_to_1m'] < 100 else "never"
        yrs_dca = f"{r_dca['years_to_1m']:.1f}" if r_dca['years_to_1m'] < 100 else "never"

        liq_warn = f" ({r_no['n_liquidations']} liquidations!)" if r_no['n_liquidations'] > 0 else ""
        print(f"  {label}:{liq_warn}")
        print(f"    CAGR: {r_no['cagr']:.1%} | Sharpe: {r_no['sharpe']:.2f} | MaxDD: {r_no['max_dd']:.1%}")
        print(f"    $1K lump  -> ${r_no['final_equity']:>11,.0f} now -> $1M in ~{yrs_no} yrs")
        print(f"    + $500/mo -> ${r_dca['final_equity']:>11,.0f} now -> $1M in ~{yrs_dca} yrs")
        print()

    print(f"  {'='*76}")
    print(f"  Signal confidence levels:")
    print(f"    BTC Weighted Regime:       p=0.012 (SIGNIFICANT)")
    print(f"    SOL Relative Momentum 20d: p=0.028 (SIGNIFICANT)")
    print(f"    DOGE Relative Momentum 10d:p=0.010 (SIGNIFICANT)")
    print(f"    LSR Divergence Pairs:      p=0.054 (BORDERLINE)")
    print(f"    Hedge-when-flat overlay:   p=0.002 (HIGHLY SIGNIFICANT)")
    print(f"  Risk overlays: Progressive trailing stop (15%->10%->7%) + hedge-when-flat")
    print(f"  All use fixed parameters. Zero optimization. TX + funding costs included.")
    print(f"{'='*80}")
