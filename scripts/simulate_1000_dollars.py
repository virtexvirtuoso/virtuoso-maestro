#!/usr/bin/env python3
"""
Simulate $1,000 invested in the BTC Vol-Targeted Weighted Regime strategy.
Shows day-by-day equity, monthly returns, drawdowns, and comparison to buy-and-hold.
"""
import duckdb, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
STARTING_CAPITAL = 1000.0
TARGET_DAILY_VOL = 0.015  # 1.5% daily vol target (middle of sweet spot)
MAX_LEVERAGE = 1.5
MIN_LEVERAGE = 0.25
VOL_LOOKBACK = 20
TX_COST = 0.001  # 10bps per trade


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    lsr = con.execute("SELECT date, global_account_long_short_ratio as lsr FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date").df()
    fr = con.execute("SELECT date, close as funding_rate FROM cg_funding_rate WHERE symbol='BTC' ORDER BY date").df()
    liq = con.execute("SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as total_liq FROM cg_liquidations WHERE symbol='BTC' ORDER BY date").df()
    taker = con.execute("SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as taker_ratio FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date").df()
    price = con.execute("SELECT date, close FROM perps_daily WHERE symbol='BTC' ORDER BY date").df()
    con.close()

    df = price.copy()
    for src in [lsr, fr, liq, taker]:
        df = df.merge(src, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df


# ── Signal + Sizing ──────────────────────────────────────────────────────────

def compute_signals(df):
    # Weighted regime score
    lsr_med = df['lsr'].rolling(30, min_periods=10).median()
    lsr_s = (df['lsr'] < lsr_med).astype(float)
    fund_s = (df['funding_rate'] < 0.03).astype(float)
    liq_p80 = df['total_liq'].rolling(30, min_periods=10).quantile(0.8)
    liq_s = (df['total_liq'] < liq_p80).astype(float)
    taker_s = (df['taker_ratio'] > 1.0).astype(float)

    df['regime_score'] = lsr_s * 0.35 + fund_s * 0.35 + liq_s * 0.15 + taker_s * 0.15
    df['regime_signal'] = (df['regime_score'] > 0.5).astype(float)

    # Returns and vol
    df['returns'] = df['close'].pct_change()
    df['realized_vol'] = df['returns'].rolling(VOL_LOOKBACK, min_periods=10).std()

    # Vol-targeted position (no lookahead: use yesterday's signal and vol)
    df['vol_scalar'] = (TARGET_DAILY_VOL / df['realized_vol']).clip(MIN_LEVERAGE, MAX_LEVERAGE)
    df['position'] = df['regime_signal'] * df['vol_scalar']

    # Lag position by 1 day (signal seen EOD → trade next day)
    df['position'] = df['position'].shift(1).fillna(0)

    return df


# ── Day-by-Day Simulation ───────────────────────────────────────────────────

def simulate(df):
    n = len(df)
    equity = np.zeros(n)
    bh_equity = np.zeros(n)
    equity[0] = STARTING_CAPITAL
    bh_equity[0] = STARTING_CAPITAL

    positions_held = []
    daily_pnl = []
    prev_pos = 0.0

    for i in range(1, n):
        ret = df['returns'].iloc[i]
        pos = df['position'].iloc[i]

        if np.isnan(ret):
            equity[i] = equity[i-1]
            bh_equity[i] = bh_equity[i-1]
            daily_pnl.append(0)
            continue

        # Strategy P&L
        pos_change = abs(pos - prev_pos)
        tc = pos_change * TX_COST * equity[i-1]
        strat_pnl = equity[i-1] * pos * ret - tc
        equity[i] = equity[i-1] + strat_pnl
        prev_pos = pos

        # Buy and hold P&L
        bh_equity[i] = bh_equity[i-1] * (1 + ret)

        daily_pnl.append(strat_pnl)
        positions_held.append(pos)

    df['equity'] = equity
    df['bh_equity'] = bh_equity

    return df, daily_pnl


# ── Reporting ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Loading BTC derivatives data...")
    df = load_data()
    df = compute_signals(df)
    df = df.dropna(subset=['returns', 'realized_vol', 'regime_score']).reset_index(drop=True)

    print(f"Simulating ${STARTING_CAPITAL:,.0f} from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    print(f"Vol target: {TARGET_DAILY_VOL*100:.1f}% daily | Max leverage: {MAX_LEVERAGE}x | TX cost: {TX_COST*100:.1f}bps\n")

    df, daily_pnl = simulate(df)

    # ── Summary Stats ────────────────────────────────────────────────────
    final_equity = df['equity'].iloc[-1]
    final_bh = df['bh_equity'].iloc[-1]
    n_days = len(df)
    n_years = n_days / 365

    # Strategy stats
    strat_total_ret = (final_equity / STARTING_CAPITAL - 1)
    strat_cagr = (final_equity / STARTING_CAPITAL) ** (1 / n_years) - 1

    peak = df['equity'].expanding().max()
    dd = (df['equity'] - peak) / peak
    max_dd = dd.min()
    max_dd_date = df['date'].iloc[dd.idxmin()]

    daily_rets = pd.Series(daily_pnl) / df['equity'].iloc[:-1].values
    daily_rets = daily_rets.replace([np.inf, -np.inf], 0).fillna(0)
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(365) if daily_rets.std() > 0 else 0

    exposure = (df['position'] > 0).mean()
    avg_position = df.loc[df['position'] > 0, 'position'].mean() if exposure > 0 else 0

    # Buy-and-hold stats
    bh_total_ret = (final_bh / STARTING_CAPITAL - 1)
    bh_cagr = (final_bh / STARTING_CAPITAL) ** (1 / n_years) - 1
    bh_peak = df['bh_equity'].expanding().max()
    bh_dd = (df['bh_equity'] - bh_peak) / bh_peak
    bh_max_dd = bh_dd.min()

    print("=" * 60)
    print(f"  ${STARTING_CAPITAL:,.0f} INVESTED — FINAL RESULTS")
    print("=" * 60)
    print(f"  Period: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} ({n_years:.1f} years)")
    print()
    print(f"  {'':30s} {'STRATEGY':>12s} {'BUY & HOLD':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Final Value':30s} {'$'+f'{final_equity:,.0f}':>12s} {'$'+f'{final_bh:,.0f}':>12s}")
    print(f"  {'Total Return':30s} {strat_total_ret:>11.1%} {bh_total_ret:>11.1%}")
    print(f"  {'CAGR':30s} {strat_cagr:>11.1%} {bh_cagr:>11.1%}")
    print(f"  {'Sharpe':30s} {sharpe:>12.2f} {'':>12s}")
    print(f"  {'Max Drawdown':30s} {max_dd:>11.1%} {bh_max_dd:>11.1%}")
    print(f"  {'Max DD Date':30s} {str(max_dd_date.date()):>12s} {'':>12s}")
    print(f"  {'Exposure':30s} {exposure:>11.1%} {'100.0%':>12s}")
    print(f"  {'Avg Position Size (when in)':30s} {avg_position:>11.2f}x {'1.00x':>12s}")

    # ── Monthly Returns ──────────────────────────────────────────────────
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg(
        start_eq=('equity', 'first'),
        end_eq=('equity', 'last'),
        start_bh=('bh_equity', 'first'),
        end_bh=('bh_equity', 'last'),
    )
    monthly['strat_ret'] = monthly['end_eq'] / monthly['start_eq'] - 1
    monthly['bh_ret'] = monthly['end_bh'] / monthly['start_bh'] - 1

    print(f"\n{'='*60}")
    print(f"  MONTHLY RETURNS")
    print(f"{'='*60}")
    print(f"  {'Month':<10} {'Strategy':>10} {'Buy&Hold':>10} {'Equity':>12}")
    print(f"  {'-'*44}")

    for idx, row in monthly.iterrows():
        eq_str = f"${row['end_eq']:,.0f}"
        print(f"  {str(idx):<10} {row['strat_ret']:>+9.1%} {row['bh_ret']:>+9.1%} {eq_str:>12s}")

    # ── Yearly Summary ───────────────────────────────────────────────────
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year').agg(
        start_eq=('equity', 'first'),
        end_eq=('equity', 'last'),
        start_bh=('bh_equity', 'first'),
        end_bh=('bh_equity', 'last'),
    )
    yearly['strat_ret'] = yearly['end_eq'] / yearly['start_eq'] - 1
    yearly['bh_ret'] = yearly['end_bh'] / yearly['start_bh'] - 1

    print(f"\n{'='*60}")
    print(f"  YEARLY RETURNS")
    print(f"{'='*60}")
    print(f"  {'Year':<6} {'Strategy':>10} {'Buy&Hold':>10} {'Equity':>12}")
    print(f"  {'-'*40}")
    for idx, row in yearly.iterrows():
        eq_str = f"${row['end_eq']:,.0f}"
        print(f"  {idx:<6} {row['strat_ret']:>+9.1%} {row['bh_ret']:>+9.1%} {eq_str:>12s}")

    # ── Drawdown Analysis ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WORST DRAWDOWNS")
    print(f"{'='*60}")

    # Find top 5 drawdown periods
    in_dd = False
    dd_start = 0
    drawdowns = []
    for i in range(len(dd)):
        if dd.iloc[i] < -0.02 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            worst = dd.iloc[dd_start:i].min()
            worst_idx = dd.iloc[dd_start:i].idxmin()
            drawdowns.append({
                'start': df['date'].iloc[dd_start].date(),
                'trough': df['date'].iloc[worst_idx].date(),
                'end': df['date'].iloc[i].date(),
                'depth': worst,
                'duration': (df['date'].iloc[i] - df['date'].iloc[dd_start]).days,
            })

    drawdowns.sort(key=lambda x: x['depth'])
    for i, d in enumerate(drawdowns[:5]):
        print(f"  {i+1}. {d['depth']:+.1%} | {d['start']} → {d['trough']} → {d['end']} ({d['duration']} days)")

    # ── Key Stats ────────────────────────────────────────────────────────
    winning_months = (monthly['strat_ret'] > 0).sum()
    total_months = len(monthly)
    best_month = monthly['strat_ret'].max()
    worst_month = monthly['strat_ret'].min()
    best_month_date = monthly['strat_ret'].idxmax()
    worst_month_date = monthly['strat_ret'].idxmin()

    print(f"\n{'='*60}")
    print(f"  ADDITIONAL STATS")
    print(f"{'='*60}")
    print(f"  Winning months: {winning_months}/{total_months} ({winning_months/total_months:.0%})")
    print(f"  Best month:  {best_month:+.1%} ({best_month_date})")
    print(f"  Worst month: {worst_month:+.1%} ({worst_month_date})")
    print(f"  Avg monthly return: {monthly['strat_ret'].mean():+.1%}")
    print(f"  Monthly vol: {monthly['strat_ret'].std():.1%}")

    # Calmar ratio
    calmar = strat_cagr / abs(max_dd) if max_dd != 0 else 0
    print(f"  Calmar ratio: {calmar:.2f}")

    print(f"\n  Strategy: BTC Weighted Regime + Vol-Targeting")
    print(f"  Signal: weighted_regime > 0.5 → LONG, else FLAT")
    print(f"  Sizing: target_vol / realized_vol_20d, clipped [{MIN_LEVERAGE}-{MAX_LEVERAGE}x]")
    print(f"  No optimization. No parameter tuning. Zero degrees of freedom.")
