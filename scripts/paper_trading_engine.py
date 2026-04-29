#!/usr/bin/env python3
"""
Paper Trading Engine — Forward test of the 4-Signal + Hedge system.

Runs daily (via cron), computes current signal state for all components,
logs theoretical positions and tracks P&L without real capital.

Usage:
  python paper_trading_engine.py              # compute today's signals
  python paper_trading_engine.py --report     # print P&L report
  python paper_trading_engine.py --backfill N # backfill last N days

Cron (run daily at 00:05 UTC):
  5 0 * * * cd ~/Desktop/maestro && python scripts/paper_trading_engine.py >> data/live/paper_trading.log 2>&1
"""
import duckdb, os, json, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
LOG_DIR = os.path.expanduser("~/Desktop/maestro/data/live")
LOG_FILE = os.path.join(LOG_DIR, "paper_trades.jsonl")
TX_COST = 0.001

WEIGHTS = {'btc': 0.45, 'sol': 0.20, 'lsr': 0.15, 'doge': 0.20}
VOL_MULT = 2.0
STOP_INITIAL = 0.15
STOP_PROFIT_20 = 0.10
STOP_PROFIT_50 = 0.07

ALL_SYMBOLS = ['BTC', 'ETH', 'SOL', 'LINK', 'AVAX', 'DOGE', 'DOT', 'NEAR',
               'ATOM', 'BNB', 'XRP', 'UNI', 'ADA', 'FIL']


# ── Data Loading ────────────────────────────────────────────────────────

def load_latest_data(lookback=60):
    """Load recent data for signal computation (need ~60 days for indicators)."""
    con = duckdb.connect(DB_PATH, read_only=True)

    btc = con.execute(
        f"SELECT date, close as btc_close FROM perps_daily "
        f"WHERE symbol='BTC' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    funding = con.execute(
        f"SELECT date, close as btc_funding FROM cg_funding_rate "
        f"WHERE symbol='BTC' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    lsr = con.execute(
        f"SELECT date, global_account_long_short_ratio as btc_lsr "
        f"FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    liq = con.execute(
        f"SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as btc_total_liq "
        f"FROM cg_liquidations WHERE symbol='BTC' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    taker = con.execute(
        f"SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as btc_taker_ratio "
        f"FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    sol = con.execute(
        f"SELECT date, close as sol_close FROM perps_daily "
        f"WHERE symbol='SOL' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    sol_lsr = con.execute(
        f"SELECT date, global_account_long_short_ratio as sol_lsr "
        f"FROM cg_lsr_global WHERE symbol='SOL' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    doge = con.execute(
        f"SELECT date, close as doge_close FROM perps_daily "
        f"WHERE symbol='DOGE' ORDER BY date DESC LIMIT {lookback}"
    ).df().sort_values('date').reset_index(drop=True)

    # LSR + prices for all symbols (for LSR pairs)
    lsr_all, price_all = {}, {}
    for sym in ALL_SYMBOLS:
        lsr_df = con.execute(
            f"SELECT date, global_account_long_short_ratio as lsr "
            f"FROM cg_lsr_global WHERE symbol='{sym}' ORDER BY date DESC LIMIT {lookback}"
        ).df().sort_values('date')
        px_df = con.execute(
            f"SELECT date, close FROM perps_daily WHERE symbol='{sym}' ORDER BY date DESC LIMIT {lookback}"
        ).df().sort_values('date')
        lsr_all[sym] = lsr_df.set_index('date')['lsr']
        price_all[sym] = px_df.set_index('date')['close']

    con.close()

    df = btc.copy()
    for src in [funding, lsr, liq, taker, sol, sol_lsr, doge]:
        df = df.merge(src, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df, lsr_all, price_all


# ── Signal Computation (current state only) ─────────────────────────────

def compute_btc_regime(df):
    """BTC Weighted Regime — returns position size for today."""
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

    # Use yesterday's signal for today (shift(1))
    idx = len(df) - 1
    pos_val = float(position.iloc[idx - 1]) if idx > 0 else 0.0

    # Component breakdown
    components = {
        'lsr_signal': float(lsr_s.iloc[idx]),
        'funding_signal': float(fund_s.iloc[idx]),
        'liquidation_signal': float(liq_s.iloc[idx]),
        'taker_signal': float(taker_s.iloc[idx]),
        'raw_score': float(score.iloc[idx]),
        'vol_scalar': float(vol_sc.iloc[idx]) if not np.isnan(vol_sc.iloc[idx]) else 1.0,
    }
    return pos_val * VOL_MULT, components


def compute_sol_relmom(df):
    """SOL Relative Momentum — binary signal."""
    btc_ret_20d = df['btc_close'].pct_change(20)
    sol_ret_20d = df['sol_close'].pct_change(20)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((sol_ret_20d > btc_ret_20d) & (df['btc_close'] > btc_sma50)).astype(float)
    idx = len(df) - 1
    pos_val = float(signal.iloc[idx - 1]) if idx > 0 else 0.0
    components = {
        'sol_20d_ret': float(sol_ret_20d.iloc[idx]),
        'btc_20d_ret': float(btc_ret_20d.iloc[idx]),
        'btc_above_sma50': int(df['btc_close'].iloc[idx] > btc_sma50.iloc[idx]),
        'outperforming': int(float(sol_ret_20d.iloc[idx]) > float(btc_ret_20d.iloc[idx])),
    }
    return pos_val, components


def compute_doge_relmom(df):
    """DOGE Relative Momentum — binary signal."""
    btc_ret_10d = df['btc_close'].pct_change(10)
    doge_ret_10d = df['doge_close'].pct_change(10)
    btc_sma50 = df['btc_close'].rolling(50, min_periods=30).mean()
    signal = ((doge_ret_10d > btc_ret_10d) & (df['btc_close'] > btc_sma50)).astype(float)
    idx = len(df) - 1
    pos_val = float(signal.iloc[idx - 1]) if idx > 0 else 0.0
    components = {
        'doge_10d_ret': float(doge_ret_10d.iloc[idx]),
        'btc_10d_ret': float(btc_ret_10d.iloc[idx]),
        'btc_above_sma50': int(df['btc_close'].iloc[idx] > btc_sma50.iloc[idx]),
        'outperforming': int(float(doge_ret_10d.iloc[idx]) > float(btc_ret_10d.iloc[idx])),
    }
    return pos_val, components


def compute_lsr_pairs(lsr_all, price_all):
    """LSR Divergence Pairs — current positions."""
    lsr_df = pd.DataFrame(lsr_all)
    latest_date = lsr_df.index[-1]
    lsr_vals = lsr_df.loc[latest_date].dropna()
    if len(lsr_vals) < 6:
        return [], [], {'available_assets': int(len(lsr_vals))}
    ranked = lsr_vals.sort_values()
    longs = list(ranked.index[:3])
    shorts = list(ranked.index[-3:])
    components = {
        'lsr_rankings': {str(k): float(v) for k, v in ranked.items()},
        'n_available': int(len(lsr_vals)),
    }
    return longs, shorts, components


def compute_hedge_state(df, btc_pos, sol_pos, doge_pos):
    """Hedge-when-flat overlay — should we be short?"""
    close = df['btc_close'].values
    sma = pd.Series(close).rolling(50, min_periods=25).mean().values
    lsr_v = df['btc_lsr'].values
    lsr_med = pd.Series(lsr_v).rolling(30, min_periods=10).median().values
    idx = len(df) - 1

    below_sma = bool(close[idx] < sma[idx]) if not np.isnan(sma[idx]) else False
    lsr_elevated = bool(lsr_v[idx] > lsr_med[idx] * 1.1) if not np.isnan(lsr_med[idx]) else False
    bearish = below_sma or lsr_elevated
    all_flat = btc_pos == 0 and sol_pos == 0 and doge_pos == 0
    hedge_active = bearish and all_flat

    components = {
        'btc_price': float(close[idx]),
        'sma50': float(sma[idx]) if not np.isnan(sma[idx]) else None,
        'below_sma50': int(below_sma),
        'btc_lsr': float(lsr_v[idx]),
        'lsr_median': float(lsr_med[idx]) if not np.isnan(lsr_med[idx]) else None,
        'lsr_elevated': int(lsr_elevated),
        'all_signals_flat': int(all_flat),
        'hedge_active': int(hedge_active),
    }
    return hedge_active, components


# ── State Management ────────────────────────────────────────────────────

def load_prev_state():
    """Load previous day's state for P&L tracking."""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def compute_pnl(prev_state, current_prices):
    """Compute P&L since last signal."""
    if prev_state is None:
        return 0.0, {}

    pnl_breakdown = {}
    total_pnl = 0.0

    # BTC regime P&L
    prev_btc_pos = prev_state['positions']['btc_regime']
    btc_ret = (current_prices['btc'] - prev_state['prices']['btc']) / prev_state['prices']['btc']
    btc_pnl = prev_btc_pos * btc_ret * WEIGHTS['btc']
    pnl_breakdown['btc_regime'] = float(btc_pnl)
    total_pnl += btc_pnl

    # SOL relmom P&L
    prev_sol_pos = prev_state['positions']['sol_relmom']
    sol_ret = (current_prices['sol'] - prev_state['prices']['sol']) / prev_state['prices']['sol']
    sol_pnl = prev_sol_pos * sol_ret * WEIGHTS['sol']
    pnl_breakdown['sol_relmom'] = float(sol_pnl)
    total_pnl += sol_pnl

    # DOGE relmom P&L
    prev_doge_pos = prev_state['positions']['doge_relmom']
    doge_ret = (current_prices['doge'] - prev_state['prices']['doge']) / prev_state['prices']['doge']
    doge_pnl = prev_doge_pos * doge_ret * WEIGHTS['doge']
    pnl_breakdown['doge_relmom'] = float(doge_pnl)
    total_pnl += doge_pnl

    # Hedge P&L
    if prev_state['positions'].get('hedge', 0) != 0:
        hedge_pnl = prev_state['positions']['hedge'] * btc_ret * WEIGHTS['btc']
        pnl_breakdown['hedge'] = float(hedge_pnl)
        total_pnl += hedge_pnl

    # TX costs for position changes
    pnl_breakdown['tx_costs'] = 0.0  # computed on next signal generation

    return float(total_pnl), pnl_breakdown


# ── Main Signal Generation ──────────────────────────────────────────────

def generate_signals():
    """Main entry: compute all signals and log."""
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[{datetime.utcnow().isoformat()}] Paper Trading Engine — Signal Generation")
    print("-" * 60)

    # Load data
    df, lsr_all, price_all = load_latest_data(lookback=60)
    latest_date = str(df['date'].iloc[-1])[:10]
    print(f"Latest data: {latest_date}")
    print(f"Data points: {len(df)}")

    # Current prices
    prices = {
        'btc': float(df['btc_close'].iloc[-1]),
        'sol': float(df['sol_close'].iloc[-1]),
        'doge': float(df['doge_close'].iloc[-1]),
    }

    # Compute signals
    btc_pos, btc_comp = compute_btc_regime(df)
    sol_pos, sol_comp = compute_sol_relmom(df)
    doge_pos, doge_comp = compute_doge_relmom(df)
    lsr_longs, lsr_shorts, lsr_comp = compute_lsr_pairs(lsr_all, price_all)
    hedge_active, hedge_comp = compute_hedge_state(df, btc_pos, sol_pos, doge_pos)

    # P&L from previous state
    prev_state = load_prev_state()
    daily_pnl, pnl_breakdown = compute_pnl(prev_state, prices)

    # Cumulative tracking
    cum_pnl = (prev_state.get('cumulative_pnl', 0.0) if prev_state else 0.0) + daily_pnl
    day_count = (prev_state.get('day_count', 0) if prev_state else 0) + 1

    # Build record
    record = {
        'timestamp': datetime.utcnow().isoformat(),
        'date': latest_date,
        'day_count': day_count,
        'prices': prices,
        'positions': {
            'btc_regime': float(btc_pos),
            'sol_relmom': float(sol_pos),
            'doge_relmom': float(doge_pos),
            'lsr_longs': lsr_longs,
            'lsr_shorts': lsr_shorts,
            'hedge': -1.0 if hedge_active else 0.0,
        },
        'components': {
            'btc_regime': btc_comp,
            'sol_relmom': sol_comp,
            'doge_relmom': doge_comp,
            'lsr_pairs': lsr_comp,
            'hedge': hedge_comp,
        },
        'daily_pnl': daily_pnl,
        'pnl_breakdown': pnl_breakdown,
        'cumulative_pnl': float(cum_pnl),
    }

    # Append to JSONL log
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Print summary
    print()
    print(f"  BTC Regime:    pos={btc_pos:.2f}  score={btc_comp['raw_score']:.2f}  vol_sc={btc_comp['vol_scalar']:.2f}")
    print(f"  SOL RelMom:    pos={sol_pos:.0f}    outperf={sol_comp['outperforming']}  sma50={sol_comp['btc_above_sma50']}")
    print(f"  DOGE RelMom:   pos={doge_pos:.0f}    outperf={doge_comp['outperforming']}  sma50={doge_comp['btc_above_sma50']}")
    print(f"  LSR Pairs:     longs={lsr_longs}  shorts={lsr_shorts}")
    print(f"  Hedge:         {'ACTIVE (short BTC)' if hedge_active else 'OFF'}")
    print()
    print(f"  Daily P&L:     {daily_pnl*100:+.3f}%")
    print(f"  Cumulative:    {cum_pnl*100:+.3f}% ({day_count} days)")
    print(f"  BTC: ${prices['btc']:,.0f}  SOL: ${prices['sol']:.2f}  DOGE: ${prices['doge']:.4f}")
    print()
    print(f"  Logged to: {LOG_FILE}")
    return record


def print_report():
    """Print performance report from paper trading log."""
    if not os.path.exists(LOG_FILE):
        print("No paper trading data yet. Run: python paper_trading_engine.py")
        return

    records = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    print("=" * 60)
    print("  PAPER TRADING REPORT")
    print("=" * 60)
    print(f"  Period: {records[0]['date']} to {records[-1]['date']}")
    print(f"  Days tracked: {len(records)}")
    print()

    # P&L summary
    daily_pnls = [r['daily_pnl'] for r in records if r['daily_pnl'] != 0]
    if daily_pnls:
        pnl_arr = np.array(daily_pnls)
        mean_bps = np.mean(pnl_arr) * 10000
        sharpe = np.mean(pnl_arr) / np.std(pnl_arr, ddof=1) * np.sqrt(365) if len(pnl_arr) > 1 else 0
        cum = records[-1]['cumulative_pnl']
        win_rate = np.mean(pnl_arr > 0) * 100

        print(f"  Cumulative P&L:  {cum*100:+.2f}%")
        print(f"  Mean daily:      {mean_bps:+.1f} bps")
        print(f"  Sharpe (ann):    {sharpe:.2f}")
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Best day:        {max(pnl_arr)*100:+.3f}%")
        print(f"  Worst day:       {min(pnl_arr)*100:+.3f}%")
    else:
        print("  No P&L data yet (need 2+ days)")

    print()

    # Signal activity
    n_btc_on = sum(1 for r in records if r['positions']['btc_regime'] > 0)
    n_sol_on = sum(1 for r in records if r['positions']['sol_relmom'] > 0)
    n_doge_on = sum(1 for r in records if r['positions']['doge_relmom'] > 0)
    n_hedge_on = sum(1 for r in records if r['positions']['hedge'] != 0)
    n = len(records)

    print(f"  Signal Activity:")
    print(f"    BTC Regime ON:  {n_btc_on}/{n} ({n_btc_on/n*100:.0f}%)")
    print(f"    SOL RelMom ON:  {n_sol_on}/{n} ({n_sol_on/n*100:.0f}%)")
    print(f"    DOGE RelMom ON: {n_doge_on}/{n} ({n_doge_on/n*100:.0f}%)")
    print(f"    Hedge Active:   {n_hedge_on}/{n} ({n_hedge_on/n*100:.0f}%)")

    # Last 5 days
    print()
    print(f"  Last 5 days:")
    print(f"  {'Date':<12} {'BTC':>6} {'SOL':>5} {'DOGE':>5} {'Hedge':>6} {'P&L':>8}")
    for r in records[-5:]:
        btc_s = f"{r['positions']['btc_regime']:.1f}"
        sol_s = f"{r['positions']['sol_relmom']:.0f}"
        doge_s = f"{r['positions']['doge_relmom']:.0f}"
        hedge_s = "SHORT" if r['positions']['hedge'] != 0 else "-"
        pnl_s = f"{r['daily_pnl']*100:+.3f}%"
        print(f"  {r['date']:<12} {btc_s:>6} {sol_s:>5} {doge_s:>5} {hedge_s:>6} {pnl_s:>8}")

    print("=" * 60)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper Trading Engine')
    parser.add_argument('--report', action='store_true', help='Print P&L report')
    parser.add_argument('--backfill', type=int, default=0, help='Backfill N days')
    args = parser.parse_args()

    if args.report:
        print_report()
    elif args.backfill > 0:
        print(f"Backfilling {args.backfill} days...")
        # Load full data
        df, lsr_all, price_all = load_latest_data(lookback=args.backfill + 60)
        dates = df['date'].values
        n = len(df)
        start = max(50, n - args.backfill)

        for day_idx in range(start, n):
            # Slice up to this day
            df_slice = df.iloc[:day_idx + 1].reset_index(drop=True)
            # Trim lsr_all/price_all to this date range
            latest_date = str(df_slice['date'].iloc[-1])[:10]

            prices = {
                'btc': float(df_slice['btc_close'].iloc[-1]),
                'sol': float(df_slice['sol_close'].iloc[-1]),
                'doge': float(df_slice['doge_close'].iloc[-1]),
            }

            btc_pos, btc_comp = compute_btc_regime(df_slice)
            sol_pos, sol_comp = compute_sol_relmom(df_slice)
            doge_pos, doge_comp = compute_doge_relmom(df_slice)

            lsr_df_slice = {s: v[v.index <= df_slice['date'].iloc[-1]] for s, v in lsr_all.items()}
            price_df_slice = {s: v[v.index <= df_slice['date'].iloc[-1]] for s, v in price_all.items()}
            lsr_longs, lsr_shorts, lsr_comp = compute_lsr_pairs(lsr_df_slice, price_df_slice)
            hedge_active, hedge_comp = compute_hedge_state(df_slice, btc_pos, sol_pos, doge_pos)

            prev_state = load_prev_state()
            daily_pnl, pnl_breakdown = compute_pnl(prev_state, prices)
            cum_pnl = (prev_state.get('cumulative_pnl', 0.0) if prev_state else 0.0) + daily_pnl
            day_count = (prev_state.get('day_count', 0) if prev_state else 0) + 1

            record = {
                'timestamp': datetime.utcnow().isoformat(),
                'date': latest_date,
                'day_count': day_count,
                'prices': prices,
                'positions': {
                    'btc_regime': float(btc_pos),
                    'sol_relmom': float(sol_pos),
                    'doge_relmom': float(doge_pos),
                    'lsr_longs': lsr_longs,
                    'lsr_shorts': lsr_shorts,
                    'hedge': -1.0 if hedge_active else 0.0,
                },
                'components': {
                    'btc_regime': btc_comp,
                    'sol_relmom': sol_comp,
                    'doge_relmom': doge_comp,
                    'lsr_pairs': lsr_comp,
                    'hedge': hedge_comp,
                },
                'daily_pnl': daily_pnl,
                'pnl_breakdown': pnl_breakdown,
                'cumulative_pnl': float(cum_pnl),
            }

            os.makedirs(LOG_DIR, exist_ok=True)
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(record) + '\n')

            sig = "B" if btc_pos > 0 else "."
            sig += "S" if sol_pos > 0 else "."
            sig += "D" if doge_pos > 0 else "."
            sig += "H" if hedge_active else "."
            print(f"  {latest_date}  [{sig}]  pnl={daily_pnl*100:+.3f}%  cum={cum_pnl*100:+.2f}%")

        print(f"\nBackfilled {n - start} days. Run --report for summary.")
    else:
        generate_signals()
