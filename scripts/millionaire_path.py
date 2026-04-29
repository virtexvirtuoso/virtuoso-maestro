#!/usr/bin/env python3
"""
Maestro Millionaire Path — How does $1,000 become $1,000,000?

Three questions, answered honestly:
  1. What max leverage on the proven regime signal gets you there fastest?
  2. Does adding ETH beta help or hurt?
  3. What if you contribute $500/month?

Uses ONLY the walk-forward validated BTC Weighted Regime signal (p=0.012).
No optimization. No speculation. Just math on proven edge.
"""
import duckdb, os, warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/Desktop/maestro/data/maestro.duckdb")
STARTING_CAPITAL = 1000.0
TARGET_DAILY_VOL = 0.015
MIN_LEVERAGE = 0.25
VOL_LOOKBACK = 20
TX_COST = 0.001  # 10bps per trade


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_btc_data():
    """Load BTC price + derivatives data for regime signal."""
    con = duckdb.connect(DB_PATH, read_only=True)
    lsr = con.execute(
        "SELECT date, global_account_long_short_ratio as lsr "
        "FROM cg_lsr_global WHERE symbol='BTC' ORDER BY date"
    ).df()
    fr = con.execute(
        "SELECT date, close as funding_rate "
        "FROM cg_funding_rate WHERE symbol='BTC' ORDER BY date"
    ).df()
    liq = con.execute(
        "SELECT date, aggregated_long_liquidation_usd + aggregated_short_liquidation_usd as total_liq "
        "FROM cg_liquidations WHERE symbol='BTC' ORDER BY date"
    ).df()
    taker = con.execute(
        "SELECT date, taker_buy_volume_usd / NULLIF(taker_sell_volume_usd, 0) as taker_ratio "
        "FROM cg_taker_volume WHERE symbol='BTC' ORDER BY date"
    ).df()
    price = con.execute(
        "SELECT date, close FROM perps_daily WHERE symbol='BTC' ORDER BY date"
    ).df()
    con.close()

    df = price.copy()
    for src in [lsr, fr, liq, taker]:
        df = df.merge(src, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_eth_prices():
    """Load ETH close prices aligned to BTC dates."""
    con = duckdb.connect(DB_PATH, read_only=True)
    eth = con.execute(
        "SELECT date, close as eth_close FROM perps_daily WHERE symbol='ETH' ORDER BY date"
    ).df()
    con.close()
    return eth


def load_sol_prices():
    """Load SOL close prices aligned to BTC dates."""
    con = duckdb.connect(DB_PATH, read_only=True)
    sol = con.execute(
        "SELECT date, close as sol_close FROM perps_daily WHERE symbol='SOL' ORDER BY date"
    ).df()
    con.close()
    return sol


# ── Signal Computation ───────────────────────────────────────────────────────

def compute_regime_signal(df):
    """Compute BTC weighted regime signal + vol scalar. No lookahead."""
    lsr_med = df["lsr"].rolling(30, min_periods=10).median()
    lsr_s = (df["lsr"] < lsr_med).astype(float)
    fund_s = (df["funding_rate"] < 0.03).astype(float)
    liq_p80 = df["total_liq"].rolling(30, min_periods=10).quantile(0.8)
    liq_s = (df["total_liq"] < liq_p80).astype(float)
    taker_s = (df["taker_ratio"] > 1.0).astype(float)

    df["regime_score"] = lsr_s * 0.35 + fund_s * 0.35 + liq_s * 0.15 + taker_s * 0.15
    df["regime_signal"] = (df["regime_score"] > 0.5).astype(float)

    df["btc_returns"] = df["close"].pct_change()
    df["realized_vol"] = df["btc_returns"].rolling(VOL_LOOKBACK, min_periods=10).std()

    return df


# ── Simulation Engine ────────────────────────────────────────────────────────

def simulate(df, max_leverage, eth_alloc=0.0, monthly_contribution=0.0,
             target_vol=TARGET_DAILY_VOL):
    """
    Day-by-day equity simulation.

    Args:
        df: DataFrame with regime_signal, realized_vol, btc_returns, eth_returns
        max_leverage: Maximum leverage cap for vol-targeting
        eth_alloc: Fraction allocated to ETH (0.0 = BTC only, 0.3 = 70/30)
        monthly_contribution: Additional capital added monthly ($0 = lump sum only)
        target_vol: Daily vol target (higher = more aggressive)

    Returns:
        dict with metrics
    """
    n = len(df)
    equity = np.zeros(n)
    equity[0] = STARTING_CAPITAL
    total_contributed = STARTING_CAPITAL
    prev_pos = 0.0
    prev_month = df["date"].iloc[0].month

    for i in range(1, n):
        btc_ret = df["btc_returns"].iloc[i]
        eth_ret = df["eth_returns"].iloc[i] if "eth_returns" in df.columns else 0.0
        signal = df["regime_signal"].iloc[i - 1]  # yesterday's signal
        vol = df["realized_vol"].iloc[i - 1]      # yesterday's vol

        if np.isnan(btc_ret) or np.isnan(vol) or vol == 0:
            equity[i] = equity[i - 1]
            continue

        # Monthly contribution
        current_month = df["date"].iloc[i].month
        if monthly_contribution > 0 and current_month != prev_month:
            equity[i - 1] += monthly_contribution
            total_contributed += monthly_contribution
        prev_month = current_month

        # Vol-targeted position sizing
        vol_scalar = np.clip(target_vol / vol, MIN_LEVERAGE, max_leverage)
        pos = signal * vol_scalar

        # Blended return: BTC portion + ETH portion
        btc_frac = 1.0 - eth_alloc
        eth_frac = eth_alloc
        blended_ret = btc_frac * btc_ret + eth_frac * (eth_ret if not np.isnan(eth_ret) else 0.0)

        # Transaction costs on position changes
        pos_change = abs(pos - prev_pos)
        tc = pos_change * TX_COST * equity[i - 1]

        # P&L
        pnl = equity[i - 1] * pos * blended_ret - tc
        equity[i] = equity[i - 1] + pnl
        prev_pos = pos

        # Ruin check
        if equity[i] <= 0:
            equity[i:] = 0
            break

    # Compute metrics
    final = equity[-1]
    valid = equity > 0
    n_valid = valid.sum()
    n_years = n_valid / 365

    total_ret = final / STARTING_CAPITAL - 1
    cagr = (final / STARTING_CAPITAL) ** (1 / n_years) - 1 if n_years > 0 else 0

    peak = np.maximum.accumulate(equity[:n_valid])
    dd = (equity[:n_valid] - peak) / np.where(peak > 0, peak, 1)
    max_dd = dd.min() if len(dd) > 0 else 0

    # Daily returns for Sharpe
    daily_rets = np.diff(equity[:n_valid]) / equity[:n_valid - 1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365)
              if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0)

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Worst month
    dates = df["date"].iloc[:n_valid]
    eq_series = pd.Series(equity[:n_valid], index=dates)
    monthly_eq = eq_series.resample("ME").last()
    monthly_rets = monthly_eq.pct_change().dropna()
    worst_month = monthly_rets.min() if len(monthly_rets) > 0 else 0
    best_month = monthly_rets.max() if len(monthly_rets) > 0 else 0

    # Time to $1M (projected)
    if cagr > 0:
        target = 1_000_000
        if final >= target:
            years_to_1m = n_years  # already there
        else:
            # With contributions
            if monthly_contribution > 0:
                # Iterative projection: compound at CAGR + add monthly
                projected = final
                months = 0
                monthly_rate = (1 + cagr) ** (1/12) - 1
                while projected < target and months < 1200:  # 100 year cap
                    projected = projected * (1 + monthly_rate) + monthly_contribution
                    months += 1
                years_to_1m = n_years + months / 12
            else:
                years_from_now = np.log(target / final) / np.log(1 + cagr)
                years_to_1m = n_years + years_from_now
    else:
        years_to_1m = float("inf")

    # Exposure
    positions = df["regime_signal"].iloc[:-1].values  # shifted
    exposure = np.mean(positions > 0) if len(positions) > 0 else 0

    return {
        "final_equity": final,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "worst_month": worst_month,
        "best_month": best_month,
        "years_to_1m": years_to_1m,
        "exposure": exposure,
        "total_contributed": total_contributed,
        "equity_curve": equity,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("  MAESTRO MILLIONAIRE PATH")
    print("  How does $1,000 become $1,000,000?")
    print("=" * 80)

    # Load data
    print("\nLoading BTC derivatives data...")
    df = load_btc_data()
    df = compute_regime_signal(df)

    print("Loading ETH price data...")
    eth = load_eth_prices()
    df = df.merge(eth, on="date", how="left")
    df["eth_returns"] = df["eth_close"].pct_change()

    df = df.dropna(subset=["btc_returns", "realized_vol", "regime_score"]).reset_index(drop=True)
    print(f"Period: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} "
          f"({len(df)/365:.1f} years, {len(df)} days)")

    # ── First: Check vol distribution ──────────────────────────────────
    vol_vals = df["realized_vol"].dropna()
    print(f"\n  Realized Vol Distribution (20-day):")
    print(f"    Median: {vol_vals.median()*100:.2f}%/day | Mean: {vol_vals.mean()*100:.2f}%/day")
    print(f"    10th pct: {vol_vals.quantile(0.10)*100:.2f}% | 90th pct: {vol_vals.quantile(0.90)*100:.2f}%")
    print(f"    At 1.5% target, median leverage = {0.015/vol_vals.median():.2f}x")

    # ── QUESTION 1: Vol Target Sweep (the REAL lever) ────────────────────
    print(f"\n{'='*80}")
    print("  QUESTION 1: What risk level maximizes your path to $1M?")
    print(f"{'='*80}")
    print(f"  Starting capital: ${STARTING_CAPITAL:,.0f} | Max leverage cap: 5.0x")
    print(f"  Signal: BTC Weighted Regime (p=0.012, walk-forward validated)")
    print(f"  Sweeping daily vol target (the REAL dial for aggressiveness)")
    print()

    vol_targets = [0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]

    header = (f"  {'VolTgt':>6} | {'AvgLev':>6} | {'Final $':>10} | {'CAGR':>7} | {'Sharpe':>6} | "
              f"{'MaxDD':>7} | {'Calmar':>6} | {'Yrs→$1M':>8} | {'Worst Mo':>9}")
    print(header)
    print(f"  {'-'*82}")

    vol_results = {}
    for vt in vol_targets:
        r = simulate(df, max_leverage=5.0, eth_alloc=0.0, target_vol=vt)
        vol_results[vt] = r
        # Compute average effective leverage
        vol_vals_clean = df["realized_vol"].dropna()
        avg_lev = np.clip(vt / vol_vals_clean, MIN_LEVERAGE, 5.0).mean()
        yrs_str = f"{r['years_to_1m']:.1f}" if r["years_to_1m"] < 100 else "never"
        print(f"  {vt*100:>5.1f}% | {avg_lev:>5.2f}x | ${r['final_equity']:>9,.0f} | {r['cagr']:>6.1%} | "
              f"{r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['calmar']:>6.2f} | "
              f"{yrs_str:>8s} | {r['worst_month']:>8.1%}")

    # ── QUESTION 2: BTC vs BTC+ETH at various risk levels ───────────────
    print(f"\n{'='*80}")
    print("  QUESTION 2: Does adding ETH amplify returns?")
    print(f"{'='*80}")
    print(f"  Same BTC regime signal applied to BTC+ETH blend\n")

    eth_allocations = [0.0, 0.3, 0.5]
    test_vol_targets = [0.015, 0.030, 0.050, 0.080]

    header2 = (f"  {'VolTgt':>6} | {'Alloc':>12} | {'Final $':>10} | {'CAGR':>7} | "
               f"{'Sharpe':>6} | {'MaxDD':>7} | {'Yrs→$1M':>8}")
    print(header2)
    print(f"  {'-'*68}")

    for vt in test_vol_targets:
        for eth_pct in eth_allocations:
            alloc_str = f"BTC {(1-eth_pct)*100:.0f}/ETH {eth_pct*100:.0f}" if eth_pct > 0 else "BTC 100%"
            r = simulate(df, max_leverage=5.0, eth_alloc=eth_pct, target_vol=vt)
            yrs_str = f"{r['years_to_1m']:.1f}" if r["years_to_1m"] < 100 else "never"
            print(f"  {vt*100:>5.1f}% | {alloc_str:>12s} | ${r['final_equity']:>9,.0f} | "
                  f"{r['cagr']:>6.1%} | {r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {yrs_str:>8s}")
        print()

    # ── QUESTION 3: With Monthly Contributions ───────────────────────────
    print(f"{'='*80}")
    print("  QUESTION 3: What if you add money monthly?")
    print(f"{'='*80}")
    print(f"  $1,000 initial + monthly contributions, regime-gated deployment\n")

    contribution_amounts = [0, 250, 500, 1000]
    rec_vol_targets = [0.015, 0.030, 0.050]

    header3 = (f"  {'VolTgt':>6} | {'Monthly':>8} | {'Final $':>10} | {'Invested':>10} | "
               f"{'Multiple':>8} | {'Yrs→$1M':>8}")
    print(header3)
    print(f"  {'-'*62}")

    for vt in rec_vol_targets:
        for contrib in contribution_amounts:
            r = simulate(df, max_leverage=5.0, eth_alloc=0.0,
                         monthly_contribution=contrib, target_vol=vt)
            contrib_str = f"${contrib:,}" if contrib > 0 else "none"
            multiple = r["final_equity"] / r["total_contributed"] if r["total_contributed"] > 0 else 0
            yrs_str = f"{r['years_to_1m']:.1f}" if r["years_to_1m"] < 100 else "never"
            print(f"  {vt*100:>5.1f}% | {contrib_str:>8s} | ${r['final_equity']:>9,.0f} | "
                  f"${r['total_contributed']:>9,.0f} | {multiple:>7.1f}x | "
                  f"{yrs_str:>8s}")
        print()

    # ── QUESTION 4: Aggressive + ETH + Contributions (the full package) ──
    print(f"{'='*80}")
    print("  QUESTION 4: The Full Package — Max everything")
    print(f"{'='*80}")
    print(f"  Best combos of vol target + ETH allocation + monthly DCA\n")

    combo_header = (f"  {'VolTgt':>6} | {'Alloc':>12} | {'Monthly':>8} | {'Final $':>10} | "
                    f"{'MaxDD':>7} | {'Sharpe':>6} | {'Yrs→$1M':>8}")
    print(combo_header)
    print(f"  {'-'*72}")

    combos = [
        (0.015, 0.0,  500,  "Conservative baseline"),
        (0.030, 0.0,  500,  "2x risk, BTC only"),
        (0.030, 0.3,  500,  "2x risk, BTC+ETH"),
        (0.050, 0.0,  500,  "3.3x risk, BTC only"),
        (0.050, 0.3,  500,  "3.3x risk, BTC+ETH"),
        (0.050, 0.3,  1000, "3.3x risk, BTC+ETH, $1K/mo"),
        (0.080, 0.0,  500,  "5x risk, BTC only"),
        (0.080, 0.3,  500,  "5x risk, BTC+ETH"),
        (0.080, 0.3,  1000, "5x risk, BTC+ETH, $1K/mo"),
        (0.100, 0.3,  1000, "YOLO: 6.7x risk, BTC+ETH, $1K/mo"),
    ]

    for vt, eth_pct, contrib, label in combos:
        alloc_str = f"BTC {(1-eth_pct)*100:.0f}/ETH {eth_pct*100:.0f}" if eth_pct > 0 else "BTC 100%"
        contrib_str = f"${contrib:,}"
        r = simulate(df, max_leverage=5.0, eth_alloc=eth_pct,
                     monthly_contribution=contrib, target_vol=vt)
        yrs_str = f"{r['years_to_1m']:.1f}" if r["years_to_1m"] < 100 else "never"
        print(f"  {vt*100:>5.1f}% | {alloc_str:>12s} | {contrib_str:>8s} | ${r['final_equity']:>9,.0f} | "
              f"{r['max_dd']:>6.1%} | {r['sharpe']:>6.2f} | {yrs_str:>8s}")

    # ── RECOMMENDATION ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  THE HONEST ANSWER")
    print(f"{'='*80}")

    # Find optimal vol target per risk tolerance
    for label, dd_limit in [("Conservative", -0.25), ("Moderate", -0.40), ("Aggressive", -0.60)]:
        best = None
        for vt, r in vol_results.items():
            if r["max_dd"] >= dd_limit:
                if best is None or r["cagr"] > best["cagr"]:
                    best = {**r, "vol_target": vt}
        if best:
            yrs = f"{best['years_to_1m']:.0f}" if best["years_to_1m"] < 100 else "never"
            # Also with $500/mo
            r_dca = simulate(df, max_leverage=5.0, monthly_contribution=500,
                             target_vol=best["vol_target"])
            yrs_dca = f"{r_dca['years_to_1m']:.0f}" if r_dca["years_to_1m"] < 100 else "never"
            avg_lev = np.clip(best["vol_target"] / vol_vals, MIN_LEVERAGE, 5.0).mean()
            print(f"\n  {label} (MaxDD < {abs(dd_limit)*100:.0f}%):")
            print(f"    Vol target: {best['vol_target']*100:.1f}%/day (~{avg_lev:.1f}x avg lev) | "
                  f"CAGR: {best['cagr']:.1%} | MaxDD: {best['max_dd']:.1%} | Sharpe: {best['sharpe']:.2f}")
            print(f"    $1K lump sum → $1M in ~{yrs} years")
            print(f"    $1K + $500/mo → $1M in ~{yrs_dca} years")

    print(f"\n  {'─'*76}")
    print(f"  NOTE: These projections assume the regime signal continues to work as")
    print(f"  validated. Past performance ≠ future results. The edge IS real (p=0.012)")
    print(f"  but crypto markets evolve. Paper trade 3 months before deploying real capital.")
    print(f"  Max leverage above 2.5x risks liquidation on perp exchanges.")
    print(f"{'='*80}")
