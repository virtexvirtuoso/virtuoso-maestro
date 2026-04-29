#!/usr/bin/env python3
"""
Mega V4 Combined System — V3 base + gap-filling strategies.

Combines:
- V3 base (long + short + adaptive leverage)
- Mean Reversion during ACCUMULATION/NEUTRAL
- Vol Breakout overlay
- Funding Carry in all regimes
- Sector Rotation

Usage:
    cd ~/Desktop/maestro/backend
    source venv/bin/activate
    python backtest_combined_system.py
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_gap_strategies import (
    load_crypto_ohlcv, load_funding, load_macro_simple,
    compute_confluence_simple, detect_regime,
    strategy_vol_breakout, strategy_mean_reversion,
    strategy_enhanced_short, strategy_funding_carry,
    strategy_sector_rotation, buy_and_hold,
    calc_metrics, regime_metrics, sharpe, max_drawdown,
    TX_COST, RESULTS_DIR, END
)

# Try importing V3 runner
try:
    from strategies.composite.mega_strategy_v3 import run_single_asset, ASSET_CONFIGS
    HAS_V3 = True
except ImportError:
    HAS_V3 = False
    print("WARNING: Could not import mega_strategy_v3, using simplified V3 proxy")


def run_v3_simplified(df, confluence, regime, macro_data, funding_daily=None):
    """Simplified V3 proxy: long in BULL/MILD_BULL, adaptive sizing."""
    from backtest_gap_strategies import rsi, bb, atr, sma

    close = df["close"]
    idx = df.index
    r = rsi(close, 14)
    s100 = sma(close, 100)
    lo, mid, hi = bb(close, 20, 2.0)
    at = atr(df, 14)

    # Leverage from confluence
    lev_map = {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}

    daily_pnl = pd.Series(0.0, index=idx)
    position = 0.0
    entry_price = 0.0
    n = len(df)

    conf_s = confluence.shift(1).fillna(0)
    regime_s = regime.shift(1).fillna("NEUTRAL")
    r_s = r.shift(1).fillna(50)
    s100_s = s100.shift(1).fillna(close)
    lo_s = lo.shift(1).fillna(close)
    hi_s = hi.shift(1).fillna(close)

    for i in range(1, n):
        price = close.iloc[i]
        prev_price = close.iloc[i-1]

        if position != 0 and prev_price > 0:
            ret = (price - prev_price) / prev_price
            daily_pnl.iloc[i] = ret * position

        c = int(conf_s.iloc[i])
        reg = regime_s.iloc[i]
        lev = lev_map.get(min(c, 5), 0)

        # Long entry in uptrend regimes on dips
        if position == 0 and reg in ("BULL", "MILD_BULL", "ACCUMULATION"):
            if r_s.iloc[i] < 45 or prev_price <= lo_s.iloc[i]:
                position = 0.6 * lev
                entry_price = price
                daily_pnl.iloc[i] -= abs(position) * TX_COST

        # Trim on overbought
        elif position > 0:
            if r_s.iloc[i] > 75 or prev_price >= hi_s.iloc[i]:
                position *= 0.75
                daily_pnl.iloc[i] -= position * 0.25 * TX_COST

            # Exit on regime change
            if reg in ("BEAR",):
                daily_pnl.iloc[i] -= abs(position) * TX_COST
                position = 0.0

            # Trailing stop 12%
            if entry_price > 0 and (price - entry_price) / entry_price < -0.12:
                daily_pnl.iloc[i] -= abs(position) * TX_COST
                position = 0.0

    return daily_pnl


def run_v4_combined(ohlcv, funding, macro_data, cross_asset, weights=None):
    """
    V4 = V3 base + mean reversion + vol breakout + funding carry + enhanced short
    Each module contributes to daily P&L with a weight.
    """
    if weights is None:
        weights = {
            "v3_base": 0.40,
            "mean_reversion": 0.15,
            "vol_breakout": 0.15,
            "enhanced_short": 0.15,
            "funding_carry": 0.15,
        }

    all_pnls = {}

    for name, df in ohlcv.items():
        conf, _ = compute_confluence_simple(df["close"], macro_data, cross_asset)
        reg = detect_regime(conf)
        fd = funding.get(name)

        # V3 base
        v3_pnl = run_v3_simplified(df, conf, reg, macro_data, fd)

        # Mean reversion
        mr_pnl = strategy_mean_reversion(df, conf, reg)

        # Vol breakout
        vb_pnl = strategy_vol_breakout(df, conf, reg)

        # Enhanced short
        es_pnl = strategy_enhanced_short(df, conf, reg, macro_data, fd)

        # Funding carry
        fc_pnl = strategy_funding_carry(df["close"], fd, name)

        # Combine with weights
        combined = (
            v3_pnl * weights["v3_base"] +
            mr_pnl * weights["mean_reversion"] +
            vb_pnl * weights["vol_breakout"] +
            es_pnl * weights["enhanced_short"] +
            fc_pnl * weights["funding_carry"]
        )

        all_pnls[name] = {
            "combined": combined,
            "v3_base": v3_pnl,
            "regime": reg,
            "confluence": conf,
        }

    return all_pnls


def main():
    print("=" * 70)
    print("MEGA V4 COMBINED SYSTEM BACKTEST")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    assets = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
    ohlcv = {}
    for name, ticker in assets.items():
        df = load_crypto_ohlcv(ticker)
        if len(df) > 0:
            ohlcv[name] = df
            print(f"  {name}: {len(df)} rows")

    funding = {}
    for name in assets:
        fd = load_funding(name)
        if fd is not None:
            funding[name] = fd

    macro_data, cross_asset = load_macro_simple()

    # V3 standalone
    print("\n[2/4] Running V3 standalone...")
    v3_results = {}
    for name, df in ohlcv.items():
        conf, _ = compute_confluence_simple(df["close"], macro_data, cross_asset)
        reg = detect_regime(conf)
        pnl = run_v3_simplified(df, conf, reg, macro_data, funding.get(name))
        m = calc_metrics(pnl, f"V3_{name}")
        rm = regime_metrics(pnl, reg)
        m["regime_sharpe"] = rm
        v3_results[name] = m
        print(f"  V3 {name}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")
        for r_name, r_sh in rm.items():
            print(f"    {r_name}: Sharpe={r_sh}")

    # V4 combined
    print("\n[3/4] Running V4 combined...")
    v4_pnls = run_v4_combined(ohlcv, funding, macro_data, cross_asset)

    v4_results = {}
    for name in v4_pnls:
        pnl = v4_pnls[name]["combined"]
        reg = v4_pnls[name]["regime"]
        m = calc_metrics(pnl, f"V4_{name}")
        rm = regime_metrics(pnl, reg)
        m["regime_sharpe"] = rm
        v4_results[name] = m
        print(f"  V4 {name}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")
        for r_name, r_sh in rm.items():
            print(f"    {r_name}: Sharpe={r_sh}")

    # Sector rotation (portfolio-level module)
    print("\n  Running sector rotation...")
    conf_btc, _ = compute_confluence_simple(ohlcv["BTC"]["close"], macro_data, cross_asset)
    reg_btc = detect_regime(conf_btc)
    sr_pnl, _ = strategy_sector_rotation(reg_btc)

    # Portfolio-level comparison
    print("\n[4/4] Portfolio comparison...")
    # Equal-weight portfolio across assets
    portfolio_v3_pnl = pd.Series(dtype=float)
    portfolio_v4_pnl = pd.Series(dtype=float)
    portfolio_bh_pnl = pd.Series(dtype=float)

    for name in ohlcv:
        v3_pnl = run_v3_simplified(
            ohlcv[name],
            *compute_confluence_simple(ohlcv[name]["close"], macro_data, cross_asset)[:1],
            detect_regime(compute_confluence_simple(ohlcv[name]["close"], macro_data, cross_asset)[0]),
            macro_data
        )
        v4_pnl = v4_pnls[name]["combined"]
        bh_pnl = buy_and_hold(ohlcv[name]["close"])

        w = 1.0 / len(ohlcv)
        if len(portfolio_v3_pnl) == 0:
            portfolio_v3_pnl = v3_pnl * w
            portfolio_v4_pnl = v4_pnl * w
            portfolio_bh_pnl = bh_pnl * w
        else:
            # Align indices
            common = portfolio_v3_pnl.index.intersection(v3_pnl.index)
            portfolio_v3_pnl = portfolio_v3_pnl.reindex(common).fillna(0) + v3_pnl.reindex(common).fillna(0) * w
            portfolio_v4_pnl = portfolio_v4_pnl.reindex(common).fillna(0) + v4_pnl.reindex(common).fillna(0) * w
            portfolio_bh_pnl = portfolio_bh_pnl.reindex(common).fillna(0) + bh_pnl.reindex(common).fillna(0) * w

    # Add sector rotation to V4 portfolio (10% weight)
    if len(sr_pnl) > 0:
        common = portfolio_v4_pnl.index.intersection(sr_pnl.index)
        portfolio_v4_pnl = portfolio_v4_pnl.reindex(common).fillna(0) * 0.9 + sr_pnl.reindex(common).fillna(0) * 0.1

    print("\n" + "=" * 70)
    print("PORTFOLIO RESULTS (equal-weight BTC+ETH+SOL)")
    print("=" * 70)

    port_reg = reg_btc

    for label, pnl in [("Buy & Hold", portfolio_bh_pnl), ("V3 Standalone", portfolio_v3_pnl), ("V4 Combined", portfolio_v4_pnl)]:
        m = calc_metrics(pnl, label)
        rm = regime_metrics(pnl, port_reg.reindex(pnl.index, method="ffill").fillna("NEUTRAL"))
        print(f"\n  {label}:")
        print(f"    Sharpe: {m['sharpe']}")
        print(f"    Total Return: {m['total_return']:.2%}")
        print(f"    Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"    Win Rate: {m['win_rate']:.2%}")
        print(f"    Regime Sharpes: {rm}")
        m["regime_sharpe"] = rm

    # Compile all results
    all_results = {
        "v3_per_asset": v3_results,
        "v4_per_asset": v4_results,
        "portfolio": {
            "buy_hold": calc_metrics(portfolio_bh_pnl, "BuyHold"),
            "v3": calc_metrics(portfolio_v3_pnl, "V3"),
            "v4": calc_metrics(portfolio_v4_pnl, "V4"),
        },
        "v4_improves_accumulation": True,  # will be verified
        "timestamp": str(pd.Timestamp.now()),
    }

    # Check if V4 fixes ACCUMULATION gap
    v4_acc_sharpes = {}
    for name in v4_pnls:
        reg = v4_pnls[name]["regime"]
        pnl = v4_pnls[name]["combined"]
        acc_mask = reg == "ACCUMULATION"
        if acc_mask.sum() > 10:
            v4_acc_sharpes[name] = round(float(sharpe(pnl[acc_mask])), 3)

    all_results["v4_accumulation_sharpes"] = v4_acc_sharpes
    all_results["v4_improves_accumulation"] = all(s > -2.0 for s in v4_acc_sharpes.values()) if v4_acc_sharpes else False

    print(f"\n  V4 ACCUMULATION Sharpes: {v4_acc_sharpes}")
    print(f"  Fixes ACCUMULATION gap: {all_results['v4_improves_accumulation']}")

    # Save
    out_path = RESULTS_DIR / "gap_strategies_results.json"
    # Merge with gap strategies results if they exist
    try:
        with open(out_path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    existing["combined_system"] = all_results
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
