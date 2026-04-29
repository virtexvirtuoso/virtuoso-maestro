#!/usr/bin/env python3
"""
VPIN v3 — Freqtrade-style Execution Simulator
===============================================
Replays the V6 trade schedule (vpin_v3_trade_schedule.csv) through the
same fee/funding/sizing math that the Maestro V6 backtest used, to prove
that the schedule CSV captures everything a downstream execution layer
needs.

This is the reconciliation gate before the Freqtrade port. If the sim
reproduces Maestro V6 (+3.43 OOS Sharpe / +56.3% / -9.67% MDD) within a
reasonable tolerance, the schedule is trustworthy and we can write the
Freqtrade strategy.

Intentionally NOT tested here:
  - Real 8h Binance funding events (data we have is daily-aggregated)
  - 1m bar fill slippage vs bucket close
  - Exchange microstructure (order book, partial fills)
These are all checked inside Freqtrade itself in the next step.

Output: stdout table + backend/research/vpin_results/vpin_v3_sim_reconcile.csv
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.research.vpin_v2_portfolio_variants import (
    load_asset_once,
    trades_to_daily_returns,
    metrics,
    FEE_PER_SIDE,
    SLIP_PER_SIDE,
    IS_FRACTION,
    Trade,
)

SCHEDULE_PATH = Path(
    "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/"
    "vpin_v3_trade_schedule.csv"
)
OUT_PATH = Path(
    "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/"
    "vpin_v3_sim_reconcile.csv"
)

V6_ASSETS = ["suiusdt", "avaxusdt", "linkusdt", "fetusdt", "arbusdt"]

# Reference numbers from V6 (taken from vpin_v2_portfolio_variants.csv)
V6_REFERENCE = {
    "oos_sharpe": 3.43,
    "oos_total": 0.5633,
    "oos_mdd": -0.0967,
    "is_sharpe": 0.44,
    "n_trades": 698,
}


def load_schedule() -> pd.DataFrame:
    df = pd.read_csv(
        SCHEDULE_PATH,
        parse_dates=["entry_time_utc", "exit_time_utc"],
    )
    return df


def schedule_row_to_trade(row: pd.Series, funding: pd.Series) -> Trade:
    """Rebuild a Trade dataclass from a schedule row + funding series."""
    entry_t = row["entry_time_utc"]
    exit_t = row["exit_time_utc"]
    side = int(row["side"])
    entry_px = float(row["entry_px"])
    exit_px = float(row["exit_px"])
    holding_days = float(row["holding_days"])

    # Apply funding exactly the same way variants.py does (for reconciliation)
    funding_cost = 0.0
    if len(funding) > 0:
        mask = (funding.index >= entry_t.normalize()) & (
            funding.index <= exit_t.normalize()
        )
        rel = funding.loc[mask]
        if len(rel) > 0:
            avg_daily = float(rel.mean())
            funding_cost = side * avg_daily * holding_days

    raw_log_ret = side * np.log(exit_px / entry_px)
    fee_cost = 2 * (FEE_PER_SIDE + SLIP_PER_SIDE)
    net_log_ret = raw_log_ret - funding_cost - fee_cost

    return Trade(
        asset=row["asset"],
        entry_time=entry_t,
        exit_time=exit_t,
        side=side,
        entry_px=entry_px,
        exit_px=exit_px,
        raw_log_ret=float(raw_log_ret),
        holding_days=holding_days,
        funding_cost=float(funding_cost),
        fee_cost=float(fee_cost),
        net_log_ret=float(net_log_ret),
    )


def main() -> None:
    print(f"Loading schedule: {SCHEDULE_PATH}")
    sched = load_schedule()
    print(f"  rows: {len(sched)}")
    print(f"  assets: {sorted(sched['asset'].unique())}")
    print(f"  span: {sched['entry_time_utc'].min()} → {sched['exit_time_utc'].max()}\n")

    # Load asset data (for funding + IS vol)
    asset_data: dict = {}
    for a in V6_ASSETS:
        asset_data[a] = load_asset_once(a)

    # Rebuild trades asset by asset
    all_trades: dict[str, list[Trade]] = {a: [] for a in V6_ASSETS}
    for _, row in sched.iterrows():
        a = row["asset"]
        funding = asset_data[a]["funding"]
        t = schedule_row_to_trade(row, funding)
        all_trades[a].append(t)

    for a in V6_ASSETS:
        trs = all_trades[a]
        if not trs:
            print(f"  {a:10s}  NO TRADES")
            continue
        raw = np.mean([t.raw_log_ret for t in trs])
        net = np.mean([t.net_log_ret for t in trs])
        longs = sum(1 for t in trs if t.side == +1)
        shorts = sum(1 for t in trs if t.side == -1)
        print(
            f"  {a:10s}  trades={len(trs):4d}  "
            f"(L={longs} S={shorts})  "
            f"raw={raw*100:+.3f}% net={net*100:+.3f}%"
        )

    # Equal-inverse-vol weighting (same as V6)
    inv_vol = {
        a: (1.0 / asset_data[a]["vol"])
        if (asset_data[a]["vol"] and np.isfinite(asset_data[a]["vol"]) and asset_data[a]["vol"] > 0)
        else 0.0
        for a in V6_ASSETS
    }
    tot_iv = sum(inv_vol.values())
    weights = {a: (inv_vol[a] / tot_iv if tot_iv > 0 else 0.0) for a in V6_ASSETS}

    print("\n  weights:", {a: f"{w*100:.1f}%" for a, w in weights.items()})

    # Span = union of all asset spans (same as V6)
    min_start = min(asset_data[a]["span"][0] for a in V6_ASSETS)
    max_end = max(asset_data[a]["span"][1] for a in V6_ASSETS)

    basket = pd.Series(
        0.0,
        index=pd.date_range(min_start.normalize(), max_end.normalize(), freq="D"),
    )
    per_asset_daily: dict[str, pd.Series] = {}
    for a in V6_ASSETS:
        ret = trades_to_daily_returns(all_trades[a], weights[a], min_start, max_end)
        if len(ret) == 0:
            continue
        ret = ret.reindex(basket.index, fill_value=0.0)
        per_asset_daily[a] = ret
        basket = basket + ret

    cutoff = min_start + (max_end - min_start) * IS_FRACTION
    is_ret = basket[basket.index < cutoff]
    oos_ret = basket[basket.index >= cutoff]

    is_m = metrics(is_ret)
    oos_m = metrics(oos_ret)
    full_m = metrics(basket)

    print("\n" + "=" * 70)
    print("VPIN v3 SIM RECONCILIATION")
    print("=" * 70)
    print(
        f"  FULL n={full_m['n_days']:4d}d  Sharpe={full_m['sharpe']:+.2f}  "
        f"total={full_m['total_ret']*100:+.2f}%  MDD={full_m['mdd']*100:+.2f}%"
    )
    print(
        f"  IS   n={is_m['n_days']:4d}d  Sharpe={is_m['sharpe']:+.2f}  "
        f"total={is_m['total_ret']*100:+.2f}%  MDD={is_m['mdd']*100:+.2f}%"
    )
    print(
        f"  OOS  n={oos_m['n_days']:4d}d  Sharpe={oos_m['sharpe']:+.2f}  "
        f"total={oos_m['total_ret']*100:+.2f}%  MDD={oos_m['mdd']*100:+.2f}%"
    )

    # Reconciliation vs V6 reference
    print("\n" + "=" * 70)
    print("RECONCILIATION vs Maestro V6")
    print("=" * 70)
    rec_rows = []

    def compare(metric: str, got: float, ref: float, tol: float = 0.05):
        if np.isnan(got) or np.isnan(ref):
            verdict = "SKIP"
            diff = float("nan")
        else:
            diff = got - ref
            if abs(ref) < 1e-9:
                verdict = "PASS" if abs(got) < tol else "FAIL"
            else:
                rel = abs(diff / ref)
                verdict = "PASS" if rel < tol else "FAIL"
        marker = "✓" if verdict == "PASS" else ("✗" if verdict == "FAIL" else "·")
        print(
            f"  {marker} {metric:20s}  sim={got:+.4f}  ref={ref:+.4f}  "
            f"diff={diff:+.4f}  [{verdict}]"
        )
        rec_rows.append({
            "metric": metric,
            "sim": got,
            "ref": ref,
            "diff": diff,
            "verdict": verdict,
        })

    n_sim = sum(len(t) for t in all_trades.values())
    compare("n_trades", n_sim, V6_REFERENCE["n_trades"], tol=0.02)
    compare("is_sharpe", is_m["sharpe"], V6_REFERENCE["is_sharpe"], tol=0.05)
    compare("oos_sharpe", oos_m["sharpe"], V6_REFERENCE["oos_sharpe"], tol=0.05)
    compare("oos_total", oos_m["total_ret"], V6_REFERENCE["oos_total"], tol=0.05)
    compare("oos_mdd", oos_m["mdd"], V6_REFERENCE["oos_mdd"], tol=0.05)

    pd.DataFrame(rec_rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    all_pass = all(r["verdict"] == "PASS" for r in rec_rows)
    if all_pass:
        print("\n" + "=" * 70)
        print("GATE: PASS — schedule CSV is a complete execution artifact.")
        print("      Safe to write Freqtrade strategy.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("GATE: FAIL — reconcile before writing Freqtrade strategy.")
        print("=" * 70)


if __name__ == "__main__":
    main()
