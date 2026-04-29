#!/usr/bin/env python3
"""
VPIN v2 — Funding Rate Filter Sensitivity Sweep
===============================================
Runs the V6 config (alt trio long/short + FET/ARB short-only) at multiple
|funding| thresholds to answer open question #3 from VPIN-v2-Results:
does requiring extreme funding at entry stack with the VPIN trigger, or
does it just shrink the sample?

V5 (5 bps/day hard filter) collapsed to 16 trades. This sweep tests a
softer grid: 0, 1, 2, 3, 5 bps/day on entry-day funding rate.

Output: backend/research/vpin_results/vpin_v2_funding_sweep.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backend.research.vpin_v2_portfolio_variants import (
    Variant,
    run_variant,
    OUT_DIR,
)

THRESHOLDS_BPS = [0, 1, 2, 3, 5]

V6_ASSETS = ["suiusdt", "avaxusdt", "linkusdt", "fetusdt", "arbusdt"]
V6_SIDES = {
    "suiusdt": "both",
    "avaxusdt": "both",
    "linkusdt": "both",
    "fetusdt": "short",
    "arbusdt": "short",
}


def build_variants() -> list[Variant]:
    out = []
    for bps in THRESHOLDS_BPS:
        out.append(
            Variant(
                name=f"V6_funding_{bps}bps",
                assets=V6_ASSETS,
                sides=V6_SIDES,
                funding_min_abs=bps / 10000.0,
                description=f"V6 config, require |funding_entry| >= {bps} bps/day",
            )
        )
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for v in build_variants():
        try:
            r = run_variant(v)
            r["funding_threshold_bps"] = float(v.funding_min_abs * 10000)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR on {v.name}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(results)
    out_file = OUT_DIR / "vpin_v2_funding_sweep.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 90)
    print("FUNDING SWEEP SUMMARY (V6 base)")
    print("=" * 90)
    print(
        f"{'threshold':>10s} {'n_tr':>5s} {'tr/yr':>6s} "
        f"{'IS_Sh':>6s} {'OOS_Sh':>7s} {'OOS_tot':>9s} {'OOS_MDD':>9s}"
    )
    for r in results:
        print(
            f"{r['funding_threshold_bps']:7.0f}bps "
            f"{r['n_trades']:5d} "
            f"{r['trades_per_year']:6.0f} "
            f"{r['is_sharpe']:+6.2f} "
            f"{r['oos_sharpe']:+7.2f} "
            f"{r['oos_total']*100:+8.2f}% "
            f"{r['oos_mdd']*100:+8.2f}%"
        )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
