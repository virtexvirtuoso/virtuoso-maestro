#!/usr/bin/env python3
"""
VPIN v2 — V7 SOL/TIA Extension Test
====================================
Tests whether adding SOL and/or TIA to the V6 production basket
(SUI/AVAX/LINK long-short + FET/ARB short-only) improves OOS edge.

6 variants layered onto V6:
  V7a: V6 + SOL (both)
  V7b: V6 + SOL (long)
  V7c: V6 + SOL (short)
  V7d: V6 + TIA (both)
  V7e: V6 + TIA (long)
  V7f: V6 + TIA (short)

Uses the same IS/OOS split, bucket sizing, and variant framework as
vpin_v2_portfolio_variants.py so results are directly comparable to V6.

SOL/TIA 1m data spans 2024-01-01 → 2026-02-24 (same window as V6 research).
Bucket sizing and vol are computed per-asset from each asset's own IS window.

Decision gate:
  V7* OOS Sharpe must exceed V6 (+3.43) by a non-trivial margin (>+0.2)
  AND MDD must not blow out beyond -12%, OR it's rejected and we ship V6.

Output: backend/research/vpin_results/vpin_v2_v7_soltia.csv
"""
from __future__ import annotations
import pandas as pd

from backend.research import vpin_v2_portfolio_variants as vpv
from backend.research.vpin_v2_portfolio_variants import (
    Variant,
    run_variant,
    OUT_DIR,
)

# The base variants module only registers funding files for the V1..V6
# universe. SOL and TIA aren't in FUNDING_MAP, so load_funding() falls
# through to `FUNDING_DIR / ""` and raises IsADirectoryError. Patch the
# map in-place here so the run can proceed without mutating research code.
vpv.FUNDING_MAP["solusdt"] = "sol_funding_full.csv"
vpv.FUNDING_MAP["tiausdt"] = "tia_funding_full.csv"


V6_BASE_ASSETS = ["suiusdt", "avaxusdt", "linkusdt", "fetusdt", "arbusdt"]
V6_BASE_SIDES = {
    "suiusdt": "both",
    "avaxusdt": "both",
    "linkusdt": "both",
    "fetusdt": "short",
    "arbusdt": "short",
}


def v7_variant(name: str, extra_asset: str, extra_side: str, desc: str) -> Variant:
    assets = V6_BASE_ASSETS + [extra_asset]
    sides = dict(V6_BASE_SIDES)
    sides[extra_asset] = extra_side
    return Variant(
        name=name,
        assets=assets,
        sides=sides,
        description=desc,
    )


V7_VARIANTS: list[Variant] = [
    # include V6 again as a baseline reference row so the CSV is self-contained
    Variant(
        name="V6_baseline_ref",
        assets=V6_BASE_ASSETS,
        sides=V6_BASE_SIDES,
        description="V6 production reference (for direct comparison)",
    ),
    v7_variant("V7a_plus_sol_both", "solusdt", "both",
               "V6 + SOL long/short"),
    v7_variant("V7b_plus_sol_long", "solusdt", "long",
               "V6 + SOL long-only"),
    v7_variant("V7c_plus_sol_short", "solusdt", "short",
               "V6 + SOL short-only"),
    v7_variant("V7d_plus_tia_both", "tiausdt", "both",
               "V6 + TIA long/short"),
    v7_variant("V7e_plus_tia_long", "tiausdt", "long",
               "V6 + TIA long-only"),
    v7_variant("V7f_plus_tia_short", "tiausdt", "short",
               "V6 + TIA short-only"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for v in V7_VARIANTS:
        try:
            results.append(run_variant(v))
        except Exception as e:
            import traceback
            print(f"  ERROR on {v.name}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(results)
    out_file = OUT_DIR / "vpin_v2_v7_soltia.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 90)
    print("V7 SOL/TIA EXTENSION SUMMARY")
    print("=" * 90)
    print(
        f"{'variant':24s} {'n_tr':>5s} {'tr/yr':>6s} "
        f"{'IS_Sh':>6s} {'OOS_Sh':>7s} {'OOS_tot':>9s} {'OOS_MDD':>9s}"
    )
    for r in results:
        print(
            f"{r['variant']:24s} "
            f"{r['n_trades']:5d} "
            f"{r['trades_per_year']:6.0f} "
            f"{r['is_sharpe']:+6.2f} "
            f"{r['oos_sharpe']:+7.2f} "
            f"{r['oos_total']*100:+8.2f}% "
            f"{r['oos_mdd']*100:+8.2f}%"
        )

    # Decision gate: anything beating V6 by > +0.2 Sharpe w/ MDD >= -12%
    v6 = next((r for r in results if r["variant"] == "V6_baseline_ref"), None)
    if v6 is not None:
        print("\n" + "=" * 90)
        print("DECISION GATE (improvement over V6)")
        print("=" * 90)
        v6_sh = v6["oos_sharpe"]
        print(f"  V6 reference: OOS Sharpe {v6_sh:+.2f}  MDD {v6['oos_mdd']*100:+.2f}%\n")
        winners = []
        for r in results:
            if r["variant"] == "V6_baseline_ref":
                continue
            d = r["oos_sharpe"] - v6_sh
            mdd_ok = r["oos_mdd"] >= -0.12
            margin_ok = d >= 0.20
            verdict = "ADOPT" if (mdd_ok and margin_ok) else \
                      "REJECT-mdd" if not mdd_ok else \
                      "REJECT-margin"
            mark = "✓" if verdict == "ADOPT" else "✗"
            print(
                f"  {mark} {r['variant']:22s}  "
                f"ΔSharpe={d:+.2f}  MDD={r['oos_mdd']*100:+.2f}%  [{verdict}]"
            )
            if verdict == "ADOPT":
                winners.append(r["variant"])
        print()
        if winners:
            print(f"  CANDIDATE(S) FOR V7: {', '.join(winners)}")
        else:
            print("  NO V7 CANDIDATES — ship V6 as the production spec.")

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
