#!/usr/bin/env python3
"""
VPIN v7 — Universe Expansion Sweep
===================================
Tests whether adding untested assets to the V6 baseline
(SUI/AVAX/LINK both + FET/ARB short) improves OOS Sharpe.

For each candidate (INJ, OP, RENDER, SOL, TAO, TIA), run:
  - V6 + candidate as "both" directions
  - V6 + candidate as "short" only
and compare vs V6 baseline.

Decision rule for adding an asset: it must improve BOTH the OOS Sharpe
AND the OOS total return without making MDD worse by >25%. This is the
same selection-bias guard the original sweep used.
"""
from __future__ import annotations
import pandas as pd

from backend.research import vpin_v2_portfolio_variants as vp
from backend.research.vpin_v2_portfolio_variants import Variant, run_variant

# Extend the funding map so load_funding() finds files for new assets
vp.FUNDING_MAP.update({
    "injusdt": "inj_funding_full.csv",
    "opusdt": "op_funding_full.csv",
    "rndrusdt": "render_funding_full.csv",
    "solusdt": "sol_funding_full.csv",
    "taousdt": "tao_funding_full.csv",
    "tiausdt": "tia_funding_full.csv",
})

V6_TRIO = ["suiusdt", "avaxusdt", "linkusdt"]
V6_SHORTS = ["fetusdt", "arbusdt"]
V6_ASSETS = V6_TRIO + V6_SHORTS
V6_SIDES = {
    "suiusdt": "both", "avaxusdt": "both", "linkusdt": "both",
    "fetusdt": "short", "arbusdt": "short",
}

CANDIDATES = ["injusdt", "opusdt", "rndrusdt", "solusdt", "taousdt", "tiausdt"]


def build_variants() -> list[Variant]:
    variants: list[Variant] = [
        Variant(
            name="V6_baseline",
            assets=list(V6_ASSETS),
            sides=dict(V6_SIDES),
            description="V6 production: trio + FET/ARB short",
        ),
    ]
    for c in CANDIDATES:
        # both directions
        variants.append(Variant(
            name=f"V7_plus_{c}_both",
            assets=V6_ASSETS + [c],
            sides={**V6_SIDES, c: "both"},
            description=f"V6 + {c} (both)",
        ))
        # short only
        variants.append(Variant(
            name=f"V7_plus_{c}_short",
            assets=V6_ASSETS + [c],
            sides={**V6_SIDES, c: "short"},
            description=f"V6 + {c} (short only)",
        ))
    return variants


def main() -> None:
    vp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = build_variants()
    print(f"Running V7 universe sweep: {len(variants)} variants\n")

    results = []
    for v in variants:
        try:
            results.append(run_variant(v))
        except Exception as e:
            import traceback
            print(f"  ERROR on {v.name}: {e}")
            traceback.print_exc()

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)

    out_csv = vp.OUT_DIR / "v7_universe_sweep.csv"
    df.to_csv(out_csv, index=False)

    # Verdict table vs V6 baseline
    baseline = df[df["variant"] == "V6_baseline"].iloc[0]
    b_oos_sh = baseline["oos_sharpe"]
    b_oos_tot = baseline["oos_total"]
    b_oos_mdd = baseline["oos_mdd"]

    print("\n" + "=" * 100)
    print("V7 UNIVERSE SWEEP — RANKED BY OOS SHARPE")
    print("=" * 100)
    print(f"{'variant':<32} {'ntrd':>6} {'is_sh':>7} {'oos_sh':>7} "
          f"{'oos_tot':>9} {'oos_mdd':>9} {'verdict':>20}")
    print("-" * 100)
    for _, row in df.iterrows():
        d_sh = row["oos_sharpe"] - b_oos_sh
        d_tot = row["oos_total"] - b_oos_tot
        # MDD is negative; "worse by >25%" means oos_mdd < 1.25*baseline_mdd (more negative)
        mdd_ok = row["oos_mdd"] >= 1.25 * b_oos_mdd
        if row["variant"] == "V6_baseline":
            verdict = "(baseline)"
        elif d_sh > 0 and d_tot > 0 and mdd_ok:
            verdict = f"ADD (+{d_sh:.2f} Sh)"
        elif d_sh > 0 and d_tot > 0:
            verdict = f"reject (MDD+{abs(row['oos_mdd']-b_oos_mdd)*100:.1f}%)"
        else:
            verdict = "reject"
        print(f"{row['variant']:<32} {row['n_trades']:>6} "
              f"{row['is_sharpe']:>+7.2f} {row['oos_sharpe']:>+7.2f} "
              f"{row['oos_total']*100:>+8.2f}% {row['oos_mdd']*100:>+8.2f}% "
              f"{verdict:>20}")
    print("=" * 100)
    print(f"\nFull CSV: {out_csv}")

    winners = df[(df["oos_sharpe"] > b_oos_sh) & (df["oos_total"] > b_oos_tot)
                 & (df["oos_mdd"] >= 1.25 * b_oos_mdd)
                 & (df["variant"] != "V6_baseline")]
    if winners.empty:
        print("\nNo candidate beat V6 baseline on all three criteria. Universe stays at 5.")
    else:
        print(f"\n{len(winners)} candidate(s) improve OOS:")
        for _, row in winners.iterrows():
            print(f"  • {row['variant']}: "
                  f"Sharpe {b_oos_sh:+.2f} → {row['oos_sharpe']:+.2f}  "
                  f"Total {b_oos_tot*100:+.2f}% → {row['oos_total']*100:+.2f}%")


if __name__ == "__main__":
    main()
