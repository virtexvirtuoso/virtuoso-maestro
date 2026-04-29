#!/usr/bin/env python3
"""
VPIN v9 — Top 20 Liquid Assets Sweep
=====================================
Tests 9 new high-liquidity assets against the v4 baseline (7 assets):
  v4 baseline: SUI/AVAX/LINK (both) + FET/ARB/OP/INJ (short)

New candidates: DOGE, XRP, BNB, ADA, 1000PEPE, NEAR, APT, WIF, DOT

For each candidate, test:
  - v4 + candidate as "both" directions
  - v4 + candidate as "short" only
and compare vs v4 baseline.

Decision rule: must improve OOS Sharpe AND OOS total return AND
MDD must not worsen by >25%.

After single-asset sweep, test all winning stacks (like v8 did).
"""
from __future__ import annotations
import pandas as pd

from backend.research import vpin_v2_portfolio_variants as vp
from backend.research.vpin_v2_portfolio_variants import Variant, run_variant

# ── Extend funding map for new assets ──
# These may not have funding CSVs yet — that's OK, run_variant handles missing gracefully
vp.FUNDING_MAP.update({
    # from v7/v8
    "injusdt": "inj_funding_full.csv",
    "opusdt": "op_funding_full.csv",
    "rndrusdt": "render_funding_full.csv",
    "solusdt": "sol_funding_full.csv",
    "taousdt": "tao_funding_full.csv",
    "tiausdt": "tia_funding_full.csv",
    # new top-20 candidates
    "dogeusdt": "doge_funding_full.csv",
    "xrpusdt": "xrp_funding_full.csv",
    "bnbusdt": "bnb_funding_full.csv",
    "adausdt": "ada_funding_full.csv",
    "1000pepeusdt": "1000pepe_funding_full.csv",
    "nearusdt": "near_funding_full.csv",
    "aptusdt": "apt_funding_full.csv",
    "wifusdt": "wif_funding_full.csv",
    "dotusdt": "dot_funding_full.csv",
})

# ── v4 baseline config (7 assets) ──
V4_BOTH = ["suiusdt", "avaxusdt", "linkusdt"]
V4_SHORTS = ["fetusdt", "arbusdt", "opusdt", "injusdt"]
V4_ASSETS = V4_BOTH + V4_SHORTS
V4_SIDES = {
    "suiusdt": "both", "avaxusdt": "both", "linkusdt": "both",
    "fetusdt": "short", "arbusdt": "short",
    "opusdt": "short", "injusdt": "short",
}

CANDIDATES = [
    "dogeusdt", "xrpusdt", "bnbusdt", "adausdt",
    "1000pepeusdt", "nearusdt", "aptusdt", "wifusdt", "dotusdt",
]


def build_single_variants() -> list[Variant]:
    """Phase 1: test each candidate individually against v4 baseline."""
    variants: list[Variant] = [
        Variant(
            name="v4_baseline",
            assets=list(V4_ASSETS),
            sides=dict(V4_SIDES),
            description="v4 production: trio both + FET/ARB/OP/INJ short",
        ),
    ]
    for c in CANDIDATES:
        # both directions
        variants.append(Variant(
            name=f"v9_{c}_both",
            assets=V4_ASSETS + [c],
            sides={**V4_SIDES, c: "both"},
            description=f"v4 + {c} (both)",
        ))
        # short only
        variants.append(Variant(
            name=f"v9_{c}_short",
            assets=V4_ASSETS + [c],
            sides={**V4_SIDES, c: "short"},
            description=f"v4 + {c} (short only)",
        ))
    return variants


def print_results(df: pd.DataFrame, baseline_name: str = "v4_baseline") -> list[str]:
    """Print ranked results and return list of winning variant names."""
    baseline = df[df["variant"] == baseline_name].iloc[0]
    b_sh = baseline["oos_sharpe"]
    b_tot = baseline["oos_total"]
    b_mdd = baseline["oos_mdd"]

    print("\n" + "=" * 110)
    print("RANKED BY OOS SHARPE")
    print("=" * 110)
    print(f"{'variant':<35} {'ntrd':>6} {'is_sh':>7} {'oos_sh':>7} "
          f"{'oos_tot':>9} {'oos_mdd':>9} {'verdict':>22}")
    print("-" * 110)

    winners = []
    for _, row in df.iterrows():
        d_sh = row["oos_sharpe"] - b_sh
        d_tot = row["oos_total"] - b_tot
        mdd_ok = row["oos_mdd"] >= 1.25 * b_mdd
        if row["variant"] == baseline_name:
            verdict = "(baseline)"
        elif d_sh > 0 and d_tot > 0 and mdd_ok:
            verdict = f"ADD (+{d_sh:.2f} Sh)"
            winners.append(row["variant"])
        elif d_sh > 0 and d_tot > 0:
            verdict = f"reject (MDD)"
        else:
            verdict = "reject"
        print(f"{row['variant']:<35} {row['n_trades']:>6} "
              f"{row['is_sharpe']:>+7.2f} {row['oos_sharpe']:>+7.2f} "
              f"{row['oos_total']*100:>+8.2f}% {row['oos_mdd']*100:>+8.2f}% "
              f"{verdict:>22}")
    print("=" * 110)
    return winners


def build_stack_variants(winner_assets: list[str]) -> list[Variant]:
    """Phase 2: test all combinations of winners stacked on v4."""
    from itertools import combinations

    variants: list[Variant] = [
        Variant(
            name="v4_baseline",
            assets=list(V4_ASSETS),
            sides=dict(V4_SIDES),
            description="v4 production baseline",
        ),
    ]

    # Extract asset name from variant name (e.g. "v9_dogeusdt_short" → "dogeusdt")
    # Determine best side for each winner
    asset_best_side: dict[str, str] = {}
    for w in winner_assets:
        parts = w.split("_")
        asset = parts[1]
        side = parts[2]
        # If both and short both win, prefer whichever had higher Sharpe
        if asset not in asset_best_side:
            asset_best_side[asset] = side

    unique_assets = list(asset_best_side.keys())
    print(f"\nWinning assets for stacking: {unique_assets}")
    print(f"Best sides: {asset_best_side}\n")

    # Test all subsets of winners (1 to all)
    for r in range(1, len(unique_assets) + 1):
        for combo in combinations(unique_assets, r):
            name = "v9_stack_" + "_".join(c.replace("usdt", "") for c in combo)
            assets = V4_ASSETS + list(combo)
            sides = {**V4_SIDES}
            for c in combo:
                sides[c] = asset_best_side[c]
            variants.append(Variant(
                name=name,
                assets=assets,
                sides=sides,
                description=f"v4 + {'+'.join(combo)}",
            ))

    return variants


def main() -> None:
    vp.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Single-asset additions ──
    print("=" * 60)
    print("PHASE 1: Single-Asset Addition Sweep (v9)")
    print("=" * 60)

    variants = build_single_variants()
    print(f"Running {len(variants)} variants (1 baseline + {len(CANDIDATES)} × 2 sides)\n")

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

    out_csv = vp.OUT_DIR / "v9_top20_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nPhase 1 results saved: {out_csv}")

    winners = print_results(df)

    if not winners:
        print("\nNo candidate beat v4 baseline. Universe stays at 7 assets.")
        return

    print(f"\n{len(winners)} variant(s) improve OOS. Proceeding to Phase 2 stacking.\n")

    # ── Phase 2: Stack combinations ──
    print("=" * 60)
    print("PHASE 2: Stacked Combinations")
    print("=" * 60)

    stack_variants = build_stack_variants(winners)
    print(f"Running {len(stack_variants)} stacked variants\n")

    stack_results = []
    for v in stack_variants:
        try:
            stack_results.append(run_variant(v))
        except Exception as e:
            import traceback
            print(f"  ERROR on {v.name}: {e}")
            traceback.print_exc()

    if not stack_results:
        print("No stack results collected.")
        return

    sdf = pd.DataFrame(stack_results)
    sdf = sdf.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)

    stack_csv = vp.OUT_DIR / "v9_top20_stacks.csv"
    sdf.to_csv(stack_csv, index=False)
    print(f"\nPhase 2 results saved: {stack_csv}")

    print_results(sdf)

    # Summary
    best = sdf.iloc[0]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best['variant']}")
    print(f"  OOS Sharpe: {best['oos_sharpe']:+.2f}")
    print(f"  OOS Total:  {best['oos_total']*100:+.2f}%")
    print(f"  OOS MDD:    {best['oos_mdd']*100:+.2f}%")
    print(f"  Trades:     {best['n_trades']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
