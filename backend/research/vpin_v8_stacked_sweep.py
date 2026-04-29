#!/usr/bin/env python3
"""
VPIN v8 — Stacked Universe Sweep
=================================
V7 tested "V6 + one candidate" and found three additions that beat the
baseline on all three criteria (OOS Sharpe, OOS total, OOS MDD):
  1. OP short   (Sharpe +2.32 → +2.99)
  2. INJ short  (Sharpe +2.32 → +2.76)
  3. TAO short  (Sharpe +2.32 → +2.54)

V8 tests whether stacking these additions compounds the edge or
cannibalises it (correlation / weight dilution). Also re-runs each
singleton as a sanity check vs V7.

Variants:
  V6_baseline         (5 assets)
  V6 + OP_s           (6)
  V6 + INJ_s          (6)
  V6 + TAO_s          (6)
  V6 + OP_s + INJ_s   (7)  top-2 stack
  V6 + OP_s + TAO_s   (7)
  V6 + INJ_s + TAO_s  (7)
  V6 + OP_s + INJ_s + TAO_s  (8)  top-3 stack
"""
from __future__ import annotations
import pandas as pd

from backend.research import vpin_v2_portfolio_variants as vp
from backend.research.vpin_v2_portfolio_variants import Variant, run_variant

vp.FUNDING_MAP.update({
    "injusdt": "inj_funding_full.csv",
    "opusdt": "op_funding_full.csv",
    "taousdt": "tao_funding_full.csv",
})

V6_ASSETS = ["suiusdt", "avaxusdt", "linkusdt", "fetusdt", "arbusdt"]
V6_SIDES = {
    "suiusdt": "both", "avaxusdt": "both", "linkusdt": "both",
    "fetusdt": "short", "arbusdt": "short",
}


def make(name: str, extras: list[str], desc: str) -> Variant:
    return Variant(
        name=name,
        assets=V6_ASSETS + extras,
        sides={**V6_SIDES, **{e: "short" for e in extras}},
        description=desc,
    )


VARIANTS: list[Variant] = [
    Variant(name="V6_baseline", assets=list(V6_ASSETS), sides=dict(V6_SIDES),
            description="trio + FET/ARB short (production)"),
    make("V8_op",        ["opusdt"],                         "V6 + OP short"),
    make("V8_inj",       ["injusdt"],                        "V6 + INJ short"),
    make("V8_tao",       ["taousdt"],                        "V6 + TAO short"),
    make("V8_op_inj",    ["opusdt", "injusdt"],              "V6 + OP + INJ short"),
    make("V8_op_tao",    ["opusdt", "taousdt"],              "V6 + OP + TAO short"),
    make("V8_inj_tao",   ["injusdt", "taousdt"],             "V6 + INJ + TAO short"),
    make("V8_op_inj_tao",["opusdt", "injusdt", "taousdt"],   "V6 + OP + INJ + TAO short"),
]


def main() -> None:
    vp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running V8 stacked sweep: {len(VARIANTS)} variants\n")

    results = []
    for v in VARIANTS:
        try:
            results.append(run_variant(v))
        except Exception as e:
            import traceback
            print(f"  ERROR on {v.name}: {e}")
            traceback.print_exc()

    if not results:
        return

    df = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    out_csv = vp.OUT_DIR / "v8_stacked_sweep.csv"
    df.to_csv(out_csv, index=False)

    baseline = df[df["variant"] == "V6_baseline"].iloc[0]
    b_sh, b_tot, b_mdd = baseline["oos_sharpe"], baseline["oos_total"], baseline["oos_mdd"]

    print("\n" + "=" * 108)
    print("V8 STACKED SWEEP — RANKED BY OOS SHARPE")
    print("=" * 108)
    print(f"{'variant':<22} {'n_assets':>8} {'ntrd':>6} {'is_sh':>7} "
          f"{'oos_sh':>7} {'Δsh':>6} {'oos_tot':>9} {'Δtot':>8} {'oos_mdd':>9} {'verdict':>18}")
    print("-" * 108)
    for _, row in df.iterrows():
        d_sh = row["oos_sharpe"] - b_sh
        d_tot = row["oos_total"] - b_tot
        mdd_ok = row["oos_mdd"] >= 1.25 * b_mdd
        n_assets = row["assets"].count("|") + 1
        if row["variant"] == "V6_baseline":
            verdict = "(baseline)"
        elif d_sh > 0 and d_tot > 0 and mdd_ok:
            verdict = f"ADD"
        elif d_sh > 0 and d_tot > 0:
            verdict = f"MDD reject"
        else:
            verdict = "reject"
        print(f"{row['variant']:<22} {n_assets:>8} {row['n_trades']:>6} "
              f"{row['is_sharpe']:>+7.2f} {row['oos_sharpe']:>+7.2f} "
              f"{d_sh:>+6.2f} {row['oos_total']*100:>+8.2f}% "
              f"{d_tot*100:>+7.2f}% {row['oos_mdd']*100:>+8.2f}% "
              f"{verdict:>18}")
    print("=" * 108)
    print(f"\nFull CSV: {out_csv}")

    best = df.iloc[0]
    print(f"\nBest OOS Sharpe: {best['variant']} — "
          f"Sharpe {best['oos_sharpe']:+.2f} "
          f"(Δ{best['oos_sharpe']-b_sh:+.2f}), "
          f"total {best['oos_total']*100:+.2f}% "
          f"(Δ{(best['oos_total']-b_tot)*100:+.2f}%), "
          f"MDD {best['oos_mdd']*100:+.2f}%")


if __name__ == "__main__":
    main()
