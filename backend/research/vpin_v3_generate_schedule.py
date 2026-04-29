#!/usr/bin/env python3
"""
VPIN v3 — Trade Schedule Generator
==================================
Emits a flat CSV of trade intents for the V6 production config so that
downstream execution layers (Freqtrade strategy, fidelity simulator, etc.)
can consume it without re-running any VPIN math.

V6 config:
  - Universe: SUI, AVAX, LINK (long/short) + FET, ARB (short-only)
  - Dollar buckets sized to median(IS daily $vol) / 50 buckets/day
  - lookback = 100, hold = 100
  - ecdf window = 90 days, threshold = 0.80
  - delta_25 (vpin - vpin.shift(25)) direction must match signed_vpin
  - No funding filter (monotonically hurt the OOS — see funding sweep)

Output columns:
  asset, entry_time_utc, exit_time_utc, side, entry_px, exit_px,
  holding_days, ecdf, signed_vpin, d25, vpin
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from backend.research.vpin_v2_portfolio_variants import (
    load_asset_once,
    compute_vpin,
    signed_vpin,
    rolling_ecdf_percentile,
    ECDF_WINDOW_DAYS,
    ECDF_THRESHOLD,
    LOOKBACK,
    HOLD,
    D25_LAG,
)

# V6 production spec
V6_ASSETS = ["suiusdt", "avaxusdt", "linkusdt", "fetusdt", "arbusdt"]
V6_SIDES = {
    "suiusdt": "both",
    "avaxusdt": "both",
    "linkusdt": "both",
    "fetusdt": "short",
    "arbusdt": "short",
}

# Default output path is the research directory. The launchd regen cron
# overrides this to a location outside ~/Desktop because macOS TCC blocks
# ~/Desktop reads for launchd-spawned rsync/cp even though Python itself
# has read access.
OUT_PATH = Path(
    os.environ.get(
        "VPIN_V3_OUT_PATH",
        "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/"
        "vpin_v3_trade_schedule.csv",
    )
)


def generate_schedule_for_asset(asset: str, allowed_sides: str) -> list[dict]:
    data = load_asset_once(asset)
    buckets = data["buckets"]

    vpin = compute_vpin(buckets)
    svpin = signed_vpin(buckets)
    ecdf = rolling_ecdf_percentile(
        vpin, buckets["time_end"].values, ECDF_WINDOW_DAYS
    )
    d25 = vpin - vpin.shift(D25_LAG)

    close = buckets["close"].values
    tend = pd.to_datetime(buckets["time_end"].values)
    n = len(buckets)

    rows: list[dict] = []
    i = 0
    last_exit_idx = -1
    while i < n - HOLD:
        if i <= last_exit_idx:
            i += 1
            continue

        e = ecdf.iloc[i]
        if np.isnan(e) or e < ECDF_THRESHOLD:
            i += 1
            continue

        sv = svpin.iloc[i]
        dd = d25.iloc[i]
        vp = vpin.iloc[i]
        if np.isnan(sv) or np.isnan(dd) or np.isnan(vp):
            i += 1
            continue

        side = 0
        if sv > 0 and dd > 0:
            side = +1
        elif sv < 0 and dd < 0:
            side = -1
        if side == 0:
            i += 1
            continue
        if allowed_sides == "long" and side != 1:
            i += 1
            continue
        if allowed_sides == "short" and side != -1:
            i += 1
            continue

        entry_idx = i
        exit_idx = i + HOLD
        entry_px = close[entry_idx]
        exit_px = close[exit_idx]
        if (
            not (np.isfinite(entry_px) and np.isfinite(exit_px))
            or entry_px <= 0
            or exit_px <= 0
        ):
            i += 1
            continue

        entry_t = tend[entry_idx]
        exit_t = tend[exit_idx]
        holding_days = (exit_t - entry_t).total_seconds() / 86400.0

        rows.append({
            "asset": asset,
            "entry_time_utc": entry_t.isoformat(),
            "exit_time_utc": exit_t.isoformat(),
            "side": int(side),
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "holding_days": float(holding_days),
            "ecdf": float(e),
            "signed_vpin": float(sv),
            "d25": float(dd),
            "vpin": float(vp),
        })

        last_exit_idx = exit_idx
        i += 1

    return rows


def main() -> None:
    all_rows: list[dict] = []
    print(f"Generating V6 trade schedule for {len(V6_ASSETS)} assets...\n")
    for a in V6_ASSETS:
        sides = V6_SIDES[a]
        rows = generate_schedule_for_asset(a, sides)
        print(f"  {a:10s}  {len(rows):4d} trades  ({sides})")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("entry_time_utc").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nTotal trades: {len(df)}")
    print(f"Saved:        {OUT_PATH}")
    print(f"Span:         {df['entry_time_utc'].min()} → {df['exit_time_utc'].max()}")
    print("\nSchema:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
