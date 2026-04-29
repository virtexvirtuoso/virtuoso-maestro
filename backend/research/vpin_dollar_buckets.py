#!/usr/bin/env python3
"""
VPIN Gap #1 — Dollar Buckets Fix
================================
Replace contract-volume buckets with dollar-volume buckets so cross-period
samples are economically comparable (Easley/López de Prado 2012).

Tests on BTC and ETH 1m data (now 785 days vs 30 in the original VPS study).

Methodology:
  - Bucket sizing: median daily $vol over the IS period / 50 buckets/day
  - Lookbacks: [25, 50, 100] buckets
  - Forward windows: [10, 25, 50] buckets
  - IS = first 60% of data, OOS = remainder
  - Metric: Spearman ρ between VPIN and forward realized vol (LdP target)
  - Pass criterion: OOS p < Bonferroni-corrected α and ρ direction stable
  - Baseline comparison: contract buckets at the same lookback grid

Output: backend/research/vpin_results/vpin_dollar_buckets.csv
"""
import sys
import gc
import numpy as np
import pandas as pd
from scipy import stats

from backend.config.data_paths import BARS_1M_V1

ASSETS = ["btcusdt", "ethusdt"]
LOOKBACKS = [25, 50, 100]
FORWARDS = [10, 25, 50]
IS_FRACTION = 0.6
BUCKETS_PER_DAY_TARGET = 50  # LdP guidance: ~50 buckets/day


def load_1m(asset: str) -> pd.DataFrame:
    fp = BARS_1M_V1 / f"{asset}_1m.csv"
    df = pd.read_csv(fp, parse_dates=["timestamp"], index_col="timestamp")
    return df


def make_dollar_buckets(df: pd.DataFrame, dollar_bucket_size: float) -> pd.DataFrame:
    """Aggregate 1m bars into equal-dollar-volume buckets.

    Each bucket has approximately `dollar_bucket_size` of traded notional.
    """
    cum_dollar = df["dollar_volume"].values.cumsum()
    bid = (cum_dollar // dollar_bucket_size).astype(np.int64)

    tmp = pd.DataFrame({
        "bid": bid,
        "buy_dollar": df["buy_dollar"].values,
        "sell_dollar": df["sell_dollar"].values,
        "dollar_volume": df["dollar_volume"].values,
        "close": df["close"].values,
        "vwap_num": df["dollar_volume"].values,  # close-weighted proxy
        "ts": df.index.values,
    })

    g = tmp.groupby("bid", sort=True)
    buckets = pd.DataFrame({
        "buy_dollar": g["buy_dollar"].sum().values,
        "sell_dollar": g["sell_dollar"].sum().values,
        "total_dollar": g["dollar_volume"].sum().values,
        "close": g["close"].last().values,
        "time_start": g["ts"].first().values,
        "time_end": g["ts"].last().values,
    })

    # VWAP via dollar-weighted close: ∑(close * vol) / ∑(vol)
    # We don't have vol per bucket directly, use last close as proxy.
    # For VPIN we only need the order imbalance — directionality from $ flow.
    buckets["oi"] = (
        np.abs(buckets["buy_dollar"] - buckets["sell_dollar"]) / buckets["total_dollar"]
    )
    return buckets


def make_contract_buckets(df: pd.DataFrame, contract_bucket_size: float) -> pd.DataFrame:
    """Original LdP-style buckets in contracts (BASELINE for comparison)."""
    cumvol = df["volume"].values.cumsum()
    bid = (cumvol // contract_bucket_size).astype(np.int64)

    tmp = pd.DataFrame({
        "bid": bid,
        "buy_vol": df["buy_vol"].values,
        "sell_vol": df["sell_vol"].values,
        "total_vol": df["volume"].values,
        "close": df["close"].values,
        "ts": df.index.values,
    })
    g = tmp.groupby("bid", sort=True)
    buckets = pd.DataFrame({
        "buy_vol": g["buy_vol"].sum().values,
        "sell_vol": g["sell_vol"].sum().values,
        "total_vol": g["total_vol"].sum().values,
        "close": g["close"].last().values,
        "time_start": g["ts"].first().values,
        "time_end": g["ts"].last().values,
    })
    buckets["oi"] = (
        np.abs(buckets["buy_vol"] - buckets["sell_vol"]) / buckets["total_vol"]
    )
    return buckets


def compute_vpin(buckets: pd.DataFrame, lookback: int) -> pd.Series:
    return buckets["oi"].rolling(lookback, min_periods=lookback).mean()


def forward_vol(buckets: pd.DataFrame, k: int) -> pd.Series:
    """Forward realized vol over next k buckets (LdP target)."""
    log_ret = np.log(buckets["close"] / buckets["close"].shift(1))
    return log_ret.rolling(k, min_periods=k).std().shift(-k)


def evaluate(vpin: pd.Series, target: pd.Series) -> dict:
    valid = pd.DataFrame({"v": vpin, "t": target}).dropna()
    if len(valid) < 100:
        return {"n": len(valid), "rho": np.nan, "p": np.nan, "ratio": np.nan}
    rho, p = stats.spearmanr(valid["v"], valid["t"])
    q20 = valid["v"].quantile(0.2)
    q80 = valid["v"].quantile(0.8)
    low = valid.loc[valid["v"] <= q20, "t"].mean()
    high = valid.loc[valid["v"] >= q80, "t"].mean()
    ratio = high / low if low and not np.isnan(low) else np.nan
    return {
        "n": len(valid),
        "rho": round(rho, 4),
        "p": p,
        "ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
    }


def run_one_asset(asset: str, results: list):
    print(f"\n{'=' * 70}\n{asset.upper()}\n{'=' * 70}")
    df = load_1m(asset)
    n = len(df)
    is_end = int(n * IS_FRACTION)

    # Bucket sizing on IS only — no look-ahead
    is_df = df.iloc[:is_end]
    daily_dollar = is_df["dollar_volume"].resample("1D").sum()
    median_daily_dollar = float(daily_dollar.median())
    dollar_bucket_size = median_daily_dollar / BUCKETS_PER_DAY_TARGET

    daily_vol = is_df["volume"].resample("1D").sum()
    median_daily_vol = float(daily_vol.median())
    contract_bucket_size = median_daily_vol / BUCKETS_PER_DAY_TARGET

    print(f"  rows: {n:,}  IS: {is_end:,}  OOS: {n - is_end:,}")
    print(f"  median daily $vol (IS): ${median_daily_dollar/1e9:.2f}B")
    print(f"  dollar bucket size: ${dollar_bucket_size/1e6:.1f}M")
    print(f"  median daily contracts (IS): {median_daily_vol:,.0f}")
    print(f"  contract bucket size: {contract_bucket_size:,.0f}")

    # Build buckets on full series; we'll mask IS/OOS later by time
    print("  building dollar buckets...")
    dbuckets = make_dollar_buckets(df, dollar_bucket_size)
    print(f"    -> {len(dbuckets):,} dollar buckets")
    print("  building contract buckets...")
    cbuckets = make_contract_buckets(df, contract_bucket_size)
    print(f"    -> {len(cbuckets):,} contract buckets")

    is_cutoff_ts = df.index[is_end]
    d_is_mask = dbuckets["time_end"].values < np.datetime64(is_cutoff_ts)
    c_is_mask = cbuckets["time_end"].values < np.datetime64(is_cutoff_ts)

    for scheme, buckets, is_mask in [
        ("dollar", dbuckets, d_is_mask),
        ("contract", cbuckets, c_is_mask),
    ]:
        for lb in LOOKBACKS:
            vpin = compute_vpin(buckets, lb)
            for fk in FORWARDS:
                fvol = forward_vol(buckets, fk)
                is_eval = evaluate(vpin[is_mask], fvol[is_mask])
                oos_eval = evaluate(vpin[~is_mask], fvol[~is_mask])
                row = {
                    "asset": asset,
                    "scheme": scheme,
                    "lookback": lb,
                    "fwd_k": fk,
                    "n_buckets": len(buckets),
                    "is_n": is_eval["n"],
                    "is_rho": is_eval["rho"],
                    "is_p": is_eval["p"],
                    "is_q80q20_ratio": is_eval["ratio"],
                    "oos_n": oos_eval["n"],
                    "oos_rho": oos_eval["rho"],
                    "oos_p": oos_eval["p"],
                    "oos_q80q20_ratio": oos_eval["ratio"],
                }
                results.append(row)
                tag = "  "
                if (
                    oos_eval["p"] is not None
                    and not np.isnan(oos_eval["p"])
                    and oos_eval["p"] < 0.05
                ):
                    tag = "✅" if oos_eval["rho"] > 0 else "⚠️"
                print(
                    f"  {tag} {scheme:8s} lb={lb:3d} fwd={fk:3d}: "
                    f"IS ρ={is_eval['rho']:+.3f} (p={is_eval['p']:.2e}) "
                    f"OOS ρ={oos_eval['rho']:+.3f} (p={oos_eval['p']:.2e}) "
                    f"q80/q20={oos_eval['ratio']}"
                )

    del df, dbuckets, cbuckets
    gc.collect()


def main():
    results: list = []
    for asset in ASSETS:
        run_one_asset(asset, results)

    df = pd.DataFrame(results)
    out_dir = (
        sys.modules["__main__"].__file__
        if False
        else "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results"
    )
    out_file = f"{out_dir}/vpin_dollar_buckets.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY — Dollar vs Contract bucket VPIN (BTC + ETH)")
    print("=" * 70)
    n = len(df)
    bonf = 0.05 / n
    print(f"Total tests: {n}, Bonferroni α: {bonf:.5f}")

    for scheme in ["dollar", "contract"]:
        sub = df[df["scheme"] == scheme]
        sig = sub[sub["oos_p"] < 0.05]
        bonf_pass = sub[sub["oos_p"] < bonf]
        pos = sig[sig["oos_rho"] > 0]
        neg = sig[sig["oos_rho"] < 0]
        print(
            f"\n  [{scheme}] sig@p<.05: {len(sig)}/{len(sub)}, "
            f"Bonferroni: {len(bonf_pass)}, "
            f"pos ρ: {len(pos)}, neg ρ: {len(neg)}"
        )
        for asset in ASSETS:
            asub = sub[sub["asset"] == asset]
            print(
                f"    {asset}: mean OOS ρ = {asub['oos_rho'].mean():+.4f}, "
                f"median q80/q20 = {asub['oos_q80q20_ratio'].median():.2f}"
            )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
