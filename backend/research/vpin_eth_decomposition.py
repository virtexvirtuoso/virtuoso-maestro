#!/usr/bin/env python3
"""
VPIN Gap #4 — ETH Direction Decomposition
==========================================
Hypothesis (gap analysis): ETH's negative VPIN-vol correlation is driven
by *seller-dominated* high-VPIN buckets (delta-neutral hedging flow from
option market makers / structured product desks).

Test: decompose high-VPIN buckets into buy-dominant vs sell-dominant and
look at forward return + forward vol conditional on the direction.

Output: backend/research/vpin_results/vpin_eth_decomposition.csv
"""
import gc
import numpy as np
import pandas as pd
from scipy import stats

from backend.config.data_paths import BARS_1M_V1

ASSETS = ["btcusdt", "ethusdt"]   # BTC as control
LOOKBACK = 100                    # strongest signal from Gap #1
FORWARDS = [10, 25, 50, 100]
IS_FRACTION = 0.6
BUCKETS_PER_DAY_TARGET = 50
HIGH_VPIN_QUANTILE = 0.80


def load_1m(asset: str) -> pd.DataFrame:
    return pd.read_csv(
        BARS_1M_V1 / f"{asset}_1m.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def make_dollar_buckets(df: pd.DataFrame, dollar_bucket_size: float) -> pd.DataFrame:
    cum_dollar = df["dollar_volume"].values.cumsum()
    bid = (cum_dollar // dollar_bucket_size).astype(np.int64)
    tmp = pd.DataFrame({
        "bid": bid,
        "buy_dollar": df["buy_dollar"].values,
        "sell_dollar": df["sell_dollar"].values,
        "dollar_volume": df["dollar_volume"].values,
        "close": df["close"].values,
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
    buckets["oi"] = (
        np.abs(buckets["buy_dollar"] - buckets["sell_dollar"]) / buckets["total_dollar"]
    )
    buckets["signed_oi"] = (
        (buckets["buy_dollar"] - buckets["sell_dollar"]) / buckets["total_dollar"]
    )
    return buckets


def compute_vpin(buckets: pd.DataFrame, lookback: int) -> pd.Series:
    return buckets["oi"].rolling(lookback, min_periods=lookback).mean()


def signed_vpin(buckets: pd.DataFrame, lookback: int) -> pd.Series:
    """Signed VPIN — positive = buy-dominant on average over the window."""
    return buckets["signed_oi"].rolling(lookback, min_periods=lookback).mean()


def forward_log_ret(buckets: pd.DataFrame, k: int) -> pd.Series:
    return np.log(buckets["close"].shift(-k) / buckets["close"])


def forward_vol(buckets: pd.DataFrame, k: int) -> pd.Series:
    log_ret = np.log(buckets["close"] / buckets["close"].shift(1))
    return log_ret.rolling(k, min_periods=k).std().shift(-k)


def run_one(asset: str, results: list):
    print(f"\n{'=' * 70}\n{asset.upper()} — direction decomposition\n{'=' * 70}")
    df = load_1m(asset)
    n = len(df)
    is_end = int(n * IS_FRACTION)
    is_df = df.iloc[:is_end]

    median_daily_dollar = is_df["dollar_volume"].resample("1D").sum().median()
    bucket_size = median_daily_dollar / BUCKETS_PER_DAY_TARGET
    print(f"  bucket size: ${bucket_size/1e6:.1f}M")

    buckets = make_dollar_buckets(df, bucket_size)
    print(f"  total buckets: {len(buckets):,}")

    is_cutoff = df.index[is_end]
    is_mask = buckets["time_end"].values < np.datetime64(is_cutoff)
    oos_mask = ~is_mask

    vpin = compute_vpin(buckets, LOOKBACK)
    s_vpin = signed_vpin(buckets, LOOKBACK)

    # Calibrate VPIN threshold on IS only
    is_vpin_valid = vpin[is_mask].dropna()
    high_thresh = float(is_vpin_valid.quantile(HIGH_VPIN_QUANTILE))
    print(f"  IS VPIN q{int(HIGH_VPIN_QUANTILE*100)}: {high_thresh:.4f}")

    for fk in FORWARDS:
        fret = forward_log_ret(buckets, fk)
        fvol = forward_vol(buckets, fk)

        df_eval = pd.DataFrame({
            "vpin": vpin,
            "s_vpin": s_vpin,
            "fret": fret,
            "fvol": fvol,
            "is_oos": np.where(is_mask, "is", "oos"),
        }).dropna()

        oos = df_eval[df_eval["is_oos"] == "oos"]
        high = oos[oos["vpin"] >= high_thresh]
        low_q = oos[oos["vpin"] < oos["vpin"].quantile(0.20)]

        # Direction split inside high-VPIN regime
        high_buy = high[high["s_vpin"] > 0]
        high_sell = high[high["s_vpin"] < 0]

        def stat_block(sub: pd.DataFrame) -> dict:
            if len(sub) < 50:
                return {"n": len(sub), "mean_ret": np.nan, "median_ret": np.nan,
                        "mean_vol": np.nan, "t_ret": np.nan, "p_ret": np.nan}
            t_stat, p_val = stats.ttest_1samp(sub["fret"].values, 0)
            return {
                "n": len(sub),
                "mean_ret": float(sub["fret"].mean()),
                "median_ret": float(sub["fret"].median()),
                "mean_vol": float(sub["fvol"].mean()),
                "t_ret": float(t_stat),
                "p_ret": float(p_val),
            }

        rows = [
            ("oos_all", oos),
            ("oos_low_vpin_q20", low_q),
            ("oos_high_vpin_q80", high),
            ("oos_high_vpin_buy_dom", high_buy),
            ("oos_high_vpin_sell_dom", high_sell),
        ]
        print(f"\n  fwd_k={fk}:")
        for label, sub in rows:
            s = stat_block(sub)
            results.append({
                "asset": asset,
                "fwd_k": fk,
                "regime": label,
                **s,
            })
            if not np.isnan(s["mean_ret"]):
                print(
                    f"    {label:28s} n={s['n']:6d} "
                    f"mean_ret={s['mean_ret']*100:+.3f}% "
                    f"mean_vol={s['mean_vol']*100:.3f}% "
                    f"t={s['t_ret']:+.2f} (p={s['p_ret']:.3f})"
                )
            else:
                print(f"    {label:28s} n={s['n']:6d}  (insufficient)")

    del df, buckets
    gc.collect()


def main():
    results: list = []
    for asset in ASSETS:
        run_one(asset, results)

    df = pd.DataFrame(results)
    out_file = "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/vpin_eth_decomposition.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY — does seller-dominated high-VPIN explain ETH's drag?")
    print("=" * 70)
    for asset in ASSETS:
        sub = df[df["asset"] == asset]
        print(f"\n  {asset.upper()}")
        for fk in FORWARDS:
            fk_sub = sub[sub["fwd_k"] == fk]
            buy = fk_sub[fk_sub["regime"] == "oos_high_vpin_buy_dom"].iloc[0]
            sell = fk_sub[fk_sub["regime"] == "oos_high_vpin_sell_dom"].iloc[0]
            allh = fk_sub[fk_sub["regime"] == "oos_high_vpin_q80"].iloc[0]
            spread = (
                buy["mean_ret"] - sell["mean_ret"]
                if not (np.isnan(buy["mean_ret"]) or np.isnan(sell["mean_ret"]))
                else np.nan
            )
            print(
                f"    fwd_k={fk:3d}: high_vpin_all={allh['mean_ret']*100:+.3f}% "
                f"buy_dom={buy['mean_ret']*100:+.3f}% (n={int(buy['n'])}) "
                f"sell_dom={sell['mean_ret']*100:+.3f}% (n={int(sell['n'])}) "
                f"spread={spread*100:+.3f}%"
            )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
