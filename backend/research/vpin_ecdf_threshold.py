#!/usr/bin/env python3
"""
VPIN Gap #2 — Empirical CDF Threshold Calibration
==================================================
Replace fixed quintile thresholds with a rolling empirical-CDF percentile.
This makes the high-VPIN trigger:
  (a) regime-adaptive (adjusts to volatility cycles), and
  (b) directly comparable across assets with different VPIN distributions.

Test plan:
  - 13 assets (full v1 universe with buy/sell-dollar split)
  - VPIN dollar buckets, lb=100 (best from Gap #1)
  - Rolling 90-day ECDF percentile of VPIN
  - High-VPIN trigger: ECDF percentile >= 0.80
  - Direction split via signed_oi within window
  - Walk-forward IS=60% / OOS=40%
  - Report directional spread (buy_dom - sell_dom forward return) per asset

Output: backend/research/vpin_results/vpin_ecdf_threshold.csv
"""
import gc
import numpy as np
import pandas as pd
from scipy import stats

from backend.config.data_paths import BARS_1M_V1

ASSETS = [
    "btcusdt", "ethusdt", "solusdt", "linkusdt", "suiusdt",
    "avaxusdt", "arbusdt", "opusdt", "injusdt", "tiausdt",
    "fetusdt", "taousdt", "rndrusdt",
]
LOOKBACK = 100
FORWARDS = [25, 50, 100]
IS_FRACTION = 0.6
BUCKETS_PER_DAY_TARGET = 50
ECDF_WINDOW_DAYS = 90
HIGH_PERCENTILE = 0.80


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
    return buckets["signed_oi"].rolling(lookback, min_periods=lookback).mean()


def rolling_ecdf_percentile(vpin: pd.Series, time_end: np.ndarray, window_days: int) -> pd.Series:
    """For each VPIN value, return its percentile rank within the prior `window_days`.

    Strictly causal: the lookback window EXCLUDES the current observation.
    """
    n = len(vpin)
    perc = np.full(n, np.nan)
    vpin_arr = vpin.values
    ts_arr = pd.to_datetime(time_end).astype("datetime64[ns]").astype(np.int64)
    win_ns = window_days * 24 * 3600 * int(1e9)

    left = 0
    for i in range(n):
        if np.isnan(vpin_arr[i]):
            continue
        cutoff = ts_arr[i] - win_ns
        while left < i and ts_arr[left] < cutoff:
            left += 1
        if i - left < 100:
            continue
        window = vpin_arr[left:i]
        valid = window[~np.isnan(window)]
        if len(valid) < 100:
            continue
        # percentile rank of vpin_arr[i] within `valid`
        perc[i] = float((valid < vpin_arr[i]).sum()) / len(valid)
    return pd.Series(perc, index=vpin.index)


def forward_log_ret(buckets: pd.DataFrame, k: int) -> pd.Series:
    return np.log(buckets["close"].shift(-k) / buckets["close"])


def run_one(asset: str, results: list):
    print(f"\n{'=' * 70}\n{asset.upper()}\n{'=' * 70}")
    df = load_1m(asset)
    n = len(df)
    is_end = int(n * IS_FRACTION)
    is_df = df.iloc[:is_end]

    median_daily_dollar = is_df["dollar_volume"].resample("1D").sum().median()
    bucket_size = median_daily_dollar / BUCKETS_PER_DAY_TARGET
    if not np.isfinite(bucket_size) or bucket_size <= 0:
        print(f"  bad bucket size for {asset}, skip")
        return
    print(f"  bucket size: ${bucket_size/1e6:.1f}M  rows={n:,}  IS_end={is_end:,}")

    buckets = make_dollar_buckets(df, bucket_size)
    print(f"  total buckets: {len(buckets):,}")

    is_cutoff = df.index[is_end]
    is_mask = buckets["time_end"].values < np.datetime64(is_cutoff)
    oos_mask = ~is_mask

    vpin = compute_vpin(buckets, LOOKBACK)
    s_vpin = signed_vpin(buckets, LOOKBACK)

    print(f"  computing rolling ECDF ({ECDF_WINDOW_DAYS}d window)...")
    ecdf_pct = rolling_ecdf_percentile(vpin, buckets["time_end"].values, ECDF_WINDOW_DAYS)

    for fk in FORWARDS:
        fret = forward_log_ret(buckets, fk)

        df_eval = pd.DataFrame({
            "vpin": vpin,
            "s_vpin": s_vpin,
            "ecdf": ecdf_pct,
            "fret": fret,
            "is_oos": np.where(is_mask, "is", "oos"),
        }).dropna()

        oos = df_eval[df_eval["is_oos"] == "oos"]

        # ECDF threshold (rolling, causal)
        high_ecdf = oos[oos["ecdf"] >= HIGH_PERCENTILE]
        h_buy = high_ecdf[high_ecdf["s_vpin"] > 0]
        h_sell = high_ecdf[high_ecdf["s_vpin"] < 0]

        # Static IS quantile threshold (Gap #4 baseline)
        is_q80 = float(df_eval[df_eval["is_oos"] == "is"]["vpin"].quantile(HIGH_PERCENTILE))
        high_static = oos[oos["vpin"] >= is_q80]
        s_buy = high_static[high_static["s_vpin"] > 0]
        s_sell = high_static[high_static["s_vpin"] < 0]

        def s(sub):
            if len(sub) < 30:
                return {"n": len(sub), "mean": np.nan, "t": np.nan, "p": np.nan}
            tt, pp = stats.ttest_1samp(sub["fret"].values, 0)
            return {
                "n": len(sub),
                "mean": float(sub["fret"].mean()),
                "t": float(tt),
                "p": float(pp),
            }

        ecdf_buy = s(h_buy)
        ecdf_sell = s(h_sell)
        stat_buy = s(s_buy)
        stat_sell = s(s_sell)

        ecdf_spread = (
            ecdf_buy["mean"] - ecdf_sell["mean"]
            if not (np.isnan(ecdf_buy["mean"]) or np.isnan(ecdf_sell["mean"]))
            else np.nan
        )
        stat_spread = (
            stat_buy["mean"] - stat_sell["mean"]
            if not (np.isnan(stat_buy["mean"]) or np.isnan(stat_sell["mean"]))
            else np.nan
        )

        results.append({
            "asset": asset,
            "fwd_k": fk,
            "n_buckets": len(buckets),
            "n_oos": len(oos),
            "ecdf_n_high": len(high_ecdf),
            "ecdf_buy_n": ecdf_buy["n"],
            "ecdf_buy_mean": ecdf_buy["mean"],
            "ecdf_buy_t": ecdf_buy["t"],
            "ecdf_sell_n": ecdf_sell["n"],
            "ecdf_sell_mean": ecdf_sell["mean"],
            "ecdf_sell_t": ecdf_sell["t"],
            "ecdf_spread": ecdf_spread,
            "static_n_high": len(high_static),
            "static_spread": stat_spread,
            "is_q80_value": is_q80,
        })

        print(
            f"  fwd_k={fk:3d}: ECDF buy={ecdf_buy['mean']*100:+.3f}% (n={ecdf_buy['n']}, t={ecdf_buy['t']:+.1f}) "
            f"sell={ecdf_sell['mean']*100:+.3f}% (n={ecdf_sell['n']}, t={ecdf_sell['t']:+.1f}) "
            f"spread={ecdf_spread*100:+.3f}% | static_spread={stat_spread*100:+.3f}%"
        )

    del df, buckets
    gc.collect()


def main():
    results: list = []
    for asset in ASSETS:
        try:
            run_one(asset, results)
        except Exception as e:
            print(f"  ERROR on {asset}: {e}")

    df = pd.DataFrame(results)
    out_file = "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/vpin_ecdf_threshold.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY — ECDF threshold across 13 assets")
    print("=" * 70)
    for fk in FORWARDS:
        sub = df[df["fwd_k"] == fk]
        sig_pos = sub[(sub["ecdf_buy_t"] > 2) & (sub["ecdf_sell_t"] < -2)]
        print(f"\n  fwd_k={fk}:")
        print(f"    assets where BOTH buy_t>2 AND sell_t<-2 (consistent direction): {len(sig_pos)}/{len(sub)}")
        for _, r in sub.sort_values("ecdf_spread", ascending=False).iterrows():
            print(
                f"    {r['asset']:10s}: ECDF spread={r['ecdf_spread']*100:+.3f}%  "
                f"buy={r['ecdf_buy_mean']*100:+.3f}% (t={r['ecdf_buy_t']:+.1f}) "
                f"sell={r['ecdf_sell_mean']*100:+.3f}% (t={r['ecdf_sell_t']:+.1f})"
            )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
