#!/usr/bin/env python3
"""
VPIN Gap #6 — Lagged VPIN Signal
=================================
Test whether the VPIN directional signal is predictive at EXECUTION lags
(i.e., how long after VPIN-high can we still enter and capture the edge?).

Rationale: our Gap #4 / Gap #2 results assume instantaneous entry at the
same bucket close. In practice we need a latency budget: the order has to
route, fills may be partial, and we may want to use minute-level timing
instead of bucket-level.

Test plan:
  - 13 assets, dollar buckets, lb=100
  - Signal: ECDF percentile >= 0.80 combined with sign(signed VPIN)
  - LAGS (in buckets): [0, 1, 2, 5, 10, 20]
    * bucket ≈ 1/50 of a day ≈ 28.8 min, so lags map roughly to
      [0, 28m, 58m, 2.4h, 4.8h, 9.6h]
  - Forward window: 100 buckets (from bucket i+lag to i+lag+100)
  - Evaluate spread (buy_dom - sell_dom) under each lag
  - OOS only (last 40%)

Output: backend/research/vpin_results/vpin_lagged_signal.csv
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
FWD = 100
LAGS = [0, 1, 2, 5, 10, 20]
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
        perc[i] = float((valid < vpin_arr[i]).sum()) / len(valid)
    return pd.Series(perc, index=vpin.index)


def forward_log_ret_from(close: pd.Series, entry_offset: int, window: int) -> pd.Series:
    """Forward log-return from `entry_offset` buckets ahead, held for `window` buckets.

    For bucket i: ln( close[i+entry_offset+window] / close[i+entry_offset] )
    """
    entry_price = close.shift(-entry_offset)
    exit_price = close.shift(-(entry_offset + window))
    return np.log(exit_price / entry_price)


def run_one(asset: str, results: list):
    print(f"\n{'=' * 70}\n{asset.upper()}\n{'=' * 70}")
    df = load_1m(asset)
    n = len(df)
    is_end = int(n * IS_FRACTION)
    is_df = df.iloc[:is_end]

    median_daily_dollar = is_df["dollar_volume"].resample("1D").sum().median()
    bucket_size = median_daily_dollar / BUCKETS_PER_DAY_TARGET
    if not np.isfinite(bucket_size) or bucket_size <= 0:
        print(f"  bad bucket size, skip")
        return

    buckets = make_dollar_buckets(df, bucket_size)
    print(f"  buckets: {len(buckets):,}")

    is_cutoff = df.index[is_end]
    is_mask = buckets["time_end"].values < np.datetime64(is_cutoff)

    vpin = compute_vpin(buckets, LOOKBACK)
    s_vpin = signed_vpin(buckets, LOOKBACK)
    ecdf = rolling_ecdf_percentile(vpin, buckets["time_end"].values, ECDF_WINDOW_DAYS)

    for lag in LAGS:
        fret = forward_log_ret_from(buckets["close"], entry_offset=lag, window=FWD)
        df_eval = pd.DataFrame({
            "vpin": vpin,
            "s_vpin": s_vpin,
            "ecdf": ecdf,
            "fret": fret,
            "is_oos": np.where(is_mask, "is", "oos"),
        }).dropna()

        oos = df_eval[df_eval["is_oos"] == "oos"]
        high = oos[oos["ecdf"] >= HIGH_PERCENTILE]
        buy = high[high["s_vpin"] > 0]
        sell = high[high["s_vpin"] < 0]

        def s(sub):
            if len(sub) < 30:
                return {"n": len(sub), "mean": np.nan, "t": np.nan}
            tt, _ = stats.ttest_1samp(sub["fret"].values, 0)
            return {"n": len(sub), "mean": float(sub["fret"].mean()), "t": float(tt)}

        b = s(buy)
        se = s(sell)
        spread = (
            b["mean"] - se["mean"]
            if not (np.isnan(b["mean"]) or np.isnan(se["mean"]))
            else np.nan
        )

        results.append({
            "asset": asset,
            "lag": lag,
            "buy_n": b["n"],
            "buy_mean": b["mean"],
            "buy_t": b["t"],
            "sell_n": se["n"],
            "sell_mean": se["mean"],
            "sell_t": se["t"],
            "spread": spread,
        })

        print(
            f"  lag={lag:3d}: buy={b['mean']*100:+.3f}% (n={b['n']}, t={b['t']:+.1f}) "
            f"sell={se['mean']*100:+.3f}% (n={se['n']}, t={se['t']:+.1f}) "
            f"spread={spread*100:+.3f}%"
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
    out_file = "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/vpin_lagged_signal.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY — signal decay vs entry lag")
    print("=" * 70)
    for lag in LAGS:
        sub = df[df["lag"] == lag]
        mean_spread = sub["spread"].mean()
        median_spread = sub["spread"].median()
        n_positive = (sub["spread"] > 0).sum()
        n_signif = ((sub["buy_t"] > 2) & (sub["sell_t"] < -2)).sum()
        print(
            f"  lag={lag:3d}: mean_spread={mean_spread*100:+.3f}%  "
            f"median={median_spread*100:+.3f}%  "
            f"pos={n_positive}/{len(sub)}  "
            f"both_sig={n_signif}/{len(sub)}"
        )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
