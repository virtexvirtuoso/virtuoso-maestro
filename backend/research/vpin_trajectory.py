#!/usr/bin/env python3
"""
VPIN Gap #3 — Trajectory Features
==================================
Test whether the *shape* of the VPIN time-series adds information beyond
the level. Three trajectory features:
  - vpin_delta_5     : VPIN_t - VPIN_{t-5}      (short-term momentum)
  - vpin_delta_25    : VPIN_t - VPIN_{t-25}     (medium-term momentum)
  - vpin_accel       : delta_5 - delta_5_{t-5}  (acceleration)
  - vpin_ar1         : VPIN_t - phi * VPIN_{t-1} (residual after AR1 fit)

For each feature, we measure Spearman correlation with the (signed)
forward log-return at fwd=100 over OOS, conditional on:
  (a) all OOS buckets
  (b) high-VPIN regime (ECDF >= 0.80)
  (c) the buy_dom slice (ECDF>=0.80 AND signed_vpin > 0)
  (d) the sell_dom slice (ECDF>=0.80 AND signed_vpin < 0)

Goal: identify whether VPIN momentum/acceleration sharpens the directional
signal we found in Gap #4 / Gap #2.

Output: backend/research/vpin_results/vpin_trajectory.csv
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


def forward_log_ret(close: pd.Series, k: int) -> pd.Series:
    return np.log(close.shift(-k) / close)


def fit_ar1_phi(x: pd.Series, n_max: int = 200_000) -> float:
    """Fit OLS AR(1): x_t = c + phi * x_{t-1}. Return phi."""
    s = x.dropna()
    if len(s) > n_max:
        s = s.iloc[-n_max:]
    if len(s) < 100:
        return np.nan
    y = s.values[1:]
    X = s.values[:-1]
    Xm = X - X.mean()
    ym = y - y.mean()
    denom = (Xm * Xm).sum()
    if denom == 0:
        return np.nan
    return float((Xm * ym).sum() / denom)


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

    # Trajectory features (causal)
    delta_5 = vpin - vpin.shift(5)
    delta_25 = vpin - vpin.shift(25)
    accel = delta_5 - delta_5.shift(5)

    # AR1 residual: fit phi on IS only, apply causally
    phi = fit_ar1_phi(vpin[is_mask])
    if np.isnan(phi):
        phi = 0.0
    ar1_resid = vpin - phi * vpin.shift(1)
    print(f"  AR1 phi (IS): {phi:.4f}")

    fret = forward_log_ret(buckets["close"], FWD)
    signed_fret = np.sign(s_vpin) * fret  # positive = aligned with direction call

    df_eval = pd.DataFrame({
        "vpin": vpin,
        "s_vpin": s_vpin,
        "ecdf": ecdf,
        "delta_5": delta_5,
        "delta_25": delta_25,
        "accel": accel,
        "ar1_resid": ar1_resid,
        "fret": fret,
        "signed_fret": signed_fret,
        "is_oos": np.where(is_mask, "is", "oos"),
    }).dropna()

    oos = df_eval[df_eval["is_oos"] == "oos"]
    high = oos[oos["ecdf"] >= HIGH_PERCENTILE]
    buy = high[high["s_vpin"] > 0]
    sell = high[high["s_vpin"] < 0]

    feat_cols = ["delta_5", "delta_25", "accel", "ar1_resid"]

    def slice_stats(label: str, sub: pd.DataFrame, target_col: str):
        if len(sub) < 100:
            return
        for f in feat_cols:
            rho, p = stats.spearmanr(sub[f].values, sub[target_col].values)
            results.append({
                "asset": asset,
                "slice": label,
                "feature": f,
                "target": target_col,
                "n": len(sub),
                "rho": float(rho) if rho is not None else np.nan,
                "p": float(p) if p is not None else np.nan,
            })

    print("\n  Spearman ρ vs forward return (fwd=100):")
    for label, sub in [("oos_all", oos), ("high_vpin", high), ("buy_dom", buy), ("sell_dom", sell)]:
        if len(sub) < 100:
            print(f"    {label:12s} (n={len(sub)} too few)")
            continue
        for f in feat_cols:
            rho_raw, _ = stats.spearmanr(sub[f].values, sub["fret"].values)
            rho_signed, _ = stats.spearmanr(sub[f].values, sub["signed_fret"].values)
            print(
                f"    {label:12s} {f:10s} ρ_raw={rho_raw:+.3f}  "
                f"ρ_signed={rho_signed:+.3f}  n={len(sub):,}"
            )
        slice_stats(label, sub, "fret")
        slice_stats(label, sub, "signed_fret")

    # Within high-VPIN: does delta_5/accel sharpen the buy vs sell separation?
    if len(high) >= 200:
        med_d5 = high["delta_5"].median()
        rising = high[high["delta_5"] > med_d5]
        falling = high[high["delta_5"] <= med_d5]

        def s(sub):
            if len(sub) < 30:
                return {"n": len(sub), "mean": np.nan, "t": np.nan}
            tt, _ = stats.ttest_1samp(sub["fret"].values, 0)
            return {"n": len(sub), "mean": float(sub["fret"].mean()), "t": float(tt)}

        sb_rise = s(rising[rising["s_vpin"] > 0])
        ss_rise = s(rising[rising["s_vpin"] < 0])
        sb_fall = s(falling[falling["s_vpin"] > 0])
        ss_fall = s(falling[falling["s_vpin"] < 0])
        print(
            f"\n  Conditional on delta_5 (median split, fwd=100):"
        )
        print(
            f"    rising  buy={sb_rise['mean']*100:+.3f}% (n={sb_rise['n']}, t={sb_rise['t']:+.1f})  "
            f"sell={ss_rise['mean']*100:+.3f}% (n={ss_rise['n']}, t={ss_rise['t']:+.1f})  "
            f"spread={(sb_rise['mean']-ss_rise['mean'])*100:+.3f}%"
        )
        print(
            f"    falling buy={sb_fall['mean']*100:+.3f}% (n={sb_fall['n']}, t={sb_fall['t']:+.1f})  "
            f"sell={ss_fall['mean']*100:+.3f}% (n={ss_fall['n']}, t={ss_fall['t']:+.1f})  "
            f"spread={(sb_fall['mean']-ss_fall['mean'])*100:+.3f}%"
        )
        for cond_label, cond_buy, cond_sell in [
            ("rising_buy", sb_rise, None),
            ("rising_sell", ss_rise, None),
            ("falling_buy", sb_fall, None),
            ("falling_sell", ss_fall, None),
        ]:
            results.append({
                "asset": asset,
                "slice": "delta5_conditional",
                "feature": cond_label,
                "target": "fret",
                "n": cond_buy["n"],
                "rho": cond_buy["mean"],
                "p": cond_buy["t"],
            })

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
    out_file = "/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results/vpin_trajectory.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY — trajectory features")
    print("=" * 70)
    feat_cols = ["delta_5", "delta_25", "accel", "ar1_resid"]
    for slc in ["oos_all", "high_vpin", "buy_dom", "sell_dom"]:
        sub = df[(df["slice"] == slc) & (df["target"] == "fret")]
        print(f"\n  [{slc}]")
        for f in feat_cols:
            fsub = sub[sub["feature"] == f]
            if len(fsub) == 0:
                continue
            mean_rho = fsub["rho"].mean()
            n_sig = ((fsub["p"] < 0.05) & (fsub["rho"].abs() > 0.05)).sum()
            print(f"    {f:10s} mean_rho={mean_rho:+.3f}  sig_assets={n_sig}/{len(fsub)}")

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
