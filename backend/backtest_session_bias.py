#!/usr/bin/env python3
"""
PowerOf3 / Session Bias Trading Strategies — Backtest & Walk-Forward
Vectorized implementation for speed.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.expanduser("~/Desktop/maestro/data/ohlcv")
RESULTS_DIR = os.path.expanduser("~/Desktop/maestro/data/backtest_results")
COMMISSION = 0.002
TOKENS = ["btc", "eth", "sol", "link"]
N_PERMUTATIONS = 200
N_FOLDS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(token):
    fp = os.path.join(DATA_DIR, f"binance_{token}_usdt_1h.csv")
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour_utc"] = df["timestamp"].dt.hour
    iso = df["timestamp"].dt.isocalendar()
    df["year_week"] = iso.year.astype(str) + "_" + iso.week.astype(str).str.zfill(2)
    return df


def calc_metrics(r):
    r = r.dropna()
    if len(r) < 5:
        return {"total_return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "n_trades": 0}
    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    dd = (cum / cum.cummax() - 1).min()
    return {
        "total_return": round(float(total_ret), 4),
        "sharpe": round(float(sharpe), 4),
        "max_dd": round(float(dd), 4),
        "win_rate": round(float((r > 0).mean()), 4),
        "n_trades": int((r != 0).sum()),
    }


def walk_forward(returns_series, n_folds=N_FOLDS):
    r = returns_series.sort_index()
    n = len(r)
    if n < n_folds * 2:
        return None
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds
    oos = []
    folds = []
    for fold in range(n_folds):
        s = min_train + fold * fold_size
        e = min(s + fold_size, n)
        test_r = r.iloc[s:e]
        m = calc_metrics(test_r)
        m["fold"] = fold
        folds.append(m)
        oos.extend(test_r.tolist())
    agg = calc_metrics(pd.Series(oos))
    agg["fold_details"] = folds
    return agg


def permutation_test(r, n_perms=N_PERMUTATIONS):
    r = r.dropna().values
    if len(r) < 10:
        return 1.0
    obs = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    count = sum(1 for _ in range(n_perms)
                if (lambda s: s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0)(np.random.permutation(r)) >= obs)
    return count / n_perms


def dow_breakdown(df, returns_series):
    daily = df.groupby("date")["dow"].first()
    merged = pd.DataFrame({"return": returns_series}).join(daily, how="inner")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = {}
    for dv in sorted(merged["dow"].unique()):
        s = merged[merged["dow"] == dv]["return"]
        out[names[dv]] = {"avg_return": round(float(s.mean()), 6), "count": len(s),
                          "win_rate": round(float((s > 0).mean()), 4) if len(s) > 0 else 0}
    return out


def evaluate(df, r, name):
    if len(r) < 20:
        print(f"    {name}: insufficient data ({len(r)} trades)")
        return {"error": "insufficient trades"}
    m = calc_metrics(r)
    wf = walk_forward(r)
    m["walk_forward"] = wf
    m["dow_breakdown"] = dow_breakdown(df, r)
    t, p = stats.ttest_1samp(r.dropna(), 0)
    m["p_value"] = round(float(p), 4)
    if p < 0.05:
        m["permutation_p"] = round(permutation_test(r), 4)
    sig = "✅" if p < 0.05 else "❌"
    print(f"    {name}: ret={m['total_return']:.2%} sharpe={m['sharpe']:.2f} p={p:.3f} n={m['n_trades']} {sig}")
    return m


# ============================================================
# STRATEGY 1: Monday Range Breakout (vectorized by week)
# ============================================================
def strategy_monday_range(df, buffer_pct=0.0, use_stop=False):
    # Build weekly summary
    weekly = df.groupby("year_week").apply(
        lambda w: pd.Series({
            "mon_high": w.loc[w["dow"] == 0, "high"].max(),
            "mon_low": w.loc[w["dow"] == 0, "low"].min(),
            "rest_high": w.loc[w["dow"] > 0, "high"].max(),
            "rest_low": w.loc[w["dow"] > 0, "low"].min(),
            "rest_open": w.loc[w["dow"] > 0, "open"].iloc[0] if (w["dow"] > 0).any() else np.nan,
            "rest_close": w.loc[w["dow"] > 0, "close"].iloc[-1] if (w["dow"] > 0).any() else np.nan,
            "fri_date": w["date"].iloc[-1],
        })
    ).dropna()

    weekly["entry_long"] = weekly["mon_high"] * (1 + buffer_pct)
    weekly["entry_short"] = weekly["mon_low"] * (1 - buffer_pct)

    results = []
    for idx, row in weekly.iterrows():
        el = row["entry_long"]
        es = row["entry_short"]
        if row["rest_high"] >= el:
            ep = el
            exit_p = row["rest_close"]
            if use_stop and row["rest_low"] <= row["mon_low"]:
                ret = (row["mon_low"] - ep) / ep - 2 * COMMISSION
            else:
                ret = (exit_p - ep) / ep - 2 * COMMISSION
            results.append({"date": row["fri_date"], "return": ret})
        elif row["rest_low"] <= es:
            ep = es
            exit_p = row["rest_close"]
            if use_stop and row["rest_high"] >= row["mon_high"]:
                ret = (ep - row["mon_high"]) / ep - 2 * COMMISSION
            else:
                ret = (ep - exit_p) / ep - 2 * COMMISSION
            results.append({"date": row["fri_date"], "return": ret})

    if not results:
        return pd.Series(dtype=float)
    rdf = pd.DataFrame(results)
    return rdf.groupby("date")["return"].sum()


# ============================================================
# STRATEGY 2: Session Bias (vectorized)
# ============================================================
def strategy_session_bias(df, variant="A"):
    asia = df[(df["hour_utc"] >= 0) & (df["hour_utc"] < 8)]
    london = df[(df["hour_utc"] >= 8) & (df["hour_utc"] < 14)]
    ny = df[(df["hour_utc"] >= 14) & (df["hour_utc"] < 21)]

    a_stats = asia.groupby("date").agg(a_high=("high", "max"), a_low=("low", "min"),
                                        a_open=("open", "first"), a_close=("close", "last"))
    l_stats = london.groupby("date").agg(l_high=("high", "max"), l_low=("low", "min"),
                                          l_open=("open", "first"), l_close=("close", "last"))
    n_stats = ny.groupby("date").agg(n_open=("open", "first"), n_close=("close", "last"))

    merged = a_stats.join(l_stats, how="inner")
    if variant == "C":
        merged = merged.join(n_stats, how="inner")

    if variant == "A":
        r = (merged["l_high"] - merged["a_low"]) / merged["a_low"] - 2 * COMMISSION
    elif variant == "B":
        asia_ret = (merged["a_close"] - merged["a_open"]) / merged["a_open"]
        london_ret = (merged["l_close"] - merged["l_open"]) / merged["l_open"]
        # Fade asia: if asia down → long london, if asia up → short london
        r = np.where(asia_ret < 0, london_ret, -london_ret) - 2 * COMMISSION
        r = pd.Series(r, index=merged.index)
    elif variant == "C":
        london_ret = (merged["l_close"] - merged["l_open"]) / merged["l_open"]
        ny_ret = (merged["n_close"] - merged["n_open"]) / merged["n_open"]
        r = np.where(london_ret > 0, ny_ret, -ny_ret) - 2 * COMMISSION
        r = pd.Series(r, index=merged.index)

    return r


# ============================================================
# STRATEGY 3: DOW Seasonality (vectorized)
# ============================================================
def strategy_dow_seasonality(df, lookback=90):
    daily = df.groupby("date").agg(open=("open", "first"), close=("close", "last"), dow=("dow", "first"))
    daily["ret"] = daily["close"] / daily["open"] - 1
    daily = daily.sort_index()

    results = []
    ret_vals = daily["ret"].values
    dow_vals = daily["dow"].values

    for i in range(lookback, len(daily)):
        window_ret = ret_vals[i - lookback:i]
        window_dow = dow_vals[i - lookback:i]
        today_dow = dow_vals[i]
        mask = window_dow == today_dow
        if mask.sum() == 0:
            results.append(0)
            continue
        dow_avg = window_ret[mask].mean()
        if dow_avg > 0:
            results.append(ret_vals[i] - 2 * COMMISSION)
        elif dow_avg < 0:
            results.append(-ret_vals[i] - 2 * COMMISSION)
        else:
            results.append(0)

    return pd.Series(results, index=daily.index[lookback:])


# ============================================================
# STRATEGY 4: PowerOf3 (vectorized by week)
# ============================================================
def strategy_power_of_3(df):
    weekly = df.groupby("year_week").apply(
        lambda w: pd.Series({
            "mon_high": w.loc[w["dow"] == 0, "high"].max(),
            "mon_low": w.loc[w["dow"] == 0, "low"].min(),
            "tue_high": w.loc[w["dow"] == 1, "high"].max(),
            "tue_low": w.loc[w["dow"] == 1, "low"].min(),
            "tue_close": w.loc[w["dow"] == 1, "close"].iloc[-1] if (w["dow"] == 1).any() else np.nan,
            "exit_close": w.loc[w["dow"].isin([3, 4]), "close"].iloc[-1] if w["dow"].isin([3, 4]).any() else np.nan,
            "exit_date": w.loc[w["dow"].isin([3, 4]), "date"].iloc[-1] if w["dow"].isin([3, 4]).any() else np.nan,
        })
    ).dropna()

    swept_low = weekly["tue_low"] < weekly["mon_low"]
    swept_high = weekly["tue_high"] > weekly["mon_high"]

    long_mask = swept_low & ~swept_high
    short_mask = swept_high & ~swept_low

    results = []
    for idx in weekly.index:
        row = weekly.loc[idx]
        if long_mask[idx]:
            ret = (row["exit_close"] - row["tue_close"]) / row["tue_close"] - 2 * COMMISSION
            results.append({"date": row["exit_date"], "return": ret})
        elif short_mask[idx]:
            ret = (row["tue_close"] - row["exit_close"]) / row["tue_close"] - 2 * COMMISSION
            results.append({"date": row["exit_date"], "return": ret})

    if not results:
        return pd.Series(dtype=float)
    rdf = pd.DataFrame(results)
    return rdf.set_index("date")["return"]


# ============================================================
# STRATEGY 5: Vol Compression (vectorized)
# ============================================================
def strategy_vol_compression(df, percentile_threshold=20, trailing_window=30):
    asia = df[(df["hour_utc"] >= 0) & (df["hour_utc"] < 8)]
    london = df[(df["hour_utc"] >= 8) & (df["hour_utc"] < 14)]
    ny = df[(df["hour_utc"] >= 14) & (df["hour_utc"] < 21)]

    a_range = asia.groupby("date").apply(lambda x: (x["high"].max() - x["low"].min()) / x["close"].mean())
    l_first = london.groupby("date").apply(lambda x: pd.Series({
        "l_open": x["open"].iloc[0], "l_first_close": x["close"].iloc[0]
    }) if len(x) > 0 else pd.Series({"l_open": np.nan, "l_first_close": np.nan}))
    n_close = ny.groupby("date")["close"].last()

    dates = sorted(a_range.index)
    results = []

    for i in range(trailing_window, len(dates)):
        d = dates[i]
        window = [a_range[dates[j]] for j in range(i - trailing_window, i)]
        thresh = np.percentile(window, percentile_threshold)
        if a_range[d] > thresh:
            continue
        if d not in l_first.index or d not in n_close.index:
            continue
        lf = l_first.loc[d]
        if pd.isna(lf["l_open"]):
            continue
        direction = 1 if lf["l_first_close"] > lf["l_open"] else -1
        entry = lf["l_open"]
        exit_p = n_close[d]
        ret = direction * (exit_p - entry) / entry - 2 * COMMISSION
        results.append({"date": d, "return": ret})

    if not results:
        return pd.Series(dtype=float)
    return pd.DataFrame(results).set_index("date")["return"]


# ============================================================
# MAIN
# ============================================================
def run_all():
    all_results = {}

    for token in TOKENS:
        print(f"\n{'='*60}")
        print(f"  {token.upper()}")
        print(f"{'='*60}")
        df = load_data(token)
        tr = {}

        # S1
        print("  S1: Monday Range Breakout")
        s1 = {}
        for buf in [0.0, 0.005, 0.01]:
            for stop in [False, True]:
                k = f"buf{buf}_stop{stop}"
                r = strategy_monday_range(df, buf, stop)
                s1[k] = evaluate(df, r, k)
        tr["monday_range_breakout"] = s1

        # S2
        print("  S2: Session Bias")
        s2 = {}
        for v in ["A", "B", "C"]:
            r = strategy_session_bias(df, v)
            s2[f"variant_{v}"] = evaluate(df, r, f"variant_{v}")
        tr["session_bias"] = s2

        # S3
        print("  S3: DOW Seasonality")
        s3 = {}
        for lb in [60, 90, 180]:
            r = strategy_dow_seasonality(df, lb)
            s3[f"lookback_{lb}"] = evaluate(df, r, f"lookback_{lb}")
        tr["dow_seasonality"] = s3

        # S4
        print("  S4: PowerOf3")
        r = strategy_power_of_3(df)
        tr["power_of_3"] = evaluate(df, r, "po3")

        # S5
        print("  S5: Vol Compression")
        s5 = {}
        for pct in [10, 20, 30]:
            for w in [20, 30, 60]:
                k = f"pct{pct}_w{w}"
                r = strategy_vol_compression(df, pct, w)
                s5[k] = evaluate(df, r, k)
        tr["vol_compression"] = s5

        all_results[token.upper()] = tr

    out = os.path.join(RESULTS_DIR, "session_bias_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Saved to {out}")

    # Summary
    print("\n" + "=" * 70)
    print("BEST PER TOKEN (by Sharpe)")
    print("=" * 70)
    for token, tr in all_results.items():
        print(f"\n{token}:")
        for sname, sdata in tr.items():
            if isinstance(sdata, dict) and "sharpe" in sdata:
                p = sdata.get("p_value", 1)
                sig = "✅" if p < 0.05 else ""
                print(f"  {sname}: {sdata['total_return']:.1%} sharpe={sdata['sharpe']:.2f} p={p:.3f} {sig}")
            elif isinstance(sdata, dict):
                best = max(((k, v) for k, v in sdata.items() if isinstance(v, dict) and "sharpe" in v),
                           key=lambda x: x[1]["sharpe"], default=None)
                if best:
                    k, v = best
                    p = v.get("p_value", 1)
                    sig = "✅" if p < 0.05 else ""
                    print(f"  {sname} [{k}]: {v['total_return']:.1%} sharpe={v['sharpe']:.2f} p={p:.3f} {sig}")


if __name__ == "__main__":
    run_all()
