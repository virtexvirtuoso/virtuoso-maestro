#!/usr/bin/env python3
"""
VPIN v2 — Portfolio Variants Sweep
===================================
Run the v2 signal through multiple portfolio configurations to address
the findings from the baseline backtest:
  - BTC/ETH are net-negative OOS despite largest weights
  - SUI/AVAX/LINK carry the entire OOS edge
  - 12 bps round-trip costs eat majors' edges

Variants tested:
  V1  baseline          — 5 bullish-tier, both directions (reference)
  V2  alt-trio          — SUI/AVAX/LINK only, both directions
  V3  alt-trio+funding  — V2 + high-|funding| filter (>|5|bps/day)
  V4  alt-trio+5assets  — V2 with FET/ARB added as short-only
  V5  alt-trio+long     — V2 long only (skip shorts)
  V6  alt-trio+short    — V2 short only (skip longs)

For each variant, report:
  - IS/OOS Sharpe, total return, max drawdown
  - Trade count, avg holding days, turnover
  - Per-asset OOS contribution
"""
from __future__ import annotations
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from backend.config.data_paths import BARS_1M_V1

# ---------- config ----------
FUNDING_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/derivatives")
FUNDING_MAP = {
    "btcusdt": "btc_funding_full.csv",
    "ethusdt": "eth_funding_full.csv",
    "suiusdt": "sui_funding_full.csv",
    "avaxusdt": "avax_funding_full.csv",
    "linkusdt": "link_funding_full.csv",
    "fetusdt": "fet_funding_full.csv",
    "arbusdt": "arb_funding_full.csv",
}

LOOKBACK = 100
HOLD = 100
BUCKETS_PER_DAY_TARGET = 50
ECDF_WINDOW_DAYS = 90
ECDF_THRESHOLD = 0.80
D25_LAG = 25
IS_FRACTION = 0.6
FEE_PER_SIDE = 0.0005
SLIP_PER_SIDE = 0.0001

OUT_DIR = Path("/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results")

Side = Literal["both", "long", "short"]


@dataclass
class Variant:
    name: str
    assets: list[str]
    sides: dict[str, Side]             # per-asset allowed sides
    funding_min_abs: float = 0.0       # minimum |daily funding rate| to trade
    description: str = ""


@dataclass
class Trade:
    asset: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_px: float
    exit_px: float
    raw_log_ret: float
    holding_days: float
    funding_cost: float
    fee_cost: float
    net_log_ret: float
    funding_at_entry: float = 0.0


# ---------- shared helpers (same math as baseline) ----------
def load_1m(asset: str) -> pd.DataFrame:
    return pd.read_csv(
        BARS_1M_V1 / f"{asset}_1m.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def load_funding(asset: str) -> pd.Series:
    fp = FUNDING_DIR / FUNDING_MAP.get(asset, "")
    if not fp.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df["funding_rate"].astype(float) / 100.0


def make_dollar_buckets(df: pd.DataFrame, bucket_size: float) -> pd.DataFrame:
    cum_dollar = df["dollar_volume"].values.cumsum()
    bid = (cum_dollar // bucket_size).astype(np.int64)
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
        "time_end": g["ts"].last().values,
    })
    buckets["oi"] = (
        np.abs(buckets["buy_dollar"] - buckets["sell_dollar"]) / buckets["total_dollar"]
    )
    buckets["signed_oi"] = (
        (buckets["buy_dollar"] - buckets["sell_dollar"]) / buckets["total_dollar"]
    )
    return buckets


def compute_vpin(buckets: pd.DataFrame) -> pd.Series:
    return buckets["oi"].rolling(LOOKBACK, min_periods=LOOKBACK).mean()


def signed_vpin(buckets: pd.DataFrame) -> pd.Series:
    return buckets["signed_oi"].rolling(LOOKBACK, min_periods=LOOKBACK).mean()


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


def compute_asset_vol(buckets: pd.DataFrame, is_mask: np.ndarray) -> float:
    close = buckets["close"].values
    log_ret = np.log(close[1:] / close[:-1])
    is_ret = log_ret[is_mask[1:]]
    if len(is_ret) < 100:
        return np.nan
    return float(np.std(is_ret) * np.sqrt(BUCKETS_PER_DAY_TARGET * 365))


def generate_trades(
    asset: str,
    buckets: pd.DataFrame,
    funding: pd.Series,
    allowed_sides: Side,
    funding_min_abs: float,
) -> list[Trade]:
    vpin = compute_vpin(buckets)
    svpin = signed_vpin(buckets)
    ecdf = rolling_ecdf_percentile(vpin, buckets["time_end"].values, ECDF_WINDOW_DAYS)
    d25 = vpin - vpin.shift(D25_LAG)

    close = buckets["close"].values
    tend = pd.to_datetime(buckets["time_end"].values)
    n = len(buckets)

    trades: list[Trade] = []
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
        if np.isnan(sv) or np.isnan(dd):
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
        if not (np.isfinite(entry_px) and np.isfinite(exit_px)) or entry_px <= 0 or exit_px <= 0:
            i += 1
            continue

        entry_t = tend[entry_idx]
        exit_t = tend[exit_idx]
        holding_days = (exit_t - entry_t).total_seconds() / 86400.0

        # funding at entry day (for filter) + avg over hold (for cost)
        funding_at_entry = 0.0
        funding_cost = 0.0
        if len(funding) > 0:
            entry_day = entry_t.normalize()
            if entry_day in funding.index:
                funding_at_entry = float(funding.loc[entry_day])
            else:
                # nearest prior day
                prior = funding.loc[:entry_day]
                if len(prior) > 0:
                    funding_at_entry = float(prior.iloc[-1])

            mask = (funding.index >= entry_t.normalize()) & (
                funding.index <= exit_t.normalize()
            )
            rel = funding.loc[mask]
            if len(rel) > 0:
                avg_daily = float(rel.mean())
                funding_cost = side * avg_daily * holding_days

        # funding filter (applied to entry-day funding)
        if funding_min_abs > 0 and abs(funding_at_entry) < funding_min_abs:
            i += 1
            continue

        raw_log_ret = side * np.log(exit_px / entry_px)
        fee_cost = 2 * (FEE_PER_SIDE + SLIP_PER_SIDE)
        net_log_ret = raw_log_ret - funding_cost - fee_cost

        trades.append(Trade(
            asset=asset,
            entry_time=entry_t,
            exit_time=exit_t,
            side=side,
            entry_px=float(entry_px),
            exit_px=float(exit_px),
            raw_log_ret=float(raw_log_ret),
            holding_days=float(holding_days),
            funding_cost=float(funding_cost),
            fee_cost=float(fee_cost),
            net_log_ret=float(net_log_ret),
            funding_at_entry=funding_at_entry,
        ))

        last_exit_idx = exit_idx
        i += 1

    return trades


def trades_to_daily_returns(
    trades: list[Trade], weight: float, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    ret = pd.Series(0.0, index=days)
    for t in trades:
        if t.holding_days <= 0:
            continue
        daily_log = t.net_log_ret / t.holding_days
        trade_days = pd.date_range(
            t.entry_time.normalize(), t.exit_time.normalize(), freq="D"
        )
        for d in trade_days:
            day_start = d
            day_end = d + pd.Timedelta(days=1)
            overlap = (min(t.exit_time, day_end) - max(t.entry_time, day_start))
            frac = max(overlap.total_seconds() / 86400.0, 0.0)
            if frac <= 0:
                continue
            if d in ret.index:
                ret.loc[d] += weight * daily_log * frac
    return ret


def metrics(returns: pd.Series) -> dict:
    if len(returns) == 0 or returns.std() == 0:
        return {"n_days": len(returns), "sharpe": np.nan, "mean_bps": np.nan,
                "std_bps": np.nan, "mdd": np.nan, "total_ret": np.nan}
    mu = returns.mean()
    sd = returns.std()
    sharpe = (mu / sd) * np.sqrt(365) if sd > 0 else np.nan
    equity = returns.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return {
        "n_days": len(returns),
        "sharpe": float(sharpe),
        "mean_bps": float(mu * 10000),
        "std_bps": float(sd * 10000),
        "mdd": float(dd.min()),
        "total_ret": float(equity.iloc[-1]),
    }


# ---------- asset data cache (avoid re-loading CSVs for each variant) ----------
ASSET_CACHE: dict[str, dict] = {}


def load_asset_once(asset: str) -> dict:
    if asset in ASSET_CACHE:
        return ASSET_CACHE[asset]
    print(f"  loading {asset}...")
    df = load_1m(asset)
    funding = load_funding(asset)
    n = len(df)
    is_end = int(n * IS_FRACTION)
    is_df = df.iloc[:is_end]
    median_daily_dollar = is_df["dollar_volume"].resample("1D").sum().median()
    bucket_size = median_daily_dollar / BUCKETS_PER_DAY_TARGET
    buckets = make_dollar_buckets(df, bucket_size)
    is_cutoff = df.index[is_end]
    is_mask = buckets["time_end"].values < np.datetime64(is_cutoff)
    vol = compute_asset_vol(buckets, is_mask)
    tend = pd.to_datetime(buckets["time_end"].values)
    ASSET_CACHE[asset] = {
        "buckets": buckets,
        "funding": funding,
        "vol": vol,
        "span": (tend[0], tend[-1]),
    }
    del df
    gc.collect()
    return ASSET_CACHE[asset]


# ---------- variant runner ----------
def run_variant(v: Variant) -> dict:
    print(f"\n{'=' * 70}\n{v.name}: {v.description}\n{'=' * 70}")

    # ensure all assets loaded
    asset_data = {a: load_asset_once(a) for a in v.assets}

    # generate trades per asset
    all_trades: dict[str, list[Trade]] = {}
    for a in v.assets:
        ad = asset_data[a]
        side = v.sides.get(a, "both")
        trades = generate_trades(
            a, ad["buckets"], ad["funding"],
            allowed_sides=side, funding_min_abs=v.funding_min_abs,
        )
        all_trades[a] = trades
        if trades:
            raw = np.mean([t.raw_log_ret for t in trades])
            net = np.mean([t.net_log_ret for t in trades])
            longs = sum(1 for t in trades if t.side == +1)
            shorts = sum(1 for t in trades if t.side == -1)
            print(
                f"  {a:10s}  trades={len(trades):4d}  "
                f"(L={longs} S={shorts})  "
                f"raw={raw*100:+.3f}% net={net*100:+.3f}%  "
                f"sides={side}"
            )
        else:
            print(f"  {a:10s}  NO TRADES  sides={side}")

    # equal-risk weights
    inv_vol = {a: (1.0 / asset_data[a]["vol"]) if (asset_data[a]["vol"] and np.isfinite(asset_data[a]["vol"]) and asset_data[a]["vol"] > 0) else 0.0 for a in v.assets}
    tot_iv = sum(inv_vol.values())
    weights = {a: (inv_vol[a] / tot_iv if tot_iv > 0 else 0.0) for a in v.assets}

    min_start = min(asset_data[a]["span"][0] for a in v.assets)
    max_end = max(asset_data[a]["span"][1] for a in v.assets)

    basket = pd.Series(0.0, index=pd.date_range(
        min_start.normalize(), max_end.normalize(), freq="D"
    ))
    per_asset_daily: dict[str, pd.Series] = {}
    for a in v.assets:
        ret = trades_to_daily_returns(all_trades[a], weights[a], min_start, max_end)
        if len(ret) == 0:
            continue
        ret = ret.reindex(basket.index, fill_value=0.0)
        per_asset_daily[a] = ret
        basket = basket + ret

    cutoff = min_start + (max_end - min_start) * IS_FRACTION
    is_ret = basket[basket.index < cutoff]
    oos_ret = basket[basket.index >= cutoff]

    is_m = metrics(is_ret)
    oos_m = metrics(oos_ret)
    full_m = metrics(basket)

    print(f"\n  BASKET:")
    print(f"    FULL n={full_m['n_days']:4d}d  Sharpe={full_m['sharpe']:+.2f}  "
          f"total={full_m['total_ret']*100:+.2f}%  MDD={full_m['mdd']*100:+.2f}%")
    print(f"    IS   n={is_m['n_days']:4d}d  Sharpe={is_m['sharpe']:+.2f}  "
          f"total={is_m['total_ret']*100:+.2f}%  MDD={is_m['mdd']*100:+.2f}%")
    print(f"    OOS  n={oos_m['n_days']:4d}d  Sharpe={oos_m['sharpe']:+.2f}  "
          f"total={oos_m['total_ret']*100:+.2f}%  MDD={oos_m['mdd']*100:+.2f}%")

    # per-asset OOS
    print("  per-asset OOS:")
    for a in v.assets:
        if a not in per_asset_daily:
            continue
        r = per_asset_daily[a][per_asset_daily[a].index >= cutoff]
        m = metrics(r)
        print(
            f"    {a:10s}  w={weights[a]*100:4.1f}%  Sharpe={m['sharpe']:+.2f}  "
            f"total={m['total_ret']*100:+.2f}%  MDD={m['mdd']*100:+.2f}%"
        )

    total_trades = sum(len(t) for t in all_trades.values())
    span_years = (max_end - min_start).total_seconds() / (365 * 86400)
    turnover_trades_per_year = total_trades / span_years if span_years > 0 else np.nan

    return {
        "variant": v.name,
        "description": v.description,
        "assets": "|".join(v.assets),
        "n_trades": total_trades,
        "trades_per_year": turnover_trades_per_year,
        "is_sharpe": is_m["sharpe"],
        "is_total": is_m["total_ret"],
        "is_mdd": is_m["mdd"],
        "oos_sharpe": oos_m["sharpe"],
        "oos_total": oos_m["total_ret"],
        "oos_mdd": oos_m["mdd"],
        "full_sharpe": full_m["sharpe"],
        "full_total": full_m["total_ret"],
        "full_mdd": full_m["mdd"],
    }


# ---------- variant definitions ----------
VARIANTS: list[Variant] = [
    Variant(
        name="V1_baseline",
        assets=["btcusdt", "ethusdt", "suiusdt", "avaxusdt", "linkusdt"],
        sides={a: "both" for a in ["btcusdt", "ethusdt", "suiusdt", "avaxusdt", "linkusdt"]},
        description="5 bullish-tier, both directions (reference)",
    ),
    Variant(
        name="V2_alt_trio",
        assets=["suiusdt", "avaxusdt", "linkusdt"],
        sides={a: "both" for a in ["suiusdt", "avaxusdt", "linkusdt"]},
        description="SUI/AVAX/LINK only, both directions",
    ),
    Variant(
        name="V3_alt_trio_long",
        assets=["suiusdt", "avaxusdt", "linkusdt"],
        sides={a: "long" for a in ["suiusdt", "avaxusdt", "linkusdt"]},
        description="Alt trio, long only",
    ),
    Variant(
        name="V4_alt_trio_short",
        assets=["suiusdt", "avaxusdt", "linkusdt"],
        sides={a: "short" for a in ["suiusdt", "avaxusdt", "linkusdt"]},
        description="Alt trio, short only",
    ),
    Variant(
        name="V5_alt_trio_funding",
        assets=["suiusdt", "avaxusdt", "linkusdt"],
        sides={a: "both" for a in ["suiusdt", "avaxusdt", "linkusdt"]},
        funding_min_abs=0.0005,  # 5 bps/day
        description="Alt trio, require |funding|>5bps/day at entry",
    ),
    Variant(
        name="V6_alt_trio_plus_fetarb_short",
        assets=["suiusdt", "avaxusdt", "linkusdt", "fetusdt", "arbusdt"],
        sides={
            "suiusdt": "both", "avaxusdt": "both", "linkusdt": "both",
            "fetusdt": "short", "arbusdt": "short",
        },
        description="Alt trio + FET/ARB short-only",
    ),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for v in VARIANTS:
        try:
            results.append(run_variant(v))
        except Exception as e:
            import traceback
            print(f"  ERROR on {v.name}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "vpin_v2_portfolio_variants.csv", index=False)

    print("\n" + "=" * 90)
    print("VARIANT SWEEP SUMMARY")
    print("=" * 90)
    print(f"{'variant':28s} {'n_tr':>5s} {'tr/yr':>6s} "
          f"{'IS_Sh':>6s} {'OOS_Sh':>7s} {'OOS_tot':>8s} {'OOS_MDD':>8s}")
    for r in results:
        print(
            f"{r['variant']:28s} "
            f"{r['n_trades']:5d} "
            f"{r['trades_per_year']:6.0f} "
            f"{r['is_sharpe']:+6.2f} "
            f"{r['oos_sharpe']:+7.2f} "
            f"{r['oos_total']*100:+7.2f}% "
            f"{r['oos_mdd']*100:+7.2f}%"
        )

    print(f"\nSaved: {OUT_DIR}/vpin_v2_portfolio_variants.csv")


if __name__ == "__main__":
    main()
