#!/usr/bin/env python3
"""
VPIN v2 — Portfolio Walk-Forward Backtest
==========================================
Take the v2 spec from `VPIN-v2-Results-2026-04-10.md` and run it as a
realistic cost-aware portfolio backtest on the 5 bullish-tier assets.

Signal (per asset, per dollar bucket):
  vpin  = rolling mean of |buy$ − sell$| / total$  (lookback=100)
  svpin = rolling mean of (buy$ − sell$) / total$  (lookback=100)
  ecdf  = 90-day causal percentile rank of vpin
  d25   = vpin_t − vpin_{t-25}

  long  iff ecdf>=0.80 and svpin>0 and d25>0
  short iff ecdf>=0.80 and svpin<0 and d25<0

Trade:
  entry  = close of signal bucket
  exit   = close of bucket signal_idx + HOLD (100)
  no overlap: if a new signal fires while a position is open, skip
  one position per asset (5 max simultaneous)

Costs:
  Taker fee: 5 bps per side → 10 bps round-trip
  Slippage:  1 bps per side → 2 bps round-trip
  Funding:   interpolated from daily funding rate, accrued over holding
             duration. Long pays when positive, short receives.

Sizing:
  Equal-risk: weight = target_risk / asset_IS_vol_annualized
  Capital normalization: weights sum to 1 across the 5 assets
  Position size per trade = weight × notional

Walk-forward:
  IS_FRACTION=0.6 used only for sizing calibration (per-asset IS vol) and
  AR1 gate is not needed (we use raw delta_25 sign). The ECDF window is
  already rolling/causal, so there's no IS/OOS leakage in the signal
  itself — the split is purely for reporting (IS/OOS Sharpe).

Output:
  backend/research/vpin_results/vpin_v2_portfolio_trades.csv
  backend/research/vpin_results/vpin_v2_portfolio_equity.csv
  stdout: IS/OOS metrics per asset + basket
"""
from __future__ import annotations
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config.data_paths import BARS_1M_V1

# ---------- config ----------
ASSETS = ["btcusdt", "ethusdt", "suiusdt", "avaxusdt", "linkusdt"]
FUNDING_DIR = Path("/Users/ffv_macmini/Desktop/maestro/data/derivatives")
FUNDING_MAP = {
    "btcusdt": "btc_funding_full.csv",
    "ethusdt": "eth_funding_full.csv",
    "suiusdt": "sui_funding_full.csv",
    "avaxusdt": "avax_funding_full.csv",
    "linkusdt": "link_funding_full.csv",
}

LOOKBACK = 100
HOLD = 100                 # exit 100 buckets after entry
BUCKETS_PER_DAY_TARGET = 50
ECDF_WINDOW_DAYS = 90
ECDF_THRESHOLD = 0.80
D25_LAG = 25
IS_FRACTION = 0.6

# Costs (decimal)
FEE_PER_SIDE = 0.0005      # 5 bps taker
SLIP_PER_SIDE = 0.0001     # 1 bps

OUT_DIR = Path("/Users/ffv_macmini/Desktop/maestro/backend/research/vpin_results")


# ---------- helpers ----------
def load_1m(asset: str) -> pd.DataFrame:
    return pd.read_csv(
        BARS_1M_V1 / f"{asset}_1m.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def load_funding(asset: str) -> pd.Series:
    """Return daily funding rate as decimal per day (e.g. 0.0001 = 1bp/day)."""
    fp = FUNDING_DIR / FUNDING_MAP[asset]
    if not fp.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # The CSV stores funding in percent (0.01 means 0.01% per day).
    # Convert to decimal.
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


# ---------- signal → trades ----------
@dataclass
class Trade:
    asset: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int                   # +1 long, -1 short
    entry_px: float
    exit_px: float
    raw_log_ret: float          # signed, before costs
    holding_days: float
    funding_cost: float         # decimal, signed (cost to P&L)
    fee_cost: float             # decimal, positive = drag
    net_log_ret: float          # after all costs


def generate_trades(asset: str, buckets: pd.DataFrame, funding: pd.Series) -> list[Trade]:
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
        # no overlap
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

        raw_log_ret = side * np.log(exit_px / entry_px)

        # funding: sum the daily funding rate × fraction-of-day overlap
        # rough trapezoid: pick the days covered by [entry_t, exit_t]
        funding_cost = 0.0
        if len(funding) > 0:
            mask = (funding.index >= entry_t.normalize()) & (
                funding.index <= exit_t.normalize()
            )
            rel = funding.loc[mask]
            if len(rel) > 0:
                # avg daily funding × holding_days, long pays positive funding
                avg_daily = float(rel.mean())
                funding_cost = side * avg_daily * holding_days

        fee_cost = 2 * (FEE_PER_SIDE + SLIP_PER_SIDE)  # round-trip, positive drag
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
        ))

        last_exit_idx = exit_idx
        i += 1

    return trades


# ---------- portfolio construction ----------
def compute_asset_vol(buckets: pd.DataFrame, is_mask: np.ndarray) -> float:
    """Annualized realized vol of bucket-to-bucket log returns on IS."""
    close = buckets["close"].values
    log_ret = np.log(close[1:] / close[:-1])
    is_ret = log_ret[is_mask[1:]]
    if len(is_ret) < 100:
        return np.nan
    # buckets/day ≈ 50; trading days/year ≈ 365 for crypto
    return float(np.std(is_ret) * np.sqrt(BUCKETS_PER_DAY_TARGET * 365))


def trades_to_daily_returns(
    trades: list[Trade], weight: float, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Convert trade list into a daily return series, attributing each trade's
    net log return linearly over its holding period."""
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
        # per-day portion: fraction of the day the trade was live
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


def metrics(returns: pd.Series, label: str) -> dict:
    if len(returns) == 0 or returns.std() == 0:
        return {"label": label, "n_days": len(returns), "sharpe": np.nan,
                "mean_bps": np.nan, "std_bps": np.nan, "mdd": np.nan,
                "total_ret": np.nan}
    mu = returns.mean()
    sd = returns.std()
    sharpe = (mu / sd) * np.sqrt(365) if sd > 0 else np.nan
    equity = returns.cumsum()  # log-space
    peak = equity.cummax()
    dd = equity - peak
    mdd = float(dd.min())
    return {
        "label": label,
        "n_days": len(returns),
        "sharpe": float(sharpe),
        "mean_bps": float(mu * 10000),
        "std_bps": float(sd * 10000),
        "mdd": mdd,
        "total_ret": float(equity.iloc[-1]),
    }


# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    asset_trades: dict[str, list[Trade]] = {}
    asset_vol: dict[str, float] = {}
    asset_span: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    for asset in ASSETS:
        print(f"\n{'=' * 70}\n{asset.upper()}\n{'=' * 70}")
        df = load_1m(asset)
        funding = load_funding(asset)
        print(f"  rows: {len(df):,}   funding rows: {len(funding)}")

        n = len(df)
        is_end = int(n * IS_FRACTION)
        is_df = df.iloc[:is_end]

        median_daily_dollar = is_df["dollar_volume"].resample("1D").sum().median()
        bucket_size = median_daily_dollar / BUCKETS_PER_DAY_TARGET
        if not np.isfinite(bucket_size) or bucket_size <= 0:
            print(f"  bad bucket size, skip")
            continue

        buckets = make_dollar_buckets(df, bucket_size)
        is_cutoff = df.index[is_end]
        is_mask = buckets["time_end"].values < np.datetime64(is_cutoff)
        print(f"  buckets: {len(buckets):,}   IS buckets: {int(is_mask.sum()):,}")

        vol = compute_asset_vol(buckets, is_mask)
        asset_vol[asset] = vol
        print(f"  IS annualized vol: {vol*100:.1f}%")

        trades = generate_trades(asset, buckets, funding)
        print(f"  trades generated: {len(trades)}")
        asset_trades[asset] = trades

        tend = pd.to_datetime(buckets["time_end"].values)
        asset_span[asset] = (tend[0], tend[-1])

        # per-asset report (unit notional)
        if trades:
            net = np.array([t.net_log_ret for t in trades])
            raw = np.array([t.raw_log_ret for t in trades])
            print(f"  raw mean: {raw.mean()*100:+.3f}%  "
                  f"net mean: {net.mean()*100:+.3f}%  "
                  f"net win%: {(net > 0).mean()*100:.1f}%")

        del df, buckets
        gc.collect()

    # equal-risk weights: w_i ∝ 1 / vol_i, normalized to sum=1
    inv_vol = {a: (1.0 / v) if (v and np.isfinite(v) and v > 0) else 0.0
               for a, v in asset_vol.items()}
    tot = sum(inv_vol.values())
    weights = {a: (inv_vol[a] / tot if tot > 0 else 0.0) for a in ASSETS}
    print("\n" + "=" * 70)
    print("Equal-risk weights:")
    for a in ASSETS:
        print(f"  {a:10s}  vol={asset_vol.get(a, np.nan)*100:5.1f}%   w={weights[a]*100:5.1f}%")

    # portfolio daily return
    min_start = min(s[0] for s in asset_span.values())
    max_end = max(s[1] for s in asset_span.values())

    basket = pd.Series(0.0, index=pd.date_range(
        min_start.normalize(), max_end.normalize(), freq="D"
    ))
    per_asset_daily: dict[str, pd.Series] = {}
    for a in ASSETS:
        ret = trades_to_daily_returns(asset_trades[a], weights[a], min_start, max_end)
        if len(ret) == 0:
            continue
        ret = ret.reindex(basket.index, fill_value=0.0)
        per_asset_daily[a] = ret
        basket = basket + ret

    # split IS / OOS on date
    cutoff = min_start + (max_end - min_start) * IS_FRACTION
    is_ret = basket[basket.index < cutoff]
    oos_ret = basket[basket.index >= cutoff]

    print("\n" + "=" * 70)
    print("BASKET METRICS")
    print("=" * 70)
    for label, r in [("FULL", basket), ("IS", is_ret), ("OOS", oos_ret)]:
        m = metrics(r, label)
        print(
            f"  {label:4s}  n={m['n_days']:4d}d  "
            f"Sharpe={m['sharpe']:+.2f}  "
            f"mean={m['mean_bps']:+6.2f}bps/d  "
            f"std={m['std_bps']:5.2f}bps/d  "
            f"MDD={m['mdd']*100:+.2f}%  "
            f"total={m['total_ret']*100:+.2f}%"
        )

    print("\nPer-asset OOS contribution:")
    for a in ASSETS:
        if a not in per_asset_daily:
            continue
        r = per_asset_daily[a]
        r_oos = r[r.index >= cutoff]
        if len(r_oos) == 0:
            continue
        m = metrics(r_oos, f"OOS {a}")
        print(
            f"  {a:10s}  Sharpe={m['sharpe']:+.2f}  "
            f"mean={m['mean_bps']:+6.2f}bps/d  "
            f"MDD={m['mdd']*100:+.2f}%  "
            f"total={m['total_ret']*100:+.2f}%"
        )

    # save trades and equity curve
    trade_rows = []
    for a, trs in asset_trades.items():
        for t in trs:
            trade_rows.append({
                "asset": t.asset,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "raw_log_ret": t.raw_log_ret,
                "holding_days": t.holding_days,
                "funding_cost": t.funding_cost,
                "fee_cost": t.fee_cost,
                "net_log_ret": t.net_log_ret,
            })
    tdf = pd.DataFrame(trade_rows)
    tdf.to_csv(OUT_DIR / "vpin_v2_portfolio_trades.csv", index=False)

    eq = pd.DataFrame({"basket": basket})
    for a, r in per_asset_daily.items():
        eq[a] = r
    eq["equity"] = eq["basket"].cumsum()
    eq.to_csv(OUT_DIR / "vpin_v2_portfolio_equity.csv")

    print(f"\nSaved: {OUT_DIR}/vpin_v2_portfolio_trades.csv")
    print(f"Saved: {OUT_DIR}/vpin_v2_portfolio_equity.csv")


if __name__ == "__main__":
    main()
