#!/usr/bin/env python3
"""
VPIN Study — Volume-Synchronized Probability of Informed Trading
================================================================
30 days of Bybit tick data → pre-aggregate to 1-min bars → VPIN computation.

Memory-efficient: processes one day at a time, aggregates to 1-min bars,
then computes VPIN on the much smaller bar dataset.

Walk-forward: IS=20 days, OOS=10 days.
Tests: VPIN predicts forward volatility (Spearman correlation) + directional accuracy.

Run on VPS: source ~/tick_collector/venv/bin/activate && python3 vpin_study.py
"""

import os, sys, glob, gc
import numpy as np
import pandas as pd
from scipy import stats

from backend.config.data_paths import TICK_TRADES, TICK_ORDERBOOK
DATA_DIR = str(TICK_TRADES)
OB_DIR = str(TICK_ORDERBOOK)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT"]

# Volume bucket sizes (in contracts) — calibrated per symbol
BUCKET_CONFIGS = {
    "BTCUSDT":  [100, 500, 2000],
    "ETHUSDT":  [1000, 5000, 20000],
    "SOLUSDT":  [10000, 50000, 200000],
    "LINKUSDT": [20000, 100000, 500000],
}
LOOKBACK_WINDOWS = [25, 50, 100]
FORWARD_WINDOWS  = [10, 25, 50]
IS_DAYS = 20


def aggregate_to_1min(symbol):
    """Load tick data day-by-day, aggregate to 1-min bars with buy/sell volume."""
    files = sorted(glob.glob(f"{DATA_DIR}/{symbol}/*.parquet"))
    if not files:
        return None
    
    all_bars = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "price", "size", "side"])
        
        # 1-minute bins
        df["minute"] = (df["timestamp"] // 60000) * 60000
        is_buy = df["side"] == "Buy"
        
        bars = df.groupby("minute").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            n_trades=("size", "count"),
        )
        bars["buy_vol"] = df[is_buy].groupby("minute")["size"].sum()
        bars["sell_vol"] = df[~is_buy].groupby("minute")["size"].sum()
        bars = bars.fillna(0)
        bars["buy_vol"] = bars["buy_vol"].astype(float)
        bars["sell_vol"] = bars["sell_vol"].astype(float)
        
        all_bars.append(bars)
        del df
        gc.collect()
    
    result = pd.concat(all_bars).sort_index()
    # Handle duplicate minutes (day boundary)
    result = result.groupby(level=0).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "n_trades": "sum", "buy_vol": "sum", "sell_vol": "sum",
    })
    
    day = files[0].split("/")[-1].replace(".parquet", "")
    dayN = files[-1].split("/")[-1].replace(".parquet", "")
    print(f"  {symbol}: {len(result):,} 1-min bars ({day} → {dayN})")
    return result


def make_volume_buckets(bars_1m, bucket_size):
    """Convert 1-min bars into equal-volume buckets."""
    cumvol = bars_1m["volume"].values.cumsum()
    bucket_id = (cumvol // bucket_size).astype(np.int64)
    
    tmp = pd.DataFrame({
        "bid": bucket_id,
        "buy_vol": bars_1m["buy_vol"].values,
        "sell_vol": bars_1m["sell_vol"].values,
        "total_vol": bars_1m["volume"].values,
        "dollar_vol": bars_1m["volume"].values * bars_1m["close"].values,
        "close": bars_1m["close"].values,
        "ts": bars_1m.index.values,
    })
    
    g = tmp.groupby("bid", sort=True)
    buckets = pd.DataFrame({
        "buy_vol": g["buy_vol"].sum().values,
        "sell_vol": g["sell_vol"].sum().values,
        "total_vol": g["total_vol"].sum().values,
        "vwap": (g["dollar_vol"].sum() / g["total_vol"].sum()).values,
        "close": g["close"].last().values,
        "time_start": g["ts"].first().values,
        "time_end": g["ts"].last().values,
    })
    
    buckets["oi"] = np.abs(buckets["buy_vol"] - buckets["sell_vol"]) / buckets["total_vol"]
    return buckets


def compute_vpin(buckets, lookback):
    """VPIN = rolling mean of order imbalance."""
    return buckets["oi"].rolling(lookback, min_periods=lookback).mean()


def compute_forward_vol(buckets, k):
    """Forward realized vol = std of log returns over next K buckets."""
    log_ret = np.log(buckets["vwap"] / buckets["vwap"].shift(1))
    return log_ret.rolling(k, min_periods=k).std().shift(-k)


def compute_forward_ret(buckets, k):
    """Forward return over K buckets."""
    return buckets["vwap"].shift(-k) / buckets["vwap"] - 1


def evaluate(vpin, target, label=""):
    """Spearman correlation + quintile analysis."""
    valid = pd.DataFrame({"vpin": vpin, "target": target}).dropna()
    if len(valid) < 50:
        return {"n": len(valid), "spearman": np.nan, "p": np.nan, "ratio": np.nan}
    
    corr, pval = stats.spearmanr(valid["vpin"], valid["target"])
    q20 = valid["vpin"].quantile(0.2)
    q80 = valid["vpin"].quantile(0.8)
    low = valid.loc[valid["vpin"] <= q20, "target"].mean()
    high = valid.loc[valid["vpin"] >= q80, "target"].mean()
    ratio = high / low if low != 0 else np.nan
    
    return {"n": len(valid), "spearman": round(corr, 4), "p": pval, "ratio": round(ratio, 3)}


def run():
    print("=" * 70)
    print("VPIN STUDY — Walk-Forward Validation (1-min pre-aggregation)")
    print(f"IS: {IS_DAYS} days, OOS: remaining")
    print("=" * 70)
    
    results = []
    
    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"{symbol}")
        print(f"{'='*50}")
        
        bars = aggregate_to_1min(symbol)
        if bars is None:
            continue
        
        # Split IS/OOS by date
        bars["date"] = pd.to_datetime(bars.index, unit="ms").date
        dates = sorted(bars["date"].unique())
        is_cutoff = dates[IS_DAYS - 1]
        
        for bsize in BUCKET_CONFIGS[symbol]:
            buckets = make_volume_buckets(bars, bsize)
            n_buckets = len(buckets)
            avg_dur = (buckets["time_end"] - buckets["time_start"]).mean() / 60000
            print(f"\n  bucket={bsize}: {n_buckets:,} buckets (avg {avg_dur:.1f} min)")
            
            buckets["date"] = pd.to_datetime(buckets["time_start"], unit="ms").dt.date
            is_mask = buckets["date"] <= is_cutoff
            
            for lb in LOOKBACK_WINDOWS:
                vpin = compute_vpin(buckets, lb)
                
                for fk in FORWARD_WINDOWS:
                    fwd_vol = compute_forward_vol(buckets, fk)
                    fwd_ret = compute_forward_ret(buckets, fk)
                    
                    # IS
                    is_vol = evaluate(vpin[is_mask], fwd_vol[is_mask])
                    # OOS
                    oos_vol = evaluate(vpin[~is_mask], fwd_vol[~is_mask])
                    
                    # Directional: high VPIN + net buy → price up?
                    net_buy = buckets["buy_vol"] - buckets["sell_vol"]
                    oos_mask = ~is_mask & vpin.notna() & fwd_ret.notna()
                    if oos_mask.sum() > 50:
                        q80 = vpin[oos_mask].quantile(0.8)
                        high_vpin = oos_mask & (vpin >= q80)
                        if high_vpin.sum() > 10:
                            correct = ((net_buy[high_vpin] > 0) & (fwd_ret[high_vpin] > 0)) | \
                                      ((net_buy[high_vpin] < 0) & (fwd_ret[high_vpin] < 0))
                            dir_acc = correct.mean()
                        else:
                            dir_acc = np.nan
                    else:
                        dir_acc = np.nan
                    
                    row = {
                        "symbol": symbol, "bucket": bsize, "lookback": lb, "fwd_k": fk,
                        "is_n": is_vol["n"], "is_rho": is_vol["spearman"], "is_p": is_vol["p"],
                        "is_vol_ratio": is_vol["ratio"],
                        "oos_n": oos_vol["n"], "oos_rho": oos_vol["spearman"], "oos_p": oos_vol["p"],
                        "oos_vol_ratio": oos_vol["ratio"],
                        "oos_dir_acc": round(dir_acc, 4) if not np.isnan(dir_acc) else np.nan,
                    }
                    results.append(row)
                    
                    if oos_vol["p"] is not None and not np.isnan(oos_vol["p"]) and oos_vol["p"] < 0.05:
                        tag = "✅" if oos_vol["spearman"] > 0.05 else "⚠️"
                        print(f"    {tag} lb={lb} fwd={fk}: IS ρ={is_vol['spearman']:.3f}(p={is_vol['p']:.4f}) "
                              f"OOS ρ={oos_vol['spearman']:.3f}(p={oos_vol['p']:.4f}) "
                              f"vol_ratio={oos_vol['ratio']:.2f} dir={dir_acc:.1%}" if not np.isnan(dir_acc) else
                              f"    {tag} lb={lb} fwd={fk}: IS ρ={is_vol['spearman']:.3f} "
                              f"OOS ρ={oos_vol['spearman']:.3f}(p={oos_vol['p']:.4f})")
            
            del buckets
            gc.collect()
        
        del bars
        gc.collect()
    
    # === SUMMARY ===
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n = len(df)
    valid = df.dropna(subset=["oos_rho"])
    n_sig = (valid["oos_p"] < 0.05).sum()
    n_pos = ((valid["oos_p"] < 0.05) & (valid["oos_rho"] > 0)).sum()
    bonf = 0.05 / n
    n_bonf = (valid["oos_p"] < bonf).sum()
    
    print(f"Total tests: {n}")
    print(f"OOS significant (p<0.05): {n_sig} ({100*n_sig/n:.1f}%)")
    print(f"OOS significant + positive ρ: {n_pos} ({100*n_pos/n:.1f}%)")
    print(f"Bonferroni (p<{bonf:.5f}): {n_bonf}")
    
    if n_pos > 0:
        print("\nTop OOS results (positive ρ, sorted by p-value):")
        top = valid[(valid["oos_p"] < 0.05) & (valid["oos_rho"] > 0)].nsmallest(15, "oos_p")
        for _, r in top.iterrows():
            print(f"  {r['symbol']} bkt={int(r['bucket'])} lb={int(r['lookback'])} fwd={int(r['fwd_k'])}: "
                  f"IS ρ={r['is_rho']:.3f} OOS ρ={r['oos_rho']:.3f} (p={r['oos_p']:.2e}) "
                  f"vol_ratio={r['oos_vol_ratio']:.2f} dir_acc={r['oos_dir_acc']:.1%}" 
                  if not np.isnan(r['oos_dir_acc']) else
                  f"  {r['symbol']} bkt={int(r['bucket'])} lb={int(r['lookback'])} fwd={int(r['fwd_k'])}: "
                  f"IS ρ={r['is_rho']:.3f} OOS ρ={r['oos_rho']:.3f} (p={r['oos_p']:.2e})")
    
    # Directional summary
    dir_valid = df.dropna(subset=["oos_dir_acc"])
    if len(dir_valid) > 0:
        print(f"\nDirectional accuracy (informed trading signal):")
        print(f"  Mean: {dir_valid['oos_dir_acc'].mean():.1%}")
        print(f"  > 55%: {(dir_valid['oos_dir_acc'] > 0.55).sum()}/{len(dir_valid)}")
        print(f"  > 60%: {(dir_valid['oos_dir_acc'] > 0.60).sum()}/{len(dir_valid)}")
    
    # Per-symbol summary
    print(f"\nPer-symbol OOS ρ (mean of positive-sig tests):")
    for sym in SYMBOLS:
        s = valid[(valid["symbol"] == sym) & (valid["oos_p"] < 0.05) & (valid["oos_rho"] > 0)]
        if len(s) > 0:
            print(f"  {sym}: {len(s)} sig tests, mean ρ={s['oos_rho'].mean():.3f}")
        else:
            print(f"  {sym}: 0 significant tests")
    
    out = os.path.expanduser("~/Desktop/maestro/backend/research/vpin_results/vpin_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run()
