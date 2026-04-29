"""
Download Binance aggTrades → rich 1m orderflow bars + keep raw files.

V2: Adds microstructure features. Raw files kept in maestro-data/raw/binance_aggtrades/{SYMBOL}/.
Output 1m bars go to maestro-data/bars/1m/v2/.
Fully vectorized aggregation for speed (~1-2s per day).
"""
import os
import zipfile
import argparse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from backend.config.data_paths import BARS_1M_V2, raw_aggtrades_dir

DATA_DIR = BARS_1M_V2  # output dir for v2 1m CSVs


def download_day(symbol: str, date: datetime) -> Path | None:
    date_str = date.strftime("%Y-%m-%d")
    fname = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{fname}"
    sym_dir = raw_aggtrades_dir(symbol)
    sym_dir.mkdir(parents=True, exist_ok=True)
    out_path = sym_dir / fname
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path
    except Exception as e:
        print(f"  ⚠ {date_str}: {e}")
        return None


def read_raw(zip_path: Path) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(z.namelist()[0]) as f:
                first = f.readline().decode().strip()
                f.seek(0)
                has_header = not first[0].isdigit()
                cols = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "is_buyer_maker"]
                df = pd.read_csv(f, header=0 if has_header else None, names=None if has_header else cols)
                if has_header:
                    df.columns = cols
    except Exception as e:
        print(f"  ⚠ Read fail: {e}")
        return None
    if len(df) == 0:
        return None
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].str.strip().str.lower() == "true"
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    ts = df["timestamp"].iloc[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us" if ts > 1e15 else "ms")
    return df


def aggregate_rich(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized rich aggregation of raw aggTrades to 1m bars."""
    df = df.copy()
    df["is_taker_buy"] = ~df["is_buyer_maker"]
    df["dollar_vol"] = df["price"] * df["qty"]
    df["buy_vol"] = df["qty"] * df["is_taker_buy"]
    df["sell_vol"] = df["qty"] * df["is_buyer_maker"]
    df["buy_dollar"] = df["dollar_vol"] * df["is_taker_buy"]
    df["sell_dollar"] = df["dollar_vol"] * df["is_buyer_maker"]
    
    # Pre-compute large trade flag using per-minute mean
    df.set_index("timestamp", inplace=True)
    minute_mean = df.groupby(pd.Grouper(freq="1min"))["qty"].transform("mean")
    df["is_large"] = df["qty"] > minute_mean * 2
    df["large_qty"] = df["qty"] * df["is_large"]
    df["large_buy"] = df["qty"] * df["is_large"] * df["is_taker_buy"]
    df["large_sell"] = df["qty"] * df["is_large"] * df["is_buyer_maker"]
    
    # Cluster detection: transitions from buy→sell or sell→buy
    df["side_change"] = df["is_buyer_maker"] != df["is_buyer_maker"].shift(1)
    df["buy_start"] = df["is_taker_buy"] & df["side_change"]
    df["sell_start"] = df["is_buyer_maker"] & df["side_change"]
    
    # Standard + microstructure agg
    bars = df.resample("1min").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        avg_trade_size=("qty", "mean"),
        max_trade_size=("qty", "max"),
        median_trade_size=("qty", "median"),
        trade_count=("qty", "count"),
        dollar_volume=("dollar_vol", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        buy_dollar=("buy_dollar", "sum"),
        sell_dollar=("sell_dollar", "sum"),
        large_trade_count=("is_large", "sum"),
        large_trade_vol=("large_qty", "sum"),
        large_buy_vol=("large_buy", "sum"),
        large_sell_vol=("large_sell", "sum"),
        buy_cluster_count=("buy_start", "sum"),
        sell_cluster_count=("sell_start", "sum"),
    )
    
    bars = bars.dropna(subset=["open"])
    if len(bars) == 0:
        return bars
    
    # Derived
    bars["delta"] = bars["buy_vol"] - bars["sell_vol"]
    bars["delta_dollar"] = bars["buy_dollar"] - bars["sell_dollar"]
    bars["buy_pct"] = bars["buy_vol"] / bars["volume"].clip(lower=1e-10)
    bars["price_impact"] = (bars["close"] - bars["open"]) / bars["open"].clip(lower=1e-10)
    bars["large_delta"] = bars["large_buy_vol"] - bars["large_sell_vol"]
    bars["large_pct_of_vol"] = bars["large_trade_vol"] / bars["volume"].clip(lower=1e-10)
    
    # Trade arrival (trades per second) — via resample to 1s then aggregate
    tps = df.resample("1s").size()
    bars["trades_per_sec_avg"] = tps.resample("1min").mean().reindex(bars.index)
    bars["trades_per_sec_max"] = tps.resample("1min").max().reindex(bars.index)
    
    return bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-02-24")
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sym_raw_dir = raw_aggtrades_dir(args.asset)
    sym_raw_dir.mkdir(parents=True, exist_ok=True)

    out_file = DATA_DIR / f"{args.asset.lower()}_1m_v2.csv"
    
    existing_dates = set()
    all_bars = []
    if out_file.exists():
        existing = pd.read_csv(out_file, parse_dates=["timestamp"], index_col="timestamp")
        existing_dates = set(existing.index.date)
        all_bars.append(existing)
        print(f"Resuming: {len(existing_dates)} days already processed")
    
    total_days = (end - start).days + 1
    downloaded = 0
    errors = 0
    
    current = start
    while current <= end:
        if current.date() in existing_dates:
            current += timedelta(days=1)
            continue
        
        day_num = (current - start).days + 1
        print(f"[{day_num}/{total_days}] {current.strftime('%Y-%m-%d')} ", end="", flush=True)
        
        zip_path = download_day(args.asset, current)
        if zip_path is None:
            errors += 1
            current += timedelta(days=1)
            continue
        
        fsize = zip_path.stat().st_size / 1e6
        print(f"({fsize:.1f}MB) ", end="", flush=True)
        
        df_raw = read_raw(zip_path)
        if df_raw is None:
            print("→ empty")
            errors += 1
            current += timedelta(days=1)
            continue
        
        n_trades = len(df_raw)
        bars = aggregate_rich(df_raw)
        
        if bars is not None and len(bars) > 0:
            all_bars.append(bars)
            downloaded += 1
            lg = bars["large_trade_count"].sum()
            print(f"→ {len(bars)} bars, {n_trades:,} trades, Δ={bars['delta'].sum():.0f}, lg={lg:.0f}")
        else:
            print("→ empty")
            errors += 1
        
        if downloaded > 0 and downloaded % 15 == 0:
            combined = pd.concat(all_bars).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.to_csv(out_file, index_label="timestamp")
            raw_size = sum(f.stat().st_size for f in sym_raw_dir.glob("*.zip")) / 1e9
            print(f"  💾 Checkpoint: {len(combined):,} bars, raw={raw_size:.1f}GB")
        
        current += timedelta(days=1)
    
    if all_bars:
        combined = pd.concat(all_bars).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.to_csv(out_file, index_label="timestamp")
        
        days = len(set(combined.index.date))
        raw_size = sum(f.stat().st_size for f in sym_raw_dir.glob("*.zip")) / 1e9
        print(f"\n{'='*60}")
        print(f"✅ {args.asset}: {len(combined):,} rich 1m bars across {days} days")
        print(f"   File: {out_file} ({out_file.stat().st_size/1e6:.1f} MB)")
        print(f"   Raw files: {raw_size:.1f} GB kept in {sym_raw_dir}")
        print(f"   Columns ({len(combined.columns)}): {list(combined.columns)}")
        print(f"   Range: {combined.index[0]} → {combined.index[-1]}")
        print(f"   Downloaded: {downloaded}, Errors: {errors}")


if __name__ == "__main__":
    main()
