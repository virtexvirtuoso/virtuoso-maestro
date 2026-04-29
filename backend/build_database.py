#!/usr/bin/env python3
"""Build Maestro DuckDB research database. Idempotent — safe to re-run."""

import duckdb
import pandas as pd
import glob
import os
import re
from datetime import datetime
from pathlib import Path

DATA = os.path.expanduser("~/Desktop/maestro/data")
DB_PATH = os.path.join(DATA, "maestro.duckdb")


def norm_symbol(s):
    """Normalize symbol: uppercase, strip USDT/USD suffix and /USDT:USDT."""
    if not isinstance(s, str):
        return s
    s = s.strip().upper()
    s = s.replace("/USDT:USDT", "").replace("/USD:USD", "")
    for suffix in ("USDT", "USD", "BUSD"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[:-len(suffix)]
    return s


def ms_to_date(col):
    """Convert millisecond timestamps to dates."""
    return pd.to_datetime(col, unit="ms").dt.date


def parse_date_col(df, candidates=("date", "dt", "time", "timestamp", "Date", "Time", "Timestamp")):
    """Find and normalize a date column, return (df, date_col_name)."""
    for c in candidates:
        if c in df.columns:
            vals = df[c]
            # Check if numeric (ms epoch or YYMMDD)
            if pd.api.types.is_numeric_dtype(vals):
                sample = vals.dropna().iloc[0] if len(vals.dropna()) > 0 else 0
                if sample < 1_000_000:
                    # YYMMDD format
                    df["date"] = pd.to_datetime(vals.astype(int).astype(str), format="%y%m%d", errors="coerce").dt.date
                elif sample < 1e12:
                    # seconds epoch
                    df["date"] = pd.to_datetime(vals, unit="s", errors="coerce").dt.date
                else:
                    df["date"] = ms_to_date(vals)
            else:
                parsed = pd.to_datetime(vals, errors="coerce")
                df["date"] = parsed.dt.date
            if c != "date":
                df.drop(columns=[c], inplace=True, errors="ignore")
            return df
    return df


def load_csv_safe(path):
    """Load CSV, return None on failure."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  ⚠ Failed to read {path}: {e}")
        return None


def extract_symbol_from_filename(fname, pattern):
    """Extract token symbol from filename using regex pattern."""
    m = re.search(pattern, fname)
    return m.group(1).upper() if m else None


def dedup(df, keys):
    """Deduplicate on given keys, keeping first."""
    present = [k for k in keys if k in df.columns]
    if present:
        df = df.drop_duplicates(subset=present, keep="first")
    return df


def build():
    print(f"Building Maestro DB at {DB_PATH}")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = duckdb.connect(DB_PATH)
    summary = []

    # ── 1. Spot Daily ──
    print("\n📊 Spot Daily")
    rows = []
    for f in sorted(glob.glob(f"{DATA}/spot/*_spot_daily.csv")):
        sym = Path(f).name.split("_")[0].upper()
        df = load_csv_safe(f)
        if df is None:
            continue
        df.columns = [c.lower() for c in df.columns]
        df = parse_date_col(df)
        df["symbol"] = sym
        rows.append(df)
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        all_df = dedup(all_df, ["symbol", "date"])
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        cols = [c for c in cols if c in all_df.columns]
        filtered = all_df[cols]
        con.execute("CREATE TABLE spot_daily AS SELECT * FROM filtered")
        summary.append(("spot_daily", len(filtered), filtered))

    # ── 2. Perps Daily (empty dir, skip gracefully) ──
    print("\n📊 Perps Daily")
    perp_files = glob.glob(f"{DATA}/perps/*_perps_daily.csv")
    if perp_files:
        rows = []
        for f in sorted(perp_files):
            sym = Path(f).name.split("_")[0].upper()
            df = load_csv_safe(f)
            if df is None:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = parse_date_col(df)
            df["symbol"] = sym
            rows.append(df)
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            all_df = dedup(all_df, ["symbol", "date"])
            cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
            cols = [c for c in cols if c in all_df.columns]
            filtered = all_df[cols]
            con.execute("CREATE TABLE perps_daily AS SELECT * FROM filtered")
            summary.append(("perps_daily", len(filtered), filtered))
    else:
        print("  ⚠ No perps daily files found, skipping")

    # ── 2b. Funding Rates Binance ──
    print("\n📊 Funding Rates (Binance/Perps)")
    fr_files = glob.glob(f"{DATA}/perps/*_funding_rates.csv")
    if fr_files:
        rows = []
        for f in sorted(fr_files):
            sym = Path(f).name.split("_")[0].upper()
            df = load_csv_safe(f)
            if df is None:
                continue
            df = parse_date_col(df)
            df["symbol"] = sym
            rows.append(df)
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            all_df = dedup(all_df, ["symbol", "date"])
            con.execute("CREATE TABLE funding_rates_binance AS SELECT * FROM all_df")
            summary.append(("funding_rates_binance", len(all_df), all_df))
    else:
        print("  ⚠ No binance funding files found, skipping")

    # ── 3. Coinalyze Derivatives ──
    deriv = f"{DATA}/derivatives"

    def load_coinalyze(suffix, table_name, extra_process=None):
        print(f"\n📊 {table_name}")
        files = sorted(glob.glob(f"{deriv}/*_{suffix}.csv"))
        if not files:
            print(f"  ⚠ No files for {suffix}")
            return
        rows = []
        for f in files:
            sym = Path(f).name.replace(f"_{suffix}.csv", "").upper()
            df = load_csv_safe(f)
            if df is None:
                continue
            df = parse_date_col(df)
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].apply(norm_symbol)
            else:
                df["symbol"] = sym
            if extra_process:
                df = extra_process(df)
            rows.append(df)
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            all_df = dedup(all_df, ["symbol", "date"])
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM all_df")
            summary.append((table_name, len(all_df), all_df))

    load_coinalyze("oi_1d", "coinalyze_oi")
    load_coinalyze("funding", "coinalyze_funding",
                    lambda df: df.rename(columns={"fundingRate": "rate"}, errors="ignore"))
    load_coinalyze("liquidations_daily", "coinalyze_liquidations")
    load_coinalyze("lsr_global", "coinalyze_lsr")

    # ── 4. CoinGlass ──
    cg = f"{DATA}/coinglass"

    def load_coinglass_type(pattern, symbol_regex, table_name, only_1d=True):
        print(f"\n📊 {table_name}")
        files = sorted(glob.glob(f"{cg}/{pattern}"))
        if only_1d:
            files = [f for f in files if "_1d.csv" in f or not re.search(r"_\dh\.csv$", f)]
        if not files:
            print(f"  ⚠ No files for {pattern}")
            return
        rows = []
        for f in files:
            sym = extract_symbol_from_filename(Path(f).name, symbol_regex)
            if not sym:
                continue
            df = load_csv_safe(f)
            if df is None:
                continue
            df = parse_date_col(df)
            df["symbol"] = sym
            rows.append(df)
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            all_df = dedup(all_df, ["symbol", "date"])
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM all_df")
            summary.append((table_name, len(all_df), all_df))

    load_coinglass_type("funding_rate_*_1d.csv", r"funding_rate_(\w+)_1d", "cg_funding_rate")
    load_coinglass_type("lsr_global_*_1d.csv", r"lsr_global_(\w+)_1d", "cg_lsr_global")
    load_coinglass_type("lsr_top_account_*_1d.csv", r"lsr_top_account_(\w+)_1d", "cg_lsr_top_account")
    load_coinglass_type("lsr_top_position_*_1d.csv", r"lsr_top_position_(\w+)_1d", "cg_lsr_top_position")
    load_coinglass_type("liquidation_aggregated_*.csv", r"liquidation_aggregated_(\w+)\.", "cg_liquidations", only_1d=False)
    load_coinglass_type("taker_buysell_*_1d.csv", r"taker_buysell_(\w+)_1d", "cg_taker_volume")

    # OI OHLC and FR OHLC (check if they exist)
    oi_files = glob.glob(f"{cg}/oi_ohlc_*.csv")
    fr_files = glob.glob(f"{cg}/fr_ohlc_*.csv")
    if oi_files:
        load_coinglass_type("oi_ohlc_*_1d.csv", r"oi_ohlc_(\w+)_", "cg_oi_ohlc")
    if fr_files:
        load_coinglass_type("fr_ohlc_*_1d.csv", r"fr_ohlc_(\w+)_", "cg_fr_ohlc")

    # Options
    opt_files = glob.glob(f"{cg}/options_*.csv")
    if opt_files:
        print(f"\n📊 cg_options (found {len(opt_files)} files)")
        # Just load them all into one table
        rows = []
        for f in sorted(opt_files):
            df = load_csv_safe(f)
            if df is None:
                continue
            df["source_file"] = Path(f).name
            rows.append(df)
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            con.execute("CREATE TABLE cg_options AS SELECT * FROM all_df")
            summary.append(("cg_options", len(all_df), all_df))

    # ── 5. Whale Trades ──
    print("\n📊 whale_trades")
    wt_path = f"{DATA}/whale_trades.csv"
    if os.path.exists(wt_path):
        df = load_csv_safe(wt_path)
        if df is not None:
            df = parse_date_col(df)
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].apply(norm_symbol)
            con.execute("CREATE TABLE whale_trades AS SELECT * FROM df")
            summary.append(("whale_trades", len(df), df))
    else:
        print("  ⚠ whale_trades.csv not found")

    # ── 6. Fear & Greed ──
    print("\n📊 fear_greed")
    fg_files = glob.glob(f"{DATA}/coinglass/fear_greed*.csv") + glob.glob(f"{DATA}/fear_greed*.csv")
    if fg_files:
        df = load_csv_safe(fg_files[0])
        if df is not None:
            df = parse_date_col(df)
            con.execute("CREATE TABLE fear_greed AS SELECT * FROM df")
            summary.append(("fear_greed", len(df), df))
    else:
        print("  ⚠ No fear & greed file found")

    # ── Metadata ──
    con.execute("""
        CREATE TABLE _metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    con.execute("INSERT INTO _metadata VALUES ('created_at', ?)", [datetime.now().isoformat()])
    con.execute("INSERT INTO _metadata VALUES ('builder', 'build_database.py')")

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"{'Table':<30} {'Rows':>10} {'Date Range':<30} Columns")
    print("=" * 70)
    for name, count, df in summary:
        date_range = ""
        if "date" in df.columns:
            mn = df["date"].min()
            mx = df["date"].max()
            date_range = f"{mn} → {mx}"
        cols = ", ".join(df.columns[:6])
        if len(df.columns) > 6:
            cols += f" (+{len(df.columns)-6})"
        print(f"{name:<30} {count:>10,} {date_range:<30} {cols}")
    print("=" * 70)

    # Table list from DB
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()
    print(f"\nTotal tables: {len(tables)}")
    for t in tables:
        rc = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        print(f"  {t[0]}: {rc:,} rows")

    con.close()
    print(f"\n✅ Database built: {DB_PATH}")
    print(f"   Size: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
