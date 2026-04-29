"""
Central data path registry for Maestro.

Single source of truth for every data directory. All research scripts, loaders,
and tools MUST import from here instead of hardcoding paths.

Override the root via the MAESTRO_DATA_ROOT environment variable for testing
or alternate mount points.

Layout on canonical volume (/Volumes/G-DRIVE/maestro-data/):

    raw/binance_aggtrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip
    bars/1m/v1/{symbol}usdt_1m.csv                (15 columns, 13 symbols, 2024-01 → 2026-02)
    bars/1m/v2/{symbol}usdt_1m_v2.csv             (29 columns, BTC/ETH only)
    tick/trades/{SYMBOL}/{YYYY-MM-DD}.parquet     (per-tick, 8 symbols, Feb 15 → Mar 24 2026)
    tick/orderbook/{SYMBOL}/{YYYY-MM-DD}.parquet  (L2 snapshots, 8 symbols, Feb 15 → Apr 9 2026)
    archive/*.tar.gz                              (historical pre-Feb 2026 archives)
"""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
MAESTRO_DATA_ROOT = Path(
    os.environ.get("MAESTRO_DATA_ROOT", "/Volumes/G-DRIVE/maestro-data")
)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
RAW = MAESTRO_DATA_ROOT / "raw"
RAW_AGGTRADES = RAW / "binance_aggtrades"

BARS = MAESTRO_DATA_ROOT / "bars"
BARS_1M_V1 = BARS / "1m" / "v1"   # 15-col aggregated (full 13 symbols)
BARS_1M_V2 = BARS / "1m" / "v2"   # 29-col microstructure (BTC/ETH only)

TICK = MAESTRO_DATA_ROOT / "tick"
TICK_TRADES = TICK / "trades"
TICK_ORDERBOOK = TICK / "orderbook"

ARCHIVE = MAESTRO_DATA_ROOT / "archive"


# ---------------------------------------------------------------------------
# Helpers — prefer these over string concatenation at call sites
# ---------------------------------------------------------------------------
def bars_1m_path(symbol: str, version: str = "v1") -> Path:
    """
    Return path to a 1-minute bars CSV.

    >>> bars_1m_path("BTCUSDT")
    PosixPath('/Volumes/G-DRIVE/maestro-data/bars/1m/v1/btcusdt_1m.csv')
    >>> bars_1m_path("ETH", "v2")
    PosixPath('/Volumes/G-DRIVE/maestro-data/bars/1m/v2/ethusdt_1m_v2.csv')
    """
    sym = symbol.lower().replace("usdt", "") + "usdt"
    if version == "v1":
        return BARS_1M_V1 / f"{sym}_1m.csv"
    if version == "v2":
        return BARS_1M_V2 / f"{sym}_1m_v2.csv"
    raise ValueError(f"unknown bars version: {version}")


def tick_trades_path(symbol: str, date: str) -> Path:
    """
    Return path to a per-tick trades parquet for a given UTC date.

    >>> tick_trades_path("BTCUSDT", "2026-02-15")
    PosixPath('/Volumes/G-DRIVE/maestro-data/tick/trades/BTCUSDT/2026-02-15.parquet')
    """
    return TICK_TRADES / symbol.upper() / f"{date}.parquet"


def tick_orderbook_path(symbol: str, date: str) -> Path:
    """Return path to a per-tick L2 orderbook parquet for a given UTC date."""
    return TICK_ORDERBOOK / symbol.upper() / f"{date}.parquet"


def tick_trades_dir(symbol: str) -> Path:
    """Directory containing all trades parquet files for a symbol."""
    return TICK_TRADES / symbol.upper()


def tick_orderbook_dir(symbol: str) -> Path:
    """Directory containing all orderbook parquet files for a symbol."""
    return TICK_ORDERBOOK / symbol.upper()


def raw_aggtrades_dir(symbol: str) -> Path:
    """Directory with raw Binance aggTrades zip files for a symbol."""
    return RAW_AGGTRADES / symbol.upper()


# ---------------------------------------------------------------------------
# Sanity check at import time (non-fatal)
# ---------------------------------------------------------------------------
def _check_layout() -> dict:
    return {
        "root_exists": MAESTRO_DATA_ROOT.exists(),
        "bars_1m_v1_count": len(list(BARS_1M_V1.glob("*.csv"))) if BARS_1M_V1.exists() else 0,
        "tick_trades_symbols": sorted(p.name for p in TICK_TRADES.iterdir()) if TICK_TRADES.exists() else [],
        "tick_orderbook_symbols": sorted(p.name for p in TICK_ORDERBOOK.iterdir()) if TICK_ORDERBOOK.exists() else [],
    }


if __name__ == "__main__":
    import json
    print(f"MAESTRO_DATA_ROOT = {MAESTRO_DATA_ROOT}")
    print(json.dumps(_check_layout(), indent=2))
