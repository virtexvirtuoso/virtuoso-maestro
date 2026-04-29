"""
Macro Data Provider V3.1 — FRED + Cross-Asset Fetcher

Fetches macro signals from FRED API and cross-asset prices from yfinance.
Caches locally in JSON, refreshes once per day.

Signals provided:
  1. M2 acceleration (M2SL month-over-month acceleration)
  2. Yield curve (T10Y2Y)
  3. Fed funds rate (FEDFUNDS)
  4. CPI (CPIAUCSL)
  5. HY spread (BAMLH0A0HYM2)

Cross-asset tickers: GLD, UUP, TLT, HYG, COPX
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "M2SL": "M2 Money Supply",
    "T10Y2Y": "10Y-2Y Yield Curve",
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "CPI All Urban",
    "BAMLH0A0HYM2": "HY OAS Spread",
}

CROSS_ASSET_TICKERS = ["GLD", "UUP", "TLT", "HYG", "COPX"]

CACHE_DIR = Path(__file__).parent / "cache_v31"
FRED_CACHE_FILE = CACHE_DIR / "fred_data.json"
CROSS_ASSET_CACHE_FILE = CACHE_DIR / "cross_asset_data.json"
CACHE_MAX_AGE_HOURS = 20  # Refresh once per day, with some margin


def _cache_is_fresh(cache_file: Path, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    """Check if cache file exists and is less than max_age_hours old."""
    if not cache_file.exists():
        return False
    try:
        data = json.loads(cache_file.read_text())
        ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
        return (datetime.utcnow() - ts).total_seconds() < max_age_hours * 3600
    except Exception:
        return False


def _save_cache(cache_file: Path, data: dict):
    """Save data to cache with timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data["timestamp"] = datetime.utcnow().isoformat()
    cache_file.write_text(json.dumps(data, default=str))
    logger.info(f"Saved cache: {cache_file}")


def _load_cache(cache_file: Path) -> Optional[dict]:
    """Load data from cache file."""
    try:
        if cache_file.exists():
            return json.loads(cache_file.read_text())
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
#  FRED DATA
# ═══════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str, start_date: str = "2020-01-01") -> pd.Series:
    """Fetch a single FRED series via API."""
    import requests

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }

    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        records = []
        for o in obs:
            try:
                records.append({"date": o["date"], "value": float(o["value"])})
            except (ValueError, KeyError):
                continue
        if not records:
            logger.warning(f"FRED {series_id}: no valid observations")
            return pd.Series(dtype=float)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"]
    except Exception as e:
        logger.error(f"FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def fetch_all_fred(start_date: str = "2020-01-01") -> Dict[str, pd.Series]:
    """Fetch all FRED series, using cache if fresh."""
    if _cache_is_fresh(FRED_CACHE_FILE):
        logger.info("Using cached FRED data")
        cached = _load_cache(FRED_CACHE_FILE)
        if cached:
            result = {}
            for sid in FRED_SERIES:
                if sid in cached:
                    s = pd.Series(cached[sid]["values"], dtype=float)
                    s.index = pd.to_datetime(cached[sid]["dates"])
                    result[sid] = s
            if len(result) == len(FRED_SERIES):
                return result

    logger.info("Fetching fresh FRED data...")
    result = {}
    cache_payload = {}
    for sid in FRED_SERIES:
        s = fetch_fred_series(sid, start_date)
        result[sid] = s
        if not s.empty:
            cache_payload[sid] = {
                "dates": [d.isoformat() for d in s.index],
                "values": s.values.tolist(),
            }
        time.sleep(0.5)  # Rate limit courtesy

    _save_cache(FRED_CACHE_FILE, cache_payload)
    return result


# ═══════════════════════════════════════════════════════════════
#  CROSS-ASSET DATA (yfinance)
# ═══════════════════════════════════════════════════════════════

def fetch_cross_asset(
    tickers: list = CROSS_ASSET_TICKERS,
    period: str = "2y",
) -> pd.DataFrame:
    """
    Fetch cross-asset close prices from yfinance.
    Returns DataFrame with columns = ticker names, index = dates.
    """
    if _cache_is_fresh(CROSS_ASSET_CACHE_FILE):
        logger.info("Using cached cross-asset data")
        cached = _load_cache(CROSS_ASSET_CACHE_FILE)
        if cached and "data" in cached:
            df = pd.DataFrame(cached["data"])
            df.index = pd.to_datetime(df.index)
            # Check we have all tickers
            if all(t in df.columns for t in tickers):
                return df

    logger.info("Fetching fresh cross-asset data from yfinance...")
    try:
        import yfinance as yf
        data = yf.download(tickers, period=period, interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", level=0, axis=1)
        else:
            df = data[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else data

        df = df.dropna(how="all")

        # Cache it
        cache_payload = {"data": df.to_dict()}
        _save_cache(CROSS_ASSET_CACHE_FILE, cache_payload)

        return df
    except Exception as e:
        logger.error(f"Cross-asset fetch failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_m2_acceleration(fred_data: Dict[str, pd.Series]) -> pd.Series:
    """
    M2 acceleration signal: 1 if M2 MoM growth is accelerating.
    M2SL is monthly — forward-fill to daily.
    """
    m2 = fred_data.get("M2SL")
    if m2 is None or m2.empty:
        return pd.Series(dtype=float)
    m2_daily = m2.resample("D").ffill()
    m2_mom = m2_daily.pct_change(30)  # ~1 month
    m2_accel = m2_mom.diff(30)  # acceleration
    signal = (m2_accel > 0).astype(int)
    return signal


def compute_yield_curve_signal(fred_data: Dict[str, pd.Series]) -> pd.Series:
    """
    Yield curve signal: 1 if T10Y2Y > 0 (normal curve = risk-on).
    Daily series, forward-fill gaps.
    """
    yc = fred_data.get("T10Y2Y")
    if yc is None or yc.empty:
        return pd.Series(dtype=float)
    yc_daily = yc.resample("D").ffill()
    return (yc_daily > 0).astype(int)


def compute_hy_spread_signal(fred_data: Dict[str, pd.Series]) -> pd.Series:
    """
    HY spread signal: 1 if spread is declining (tightening = risk-on).
    BAMLH0A0HYM2 is daily.
    """
    hy = fred_data.get("BAMLH0A0HYM2")
    if hy is None or hy.empty:
        return pd.Series(dtype=float)
    hy_daily = hy.resample("D").ffill()
    hy_sma = hy_daily.rolling(20).mean()
    return (hy_daily < hy_sma).astype(int)


def compute_liquidity_proxy(cross_asset: pd.DataFrame) -> pd.Series:
    """
    Liquidity proxy: 3-of-4 rule over 20 days.
    Bullish if 3 of {GLD up, UUP down, TLT up, HYG up} over 20d.
    """
    if cross_asset.empty:
        return pd.Series(dtype=float)

    signals = pd.DataFrame(index=cross_asset.index)

    for ticker, direction in [("GLD", 1), ("UUP", -1), ("TLT", 1), ("HYG", 1)]:
        if ticker in cross_asset.columns:
            ret_20 = cross_asset[ticker].pct_change(20)
            signals[ticker] = ((ret_20 * direction) > 0).astype(int)
        else:
            signals[ticker] = 0

    # 3 of 4 must be true
    total = signals.sum(axis=1)
    return (total >= 3).astype(int)


def compute_cross_asset_momentum(cross_asset: pd.DataFrame) -> pd.Series:
    """
    Cross-asset momentum: positive if COPX (copper miners) 20d momentum > 0.
    Copper = global growth proxy.
    """
    if "COPX" not in cross_asset.columns:
        return pd.Series(dtype=float)
    ret_20 = cross_asset["COPX"].pct_change(20)
    return (ret_20 > 0).astype(int)


def get_all_macro_signals(start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Fetch and compute all macro signals. Returns daily DataFrame with columns:
      m2_accel, liquidity_proxy, yield_curve, cross_asset_mom, hy_spread
    Plus raw FRED data for reference.
    """
    fred = fetch_all_fred(start_date)
    cross = fetch_cross_asset()

    signals = pd.DataFrame()

    # 1. M2 acceleration
    m2_sig = compute_m2_acceleration(fred)
    if not m2_sig.empty:
        signals["m2_accel"] = m2_sig

    # 2. Liquidity proxy (3-of-4)
    liq_sig = compute_liquidity_proxy(cross)
    if not liq_sig.empty:
        signals["liquidity_proxy"] = liq_sig

    # 3. Yield curve
    yc_sig = compute_yield_curve_signal(fred)
    if not yc_sig.empty:
        signals["yield_curve"] = yc_sig

    # 4. Cross-asset momentum (COPX)
    cam_sig = compute_cross_asset_momentum(cross)
    if not cam_sig.empty:
        signals["cross_asset_mom"] = cam_sig

    # 5. HY spread (used as proxy in confluence)
    hy_sig = compute_hy_spread_signal(fred)
    if not hy_sig.empty:
        signals["hy_spread"] = hy_sig

    if signals.empty:
        logger.warning("No macro signals computed — all fetches may have failed")
        return pd.DataFrame()

    # Forward-fill and align to common daily index
    signals = signals.resample("D").ffill().dropna(how="all")
    signals = signals.fillna(0).astype(int)

    logger.info(
        f"Macro signals computed: {len(signals)} days, "
        f"columns={list(signals.columns)}"
    )
    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_all_macro_signals()
    print(f"\nMacro signals shape: {df.shape}")
    print(df.tail(10))
