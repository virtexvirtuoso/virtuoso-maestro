"""
Stablecoin Supply Loader — fetches USDT/USDC market cap history from DefiLlama.
Used as a crypto-native liquidity proxy (alternative to M2).
"""

import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFILLAMA_URL = "https://stablecoins.llama.fi/stablecoin"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "stablecoin"
CACHE_TTL = 86400  # 1 day

# DefiLlama stablecoin IDs
STABLECOIN_IDS = {"tether": 1, "usd-coin": 2}


class StablecoinLoader:
    """Downloads and caches stablecoin market cap (supply proxy) from DefiLlama."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_call = time.time()

    def _cache_path(self, coin_id: str) -> Path:
        return self.cache_dir / f"{coin_id}_mcap.parquet"

    def _is_stale(self, path: Path) -> bool:
        if not path.exists():
            return True
        return (time.time() - path.stat().st_mtime) > CACHE_TTL

    def get_supply(self, coin_id: str = "tether", **kwargs) -> pd.Series:
        """Fetch daily market cap as a proxy for circulating supply."""
        cache = self._cache_path(coin_id)

        if not self._is_stale(cache):
            logger.info(f"Loading {coin_id} supply from cache")
            df = pd.read_parquet(cache)
            return df.iloc[:, 0]

        llama_id = STABLECOIN_IDS.get(coin_id, 1)
        logger.info(f"Downloading {coin_id} (id={llama_id}) market cap from DefiLlama")
        self._rate_limit()
        url = f"{DEFILLAMA_URL}/{llama_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # DefiLlama returns chainBalances with "totalCirculating" per day
        records = []
        for entry in data.get("tokens", []):
            ts = pd.Timestamp(entry["date"], unit="s").normalize()
            circ = entry.get("circulating", {})
            # Sum across all chains
            total = sum(v if isinstance(v, (int, float)) else 0 for v in circ.values())
            if total > 0:
                records.append((ts, total))

        s = pd.Series(
            dict(records),
            name=f"{coin_id}_mcap",
            dtype=float,
        )
        s.index.name = "timestamp"
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.to_frame().to_parquet(cache)
        return s

    def get_total_stablecoin_supply(self, **kwargs) -> pd.Series:
        """USDT + USDC combined supply."""
        usdt = self.get_supply("tether")
        usdc = self.get_supply("usd-coin")
        total = usdt.add(usdc, fill_value=0)
        total.name = "total_stablecoin_supply"
        return total

    def get_supply_growth(self, period: int = 30, **kwargs) -> pd.Series:
        """N-day supply growth rate."""
        total = self.get_total_stablecoin_supply()
        growth = total.pct_change(period)
        growth.name = f"supply_growth_{period}d"
        return growth
