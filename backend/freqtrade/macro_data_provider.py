"""
Macro Data Provider for Freqtrade Strategy.

Fetches FRED macro data (M2, yield curve, unemployment) and caches
as JSON for the strategy to read. Run via cron (daily or weekly).

Usage:
    # Update cache (run via cron):
    python macro_data_provider.py --update

    # Read current score:
    python macro_data_provider.py --score
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

CACHE_PATH = Path(__file__).parent / "macro_cache.json"

# FRED series IDs
FRED_SERIES = {
    "M2SL": "M2 Money Supply",
    "T10Y2Y": "10Y-2Y Yield Spread",
    "UNRATE": "Unemployment Rate",
    "DFF": "Fed Funds Rate",
    "DTWEXBGS": "Trade-Weighted Dollar Index",
}


class MacroDataProvider:
    """Fetches and caches macro data for the trading strategy.

    The strategy reads from a JSON cache file. This provider is run
    separately (cron job) to keep the cache fresh.
    """

    def __init__(self, fred_api_key: Optional[str] = None, cache_path: Optional[Path] = None):
        self.api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        self.cache_path = cache_path or CACHE_PATH

    def update_macro_data(self) -> dict:
        """Fetch latest FRED data, compute macro score, save to JSON.

        Returns the computed macro data dict.
        """
        data = {}

        if not self.api_key:
            print("WARNING: No FRED_API_KEY set. Using fallback neutral scores.")
            data = self._fallback_data()
        else:
            try:
                import requests

                for series_id, name in FRED_SERIES.items():
                    url = (
                        f"https://api.stlouisfed.org/fred/series/observations"
                        f"?series_id={series_id}&api_key={self.api_key}"
                        f"&file_type=json&sort_order=desc&limit=60"
                    )
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    obs = resp.json().get("observations", [])

                    # Get latest valid value
                    values = []
                    for o in obs:
                        try:
                            values.append(float(o["value"]))
                        except (ValueError, KeyError):
                            continue

                    if values:
                        data[series_id] = {
                            "name": name,
                            "latest": values[0],
                            "prev_month": values[1] if len(values) > 1 else values[0],
                            "prev_quarter": values[3] if len(values) > 3 else values[0],
                            "values_12m": values[:12],
                        }
            except Exception as e:
                print(f"FRED fetch failed: {e}. Using fallback.")
                data = self._fallback_data()

        # Compute scores
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "series": data,
            "macro_score": self._compute_macro_score(data),
            "m2_accelerating": self._check_m2_accelerating(data),
            "confluence_boost": self._compute_confluence_boost(data),
        }

        self.cache_path.write_text(json.dumps(result, indent=2))
        print(f"Macro cache updated: {self.cache_path}")
        return result

    def get_macro_score(self) -> float:
        """Read latest macro score from cached JSON. Returns 0.5 (neutral) on failure."""
        try:
            data = json.loads(self.cache_path.read_text())
            return data.get("macro_score", 0.5)
        except Exception:
            return 0.5

    def get_m2_accelerating(self) -> bool:
        """Is M2 currently accelerating?"""
        try:
            data = json.loads(self.cache_path.read_text())
            return data.get("m2_accelerating", False)
        except Exception:
            return False

    def get_confluence_boost(self) -> int:
        """Additional confluence points from macro data (0, 1, or 2)."""
        try:
            data = json.loads(self.cache_path.read_text())
            return data.get("confluence_boost", 0)
        except Exception:
            return 0

    def _compute_macro_score(self, data: dict) -> float:
        """Compute overall macro score (0.0 = bearish, 1.0 = bullish).

        Factors:
          - M2 growth (expanding = bullish)
          - Yield curve (positive = bullish)
          - Unemployment (low/falling = bullish)
          - Fed funds (low/falling = bullish)
          - Dollar (weak = bullish for crypto)
        """
        score = 0.5  # neutral default
        count = 0

        # M2 growth
        m2 = data.get("M2SL", {})
        if m2:
            latest, prev = m2.get("latest", 0), m2.get("prev_quarter", 0)
            if prev > 0:
                growth = (latest - prev) / prev
                score += 0.2 if growth > 0.01 else (-0.1 if growth < -0.01 else 0)
                count += 1

        # Yield curve
        yc = data.get("T10Y2Y", {})
        if yc:
            spread = yc.get("latest", 0)
            score += 0.15 if spread > 0.5 else (-0.15 if spread < -0.5 else 0)
            count += 1

        # Unemployment
        ur = data.get("UNRATE", {})
        if ur:
            latest, prev = ur.get("latest", 5), ur.get("prev_quarter", 5)
            score += 0.1 if latest < prev else (-0.1 if latest > prev + 0.3 else 0)
            count += 1

        # Dollar index (inverse for crypto)
        dxy = data.get("DTWEXBGS", {})
        if dxy:
            latest, prev = dxy.get("latest", 100), dxy.get("prev_quarter", 100)
            if prev > 0:
                change = (latest - prev) / prev
                score += 0.1 if change < -0.02 else (-0.1 if change > 0.02 else 0)
                count += 1

        return max(0.0, min(1.0, score))

    def _check_m2_accelerating(self, data: dict) -> bool:
        """Check if M2 money supply is accelerating (QoQ growth positive)."""
        m2 = data.get("M2SL", {})
        if not m2:
            return False
        latest = m2.get("latest", 0)
        prev = m2.get("prev_quarter", 0)
        return prev > 0 and (latest - prev) / prev > 0.005

    def _compute_confluence_boost(self, data: dict) -> int:
        """Compute 0–2 extra confluence points from macro conditions.

        +1 if M2 accelerating
        +1 if macro_score > 0.65
        """
        boost = 0
        if self._check_m2_accelerating(data):
            boost += 1
        if self._compute_macro_score(data) > 0.65:
            boost += 1
        return boost

    def _fallback_data(self) -> dict:
        """Neutral fallback when FRED is unavailable."""
        return {
            series_id: {
                "name": name,
                "latest": 0,
                "prev_month": 0,
                "prev_quarter": 0,
                "values_12m": [],
            }
            for series_id, name in FRED_SERIES.items()
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Macro Data Provider")
    parser.add_argument("--update", action="store_true", help="Update macro cache")
    parser.add_argument("--score", action="store_true", help="Print current macro score")
    parser.add_argument("--api-key", type=str, help="FRED API key")
    args = parser.parse_args()

    provider = MacroDataProvider(fred_api_key=args.api_key)

    if args.update:
        result = provider.update_macro_data()
        print(json.dumps(result, indent=2))
    elif args.score:
        print(f"Macro Score: {provider.get_macro_score():.2f}")
        print(f"M2 Accelerating: {provider.get_m2_accelerating()}")
        print(f"Confluence Boost: {provider.get_confluence_boost()}")
    else:
        parser.print_help()
