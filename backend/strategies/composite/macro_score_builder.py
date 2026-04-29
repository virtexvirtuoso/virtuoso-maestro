"""
Macro Score Builder
Computes a 0-6 macro favorability score from FRED data.

Components (each 0 or 1):
1. yield_curve: T10Y2Y > 0 (not inverted)
2. m2_expanding: M2SL 3-month change > 0
3. m2_accelerating: M2SL 3-month change > 6-month change
4. cpi_declining: CPI YoY decreasing over last 3 months
5. fed_not_hiking: FEDFUNDS not increasing over last 3 months
6. hy_not_tightening: HY spread (BAMLH0A0HYM2) not rising over last 3 months
"""
import pandas as pd
import numpy as np
from typing import Optional


FRED_SERIES = {
    "yield_curve": "T10Y2Y",
    "m2": "M2SL",
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "hy_spread": "BAMLH0A0HYM2",
}


def compute_macro_score(
    fred_loader,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Compute macro score (0-6) from FRED data via MacroDataLoader.
    Returns daily-frequency Series (forward-filled from monthly data).
    """
    macro_df = fred_loader.get_multiple(
        FRED_SERIES, start_date=start_date, end_date=end_date
    )
    return compute_macro_score_from_df(macro_df)


def compute_macro_score_from_df(macro_df: pd.DataFrame) -> pd.Series:
    """
    Compute macro score from pre-loaded DataFrame.
    Expected columns: yield_curve, m2, fed_funds, cpi, hy_spread
    """
    df = macro_df.copy()
    # Forward-fill monthly data to daily
    df = df.asfreq("D", method="ffill") if not df.empty else df
    df = df.ffill()

    score = pd.Series(0, index=df.index, dtype=int)

    # 1. Yield curve positive
    if "yield_curve" in df.columns:
        score += (df["yield_curve"] > 0).astype(int)

    # 2. M2 expanding (3-month change > 0)
    if "m2" in df.columns:
        m2_3m = df["m2"].pct_change(90).fillna(0)
        score += (m2_3m > 0).astype(int)

        # 3. M2 accelerating (3-month change > 6-month change)
        m2_6m = df["m2"].pct_change(180).fillna(0)
        score += (m2_3m > m2_6m).astype(int)

    # 4. CPI declining (YoY rate falling over 90 days)
    if "cpi" in df.columns:
        cpi_yoy = df["cpi"].pct_change(365).fillna(0)
        cpi_yoy_chg = cpi_yoy - cpi_yoy.shift(90)
        score += (cpi_yoy_chg < 0).astype(int).fillna(0).astype(int)

    # 5. Fed not hiking (fed funds not increasing over 90 days)
    if "fed_funds" in df.columns:
        ff_chg = df["fed_funds"] - df["fed_funds"].shift(90)
        score += (ff_chg <= 0).astype(int).fillna(0).astype(int)

    # 6. HY spread not tightening (spread not rising over 90 days)
    if "hy_spread" in df.columns:
        hy_chg = df["hy_spread"] - df["hy_spread"].shift(90)
        score += (hy_chg <= 0).astype(int).fillna(0).astype(int)

    score.name = "macro_score"
    return score.clip(0, 6)
