"""
Cycle Position Score (CPS), Top Probability Indicator (TPI), and Bottom Probability Indicator (BPI).

Virtuoso-style macro regime filters built from price, volume, and derivatives data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DERIVATIVES_DIR = Path.home() / "Desktop" / "maestro" / "data" / "derivatives"


def _clip(x, lo=0, hi=10):
    return np.clip(x, lo, hi)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def _load_funding(token: str) -> pd.Series:
    """Load daily funding rate for token, return Series indexed by date."""
    for suffix in ["_funding_full.csv", "_funding.csv"]:
        p = DERIVATIVES_DIR / f"{token}{suffix}"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp"])
            col = "funding_rate" if "funding_rate" in df.columns else "fundingRate"
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            # Average across intraday readings
            daily = df.groupby("date")[col].mean()
            daily.index = pd.to_datetime(daily.index)
            return daily
    return pd.Series(dtype=float)


def _load_oi(token: str) -> pd.Series:
    """Load daily OI close for token."""
    for suffix in ["_oi_daily_full.csv", "_oi_1d.csv"]:
        p = DERIVATIVES_DIR / f"{token}{suffix}"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp"])
            col = "c" if "c" in df.columns else "sumOpenInterest" if "sumOpenInterest" in df.columns else None
            if col is None:
                continue
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            daily = df.groupby("date")[col].last()
            daily.index = pd.to_datetime(daily.index)
            return daily
    return pd.Series(dtype=float)


def compute_indices(ohlcv: pd.DataFrame, token: str = "btc") -> pd.DataFrame:
    """
    Compute CPS, TPI, BPI for a token.

    Parameters
    ----------
    ohlcv : DataFrame with columns [open, high, low, close, volume] and DatetimeIndex.
    token : str, lowercase token name for loading derivatives data.

    Returns
    -------
    DataFrame with columns: cps, tpi, bpi (and sub-components).
    """
    df = ohlcv.copy()
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # Moving averages
    sma200 = close.rolling(200).mean()
    sma50 = close.rolling(50).mean()
    std200 = close.rolling(200).std()

    # RSI
    rsi = _rsi(close, 14)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - macd_signal

    # Realized vol (90d)
    log_ret = np.log(close / close.shift(1))
    rvol90 = log_ret.rolling(90).std() * np.sqrt(365)
    rvol_pct = rvol90.rolling(365).rank(pct=True)  # percentile over trailing year

    # Returns
    ret30 = close.pct_change(30)
    ret90 = close.pct_change(90)

    # Volume trend
    vol_sma20 = volume.rolling(20).mean()
    vol_sma50 = volume.rolling(50).mean()

    # Funding & OI
    funding = _load_funding(token)
    oi = _load_oi(token)

    # Align funding/OI to ohlcv index
    funding_aligned = funding.reindex(close.index).ffill()
    oi_aligned = oi.reindex(close.index).ffill()

    # Funding z-score (90d rolling)
    funding_mean = funding_aligned.rolling(90).mean()
    funding_std = funding_aligned.rolling(90).std()
    funding_z = (funding_aligned - funding_mean) / funding_std.replace(0, np.nan)

    # OI change 30d
    oi_chg30 = oi_aligned.pct_change(30)

    # ===================== CPS Components (each 0-10) =====================

    # 1. Price vs 200d SMA
    dist200 = (close - sma200) / sma200
    cps1 = _clip(5 + dist200 * 20)  # ±25% maps to 0-10

    # 2. Price vs 50d SMA
    dist50 = (close - sma50) / sma50
    cps2 = _clip(5 + dist50 * 30)

    # 3. RSI mapping
    cps3 = _clip((rsi - 30) / 4)  # 30->0, 70->10

    # 4. MACD histogram
    macd_norm = macd_hist / close * 100  # normalize by price
    cps4 = _clip(5 + macd_norm * 10)

    # 5. Realized vol percentile (inverted U - mid vol = bullish)
    # Low vol early bull, high vol tops. Use inverted: high pct = lower score
    cps5 = _clip(10 * (1 - rvol_pct))

    # 6. 30d return momentum
    cps6 = _clip(5 + ret30 * 20)

    # 7. 90d return momentum
    cps7 = _clip(5 + ret90 * 10)

    # 8. Volume trend
    vol_ratio = vol_sma20 / vol_sma50.replace(0, np.nan)
    cps8 = _clip((vol_ratio - 0.5) * 10)

    # 9. Funding z-score (moderate positive = bullish, extreme = bearish)
    # Bell curve: z=0-1 bullish, z>2 overbought
    cps9 = _clip(5 + funding_z * 2 - funding_z.clip(lower=1.5) * 3)

    # 10. OI change 30d (rising OI + rising price = healthy)
    price_rising = (ret30 > 0).astype(float)
    oi_score = oi_chg30 * 50 * price_rising + oi_chg30.clip(upper=0) * 20 * (1 - price_rising)
    cps10 = _clip(5 + oi_score)

    cps = cps1 + cps2 + cps3 + cps4 + cps5 + cps6 + cps7 + cps8 + cps9 + cps10

    # ===================== TPI (0-8) =====================
    tpi1 = (close > sma200 + 2 * std200).astype(float)
    tpi2 = (rsi > 80).astype(float)
    tpi3 = (ret90 > 1.0).astype(float)
    tpi4 = (funding_aligned > 0.0005).astype(float)  # 0.05% per 8h

    # Volume declining while price rising (20d)
    vol_slope = volume.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True)
    price_slope = close.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True)
    tpi5 = ((vol_slope < 0) & (price_slope > 0)).astype(float)

    # 30d vol > 80th pct
    vol30 = log_ret.rolling(30).std() * np.sqrt(365)
    vol30_pct = vol30.rolling(365).rank(pct=True)
    tpi6 = (vol30_pct > 0.8).astype(float)

    tpi7 = (close > 1.5 * sma200).astype(float)

    # Consecutive extreme positive funding (>0.03% for 3+ days)
    extreme_funding = (funding_aligned > 0.0003).astype(float)
    consec_fund = extreme_funding.rolling(3).sum()
    tpi8 = (consec_fund >= 3).astype(float)

    tpi = tpi1 + tpi2 + tpi3 + tpi4 + tpi5 + tpi6 + tpi7 + tpi8

    # ===================== BPI (0-7) =====================
    bpi1 = (close < sma200 - 2 * std200).astype(float)
    bpi2 = (rsi < 20).astype(float)
    bpi3 = (ret90 < -0.5).astype(float)
    bpi4 = (funding_aligned < -0.0003).astype(float)

    # Volume spike (capitulation) - volume > 3x 50d avg
    bpi5 = (volume > 3 * vol_sma50).astype(float)

    bpi6 = (close < 0.7 * sma200).astype(float)

    # Consecutive negative funding (3+ days)
    neg_funding = (funding_aligned < -0.0001).astype(float)
    consec_neg = neg_funding.rolling(3).sum()
    bpi7 = (consec_neg >= 3).astype(float)

    bpi = bpi1 + bpi2 + bpi3 + bpi4 + bpi5 + bpi6 + bpi7

    result = pd.DataFrame({
        "cps": cps, "tpi": tpi, "bpi": bpi,
        "rsi": rsi, "funding": funding_aligned, "oi": oi_aligned,
    }, index=df.index)

    return result
