"""
Feature Engine — Build comprehensive feature matrix from all data sources.
~70 features grouped by: price, cross-asset, macro, factor, derived.
All features are lagged by 1 day (.shift(1)) before model consumption.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict


# ── Technical Helpers ──────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    lower = mid - std * s
    upper = mid + std * s
    return lower, mid, upper


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Price Features (per asset) ────────────────────────────────────────────

def _price_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Build price-based features for a single asset OHLCV DataFrame."""
    c = df["close"]
    v = df.get("volume", pd.Series(0, index=df.index))
    h = df.get("high", c)
    l = df.get("low", c)
    feats = {}

    # Returns
    for d in [1, 5, 10, 30, 60]:
        feats[f"{prefix}_ret_{d}d"] = c.pct_change(d)

    # SMA ratios
    for w in [20, 50, 100, 200]:
        sma = c.rolling(w).mean()
        feats[f"{prefix}_sma{w}_ratio"] = c / sma.replace(0, np.nan)

    # RSI
    feats[f"{prefix}_rsi14"] = _rsi(c, 14)
    feats[f"{prefix}_rsi28"] = _rsi(c, 28)

    # Bollinger Band position & width
    bb_l, bb_m, bb_u = _bb(c, 20, 2.0)
    bb_range = (bb_u - bb_l).replace(0, np.nan)
    feats[f"{prefix}_bb_pos"] = (c - bb_l) / bb_range
    feats[f"{prefix}_bbw"] = bb_range / bb_m.replace(0, np.nan)

    # Normalized ATR
    atr = _atr(h, l, c, 14)
    feats[f"{prefix}_natr"] = atr / c.replace(0, np.nan)

    # Volume ratio
    if (v != 0).any():
        vol_sma = v.rolling(20).mean().replace(0, np.nan)
        feats[f"{prefix}_vol_ratio"] = v / vol_sma

    # High/Low range ratio
    feats[f"{prefix}_hl_ratio"] = (h - l) / c.replace(0, np.nan)

    return pd.DataFrame(feats, index=df.index)


# ── Cross-Asset Features ──────────────────────────────────────────────────

def _cross_asset_features(cross: pd.DataFrame) -> pd.DataFrame:
    """Build cross-asset features from DataFrame with columns: gold/GLD, dxy/UUP, bonds/TLT, hyg/HYG, copper/COPX."""
    feats = {}
    col_map = {
        "GLD": ["gold", "GLD"], "UUP": ["dxy", "UUP"],
        "TLT": ["bonds", "TLT"], "HYG": ["hyg", "HYG"], "COPX": ["copper", "COPX"],
    }

    def _find(names):
        for n in names:
            if n in cross.columns:
                return cross[n]
        return None

    for label, names in col_map.items():
        s = _find(names)
        if s is None:
            continue
        for d in [5, 20]:
            feats[f"{label}_ret_{d}d"] = s.pct_change(d)

    gld = _find(col_map["GLD"])
    uup = _find(col_map["UUP"])
    copx = _find(col_map["COPX"])

    if gld is not None and uup is not None:
        ratio = gld / uup.replace(0, np.nan)
        feats["gld_uup_ratio_mom"] = ratio.pct_change(20)

    if copx is not None and gld is not None:
        feats["copper_gold_ratio"] = copx / gld.replace(0, np.nan)

    return pd.DataFrame(feats, index=cross.index) if feats else pd.DataFrame(index=cross.index)


# ── Macro Features ────────────────────────────────────────────────────────

def _macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Build macro features from DataFrame with: yield_curve/T10Y2Y, m2/M2SL, fed_funds, cpi, hy_spread."""
    feats = {}

    def _col(names):
        for n in names:
            if n in macro.columns:
                return macro[n]
        return None

    yc = _col(["yield_curve", "T10Y2Y"])
    if yc is not None:
        feats["yc_level"] = yc
        feats["yc_30d_chg"] = yc - yc.shift(30)

    m2 = _col(["m2", "M2SL"])
    if m2 is not None:
        m2_yoy = m2.pct_change(365)
        feats["m2_yoy"] = m2_yoy
        feats["m2_accel"] = m2_yoy - m2_yoy.rolling(180).mean()

    ff = _col(["fed_funds", "FEDFUNDS"])
    if ff is not None:
        feats["fed_funds_level"] = ff
        feats["fed_funds_90d_chg"] = ff - ff.shift(90)

    cpi = _col(["cpi", "CPIAUCSL"])
    if cpi is not None:
        cpi_yoy = cpi.pct_change(365)
        feats["cpi_yoy"] = cpi_yoy
        feats["cpi_trend"] = cpi_yoy - cpi_yoy.shift(90)

    hy = _col(["hy_spread", "BAMLH0A0HYM2"])
    if hy is not None:
        feats["hy_spread_level"] = hy
        feats["hy_spread_30d_chg"] = hy - hy.shift(30)

    # Macro score components as binary
    if yc is not None:
        feats["macro_yc_positive"] = (yc > 0).astype(int)
    if m2 is not None:
        m2_3m = m2.pct_change(90)
        m2_6m = m2.pct_change(180)
        feats["macro_m2_expanding"] = (m2_3m > 0).astype(int)
        feats["macro_m2_accelerating"] = (m2_3m > m2_6m).astype(int)
    if cpi is not None:
        cpi_yoy_val = cpi.pct_change(365)
        feats["macro_cpi_declining"] = ((cpi_yoy_val - cpi_yoy_val.shift(90)) < 0).astype(int)
    if ff is not None:
        feats["macro_fed_not_hiking"] = ((ff - ff.shift(90)) <= 0).astype(int)
    if hy is not None:
        feats["macro_hy_not_tight"] = ((hy - hy.shift(90)) <= 0).astype(int)

    return pd.DataFrame(feats, index=macro.index) if feats else pd.DataFrame(index=macro.index)


# ── Factor Features ───────────────────────────────────────────────────────

def _factor_features(ff_data: pd.DataFrame) -> pd.DataFrame:
    """FF5 + Momentum factors (monthly, will be ffilled to daily)."""
    cols = []
    for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]:
        if c in ff_data.columns:
            cols.append(c)
    return ff_data[cols].copy() if cols else pd.DataFrame(index=ff_data.index)


# ── Derived / Confluence Features ─────────────────────────────────────────

def _derived_features(
    btc_close: pd.Series,
    cross_asset: pd.DataFrame,
    confluence_breakdown: Optional[pd.DataFrame] = None,
    regime_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    feats = {}

    # Realized vol
    ret = btc_close.pct_change()
    feats["rvol_20d"] = ret.rolling(20).std() * np.sqrt(252)
    feats["rvol_60d"] = ret.rolling(60).std() * np.sqrt(252)
    rvol_1y = ret.rolling(252).std() * np.sqrt(252)
    feats["vol_regime"] = (ret.rolling(20).std() * np.sqrt(252)) / rvol_1y.replace(0, np.nan)

    # Cross-asset correlations
    def _find(names):
        for n in names:
            if n in cross_asset.columns:
                return cross_asset[n]
        return None

    gld = _find(["gold", "GLD"])
    spy = _find(["SPY", "spy"])

    btc_ret = btc_close.pct_change()
    if gld is not None:
        gld_ret = gld.pct_change()
        feats["corr_btc_gld_30d"] = btc_ret.rolling(30).corr(gld_ret)
    if spy is not None:
        spy_ret = spy.pct_change()
        feats["corr_btc_spy_30d"] = btc_ret.rolling(30).corr(spy_ret)

    # Confluence signals
    if confluence_breakdown is not None:
        for col in ["m2_accel", "liquidity_proxy", "yield_curve", "cross_asset_mom", "crypto_momentum"]:
            if col in confluence_breakdown.columns:
                feats[f"sig_{col}"] = confluence_breakdown[col]
        if "confluence" in confluence_breakdown.columns:
            feats["confluence_score"] = confluence_breakdown["confluence"]

    # Regime encoded
    if regime_series is not None:
        regime_map = {"BULL": 4, "MILD_BULL": 3, "ACCUMULATION": 2, "NEUTRAL": 1, "BEAR": 0}
        feats["regime_encoded"] = regime_series.map(regime_map).fillna(1)

    return pd.DataFrame(feats, index=btc_close.index)


# ── Target Variables ──────────────────────────────────────────────────────

def _build_targets(btc_close: pd.Series, regime_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """Forward-looking targets (NOT shifted — caller must ensure no lookahead)."""
    ret = btc_close.pct_change()
    targets = {}
    targets["fwd_ret_1d"] = ret.shift(-1)
    targets["fwd_ret_5d"] = btc_close.pct_change(5).shift(-5)
    targets["fwd_ret_20d"] = btc_close.pct_change(20).shift(-20)

    # Forward 5d max drawdown
    fwd_5d_dd = pd.Series(np.nan, index=btc_close.index)
    prices = btc_close.values
    for i in range(len(prices) - 5):
        window = prices[i + 1: i + 6]
        peak = prices[i]
        dd = (window / peak - 1).min()
        fwd_5d_dd.iloc[i] = dd
    targets["fwd_5d_max_dd"] = fwd_5d_dd

    # Binary classification target
    targets["fwd_ret_20d_positive"] = (targets["fwd_ret_20d"] > 0).astype(float)

    # Regime label
    if regime_series is not None:
        targets["regime_label"] = regime_series

    return pd.DataFrame(targets, index=btc_close.index)


# ── Main Entry Point ──────────────────────────────────────────────────────

def build_feature_matrix(
    btc: pd.DataFrame,
    eth: Optional[pd.DataFrame] = None,
    sol: Optional[pd.DataFrame] = None,
    link: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    ff_data: Optional[pd.DataFrame] = None,
    cross_asset: Optional[pd.DataFrame] = None,
    confluence_breakdown: Optional[pd.DataFrame] = None,
    regime_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build daily feature matrix from all data sources.
    Returns DataFrame with ~50-80 features + target columns.
    ALL FEATURES ARE LAGGED BY 1 DAY to prevent lookahead.
    Targets are NOT lagged (they are forward-looking by definition).
    """
    idx = btc.index
    parts = []

    # Price features per asset
    parts.append(_price_features(btc, "btc"))
    if eth is not None and not eth.empty:
        parts.append(_price_features(eth, "eth").reindex(idx))
    if sol is not None and not sol.empty:
        parts.append(_price_features(sol, "sol").reindex(idx))
    if link is not None and not link.empty:
        parts.append(_price_features(link, "link").reindex(idx))

    # Cross-asset
    if cross_asset is not None and not cross_asset.empty:
        ca = cross_asset.reindex(idx, method="ffill")
        parts.append(_cross_asset_features(ca))

    # Macro
    if macro_df is not None and not macro_df.empty:
        macro_daily = macro_df.reindex(idx, method="ffill")
        parts.append(_macro_features(macro_daily))

    # Factors (monthly → daily ffill)
    if ff_data is not None and not ff_data.empty:
        ff_daily = ff_data.reindex(idx, method="ffill")
        parts.append(_factor_features(ff_daily))

    # Derived
    parts.append(_derived_features(
        btc["close"], cross_asset if cross_asset is not None else pd.DataFrame(index=idx),
        confluence_breakdown, regime_series,
    ))

    # Combine features
    features = pd.concat(parts, axis=1)

    # LAG ALL FEATURES BY 1 DAY — critical for no lookahead
    features = features.shift(1)

    # Targets (NOT lagged — they are forward-looking)
    targets = _build_targets(btc["close"], regime_series)

    # Combine
    result = pd.concat([features, targets], axis=1)
    return result


def get_feature_columns(matrix: pd.DataFrame) -> list:
    """Return list of feature column names (excludes targets)."""
    target_cols = {"fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d",
                   "fwd_5d_max_dd", "fwd_ret_20d_positive", "regime_label"}
    return [c for c in matrix.columns if c not in target_cols]
