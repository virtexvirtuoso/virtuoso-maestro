"""
MegaStrategyV3 — Adaptive Leverage Long/Short Multi-Asset Macro Momentum

Architecture:
- 5-signal confluence engine (M2, liquidity proxy, yield curve, cross-asset, crypto momentum)
- Regime detector (BULL/MILD_BULL/NEUTRAL/BEAR/ACCUMULATION)
- Long side: dip-buying within uptrends with pyramiding (from V2)
- Short side: fade rallies in downtrends, earn funding carry
- Adaptive leverage: confluence-mapped with safety overrides (vol, DD)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

NAME = "MegaStrategyV3"
CATEGORY = "composite"
DESCRIPTION = "Adaptive leverage long/short multi-asset macro momentum with confluence scoring"
REQUIRES_DERIVATIVES = False

# Per-asset optimized params
ASSET_CONFIGS = {
    "BTC": dict(
        ticker="BTC-USD", sma_slow=100, momentum_period=35, rsi_entry=52,
        trail_stop_pct=0.12, max_position=1.50, initial_size=0.60,
        rsi_exit=75, ema_period=21, bb_period=20, bb_std=2.0,
        roc_threshold=0.0, pyramid_size=0.20, trim_pct=0.25,
        atr_exit_mult=2.0, macro_weight=1.0,
    ),
    "ETH": dict(
        ticker="ETH-USD", sma_slow=140, momentum_period=15, rsi_entry=32,
        trail_stop_pct=0.20, max_position=1.60, initial_size=0.70,
        rsi_exit=75, ema_period=21, bb_period=20, bb_std=2.0,
        roc_threshold=0.0, pyramid_size=0.20, trim_pct=0.25,
        atr_exit_mult=2.0, macro_weight=1.0,
    ),
    "SOL": dict(
        ticker="SOL-USD", sma_slow=70, momentum_period=20, rsi_entry=30,
        trail_stop_pct=0.10, max_position=1.10, initial_size=0.60,
        rsi_exit=75, ema_period=21, bb_period=20, bb_std=2.0,
        roc_threshold=0.0, pyramid_size=0.20, trim_pct=0.25,
        atr_exit_mult=2.0, macro_weight=1.0,
    ),
    "LINK": dict(
        ticker="LINK-USD", sma_slow=190, momentum_period=25, rsi_entry=52,
        trail_stop_pct=0.086, max_position=1.50, initial_size=0.60,
        rsi_exit=75, ema_period=21, bb_period=20, bb_std=2.0,
        roc_threshold=0.0, pyramid_size=0.20, trim_pct=0.25,
        atr_exit_mult=2.0, macro_weight=1.0,
    ),
}

DEFAULT_LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
DEFAULT_SHORT_PARAMS = dict(
    short_max_size=0.5,
    short_rsi_entry=65,
    short_rsi_exit=25,
    short_trail_stop=0.12,
    short_sma_slow=None,  # uses asset sma_slow if None
)
DEFAULT_SAFETY_PARAMS = dict(
    max_portfolio_leverage=2.5,
    vol_ceiling=1.0,  # 100% annualized
    vol_lookback=20,
    dd_reduction_threshold=0.10,
    dd_reduction_factor=0.5,
)

TX_COST = 0.001  # 0.1% per trade
SHORT_BORROW_COST_DAILY = 0.0003  # 0.03%/day ~11%/yr
FUNDING_CARRY_DAILY = 0.0001  # 0.01%/day default


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid - std * s, mid, mid + std * s


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _realized_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Annualized realized volatility."""
    ret = close.pct_change()
    return ret.rolling(lookback).std() * np.sqrt(365)  # crypto trades 365 days/yr


# ---------------------------------------------------------------------------
# Confluence Engine
# ---------------------------------------------------------------------------

def compute_confluence(
    crypto_close: pd.Series,
    macro_data: pd.DataFrame,
    cross_asset_data: pd.DataFrame,
    sma_slow: int = 100,
    momentum_period: int = 35,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute daily confluence score 0-5 from 5 independent signals.
    All signals are lagged by 1 day to avoid lookahead.

    Args:
        crypto_close: daily close price for the crypto asset
        macro_data: DataFrame with columns: m2, yield_curve, fed_funds, cpi, hy_spread
        cross_asset_data: DataFrame with columns: gold, dxy, bonds, hyg, copper

    Returns:
        (confluence_score: Series 0-5, breakdown: DataFrame with individual signals)
    """
    idx = crypto_close.index
    n = len(idx)

    # Initialize component signals
    sig1 = pd.Series(0, index=idx, dtype=int)  # M2 acceleration
    sig2 = pd.Series(0, index=idx, dtype=int)  # Liquidity proxy
    sig3 = pd.Series(0, index=idx, dtype=int)  # Yield curve
    sig4 = pd.Series(0, index=idx, dtype=int)  # Cross-asset momentum
    sig5 = pd.Series(0, index=idx, dtype=int)  # Crypto momentum

    # Signal 1: M2 Acceleration (monthly, ffill daily)
    if macro_data is not None and "m2" in macro_data.columns:
        m2 = macro_data["m2"].reindex(idx, method="ffill").ffill()
        m2_yoy = m2.pct_change(365).fillna(0)
        m2_yoy_6m_ma = m2_yoy.rolling(180).mean()
        sig1 = (m2_yoy > m2_yoy_6m_ma).astype(int).fillna(0).astype(int)

    # Signal 2: Real-Time Liquidity Proxy (20d lookback, 3-of-4 consensus)
    # NOTE: 120d proxy wins in isolation (OOS 1.12) but 20d wins INSIDE V3
    # system (OOS 0.54 vs 0.31) because V3's other 4 signals already capture
    # macro trends — the fast 20d proxy adds complementary crash detection.
    # Validated via 3-way comparison: 2026-02-13
    # DXY declining + Gold rising + Bonds (TLT) rising + HYG rising
    PROXY_LOOKBACK = 20   # Fast — complements slow M2/yield/cross-asset signals
    PROXY_THRESHOLD = 3   # 3-of-4 consensus (stricter = fewer false positives)
    if cross_asset_data is not None:
        liq_score = pd.Series(0.0, index=idx)
        if "dxy" in cross_asset_data.columns:
            dxy = cross_asset_data["dxy"].reindex(idx, method="ffill").ffill()
            liq_score += (dxy.pct_change(PROXY_LOOKBACK) < 0).astype(float).fillna(0)
        if "gold" in cross_asset_data.columns:
            gold = cross_asset_data["gold"].reindex(idx, method="ffill").ffill()
            liq_score += (gold.pct_change(PROXY_LOOKBACK) > 0).astype(float).fillna(0)
        if "bonds" in cross_asset_data.columns:
            bonds = cross_asset_data["bonds"].reindex(idx, method="ffill").ffill()
            liq_score += (bonds.pct_change(PROXY_LOOKBACK) > 0).astype(float).fillna(0)
        if "hyg" in cross_asset_data.columns:
            hyg = cross_asset_data["hyg"].reindex(idx, method="ffill").ffill()
            liq_score += (hyg.pct_change(PROXY_LOOKBACK) > 0).astype(float).fillna(0)
        sig2 = (liq_score >= PROXY_THRESHOLD).astype(int)

    # Signal 3: Yield Curve (T10Y2Y > 0 OR steepening from inversion)
    if macro_data is not None and "yield_curve" in macro_data.columns:
        yc = macro_data["yield_curve"].reindex(idx, method="ffill").ffill()
        yc_positive = yc > 0
        yc_steepening = yc.diff(20) > 0  # Steepening over 20 days
        yc_was_inverted = yc.rolling(60).min() < 0  # Was inverted recently
        sig3 = (yc_positive | (yc_steepening & yc_was_inverted)).astype(int).fillna(0).astype(int)

    # Signal 4: Cross-Asset Momentum (Gold breakout + DXY breakdown + Copper/Gold rising)
    if cross_asset_data is not None:
        xam_score = pd.Series(0.0, index=idx)
        lookback_long = 60
        if "gold" in cross_asset_data.columns:
            gold = cross_asset_data["gold"].reindex(idx, method="ffill").ffill()
            gold_sma = gold.rolling(lookback_long).mean()
            xam_score += (gold > gold_sma).astype(float).fillna(0)
        if "dxy" in cross_asset_data.columns:
            dxy = cross_asset_data["dxy"].reindex(idx, method="ffill").ffill()
            dxy_sma = dxy.rolling(lookback_long).mean()
            xam_score += (dxy < dxy_sma).astype(float).fillna(0)
        if "copper" in cross_asset_data.columns and "gold" in cross_asset_data.columns:
            copper = cross_asset_data["copper"].reindex(idx, method="ffill").ffill()
            gold = cross_asset_data["gold"].reindex(idx, method="ffill").ffill()
            ratio = (copper / gold.replace(0, np.nan)).fillna(method="ffill")
            xam_score += (ratio.pct_change(lookback_long) > 0).astype(float).fillna(0)
        sig4 = (xam_score >= 2).astype(int)

    # Signal 5: Crypto Momentum (price > SMA + ROC > 0)
    sma = crypto_close.rolling(sma_slow).mean()
    roc = crypto_close.pct_change(momentum_period)
    sig5 = ((crypto_close > sma) & (roc > 0)).astype(int).fillna(0).astype(int)

    # Lag all signals by 1 day (no lookahead)
    sig1 = sig1.shift(1).fillna(0).astype(int)
    sig2 = sig2.shift(1).fillna(0).astype(int)
    sig3 = sig3.shift(1).fillna(0).astype(int)
    sig4 = sig4.shift(1).fillna(0).astype(int)
    sig5 = sig5.shift(1).fillna(0).astype(int)

    confluence = sig1 + sig2 + sig3 + sig4 + sig5
    breakdown = pd.DataFrame({
        "m2_accel": sig1, "liquidity_proxy": sig2, "yield_curve": sig3,
        "cross_asset_mom": sig4, "crypto_momentum": sig5, "confluence": confluence,
    }, index=idx)

    return confluence, breakdown


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

def detect_regime(
    confluence: pd.Series,
    prev_regime: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Map confluence score to regime string.
    ACCUMULATION = transition from BEAR (confluence was 0 recently, now rising).
    """
    regime = pd.Series("NEUTRAL", index=confluence.index)
    regime[confluence >= 4] = "BULL"
    regime[(confluence >= 2) & (confluence < 4)] = "MILD_BULL"
    regime[confluence == 1] = "NEUTRAL"
    regime[confluence == 0] = "BEAR"

    # Detect ACCUMULATION: confluence was 0 in last 30 days and now >= 2
    was_bear = (confluence.rolling(30).min() == 0)
    regime[(was_bear) & (confluence >= 2) & (confluence < 4)] = "ACCUMULATION"

    return regime


# ---------------------------------------------------------------------------
# Long Signal Generation
# ---------------------------------------------------------------------------

def generate_long_signals(
    df: pd.DataFrame,
    regime: pd.Series,
    confluence: pd.Series,
    **params,
) -> pd.DataFrame:
    """
    Long side: dip-buying within uptrends with adaptive sizing (from V2 enhanced).
    Returns DataFrame: position_size (0 to max_position), entry_type
    """
    p = {**ASSET_CONFIGS.get("BTC", {}), **params}
    close = df["close"].copy()
    n = len(close)

    # Indicators
    sma_slow = close.rolling(p["sma_slow"]).mean()
    roc = close.pct_change(p["momentum_period"])
    rsi = _rsi(close, 14)
    ema = close.ewm(span=p["ema_period"], adjust=False).mean()
    bb_lower, bb_mid, bb_upper = _bb(close, p["bb_period"], p["bb_std"])
    atr = _atr(df, 14)

    # Trend regime from crypto momentum (for long side, require price > SMA)
    trend_up = (close > sma_slow).shift(1).fillna(False)
    momentum_pos = (roc > p["roc_threshold"]).shift(1).fillna(False)

    # Dip conditions (shifted to avoid lookahead)
    rsi_shifted = rsi.shift(1).fillna(50)
    ema_shifted = ema.shift(1).fillna(close)
    bb_lower_shifted = bb_lower.shift(1).fillna(close)
    bb_upper_shifted = bb_upper.shift(1).fillna(close)
    atr_shifted = atr.shift(1).fillna(0)

    rsi_dip = rsi_shifted < p["rsi_entry"]
    ema_dip = close.shift(1) < ema_shifted
    bb_dip = close.shift(1) <= bb_lower_shifted
    rsi_hot = rsi_shifted > p["rsi_exit"]
    bb_hot = close.shift(1) > bb_upper_shifted

    # Long is allowed in BULL, MILD_BULL, ACCUMULATION
    long_allowed = regime.isin(["BULL", "MILD_BULL", "ACCUMULATION"])
    in_trend = trend_up & momentum_pos & long_allowed

    any_dip = rsi_dip | ema_dip | bb_dip

    # Simulate pyramiding
    pos_sizes = np.zeros(n, dtype=float)
    entry_types = [""] * n
    pos = 0.0
    avg_entry = 0.0
    total_cost = 0.0
    peak_eq = 1.0
    eq = 1.0

    warmup = max(p["sma_slow"], p["bb_period"], p["momentum_period"]) + 2

    for i in range(warmup, n):
        price = close.iloc[i]
        prev_price = close.iloc[i - 1] if i > 0 else price

        # Update equity
        if pos > 0 and prev_price > 0:
            daily_ret = (price - prev_price) / prev_price
            eq *= (1 + daily_ret * pos)
        peak_eq = max(peak_eq, eq)

        # Trailing stop
        if pos > 0:
            dd = 1 - eq / peak_eq
            if dd >= p["trail_stop_pct"]:
                entry_types[i] = "long_trail_stop"
                pos = 0.0
                avg_entry = 0.0
                total_cost = 0.0
                pos_sizes[i] = 0.0
                continue

        # Regime exit: if regime turns BEAR or NEUTRAL, close longs
        if pos > 0 and regime.iloc[i] in ("BEAR", "NEUTRAL"):
            entry_types[i] = "regime_exit"
            pos = 0.0
            avg_entry = 0.0
            total_cost = 0.0
            pos_sizes[i] = 0.0
            continue

        # Trim on overbought
        if pos > 0:
            trimmed = False
            if rsi_hot.iloc[i]:
                pos -= pos * p["trim_pct"]
                entry_types[i] = "trim_rsi"
                trimmed = True
            elif bb_hot.iloc[i]:
                pos -= pos * p["trim_pct"]
                entry_types[i] = "trim_bb"
                trimmed = True
            elif avg_entry > 0 and atr_shifted.iloc[i] > 0:
                ext = (price - avg_entry) / atr_shifted.iloc[i]
                if ext > p["atr_exit_mult"]:
                    pos -= pos * p["trim_pct"]
                    entry_types[i] = "trim_atr"
                    trimmed = True
            if trimmed:
                pos = max(pos, 0.0)
                if pos < 0.01:
                    pos = 0.0
                    avg_entry = 0.0
                    total_cost = 0.0
                pos_sizes[i] = pos
                continue

        # Entry / pyramid
        if in_trend.iloc[i] and any_dip.iloc[i]:
            max_pos = p["max_position"]
            if pos == 0:
                add = p["initial_size"]
                add = min(add, max_pos)
                pos = add
                avg_entry = price
                total_cost = price * add
                entry_types[i] = "long_entry"
            elif pos < max_pos:
                add = min(p["pyramid_size"], max_pos - pos)
                if add > 0.01:
                    total_cost += price * add
                    pos += add
                    avg_entry = total_cost / pos
                    entry_types[i] = "long_pyramid"

        pos_sizes[i] = pos

    result = pd.DataFrame({
        "long_position": pos_sizes,
        "long_entry_type": entry_types,
    }, index=df.index)
    return result


# ---------------------------------------------------------------------------
# Short Signal Generation
# ---------------------------------------------------------------------------

def generate_short_signals(
    df: pd.DataFrame,
    regime: pd.Series,
    confluence: pd.Series,
    m2_decelerating: Optional[pd.Series] = None,
    funding_zscore: Optional[pd.Series] = None,
    **params,
) -> pd.DataFrame:
    """
    Short side: fade rallies in downtrends, earn funding carry.
    Returns DataFrame: short_position (0 to short_max_size), short_entry_type
    """
    sp = {**DEFAULT_SHORT_PARAMS, **params}
    close = df["close"].copy()
    n = len(close)
    sma_period = sp.get("short_sma_slow") or sp.get("sma_slow", 100)

    sma = close.rolling(sma_period).mean()
    rsi = _rsi(close, 14)

    # M2 decelerating (external or default False — conservative)
    if m2_decelerating is not None:
        m2_dec = m2_decelerating.reindex(df.index, method="ffill").fillna(False).astype(bool)
    else:
        m2_dec = pd.Series(False, index=df.index)

    # Funding z-score
    if funding_zscore is not None:
        fz = funding_zscore.reindex(df.index, method="ffill").fillna(0)
    else:
        fz = pd.Series(0.0, index=df.index)

    # Shift for no lookahead
    sma_shifted = sma.shift(1).fillna(close)
    rsi_shifted = rsi.shift(1).fillna(50)
    m2_dec_shifted = m2_dec.shift(1).fillna(False)
    fz_shifted = fz.shift(1).fillna(0)
    close_shifted = close.shift(1).fillna(close)

    # Short allowed in BEAR regime, or NEUTRAL with low confluence
    short_allowed = regime.isin(["BEAR"]) | ((regime == "NEUTRAL") & (confluence <= 1))

    # Short entry conditions
    downtrend = close_shifted < sma_shifted  # price below SMA
    rsi_overbought = rsi_shifted > sp["short_rsi_entry"]
    sma_rejection = (close_shifted < sma_shifted) & (close.shift(2).fillna(0) < sma.shift(2).fillna(0))
    funding_overleveraged = fz_shifted > 1.5

    short_trigger = rsi_overbought | sma_rejection | funding_overleveraged

    # All conditions for short entry
    short_entry_cond = short_allowed & m2_dec_shifted & downtrend & short_trigger

    # Scale short size by confluence
    def _short_size(conf_val):
        if conf_val == 0:
            return sp["short_max_size"]
        elif conf_val == 1:
            return sp["short_max_size"] * 0.5
        return 0.0

    # Simulate short positions
    short_positions = np.zeros(n, dtype=float)
    short_entry_types = [""] * n
    short_pos = 0.0
    short_entry_price = 0.0
    short_low = np.inf  # track lowest price for trailing stop

    warmup = sma_period + 5

    for i in range(warmup, n):
        price = close.iloc[i]

        # If in short position, check exits
        if short_pos > 0:
            short_low = min(short_low, price)

            # RSI oversold → cover
            if rsi_shifted.iloc[i] < sp["short_rsi_exit"]:
                short_entry_types[i] = "short_cover_rsi"
                short_pos = 0.0
                short_positions[i] = 0.0
                continue

            # Price reclaims SMA → cover
            if close_shifted.iloc[i] > sma_shifted.iloc[i]:
                short_entry_types[i] = "short_cover_sma"
                short_pos = 0.0
                short_positions[i] = 0.0
                continue

            # Trailing stop from low
            if short_low > 0:
                bounce = (price - short_low) / short_low
                if bounce >= sp["short_trail_stop"]:
                    short_entry_types[i] = "short_trail_stop"
                    short_pos = 0.0
                    short_positions[i] = 0.0
                    continue

            # M2 starts accelerating → close shorts
            if not m2_dec_shifted.iloc[i]:
                short_entry_types[i] = "short_cover_m2"
                short_pos = 0.0
                short_positions[i] = 0.0
                continue

        # Short entry
        if short_pos == 0 and i < n and short_entry_cond.iloc[i]:
            conf_val = int(confluence.iloc[i])
            size = _short_size(conf_val)
            if size > 0:
                short_pos = size
                short_entry_price = price
                short_low = price
                short_entry_types[i] = "short_entry"

        short_positions[i] = short_pos

    result = pd.DataFrame({
        "short_position": short_positions,
        "short_entry_type": short_entry_types,
    }, index=df.index)
    return result


# ---------------------------------------------------------------------------
# Adaptive Leverage
# ---------------------------------------------------------------------------

def adaptive_leverage(
    confluence: pd.Series,
    realized_vol: pd.Series,
    current_dd: pd.Series,
    leverage_map: Optional[Dict[int, float]] = None,
    **params,
) -> pd.Series:
    """
    Map confluence to leverage with safety overrides.
    Returns daily leverage multiplier.
    """
    sp = {**DEFAULT_SAFETY_PARAMS, **params}
    lmap = leverage_map or DEFAULT_LEVERAGE_MAP

    # Base leverage from confluence
    lev = confluence.map(lambda c: lmap.get(min(int(c), 5), lmap.get(0, 0.0)))

    # Safety 1: vol ceiling — halve if realized vol > ceiling
    vol_ceil = sp["vol_ceiling"]
    high_vol = realized_vol > vol_ceil
    lev = lev.where(~high_vol, lev * 0.5)

    # Safety 2: drawdown reduction
    dd_thresh = sp["dd_reduction_threshold"]
    dd_factor = sp["dd_reduction_factor"]
    in_dd = current_dd.abs() > dd_thresh
    lev = lev.where(~in_dd, lev * dd_factor)

    # Safety 3: cap at max portfolio leverage
    lev = lev.clip(upper=sp["max_portfolio_leverage"])

    return lev


# ---------------------------------------------------------------------------
# Full Strategy Runner
# ---------------------------------------------------------------------------

def run_single_asset(
    df: pd.DataFrame,
    macro_data: pd.DataFrame,
    cross_asset_data: pd.DataFrame,
    funding_data: Optional[pd.Series] = None,
    asset_name: str = "BTC",
    leverage_map: Optional[Dict[int, float]] = None,
    enable_long: bool = True,
    enable_short: bool = True,
    enable_adaptive_leverage: bool = True,
    funding_carry_daily: float = FUNDING_CARRY_DAILY,
    **override_params,
) -> pd.DataFrame:
    """
    Run full strategy for a single asset.
    Returns DataFrame with: position, leverage, regime, confluence,
    long_pos, short_pos, daily_pnl, long_pnl, short_pnl, funding_pnl
    """
    # Merge asset params with overrides
    asset_cfg = {**ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"]), **override_params}
    close = df["close"]
    idx = df.index

    # 1. Confluence
    confluence, breakdown = compute_confluence(
        close, macro_data, cross_asset_data,
        sma_slow=asset_cfg["sma_slow"],
        momentum_period=asset_cfg["momentum_period"],
    )

    # 2. Regime
    regime = detect_regime(confluence)

    # 3. M2 decelerating for short side
    m2_dec = pd.Series(False, index=idx)
    if macro_data is not None and "m2" in macro_data.columns:
        m2 = macro_data["m2"].reindex(idx, method="ffill").ffill()
        m2_yoy = m2.pct_change(365).fillna(0)
        m2_yoy_6m_ma = m2_yoy.rolling(180).mean()
        m2_dec = (m2_yoy < m2_yoy_6m_ma).shift(1).fillna(False)

    # 4. Long signals
    if enable_long:
        long_df = generate_long_signals(df, regime, confluence, **asset_cfg)
    else:
        long_df = pd.DataFrame({"long_position": 0.0, "long_entry_type": ""}, index=idx)

    # 5. Short signals
    if enable_short:
        funding_z = None
        if funding_data is not None:
            mean = funding_data.rolling(30).mean()
            std = funding_data.rolling(30).std().replace(0, np.nan)
            funding_z = ((funding_data - mean) / std).fillna(0)
        short_df = generate_short_signals(
            df, regime, confluence, m2_decelerating=m2_dec,
            funding_zscore=funding_z, sma_slow=asset_cfg["sma_slow"],
            **{k: v for k, v in override_params.items() if k.startswith("short_")},
        )
    else:
        short_df = pd.DataFrame({"short_position": 0.0, "short_entry_type": ""}, index=idx)

    # 6. Realized vol and drawdown for leverage calc
    rvol = _realized_vol(close, DEFAULT_SAFETY_PARAMS["vol_lookback"])

    # Compute running drawdown (from cumulative return)
    daily_ret = close.pct_change().fillna(0)
    cum_ret = (1 + daily_ret).cumprod()
    running_dd = cum_ret / cum_ret.cummax() - 1

    # 7. Adaptive leverage
    if enable_adaptive_leverage:
        lev = adaptive_leverage(
            confluence, rvol, running_dd,
            leverage_map=leverage_map,
            **{k: v for k, v in override_params.items()
               if k in DEFAULT_SAFETY_PARAMS},
        )
    else:
        lev = pd.Series(1.0, index=idx)

    # 8. Combine: net position = long - short, scaled by leverage
    long_pos = long_df["long_position"].fillna(0)
    short_pos = short_df["short_position"].fillna(0)

    # Apply leverage to long side
    long_pos_levered = long_pos * lev
    # Shorts don't get extra leverage (already sized conservatively)
    short_pos_final = short_pos

    net_position = long_pos_levered - short_pos_final

    # 9. Daily P&L simulation
    daily_ret_series = close.pct_change().fillna(0)
    pos_change_long = long_pos_levered.diff().abs().fillna(0)
    pos_change_short = short_pos_final.diff().abs().fillna(0)

    long_pnl = daily_ret_series * long_pos_levered - pos_change_long * TX_COST
    short_pnl = -daily_ret_series * short_pos_final - pos_change_short * TX_COST

    # Short borrowing cost
    short_borrow = short_pos_final * SHORT_BORROW_COST_DAILY
    short_pnl -= short_borrow

    # Funding carry (shorts earn when funding is positive)
    funding_pnl = short_pos_final * funding_carry_daily

    total_pnl = long_pnl + short_pnl + funding_pnl

    result = pd.DataFrame({
        "position": net_position,
        "long_position": long_pos_levered,
        "short_position": short_pos_final,
        "leverage": lev,
        "regime": regime,
        "confluence": confluence,
        "daily_pnl": total_pnl,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "funding_pnl": funding_pnl,
    }, index=idx)

    return result


def run_full_strategy(
    asset_data: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    cross_asset_data: pd.DataFrame,
    funding_data: Optional[Dict[str, pd.Series]] = None,
    asset_configs: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None,
    leverage_map: Optional[Dict[int, float]] = None,
    enable_long: bool = True,
    enable_short: bool = True,
    enable_adaptive_leverage: bool = True,
    funding_carry_daily: float = FUNDING_CARRY_DAILY,
    **override_params,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Complete multi-asset strategy: long + short + adaptive leverage.

    Returns:
        portfolio_df: DataFrame with portfolio-level daily_pnl, equity, regime, etc.
        per_asset: dict of per-asset result DataFrames
    """
    configs = asset_configs or ASSET_CONFIGS
    if weights is None:
        weights = {a: 1.0 / len(asset_data) for a in asset_data}

    per_asset = {}
    for asset, df in asset_data.items():
        cfg = configs.get(asset, configs.get("BTC", {}))
        asset_params = {k: v for k, v in cfg.items() if k != "ticker"}
        asset_params.update(override_params)

        fd = None
        if funding_data and asset in funding_data:
            fd = funding_data[asset]

        res = run_single_asset(
            df, macro_data, cross_asset_data,
            funding_data=fd, asset_name=asset,
            leverage_map=leverage_map,
            enable_long=enable_long,
            enable_short=enable_short,
            enable_adaptive_leverage=enable_adaptive_leverage,
            funding_carry_daily=funding_carry_daily,
            **asset_params,
        )
        per_asset[asset] = res

    # Portfolio aggregation
    assets = list(per_asset.keys())
    common_idx = per_asset[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(per_asset[a].index)

    portfolio_pnl = pd.Series(0.0, index=common_idx)
    portfolio_long_pnl = pd.Series(0.0, index=common_idx)
    portfolio_short_pnl = pd.Series(0.0, index=common_idx)
    portfolio_funding_pnl = pd.Series(0.0, index=common_idx)
    total_leverage = pd.Series(0.0, index=common_idx)

    for a in assets:
        w = weights.get(a, 1.0 / len(assets))
        res = per_asset[a].reindex(common_idx)
        portfolio_pnl += res["daily_pnl"] * w
        portfolio_long_pnl += res["long_pnl"].fillna(0) * w
        portfolio_short_pnl += res["short_pnl"].fillna(0) * w
        portfolio_funding_pnl += res["funding_pnl"].fillna(0) * w
        total_leverage += (res["long_position"].fillna(0) + res["short_position"].fillna(0)) * w

    equity = (1 + portfolio_pnl).cumprod()

    # Use BTC's regime as portfolio regime (dominant asset)
    regime = per_asset.get("BTC", per_asset[assets[0]]).reindex(common_idx)["regime"]
    confluence = per_asset.get("BTC", per_asset[assets[0]]).reindex(common_idx)["confluence"]

    portfolio_df = pd.DataFrame({
        "daily_pnl": portfolio_pnl,
        "equity": equity,
        "long_pnl": portfolio_long_pnl,
        "short_pnl": portfolio_short_pnl,
        "funding_pnl": portfolio_funding_pnl,
        "total_leverage": total_leverage,
        "regime": regime,
        "confluence": confluence,
    }, index=common_idx)

    return portfolio_df, per_asset
