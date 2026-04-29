"""
Macro Momentum V5 — Decoupled Architecture
============================================
V3.1 entangles entry and sizing through a 5-signal confluence score.
The M2 edge gets diluted: a strong M2 signal only adds +1 to a 0-5 score,
meaning M2 acceleration changes position size by at most 20%.

V5 fixes this by separating concerns:
  - ENTRY is pure momentum (SMA50 cross + ROC > 0). No macro gating.
  - SIZING is binary M2 regime (accelerating → 1.0x, not → 0.3x).
  - EXIT is a 15% trailing stop from peak.

M2 never prevents entry — it only scales how much capital is deployed.
This lets the momentum signal catch every trend while M2 controls risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# ─── Module Constants ────────────────────────────────────────────────────────

NAME = "MacroMomentumV5"
CATEGORY = "composite"
DESCRIPTION = "Decoupled architecture: momentum entry + M2 sizing + trailing stop"
REQUIRES_DERIVATIVES = False

SMA_WINDOW = 50
ROC_PERIOD = 20
TRAILING_STOP_PCT = 0.15
VOL_CEILING = 0.80
VOL_LOOKBACK = 30
TX_COST = 0.001
WEIGHTS = {"BTC": 0.70, "ETH": 0.30}


# ─── Entry Signal ────────────────────────────────────────────────────────────

def generate_entry_signal(close: pd.Series, sma_window: int = SMA_WINDOW,
                          roc_period: int = ROC_PERIOD) -> pd.Series:
    """Pure momentum entry: close > SMA(50) AND ROC(20) > 0. Shifted by 1 day."""
    sma = close.rolling(sma_window).mean()
    roc = close.pct_change(roc_period)
    signal = ((close > sma) & (roc > 0)).astype(float)
    signal.iloc[:sma_window] = 0  # warmup
    return signal.shift(1).fillna(0)


# ─── M2 Sizing ───────────────────────────────────────────────────────────────

def m2_sizing(macro_data: Optional[pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """Binary M2 regime sizing: 1.0x when accelerating, 0.3x when not. Shifted by 1 day."""
    if macro_data is None or "m2" not in macro_data.columns:
        return pd.Series(1.0, index=index)
    m2 = macro_data["m2"].reindex(index, method="ffill").ffill()
    m2_yoy = m2.pct_change(365).fillna(0)
    m2_yoy_6m_ma = m2_yoy.rolling(180).mean()
    accelerating = (m2_yoy > m2_yoy_6m_ma).astype(float)
    sizing = accelerating.where(accelerating == 1.0, 0.3)
    return sizing.shift(1).fillna(0.3)


# ─── Risk Layers ─────────────────────────────────────────────────────────────

def trailing_stop(close: pd.Series, position: pd.Series,
                  stop_pct: float = TRAILING_STOP_PCT) -> pd.Series:
    """Iterative trailing stop. Exits when price drops stop_pct from peak while in position."""
    result = position.copy()
    peak = close.iloc[0]
    in_position = False
    stopped_out = False

    for i in range(len(close)):
        if position.iloc[i] > 0 and not stopped_out:
            if not in_position:
                peak = close.iloc[i]
                in_position = True
            else:
                peak = max(peak, close.iloc[i])
            if close.iloc[i] < peak * (1 - stop_pct):
                result.iloc[i] = 0
                stopped_out = True
                in_position = False
        elif position.iloc[i] > 0 and stopped_out:
            result.iloc[i] = 0  # stay out until signal resets
        else:
            in_position = False
            stopped_out = False
            peak = close.iloc[i]

    return result


def vol_ceiling_filter(close: pd.Series, position: pd.Series,
                       lookback: int = VOL_LOOKBACK,
                       ceiling: float = VOL_CEILING) -> pd.Series:
    """Halve position when 30d annualized vol > 80%. Uses sqrt(365) for crypto."""
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(lookback).std() * np.sqrt(365)
    result = position.copy()
    high_vol = realized_vol > ceiling
    result[high_vol] = result[high_vol] * 0.5
    return result


# ─── Single Asset Runner ─────────────────────────────────────────────────────

def run_single_asset(df: pd.DataFrame, macro_data: Optional[pd.DataFrame],
                     asset_name: str, stop_pct: float = TRAILING_STOP_PCT) -> pd.DataFrame:
    """Run V5 on one asset. Returns DataFrame with daily returns and position info."""
    close = df["close"]
    idx = close.index

    # Entry: pure momentum
    entry = generate_entry_signal(close)

    # Sizing: M2 regime
    sizing = m2_sizing(macro_data, idx)

    # Raw position = entry * sizing
    position = entry * sizing

    # Risk layer 1: trailing stop
    position = trailing_stop(close, position, stop_pct)

    # Risk layer 2: vol ceiling
    position = vol_ceiling_filter(close, position)

    # Returns with transaction costs
    daily_ret = close.pct_change().fillna(0)
    pos_change = position.diff().abs().fillna(0)
    strategy_ret = position * daily_ret - pos_change * TX_COST

    return pd.DataFrame({
        "close": close,
        "position": position,
        "entry_signal": entry,
        "m2_sizing": sizing,
        "strategy_return": strategy_ret,
        "buy_hold_return": daily_ret,
    }, index=idx)


# ─── Portfolio Runner ─────────────────────────────────────────────────────────

def run_portfolio(asset_data: Dict[str, pd.DataFrame],
                  macro_data: Optional[pd.DataFrame],
                  weights: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run V5 across BTC/ETH portfolio. Returns (portfolio_df, per_asset_dict)."""
    w = weights or WEIGHTS
    per_asset = {}

    for asset, df in asset_data.items():
        if asset not in w:
            continue
        per_asset[asset] = run_single_asset(df, macro_data, asset)

    # Common index
    common_idx = per_asset[list(per_asset.keys())[0]].index
    for asset in list(per_asset.keys())[1:]:
        common_idx = common_idx.intersection(per_asset[asset].index)
    common_idx = common_idx.sort_values()

    # Weighted portfolio returns
    port_ret = pd.Series(0.0, index=common_idx)
    port_bh = pd.Series(0.0, index=common_idx)
    for asset in per_asset:
        r = per_asset[asset]["strategy_return"].reindex(common_idx).fillna(0)
        bh = per_asset[asset]["buy_hold_return"].reindex(common_idx).fillna(0)
        port_ret += r * w[asset]
        port_bh += bh * w[asset]

    port_equity = (1 + port_ret).cumprod()
    bh_equity = (1 + port_bh).cumprod()

    portfolio_df = pd.DataFrame({
        "strategy_return": port_ret,
        "buy_hold_return": port_bh,
        "strategy_equity": port_equity,
        "buy_hold_equity": bh_equity,
    }, index=common_idx)

    return portfolio_df, per_asset
