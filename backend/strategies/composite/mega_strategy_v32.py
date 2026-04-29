"""
MegaStrategyV3.2 — E3 Configuration: DipBuy + Pyramid + Per-Asset Trails

Evolution from V3.1-H2:
- KEPT: Direct sizing from confluence score (V3.1 fix)
- KEPT: Score 4 CryptoMom demotion to 0.5x
- KEPT: Vol ceiling (halve when BTC 30d vol > 80%)
- KEPT: Bear filter (flat after 30 consecutive days at score <= 1)
- ADDED: RSI<30 dip-buying — add 0.3x to existing positions on deep dips
- ADDED: 20d breakout pyramiding — add 0.2x when price breaks above 20d high
- REPLACED: Fixed 10% trail stop → per-asset trail stops (BTC 12%, ETH 15%, SOL 8%, LINK 8%)
- Additional position cap at 2.5x to prevent over-leverage

Key difference from V3's broken dip-buying:
V3 used RSI/BB as ENTRY GATES → blocked capital deployment (flat 72% of time)
V3.2 uses RSI/BB as POSITION ADDITIONS → enhances already-deployed capital

Performance:
  V3.1-H2: IS Sharpe 1.251, OOS 0.447, CAGR 41.8%, MaxDD -40.9%
  V3.2-E3: IS Sharpe 1.948, OOS 1.166, CAGR 106.9%, MaxDD -29.7%, p=0.004
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from strategies.composite.mega_strategy_v3 import (
    compute_confluence, detect_regime, ASSET_CONFIGS,
    _realized_vol, TX_COST,
)

NAME = "MegaStrategyV3.2"
CATEGORY = "composite"
DESCRIPTION = "E3: DipBuy + Pyramid + Per-Asset Trails on direct sizing base"

# ── Configuration ─────────────────────────────────────────────────

# Base leverage from confluence score (unchanged from V3.1)
LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}

# Score 4 CryptoMom demotion (unchanged from V3.1)
SCORE4_CRYPTO_MOM_OVERRIDE = 0.5

# Vol ceiling (unchanged from V3.1)
VOL_CEILING = 0.80
VOL_LOOKBACK = 30

# Bear filter (unchanged from V3.1)
BEAR_FILTER_DAYS = 30

# ── NEW: Dip-buying (position addition, NOT entry gate) ──────────
DIP_BUY_RSI_PERIOD = 14
DIP_BUY_RSI_THRESHOLD = 30    # Add when RSI < 30
DIP_BUY_ADD_LEVERAGE = 0.3    # Add 0.3x on dip
MAX_TOTAL_LEVERAGE = 2.5       # Hard cap on total exposure

# ── NEW: Pyramiding (20d breakout) ───────────────────────────────
PYRAMID_LOOKBACK = 20          # Buy breakout above 20d high
PYRAMID_ADD_LEVERAGE = 0.2     # Add 0.2x on breakout
# MAX_TOTAL_LEVERAGE shared with dip-buying

# ── NEW: Per-asset trail stops (replaces fixed 10%) ──────────────
PER_ASSET_TRAIL_STOPS = {
    "BTC": 0.12,   # 12% — lower vol, tighter stop
    "ETH": 0.15,   # 15% — higher vol, more room
    "SOL": 0.08,   # 8% — high vol but smaller position
    "LINK": 0.08,  # 8% — high vol but smallest position
}
DEFAULT_TRAIL_STOP = 0.10

TRAIL_REDUCE_FACTOR = 0.3         # Reduce to 30% of target during drawdown
TRAIL_RECOVERY_DAYS = 30
TRAIL_RECOVERY_THRESHOLD = 0.95

WEIGHTS = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15}


# ── Helper: RSI ───────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Core Engine ───────────────────────────────────────────────────

def run_full_strategy(
    asset_data: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    cross_asset_data: pd.DataFrame,
    leverage_map: Optional[Dict[int, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    vol_ceiling: float = VOL_CEILING,
    vol_lookback: int = VOL_LOOKBACK,
    bear_filter_days: int = BEAR_FILTER_DAYS,
    s4_override: float = SCORE4_CRYPTO_MOM_OVERRIDE,
    dip_rsi_period: int = DIP_BUY_RSI_PERIOD,
    dip_rsi_threshold: float = DIP_BUY_RSI_THRESHOLD,
    dip_add_leverage: float = DIP_BUY_ADD_LEVERAGE,
    pyramid_lookback: int = PYRAMID_LOOKBACK,
    pyramid_add_leverage: float = PYRAMID_ADD_LEVERAGE,
    max_total_leverage: float = MAX_TOTAL_LEVERAGE,
    per_asset_trail_stops: Optional[Dict[str, float]] = None,
    trail_reduce_factor: float = TRAIL_REDUCE_FACTOR,
    trail_recovery_days: int = TRAIL_RECOVERY_DAYS,
    trail_recovery_threshold: float = TRAIL_RECOVERY_THRESHOLD,
    tx_cost: float = TX_COST,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Full multi-asset V3.2 strategy with dip-buying, pyramiding, and per-asset trails.
    """
    w = weights or WEIGHTS
    lmap = leverage_map or LEVERAGE_MAP
    trail_stops = per_asset_trail_stops or PER_ASSET_TRAIL_STOPS
    assets = list(asset_data.keys())

    # Common index
    common_idx = asset_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(asset_data[a].index)

    # Pre-compute per-asset signals
    per_asset_conf = {}
    per_asset_lev = {}
    per_asset_bd = {}
    per_asset_ret = {}
    per_asset_close = {}
    per_asset_rsi = {}
    per_asset_high20 = {}

    for asset in assets:
        df = asset_data[asset].reindex(common_idx)
        close = df["close"]

        cfg = ASSET_CONFIGS.get(asset, ASSET_CONFIGS["BTC"])
        conf, bd = compute_confluence(
            close, macro_data, cross_asset_data,
            sma_slow=cfg["sma_slow"], momentum_period=cfg["momentum_period"]
        )
        conf = conf.reindex(common_idx).fillna(0)
        bd = bd.reindex(common_idx).fillna(0)

        # Base leverage
        base_lev = conf.map(lambda c: lmap.get(min(int(c), 5), 0.0))
        is_s4 = conf == 4
        cm_off = bd["crypto_momentum"] == 0
        base_lev = base_lev.where(~(is_s4 & cm_off), s4_override)
        base_lev = base_lev.shift(1).fillna(0)

        per_asset_conf[asset] = conf
        per_asset_lev[asset] = base_lev
        per_asset_bd[asset] = bd
        per_asset_ret[asset] = close.pct_change().reindex(common_idx).fillna(0)
        per_asset_close[asset] = close

        # Dip-buying indicator: RSI
        per_asset_rsi[asset] = _rsi(close, dip_rsi_period)

        # Pyramiding indicator: 20d high
        per_asset_high20[asset] = close.rolling(pyramid_lookback).max().shift(1)

    # BTC drives portfolio regime
    btc_conf = per_asset_conf.get("BTC", per_asset_conf[assets[0]])
    regime = detect_regime(btc_conf)

    # ── Day-by-day simulation ─────────────────────────────────────
    n = len(common_idx)
    daily_pnl = np.zeros(n)
    total_leverage = np.zeros(n)
    regimes = [""] * n
    confluences = np.zeros(n)

    equity = 1.0
    peak_eq = 1.0

    # Per-asset drawdown tracking
    asset_peaks = {a: 1.0 for a in assets}
    asset_in_dd = {a: False for a in assets}
    asset_dd_start = {a: 0 for a in assets}

    consecutive_low = 0
    prev_exposure = 0.0

    for i in range(1, n):
        # ── Layer 1: Base exposure from confluence ──
        base_exposure = 0.0
        per_asset_exposure = {}
        for asset in assets:
            asset_lev = per_asset_lev[asset].iloc[i]
            asset_exp = w.get(asset, 0.25) * asset_lev
            per_asset_exposure[asset] = asset_exp
            base_exposure += asset_exp

        # ── Layer 2: Dip-buying (add to existing positions) ──
        for asset in assets:
            if per_asset_exposure[asset] > 0:  # Already long
                rsi_val = per_asset_rsi[asset].iloc[i - 1]  # Lagged
                if not np.isnan(rsi_val) and rsi_val < dip_rsi_threshold:
                    add = w.get(asset, 0.25) * dip_add_leverage
                    per_asset_exposure[asset] += add

        # ── Layer 3: Pyramiding (20d breakout) ──
        for asset in assets:
            if per_asset_exposure[asset] > 0:  # Already long
                close_prev = per_asset_close[asset].iloc[i - 1]
                high20 = per_asset_high20[asset].iloc[i - 1]
                if not np.isnan(high20) and close_prev > high20:
                    add = w.get(asset, 0.25) * pyramid_add_leverage
                    per_asset_exposure[asset] += add

        # Total exposure with cap
        target_exposure = sum(per_asset_exposure.values())
        if target_exposure > max_total_leverage:
            scale = max_total_leverage / target_exposure
            per_asset_exposure = {a: v * scale for a, v in per_asset_exposure.items()}
            target_exposure = max_total_leverage

        # ── Layer 4: Vol ceiling ──
        if vol_ceiling > 0:
            btc_close = per_asset_close.get("BTC", per_asset_close[assets[0]])
            if i >= vol_lookback:
                window = btc_close.iloc[max(0, i - vol_lookback):i]
                rvol = window.pct_change().std() * np.sqrt(252)
                if rvol > vol_ceiling:
                    target_exposure *= 0.5
                    per_asset_exposure = {a: v * 0.5 for a, v in per_asset_exposure.items()}

        # ── Layer 5: Bear filter ──
        btc_score = int(btc_conf.iloc[i - 1])
        if btc_score <= 1:
            consecutive_low += 1
        else:
            consecutive_low = 0

        if consecutive_low >= bear_filter_days:
            target_exposure = 0.0
            per_asset_exposure = {a: 0.0 for a in assets}

        # ── Layer 6: Per-asset trailing stops ──
        for asset in assets:
            asset_eq = per_asset_close[asset].iloc[i - 1]
            if asset_eq > asset_peaks[asset]:
                asset_peaks[asset] = asset_eq

            trail_pct = trail_stops.get(asset, DEFAULT_TRAIL_STOP)
            dd_from_peak = 1 - asset_eq / asset_peaks[asset] if asset_peaks[asset] > 0 else 0

            if dd_from_peak > trail_pct:
                if not asset_in_dd[asset]:
                    asset_in_dd[asset] = True
                    asset_dd_start[asset] = i
                per_asset_exposure[asset] *= trail_reduce_factor
            elif asset_in_dd[asset]:
                days_since = i - asset_dd_start[asset]
                if days_since > trail_recovery_days and dd_from_peak < (1 - trail_recovery_threshold):
                    asset_in_dd[asset] = False
                    asset_peaks[asset] = asset_eq  # Reset peak

        target_exposure = sum(per_asset_exposure.values())

        # ── Compute P&L ──
        port_ret = 0.0
        exp_sum = sum(per_asset_exposure.values())
        for asset in assets:
            if exp_sum > 0:
                asset_exposure = per_asset_exposure[asset]
            else:
                asset_exposure = 0.0
            ret = per_asset_ret[asset].iloc[i]
            port_ret += asset_exposure * ret

        # Transaction costs
        lev_change = abs(target_exposure - prev_exposure)
        port_ret -= lev_change * tx_cost
        prev_exposure = target_exposure

        daily_pnl[i] = port_ret
        total_leverage[i] = target_exposure
        regimes[i] = regime.iloc[i] if i < len(regime) else ""
        confluences[i] = btc_conf.iloc[i] if i < len(btc_conf) else 0

        equity *= (1 + port_ret)
        peak_eq = max(peak_eq, equity)

    # Build output
    portfolio_df = pd.DataFrame({
        "daily_pnl": daily_pnl,
        "equity": np.cumprod(1 + daily_pnl),
        "long_pnl": daily_pnl,
        "short_pnl": 0.0,
        "funding_pnl": 0.0,
        "total_leverage": total_leverage,
        "regime": regimes,
        "confluence": confluences,
    }, index=common_idx)

    per_asset_results = {}
    for asset in assets:
        per_asset_results[asset] = pd.DataFrame({
            "confluence": per_asset_conf[asset],
            "base_leverage": per_asset_lev[asset],
            "regime": regime,
            "daily_return": per_asset_ret[asset],
        }, index=common_idx)

    return portfolio_df, per_asset_results
