"""
MegaStrategyV3.1 — Direct Sizing with Protection Layers

Architecture change from V3:
- REMOVED: Dip-buying entry logic, pyramiding, RSI/BB entry filters
- ADDED: Confluence score → direct portfolio exposure
- ADDED: Score 4 smart demotion (0.5x when CryptoMom is off)
- ADDED: Portfolio-level trailing stop (10% DD → 30% exposure)
- ADDED: Bear filter (flat after 30 consecutive days at score ≤ 1)
- KEPT: Vol ceiling (halve when BTC 30d vol > 80%)
- KEPT: 5-signal confluence engine (unchanged from V3)
- KEPT: Per-asset weighting (BTC 40%, ETH 25%, SOL 20%, LINK 15%)

Rationale: V3 had IS Sharpe 1.71 but CAGR 9.3% because dip-buying logic
caused 72% flat time. The confluence signals were correct (score 3+ for 82% 
of 2024 when BTC did +94%) but positions were never entered.

V3.1 H2: IS Sharpe 1.25, OOS Sharpe 1.07, CAGR 41.8%, MaxDD -40.9%
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# Import confluence engine from V3 (unchanged)
from strategies.composite.mega_strategy_v3 import (
    compute_confluence, detect_regime, ASSET_CONFIGS,
    _realized_vol, TX_COST,
)

NAME = "MegaStrategyV3.1"
CATEGORY = "composite"
DESCRIPTION = "Direct sizing with protection layers — fixes V3 capital efficiency"

# ── Configuration ─────────────────────────────────────────────────
# H2 production config — validated via 8-variant comparison + walk-forward

LEVERAGE_MAP = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.8, 1: 0.3, 0: 0.0}

SCORE4_CRYPTO_MOM_OVERRIDE = 0.5  # Demote to 0.5x when CryptoMom is the dissenter
# Evidence: CryptoMom OFF at score 4 = -48.6%/yr (37% of score-4 days)

VOL_CEILING = 0.80          # Halve positions when 30d annualized vol > 80%
VOL_LOOKBACK = 30           # Days for realized vol calculation

BEAR_FILTER_DAYS = 30       # Go flat after N consecutive days at score ≤ 1

PORTFOLIO_TRAIL_STOP = 0.10       # Trigger at -10% portfolio drawdown
TRAIL_REDUCE_FACTOR = 0.3         # Reduce to 30% of target during drawdown
TRAIL_RECOVERY_DAYS = 30          # Days needed to restore full sizing
TRAIL_RECOVERY_THRESHOLD = 0.95   # Equity must recover to 95% of peak

WEIGHTS = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "LINK": 0.15}


# ── Core Engine ───────────────────────────────────────────────────

def run_single_asset(
    df: pd.DataFrame,
    macro_data: pd.DataFrame,
    cross_asset_data: pd.DataFrame,
    asset_name: str = "BTC",
    leverage_map: Optional[Dict[int, float]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute confluence + base leverage for a single asset.
    Returns (confluence, base_leverage, breakdown).
    Does NOT apply protection layers — those are portfolio-level.
    """
    cfg = ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"])
    lmap = leverage_map or LEVERAGE_MAP
    close = df["close"]

    # Confluence (unchanged from V3)
    confluence, breakdown = compute_confluence(
        close, macro_data, cross_asset_data,
        sma_slow=cfg["sma_slow"],
        momentum_period=cfg["momentum_period"],
    )

    # Base leverage from map (lagged 1 day)
    base_lev = confluence.map(lambda c: lmap.get(min(int(c), 5), 0.0))

    # Score 4 smart demotion
    is_score4 = confluence == 4
    crypto_mom_off = breakdown["crypto_momentum"] == 0
    override_mask = is_score4 & crypto_mom_off
    base_lev = base_lev.where(~override_mask, SCORE4_CRYPTO_MOM_OVERRIDE)

    # Lag by 1 day (no lookahead)
    base_lev = base_lev.shift(1).fillna(0)

    return confluence, base_lev, breakdown


def run_full_strategy(
    asset_data: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    cross_asset_data: pd.DataFrame,
    leverage_map: Optional[Dict[int, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    vol_ceiling: float = VOL_CEILING,
    vol_lookback: int = VOL_LOOKBACK,
    bear_filter_days: int = BEAR_FILTER_DAYS,
    portfolio_trail_stop: float = PORTFOLIO_TRAIL_STOP,
    trail_reduce_factor: float = TRAIL_REDUCE_FACTOR,
    trail_recovery_days: int = TRAIL_RECOVERY_DAYS,
    trail_recovery_threshold: float = TRAIL_RECOVERY_THRESHOLD,
    s4_override: float = SCORE4_CRYPTO_MOM_OVERRIDE,
    tx_cost: float = TX_COST,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Full multi-asset V3.1 strategy with protection layers.
    """
    w = weights or WEIGHTS
    lmap = leverage_map or LEVERAGE_MAP
    assets = list(asset_data.keys())

    # Common index
    common_idx = asset_data[assets[0]].index
    for a in assets[1:]:
        common_idx = common_idx.intersection(asset_data[a].index)

    # Pre-compute per-asset data
    per_asset_conf = {}
    per_asset_lev = {}
    per_asset_bd = {}
    per_asset_ret = {}
    per_asset_close = {}

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

    # BTC drives portfolio regime
    btc_conf = per_asset_conf.get("BTC", per_asset_conf[assets[0]])
    regime = detect_regime(btc_conf)

    # ── Day-by-day simulation with protection layers ──
    n = len(common_idx)
    daily_pnl = np.zeros(n)
    daily_long_pnl = np.zeros(n)
    total_leverage = np.zeros(n)
    regimes = [""] * n
    confluences = np.zeros(n)

    equity = 1.0
    peak_eq = 1.0
    in_drawdown = False
    dd_start_idx = 0
    consecutive_low = 0
    prev_exposure = 0.0

    for i in range(1, n):
        # ── Layer 1: Base exposure from per-asset confluence ──
        target_exposure = 0.0
        for asset in assets:
            asset_lev = per_asset_lev[asset].iloc[i]
            target_exposure += w.get(asset, 0.25) * asset_lev

        # ── Layer 2: Vol ceiling ──
        if vol_ceiling > 0:
            btc_close = per_asset_close.get("BTC", per_asset_close[assets[0]])
            if i >= vol_lookback:
                window = btc_close.iloc[max(0, i - vol_lookback):i]
                rvol = window.pct_change().std() * np.sqrt(365)  # crypto trades 365 days/yr
                if rvol > vol_ceiling:
                    target_exposure *= 0.5

        # ── Layer 3: Bear filter ──
        btc_score = int(btc_conf.iloc[i - 1])  # lagged
        if btc_score <= 1:
            consecutive_low += 1
        else:
            consecutive_low = 0

        if consecutive_low >= bear_filter_days:
            target_exposure = 0.0

        # ── Layer 4: Portfolio trailing stop ──
        if portfolio_trail_stop > 0:
            dd = 1 - equity / peak_eq
            if dd > portfolio_trail_stop:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start_idx = i
                target_exposure *= trail_reduce_factor
            elif in_drawdown:
                days_since = i - dd_start_idx
                if days_since > trail_recovery_days and equity > peak_eq * trail_recovery_threshold:
                    in_drawdown = False

        # ── Compute P&L ──
        port_ret = 0.0
        for asset in assets:
            asset_weight = w.get(asset, 0.25)
            weight_sum = sum(w.values())
            asset_exposure = target_exposure * (asset_weight / weight_sum)
            ret = per_asset_ret[asset].iloc[i]
            port_ret += asset_exposure * ret

        # Transaction costs
        lev_change = abs(target_exposure - prev_exposure)
        port_ret -= lev_change * tx_cost
        prev_exposure = target_exposure

        daily_pnl[i] = port_ret
        daily_long_pnl[i] = port_ret  # V3.1 is long-only
        total_leverage[i] = target_exposure
        regimes[i] = regime.iloc[i] if i < len(regime) else ""
        confluences[i] = btc_conf.iloc[i] if i < len(btc_conf) else 0

        equity *= (1 + port_ret)
        peak_eq = max(peak_eq, equity)

    # Build output
    portfolio_df = pd.DataFrame({
        "daily_pnl": daily_pnl,
        "equity": np.cumprod(1 + daily_pnl),
        "long_pnl": daily_long_pnl,
        "short_pnl": 0.0,
        "funding_pnl": 0.0,
        "total_leverage": total_leverage,
        "regime": regimes,
        "confluence": confluences,
    }, index=common_idx)

    # Per-asset results
    per_asset_results = {}
    for asset in assets:
        per_asset_results[asset] = pd.DataFrame({
            "confluence": per_asset_conf[asset],
            "base_leverage": per_asset_lev[asset],
            "regime": regime,
            "daily_return": per_asset_ret[asset],
        }, index=common_idx)

    return portfolio_df, per_asset_results
