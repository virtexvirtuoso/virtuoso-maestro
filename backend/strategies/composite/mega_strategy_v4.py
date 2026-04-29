"""
MegaStrategyV4 — V3 + Vol Breakout + EMARibbon + Fama-French Bridge + MultiTF Confluence

Architecture:
    V3 Core (50% risk budget)
    Vol Breakout Module (15% risk budget)
    EMARibbon Module (15% risk budget)
    Fama-French Factor Bridge (10% risk budget)
    Multi-Timeframe Confluence (10% risk budget)
    Portfolio Risk Manager (leverage cap, DD breakers, vol targeting)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

NAME = "MegaStrategyV4"
CATEGORY = "composite"
DESCRIPTION = "V3 + Vol Breakout + EMARibbon + FF Bridge + MultiTF with portfolio risk management"
REQUIRES_DERIVATIVES = False

from strategies.composite.mega_strategy_v3 import (
    ASSET_CONFIGS, DEFAULT_LEVERAGE_MAP, DEFAULT_SAFETY_PARAMS,
    TX_COST, SHORT_BORROW_COST_DAILY, FUNDING_CARRY_DAILY,
    compute_confluence, detect_regime, adaptive_leverage,
    generate_long_signals, generate_short_signals,
    run_single_asset, _rsi, _bb, _atr, _realized_vol,
)

# Try importing EMARibbon
try:
    from strategies.scalping.ema_ribbon import generate_signals as ema_ribbon_signals
except ImportError:
    def ema_ribbon_signals(df):
        """Fallback EMARibbon."""
        signals = pd.Series(0, index=df.index)
        emas = [df['close'].ewm(span=s).mean() for s in [8, 13, 21, 34, 55]]
        bullish = (emas[0] > emas[1]) & (emas[1] > emas[2]) & (emas[2] > emas[3]) & (emas[3] > emas[4])
        bearish = (emas[0] < emas[1]) & (emas[1] < emas[2]) & (emas[2] < emas[3]) & (emas[3] < emas[4])
        signals[bullish] = 1
        signals[bearish] = -1
        return signals

DEFAULT_RISK_BUDGETS = {
    "v3_core": 0.50,
    "vol_breakout": 0.15,
    "ema_ribbon": 0.15,
    "ff_bridge": 0.10,
    "multitf": 0.10,
}

DEFAULT_V4_PARAMS = dict(
    max_total_leverage=2.5,
    vol_target=0.30,
    vol_lookback=20,
    module_dd_breaker=0.08,
    corr_threshold=0.7,
    bb_squeeze_pctile=20,
    bb_squeeze_lookback=100,
    bb_atr_mult=1.5,
    ff_lookback_months=3,
    mtf_fast=5,
    mtf_medium=25,
    mtf_slow=120,
)


# ---------------------------------------------------------------------------
# Module 1: Vol Breakout
# ---------------------------------------------------------------------------

def run_vol_breakout(
    asset_data: pd.DataFrame,
    confluence: pd.Series,
    bb_squeeze_pctile: int = 20,
    bb_squeeze_lookback: int = 100,
    bb_atr_mult: float = 1.5,
    **params,
) -> dict:
    """
    Vol Breakout: detect BB squeeze and trade the release.
    Returns dict with 'returns' (daily), 'positions', 'trades' count.
    """
    close = asset_data["close"]
    idx = close.index
    n = len(idx)

    # Bollinger Band Width
    bb_lower, bb_mid, bb_upper = _bb(close, 20, 2.0)
    bbw = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bbw = bbw.fillna(method="ffill").fillna(0)

    # ATR for stops
    atr = _atr(asset_data, 14).fillna(0)

    # Rolling percentile of BBW (squeeze detection)
    bbw_pctile = bbw.rolling(bb_squeeze_lookback, min_periods=20).rank(pct=True) * 100

    # Squeeze: BBW below threshold percentile
    in_squeeze = (bbw_pctile < bb_squeeze_pctile).shift(1).fillna(False)

    # Squeeze release: was in squeeze, now BBW expanding
    squeeze_release = in_squeeze & (bbw > bbw.shift(1))

    # Direction from confluence (shifted for no lookahead)
    conf_shifted = confluence.shift(1).fillna(2)

    # Simulate positions
    positions = np.zeros(n)
    stop_price = np.zeros(n)
    trades = 0

    for i in range(1, n):
        prev_pos = positions[i - 1]
        price = close.iloc[i]

        # Check stop
        if prev_pos != 0 and stop_price[i - 1] > 0:
            if prev_pos > 0 and price < stop_price[i - 1]:
                positions[i] = 0
                stop_price[i] = 0
                trades += 1
                continue
            if prev_pos < 0 and price > stop_price[i - 1]:
                positions[i] = 0
                stop_price[i] = 0
                trades += 1
                continue

        # New entry on squeeze release
        if prev_pos == 0 and squeeze_release.iloc[i]:
            c = conf_shifted.iloc[i]
            atr_val = atr.iloc[i - 1] if i > 0 else 0
            if c >= 3:
                positions[i] = 1.0
                stop_price[i] = price - bb_atr_mult * atr_val
                trades += 1
                continue
            elif c <= 1:
                positions[i] = -1.0
                stop_price[i] = price + bb_atr_mult * atr_val
                trades += 1
                continue

        positions[i] = prev_pos
        stop_price[i] = stop_price[i - 1]

    pos_series = pd.Series(positions, index=idx)
    daily_ret = close.pct_change().fillna(0)
    pos_shifted = pos_series.shift(1).fillna(0)
    pos_changes = pos_shifted.diff().abs().fillna(0)
    module_returns = pos_shifted * daily_ret - pos_changes * TX_COST

    return {
        "returns": module_returns,
        "positions": pos_series,
        "trades": trades,
        "name": "vol_breakout",
    }


# ---------------------------------------------------------------------------
# Module 2: EMARibbon + Adaptive Sizing
# ---------------------------------------------------------------------------

def run_ema_ribbon(
    asset_data: pd.DataFrame,
    confluence: pd.Series,
    **params,
) -> dict:
    """
    EMARibbon with confluence-based adaptive sizing.
    """
    close = asset_data["close"]
    idx = close.index

    # Get raw EMA ribbon signals
    raw_signals = ema_ribbon_signals(asset_data)

    # Shift to avoid lookahead
    signals = raw_signals.shift(1).fillna(0)

    # Adaptive sizing from confluence (shifted)
    conf_shifted = confluence.shift(1).fillna(2)
    leverage_map = {5: 2.0, 4: 1.5, 3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
    sizing = conf_shifted.map(lambda c: leverage_map.get(min(int(c), 5), 0.0))

    # Position = signal direction * sizing
    positions = signals * sizing

    daily_ret = close.pct_change().fillna(0)
    pos_shifted = positions.shift(1).fillna(0)
    pos_changes = pos_shifted.diff().abs().fillna(0)
    module_returns = pos_shifted * daily_ret - pos_changes * TX_COST

    trades = int((signals.diff().abs() > 0).sum())

    return {
        "returns": module_returns,
        "positions": positions,
        "trades": trades,
        "name": "ema_ribbon",
    }


# ---------------------------------------------------------------------------
# Module 3: Fama-French Factor Bridge
# ---------------------------------------------------------------------------

def run_ff_bridge(
    asset_data: pd.DataFrame,
    ff_data: pd.DataFrame,
    ff_lookback_months: int = 3,
    **params,
) -> dict:
    """
    Fama-French Factor Bridge: monthly signal from FF factors.
    When Mkt-RF and RMW underperform → overweight crypto.
    """
    close = asset_data["close"]
    idx = close.index
    n = len(idx)

    if ff_data is None or len(ff_data) < 12:
        return {
            "returns": pd.Series(0.0, index=idx),
            "positions": pd.Series(0.0, index=idx),
            "trades": 0,
            "name": "ff_bridge",
        }

    # Resample FF data to monthly if not already
    ff = ff_data.copy()
    if not isinstance(ff.index, pd.DatetimeIndex):
        ff.index = pd.to_datetime(ff.index)

    # Compute rolling averages of Mkt-RF and RMW
    mkt_col = "Mkt-RF" if "Mkt-RF" in ff.columns else ff.columns[0]
    rmw_col = "RMW" if "RMW" in ff.columns else None

    mkt_rf_roll = ff[mkt_col].rolling(ff_lookback_months, min_periods=1).mean()

    if rmw_col and rmw_col in ff.columns:
        rmw_roll = ff[rmw_col].rolling(ff_lookback_months, min_periods=1).mean()
    else:
        rmw_roll = pd.Series(0.0, index=ff.index)

    # Signal: when both are underperforming (negative), overweight crypto
    # When both outperforming, underweight
    ff_signal = pd.Series(0.0, index=ff.index)
    both_weak = (mkt_rf_roll < 0) & (rmw_roll < 0)
    both_strong = (mkt_rf_roll > 0) & (rmw_roll > 0)
    ff_signal[both_weak] = 1.0   # overweight crypto
    ff_signal[both_strong] = -0.3  # slight underweight
    ff_signal[(~both_weak) & (~both_strong)] = 0.5  # neutral-ish

    # Forward-fill to daily (monthly rebalance, shift to avoid lookahead)
    ff_daily = ff_signal.reindex(idx, method="ffill").fillna(0).shift(1).fillna(0)

    daily_ret = close.pct_change().fillna(0)
    pos_shifted = ff_daily.shift(1).fillna(0)
    pos_changes = pos_shifted.diff().abs().fillna(0)
    module_returns = pos_shifted * daily_ret - pos_changes * TX_COST

    trades = int((ff_daily.diff().abs() > 0.01).sum())

    return {
        "returns": module_returns,
        "positions": ff_daily,
        "trades": trades,
        "name": "ff_bridge",
    }


# ---------------------------------------------------------------------------
# Module 4: Multi-Timeframe Confluence
# ---------------------------------------------------------------------------

def run_multitf(
    asset_data: pd.DataFrame,
    macro_score: pd.Series,
    mtf_fast: int = 5,
    mtf_medium: int = 25,
    mtf_slow: int = 120,
    **params,
) -> dict:
    """
    Multi-Timeframe Confluence: daily + weekly + monthly must all agree.
    Uses different lookback periods on daily data.
    """
    close = asset_data["close"]
    idx = close.index

    # Daily trend (fast): ROC over fast period
    daily_trend = (close.pct_change(mtf_fast) > 0).astype(int).shift(1).fillna(0)

    # Weekly momentum (medium): ROC over medium period
    weekly_mom = (close.pct_change(mtf_medium) > 0).astype(int).shift(1).fillna(0)

    # Monthly regime (slow): price vs SMA
    sma_slow = close.rolling(mtf_slow).mean()
    monthly_regime = (close > sma_slow).astype(int).shift(1).fillna(0)

    # Macro regime agreement (shifted)
    if macro_score is not None:
        macro_bull = (macro_score >= 3).astype(int).shift(1).fillna(0)
    else:
        macro_bull = pd.Series(1, index=idx)

    # All must agree for entry
    all_bull = (daily_trend == 1) & (weekly_mom == 1) & (monthly_regime == 1) & (macro_bull == 1)
    all_bear = (daily_trend == 0) & (weekly_mom == 0) & (monthly_regime == 0) & (macro_bull == 0)

    positions = pd.Series(0.0, index=idx)
    positions[all_bull] = 1.0
    positions[all_bear] = -0.5  # more conservative shorts

    daily_ret = close.pct_change().fillna(0)
    pos_shifted = positions.shift(1).fillna(0)
    pos_changes = pos_shifted.diff().abs().fillna(0)
    module_returns = pos_shifted * daily_ret - pos_changes * TX_COST

    trades = int((positions.diff().abs() > 0).sum())

    return {
        "returns": module_returns,
        "positions": positions,
        "trades": trades,
        "name": "multitf",
    }


# ---------------------------------------------------------------------------
# V3 Core Wrapper
# ---------------------------------------------------------------------------

def run_v3_core(
    asset_data: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    cross_asset: pd.DataFrame,
    **params,
) -> dict:
    """
    Run V3 core and return module-format result.
    """
    portfolio_df, per_asset = run_single_asset_portfolio(
        asset_data, macro_data, cross_asset, **params,
    )
    return {
        "returns": portfolio_df["daily_pnl"],
        "positions": portfolio_df.get("total_leverage", pd.Series(0, index=portfolio_df.index)),
        "trades": 0,  # counted internally
        "name": "v3_core",
        "portfolio_df": portfolio_df,
        "per_asset": per_asset,
    }


def run_single_asset_portfolio(
    asset_data: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    cross_asset: pd.DataFrame,
    **params,
) -> Tuple[pd.DataFrame, Dict]:
    """Wrapper around V3 run_full_strategy with default configs."""
    from strategies.composite.mega_strategy_v3 import run_full_strategy
    return run_full_strategy(
        asset_data, macro_data, cross_asset,
        enable_long=True, enable_short=True,
        enable_adaptive_leverage=True,
        **params,
    )


# ---------------------------------------------------------------------------
# Portfolio Risk Manager
# ---------------------------------------------------------------------------

def apply_risk_management(
    module_returns: Dict[str, pd.Series],
    risk_budgets: Dict[str, float],
    max_total_leverage: float = 2.5,
    vol_target: float = 0.30,
    vol_lookback: int = 20,
    module_dd_breaker: float = 0.08,
    corr_threshold: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply portfolio risk management across modules.
    Returns (combined_returns, module_weights_over_time).
    """
    names = list(module_returns.keys())
    if not names:
        return pd.DataFrame(), pd.DataFrame()

    # Align all to common index
    common_idx = module_returns[names[0]].index
    for n in names[1:]:
        common_idx = common_idx.intersection(module_returns[n].index)

    ret_df = pd.DataFrame({n: module_returns[n].reindex(common_idx).fillna(0) for n in names})
    n_days = len(common_idx)

    # Normalize risk budgets
    total_budget = sum(risk_budgets.get(n, 0) for n in names)
    if total_budget <= 0:
        total_budget = 1.0
    norm_budgets = {n: risk_budgets.get(n, 0) / total_budget for n in names}

    # Initialize weights
    weights = pd.DataFrame(
        {n: [norm_budgets[n]] * n_days for n in names},
        index=common_idx,
    )

    # Circuit breaker: rolling module drawdown
    for name in names:
        cum = (1 + ret_df[name]).cumprod()
        dd = cum / cum.cummax() - 1

        # Monthly circuit breaker
        month_groups = pd.Series(common_idx).dt.to_period("M").values
        breaker_active = pd.Series(False, index=common_idx)

        for i in range(1, n_days):
            if dd.iloc[i] < -module_dd_breaker:
                # Halve for rest of month
                current_month = month_groups[i]
                mask = pd.Series(month_groups) == current_month
                mask_idx = mask[mask].index
                remaining = [j for j in mask_idx if j >= i]
                for j in remaining:
                    weights.iloc[j][name] *= 0.5

    # Correlation monitor: if rolling corr between any pair > threshold, reduce both
    if len(names) >= 2:
        for i_n in range(len(names)):
            for j_n in range(i_n + 1, len(names)):
                roll_corr = ret_df[names[i_n]].rolling(60, min_periods=20).corr(ret_df[names[j_n]])
                high_corr = roll_corr > corr_threshold
                for k in range(len(common_idx)):
                    if high_corr.iloc[k] if k < len(high_corr) else False:
                        weights.iloc[k][names[i_n]] *= 0.7
                        weights.iloc[k][names[j_n]] *= 0.7

    # Renormalize weights each day
    row_sums = weights.sum(axis=1).replace(0, 1)
    weights = weights.div(row_sums, axis=0)

    # Vol targeting: scale total position
    combined_unscaled = (ret_df * weights).sum(axis=1)
    realized = combined_unscaled.rolling(vol_lookback, min_periods=5).std() * np.sqrt(252)
    vol_scalar = (vol_target / realized.replace(0, np.nan)).fillna(1.0).clip(0.2, max_total_leverage)

    combined = combined_unscaled * vol_scalar

    # Cap at max leverage equivalent
    # (vol_scalar effectively serves as leverage scaler)

    module_contrib = ret_df * weights * vol_scalar.values.reshape(-1, 1)

    return combined, module_contrib


# ---------------------------------------------------------------------------
# Combine Modules
# ---------------------------------------------------------------------------

def combine_modules(
    module_results: Dict[str, dict],
    risk_budgets: Dict[str, float],
    risk_params: dict,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Combine all module returns with risk management.
    Returns (portfolio_returns, module_contributions).
    """
    mod_returns = {name: res["returns"] for name, res in module_results.items()}
    combined, contrib = apply_risk_management(
        mod_returns, risk_budgets,
        max_total_leverage=risk_params.get("max_total_leverage", 2.5),
        vol_target=risk_params.get("vol_target", 0.30),
        vol_lookback=risk_params.get("vol_lookback", 20),
        module_dd_breaker=risk_params.get("module_dd_breaker", 0.08),
        corr_threshold=risk_params.get("corr_threshold", 0.7),
    )
    return combined, contrib


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_mega_v4(
    asset_data: Dict[str, pd.DataFrame],
    macro_data: pd.DataFrame,
    cross_asset: pd.DataFrame,
    ff_data: Optional[pd.DataFrame] = None,
    risk_budgets: Optional[Dict[str, float]] = None,
    disabled_modules: Optional[list] = None,
    **all_params,
) -> dict:
    """
    Run the complete V4 system.

    Args:
        asset_data: dict of asset_name -> OHLCV DataFrame
        macro_data: macro DataFrame (m2, yield_curve, etc.)
        cross_asset: cross-asset DataFrame (gold, dxy, bonds, hyg, copper)
        ff_data: Fama-French factor DataFrame (monthly)
        risk_budgets: override risk budgets
        disabled_modules: list of module names to disable (for ablation)
        **all_params: override V4 params

    Returns:
        dict with 'portfolio_returns', 'module_results', 'module_contributions',
        'equity', 'metrics', 'regime', 'confluence'
    """
    params = {**DEFAULT_V4_PARAMS, **all_params}
    budgets = dict(risk_budgets or DEFAULT_RISK_BUDGETS)
    disabled = set(disabled_modules or [])

    # Get a representative asset for module signals (BTC as primary)
    primary_asset = "BTC" if "BTC" in asset_data else list(asset_data.keys())[0]
    primary_df = asset_data[primary_asset]

    # Compute confluence from V3 (for use by other modules)
    from strategies.composite.mega_strategy_v3 import compute_confluence as v3_confluence
    asset_cfg = ASSET_CONFIGS.get(primary_asset, ASSET_CONFIGS["BTC"])
    confluence, breakdown = v3_confluence(
        primary_df["close"], macro_data, cross_asset,
        sma_slow=asset_cfg["sma_slow"],
        momentum_period=asset_cfg["momentum_period"],
    )
    regime = detect_regime(confluence)

    module_results = {}

    # --- V3 Core ---
    if "v3_core" not in disabled:
        v3_result = run_v3_core(asset_data, macro_data, cross_asset, **all_params)
        module_results["v3_core"] = v3_result
    else:
        budgets["v3_core"] = 0

    # --- Vol Breakout (run per-asset, then average) ---
    if "vol_breakout" not in disabled:
        vb_returns_list = []
        vb_trades = 0
        for asset_name, df in asset_data.items():
            a_conf, _ = v3_confluence(
                df["close"], macro_data, cross_asset,
                sma_slow=ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"])["sma_slow"],
                momentum_period=ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"])["momentum_period"],
            )
            vb = run_vol_breakout(
                df, a_conf,
                bb_squeeze_pctile=params["bb_squeeze_pctile"],
                bb_squeeze_lookback=params.get("bb_squeeze_lookback", 100),
                bb_atr_mult=params["bb_atr_mult"],
            )
            vb_returns_list.append(vb["returns"])
            vb_trades += vb["trades"]

        # Equal-weight average across assets
        vb_combined = pd.concat(vb_returns_list, axis=1).mean(axis=1)
        module_results["vol_breakout"] = {
            "returns": vb_combined,
            "positions": pd.Series(0, index=vb_combined.index),
            "trades": vb_trades,
            "name": "vol_breakout",
        }
    else:
        budgets["vol_breakout"] = 0

    # --- EMARibbon (per-asset average) ---
    if "ema_ribbon" not in disabled:
        er_returns_list = []
        er_trades = 0
        for asset_name, df in asset_data.items():
            a_conf, _ = v3_confluence(
                df["close"], macro_data, cross_asset,
                sma_slow=ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"])["sma_slow"],
                momentum_period=ASSET_CONFIGS.get(asset_name, ASSET_CONFIGS["BTC"])["momentum_period"],
            )
            er = run_ema_ribbon(df, a_conf)
            er_returns_list.append(er["returns"])
            er_trades += er["trades"]

        er_combined = pd.concat(er_returns_list, axis=1).mean(axis=1)
        module_results["ema_ribbon"] = {
            "returns": er_combined,
            "positions": pd.Series(0, index=er_combined.index),
            "trades": er_trades,
            "name": "ema_ribbon",
        }
    else:
        budgets["ema_ribbon"] = 0

    # --- FF Bridge (per-asset average) ---
    if "ff_bridge" not in disabled:
        ff_returns_list = []
        ff_trades = 0
        for asset_name, df in asset_data.items():
            ffb = run_ff_bridge(
                df, ff_data,
                ff_lookback_months=params.get("ff_lookback_months", params.get("ff_lookback", 3)),
            )
            ff_returns_list.append(ffb["returns"])
            ff_trades += ffb["trades"]

        ff_combined = pd.concat(ff_returns_list, axis=1).mean(axis=1)
        module_results["ff_bridge"] = {
            "returns": ff_combined,
            "positions": pd.Series(0, index=ff_combined.index),
            "trades": ff_trades,
            "name": "ff_bridge",
        }
    else:
        budgets["ff_bridge"] = 0

    # --- MultiTF (per-asset average) ---
    if "multitf" not in disabled:
        mtf_returns_list = []
        mtf_trades = 0
        for asset_name, df in asset_data.items():
            mtf = run_multitf(
                df, confluence,
                mtf_fast=params["mtf_fast"],
                mtf_medium=params["mtf_medium"],
                mtf_slow=params["mtf_slow"],
            )
            mtf_returns_list.append(mtf["returns"])
            mtf_trades += mtf["trades"]

        mtf_combined = pd.concat(mtf_returns_list, axis=1).mean(axis=1)
        module_results["multitf"] = {
            "returns": mtf_combined,
            "positions": pd.Series(0, index=mtf_combined.index),
            "trades": mtf_trades,
            "name": "multitf",
        }
    else:
        budgets["multitf"] = 0

    # --- Combine with risk management ---
    risk_params = {
        "max_total_leverage": params["max_total_leverage"],
        "vol_target": params["vol_target"],
        "vol_lookback": params.get("vol_lookback", 20),
        "module_dd_breaker": params["module_dd_breaker"],
        "corr_threshold": params["corr_threshold"],
    }

    portfolio_returns, module_contrib = combine_modules(module_results, budgets, risk_params)

    # Equity curve
    equity = (1 + portfolio_returns).cumprod()

    return {
        "portfolio_returns": portfolio_returns,
        "equity": equity,
        "module_results": module_results,
        "module_contributions": module_contrib,
        "regime": regime,
        "confluence": confluence,
        "risk_budgets": budgets,
        "params": params,
    }
