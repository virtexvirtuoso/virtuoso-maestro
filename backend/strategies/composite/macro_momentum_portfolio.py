"""
MacroMomentum Portfolio — Multi-asset macro-filtered momentum with per-asset optimization.

Combines BTC, ETH, SOL, LINK each running MacroMomentumV2 with asset-specific params,
then allocates across assets using equal-weight, risk-parity, or momentum-weighted schemes.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

NAME = "MacroMomentumPortfolio"
CATEGORY = "composite"
DESCRIPTION = "Multi-asset macro-filtered momentum with dip-buying and per-asset optimization"
REQUIRES_DERIVATIVES = False

# Per-asset optimized params from Optuna runs
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
        ticker="LINK-USD", sma_slow=100, momentum_period=35, rsi_entry=52,
        trail_stop_pct=0.12, max_position=1.50, initial_size=0.60,
        rsi_exit=75, ema_period=21, bb_period=20, bb_std=2.0,
        roc_threshold=0.0, pyramid_size=0.20, trim_pct=0.25,
        atr_exit_mult=2.0, macro_weight=1.0,
    ),
}

TX_COST = 0.001


def generate_portfolio_signals(
    asset_data: Dict[str, pd.DataFrame],
    macro_score: pd.Series,
    m2_accelerating: pd.Series,
    asset_configs: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate per-asset V2 signals using asset-specific params.
    
    Args:
        asset_data: {"BTC": df_btc, "ETH": df_eth, ...} with OHLCV DataFrames
        macro_score: daily macro score (0-6)
        m2_accelerating: bool Series for M2 acceleration filter
        asset_configs: override ASSET_CONFIGS if provided
        
    Returns:
        dict of per-asset signal DataFrames with columns: signal, position_size, entry_type
    """
    from strategies.composite.macro_momentum_v2 import generate_signals
    
    configs = asset_configs or ASSET_CONFIGS
    signals = {}
    
    for asset, df in asset_data.items():
        if asset not in configs:
            continue
        cfg = {k: v for k, v in configs[asset].items() if k != "ticker"}
        sig = generate_signals(df, macro_score=macro_score, m2_accelerating=m2_accelerating, **cfg)
        signals[asset] = sig
    
    return signals


def compute_portfolio_allocation(
    signals: Dict[str, pd.DataFrame],
    asset_data: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    mode: str = "equal",
    max_portfolio_leverage: float = 1.0,
    vol_lookback: int = 60,
    momentum_lookback: int = 20,
    rebalance_freq: str = "monthly",
) -> Dict[str, pd.Series]:
    """
    Compute portfolio-level position sizes from per-asset signals.
    
    Args:
        signals: per-asset signal DataFrames from generate_portfolio_signals
        asset_data: OHLCV data per asset
        weights: custom weights dict (used when mode="custom")
        mode: "equal", "risk_parity", "momentum_weighted", "custom"
        max_portfolio_leverage: max total portfolio exposure
        vol_lookback: lookback for risk parity volatility calc
        momentum_lookback: lookback for momentum weighting
        rebalance_freq: "daily", "weekly", "monthly"
        
    Returns:
        dict of per-asset position size Series
    """
    assets = list(signals.keys())
    if not assets:
        return {}
    
    # Get common index
    all_idx = signals[assets[0]].index
    for a in assets[1:]:
        all_idx = all_idx.intersection(signals[a].index)
    
    n = len(all_idx)
    
    # Compute weights based on mode
    if mode == "custom" and weights:
        total = sum(weights.values())
        w = {a: weights.get(a, 0) / total for a in assets}
        weight_series = {a: pd.Series(w[a], index=all_idx) for a in assets}
    elif mode == "risk_parity":
        weight_series = _risk_parity_weights(asset_data, assets, all_idx, vol_lookback, rebalance_freq)
    elif mode == "momentum_weighted":
        weight_series = _momentum_weights(asset_data, assets, all_idx, momentum_lookback, rebalance_freq)
    else:  # equal
        eq_w = 1.0 / len(assets)
        weight_series = {a: pd.Series(eq_w, index=all_idx) for a in assets}
    
    # Apply weights to per-asset position sizes
    allocated = {}
    for a in assets:
        raw_pos = signals[a]["position_size"].reindex(all_idx).fillna(0)
        # Scale by weight and normalize by max_position so weight represents fraction of portfolio
        cfg = ASSET_CONFIGS.get(a, {})
        max_pos = cfg.get("max_position", 1.0)
        # Normalized position: raw / max_pos gives [0, 1], multiply by weight
        norm_pos = (raw_pos / max_pos).clip(0, 1) * weight_series[a]
        allocated[a] = norm_pos
    
    # Enforce max portfolio leverage
    for i in range(n):
        idx = all_idx[i]
        total_exp = sum(allocated[a].iloc[i] for a in assets)
        if total_exp > max_portfolio_leverage:
            scale = max_portfolio_leverage / total_exp
            for a in assets:
                allocated[a].iloc[i] *= scale
    
    return allocated


def _risk_parity_weights(
    asset_data: Dict[str, pd.DataFrame],
    assets: list,
    index: pd.DatetimeIndex,
    lookback: int,
    rebalance_freq: str,
) -> Dict[str, pd.Series]:
    """Inverse-volatility weights, rebalanced per rebalance_freq."""
    # Compute daily returns
    rets = {}
    for a in assets:
        r = asset_data[a]["close"].reindex(index).pct_change().fillna(0)
        rets[a] = r
    
    weight_df = pd.DataFrame(index=index, columns=assets, dtype=float)
    
    # Determine rebalance dates
    rebal_dates = _get_rebalance_dates(index, rebalance_freq)
    
    current_weights = {a: 1.0 / len(assets) for a in assets}
    
    for dt in index:
        if dt in rebal_dates:
            vols = {}
            for a in assets:
                loc = rets[a].index.get_loc(dt)
                start = max(0, loc - lookback)
                vol = rets[a].iloc[start:loc+1].std() * np.sqrt(252)
                vols[a] = max(vol, 0.01)
            inv_vol = {a: 1.0 / vols[a] for a in assets}
            total = sum(inv_vol.values())
            current_weights = {a: inv_vol[a] / total for a in assets}
        
        for a in assets:
            weight_df.loc[dt, a] = current_weights[a]
    
    return {a: weight_df[a].astype(float) for a in assets}


def _momentum_weights(
    asset_data: Dict[str, pd.DataFrame],
    assets: list,
    index: pd.DatetimeIndex,
    lookback: int,
    rebalance_freq: str,
) -> Dict[str, pd.Series]:
    """Momentum-weighted: overweight assets with strongest recent returns."""
    weight_df = pd.DataFrame(index=index, columns=assets, dtype=float)
    rebal_dates = _get_rebalance_dates(index, rebalance_freq)
    current_weights = {a: 1.0 / len(assets) for a in assets}
    
    for dt in index:
        if dt in rebal_dates:
            moms = {}
            for a in assets:
                close = asset_data[a]["close"].reindex(index)
                loc = close.index.get_loc(dt)
                start = max(0, loc - lookback)
                if close.iloc[start] > 0:
                    moms[a] = max(close.iloc[loc] / close.iloc[start] - 1, 0.001)
                else:
                    moms[a] = 0.001
            total = sum(moms.values())
            current_weights = {a: moms[a] / total for a in assets}
        
        for a in assets:
            weight_df.loc[dt, a] = current_weights[a]
    
    return {a: weight_df[a].astype(float) for a in assets}


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """Get set of rebalance dates from index."""
    if freq == "daily":
        return set(index)
    elif freq == "weekly":
        return set(index[index.weekday == 0])  # Mondays
    else:  # monthly
        # First trading day of each month
        months = pd.Series(index, index=index).groupby([index.year, index.month]).first()
        return set(months.values)


def run_portfolio_backtest(
    asset_data: Dict[str, pd.DataFrame],
    macro_score: pd.Series,
    m2_accelerating: pd.Series,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "equal",
    max_portfolio_leverage: float = 1.0,
    vol_target: Optional[float] = None,
    rebalance_freq: str = "monthly",
    correlation_filter: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    asset_configs: Optional[Dict] = None,
) -> dict:
    """
    Full portfolio backtest simulation.
    
    Returns comprehensive results dict with portfolio and per-asset metrics.
    """
    # Filter end date on data, but keep full history for indicator warmup
    if end_date:
        asset_data = {a: df[df.index <= pd.Timestamp(end_date)] for a, df in asset_data.items()}
    
    # Remove empty assets
    asset_data = {a: df for a, df in asset_data.items() if len(df) > 200}
    assets = list(asset_data.keys())
    
    if not assets:
        return {"error": "No assets with sufficient data"}
    
    # Generate signals
    signals = generate_portfolio_signals(asset_data, macro_score, m2_accelerating, asset_configs)
    
    # Compute allocation
    allocated = compute_portfolio_allocation(
        signals, asset_data, weights=weights, mode=mode,
        max_portfolio_leverage=max_portfolio_leverage,
        rebalance_freq=rebalance_freq,
    )
    
    # Common index
    common_idx = allocated[assets[0]].index
    
    # Filter to evaluation period (start_date onward) for P&L calculation
    if start_date:
        eval_start = pd.Timestamp(start_date)
        common_idx = common_idx[common_idx >= eval_start]
    
    if len(common_idx) < 5:
        return {"error": "Insufficient evaluation period"}
    
    # Simulate daily P&L
    portfolio_ret = pd.Series(0.0, index=common_idx)
    asset_rets = {}
    asset_trades = {}
    
    for a in assets:
        close = asset_data[a]["close"].reindex(common_idx).ffill()
        daily_ret = close.pct_change().fillna(0)
        pos = allocated[a]
        pos_change = pos.diff().fillna(0).abs()
        
        # Strategy return = position * daily_ret - transaction costs
        strat_ret = daily_ret * pos - pos_change * TX_COST
        
        # Vol targeting
        if vol_target is not None:
            realized_vol = strat_ret.rolling(60).std() * np.sqrt(252)
            vol_scale = (vol_target / len(assets)) / realized_vol.clip(lower=0.05)
            vol_scale = vol_scale.clip(0.1, 3.0).fillna(1.0)
            strat_ret = strat_ret * vol_scale
        
        # Correlation filter
        if correlation_filter and len(assets) > 1:
            strat_ret = _apply_correlation_filter(strat_ret, asset_data, a, assets, common_idx)
        
        asset_rets[a] = strat_ret
        portfolio_ret += strat_ret
        
        # Count trades
        entries = (pos.diff().fillna(0) > 0).sum()
        exits = (pos.diff().fillna(0) < 0).sum()
        asset_trades[a] = int(entries + exits)
    
    # Compute portfolio equity and metrics
    equity = (1 + portfolio_ret).cumprod()
    
    results = _compute_metrics(portfolio_ret, equity, common_idx)
    results["assets"] = assets
    results["mode"] = mode
    results["max_portfolio_leverage"] = max_portfolio_leverage
    
    # Per-asset metrics
    results["per_asset"] = {}
    for a in assets:
        a_equity = (1 + asset_rets[a]).cumprod()
        a_metrics = _compute_metrics(asset_rets[a], a_equity, common_idx)
        a_metrics["trades"] = asset_trades[a]
        a_metrics["contribution"] = float(asset_rets[a].sum() / portfolio_ret.sum()) if portfolio_ret.sum() != 0 else 0
        results["per_asset"][a] = a_metrics
    
    results["total_trades"] = sum(asset_trades.values())
    results["equity_curve"] = equity
    results["daily_returns"] = portfolio_ret
    results["asset_returns"] = asset_rets
    
    # Crash analysis
    results["crash_analysis"] = _crash_analysis(equity, portfolio_ret, asset_rets, common_idx)
    
    # Risk metrics
    results["risk"] = _risk_metrics(portfolio_ret)
    
    return results


def _apply_correlation_filter(
    strat_ret: pd.Series, asset_data: dict, asset: str, 
    assets: list, index: pd.DatetimeIndex
) -> pd.Series:
    """Reduce position when inter-asset correlation spikes above 0.7."""
    other_rets = []
    for a in assets:
        if a != asset:
            r = asset_data[a]["close"].reindex(index).pct_change().fillna(0)
            other_rets.append(r)
    
    if not other_rets:
        return strat_ret
    
    asset_ret_raw = asset_data[asset]["close"].reindex(index).pct_change().fillna(0)
    avg_corr = pd.Series(0.0, index=index)
    for r in other_rets:
        corr = asset_ret_raw.rolling(60).corr(r).fillna(0)
        avg_corr += corr / len(other_rets)
    
    # Scale down when correlation > 0.7
    scale = pd.Series(1.0, index=index)
    high_corr = avg_corr > 0.7
    scale[high_corr] = 0.5
    
    return strat_ret * scale


def _compute_metrics(returns: pd.Series, equity: pd.Series, index: pd.DatetimeIndex) -> dict:
    """Compute standard performance metrics."""
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
    n_years = len(index) / 252
    cagr = float((equity.iloc[-1]) ** (1 / n_years) - 1) if n_years > 0 else 0
    
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0
    
    # Worst drawdown period
    dd_end = drawdown.idxmin()
    dd_peak = equity[:dd_end].idxmax()
    
    # Recovery
    post_dd = equity[dd_end:]
    recovered = post_dd[post_dd >= equity[dd_peak]]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None
    
    # Time in market
    time_in_market = float((returns != 0).mean())
    
    # Monthly returns for win rate
    monthly = returns.resample("ME").sum()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0
    
    # Longest drawdown duration
    in_dd = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    dd_lengths = in_dd.groupby(dd_groups).sum()
    longest_dd_days = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    
    return {
        "total_return": round(total_ret * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_monthly": round(win_rate * 100, 1),
        "time_in_market": round(time_in_market * 100, 1),
        "ann_return": round(float(ann_ret * 100), 2),
        "ann_vol": round(float(ann_vol * 100), 2),
        "worst_dd_start": str(dd_peak.date()) if hasattr(dd_peak, 'date') else str(dd_peak),
        "worst_dd_end": str(dd_end.date()) if hasattr(dd_end, 'date') else str(dd_end),
        "recovery_date": str(recovery_date.date()) if recovery_date is not None and hasattr(recovery_date, 'date') else None,
        "longest_dd_days": longest_dd_days,
    }


def _crash_analysis(
    equity: pd.Series, portfolio_ret: pd.Series,
    asset_rets: Dict[str, pd.Series], index: pd.DatetimeIndex,
) -> dict:
    """Analyze performance during known crypto crash periods."""
    periods = {
        "2018_bear": ("2018-01-01", "2018-12-31"),
        "covid_mar2020": ("2020-02-15", "2020-04-15"),
        "may2021_crash": ("2021-04-15", "2021-07-20"),
        "2022_bear": ("2022-01-01", "2022-12-31"),
    }
    
    analysis = {}
    for name, (start, end) in periods.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        mask = (index >= s) & (index <= e)
        if mask.sum() < 5:
            continue
        
        period_ret = portfolio_ret[mask]
        period_eq = (1 + period_ret).cumprod()
        period_dd = float((period_eq / period_eq.cummax() - 1).min())
        period_total = float(period_eq.iloc[-1] - 1) if len(period_eq) > 0 else 0
        
        per_asset_dd = {}
        for a, ar in asset_rets.items():
            a_period = ar[mask]
            if len(a_period) > 0:
                a_eq = (1 + a_period).cumprod()
                per_asset_dd[a] = round(float((a_eq / a_eq.cummax() - 1).min()) * 100, 2)
        
        analysis[name] = {
            "portfolio_return": round(period_total * 100, 2),
            "portfolio_max_dd": round(period_dd * 100, 2),
            "per_asset_dd": per_asset_dd,
        }
    
    return analysis


def _risk_metrics(returns: pd.Series) -> dict:
    """VaR, CVaR, and other risk metrics."""
    var_95 = float(returns.quantile(0.05))
    var_99 = float(returns.quantile(0.01))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95
    cvar_99 = float(returns[returns <= var_99].mean()) if (returns <= var_99).any() else var_99
    
    return {
        "var_95_daily": round(var_95 * 100, 3),
        "var_99_daily": round(var_99 * 100, 3),
        "cvar_95_daily": round(cvar_95 * 100, 3),
        "cvar_99_daily": round(cvar_99 * 100, 3),
    }
