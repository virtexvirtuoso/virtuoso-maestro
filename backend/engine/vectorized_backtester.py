"""
Vectorized Backtester for Strategy Screening

Fast backtesting using vectorized operations for screening all strategies
across all symbols and timeframes.

Fixed issues:
- Timeframe-aware Sharpe ratio annualization
- Proper commission handling (per-trade, not per-return)
- Compound trade returns (not sum)
- Numerical stability via log returns
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union


# Timeframe to periods-per-year mapping (crypto markets: 24/7/365)
TIMEFRAME_PERIODS = {
    '1m': 365.25 * 24 * 60,      # 525,960
    '3m': 365.25 * 24 * 20,      # 175,320
    '5m': 365.25 * 24 * 12,      # 105,192
    '15m': 365.25 * 24 * 4,      # 35,064
    '30m': 365.25 * 24 * 2,      # 17,532
    '1h': 365.25 * 24,           # 8,766
    '2h': 365.25 * 12,           # 4,383
    '4h': 365.25 * 6,            # 2,191.5
    '6h': 365.25 * 4,            # 1,461
    '8h': 365.25 * 3,            # 1,095.75
    '12h': 365.25 * 2,           # 730.5
    '1d': 365.25,                # 365.25
    '3d': 365.25 / 3,            # 121.75
    '1w': 52.18,                 # 52.18
    '1M': 12,                    # 12
}


def infer_timeframe(df: pd.DataFrame) -> str:
    """
    Infer timeframe from timestamp differences.

    Args:
        df: DataFrame with 'timestamp' column

    Returns:
        Timeframe string (e.g., '1h', '1d')
    """
    if 'timestamp' not in df.columns or len(df) < 2:
        return '1h'  # Default fallback

    # Get median time delta (robust to gaps)
    timestamps = pd.to_datetime(df['timestamp'])
    deltas = timestamps.diff().dropna()

    if len(deltas) == 0:
        return '1h'

    median_delta = deltas.median()
    minutes = median_delta.total_seconds() / 60

    # Map to standard timeframes
    if minutes <= 1.5:
        return '1m'
    elif minutes <= 4:
        return '3m'
    elif minutes <= 7:
        return '5m'
    elif minutes <= 20:
        return '15m'
    elif minutes <= 45:
        return '30m'
    elif minutes <= 90:
        return '1h'
    elif minutes <= 150:
        return '2h'
    elif minutes <= 300:
        return '4h'
    elif minutes <= 420:
        return '6h'
    elif minutes <= 600:
        return '8h'
    elif minutes <= 900:
        return '12h'
    elif minutes <= 2000:
        return '1d'
    elif minutes <= 5000:
        return '3d'
    elif minutes <= 15000:
        return '1w'
    else:
        return '1M'


def get_periods_per_year(timeframe: str) -> float:
    """Get annualization factor for a timeframe."""
    return TIMEFRAME_PERIODS.get(timeframe, 8766)  # Default to 1h


def calculate_returns(df: pd.DataFrame, signals: pd.Series,
                      commission: float = 0.001,
                      slippage: float = 0.0005) -> pd.Series:
    """
    Calculate strategy returns from signals.

    Args:
        df: OHLCV DataFrame
        signals: Series of signals (1=long, -1=short, 0=flat)
        commission: Trading commission per trade (default 0.1%)
        slippage: Slippage per trade (default 0.05%)

    Returns:
        Series of strategy returns (log returns for stability)
    """
    # Use log returns for numerical stability on high-frequency data
    # log(1 + r) ≈ r for small r, but compounds correctly: sum(log_returns) = log(total_return)
    price_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    # Position from previous bar's signal (enter on next bar)
    position = signals.shift(1).fillna(0)

    # Strategy returns = position * price returns
    strategy_returns = position * price_returns

    # Identify trade entries/exits (position changes)
    position_changes = position.diff().abs().fillna(0)

    # Commission + slippage cost per trade (as log return impact)
    # For a round-trip trade: entry + exit = 2 * (commission + slippage)
    trade_cost = commission + slippage

    # Apply costs only when position changes (not on every bar)
    # Use log approximation: log(1 - cost) ≈ -cost for small cost
    strategy_returns = strategy_returns - (position_changes > 0) * trade_cost

    return strategy_returns


def calculate_metrics(returns: pd.Series, signals: pd.Series,
                      timeframe: str = '1h') -> Dict[str, Any]:
    """
    Calculate performance metrics from returns.

    Args:
        returns: Series of strategy log returns
        signals: Series of signals
        timeframe: Data timeframe for Sharpe annualization

    Returns:
        Dictionary of performance metrics
    """
    # Handle edge cases
    if len(returns) < 10 or returns.std() == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'avg_trade': 0.0,
            'calmar_ratio': 0.0,
        }

    # Total return from log returns: exp(sum(log_returns)) - 1
    total_log_return = returns.sum()
    total_return = np.expm1(total_log_return)  # exp(x) - 1, numerically stable

    # Clip extreme returns for display (but keep actual calculation)
    total_return_display = np.clip(total_return, -0.9999, 100.0)

    # Sharpe ratio with CORRECT annualization
    periods_per_year = get_periods_per_year(timeframe)
    annualization_factor = np.sqrt(periods_per_year)

    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return * annualization_factor) if std_return > 0 else 0

    # Sortino ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
    sortino = (mean_return / downside_std * annualization_factor) if downside_std > 0 else 0

    # Max drawdown from cumulative log returns
    cumulative_log = returns.cumsum()
    running_max = cumulative_log.cummax()
    drawdown = cumulative_log - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = np.expm1(max_drawdown)  # Convert log drawdown to percentage

    # Calmar ratio (annual return / max drawdown)
    # Approximate annual return from mean
    annual_return = mean_return * periods_per_year
    calmar = abs(annual_return / max_drawdown) if max_drawdown < -0.001 else 0

    # Trade analysis
    position = signals.shift(1).fillna(0)
    position_changes = position.diff().fillna(0)

    # A trade starts when position changes from 0 or changes direction
    trade_entries = (position_changes != 0) & (position != 0)
    trade_exits = (position_changes != 0) & (position.shift(-1).fillna(0) != position)

    # Group returns by trade using cumulative entry count
    trade_id = trade_entries.cumsum()

    # Calculate compound return per trade (using log returns)
    trade_log_returns = returns.groupby(trade_id).sum()
    trade_returns = np.expm1(trade_log_returns)  # Convert back to simple returns

    # Filter out periods with no position (trade_id 0 before first trade)
    if 0 in trade_returns.index:
        trade_returns = trade_returns.drop(0)

    # Remove zero-return "trades" (flat periods)
    trade_returns = trade_returns[trade_returns.abs() > 1e-10]

    total_trades = len(trade_returns)

    if total_trades > 0:
        winning_trades = (trade_returns > 0).sum()
        win_rate = winning_trades / total_trades

        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float('inf') if gross_profit > 0 else 0
        )

        avg_trade = trade_returns.mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0

    return {
        'total_return': round(total_return_display * 100, 2),  # Percentage
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'max_drawdown': round(max_drawdown_pct * 100, 2),  # Percentage (negative)
        'win_rate': round(win_rate * 100, 2),  # Percentage
        'profit_factor': round(min(profit_factor, 999.99), 2),  # Cap display
        'total_trades': total_trades,
        'avg_trade': round(avg_trade * 100, 4),  # Percentage
        'calmar_ratio': round(min(calmar, 99.99), 2),  # Cap display
    }


def backtest_strategy(df: pd.DataFrame, strategy_func,
                      strategy_name: str,
                      timeframe: Optional[str] = None,
                      commission: float = 0.001,
                      slippage: float = 0.0005,
                      **kwargs) -> Dict[str, Any]:
    """
    Run a single strategy backtest.

    Args:
        df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
        strategy_func: Strategy function that takes df and returns signals Series
        strategy_name: Name of the strategy
        timeframe: Data timeframe (inferred from data if not provided)
        commission: Trading commission per trade
        slippage: Slippage per trade
        **kwargs: Additional parameters for strategy

    Returns:
        Dictionary with strategy name and metrics
    """
    try:
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return {'strategy': strategy_name, 'error': 'Missing required columns'}

        # Infer timeframe if not provided
        if timeframe is None:
            timeframe = infer_timeframe(df)

        # Generate signals
        signals = strategy_func(df, **kwargs)

        # Validate signals
        if signals is None or len(signals) == 0:
            return {'strategy': strategy_name, 'error': 'No signals generated'}

        # Ensure signals are numeric and aligned
        signals = pd.Series(signals, index=df.index).fillna(0)

        # Calculate returns and metrics
        returns = calculate_returns(df, signals, commission, slippage)
        metrics = calculate_metrics(returns, signals, timeframe)

        return {
            'strategy': strategy_name,
            'timeframe_detected': timeframe,
            **metrics,
            'error': None
        }

    except Exception as e:
        return {
            'strategy': strategy_name,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'avg_trade': 0.0,
            'calmar_ratio': 0.0,
            'error': str(e)
        }


def run_all_strategies(df: pd.DataFrame, strategies: Dict[str, callable],
                       timeframe: Optional[str] = None) -> pd.DataFrame:
    """
    Run all strategies on a single dataset.

    Args:
        df: OHLCV DataFrame
        strategies: Dict of {name: function}
        timeframe: Data timeframe (optional)

    Returns:
        DataFrame with results for all strategies
    """
    results = []
    for name, func in strategies.items():
        result = backtest_strategy(df, func, name, timeframe=timeframe)
        results.append(result)

    return pd.DataFrame(results)
