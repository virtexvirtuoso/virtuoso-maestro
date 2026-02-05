"""
Result Converter - Convert V2 (VectorBT) results to V1 schema for frontend compatibility.

This module provides conversion functions that map VectorBT engine results to the
Backtrader-compatible schema expected by the existing frontend. This ensures that
V2 results display correctly in the React UI without frontend modifications.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .vectorbt_engine import BacktestResult
from .walk_forward_optuna import WalkForwardResult


def convert_v2_to_v1_schema(result: BacktestResult) -> dict[str, Any]:
    """
    Convert a V2 BacktestResult to V1-compatible schema.

    Maps VectorBT metrics to the Backtrader analyzer format expected by frontend:
    - result.sharpe_ratio -> analyzers.PyFolio['Sharpe ratio']
    - result.annual_return -> analyzers.PyFolio['Annual return']
    - result.volatility -> analyzers.PyFolio['Annual volatility']
    - result.max_drawdown -> analyzers.PyFolio['Max drawdown']
    - result.vwr -> analyzers.PyFolio['VWR']
    - result.num_trades -> analyzers.TradeAnalyzer.total.total

    Args:
        result: BacktestResult from VectorBT engine

    Returns:
        Dict matching V1 optimization result schema
    """
    # Extract buy/sell markers for chart
    observers = extract_buysell_from_v2(result)

    # Build PyFolio analyzer dict (V1 uses space-separated keys with specific casing)
    pyfolio_metrics = {
        'Sharpe ratio': _safe_float(result.sharpe_ratio),
        'Annual return': _safe_float(result.annual_return),
        'Annual volatility': _safe_float(result.volatility),
        'Max drawdown': _safe_float(result.max_drawdown),
        'Calmar ratio': _safe_float(result.calmar_ratio),
        'Sortino ratio': _safe_float(result.sortino_ratio),
        'Cumulative returns': _safe_float(result.total_return),
        'VWR': _safe_float(result.vwr),
    }

    # Build TradeAnalyzer dict (V1 nested structure)
    trade_analyzer = {
        'total': {
            'total': result.num_trades,
            'open': 0,  # VectorBT doesn't expose open trades in same way
            'closed': result.num_trades,
        },
        'won': {
            'total': int(result.num_trades * result.win_rate) if result.win_rate else 0,
        },
        'lost': {
            'total': result.num_trades - int(result.num_trades * result.win_rate) if result.win_rate else 0,
        },
        'pnl': {
            'gross': {
                'total': _safe_float(result.total_return * 100000 if result.total_return else 0),
            },
            'net': {
                'total': _safe_float(result.total_return * 100000 if result.total_return else 0),
            }
        }
    }

    # Build complete V1-compatible result
    v1_result = {
        'analyzers': {
            'PyFolio': pyfolio_metrics,
            'TradeAnalyzer': trade_analyzer,
        },
        'observers': observers,
        'parameters': result.parameters or {},
        'processing_time': result.processing_time,
        'start_date': _datetime_to_timestamp_ms(result.start_date),
        'end_date': _datetime_to_timestamp_ms(result.end_date),
    }

    return v1_result


def extract_buysell_from_v2(result: BacktestResult) -> dict[str, Any]:
    """
    Extract buy/sell markers from V2 BacktestResult for chart display.

    The frontend expects observers in format:
    {
        'BuySell': {
            'buy': [[timestamp_ms, price], ...],
            'sell': [[timestamp_ms, price], ...]
        }
    }

    Args:
        result: BacktestResult with trades DataFrame

    Returns:
        Dict with BuySell observer data for chart markers
    """
    buys: list[list[float]] = []
    sells: list[list[float]] = []

    if result.trades is not None and len(result.trades) > 0:
        trades_df = result.trades

        # VectorBT trades DataFrame has columns like:
        # Entry Timestamp, Exit Timestamp, Entry Price, Exit Price, Size, PnL, etc.
        # Column names vary slightly between vectorbt and vectorbtpro

        # Detect column names
        entry_ts_col = _find_column(trades_df, ['Entry Timestamp', 'entry_timestamp', 'Entry Time'])
        exit_ts_col = _find_column(trades_df, ['Exit Timestamp', 'exit_timestamp', 'Exit Time'])
        entry_price_col = _find_column(trades_df, ['Entry Price', 'entry_price', 'Avg Entry Price'])
        exit_price_col = _find_column(trades_df, ['Exit Price', 'exit_price', 'Avg Exit Price'])
        direction_col = _find_column(trades_df, ['Direction', 'direction', 'Side'])

        for _, trade in trades_df.iterrows():
            # Get entry timestamp and price
            if entry_ts_col and entry_price_col:
                entry_ts = trade.get(entry_ts_col)
                entry_price = trade.get(entry_price_col)

                if entry_ts is not None and entry_price is not None:
                    ts_ms = _to_timestamp_ms(entry_ts)
                    if ts_ms is not None:
                        # Determine if buy or sell based on direction
                        is_long = True
                        if direction_col:
                            direction = trade.get(direction_col)
                            if direction is not None:
                                is_long = str(direction).lower() in ['long', '1', '1.0', 'buy']

                        if is_long:
                            buys.append([ts_ms, float(entry_price)])
                        else:
                            sells.append([ts_ms, float(entry_price)])

            # Get exit timestamp and price (opposite action)
            if exit_ts_col and exit_price_col:
                exit_ts = trade.get(exit_ts_col)
                exit_price = trade.get(exit_price_col)

                if exit_ts is not None and exit_price is not None:
                    ts_ms = _to_timestamp_ms(exit_ts)
                    if ts_ms is not None:
                        # Exit is opposite of entry direction
                        is_long = True
                        if direction_col:
                            direction = trade.get(direction_col)
                            if direction is not None:
                                is_long = str(direction).lower() in ['long', '1', '1.0', 'buy']

                        if is_long:
                            sells.append([ts_ms, float(exit_price)])
                        else:
                            buys.append([ts_ms, float(exit_price)])

    return {
        'BuySell': {
            'buy': sorted(buys, key=lambda x: x[0]) if buys else [],
            'sell': sorted(sells, key=lambda x: x[0]) if sells else [],
        }
    }


def convert_walkforward_v2_to_v1(wf_result: WalkForwardResult) -> dict[str, Any]:
    """
    Convert a V2 WalkForwardResult to V1-compatible schema.

    Args:
        wf_result: WalkForwardResult from WalkForwardOptuna engine

    Returns:
        Dict matching V1 walk-forward optimization result schema
    """
    fold_results_v1 = []

    for i, fold_result in enumerate(wf_result.fold_results):
        v1_fold = convert_v2_to_v1_schema(fold_result)
        v1_fold['num_split'] = i
        v1_fold['parameters'] = (
            wf_result.optimal_params_per_fold[i]
            if i < len(wf_result.optimal_params_per_fold)
            else fold_result.parameters
        )
        fold_results_v1.append(v1_fold)

    return {
        'fold_results': fold_results_v1,
        'aggregate_metrics': wf_result.aggregate_metrics,
        'processing_time': wf_result.total_processing_time,
    }


def _safe_float(value: Any) -> float | None:
    """Convert value to float, returning None for NaN/inf."""
    if value is None:
        return None
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def _datetime_to_timestamp_ms(dt: datetime | None) -> int | None:
    """Convert datetime to millisecond timestamp."""
    if dt is None:
        return None
    try:
        return int(dt.timestamp() * 1000)
    except (AttributeError, TypeError):
        return None


def _to_timestamp_ms(value: Any) -> int | None:
    """Convert various timestamp formats to milliseconds."""
    if value is None:
        return None

    try:
        # Already a timestamp in seconds
        if isinstance(value, (int, float)):
            # Assume seconds if small, milliseconds if large
            if value > 1e12:
                return int(value)
            return int(value * 1000)

        # pandas Timestamp
        if isinstance(value, pd.Timestamp):
            return int(value.timestamp() * 1000)

        # numpy datetime64
        if isinstance(value, np.datetime64):
            ts = pd.Timestamp(value)
            return int(ts.timestamp() * 1000)

        # datetime
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)

        # Try parsing as string
        if isinstance(value, str):
            ts = pd.to_datetime(value)
            return int(ts.timestamp() * 1000)

    except Exception:
        pass

    return None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column name from candidates list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None
