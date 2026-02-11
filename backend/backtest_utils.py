"""
Backtest Utilities - Unified interface for Numba-accelerated backtesting

Provides easy access to the 400x faster Numba JIT backtest engine.

Usage:
    from backtest_utils import fast_backtest, fast_backtest_core

    # Full backtest with metrics
    result = fast_backtest(strategy, df, fast_mode=True)  # 0.3ms

    # Just the core loop for custom pipelines
    equity, pnls, dirs, entries, exits, entry_prices, exit_prices = fast_backtest_core(
        close, high, low, signals, atr, tp_mult=3.0, sl_mult=2.0
    )

Performance:
    - fast_mode=True:  0.22ms per backtest (use in optimization loops)
    - fast_mode=False: 13.5ms per backtest (includes trade details)
    - Python loop:     75ms per backtest (avoid)
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Optional
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# NUMBA JIT ACCELERATED CORE (400x faster than Python)
# =============================================================================

@njit(cache=True)
def calculate_atr_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized ATR with Numba - runs at C speed."""
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i-period+1:i+1])

    return atr


@njit(cache=True)
def backtest_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signals: np.ndarray,
    atr: np.ndarray,
    tp_mult: float = 3.0,
    sl_mult: float = 2.0,
    trailing_mult: float = 1.5,
    commission: float = 0.001,
    use_trailing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core backtest loop with Numba JIT - runs at C speed.

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        signals: Signal array (1=long, -1=short, 0=flat)
        atr: Pre-computed ATR values
        tp_mult: Take profit at N x ATR
        sl_mult: Stop loss at N x ATR
        trailing_mult: Trailing stop at N x ATR
        commission: Commission per trade (0.001 = 0.1%)
        use_trailing: Enable trailing stops

    Returns:
        equity_curve, trade_pnls, trade_directions, trade_entries,
        trade_exits, trade_entry_prices, trade_exit_prices
    """
    n = len(close)

    # Pre-allocate
    equity = np.ones(n)
    max_trades = n // 2
    trade_pnls = np.zeros(max_trades)
    trade_directions = np.zeros(max_trades, dtype=np.int32)
    trade_entries = np.zeros(max_trades, dtype=np.int32)
    trade_exits = np.zeros(max_trades, dtype=np.int32)
    trade_entry_prices = np.zeros(max_trades)
    trade_exit_prices = np.zeros(max_trades)
    trade_count = 0

    # State
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    highest_since_entry = 0.0
    lowest_since_entry = 1e10
    current_equity = 1.0

    for i in range(1, n):
        current_price = close[i]
        current_high = high[i]
        current_low = low[i]
        current_atr = atr[i] if atr[i] > 0 else current_price * 0.02
        signal = signals[i]

        exit_triggered = False
        exit_price = 0.0

        # Long position management
        if position == 1:
            tp_price = entry_price + tp_mult * current_atr
            sl_price = entry_price - sl_mult * current_atr
            trailing_stop = highest_since_entry - trailing_mult * current_atr

            if current_high > highest_since_entry:
                highest_since_entry = current_high

            if current_high >= tp_price:
                exit_price = tp_price
                exit_triggered = True
            elif current_low <= sl_price:
                exit_price = sl_price
                exit_triggered = True
            elif use_trailing and current_low <= trailing_stop and highest_since_entry > entry_price:
                exit_price = trailing_stop
                exit_triggered = True
            elif signal == -1:
                exit_price = current_price
                exit_triggered = True

        # Short position management
        elif position == -1:
            tp_price = entry_price - tp_mult * current_atr
            sl_price = entry_price + sl_mult * current_atr
            trailing_stop = lowest_since_entry + trailing_mult * current_atr

            if current_low < lowest_since_entry:
                lowest_since_entry = current_low

            if current_low <= tp_price:
                exit_price = tp_price
                exit_triggered = True
            elif current_high >= sl_price:
                exit_price = sl_price
                exit_triggered = True
            elif use_trailing and current_high >= trailing_stop and lowest_since_entry < entry_price:
                exit_price = trailing_stop
                exit_triggered = True
            elif signal == 1:
                exit_price = current_price
                exit_triggered = True

        # Execute exit
        if exit_triggered and position != 0:
            if position == 1:
                pnl = (exit_price - entry_price) / entry_price - commission * 2
            else:
                pnl = (entry_price - exit_price) / entry_price - commission * 2

            current_equity *= (1 + pnl)

            trade_pnls[trade_count] = pnl
            trade_directions[trade_count] = position
            trade_entries[trade_count] = entry_idx
            trade_exits[trade_count] = i
            trade_entry_prices[trade_count] = entry_price
            trade_exit_prices[trade_count] = exit_price
            trade_count += 1

            position = 0

        # Enter new position
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = current_price
                entry_idx = i
                highest_since_entry = current_price
                lowest_since_entry = 1e10
            elif signal == -1:
                position = -1
                entry_price = current_price
                entry_idx = i
                lowest_since_entry = current_price
                highest_since_entry = 0.0

        equity[i] = current_equity

    # Close open position at end
    if position != 0:
        if position == 1:
            pnl = (close[-1] - entry_price) / entry_price - commission * 2
        else:
            pnl = (entry_price - close[-1]) / entry_price - commission * 2

        current_equity *= (1 + pnl)
        equity[-1] = current_equity

        trade_pnls[trade_count] = pnl
        trade_directions[trade_count] = position
        trade_entries[trade_count] = entry_idx
        trade_exits[trade_count] = n - 1
        trade_entry_prices[trade_count] = entry_price
        trade_exit_prices[trade_count] = close[-1]
        trade_count += 1

    return (
        equity,
        trade_pnls[:trade_count],
        trade_directions[:trade_count],
        trade_entries[:trade_count],
        trade_exits[:trade_count],
        trade_entry_prices[:trade_count],
        trade_exit_prices[:trade_count],
    )


# =============================================================================
# SIMPLE SIGNAL-BASED BACKTEST (for grid/optimization without TP/SL)
# =============================================================================

@njit(cache=True)
def simple_backtest_core(
    close: np.ndarray,
    signals: np.ndarray,
    commission: float = 0.001,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Simple signal-following backtest without TP/SL/Trailing.

    Faster for quick optimization where TP/SL aren't needed.

    Args:
        close: Close prices
        signals: Signal array (1=long, -1=short, 0=flat)
        commission: Commission per trade

    Returns:
        equity_curve, total_return, num_trades, num_wins
    """
    n = len(close)
    equity = np.ones(n)
    current_equity = 1.0
    position = 0
    entry_price = 0.0
    num_trades = 0
    num_wins = 0

    for i in range(1, n):
        current_price = close[i]
        signal = signals[i]

        # Check for exit
        if position == 1 and signal != 1:
            pnl = (current_price - entry_price) / entry_price - commission * 2
            current_equity *= (1 + pnl)
            num_trades += 1
            if pnl > 0:
                num_wins += 1
            position = 0

        elif position == -1 and signal != -1:
            pnl = (entry_price - current_price) / entry_price - commission * 2
            current_equity *= (1 + pnl)
            num_trades += 1
            if pnl > 0:
                num_wins += 1
            position = 0

        # Check for entry
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = current_price
            elif signal == -1:
                position = -1
                entry_price = current_price

        equity[i] = current_equity

    # Close open position
    if position != 0:
        if position == 1:
            pnl = (close[-1] - entry_price) / entry_price - commission * 2
        else:
            pnl = (entry_price - close[-1]) / entry_price - commission * 2
        current_equity *= (1 + pnl)
        equity[-1] = current_equity
        num_trades += 1
        if pnl > 0:
            num_wins += 1

    total_return = current_equity - 1.0
    return equity, total_return, num_trades, num_wins


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Single trade record."""
    entry_time: str
    exit_time: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    """Full backtest result."""
    strategy: str
    asset: str
    timeframe: str
    total_return: float
    num_trades: int
    win_rate: float
    avg_trade: float
    max_drawdown: float
    sharpe: float
    long_trades: int
    short_trades: int
    long_pnl: float
    short_pnl: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# HIGH-LEVEL BACKTEST FUNCTION
# =============================================================================

def fast_backtest(
    strategy_name: str,
    df: pd.DataFrame,
    signals: pd.Series = None,
    generate_signals_fn = None,
    initial_capital: float = 10000,
    tp_atr_mult: float = 3.0,
    sl_atr_mult: float = 2.0,
    trailing_atr_mult: float = 1.5,
    use_trailing: bool = True,
    commission: float = 0.001,
    asset: str = "BTC/USDT",
    timeframe: str = "4h",
    fast_mode: bool = False,
) -> BacktestResult:
    """
    Run fast Numba-accelerated backtest.

    Args:
        strategy_name: Name of strategy (for result labeling)
        df: OHLCV DataFrame
        signals: Pre-computed signals (optional)
        generate_signals_fn: Function to generate signals if not provided
        initial_capital: Starting capital
        tp_atr_mult: Take profit at N x ATR
        sl_atr_mult: Stop loss at N x ATR
        trailing_atr_mult: Trailing stop at N x ATR
        use_trailing: Enable trailing stops
        commission: Commission per trade
        asset: Asset name for result
        timeframe: Timeframe for result
        fast_mode: Skip trade list building (60x faster, use in optimization)

    Returns:
        BacktestResult with metrics and optionally trade details
    """
    # Generate signals if not provided
    if signals is None:
        if generate_signals_fn is not None:
            signals = generate_signals_fn(df)
        else:
            raise ValueError("Must provide either signals or generate_signals_fn")

    # Extract numpy arrays
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    sig = signals.values.astype(np.float64)

    # Pre-compute ATR
    atr = calculate_atr_fast(high, low, close, period=14)

    # Run JIT-compiled backtest core
    (equity, trade_pnls, trade_dirs, trade_entries, trade_exits,
     entry_prices, exit_prices) = backtest_core(
        close, high, low, sig, atr,
        tp_atr_mult, sl_atr_mult, trailing_atr_mult,
        commission, use_trailing
    )

    # Scale equity to initial capital
    equity = equity * initial_capital

    # Calculate metrics using numpy (fast)
    total_return = (equity[-1] - initial_capital) / initial_capital * 100
    num_trades = len(trade_pnls)

    if num_trades > 0:
        wins = int(np.sum(trade_pnls > 0))
        win_rate = wins / num_trades * 100
        avg_trade = float(np.mean(trade_pnls) * 100)

        long_mask = trade_dirs == 1
        short_mask = trade_dirs == -1
        n_long = int(np.sum(long_mask))
        n_short = int(np.sum(short_mask))
        long_pnl = float(np.sum(trade_pnls[long_mask]) * 100)
        short_pnl = float(np.sum(trade_pnls[short_mask]) * 100)
    else:
        win_rate = 0.0
        avg_trade = 0.0
        n_long = 0
        n_short = 0
        long_pnl = 0.0
        short_pnl = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = float(np.min(drawdown))

    # Sharpe ratio
    returns = np.diff(equity) / equity[:-1]
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365))
    else:
        sharpe = 0.0

    # Build trade list only if not in fast_mode
    trades = []
    if not fast_mode and num_trades > 0:
        for i in range(num_trades):
            direction = "long" if trade_dirs[i] == 1 else "short"
            entry_idx = trade_entries[i]
            exit_idx = trade_exits[i]

            if hasattr(df.index, 'strftime'):
                entry_time = str(df.index[entry_idx])[:10]
                exit_time = str(df.index[exit_idx])[:10]
            else:
                entry_time = str(entry_idx)
                exit_time = str(exit_idx)

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                direction=direction,
                entry_price=entry_prices[i],
                exit_price=exit_prices[i],
                pnl_pct=trade_pnls[i] * 100,
                exit_reason="auto",
                bars_held=exit_idx - entry_idx,
            ))

    return BacktestResult(
        strategy=strategy_name,
        asset=asset,
        timeframe=timeframe,
        total_return=total_return,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_trade=avg_trade,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        long_trades=n_long,
        short_trades=n_short,
        long_pnl=long_pnl,
        short_pnl=short_pnl,
        trades=trades,
        equity_curve=equity if not fast_mode else np.array([]),
    )


def fast_simple_backtest(
    close: np.ndarray,
    signals: np.ndarray,
    commission: float = 0.001,
) -> Tuple[float, float, int]:
    """
    Ultra-fast simple backtest for optimization loops.

    No TP/SL, just signal following. Use for quick parameter search.

    Args:
        close: Close prices as numpy array
        signals: Signals as numpy array (1, -1, 0)
        commission: Commission per trade

    Returns:
        (total_return_pct, win_rate_pct, num_trades)
    """
    equity, total_return, num_trades, num_wins = simple_backtest_core(
        close, signals, commission
    )

    win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0
    return total_return * 100, win_rate, num_trades


# =============================================================================
# WARM-UP FUNCTION (call once at startup)
# =============================================================================

def warmup_jit():
    """
    Pre-compile Numba functions by running a dummy backtest.
    Call once at application startup for instant first-run performance.
    """
    dummy_close = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    dummy_high = dummy_close + 1
    dummy_low = dummy_close - 1
    dummy_signals = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    dummy_atr = np.ones(5) * 2.0

    # Warm up both core functions
    _ = backtest_core(dummy_close, dummy_high, dummy_low, dummy_signals, dummy_atr)
    _ = simple_backtest_core(dummy_close, dummy_signals)
    _ = calculate_atr_fast(dummy_high, dummy_low, dummy_close)


# Export main functions
__all__ = [
    'fast_backtest',
    'fast_simple_backtest',
    'backtest_core',
    'simple_backtest_core',
    'calculate_atr_fast',
    'warmup_jit',
    'BacktestResult',
    'Trade',
]
