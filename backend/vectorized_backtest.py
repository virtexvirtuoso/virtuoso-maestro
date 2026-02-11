"""
Vectorized Perpetual Futures Backtest Engine

Uses Numba JIT compilation for 50-100x speedup over pure Python loops.
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from dataclasses import dataclass
from typing import Tuple


@njit(cache=True)
def calculate_atr_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized ATR calculation with Numba."""
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # ATR (simple moving average of TR)
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i-period+1:i+1])

    return atr


@njit(cache=True)
def run_backtest_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signals: np.ndarray,
    atr: np.ndarray,
    tp_mult: float = 3.0,
    sl_mult: float = 2.0,
    trailing_mult: float = 1.5,
    commission: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core backtest loop with Numba JIT.

    Returns:
        equity: Equity curve
        positions: Position at each bar (0=flat, 1=long, -1=short)
        trade_pnls: PnL of each closed trade
        trade_directions: Direction of each trade (1=long, -1=short)
        trade_exits: Exit bar index of each trade
    """
    n = len(close)

    # Pre-allocate output arrays
    equity = np.ones(n)
    positions = np.zeros(n, dtype=np.int32)

    # Trade tracking (max possible trades = n/2)
    max_trades = n // 2
    trade_pnls = np.zeros(max_trades)
    trade_directions = np.zeros(max_trades, dtype=np.int32)
    trade_exits = np.zeros(max_trades, dtype=np.int32)
    trade_count = 0

    # State variables
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
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

        # Check exit conditions for long position
        if position == 1:
            tp_price = entry_price + tp_mult * current_atr
            sl_price = entry_price - sl_mult * current_atr
            trailing_stop = highest_since_entry - trailing_mult * current_atr

            # Update highest
            if current_high > highest_since_entry:
                highest_since_entry = current_high

            # Check exits (use high/low for intrabar)
            if current_high >= tp_price:
                exit_price = tp_price
                exit_triggered = True
            elif current_low <= sl_price:
                exit_price = sl_price
                exit_triggered = True
            elif current_low <= trailing_stop and highest_since_entry > entry_price:
                exit_price = trailing_stop
                exit_triggered = True
            elif signal == -1:
                exit_price = current_price
                exit_triggered = True

        # Check exit conditions for short position
        elif position == -1:
            tp_price = entry_price - tp_mult * current_atr
            sl_price = entry_price + sl_mult * current_atr
            trailing_stop = lowest_since_entry + trailing_mult * current_atr

            # Update lowest
            if current_low < lowest_since_entry:
                lowest_since_entry = current_low

            # Check exits
            if current_low <= tp_price:
                exit_price = tp_price
                exit_triggered = True
            elif current_high >= sl_price:
                exit_price = sl_price
                exit_triggered = True
            elif current_high >= trailing_stop and lowest_since_entry < entry_price:
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

            # Record trade
            trade_pnls[trade_count] = pnl
            trade_directions[trade_count] = position
            trade_exits[trade_count] = i
            trade_count += 1

            position = 0

        # Enter new position
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = current_price
                highest_since_entry = current_price
                lowest_since_entry = 1e10
            elif signal == -1:
                position = -1
                entry_price = current_price
                lowest_since_entry = current_price
                highest_since_entry = 0.0

        equity[i] = current_equity
        positions[i] = position

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
        trade_exits[trade_count] = n - 1
        trade_count += 1

    return equity, positions, trade_pnls[:trade_count], trade_directions[:trade_count], trade_exits[:trade_count]


@dataclass
class FastBacktestResult:
    """Backtest result container."""
    strategy: str
    asset: str
    total_return: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    sharpe: float
    long_trades: int
    short_trades: int
    long_pnl: float
    short_pnl: float
    equity_curve: np.ndarray


def run_vectorized_backtest(
    strategy_name: str,
    df: pd.DataFrame,
    signals: pd.Series,
    tp_mult: float = 3.0,
    sl_mult: float = 2.0,
    trailing_mult: float = 1.5,
    commission: float = 0.001,
    asset: str = "BTC",
) -> FastBacktestResult:
    """
    Run vectorized backtest using Numba-accelerated core.

    Args:
        strategy_name: Name of the strategy
        df: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume'
        signals: Series of signals (1=long, -1=short, 0=flat)
        tp_mult: Take profit ATR multiplier
        sl_mult: Stop loss ATR multiplier
        trailing_mult: Trailing stop ATR multiplier
        commission: Commission per trade (0.001 = 0.1%)
        asset: Asset name for result
    """
    # Extract numpy arrays for Numba
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    sig = signals.values.astype(np.float64)

    # Pre-compute ATR
    atr = calculate_atr_fast(high, low, close, period=14)

    # Run core backtest
    equity, positions, trade_pnls, trade_dirs, trade_exits = run_backtest_core(
        close, high, low, sig, atr,
        tp_mult, sl_mult, trailing_mult, commission
    )

    # Calculate metrics
    total_return = (equity[-1] - 1.0) * 100
    num_trades = len(trade_pnls)

    if num_trades > 0:
        wins = np.sum(trade_pnls > 0)
        win_rate = wins / num_trades * 100

        long_mask = trade_dirs == 1
        short_mask = trade_dirs == -1
        long_trades = np.sum(long_mask)
        short_trades = np.sum(short_mask)
        long_pnl = np.sum(trade_pnls[long_mask]) * 100
        short_pnl = np.sum(trade_pnls[short_mask]) * 100
    else:
        win_rate = 0.0
        long_trades = 0
        short_trades = 0
        long_pnl = 0.0
        short_pnl = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = np.min(drawdown)

    # Sharpe ratio
    returns = np.diff(equity) / equity[:-1]
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365)
    else:
        sharpe = 0.0

    return FastBacktestResult(
        strategy=strategy_name,
        asset=asset,
        total_return=total_return,
        num_trades=num_trades,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        long_trades=long_trades,
        short_trades=short_trades,
        long_pnl=long_pnl,
        short_pnl=short_pnl,
        equity_curve=equity,
    )


def benchmark_vs_original():
    """Benchmark vectorized vs original implementation."""
    import time
    import duckdb
    from strategies import run_strategy

    # Load data
    conn = duckdb.connect("data/maestro.duckdb", read_only=True)
    df = conn.execute("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_futures
        WHERE symbol = 'BTC' AND timeframe = '4h'
        ORDER BY timestamp
    """).fetchdf()
    conn.close()

    # Generate signals
    signals = run_strategy("TSMOM", df)

    print("Vectorized Backtest Benchmark")
    print("=" * 50)
    print(f"Data: {len(df):,} bars")

    # Warm up JIT
    _ = run_vectorized_backtest("TSMOM", df, signals, asset="BTC")

    # Benchmark vectorized
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = run_vectorized_backtest("TSMOM", df, signals, asset="BTC")
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_vectorized = np.mean(times)
    print(f"\nVectorized (Numba): {avg_vectorized:.2f}ms per backtest")
    print(f"  Return: {result.total_return:+.1f}%")
    print(f"  Trades: {result.num_trades} (L:{result.long_trades} S:{result.short_trades})")
    print(f"  Win rate: {result.win_rate:.1f}%")

    # Compare to simple Python loop
    def python_loop_backtest(signals, df):
        pnl = 0.0
        position = 0
        entry_price = 0.0
        for i in range(1, len(signals)):
            if position == 1:
                pnl += (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            elif position == -1:
                pnl -= (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]

            if signals.iloc[i] == 1 and position != 1:
                position = 1
            elif signals.iloc[i] == -1 and position != -1:
                position = -1
        return pnl

    times_python = []
    for _ in range(10):
        start = time.perf_counter()
        _ = python_loop_backtest(signals, df)
        elapsed = (time.perf_counter() - start) * 1000
        times_python.append(elapsed)

    avg_python = np.mean(times_python)
    print(f"\nPure Python loop: {avg_python:.2f}ms per backtest")
    print(f"\nSpeedup: {avg_python / avg_vectorized:.0f}x faster")


if __name__ == "__main__":
    benchmark_vs_original()
