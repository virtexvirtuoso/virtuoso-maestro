"""
Tests for transaction-cost realism (feature/honest-metrics-dsr-pbo, Workstream A).

Proves:
  1. slippage actually flows through VectorBTEngine into net performance
     (a real backtest, not a mock — a passing test proves the cost path works).
  2. slippage impact scales with turnover (zero-turnover is ~immune).
  3. the walk-forward entry points now expose a non-zero slippage default,
     so the WF pipeline no longer reports slippage-free Sharpe.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from engine_v2.vectorbt_engine import BacktestConfig, VectorBTEngine


def _trending_ohlcv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    # random walk with mild drift so a trend-follower actually trades
    steps = rng.standard_normal(n) * 0.02 + 0.001
    close = 100 * np.exp(np.cumsum(steps))
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000.0},
        index=idx,
    )
    return df


def _churn_signals(df, fast=5, slow=20):
    """SMA crossover -> repeated entries/exits -> real turnover."""
    close = df["close"]
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    entries = (f > s) & (f.shift(1) <= s.shift(1))
    exits = (f < s) & (f.shift(1) >= s.shift(1))
    return entries.fillna(False), exits.fillna(False)


def test_slippage_reduces_net_return():
    df = _trending_ohlcv()
    entries, exits = _churn_signals(df)

    no_slip = VectorBTEngine(BacktestConfig(commission=0.001, slippage=0.0, allow_short=False))
    with_slip = VectorBTEngine(BacktestConfig(commission=0.001, slippage=0.005, allow_short=False))

    r0 = no_slip.run(df, entries, exits)
    r1 = with_slip.run(df, entries, exits)

    assert r0.num_trades > 0, "test needs a strategy that actually trades"
    assert r1.total_return < r0.total_return, "slippage must reduce net return"


def test_zero_turnover_is_immune_to_slippage():
    df = _trending_ohlcv(seed=1)
    # enter once on day 25, never exit -> ~1 trade, minimal slippage exposure
    entries = pd.Series(False, index=df.index)
    entries.iloc[25] = True
    exits = pd.Series(False, index=df.index)

    r0 = VectorBTEngine(BacktestConfig(slippage=0.0, allow_short=False)).run(df, entries, exits)
    r1 = VectorBTEngine(BacktestConfig(slippage=0.01, allow_short=False)).run(df, entries, exits)

    # a single entry: net returns should be nearly identical (one fill's slippage)
    assert r0.total_return == pytest.approx(r1.total_return, rel=0.05)


@pytest.mark.parametrize("fn_path", [
    "engine_v2.walk_forward_optuna.run_simple_walkforward",
    "engine_v2.parallel_walk_forward.run_parallel_walkforward",
])
def test_wf_entrypoints_expose_nonzero_slippage_default(fn_path):
    mod_name, fn_name = fn_path.rsplit(".", 1)
    mod = __import__(mod_name, fromlist=[fn_name])
    fn = getattr(mod, fn_name)
    params = inspect.signature(fn).parameters
    assert "slippage" in params, f"{fn_path} must accept slippage"
    assert params["slippage"].default > 0.0, "slippage default must be non-zero (was silently 0.0)"
