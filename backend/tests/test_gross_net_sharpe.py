"""
Tests for A3: gross vs net Sharpe split (spec Walk-Forward-Methodology-Upgrade-Spec-2026-06-30 §A3).

Gross Sharpe = same signals with fees=0, slippage=0. Net Sharpe = costed run.
Optuna keeps optimizing on net; gross is reporting-layer only.

No mocks — every test runs the real VectorBT engine on synthetic data
(a passing test proves the cost path, per the repo anti-cheating rule).
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v2.vectorbt_engine import VectorBTEngine, BacktestConfig, BacktestResult
from engine_v2.walk_forward_optuna import WalkForwardResult


@pytest.fixture
def price_data():
    """Random-walk OHLCV, 1000 daily bars."""
    np.random.seed(42)
    n = 1000
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


def _high_turnover_signals(index):
    """Enter/exit on alternating bars — maximal turnover."""
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)
    entries.iloc[0::4] = True
    exits.iloc[2::4] = True
    return entries, exits


def _buy_and_hold_signals(index):
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)
    entries.iloc[0] = True
    return entries, exits


def test_costless_run_gross_equals_net(price_data):
    """With zero costs, gross and net Sharpe are the same number."""
    engine = VectorBTEngine(BacktestConfig(cash=100_000, commission=0.0, slippage=0.0))
    entries, exits = _high_turnover_signals(price_data.index)
    result = engine.run(price_data, entries, exits)
    assert result.sharpe_gross == result.sharpe_ratio


def test_high_turnover_net_strictly_below_gross(price_data):
    """Costs on a high-turnover strategy must strictly reduce Sharpe."""
    engine = VectorBTEngine(BacktestConfig(cash=100_000, commission=0.001, slippage=0.0005))
    entries, exits = _high_turnover_signals(price_data.index)
    result = engine.run(price_data, entries, exits)
    assert result.num_trades > 100  # sanity: turnover actually high
    assert result.sharpe_gross > result.sharpe_ratio


def test_buy_and_hold_gross_close_to_net(price_data):
    """Zero-turnover: one entry fee only, gross ≈ net."""
    engine = VectorBTEngine(BacktestConfig(cash=100_000, commission=0.001, slippage=0.0005))
    entries, exits = _buy_and_hold_signals(price_data.index)
    result = engine.run(price_data, entries, exits)
    assert result.sharpe_gross >= result.sharpe_ratio  # costs never help
    assert abs(result.sharpe_gross - result.sharpe_ratio) < 0.05


def test_gross_sharpe_reflects_costless_portfolio(price_data):
    """sharpe_gross of a costed run equals sharpe_ratio of a costless run
    with identical signals — the definition of 'gross'."""
    entries, exits = _high_turnover_signals(price_data.index)
    costed = VectorBTEngine(BacktestConfig(cash=100_000, commission=0.001, slippage=0.0005))
    costless = VectorBTEngine(BacktestConfig(cash=100_000, commission=0.0, slippage=0.0))
    r_costed = costed.run(price_data, entries, exits)
    r_costless = costless.run(price_data, entries, exits)
    assert r_costed.sharpe_gross == pytest.approx(r_costless.sharpe_ratio, rel=1e-9)


def test_walkforward_aggregate_includes_avg_sharpe_gross():
    """WalkForwardResult aggregate metrics carry avg_sharpe_gross from folds."""

    def fold(sharpe_net, sharpe_gross):
        return BacktestResult(
            total_return=0.1,
            sharpe_ratio=sharpe_net,
            max_drawdown=0.05,
            win_rate=0.5,
            profit_factor=1.2,
            num_trades=10,
            annual_return=0.1,
            volatility=0.2,
            calmar_ratio=1.0,
            sortino_ratio=1.0,
            vwr=0.1,
            sharpe_gross=sharpe_gross,
        )

    wf = WalkForwardResult(
        fold_results=[fold(0.8, 1.2), fold(1.0, 1.6)],
        optimal_params_per_fold=[{}, {}],
    )
    assert wf.aggregate_metrics["avg_sharpe_gross"] == pytest.approx(1.4)
    assert wf.aggregate_metrics["avg_sharpe"] == pytest.approx(0.9)
