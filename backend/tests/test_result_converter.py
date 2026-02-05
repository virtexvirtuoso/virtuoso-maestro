"""
Tests for result_converter.py - V2 to V1 schema conversion.

Validates that V2 (VectorBT) results convert correctly to the V1 (Backtrader)
schema expected by the frontend.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v2.result_converter import (
    _datetime_to_timestamp_ms,
    _safe_float,
    _to_timestamp_ms,
    convert_v2_to_v1_schema,
    convert_walkforward_v2_to_v1,
    extract_buysell_from_v2,
)
from engine_v2.vectorbt_engine import BacktestResult
from engine_v2.walk_forward_optuna import WalkForwardResult


class TestConvertV2ToV1Schema:
    """Tests for convert_v2_to_v1_schema function."""

    def test_basic_conversion(self):
        """Test basic BacktestResult to V1 schema conversion."""
        result = BacktestResult(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            win_rate=0.55,
            profit_factor=1.8,
            num_trades=50,
            annual_return=0.25,
            volatility=0.12,
            calmar_ratio=3.1,
            sortino_ratio=2.2,
            vwr=0.75,
            processing_time=1.5,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            parameters={'fast_period': 10, 'slow_period': 30},
        )

        v1_result = convert_v2_to_v1_schema(result)

        # Check structure
        assert 'analyzers' in v1_result
        assert 'observers' in v1_result
        assert 'parameters' in v1_result

        # Check PyFolio analyzer
        assert 'PyFolio' in v1_result['analyzers']
        pyfolio = v1_result['analyzers']['PyFolio']
        assert pyfolio['Sharpe ratio'] == 1.5
        assert pyfolio['Annual return'] == 0.25
        assert pyfolio['Annual volatility'] == 0.12
        assert pyfolio['Max drawdown'] == 0.08
        assert pyfolio['VWR'] == 0.75

        # Check TradeAnalyzer
        assert 'TradeAnalyzer' in v1_result['analyzers']
        trade_analyzer = v1_result['analyzers']['TradeAnalyzer']
        assert trade_analyzer['total']['total'] == 50

        # Check observers
        assert 'BuySell' in v1_result['observers']

        # Check parameters
        assert v1_result['parameters'] == {'fast_period': 10, 'slow_period': 30}

    def test_conversion_with_nan_values(self):
        """Test conversion handles NaN values correctly."""
        result = BacktestResult(
            total_return=float('nan'),
            sharpe_ratio=float('inf'),
            max_drawdown=0.0,
            win_rate=None,
            profit_factor=1.0,
            num_trades=0,
            annual_return=float('nan'),
            volatility=float('-inf'),
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            vwr=0.0,
        )

        v1_result = convert_v2_to_v1_schema(result)

        # NaN/inf values should be converted to None
        pyfolio = v1_result['analyzers']['PyFolio']
        assert pyfolio['Sharpe ratio'] is None  # inf -> None
        assert pyfolio['Annual return'] is None  # nan -> None
        assert pyfolio['Annual volatility'] is None  # -inf -> None

    def test_conversion_with_zero_trades(self):
        """Test conversion when no trades were executed."""
        result = BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            num_trades=0,
            annual_return=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            vwr=0.0,
        )

        v1_result = convert_v2_to_v1_schema(result)

        trade_analyzer = v1_result['analyzers']['TradeAnalyzer']
        assert trade_analyzer['total']['total'] == 0
        assert trade_analyzer['won']['total'] == 0
        assert trade_analyzer['lost']['total'] == 0


class TestExtractBuySellFromV2:
    """Tests for extract_buysell_from_v2 function."""

    def test_empty_trades(self):
        """Test extraction with no trades."""
        result = BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            num_trades=0,
            annual_return=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            vwr=0.0,
            trades=None,
        )

        observers = extract_buysell_from_v2(result)

        assert observers['BuySell']['buy'] == []
        assert observers['BuySell']['sell'] == []

    def test_with_trades_dataframe(self):
        """Test extraction with a trades DataFrame."""
        trades_df = pd.DataFrame({
            'Entry Timestamp': [
                datetime(2023, 1, 15),
                datetime(2023, 2, 20),
            ],
            'Exit Timestamp': [
                datetime(2023, 1, 25),
                datetime(2023, 3, 1),
            ],
            'Entry Price': [100.0, 110.0],
            'Exit Price': [105.0, 108.0],
            'Direction': ['Long', 'Long'],
        })

        result = BacktestResult(
            total_return=0.1,
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            win_rate=0.5,
            profit_factor=1.2,
            num_trades=2,
            annual_return=0.1,
            volatility=0.1,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            vwr=0.5,
            trades=trades_df,
        )

        observers = extract_buysell_from_v2(result)

        # Should have buy markers at entry times (long positions)
        assert len(observers['BuySell']['buy']) == 2
        # Should have sell markers at exit times (long positions)
        assert len(observers['BuySell']['sell']) == 2

        # Check buy prices
        buy_prices = [b[1] for b in observers['BuySell']['buy']]
        assert 100.0 in buy_prices
        assert 110.0 in buy_prices

    def test_timestamp_format(self):
        """Test that timestamps are in milliseconds."""
        trades_df = pd.DataFrame({
            'Entry Timestamp': [datetime(2023, 6, 15, 12, 0, 0)],
            'Exit Timestamp': [datetime(2023, 6, 20, 12, 0, 0)],
            'Entry Price': [100.0],
            'Exit Price': [110.0],
            'Direction': ['Long'],
        })

        result = BacktestResult(
            total_return=0.1,
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            win_rate=1.0,
            profit_factor=2.0,
            num_trades=1,
            annual_return=0.1,
            volatility=0.1,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            vwr=0.5,
            trades=trades_df,
        )

        observers = extract_buysell_from_v2(result)

        # Timestamps should be in milliseconds (> 1e12)
        for buy in observers['BuySell']['buy']:
            assert buy[0] > 1e12, "Timestamp should be in milliseconds"
        for sell in observers['BuySell']['sell']:
            assert sell[0] > 1e12, "Timestamp should be in milliseconds"


class TestConvertWalkforwardV2ToV1:
    """Tests for convert_walkforward_v2_to_v1 function."""

    def test_walkforward_conversion(self):
        """Test WalkForwardResult to V1 schema conversion."""
        # Create fold results
        fold_results = [
            BacktestResult(
                total_return=0.05,
                sharpe_ratio=1.0,
                max_drawdown=0.03,
                win_rate=0.5,
                profit_factor=1.2,
                num_trades=10,
                annual_return=0.08,
                volatility=0.1,
                calmar_ratio=2.0,
                sortino_ratio=1.5,
                vwr=0.4,
            ),
            BacktestResult(
                total_return=0.08,
                sharpe_ratio=1.5,
                max_drawdown=0.04,
                win_rate=0.6,
                profit_factor=1.5,
                num_trades=15,
                annual_return=0.12,
                volatility=0.08,
                calmar_ratio=3.0,
                sortino_ratio=2.0,
                vwr=0.6,
            ),
        ]

        wf_result = WalkForwardResult(
            fold_results=fold_results,
            optimal_params_per_fold=[
                {'fast_period': 10, 'slow_period': 30},
                {'fast_period': 12, 'slow_period': 28},
            ],
            total_processing_time=5.0,
        )

        v1_result = convert_walkforward_v2_to_v1(wf_result)

        # Check structure
        assert 'fold_results' in v1_result
        assert 'aggregate_metrics' in v1_result
        assert 'processing_time' in v1_result

        # Check fold results
        assert len(v1_result['fold_results']) == 2
        assert v1_result['fold_results'][0]['num_split'] == 0
        assert v1_result['fold_results'][1]['num_split'] == 1

        # Check parameters assigned correctly
        assert v1_result['fold_results'][0]['parameters']['fast_period'] == 10
        assert v1_result['fold_results'][1]['parameters']['fast_period'] == 12

        # Check aggregate metrics
        assert 'avg_sharpe' in v1_result['aggregate_metrics']
        assert 'total_trades' in v1_result['aggregate_metrics']


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_safe_float_normal(self):
        """Test _safe_float with normal values."""
        assert _safe_float(1.5) == 1.5
        assert _safe_float(0.0) == 0.0
        assert _safe_float(-1.5) == -1.5

    def test_safe_float_special(self):
        """Test _safe_float with special values."""
        assert _safe_float(float('nan')) is None
        assert _safe_float(float('inf')) is None
        assert _safe_float(float('-inf')) is None
        assert _safe_float(None) is None

    def test_datetime_to_timestamp_ms(self):
        """Test _datetime_to_timestamp_ms conversion."""
        dt = datetime(2023, 6, 15, 12, 0, 0)
        ts_ms = _datetime_to_timestamp_ms(dt)

        assert ts_ms is not None
        assert ts_ms > 1e12  # Milliseconds
        assert ts_ms == int(dt.timestamp() * 1000)

    def test_datetime_to_timestamp_ms_none(self):
        """Test _datetime_to_timestamp_ms with None."""
        assert _datetime_to_timestamp_ms(None) is None

    def test_to_timestamp_ms_various_formats(self):
        """Test _to_timestamp_ms with various input formats."""
        # datetime
        dt = datetime(2023, 6, 15)
        ts_from_dt = _to_timestamp_ms(dt)
        assert ts_from_dt > 1e12

        # pandas Timestamp
        pd_ts = pd.Timestamp('2023-06-15')
        ts_from_pd = _to_timestamp_ms(pd_ts)
        assert ts_from_pd > 1e12

        # seconds (small number)
        ts_seconds = 1686830400  # June 15, 2023 12:00:00 UTC
        ts_from_seconds = _to_timestamp_ms(ts_seconds)
        assert ts_from_seconds == ts_seconds * 1000

        # milliseconds (large number)
        ts_millis = 1686830400000
        ts_from_millis = _to_timestamp_ms(ts_millis)
        assert ts_from_millis == ts_millis


class TestV1SchemaValidation:
    """Tests to validate V1 schema structure matches frontend expectations."""

    def test_pyfolio_keys_match_v1(self):
        """Ensure PyFolio keys match V1 format (space-separated, specific casing)."""
        result = BacktestResult(
            total_return=0.1,
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            win_rate=0.5,
            profit_factor=1.2,
            num_trades=10,
            annual_return=0.1,
            volatility=0.1,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            vwr=0.5,
        )

        v1_result = convert_v2_to_v1_schema(result)
        pyfolio = v1_result['analyzers']['PyFolio']

        # V1 frontend expects these exact keys (from optimization_engine.py)
        expected_keys = {
            'Sharpe ratio',
            'Annual return',
            'Annual volatility',
            'Max drawdown',
            'Calmar ratio',
            'Sortino ratio',
            'Cumulative returns',
            'VWR',
        }

        for key in expected_keys:
            assert key in pyfolio, f"Missing PyFolio key: {key}"

    def test_buysell_format_matches_v1(self):
        """Ensure BuySell format matches V1 observer structure."""
        trades_df = pd.DataFrame({
            'Entry Timestamp': [datetime(2023, 1, 15)],
            'Exit Timestamp': [datetime(2023, 1, 25)],
            'Entry Price': [100.0],
            'Exit Price': [105.0],
            'Direction': ['Long'],
        })

        result = BacktestResult(
            total_return=0.05,
            sharpe_ratio=1.0,
            max_drawdown=0.02,
            win_rate=1.0,
            profit_factor=2.0,
            num_trades=1,
            annual_return=0.05,
            volatility=0.05,
            calmar_ratio=2.5,
            sortino_ratio=1.5,
            vwr=0.3,
            trades=trades_df,
        )

        v1_result = convert_v2_to_v1_schema(result)

        # V1 frontend expects observers.BuySell.buy/sell as [[timestamp_ms, price], ...]
        buysell = v1_result['observers']['BuySell']
        assert isinstance(buysell['buy'], list)
        assert isinstance(buysell['sell'], list)

        if buysell['buy']:
            assert isinstance(buysell['buy'][0], list)
            assert len(buysell['buy'][0]) == 2  # [timestamp, price]

    def test_trade_analyzer_structure(self):
        """Ensure TradeAnalyzer matches V1 nested structure."""
        result = BacktestResult(
            total_return=0.1,
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            win_rate=0.6,
            profit_factor=1.5,
            num_trades=20,
            annual_return=0.1,
            volatility=0.1,
            calmar_ratio=2.0,
            sortino_ratio=1.5,
            vwr=0.5,
        )

        v1_result = convert_v2_to_v1_schema(result)
        trade_analyzer = v1_result['analyzers']['TradeAnalyzer']

        # Check nested structure matches V1
        assert 'total' in trade_analyzer
        assert 'total' in trade_analyzer['total']
        assert 'won' in trade_analyzer
        assert 'total' in trade_analyzer['won']
        assert 'lost' in trade_analyzer
        assert 'total' in trade_analyzer['lost']
        assert 'pnl' in trade_analyzer
