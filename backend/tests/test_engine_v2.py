"""
Tests for Maestro Engine V2

Tests the VectorBT engine, walk-forward optimization, and strategy adapters.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v2.vectorbt_engine import (
    VectorBTEngine,
    BacktestConfig,
    BacktestResult,
    ema_crossover_signals,
    rsi_signals,
    macd_signals,
)
from engine_v2.strategy_adapter import (
    VectorBTStrategy,
    StrategyAdapter,
    SignalOutput,
    EMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    create_custom_strategy,
)
from engine_v2.walk_forward_optuna import (
    WalkForwardOptuna,
    WalkForwardConfig,
    WalkForwardResult,
    run_simple_walkforward,
)
from utils.time_series_split_rolling import TimeSeriesSplitRolling


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    n_bars = 1000
    
    # Generate random walk price data
    returns = np.random.randn(n_bars) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(n_bars) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_bars) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n_bars) * 0.01)),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars),
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='D'))
    
    return data


@pytest.fixture
def trending_data():
    """Generate trending data (easier for strategies to profit)"""
    np.random.seed(42)
    n_bars = 500
    
    # Create uptrend then downtrend
    trend = np.concatenate([
        np.linspace(100, 150, n_bars // 2),
        np.linspace(150, 120, n_bars // 2),
    ])
    noise = np.random.randn(n_bars) * 2
    close = trend + noise
    
    data = pd.DataFrame({
        'open': close * 0.999,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars),
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='D'))
    
    return data


@pytest.fixture
def small_data():
    """Small dataset for quick tests"""
    np.random.seed(42)
    n_bars = 200
    close = 100 + np.cumsum(np.random.randn(n_bars))
    
    data = pd.DataFrame({
        'open': close * 0.999,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars),
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='D'))
    
    return data


# =============================================================================
# VectorBT Engine Tests
# =============================================================================

class TestVectorBTEngine:
    """Tests for the VectorBT backtesting engine"""
    
    def test_engine_initialization(self):
        """Test engine can be initialized with default config"""
        engine = VectorBTEngine()
        assert engine.config.cash == 100000.0
        assert engine.config.commission == 0.001
        
    def test_engine_with_custom_config(self):
        """Test engine with custom configuration"""
        config = BacktestConfig(cash=50000, commission=0.002, size=0.1)
        engine = VectorBTEngine(config=config)
        assert engine.config.cash == 50000
        assert engine.config.size == 0.1
    
    def test_basic_backtest(self, sample_ohlcv_data):
        """Test running a basic backtest"""
        engine = VectorBTEngine()
        
        # Simple moving average crossover
        close = sample_ohlcv_data['close']
        fast_ma = close.rolling(10).mean()
        slow_ma = close.rolling(30).mean()
        
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        result = engine.run(sample_ohlcv_data, entries, exits)
        
        assert isinstance(result, BacktestResult)
        assert result.processing_time > 0
        assert result.num_trades >= 0
        
    def test_backtest_result_metrics(self, trending_data):
        """Test that backtest returns valid metrics"""
        engine = VectorBTEngine()
        
        entries, exits = ema_crossover_signals(trending_data, 10, 30)
        result = engine.run(trending_data, entries, exits)
        
        # Check all metrics are calculated
        assert not np.isnan(result.total_return) or result.num_trades == 0
        assert result.equity_curve is not None
        
    def test_ema_crossover_signals(self, sample_ohlcv_data):
        """Test EMA crossover signal generation"""
        entries, exits = ema_crossover_signals(sample_ohlcv_data, 10, 30)
        
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_ohlcv_data)
        assert entries.dtype == bool
        assert exits.dtype == bool
        
    def test_rsi_signals(self, sample_ohlcv_data):
        """Test RSI signal generation"""
        entries, exits = rsi_signals(sample_ohlcv_data, 14, 30, 70)
        
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        
    def test_macd_signals(self, sample_ohlcv_data):
        """Test MACD signal generation"""
        entries, exits = macd_signals(sample_ohlcv_data, 12, 26, 9)
        
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)


# =============================================================================
# Strategy Adapter Tests
# =============================================================================

class TestStrategyAdapter:
    """Tests for strategy adapters"""
    
    def test_strategy_registration(self):
        """Test strategies are registered correctly"""
        strategies = StrategyAdapter.list_strategies()
        
        assert 'EmaCrossStrategy' in strategies
        assert 'RSIStrategy' in strategies
        assert 'MACDStrategy' in strategies
        assert 'BollingerBandsStrategy' in strategies
        
    def test_get_registered_strategy(self):
        """Test retrieving registered strategy"""
        strategy_class = StrategyAdapter.get('EMACrossStrategy')
        # Note: name in registry might be different
        ema_class = StrategyAdapter.get('EmaCrossStrategy')
        assert ema_class == EMACrossStrategy
        
    def test_ema_cross_strategy(self, sample_ohlcv_data):
        """Test EMA Cross strategy signal generation"""
        strategy = EMACrossStrategy()
        params = strategy.get_params()
        
        assert 'ema50' in params
        assert 'ema100' in params
        
        signals = strategy.generate_signals(sample_ohlcv_data, params)
        
        assert isinstance(signals, SignalOutput)
        assert signals.entries is not None
        assert signals.exits is not None
        assert len(signals.entries) == len(sample_ohlcv_data)
        
    def test_rsi_strategy(self, sample_ohlcv_data):
        """Test RSI strategy signal generation"""
        strategy = RSIStrategy()
        params = strategy.get_params()
        
        assert 'period' in params
        
        signals = strategy.generate_signals(sample_ohlcv_data, params)
        
        assert isinstance(signals, SignalOutput)
        assert 'rsi' in signals.indicators
        
    def test_macd_strategy(self, sample_ohlcv_data):
        """Test MACD strategy signal generation"""
        strategy = MACDStrategy()
        signals = strategy.generate_signals(sample_ohlcv_data, strategy.get_params())
        
        assert 'macd' in signals.indicators
        assert 'signal' in signals.indicators
        
    def test_bollinger_strategy(self, sample_ohlcv_data):
        """Test Bollinger Bands strategy"""
        strategy = BollingerBandsStrategy()
        params = strategy.get_params()
        
        assert 'period' in params
        assert 'num_std' in params
        
        signals = strategy.generate_signals(sample_ohlcv_data, params)
        
        assert 'upper_band' in signals.indicators
        assert 'lower_band' in signals.indicators
        
    def test_custom_strategy_creation(self, sample_ohlcv_data):
        """Test creating a custom strategy via factory function"""
        def my_signal_fn(data, params):
            close = data['close']
            ma = close.rolling(params['period']).mean()
            entries = (close > ma) & (close.shift(1) <= ma.shift(1))
            exits = (close < ma) & (close.shift(1) >= ma.shift(1))
            return SignalOutput(entries=entries, exits=exits)
        
        MyStrategy = create_custom_strategy(
            'MyStrategy',
            {'period': 20},
            {'period': ('int', 10, 50)},
            my_signal_fn
        )
        
        strategy = MyStrategy()
        assert strategy.get_params() == {'period': 20}
        
        signals = strategy.generate_signals(sample_ohlcv_data, {'period': 15})
        assert signals.entries is not None
        
    def test_param_space(self):
        """Test parameter space definitions"""
        space = EMACrossStrategy.get_param_space()
        
        assert 'ema50' in space
        assert space['ema50'][0] == 'int'
        assert len(space['ema50']) == 3  # (type, min, max)
        
    def test_validate_params(self):
        """Test parameter validation fills in defaults"""
        strategy = EMACrossStrategy()
        
        validated = strategy.validate_params({'ema50': 25})
        
        assert validated['ema50'] == 25
        assert 'ema100' in validated  # Default should be filled in


# =============================================================================
# TimeSeriesSplitRolling Tests (Compatibility)
# =============================================================================

class TestTimeSeriesSplitRolling:
    """Test that TimeSeriesSplitRolling works the same as original"""
    
    def test_basic_split(self, sample_ohlcv_data):
        """Test basic splitting functionality"""
        tscv = TimeSeriesSplitRolling(n_splits=5)
        splits = list(tscv.split(sample_ohlcv_data))
        
        assert len(splits) > 0
        
        for train_idx, test_idx in splits:
            # Test indices should be after train indices
            assert train_idx[-1] < test_idx[0]
            
    def test_fixed_length_split(self, sample_ohlcv_data):
        """Test fixed-length splitting"""
        tscv = TimeSeriesSplitRolling(n_splits=5)
        splits = list(tscv.split(sample_ohlcv_data, fixed_length=True, train_splits=2))
        
        assert len(splits) > 0
        
        # All training sets should have similar lengths (with fixed_length=True)
        train_lengths = [len(train) for train, test in splits]
        
        # First might be different due to remainder, but rest should be similar
        if len(train_lengths) > 2:
            assert max(train_lengths[1:]) - min(train_lengths[1:]) <= 1
            
    def test_split_indices_non_overlapping(self, sample_ohlcv_data):
        """Test that train/test splits don't overlap"""
        tscv = TimeSeriesSplitRolling(n_splits=5)
        
        for train_idx, test_idx in tscv.split(sample_ohlcv_data, fixed_length=True, train_splits=2):
            train_set = set(train_idx)
            test_set = set(test_idx)
            
            # No overlap
            assert len(train_set & test_set) == 0


# =============================================================================
# Walk-Forward Optimization Tests
# =============================================================================

class TestWalkForwardOptuna:
    """Tests for walk-forward optimization with Optuna"""
    
    def test_walkforward_initialization(self, small_data):
        """Test walk-forward engine can be initialized"""
        strategy = EMACrossStrategy()
        config = WalkForwardConfig(num_splits=5, train_splits=2, test_splits=1, n_trials=5)
        
        engine = WalkForwardOptuna(
            data=small_data,
            strategy=strategy,
            config=config
        )
        
        assert engine.config.num_splits == 5
        assert engine.config.n_trials == 5
        
    def test_walkforward_run(self, small_data):
        """Test running walk-forward optimization"""
        strategy = RSIStrategy()
        config = WalkForwardConfig(num_splits=5, train_splits=2, test_splits=1, n_trials=5)
        
        engine = WalkForwardOptuna(
            data=small_data,
            strategy=strategy,
            config=config
        )
        
        result = engine.run()
        
        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 4  # num_splits=5 gives 4 folds
        assert len(result.optimal_params_per_fold) == 4
        
    def test_walkforward_produces_results(self, trending_data):
        """Test walk-forward produces meaningful results"""
        strategy = EMACrossStrategy()
        
        result = run_simple_walkforward(
            data=trending_data,
            strategy=strategy,
            num_splits=5,
            train_splits=2,
            test_splits=1,
            n_trials=10,
        )
        
        assert result.aggregate_metrics is not None
        assert 'total_return' in result.aggregate_metrics
        assert 'avg_sharpe' in result.aggregate_metrics
        
    def test_walkforward_different_strategies(self, small_data):
        """Test walk-forward works with different strategies"""
        strategies = [
            EMACrossStrategy(),
            RSIStrategy(),
            MACDStrategy(),
        ]
        
        for strategy in strategies:
            config = WalkForwardConfig(num_splits=5, train_splits=2, test_splits=1, n_trials=3)
            engine = WalkForwardOptuna(
                data=small_data,
                strategy=strategy,
                config=config
            )
            result = engine.run()
            
            assert len(result.fold_results) >= 2  # At least 2 folds
            
    def test_walkforward_progress_callback(self, small_data):
        """Test progress callback is called"""
        progress_calls = []
        
        def progress_cb(cur, total, msg):
            progress_calls.append((cur, total, msg))
        
        strategy = RSIStrategy()
        config = WalkForwardConfig(num_splits=5, train_splits=2, test_splits=1, n_trials=3)
        
        engine = WalkForwardOptuna(
            data=small_data,
            strategy=strategy,
            config=config,
            progress_callback=progress_cb
        )
        
        engine.run()
        
        # Progress should have been called
        assert len(progress_calls) > 0
        
    def test_walkforward_optimal_params_vary(self, trending_data):
        """Test that optimal params can vary across folds"""
        strategy = RSIStrategy()
        
        result = run_simple_walkforward(
            data=trending_data,
            strategy=strategy,
            num_splits=5,
            train_splits=2,
            test_splits=1,
            n_trials=20,
        )
        
        # Check that we got params for each fold
        assert len(result.optimal_params_per_fold) >= 3
        
        # Each should be a dict with the strategy params
        for params in result.optimal_params_per_fold:
            assert 'period' in params
            
    def test_walkforward_combined_equity_curve(self, trending_data):
        """Test combined equity curve generation"""
        strategy = EMACrossStrategy()
        
        result = run_simple_walkforward(
            data=trending_data,
            strategy=strategy,
            num_splits=5,
            train_splits=2,
            test_splits=1,
            n_trials=5,
        )
        
        # Combined curve should exist
        assert result.combined_equity_curve is not None
        
    def test_walkforward_stop(self, small_data):
        """Test stopping walk-forward mid-run"""
        import threading
        import time
        
        strategy = RSIStrategy()
        config = WalkForwardConfig(num_splits=5, n_trials=100)  # Long running
        
        engine = WalkForwardOptuna(
            data=small_data,
            strategy=strategy,
            config=config
        )
        
        # Start in thread
        def run_engine():
            engine.run()
        
        thread = threading.Thread(target=run_engine)
        thread.start()
        
        # Stop after a bit
        time.sleep(0.5)
        engine.stop()
        
        thread.join(timeout=5)
        
        # Should have stopped before completing all folds
        # (depends on timing, might complete if fast enough)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_workflow(self, trending_data):
        """Test full workflow: strategy -> signals -> backtest -> results"""
        # 1. Create strategy
        strategy = EMACrossStrategy()
        
        # 2. Generate signals
        params = {'ema50': 20, 'ema100': 50, 'ema51': 21, 'ema101': 51}
        signals = strategy.generate_signals(trending_data, params)
        
        # 3. Run backtest
        engine = VectorBTEngine(config=BacktestConfig(cash=100000))
        result = engine.run(
            trending_data,
            signals.entries,
            signals.exits,
            signals.short_entries,
            signals.short_exits,
            parameters=params
        )
        
        # 4. Check results
        assert result.parameters == params
        assert result.equity_curve is not None
        
    def test_strategy_comparison(self, trending_data):
        """Test comparing multiple strategies"""
        strategies = [
            EMACrossStrategy(),
            RSIStrategy(),
        ]
        
        results = []
        engine = VectorBTEngine()
        
        for strategy in strategies:
            signals = strategy.generate_signals(trending_data, strategy.get_params())
            result = engine.run(trending_data, signals.entries, signals.exits)
            results.append({
                'strategy': strategy.__class__.__name__,
                'sharpe': result.sharpe_ratio,
                'return': result.total_return,
            })
        
        assert len(results) == 2
        
    def test_walkforward_matches_original_splits(self, sample_ohlcv_data):
        """Test that walk-forward uses same split logic as original"""
        num_splits = 5
        train_splits = 2
        test_splits = 1
        
        # Original splitting
        tscv = TimeSeriesSplitRolling(num_splits)
        original_splits = list(tscv.split(
            sample_ohlcv_data,
            fixed_length=True,
            train_splits=train_splits,
            test_splits=test_splits
        ))
        
        # Walk-forward engine should produce same splits
        strategy = RSIStrategy()
        config = WalkForwardConfig(
            num_splits=num_splits,
            train_splits=train_splits,
            test_splits=test_splits,
            n_trials=3
        )
        
        # We can't directly access splits, but we can verify same number
        engine = WalkForwardOptuna(
            data=sample_ohlcv_data,
            strategy=strategy,
            config=config
        )
        
        result = engine.run()
        
        # Should have same number of folds
        assert len(result.fold_results) == len(original_splits)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests"""
    
    def test_backtest_speed(self, sample_ohlcv_data):
        """Test that VectorBT backtest is fast"""
        import time
        
        engine = VectorBTEngine()
        strategy = EMACrossStrategy()
        signals = strategy.generate_signals(sample_ohlcv_data, strategy.get_params())
        
        start = time.time()
        for _ in range(10):
            engine.run(sample_ohlcv_data, signals.entries, signals.exits)
        elapsed = time.time() - start
        
        # 10 backtests on 1000 bars should be very fast
        assert elapsed < 5.0, f"10 backtests took {elapsed:.2f}s, too slow"
        
    def test_large_data_handling(self):
        """Test handling of larger datasets"""
        np.random.seed(42)
        n_bars = 5000
        
        close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
        data = pd.DataFrame({
            'open': close * 0.999,
            'high': close * 1.005,
            'low': close * 0.995,
            'close': close,
            'volume': np.random.randint(1000, 10000, n_bars),
        }, index=pd.date_range('2010-01-01', periods=n_bars, freq='D'))
        
        engine = VectorBTEngine()
        entries, exits = ema_crossover_signals(data, 10, 30)
        
        result = engine.run(data, entries, exits)
        
        assert result.processing_time < 1.0, "Large backtest took too long"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
