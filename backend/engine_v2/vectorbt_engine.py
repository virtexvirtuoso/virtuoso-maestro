"""
VectorBT Engine - Core vectorized backtesting wrapper

Provides a high-performance backtesting engine using VectorBT for signal-based strategies.
Designed to be 100-1000x faster than the Backtrader-based engine.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Tuple, List
from datetime import datetime, timedelta
import logging

try:
    import vectorbtpro as vbt
    VBT_PRO = True
except ImportError:
    import vectorbt as vbt
    VBT_PRO = False


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    cash: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0
    size_type: str = 'percent'  # 'percent' or 'fixed'
    size: float = 0.05  # 5% position size
    allow_short: bool = True
    freq: str = 'D'  # Data frequency


@dataclass 
class BacktestResult:
    """Results from a backtest run"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    annual_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Detailed data
    equity_curve: pd.Series = field(default=None)
    trades: pd.DataFrame = field(default=None)
    returns: pd.Series = field(default=None)
    positions: pd.Series = field(default=None)
    
    # Parameters used
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    start_date: datetime = None
    end_date: datetime = None
    processing_time: float = 0.0


class VectorBTEngine:
    """
    High-performance vectorized backtesting engine using VectorBT.
    
    This engine takes signal arrays (entries/exits) and runs backtests
    much faster than event-driven systems like Backtrader.
    
    Example usage:
        engine = VectorBTEngine(config=BacktestConfig(cash=100000))
        
        # Generate signals from your strategy
        entries = close > close.rolling(20).mean()
        exits = close < close.rolling(20).mean()
        
        result = engine.run(ohlcv_data, entries, exits)
        print(f"Sharpe: {result.sharpe_ratio}")
    """
    
    def __init__(self, config: BacktestConfig = None, logger: logging.Logger = None):
        self.config = config or BacktestConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._portfolio = None
        
    def run(
        self,
        data: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        short_entries: pd.Series = None,
        short_exits: pd.Series = None,
        parameters: Dict[str, Any] = None
    ) -> BacktestResult:
        """
        Run a vectorized backtest with the given signals.
        
        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            entries: Boolean Series indicating long entry signals
            exits: Boolean Series indicating long exit signals  
            short_entries: Optional boolean Series for short entry signals
            short_exits: Optional boolean Series for short exit signals
            parameters: Strategy parameters used (for record keeping)
            
        Returns:
            BacktestResult with performance metrics and detailed data
        """
        start_time = datetime.utcnow()
        
        # Ensure data has proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        
        # Normalize column names
        data.columns = [c.lower() for c in data.columns]
        close = data['close']
        
        # Align signals with data
        entries = entries.reindex(data.index).fillna(False).astype(bool)
        exits = exits.reindex(data.index).fillna(False).astype(bool)
        
        # Build portfolio
        if self.config.allow_short and short_entries is not None:
            short_entries = short_entries.reindex(data.index).fillna(False).astype(bool)
            short_exits = short_exits.reindex(data.index).fillna(False).astype(bool)
            
            # Create combined portfolio with longs and shorts
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=self.config.cash,
                fees=self.config.commission,
                slippage=self.config.slippage,
                size=self.config.size,
                size_type='percent' if self.config.size_type == 'percent' else 'amount',
                freq=self.config.freq,
            )
        else:
            # Long-only portfolio
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=self.config.cash,
                fees=self.config.commission,
                slippage=self.config.slippage,
                size=self.config.size,
                size_type='percent' if self.config.size_type == 'percent' else 'amount',
                freq=self.config.freq,
            )
        
        self._portfolio = pf
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return self._extract_results(pf, parameters or {}, processing_time)
    
    def run_multi(
        self,
        data: pd.DataFrame,
        entries_matrix: pd.DataFrame,
        exits_matrix: pd.DataFrame,
        param_combinations: List[Dict[str, Any]],
    ) -> List[BacktestResult]:
        """
        Run multiple backtests in parallel with different parameter combinations.
        
        VectorBT can run thousands of backtests simultaneously by broadcasting
        signals across parameter dimensions.
        
        Args:
            data: OHLCV DataFrame
            entries_matrix: DataFrame where each column is entry signals for one param combo
            exits_matrix: DataFrame where each column is exit signals for one param combo
            param_combinations: List of parameter dicts, one per column
            
        Returns:
            List of BacktestResult objects
        """
        start_time = datetime.utcnow()
        
        data.columns = [c.lower() for c in data.columns]
        close = data['close']
        
        # Run all combinations at once
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_matrix,
            exits=exits_matrix,
            init_cash=self.config.cash,
            fees=self.config.commission,
            slippage=self.config.slippage,
            size=self.config.size,
            size_type='percent' if self.config.size_type == 'percent' else 'amount',
            freq=self.config.freq,
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extract results for each parameter combination
        results = []
        for i, params in enumerate(param_combinations):
            if hasattr(pf, 'iloc'):
                pf_i = pf.iloc[i] if len(param_combinations) > 1 else pf
            else:
                pf_i = pf
            results.append(self._extract_results(pf_i, params, processing_time / len(param_combinations)))
            
        return results
    
    def _extract_results(
        self,
        pf,
        parameters: Dict[str, Any],
        processing_time: float
    ) -> BacktestResult:
        """Extract BacktestResult from a VectorBT Portfolio object"""
        
        # Get stats safely with fallbacks
        def safe_get(fn, default=0.0):
            try:
                val = fn()
                if pd.isna(val) or np.isinf(val):
                    return default
                return float(val)
            except Exception:
                return default
        
        total_return = safe_get(lambda: pf.total_return())
        sharpe = safe_get(lambda: pf.sharpe_ratio())
        max_dd = safe_get(lambda: pf.max_drawdown())
        
        # Trade statistics
        try:
            trades = pf.trades.records_readable if hasattr(pf.trades, 'records_readable') else None
            num_trades = len(trades) if trades is not None and len(trades) > 0 else 0
            if num_trades > 0:
                win_rate = safe_get(lambda: pf.trades.win_rate())
                profit_factor = safe_get(lambda: pf.trades.profit_factor(), 1.0)
            else:
                win_rate = 0.0
                profit_factor = 1.0
        except Exception:
            trades = None
            num_trades = 0
            win_rate = 0.0
            profit_factor = 1.0
            
        # Additional metrics
        annual_return = safe_get(lambda: pf.annualized_return())
        volatility = safe_get(lambda: pf.annualized_volatility())
        sortino = safe_get(lambda: pf.sortino_ratio())
        calmar = safe_get(lambda: pf.calmar_ratio())
        
        # Time series data
        try:
            equity_curve = pf.value()
            returns = pf.returns()
            positions = pf.positions.records_readable if hasattr(pf.positions, 'records_readable') else None
        except Exception:
            equity_curve = None
            returns = None
            positions = None
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            annual_return=annual_return,
            volatility=volatility,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            equity_curve=equity_curve,
            trades=trades,
            returns=returns,
            positions=positions,
            parameters=parameters,
            start_date=equity_curve.index[0] if equity_curve is not None and len(equity_curve) > 0 else None,
            end_date=equity_curve.index[-1] if equity_curve is not None and len(equity_curve) > 0 else None,
            processing_time=processing_time,
        )
    
    @property
    def portfolio(self):
        """Access the underlying VectorBT Portfolio object for advanced analysis"""
        return self._portfolio
    
    @staticmethod
    def create_indicator_signals(
        data: pd.DataFrame,
        indicator_fn: Callable[[pd.DataFrame, Dict], Tuple[pd.Series, pd.Series]],
        params: Dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Helper to create entry/exit signals from an indicator function.
        
        Args:
            data: OHLCV DataFrame
            indicator_fn: Function that takes (data, params) and returns (entries, exits)
            params: Parameters to pass to the indicator function
            
        Returns:
            Tuple of (entries, exits) boolean Series
        """
        return indicator_fn(data, params)


# Convenience functions for common signal patterns
def ema_crossover_signals(data: pd.DataFrame, fast_period: int, slow_period: int) -> Tuple[pd.Series, pd.Series]:
    """Generate EMA crossover signals"""
    close = data['close'] if 'close' in data.columns else data['Close']
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    
    return entries, exits


def rsi_signals(data: pd.DataFrame, period: int, oversold: int = 30, overbought: int = 70) -> Tuple[pd.Series, pd.Series]:
    """Generate RSI overbought/oversold signals"""
    close = data['close'] if 'close' in data.columns else data['Close']
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)
    
    return entries, exits


def macd_signals(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Generate MACD crossover signals"""
    close = data['close'] if 'close' in data.columns else data['Close']
    
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    
    return entries, exits
