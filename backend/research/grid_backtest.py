"""
Grid Backtest Runner - Test strategies across asset/timeframe combinations

Usage:
    grid = GridBacktester(strategies=['BollingerBreakout', 'MACD'])
    grid.add_assets(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
    grid.add_timeframes(['1h', '4h', '1d'])
    results = grid.run()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False
    print("Warning: vectorbt not installed. Install with: pip install vectorbt")

# Import Numba-accelerated backtest
try:
    import sys
    sys.path.insert(0, '..')
    from backtest_utils import simple_backtest_core, warmup_jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@dataclass
class BacktestResult:
    """Single backtest result."""
    strategy: str
    asset: str
    timeframe: str
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade stats
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    
    # Meta
    start_date: str = ""
    end_date: str = ""
    bars: int = 0
    final_value: float = 10000.0
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GridConfig:
    """Configuration for grid backtest."""
    strategies: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    
    # Backtest params
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    
    # Data params
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_source: str = "ccxt"  # ccxt, csv, rethinkdb
    exchange: str = "binance"
    
    # Execution
    parallel: bool = True
    max_workers: int = 4
    
    def total_combinations(self) -> int:
        return len(self.strategies) * len(self.assets) * len(self.timeframes)


class GridBacktester:
    """
    Run backtests across a grid of strategies × assets × timeframes.
    """
    
    # Built-in strategy implementations
    STRATEGIES = {
        'BollingerBreakout': '_strategy_bollinger_breakout',
        'BollingerBands': '_strategy_bollinger_breakout',
        'VolumeBreakout': '_strategy_volume_breakout',
        'MACD': '_strategy_macd',
        'MACDMomentum': '_strategy_macd',
        'RSI': '_strategy_rsi',
        'OBV': '_strategy_obv',
        'OBVStrategy': '_strategy_obv',
        'EMA_Cross': '_strategy_ema_cross',
        'SMA_Cross': '_strategy_sma_cross',
        'Momentum': '_strategy_momentum',
        'MeanReversion': '_strategy_mean_reversion',
        'CapitulationReversal': '_strategy_capitulation_reversal',
    }
    
    def __init__(self, config: Optional[GridConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or GridConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.results: List[BacktestResult] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
    def add_strategies(self, strategies: List[str]) -> 'GridBacktester':
        """Add strategies to test."""
        self.config.strategies.extend(strategies)
        return self
    
    def add_assets(self, assets: List[str]) -> 'GridBacktester':
        """Add assets to test."""
        self.config.assets.extend(assets)
        return self
    
    def add_timeframes(self, timeframes: List[str]) -> 'GridBacktester':
        """Add timeframes to test."""
        self.config.timeframes.extend(timeframes)
        return self
    
    def run(self, progress_callback=None) -> List[BacktestResult]:
        """
        Run all backtest combinations.
        
        Returns:
            List of BacktestResult objects
        """
        combinations = list(itertools.product(
            self.config.strategies,
            self.config.assets,
            self.config.timeframes
        ))
        
        total = len(combinations)
        self.logger.info(f"Running {total} backtest combinations")
        
        self.results = []
        
        if self.config.parallel and total > 1:
            self._run_parallel(combinations, progress_callback)
        else:
            self._run_sequential(combinations, progress_callback)
        
        return self.results
    
    def _run_sequential(self, combinations, progress_callback=None):
        """Run backtests sequentially."""
        for i, (strategy, asset, timeframe) in enumerate(combinations):
            result = self._run_single_backtest(strategy, asset, timeframe)
            self.results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(combinations), result)
    
    def _run_parallel(self, combinations, progress_callback=None):
        """Run backtests in parallel."""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._run_single_backtest, s, a, t): (s, a, t)
                for s, a, t in combinations
            }
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                self.results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(combinations), result)
    
    def _run_single_backtest(self, strategy: str, asset: str, timeframe: str) -> BacktestResult:
        """Run a single backtest."""
        result = BacktestResult(strategy=strategy, asset=asset, timeframe=timeframe)
        
        try:
            # Fetch data
            df = self._fetch_data(asset, timeframe)
            if df is None or df.empty:
                result.success = False
                result.error = f"No data for {asset} {timeframe}"
                return result
            
            result.bars = len(df)
            result.start_date = str(df.index[0])
            result.end_date = str(df.index[-1])
            
            # Get strategy method
            strategy_method = self.STRATEGIES.get(strategy)
            if not strategy_method:
                result.success = False
                result.error = f"Unknown strategy: {strategy}"
                return result
            
            # Run strategy
            method = getattr(self, strategy_method)
            signals = method(df)
            
            if signals is None or len(signals) == 0:
                result.success = False
                result.error = "No signals generated"
                return result
            
            # Run backtest with VectorBT
            if HAS_VECTORBT:
                result = self._vectorbt_backtest(df, signals, result)
            else:
                result = self._simple_backtest(df, signals, result)
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            self.logger.error(f"Backtest failed for {strategy}/{asset}/{timeframe}: {e}")
        
        return result
    
    def _fetch_data(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data."""
        cache_key = f"{asset}_{timeframe}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        try:
            if self.config.data_source == "ccxt":
                df = self._fetch_ccxt(asset, timeframe)
            elif self.config.data_source == "csv":
                df = self._fetch_csv(asset, timeframe)
            else:
                df = self._fetch_ccxt(asset, timeframe)
            
            if df is not None:
                self._data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {asset} {timeframe}: {e}")
            return None
    
    def _fetch_ccxt(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data via CCXT."""
        try:
            import ccxt
            
            exchange = getattr(ccxt, self.config.exchange)()
            exchange.load_markets()
            
            # Fetch last 1000 candles
            ohlcv = exchange.fetch_ohlcv(asset, timeframe, limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"CCXT fetch failed: {e}")
            return None
    
    def _fetch_csv(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from CSV file."""
        # Look for CSV in standard locations
        symbol = asset.replace('/', '_').replace(':', '_')
        paths = [
            Path(f"data/{symbol}_{timeframe}.csv"),
            Path(f"~/Desktop/_Personal/maestro/data/{symbol}_{timeframe}.csv").expanduser(),
        ]
        
        for path in paths:
            if path.exists():
                df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
                return df
        
        return None
    
    def _vectorbt_backtest(self, df: pd.DataFrame, signals: pd.Series, result: BacktestResult) -> BacktestResult:
        """Run backtest using VectorBT."""
        # Convert signals to entries/exits
        entries = signals == 1
        exits = signals == -1
        
        # Run portfolio simulation
        pf = vbt.Portfolio.from_signals(
            df['close'],
            entries=entries,
            exits=exits,
            init_cash=self.config.initial_capital,
            fees=self.config.commission,
            slippage=self.config.slippage,
        )
        
        # Extract metrics
        result.total_return = float(pf.total_return() * 100)
        result.sharpe_ratio = float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0.0
        result.sortino_ratio = float(pf.sortino_ratio()) if not np.isnan(pf.sortino_ratio()) else 0.0
        result.max_drawdown = float(pf.max_drawdown() * 100)
        result.total_trades = int(pf.trades.count())
        result.win_rate = float(pf.trades.win_rate() * 100) if pf.trades.count() > 0 else 0.0
        result.final_value = float(pf.final_value())
        
        return result
    
    def _simple_backtest(self, df: pd.DataFrame, signals: pd.Series, result: BacktestResult) -> BacktestResult:
        """Simple backtest - uses Numba JIT if available (100x faster)."""
        if HAS_NUMBA:
            # Use Numba-accelerated backtest (100x faster)
            import numpy as np
            close = df['close'].values.astype(np.float64)
            sig = signals.values.astype(np.float64)

            equity, total_return, num_trades, num_wins = simple_backtest_core(
                close, sig, self.config.commission
            )

            result.final_value = self.config.initial_capital * (1 + total_return)
            result.total_return = total_return * 100
            result.total_trades = num_trades
            result.win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
            return result

        # Fallback to Python loop (slow)
        capital = self.config.initial_capital
        position = 0
        trades = []
        entry_price = 0

        for i in range(len(df)):
            price = df['close'].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0

            if signal == 1 and position == 0:  # Buy
                position = capital / price
                entry_price = price
                capital = 0
            elif signal == -1 and position > 0:  # Sell
                capital = position * price * (1 - self.config.commission)
                pnl = (price - entry_price) / entry_price
                trades.append(pnl)
                position = 0

        # Close any open position
        if position > 0:
            capital = position * df['close'].iloc[-1]

        result.final_value = capital
        result.total_return = (capital / self.config.initial_capital - 1) * 100
        result.total_trades = len(trades)
        result.win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0

        return result
    
    # ===== STRATEGY IMPLEMENTATIONS =====
    
    def _strategy_bollinger_breakout(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Bollinger Bands breakout strategy."""
        close = df['close']
        sma = close.rolling(period).mean()
        std_dev = close.rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        
        signals = pd.Series(0, index=df.index)
        signals[close > upper] = 1   # Breakout above = buy
        signals[close < lower] = -1  # Breakout below = sell
        
        return signals
    
    def _strategy_volume_breakout(self, df: pd.DataFrame, vol_mult: float = 2.0, period: int = 20) -> pd.Series:
        """Volume spike breakout strategy."""
        close = df['close']
        volume = df['volume']
        
        vol_sma = volume.rolling(period).mean()
        price_change = close.pct_change()
        
        signals = pd.Series(0, index=df.index)
        high_vol = volume > vol_sma * vol_mult
        
        signals[(high_vol) & (price_change > 0)] = 1   # High volume up = buy
        signals[(high_vol) & (price_change < 0)] = -1  # High volume down = sell
        
        return signals
    
    def _strategy_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD crossover strategy."""
        close = df['close']
        
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[macd > macd_signal] = 1
        signals[macd < macd_signal] = -1
        
        # Only signal on crossovers
        signals = signals.diff().fillna(0)
        signals[signals > 0] = 1
        signals[signals < 0] = -1
        
        return signals
    
    def _strategy_rsi(self, df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
        """RSI mean reversion strategy."""
        close = df['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        signals[rsi < oversold] = 1   # Oversold = buy
        signals[rsi > overbought] = -1  # Overbought = sell
        
        return signals
    
    def _strategy_obv(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """On-Balance Volume strategy."""
        close = df['close']
        volume = df['volume']
        
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_sma = obv.rolling(period).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[obv > obv_sma] = 1
        signals[obv < obv_sma] = -1
        
        return signals
    
    def _strategy_ema_cross(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """EMA crossover strategy."""
        close = df['close']
        
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[ema_fast > ema_slow] = 1
        signals[ema_fast < ema_slow] = -1
        
        return signals
    
    def _strategy_sma_cross(self, df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
        """SMA crossover strategy."""
        close = df['close']
        
        sma_fast = close.rolling(fast).mean()
        sma_slow = close.rolling(slow).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[sma_fast > sma_slow] = 1
        signals[sma_fast < sma_slow] = -1
        
        return signals
    
    def _strategy_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Momentum strategy."""
        close = df['close']
        momentum = close / close.shift(period) - 1
        
        signals = pd.Series(0, index=df.index)
        signals[momentum > 0] = 1
        signals[momentum < 0] = -1
        
        return signals
    
    def _strategy_mean_reversion(self, df: pd.DataFrame, period: int = 20, threshold: float = 2.0) -> pd.Series:
        """Mean reversion z-score strategy."""
        close = df['close']
        
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        zscore = (close - sma) / std
        
        signals = pd.Series(0, index=df.index)
        signals[zscore < -threshold] = 1   # Below mean = buy
        signals[zscore > threshold] = -1   # Above mean = sell
        
        return signals
    
    def _strategy_capitulation_reversal(self, df: pd.DataFrame, vol_mult: float = 4.0, period: int = 20) -> pd.Series:
        """
        Capitulation reversal - buy after extreme volume + price drop.
        Based on liquidation cascade patterns.
        """
        close = df['close']
        volume = df['volume']
        
        vol_sma = volume.rolling(period).mean()
        price_change = close.pct_change()
        
        signals = pd.Series(0, index=df.index)
        
        # Capitulation: extreme volume + sharp drop
        capitulation = (volume > vol_sma * vol_mult) & (price_change < -0.03)
        
        # Buy signal on next bar after capitulation
        signals[capitulation.shift(1) == True] = 1
        
        # Exit after 3-5 bars or on reversal
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                # Set exit 3 bars later
                exit_idx = min(i + 3, len(signals) - 1)
                signals.iloc[exit_idx] = -1
        
        return signals
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export results to JSON."""
        data = {
            'config': asdict(self.config),
            'results': [r.to_dict() for r in self.results],
            'generated_at': datetime.now().isoformat(),
        }
        
        json_str = json.dumps(data, indent=2, default=str)
        
        if path:
            Path(path).write_text(json_str)
        
        return json_str
    
    def summary(self) -> str:
        """Generate summary table."""
        if not self.results:
            return "No results"
        
        df = self.to_dataframe()
        
        # Pivot table: strategy × asset/timeframe
        summary_lines = [
            f"\n{'='*80}",
            f"GRID BACKTEST RESULTS ({len(self.results)} combinations)",
            f"{'='*80}\n",
        ]
        
        for strategy in self.config.strategies:
            strategy_results = df[df['strategy'] == strategy]
            if strategy_results.empty:
                continue
            
            summary_lines.append(f"\n📊 {strategy}")
            summary_lines.append("-" * 60)
            summary_lines.append(f"{'Asset/TF':<20} {'Return %':>12} {'Sharpe':>10} {'MaxDD %':>10} {'Trades':>8}")
            summary_lines.append("-" * 60)
            
            for _, row in strategy_results.iterrows():
                label = f"{row['asset']}/{row['timeframe']}"
                ret = f"{row['total_return']:+.2f}%" if row['success'] else "ERROR"
                sharpe = f"{row['sharpe_ratio']:.4f}" if row['success'] else "-"
                maxdd = f"{row['max_drawdown']:.2f}%" if row['success'] else "-"
                trades = str(row['total_trades']) if row['success'] else "-"
                
                summary_lines.append(f"{label:<20} {ret:>12} {sharpe:>10} {maxdd:>10} {trades:>8}")
        
        return "\n".join(summary_lines)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid Backtest Runner")
    parser.add_argument('--strategies', nargs='+', default=['BollingerBreakout', 'MACD', 'RSI'])
    parser.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    parser.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    parser.add_argument('--output', default='grid_results.json')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    grid = GridBacktester()
    grid.add_strategies(args.strategies)
    grid.add_assets(args.assets)
    grid.add_timeframes(args.timeframes)
    
    print(f"Running {grid.config.total_combinations()} backtests...")
    
    def progress(current, total, result):
        status = "✅" if result.success else "❌"
        print(f"[{current}/{total}] {status} {result.strategy}/{result.asset}/{result.timeframe}")
    
    results = grid.run(progress_callback=progress)
    
    print(grid.summary())
    grid.to_json(args.output)
    print(f"\nResults saved to {args.output}")
