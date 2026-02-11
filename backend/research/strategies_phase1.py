"""
Phase 1 Strategies - Funding Rate, Volatility Regime, Session Momentum

These strategies extend the GridBacktester with new alpha sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple
import ccxt


class FundingRateStrategy:
    """
    Funding Rate Arbitrage Strategy
    
    Edge: Extreme funding rates indicate crowded positioning.
    - Negative funding = overleveraged shorts = long opportunity
    - High positive funding = overleveraged longs = short opportunity
    """
    
    def __init__(self, exchange: str = 'binance'):
        self.exchange_name = exchange
        self._exchange = None
        self._funding_cache: Dict[str, pd.DataFrame] = {}
    
    @property
    def exchange(self):
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        return self._exchange
    
    def fetch_funding_history(self, symbol: str, since: Optional[int] = None, limit: int = 500) -> pd.DataFrame:
        """Fetch historical funding rates."""
        try:
            # Binance futures funding rate history
            if self.exchange_name == 'binance':
                rates = self.exchange.fapiPublicGetFundingRate({
                    'symbol': symbol.replace('/', ''),
                    'limit': limit
                })
                df = pd.DataFrame(rates)
                df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['funding_rate'] = df['fundingRate'].astype(float)
                df.set_index('timestamp', inplace=True)
                return df[['funding_rate']]
            else:
                # Generic CCXT funding rate
                rates = self.exchange.fetch_funding_rate_history(symbol, since, limit)
                df = pd.DataFrame(rates)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        except Exception as e:
            print(f"Error fetching funding rates: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, df: pd.DataFrame, funding_df: Optional[pd.DataFrame] = None,
                        short_threshold: float = -0.0001,  # -0.01% (8h) = -0.03%/day
                        long_threshold: float = 0.0005,    # 0.05% (8h) = 0.15%/day
                        exit_threshold: float = 0.0001) -> pd.Series:
        """
        Generate trading signals based on funding rate extremes.
        
        Args:
            df: OHLCV DataFrame
            funding_df: Funding rate DataFrame (if None, uses synthetic proxy)
            short_threshold: Go long when funding below this (shorts paying)
            long_threshold: Go short when funding above this (longs paying)
            exit_threshold: Exit when funding normalizes to this range
        """
        signals = pd.Series(0, index=df.index)
        
        if funding_df is not None and not funding_df.empty:
            # Merge funding rates with price data
            merged = df.join(funding_df, how='left')
            merged['funding_rate'] = merged['funding_rate'].ffill()
            
            # Generate signals
            signals[merged['funding_rate'] < short_threshold] = 1   # Negative funding = LONG
            signals[merged['funding_rate'] > long_threshold] = -1   # High positive = SHORT
        else:
            # Synthetic funding proxy using price momentum + volume
            # When price pumps hard with high volume = likely high funding
            returns = df['close'].pct_change(8)  # 8-bar momentum
            vol_ratio = df['volume'] / df['volume'].rolling(24).mean()
            
            funding_proxy = returns * vol_ratio * 0.1  # Scale to funding-like values
            funding_proxy = funding_proxy.rolling(3).mean()  # Smooth
            
            signals[funding_proxy < short_threshold * 10] = 1   # Adjusted thresholds
            signals[funding_proxy > long_threshold * 10] = -1
        
        return signals


class VolatilityRegimeFilter:
    """
    Volatility Regime Filter
    
    Edge: Different strategies work in different volatility environments.
    - Low vol: Mean reversion, range trading
    - High vol: Breakout, momentum, capitulation
    - Normal vol: All strategies with standard sizing
    
    This is primarily a META-strategy that filters other signals.
    """
    
    def __init__(self, lookback: int = 90, atr_period: int = 14):
        self.lookback = lookback
        self.atr_period = atr_period
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        return atr
    
    def calculate_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility regime.
        
        Returns:
            Series with values: 'low', 'normal', 'high'
        """
        atr = self.calculate_atr(df)
        atr_pct = atr / df['close'] * 100  # ATR as % of price
        
        # Rolling percentile
        rolling_percentile = atr_pct.rolling(self.lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        regime = pd.Series('normal', index=df.index)
        regime[rolling_percentile < 0.25] = 'low'
        regime[rolling_percentile > 0.75] = 'high'
        
        return regime
    
    def generate_signals(self, df: pd.DataFrame, 
                        strategy_type: str = 'breakout') -> pd.Series:
        """
        Generate signals based on volatility regime.
        
        For breakout strategies: Only trade in high vol
        For mean reversion: Only trade in low vol
        For momentum: Trade in normal-high vol
        
        Args:
            df: OHLCV DataFrame
            strategy_type: 'breakout', 'mean_reversion', 'momentum', 'all'
        """
        regime = self.calculate_regime(df)
        atr = self.calculate_atr(df)
        atr_pct = atr / df['close'] * 100
        
        signals = pd.Series(0, index=df.index)
        
        if strategy_type == 'breakout':
            # Breakout in high volatility
            close = df['close']
            high_20 = close.rolling(20).max()
            low_20 = close.rolling(20).min()
            
            # Only signal in high vol regime
            high_vol = regime == 'high'
            signals[(close >= high_20) & high_vol] = 1
            signals[(close <= low_20) & high_vol] = -1
            
        elif strategy_type == 'mean_reversion':
            # Mean reversion in low volatility
            close = df['close']
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            z_score = (close - sma) / std
            
            low_vol = regime == 'low'
            signals[(z_score < -2) & low_vol] = 1   # Oversold in calm market
            signals[(z_score > 2) & low_vol] = -1   # Overbought in calm market
            
        elif strategy_type == 'momentum':
            # Momentum in normal-high volatility
            close = df['close']
            mom = close.pct_change(10)
            
            active_vol = regime.isin(['normal', 'high'])
            signals[(mom > 0.05) & active_vol] = 1   # Strong up momentum
            signals[(mom < -0.05) & active_vol] = -1  # Strong down momentum
            
        else:  # 'all' - pure regime indicator
            # Output regime as signal for filtering
            signals[regime == 'low'] = -1    # Defensive
            signals[regime == 'normal'] = 0  # Neutral
            signals[regime == 'high'] = 1    # Aggressive
        
        return signals
    
    def filter_signals(self, signals: pd.Series, df: pd.DataFrame,
                      strategy_type: str = 'breakout') -> pd.Series:
        """
        Filter existing signals based on volatility regime.
        
        Use this to wrap other strategies.
        """
        regime = self.calculate_regime(df)
        filtered = signals.copy()
        
        if strategy_type == 'breakout':
            # Only allow breakout signals in high vol
            filtered[regime != 'high'] = 0
        elif strategy_type == 'mean_reversion':
            # Only allow mean reversion in low vol
            filtered[regime != 'low'] = 0
        elif strategy_type == 'momentum':
            # Allow momentum in normal-high vol
            filtered[regime == 'low'] = 0
        
        return filtered


class SessionMomentumStrategy:
    """
    Session Momentum Strategy
    
    Edge: Different trading sessions have distinct characteristics.
    - Asian (00:00-08:00 UTC): Range-bound, accumulation
    - London (08:00-12:00 UTC): Breakout of Asian range
    - NY (12:00-21:00 UTC): Continuation or reversal
    
    Primary setup: Trade breakout of Asian session range during London/NY.
    """
    
    # Session definitions (UTC hours)
    SESSIONS = {
        'asian': (0, 8),
        'london': (8, 12),
        'ny': (12, 21),
        'off_hours': (21, 24),
    }
    
    def __init__(self):
        pass
    
    def get_session(self, timestamp: pd.Timestamp) -> str:
        """Determine which session a timestamp belongs to."""
        hour = timestamp.hour
        for session, (start, end) in self.SESSIONS.items():
            if start <= hour < end:
                return session
        return 'off_hours'
    
    def calculate_asian_range(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate the Asian session high/low for each day."""
        df = df.copy()
        
        # Ensure timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Mark Asian session
        df['session'] = df.index.map(self.get_session)
        df['date'] = df.index.date
        
        # Calculate Asian range per day
        asian_df = df[df['session'] == 'asian']
        daily_asian_high = asian_df.groupby('date')['high'].max()
        daily_asian_low = asian_df.groupby('date')['low'].min()
        
        # Map back to full dataframe
        df['asian_high'] = df['date'].map(daily_asian_high)
        df['asian_low'] = df['date'].map(daily_asian_low)
        
        # Forward fill for non-Asian hours
        df['asian_high'] = df['asian_high'].ffill()
        df['asian_low'] = df['asian_low'].ffill()
        
        return df['asian_high'], df['asian_low']
    
    def generate_signals(self, df: pd.DataFrame,
                        breakout_buffer: float = 0.001,  # 0.1% buffer
                        trade_sessions: list = None) -> pd.Series:
        """
        Generate signals based on Asian range breakout.
        
        Args:
            df: OHLCV DataFrame with UTC timestamps
            breakout_buffer: % buffer above/below range for confirmation
            trade_sessions: Sessions to trade in (default: ['london', 'ny'])
        """
        if trade_sessions is None:
            trade_sessions = ['london', 'ny']
        
        signals = pd.Series(0, index=df.index)
        df = df.copy()
        
        # Handle timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Get Asian range
        try:
            asian_high, asian_low = self.calculate_asian_range(df)
        except Exception as e:
            print(f"Error calculating Asian range: {e}")
            # Fallback: use rolling 8-bar high/low
            asian_high = df['high'].rolling(8).max().shift(1)
            asian_low = df['low'].rolling(8).min().shift(1)
        
        # Determine current session
        df['session'] = df.index.map(self.get_session)
        
        # Calculate breakout levels with buffer
        upper_break = asian_high * (1 + breakout_buffer)
        lower_break = asian_low * (1 - breakout_buffer)
        
        # Only trade during specified sessions
        in_session = df['session'].isin(trade_sessions)
        
        # Breakout signals
        signals[(df['close'] > upper_break) & in_session] = 1   # Bullish breakout
        signals[(df['close'] < lower_break) & in_session] = -1  # Bearish breakout
        
        return signals
    
    def generate_signals_simple(self, df: pd.DataFrame, lookback: int = 8) -> pd.Series:
        """
        Simplified version without timezone requirements.
        Uses rolling range instead of exact session times.
        
        Good for backtesting on daily/4h data where session times don't align.
        """
        signals = pd.Series(0, index=df.index)
        
        # Rolling "Asian range" proxy
        rolling_high = df['high'].rolling(lookback).max().shift(1)
        rolling_low = df['low'].rolling(lookback).min().shift(1)
        range_size = rolling_high - rolling_low
        
        # Breakout with momentum confirmation
        close = df['close']
        volume = df['volume']
        vol_avg = volume.rolling(lookback).mean()
        
        # Bullish breakout: price > range high + volume confirmation
        bull_break = (close > rolling_high) & (volume > vol_avg * 1.2)
        # Bearish breakout: price < range low + volume confirmation  
        bear_break = (close < rolling_low) & (volume > vol_avg * 1.2)
        
        signals[bull_break] = 1
        signals[bear_break] = -1
        
        return signals


# ============================================================================
# Integration functions for GridBacktester
# ============================================================================

def add_phase1_strategies(grid_backtest_class):
    """
    Add Phase 1 strategies to an existing GridBacktester class.
    
    Usage:
        from strategies_phase1 import add_phase1_strategies
        add_phase1_strategies(GridBacktester)
    """
    
    # Add strategy methods
    def _strategy_funding_rate(self, df: pd.DataFrame, 
                               short_threshold: float = -0.001,
                               long_threshold: float = 0.005) -> pd.Series:
        """Funding rate arbitrage strategy."""
        strategy = FundingRateStrategy()
        return strategy.generate_signals(df, None, short_threshold, long_threshold)
    
    def _strategy_volatility_regime(self, df: pd.DataFrame,
                                    strategy_type: str = 'breakout') -> pd.Series:
        """Volatility regime-based strategy."""
        filter_obj = VolatilityRegimeFilter()
        return filter_obj.generate_signals(df, strategy_type)
    
    def _strategy_session_momentum(self, df: pd.DataFrame,
                                   lookback: int = 8) -> pd.Series:
        """Session momentum / range breakout strategy."""
        strategy = SessionMomentumStrategy()
        return strategy.generate_signals_simple(df, lookback)
    
    # Register strategies
    grid_backtest_class._strategy_funding_rate = _strategy_funding_rate
    grid_backtest_class._strategy_volatility_regime = _strategy_volatility_regime
    grid_backtest_class._strategy_session_momentum = _strategy_session_momentum
    
    # Add to STRATEGIES dict
    grid_backtest_class.STRATEGIES['FundingRate'] = '_strategy_funding_rate'
    grid_backtest_class.STRATEGIES['FundingRateArbitrage'] = '_strategy_funding_rate'
    grid_backtest_class.STRATEGIES['VolatilityRegime'] = '_strategy_volatility_regime'
    grid_backtest_class.STRATEGIES['VolRegime'] = '_strategy_volatility_regime'
    grid_backtest_class.STRATEGIES['SessionMomentum'] = '_strategy_session_momentum'
    grid_backtest_class.STRATEGIES['AsianBreakout'] = '_strategy_session_momentum'
    
    return grid_backtest_class


# Standalone signal generators for direct use
def funding_rate_signals(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Generate funding rate signals."""
    return FundingRateStrategy().generate_signals(df, **kwargs)

def volatility_regime_signals(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Generate volatility regime signals."""
    return VolatilityRegimeFilter().generate_signals(df, **kwargs)

def session_momentum_signals(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Generate session momentum signals."""
    return SessionMomentumStrategy().generate_signals_simple(df, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Phase 1 Strategies loaded:")
    print("  - FundingRateStrategy")
    print("  - VolatilityRegimeFilter")
    print("  - SessionMomentumStrategy")
    print("\nTo integrate with GridBacktester:")
    print("  from strategies_phase1 import add_phase1_strategies")
    print("  add_phase1_strategies(GridBacktester)")
