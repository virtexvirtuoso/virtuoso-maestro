"""
Strategy Adapter - Convert Backtrader strategies to VectorBT signals

This module provides adapters to convert existing Backtrader-based strategies
to vectorized signal generators compatible with VectorBT.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Type, Optional, Callable
from dataclasses import dataclass
import logging


@dataclass
class SignalOutput:
    """Output from a vectorized strategy signal generator"""
    entries: pd.Series  # Long entry signals
    exits: pd.Series    # Long exit signals
    short_entries: Optional[pd.Series] = None  # Short entry signals
    short_exits: Optional[pd.Series] = None    # Short exit signals
    indicators: Dict[str, pd.Series] = None    # Calculated indicators


class VectorBTStrategy(ABC):
    """
    Base class for VectorBT-compatible strategies.
    
    Subclass this to create new vectorized strategies that can be used
    with the VectorBT engine and Optuna optimization.
    """
    
    @staticmethod
    @abstractmethod
    def get_params() -> Dict[str, Any]:
        """Get default parameters for this strategy."""
        raise NotImplementedError()
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        """Get the parameter search space for optimization."""
        params = VectorBTStrategy.get_params()
        space = {}
        for k, v in params.items():
            if isinstance(v, int):
                space[k] = ('int', max(1, v // 2), v * 2)
            elif isinstance(v, float):
                space[k] = ('float', v / 2, v * 2)
        return space
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        """Generate entry/exit signals from the data using given parameters."""
        raise NotImplementedError()
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill in missing parameters with defaults"""
        defaults = self.get_params()
        validated = defaults.copy()
        validated.update(params)
        return validated
    
    @staticmethod
    def _get_close(data: pd.DataFrame) -> pd.Series:
        """Helper to get close price from data"""
        return data['close'] if 'close' in data.columns else data['Close']
    
    @staticmethod
    def _get_high(data: pd.DataFrame) -> pd.Series:
        """Helper to get high price from data"""
        return data['high'] if 'high' in data.columns else data['High']
    
    @staticmethod
    def _get_low(data: pd.DataFrame) -> pd.Series:
        """Helper to get low price from data"""
        return data['low'] if 'low' in data.columns else data['Low']
    
    @staticmethod
    def _get_volume(data: pd.DataFrame) -> pd.Series:
        """Helper to get volume from data"""
        return data['volume'] if 'volume' in data.columns else data['Volume']


class StrategyAdapter:
    """Adapter to convert Backtrader strategy definitions to VectorBT signals."""
    
    _registry: Dict[str, Type[VectorBTStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[VectorBTStrategy]):
        """Register a VectorBT strategy adapter"""
        cls._registry[name] = strategy_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[VectorBTStrategy]]:
        """Get a registered strategy by name"""
        return cls._registry.get(name)
    
    @classmethod
    def list_strategies(cls) -> list:
        """List all registered strategy names"""
        return list(cls._registry.keys())


# ============================================================================
# EXISTING STRATEGIES (Already converted)
# ============================================================================

class EMACrossStrategy(VectorBTStrategy):
    """Vectorized EMA Cross Strategy"""
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'ema50': 50,
            'ema51': 51,
            'ema100': 100,
            'ema101': 101,
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'ema50': ('int', 20, 100),
            'ema51': ('int', 20, 100),
            'ema100': ('int', 50, 200),
            'ema101': ('int', 50, 200),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        ema50 = close.ewm(span=params['ema50'], adjust=False).mean()
        ema100 = close.ewm(span=params['ema100'], adjust=False).mean()
        ema51 = close.shift(1).ewm(span=params['ema51'], adjust=False).mean()
        ema101 = close.shift(1).ewm(span=params['ema101'], adjust=False).mean()
        
        even_crossover = (ema50 > ema100).astype(int).diff()
        odd_crossover = (ema101 > ema51).astype(int).diff()
        
        entries = (even_crossover > 0) & (odd_crossover > 0)
        exits = (even_crossover < 0) & (odd_crossover < 0)
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'ema50': ema50, 'ema100': ema100, 'ema51': ema51, 'ema101': ema101}
        )


class RSIStrategy(VectorBTStrategy):
    """Vectorized RSI Strategy"""
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'period': 20, 'oversold': 30, 'overbought': 70}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'period': ('int', 5, 50),
            'oversold': ('int', 20, 40),
            'overbought': ('int', 60, 80),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=params['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        entries = (rsi > params['oversold']) & (rsi.shift(1) <= params['oversold'])
        exits = (rsi > params['overbought']) & (rsi.shift(1) <= params['overbought'])
        short_entries = (rsi < params['overbought']) & (rsi.shift(1) >= params['overbought'])
        short_exits = (rsi < params['oversold']) & (rsi.shift(1) >= params['oversold'])
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'rsi': rsi}
        )


class MACDStrategy(VectorBTStrategy):
    """Vectorized MACD Strategy"""
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'fast_period': ('int', 6, 24),
            'slow_period': ('int', 18, 52),
            'signal_period': ('int', 5, 15),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        fast_ema = close.ewm(span=params['fast_period'], adjust=False).mean()
        slow_ema = close.ewm(span=params['slow_period'], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=params['signal_period'], adjust=False).mean()
        
        mcross = (macd_line > signal_line).astype(int)
        entries = (mcross > 0) & (mcross.shift(1) <= 0)
        exits = (mcross < mcross.shift(1))
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'macd': macd_line, 'signal': signal_line, 'histogram': macd_line - signal_line}
        )


class BollingerBandsStrategy(VectorBTStrategy):
    """Vectorized Bollinger Bands Strategy"""
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'period': 20, 'num_std': 2.0}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {'period': ('int', 10, 50), 'num_std': ('float', 1.5, 3.0)}
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        sma = close.rolling(window=params['period']).mean()
        std = close.rolling(window=params['period']).std()
        upper_band = sma + params['num_std'] * std
        lower_band = sma - params['num_std'] * std
        
        entries = (close < lower_band) & (close.shift(1) >= lower_band.shift(1))
        exits = (close > upper_band) & (close.shift(1) <= upper_band.shift(1))
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'sma': sma, 'upper_band': upper_band, 'lower_band': lower_band}
        )


# ============================================================================
# CLASSIC STRATEGIES (5)
# ============================================================================

class ChannelStrategy(VectorBTStrategy):
    """
    Vectorized Channel Strategy
    
    Uses highest-high channels for breakout/breakdown signals.
    Buy when price breaks below short-term channel while long-term channel is higher.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'period_long_term': 50, 'period_short_term': 20}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'period_long_term': ('int', 30, 100),
            'period_short_term': ('int', 10, 40),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        
        # Calculate channels on previous bar's high
        high_prev = high.shift(1)
        channel_long = high_prev.rolling(window=params['period_long_term']).max()
        channel_short = high_prev.rolling(window=params['period_short_term']).max()
        
        # Entry: close < short channel < long channel
        entries = (close < channel_short) & (channel_short < channel_long)
        
        # Exit: close > long channel > short channel
        exits = (close > channel_long) & (channel_long > channel_short)
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'channel_long': channel_long, 'channel_short': channel_short}
        )


class IchimokuStrategy(VectorBTStrategy):
    """
    Vectorized Ichimoku Cloud Strategy
    
    Enter long when price closes above cloud but low dips into it.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_span_b_period': 52,
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'tenkan_period': ('int', 5, 15),
            'kijun_period': ('int', 15, 40),
            'senkou_span_b_period': ('int', 30, 80),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(params['tenkan_period']).max() + 
                  low.rolling(params['tenkan_period']).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun = (high.rolling(params['kijun_period']).max() + 
                 low.rolling(params['kijun_period']).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(params['kijun_period'])
        
        # Senkou Span B (Leading Span B)
        senkou_b = ((high.rolling(params['senkou_span_b_period']).max() + 
                     low.rolling(params['senkou_span_b_period']).min()) / 2).shift(params['kijun_period'])
        
        # Entry: close above both spans AND low touches/below at least one span
        above_cloud = (close > senkou_a) & (close > senkou_b)
        low_dips = (low < senkou_a) | (low < senkou_b)
        entries = above_cloud & low_dips
        
        # Exit: opposite condition
        exits = ~(above_cloud & low_dips) & above_cloud.shift(1) & low_dips.shift(1)
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b}
        )


class MaCrossStrategy(VectorBTStrategy):
    """
    Vectorized MA Cross Strategy
    
    Uses dual SMA crossovers with confirmation from previous bar.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'sma_1': 60, 'sma_2': 120, 'prev_sma_1': 60, 'prev_sma_2': 120}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'sma_1': ('int', 20, 100),
            'sma_2': ('int', 50, 200),
            'prev_sma_1': ('int', 20, 100),
            'prev_sma_2': ('int', 50, 200),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        close_prev = close.shift(1)
        
        sma_1 = close.rolling(params['sma_1']).mean()
        sma_2 = close.rolling(params['sma_2']).mean()
        prev_sma_1 = close_prev.rolling(params['prev_sma_1']).mean()
        prev_sma_2 = close_prev.rolling(params['prev_sma_2']).mean()
        
        # Crossover signals
        sma_cross = (sma_1 > sma_2).astype(int).diff()
        prev_sma_cross = (prev_sma_2 > prev_sma_1).astype(int).diff()
        
        # Entry: both crossovers positive OR price above fast SMA
        entries = ((sma_cross > 0) & (prev_sma_cross > 0)) | (close > sma_1)
        
        # Exit: both crossovers negative
        exits = (sma_cross < 0) & (prev_sma_cross < 0)
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'sma_1': sma_1, 'sma_2': sma_2}
        )


class FernandoStrategy(VectorBTStrategy):
    """
    Vectorized Fernando Strategy (Volatility Level Index)
    
    Based on Alain Glucksmann's thesis. Uses Bollinger Band Width and
    Volatility Level Index to identify entry points.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'vli_fast': 20,
            'vli_slow': 100,
            'vol_fast': 10,
            'vol_slow': 50,
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'bb_period': ('int', 10, 40),
            'bb_std': ('float', 1.5, 3.0),
            'vli_fast': ('int', 10, 40),
            'vli_slow': ('int', 50, 150),
            'vol_fast': ('int', 5, 20),
            'vol_slow': ('int', 30, 80),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        volume = self._get_volume(data)
        low = self._get_low(data)
        
        # Bollinger Bands
        bb_mid = close.rolling(params['bb_period']).mean()
        bb_std = close.rolling(params['bb_period']).std()
        bb_top = bb_mid + params['bb_std'] * bb_std
        bb_bot = bb_mid - params['bb_std'] * bb_std
        
        # Bollinger Band Width
        bbw = (bb_top - bb_bot) / bb_mid
        
        # Volatility Level Index
        vli_fast = bbw.rolling(params['vli_fast']).mean()
        vli_slow = bbw.rolling(params['vli_slow']).mean()
        vli_std = bbw.rolling(params['vli_slow']).std()
        vli_top = vli_slow + 2 * vli_std
        
        # Volume condition
        vol_fast = volume.rolling(params['vol_fast']).mean()
        vol_slow = volume.rolling(params['vol_slow']).mean()
        vol_condition = vol_fast > vol_slow
        
        # SMAs for trend
        sma_fast = close.rolling(20).mean()
        sma_mid = close.rolling(50).mean()
        sma_slow = close.rolling(100).mean()
        sma_veryslow = close.rolling(200).mean()
        
        # Cross-down of close with upper band
        crossdown_top = (close < bb_top) & (close.shift(1) >= bb_top.shift(1))
        
        # Cross-up of close with lower band
        crossup_bot = (close > bb_bot) & (close.shift(1) <= bb_bot.shift(1))
        
        # Entry conditions (simplified from original complex logic)
        low_volatility = vli_fast < vli_slow
        trend_up = sma_mid > sma_veryslow
        
        # Long entry: crossdown top + volume + (low vol with trend OR any vol)
        entries = crossdown_top & vol_condition & (
            (low_volatility & trend_up) | ~low_volatility | (sma_slow > sma_veryslow)
        )
        
        # Exit: crossup bottom with volume
        exits = crossup_bot & vol_condition
        
        # Short entry: crossup bottom with volume
        short_entries = crossup_bot & vol_condition
        short_exits = crossdown_top & vol_condition
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={
                'bb_top': bb_top, 'bb_bot': bb_bot, 'bb_mid': bb_mid,
                'bbw': bbw, 'vli_fast': vli_fast, 'vli_slow': vli_slow
            }
        )


class MACDStrategyX(VectorBTStrategy):
    """
    Vectorized MACD Strategy Extended
    
    MACD crossover with SMA trend filter.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'macd1': 12, 'macd2': 26, 'macdsig': 9,
            'smaperiod': 30, 'dirperiod': 10,
            'trailpercent': 0.40,
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'macd1': ('int', 6, 20),
            'macd2': ('int', 18, 40),
            'macdsig': ('int', 5, 15),
            'smaperiod': ('int', 15, 50),
            'dirperiod': ('int', 5, 20),
            'trailpercent': ('float', 0.1, 0.6),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        # MACD calculation
        fast_ema = close.ewm(span=params['macd1'], adjust=False).mean()
        slow_ema = close.ewm(span=params['macd2'], adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=params['macdsig'], adjust=False).mean()
        
        # SMA direction
        sma = close.rolling(params['smaperiod']).mean()
        smadir = sma - sma.shift(params['dirperiod'])
        
        # Entry: MACD crosses above signal AND sma direction is negative (contrarian)
        mcross = (macd > signal).astype(int).diff()
        entries = (mcross > 0) & (smadir < 0)
        
        # Exit: trailing stop triggered (price drops by trailpercent from high)
        rolling_high = close.rolling(params['dirperiod']).max()
        trail_stop = rolling_high * (1 - params['trailpercent'])
        exits = close < trail_stop
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=pd.Series(False, index=data.index),
            short_exits=pd.Series(False, index=data.index),
            indicators={'macd': macd, 'signal': signal, 'sma': sma, 'smadir': smadir}
        )


# ============================================================================
# SCALPING STRATEGIES (5)
# ============================================================================

class VWAPStrategy(VectorBTStrategy):
    """
    Vectorized VWAP Mean Reversion Strategy
    
    Buy below VWAP, sell above VWAP with standard deviation bands.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'period': 20, 'dev_factor': 1.5}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'period': ('int', 10, 50),
            'dev_factor': ('float', 1.0, 3.0),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        # VWAP calculation (weighted moving average)
        vwap = (close * volume).rolling(params['period']).sum() / volume.rolling(params['period']).sum()
        std = close.rolling(params['period']).std()
        
        # Entry: price below VWAP - dev * std
        entries = close < (vwap - params['dev_factor'] * std)
        
        # Exit: price above VWAP + dev * std
        exits = close > (vwap + params['dev_factor'] * std)
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'vwap': vwap, 'upper': vwap + params['dev_factor'] * std, 
                       'lower': vwap - params['dev_factor'] * std}
        )


class ScalpRSIStrategy(VectorBTStrategy):
    """
    Vectorized Quick RSI Scalping Strategy
    
    Tight entries/exits on oversold/overbought.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'rsi_period': 7, 'oversold': 25, 'overbought': 75}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'rsi_period': ('int', 3, 14),
            'oversold': ('int', 15, 35),
            'overbought': ('int', 65, 85),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        # RSI calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        entries = rsi < params['oversold']
        exits = rsi > params['overbought']
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'rsi': rsi}
        )


class MomentumBreakoutStrategy(VectorBTStrategy):
    """
    Vectorized Momentum Breakout Strategy
    
    Breakout with volume confirmation.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'lookback': 20, 'vol_mult': 1.5}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'lookback': ('int', 10, 50),
            'vol_mult': ('float', 1.0, 3.0),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        volume = self._get_volume(data)
        
        highest = high.rolling(params['lookback']).max()
        lowest = low.rolling(params['lookback']).min()
        vol_sma = volume.rolling(params['lookback']).mean()
        
        vol_spike = volume > (vol_sma * params['vol_mult'])
        
        # Breakout above previous high with volume
        entries = (close > highest.shift(1)) & vol_spike
        
        # Breakdown below previous low with volume
        exits = (close < lowest.shift(1)) & vol_spike
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'highest': highest, 'lowest': lowest, 'vol_sma': vol_sma}
        )


class StochRSIStrategy(VectorBTStrategy):
    """
    Vectorized Stochastic RSI Strategy
    
    Popular for scalping - combines RSI with Stochastic oscillator.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'rsi_period': 14, 'stoch_period': 14, 'k_period': 3, 'd_period': 3}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'rsi_period': ('int', 7, 21),
            'stoch_period': ('int', 7, 21),
            'k_period': ('int', 2, 7),
            'd_period': ('int', 2, 7),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        # RSI calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic of RSI
        rsi_low = rsi.rolling(params['stoch_period']).min()
        rsi_high = rsi.rolling(params['stoch_period']).max()
        stoch_k = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
        stoch_d = stoch_k.rolling(params['d_period']).mean()
        
        # Entry: K below 20 and turning up
        entries = (stoch_k < 20) & (stoch_k > stoch_k.shift(1))
        
        # Exit: K above 80 and turning down
        exits = (stoch_k > 80) & (stoch_k < stoch_k.shift(1))
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'rsi': rsi, 'stoch_k': stoch_k, 'stoch_d': stoch_d}
        )


class EMARibbonStrategy(VectorBTStrategy):
    """
    Vectorized EMA Ribbon Strategy
    
    Trend strength indicator using stacked EMAs.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'fast': 8, 'medium': 13, 'slow': 21}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'fast': ('int', 3, 15),
            'medium': ('int', 8, 20),
            'slow': ('int', 15, 40),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        ema_fast = close.ewm(span=params['fast'], adjust=False).mean()
        ema_med = close.ewm(span=params['medium'], adjust=False).mean()
        ema_slow = close.ewm(span=params['slow'], adjust=False).mean()
        
        # Bullish ribbon: fast > med > slow
        bullish = (ema_fast > ema_med) & (ema_med > ema_slow)
        bullish_prev = (ema_fast.shift(1) <= ema_med.shift(1))
        
        # Bearish ribbon: fast < med < slow
        bearish = (ema_fast < ema_med) & (ema_med < ema_slow)
        bearish_prev = (ema_fast.shift(1) >= ema_med.shift(1))
        
        entries = bullish & bullish_prev  # Just crossed into bullish
        exits = bearish & bearish_prev    # Just crossed into bearish
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'ema_fast': ema_fast, 'ema_med': ema_med, 'ema_slow': ema_slow}
        )


# ============================================================================
# DERIVATIVE STRATEGIES (8)
# ============================================================================

class FundingRateArbitrage(VectorBTStrategy):
    """
    Vectorized Funding Rate Strategy
    
    Go long when funding is very negative (shorts pay longs).
    Uses momentum as proxy for funding rate.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'funding_threshold': -0.01, 'exit_threshold': 0.005, 'period': 8}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'funding_threshold': ('float', -0.05, -0.005),
            'exit_threshold': ('float', 0.001, 0.02),
            'period': ('int', 4, 24),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        # Rate of change as funding proxy
        momentum = close.pct_change(params['period'])
        
        entries = momentum < params['funding_threshold']
        exits = momentum > params['exit_threshold']
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=pd.Series(False, index=data.index),
            short_exits=pd.Series(False, index=data.index),
            indicators={'momentum': momentum}
        )


class BasisTradingStrategy(VectorBTStrategy):
    """
    Vectorized Basis/Contango Strategy
    
    Trade the spread between spot and futures using z-score.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'lookback': 20, 'entry_std': 2.0, 'exit_std': 0.5}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'lookback': ('int', 10, 50),
            'entry_std': ('float', 1.5, 3.0),
            'exit_std': ('float', 0.2, 1.0),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        sma = close.rolling(params['lookback']).mean()
        std = close.rolling(params['lookback']).std()
        zscore = (close - sma) / std
        
        # Long when at discount (negative z-score)
        entries = zscore < -params['entry_std']
        
        # Short when at premium (positive z-score)
        short_entries = zscore > params['entry_std']
        
        # Exit near mean
        exits = zscore > -params['exit_std']
        short_exits = zscore < params['exit_std']
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'zscore': zscore, 'sma': sma}
        )


class OpenInterestDivergence(VectorBTStrategy):
    """
    Vectorized OI Divergence Strategy
    
    Price up + OI/volume down = weak rally, fade it.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'price_period': 10, 'vol_period': 10, 'price_thresh': 0.02, 'vol_thresh': 0.1}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'price_period': ('int', 5, 20),
            'vol_period': ('int', 5, 20),
            'price_thresh': ('float', 0.01, 0.05),
            'vol_thresh': ('float', 0.05, 0.2),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        price_roc = close.pct_change(params['price_period'])
        vol_roc = volume.pct_change(params['vol_period'])
        
        # Bearish divergence: price up, volume down
        short_entries = (price_roc > params['price_thresh']) & (vol_roc < -params['vol_thresh'])
        
        # Bullish divergence: price down, volume up
        entries = (price_roc < -params['price_thresh']) & (vol_roc > params['vol_thresh'])
        
        exits = short_entries  # Exit long on bearish divergence
        short_exits = entries  # Exit short on bullish divergence
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'price_roc': price_roc, 'vol_roc': vol_roc}
        )


class LiquidationHuntStrategy(VectorBTStrategy):
    """
    Vectorized Liquidation Hunt Strategy
    
    After sharp moves, anticipate mean reversion as liquidations exhaust.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'atr_period': 14, 'atr_mult': 3.0, 'sma_period': 20}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'atr_period': ('int', 7, 21),
            'atr_mult': ('float', 2.0, 5.0),
            'sma_period': ('int', 10, 40),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        # ATR calculation
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(params['atr_period']).mean()
        
        # Move from previous bar
        move = close - close.shift(1)
        threshold = atr * params['atr_mult']
        
        # Sharp drop = long for bounce
        entries = move < -threshold
        
        # Sharp pump = short for pullback
        short_entries = move > threshold
        
        exits = short_entries
        short_exits = entries
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'atr': atr, 'move': move, 'threshold': threshold}
        )


class GridTradingStrategy(VectorBTStrategy):
    """
    Vectorized Grid Trading Strategy
    
    Trade at Bollinger Band extremes in ranging markets.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'period': 20, 'num_std': 2.0}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'period': ('int', 10, 40),
            'num_std': ('float', 1.5, 3.0),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        sma = close.rolling(params['period']).mean()
        std = close.rolling(params['period']).std()
        upper = sma + params['num_std'] * std
        lower = sma - params['num_std'] * std
        
        entries = close < lower
        exits = close > upper
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=exits.copy(), short_exits=entries.copy(),
            indicators={'sma': sma, 'upper': upper, 'lower': lower}
        )


class VolatilityBreakoutStrategy(VectorBTStrategy):
    """
    Vectorized Volatility Breakout Strategy
    
    Enter on expansion after compression (Keltner Channel squeeze).
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'atr_period': 20, 'atr_mult': 2.0, 'squeeze_threshold': 0.5}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'atr_period': ('int', 10, 40),
            'atr_mult': ('float', 1.0, 3.0),
            'squeeze_threshold': ('float', 0.3, 0.8),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        # ATR calculation
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(params['atr_period']).mean()
        atr_sma = atr.rolling(params['atr_period']).mean()
        
        ema = close.ewm(span=params['atr_period'], adjust=False).mean()
        
        # Squeeze: ATR below average
        squeeze = atr < (atr_sma * params['squeeze_threshold'])
        expansion = ~squeeze
        
        # Breakout in expansion
        upper_break = close > (ema + atr * params['atr_mult'])
        lower_break = close < (ema - atr * params['atr_mult'])
        
        entries = expansion & upper_break
        short_entries = expansion & lower_break
        exits = short_entries
        short_exits = entries
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'atr': atr, 'ema': ema, 'squeeze': squeeze.astype(int)}
        )


class TrendFollowingATR(VectorBTStrategy):
    """
    Vectorized ATR Trend Following Strategy
    
    Classic trend system with ATR-based channel breakouts.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'trend_period': 50, 'atr_period': 14, 'atr_mult': 2.0}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'trend_period': ('int', 20, 100),
            'atr_period': ('int', 7, 21),
            'atr_mult': ('float', 1.0, 3.0),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(params['atr_period']).mean()
        
        highest = high.rolling(params['trend_period']).max()
        lowest = low.rolling(params['trend_period']).min()
        
        # Breakout above highest
        entries = close > highest.shift(1)
        
        # Breakdown below lowest
        short_entries = close < lowest.shift(1)
        
        exits = short_entries
        short_exits = entries
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'atr': atr, 'highest': highest, 'lowest': lowest}
        )


class MeanReversionBands(VectorBTStrategy):
    """
    Vectorized Mean Reversion with Dynamic Bands
    
    Fade moves to extremes with exit near mean.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'period': 20, 'num_std': 2.5, 'exit_std': 0.5}
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'period': ('int', 10, 40),
            'num_std': ('float', 2.0, 3.5),
            'exit_std': ('float', 0.3, 1.0),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        sma = close.rolling(params['period']).mean()
        std = close.rolling(params['period']).std()
        
        # Entry bands
        upper = sma + params['num_std'] * std
        lower = sma - params['num_std'] * std
        
        # Exit bands (near mean)
        exit_upper = sma + params['exit_std'] * std
        exit_lower = sma - params['exit_std'] * std
        
        entries = close < lower
        short_entries = close > upper
        
        # Exit near mean
        exits = close > exit_upper
        short_exits = close < exit_lower
        
        return SignalOutput(
            entries=entries, exits=exits,
            short_entries=short_entries, short_exits=short_exits,
            indicators={'sma': sma, 'upper': upper, 'lower': lower}
        )


# ============================================================================
# RESEARCH PAPER STRATEGIES (3)
# ============================================================================

class CointegrationPairsStrategy(VectorBTStrategy):
    """
    Cointegration-Based Pairs Trading Strategy
    
    Based on: Tadi (2021) - Cointegration Pairs Trading
    
    Concept: Trade pairs of correlated assets when their spread diverges from
    equilibrium. For single-asset mode, we use the spread between price and its
    rolling regression against a synthetic pair (lagged self or moving average).
    
    - Calculate spread using rolling hedge ratio
    - Enter when spread z-score exceeds threshold
    - Exit when spread reverts to mean
    
    Note: For true pairs trading, use with multi-asset data. In single-asset
    mode, this behaves as a sophisticated mean-reversion strategy using the
    spread between price and its smoothed version.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'lookback': 60,           # Rolling window for spread calculation
            'entry_zscore': 2.0,      # Z-score threshold to enter
            'exit_zscore': 0.5,       # Z-score threshold to exit
            'hedge_ratio_window': 30  # Window for hedge ratio calculation
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'lookback': ('int', 30, 120),
            'entry_zscore': ('float', 1.5, 3.0),
            'exit_zscore': ('float', 0.2, 1.0),
            'hedge_ratio_window': ('int', 15, 60)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        # Create synthetic pair using smoothed price (as proxy for second asset)
        synthetic_pair = close.rolling(params['hedge_ratio_window']).mean()
        
        # Calculate rolling hedge ratio (beta) using covariance/variance
        rolling_cov = close.rolling(params['lookback']).cov(synthetic_pair)
        rolling_var = synthetic_pair.rolling(params['lookback']).var()
        hedge_ratio = rolling_cov / rolling_var
        hedge_ratio = hedge_ratio.fillna(1.0)  # Default to 1:1 ratio
        
        # Calculate spread: price - hedge_ratio * synthetic_pair
        spread = close - hedge_ratio * synthetic_pair
        
        # Calculate z-score of the spread
        spread_mean = spread.rolling(params['lookback']).mean()
        spread_std = spread.rolling(params['lookback']).std()
        zscore = (spread - spread_mean) / spread_std
        zscore = zscore.fillna(0)
        
        # Entry signals: spread diverged significantly
        # Long when spread is very negative (price too low relative to synthetic)
        entries = zscore < -params['entry_zscore']
        
        # Short when spread is very positive (price too high relative to synthetic)
        short_entries = zscore > params['entry_zscore']
        
        # Exit signals: spread reverted to mean
        exits = zscore > -params['exit_zscore']
        short_exits = zscore < params['exit_zscore']
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'spread': spread,
                'zscore': zscore,
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread_mean
            }
        )


class MACDDivergenceStrategy(VectorBTStrategy):
    """
    MACD Divergence Detection Strategy
    
    Based on: Chio (2022) - MACD Trading Strategies
    
    Concept: Detect divergence between price and MACD for stronger signals.
    Divergences indicate weakening momentum and potential reversals.
    
    - Bullish divergence: Price makes lower low, MACD makes higher low
    - Bearish divergence: Price makes higher high, MACD makes lower high
    - Much stronger signal than regular MACD crossover
    
    Uses vectorized pivot detection for efficient signal generation.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'divergence_lookback': 14,  # Bars to look for divergence
            'min_divergence_bars': 3    # Min bars between pivots
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'fast_period': ('int', 6, 20),
            'slow_period': ('int', 18, 40),
            'signal_period': ('int', 5, 15),
            'divergence_lookback': ('int', 7, 28),
            'min_divergence_bars': ('int', 2, 7)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        low = self._get_low(data)
        high = self._get_high(data)
        
        # Calculate MACD
        fast_ema = close.ewm(span=params['fast_period'], adjust=False).mean()
        slow_ema = close.ewm(span=params['slow_period'], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=params['signal_period'], adjust=False).mean()
        histogram = macd_line - signal_line
        
        lookback = params['divergence_lookback']
        min_bars = params['min_divergence_bars']
        
        # Find local minima and maxima using rolling windows
        # Local low: current low is the minimum in the window
        is_local_low = low == low.rolling(window=min_bars * 2 + 1, center=True).min()
        
        # Local high: current high is the maximum in the window
        is_local_high = high == high.rolling(window=min_bars * 2 + 1, center=True).max()
        
        # Get rolling min/max of price and MACD for divergence detection
        price_low_recent = low.rolling(lookback).min()
        price_low_prev = low.shift(min_bars).rolling(lookback).min()
        
        price_high_recent = high.rolling(lookback).max()
        price_high_prev = high.shift(min_bars).rolling(lookback).max()
        
        macd_low_recent = macd_line.rolling(lookback).min()
        macd_low_prev = macd_line.shift(min_bars).rolling(lookback).min()
        
        macd_high_recent = macd_line.rolling(lookback).max()
        macd_high_prev = macd_line.shift(min_bars).rolling(lookback).max()
        
        # Bullish divergence: price lower low + MACD higher low
        bullish_divergence = (
            (price_low_recent < price_low_prev) &  # Price made lower low
            (macd_low_recent > macd_low_prev) &    # MACD made higher low
            (macd_line > macd_line.shift(1)) &     # MACD turning up
            is_local_low.shift(1).fillna(False)    # Near a local low
        )
        
        # Bearish divergence: price higher high + MACD lower high
        bearish_divergence = (
            (price_high_recent > price_high_prev) &  # Price made higher high
            (macd_high_recent < macd_high_prev) &    # MACD made lower high
            (macd_line < macd_line.shift(1)) &       # MACD turning down
            is_local_high.shift(1).fillna(False)     # Near a local high
        )
        
        # Also include standard MACD crossover as confirmation
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # Entry: divergence OR strong crossover with histogram momentum
        entries = bullish_divergence | (macd_cross_up & (histogram > histogram.shift(1)))
        exits = bearish_divergence | (macd_cross_down & (histogram < histogram.shift(1)))
        
        short_entries = bearish_divergence
        short_exits = bullish_divergence
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram,
                'bullish_div': bullish_divergence.astype(int),
                'bearish_div': bearish_divergence.astype(int)
            }
        )


class WhaleActivityStrategy(VectorBTStrategy):
    """
    Whale Activity Detection Strategy
    
    Based on: Herremans (2022) - Bitcoin Volatility and Whale Activity
    
    Concept: React to large whale transactions that precede volatility.
    Uses volume spikes as a proxy for whale activity (in absence of on-chain data).
    
    - Detect unusual volume spikes (volume > N times average)
    - Volume spike + price rising = momentum continuation (long)
    - Volume spike + price falling = potential reversal opportunity
    - Cooldown period to avoid overtrading after signals
    
    The paper found that whale transactions often precede significant
    price movements, making volume spikes a useful leading indicator.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'volume_mult': 3.0,         # Multiple of avg volume for "whale" detection
            'volume_lookback': 20,      # Window for average volume calculation
            'price_change_thresh': 0.02, # 2% price move threshold
            'cooldown_bars': 5          # Bars to wait after signal
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'volume_mult': ('float', 2.0, 5.0),
            'volume_lookback': ('int', 10, 40),
            'price_change_thresh': ('float', 0.01, 0.05),
            'cooldown_bars': ('int', 2, 10)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        volume = self._get_volume(data)
        
        # Calculate average volume and detect spikes
        avg_volume = volume.rolling(params['volume_lookback']).mean()
        volume_spike = volume > (avg_volume * params['volume_mult'])
        
        # Calculate price change
        price_change = close.pct_change(params['cooldown_bars'])
        
        # Momentum detection: is price rising or falling?
        price_rising = price_change > params['price_change_thresh']
        price_falling = price_change < -params['price_change_thresh']
        
        # Calculate trend context using SMA
        sma_fast = close.rolling(10).mean()
        sma_slow = close.rolling(30).mean()
        uptrend = sma_fast > sma_slow
        downtrend = sma_fast < sma_slow
        
        # Whale volume spike with momentum = continuation
        # Long entry: volume spike + price rising + overall uptrend
        raw_long_entries = volume_spike & price_rising & uptrend
        
        # Counter-trend: volume spike + sharp drop = potential reversal (contrarian long)
        reversal_long = volume_spike & price_falling & (price_change < -params['price_change_thresh'] * 2)
        
        # Short entry: volume spike + price falling + overall downtrend
        raw_short_entries = volume_spike & price_falling & downtrend
        
        # Apply cooldown: suppress signals within N bars of previous signal
        # We use rolling sum to detect if there was a recent signal
        long_signal_count = raw_long_entries.astype(int).rolling(params['cooldown_bars']).sum()
        short_signal_count = raw_short_entries.astype(int).rolling(params['cooldown_bars']).sum()
        
        # Only trigger if this is the first signal in the cooldown window
        entries = (raw_long_entries | reversal_long) & (long_signal_count.shift(1).fillna(0) == 0)
        short_entries = raw_short_entries & (short_signal_count.shift(1).fillna(0) == 0)
        
        # Exit on opposite condition or after cooldown period
        exits = short_entries | (~uptrend & uptrend.shift(1))  # Exit when trend flips
        short_exits = entries | (~downtrend & downtrend.shift(1))
        
        # Volume intensity indicator (normalized)
        volume_intensity = volume / avg_volume
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'volume_intensity': volume_intensity,
                'avg_volume': avg_volume,
                'whale_spike': volume_spike.astype(int),
                'price_change': price_change,
                'uptrend': uptrend.astype(int)
            }
        )


# ============================================================================
# ADVANCED STRATEGIES (4) - Multi-Timeframe, Session, Order Flow, Kelly
# ============================================================================

class MultiTimeframeStrategy(VectorBTStrategy):
    """
    Multi-Timeframe Trend Confirmation Strategy
    
    Concept: Only take signals when higher timeframe trend aligns with
    lower timeframe signals. Dramatically reduces false signals.
    
    - Calculate trend on LTF using fast/slow SMA crossover
    - Calculate trend on HTF using longer period SMA (simulates higher timeframe)
    - Only take long signals when HTF is bullish (price > HTF SMA)
    - Only take short signals when HTF is bearish (price < HTF SMA)
    
    The HTF filter acts as a regime filter, preventing counter-trend trades
    that often result in losses. This is a core principle in professional
    trading: "trade with the trend of the higher timeframe."
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'ltf_fast': 10,           # LTF fast SMA period
            'ltf_slow': 30,           # LTF slow SMA period  
            'htf_period': 100,        # HTF trend period (simulates higher TF)
            'htf_filter': True,       # Enable HTF filter
            'require_alignment': True  # Require trend alignment
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'ltf_fast': ('int', 5, 20),
            'ltf_slow': ('int', 20, 60),
            'htf_period': ('int', 50, 200),
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        # LTF (Lower Timeframe) signals - fast/slow SMA crossover
        ltf_fast_sma = close.rolling(params['ltf_fast']).mean()
        ltf_slow_sma = close.rolling(params['ltf_slow']).mean()
        
        # LTF crossover signals
        ltf_bullish_cross = (ltf_fast_sma > ltf_slow_sma) & (ltf_fast_sma.shift(1) <= ltf_slow_sma.shift(1))
        ltf_bearish_cross = (ltf_fast_sma < ltf_slow_sma) & (ltf_fast_sma.shift(1) >= ltf_slow_sma.shift(1))
        
        # HTF (Higher Timeframe) trend filter
        htf_sma = close.rolling(params['htf_period']).mean()
        htf_bullish = close > htf_sma  # Price above HTF SMA = bullish regime
        htf_bearish = close < htf_sma  # Price below HTF SMA = bearish regime
        
        # Apply HTF filter if enabled
        if params.get('htf_filter', True):
            if params.get('require_alignment', True):
                # Only long when HTF is bullish, only short when HTF is bearish
                entries = ltf_bullish_cross & htf_bullish
                short_entries = ltf_bearish_cross & htf_bearish
            else:
                # Just use HTF as additional confirmation
                entries = ltf_bullish_cross & (htf_bullish | ltf_fast_sma > ltf_slow_sma)
                short_entries = ltf_bearish_cross & (htf_bearish | ltf_fast_sma < ltf_slow_sma)
        else:
            # No filter - just use LTF signals
            entries = ltf_bullish_cross
            short_entries = ltf_bearish_cross
        
        # Exit signals
        exits = ltf_bearish_cross | (htf_bullish.shift(1) & ~htf_bullish)  # LTF bearish or HTF flip
        short_exits = ltf_bullish_cross | (htf_bearish.shift(1) & ~htf_bearish)
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'ltf_fast': ltf_fast_sma,
                'ltf_slow': ltf_slow_sma,
                'htf_sma': htf_sma,
                'htf_bullish': htf_bullish.astype(int),
                'htf_bearish': htf_bearish.astype(int)
            }
        )


class SessionBasedStrategy(VectorBTStrategy):
    """
    Session-Aware Trading Strategy
    
    Concept: Different trading sessions have distinct behavioral patterns.
    - Asian session (00:00-08:00 UTC): Range-bound, mean reversion works
    - London session (08:00-16:00 UTC): Breakouts, trends start here
    - NY session (13:00-21:00 UTC): Continuation or reversal patterns
    
    The strategy adapts its approach based on the detected session:
    - 'breakout' mode: Trade breakouts during London/NY, avoid in Asian
    - 'reversion' mode: Trade mean reversion during Asian, avoid in London/NY
    
    Note: If datetime index is available, uses hour-of-day directly.
    Otherwise, uses rolling volatility patterns as a proxy for session detection
    (Asian sessions typically have lower volatility).
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'asian_start': 0,          # UTC hour start of Asian session
            'asian_end': 8,            # UTC hour end of Asian session
            'london_start': 8,         # UTC hour start of London session
            'london_end': 16,          # UTC hour end of London session
            'ny_start': 13,            # UTC hour start of NY session
            'ny_end': 21,              # UTC hour end of NY session
            'session_mode': 'breakout', # 'breakout' or 'reversion'
            'volatility_lookback': 24,  # Bars for volatility calculation
            'breakout_period': 20,      # Period for breakout levels
            'reversion_std': 2.0        # Std devs for reversion bands
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'volatility_lookback': ('int', 12, 48),
            'breakout_period': ('int', 10, 40),
            'reversion_std': ('float', 1.5, 3.0)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        
        n = len(close)
        
        # Try to extract hour from datetime index
        try:
            if hasattr(data.index, 'hour'):
                hour = data.index.hour
            elif hasattr(data.index, 'to_series'):
                hour = pd.to_datetime(data.index).hour
            else:
                hour = None
        except Exception:
            hour = None
        
        # Detect session (or use volatility proxy)
        if hour is not None:
            # Direct session detection from timestamp
            is_asian = (hour >= params['asian_start']) & (hour < params['asian_end'])
            is_london = (hour >= params['london_start']) & (hour < params['london_end'])
            is_ny = (hour >= params['ny_start']) & (hour < params['ny_end'])
        else:
            # Volatility-based proxy for session detection
            # Calculate rolling volatility (ATR-like)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            
            rolling_vol = tr.rolling(params['volatility_lookback']).mean()
            vol_percentile = rolling_vol.rolling(params['volatility_lookback'] * 3).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
                raw=False
            ).fillna(0.5)
            
            # Low volatility = Asian-like, High volatility = London/NY-like
            is_asian = vol_percentile < 0.33
            is_london = (vol_percentile >= 0.33) & (vol_percentile < 0.66)
            is_ny = vol_percentile >= 0.66
        
        # Convert to Series if needed
        is_asian = pd.Series(is_asian, index=data.index)
        is_london = pd.Series(is_london, index=data.index)
        is_ny = pd.Series(is_ny, index=data.index)
        
        # Calculate breakout levels
        highest = high.rolling(params['breakout_period']).max()
        lowest = low.rolling(params['breakout_period']).min()
        
        # Calculate reversion levels (Bollinger-style)
        sma = close.rolling(params['breakout_period']).mean()
        std = close.rolling(params['breakout_period']).std()
        upper_band = sma + params['reversion_std'] * std
        lower_band = sma - params['reversion_std'] * std
        
        # Breakout signals
        breakout_long = close > highest.shift(1)
        breakout_short = close < lowest.shift(1)
        
        # Reversion signals
        reversion_long = close < lower_band
        reversion_short = close > upper_band
        
        if params['session_mode'] == 'breakout':
            # Trade breakouts in London/NY, avoid during Asian
            active_session = is_london | is_ny
            entries = breakout_long & active_session
            short_entries = breakout_short & active_session
            exits = breakout_short | is_asian  # Exit if entering Asian session
            short_exits = breakout_long | is_asian
        else:  # 'reversion' mode
            # Trade mean reversion in Asian, avoid during London/NY
            active_session = is_asian
            entries = reversion_long & active_session
            short_entries = reversion_short & active_session
            exits = (close > sma) | is_london  # Exit near mean or if London starts
            short_exits = (close < sma) | is_london
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'highest': highest,
                'lowest': lowest,
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'is_asian': is_asian.astype(int),
                'is_london': is_london.astype(int),
                'is_ny': is_ny.astype(int)
            }
        )


class OrderBookImbalanceStrategy(VectorBTStrategy):
    """
    Order Book Imbalance Simulation Strategy
    
    Concept: Trade based on buy/sell pressure imbalance. Without real L2/L3
    order book data, we simulate order flow using price action and volume:
    
    - Buying pressure: Close near the high of the bar + elevated volume
    - Selling pressure: Close near the low of the bar + elevated volume
    - Imbalance ratio signals directional bias
    
    Price position in range formula: (close - low) / (high - low)
    - 1.0 = closed at the high (strong buying)
    - 0.0 = closed at the low (strong selling)
    - 0.5 = closed at the midpoint (neutral)
    
    When this pressure is confirmed by above-average volume, it indicates
    real conviction behind the move.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'lookback': 14,              # Bars for average calculations
            'imbalance_threshold': 0.7,  # 70% buy or sell pressure threshold
            'volume_confirm': True,      # Require volume confirmation
            'volume_mult': 1.5,          # Volume must be N times average
            'smoothing': 3               # Smooth the imbalance reading
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'lookback': ('int', 7, 28),
            'imbalance_threshold': ('float', 0.6, 0.85),
            'volume_mult': ('float', 1.0, 2.5),
            'smoothing': ('int', 1, 7)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        high = self._get_high(data)
        low = self._get_low(data)
        volume = self._get_volume(data)
        
        # Calculate price position within bar range (0 = low, 1 = high)
        bar_range = high - low
        # Avoid division by zero for doji candles
        bar_range = bar_range.replace(0, np.nan)
        price_position = (close - low) / bar_range
        price_position = price_position.fillna(0.5)  # Neutral for doji
        
        # Smooth the price position to reduce noise
        if params['smoothing'] > 1:
            smoothed_position = price_position.rolling(params['smoothing']).mean()
        else:
            smoothed_position = price_position
        
        # Calculate rolling average for trend
        avg_position = smoothed_position.rolling(params['lookback']).mean()
        
        # Detect strong buying/selling pressure
        buying_pressure = smoothed_position > params['imbalance_threshold']
        selling_pressure = smoothed_position < (1 - params['imbalance_threshold'])
        
        # Volume confirmation
        avg_volume = volume.rolling(params['lookback']).mean()
        high_volume = volume > (avg_volume * params['volume_mult'])
        
        # Calculate cumulative imbalance (like delta/cumulative volume delta)
        # +1 for buying pressure bars, -1 for selling pressure bars
        pressure_sign = np.where(smoothed_position > 0.5, 1, -1)
        weighted_pressure = pressure_sign * (volume / avg_volume)
        cumulative_imbalance = pd.Series(weighted_pressure, index=data.index).rolling(params['lookback']).sum()
        
        if params['volume_confirm']:
            # Require volume confirmation
            entries = buying_pressure & high_volume & (cumulative_imbalance > 0)
            short_entries = selling_pressure & high_volume & (cumulative_imbalance < 0)
        else:
            # Pure price position based
            entries = buying_pressure & (cumulative_imbalance > 0)
            short_entries = selling_pressure & (cumulative_imbalance < 0)
        
        # Exit when imbalance flips
        exits = selling_pressure | (cumulative_imbalance < 0)
        short_exits = buying_pressure | (cumulative_imbalance > 0)
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'price_position': smoothed_position,
                'avg_position': avg_position,
                'cumulative_imbalance': cumulative_imbalance,
                'high_volume': high_volume.astype(int),
                'buying_pressure': buying_pressure.astype(int),
                'selling_pressure': selling_pressure.astype(int)
            }
        )


class KellyCriterionOverlay(VectorBTStrategy):
    """
    Kelly Criterion Position Sizing Overlay
    
    Concept: Calculate optimal position size based on historical win rate and
    payoff ratio. This is not a signal generator but a sizing overlay that
    returns a position size multiplier (0 to kelly_fraction).
    
    Kelly Formula: K% = W - [(1-W) / R]
    Where:
    - W = Win probability (historical win rate)
    - R = Win/Loss ratio (average win / average loss)
    
    The result is the optimal fraction of capital to bet. Since full Kelly
    is aggressive and volatility-inducing, we use fractional Kelly (typically
    half-Kelly or quarter-Kelly) for more stable equity curves.
    
    Usage: Combine with any other strategy. The 'entries' signal indicates
    bars where position sizing is recommended (Kelly > 0). The 'kelly_size'
    indicator provides the exact position size multiplier.
    
    Note: For proper Kelly calculation, this strategy needs a base signal
    to calculate historical performance on. It uses momentum as the base.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'lookback': 50,           # Bars to calculate win rate from
            'kelly_fraction': 0.5,    # Half-Kelly for safety (0.25 = quarter)
            'min_trades': 10,         # Min trades before using Kelly
            'max_position': 1.0,      # Maximum position size cap
            'signal_period': 10,      # Period for base momentum signal
            'signal_threshold': 0.0   # Momentum threshold for base signal
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'lookback': ('int', 30, 100),
            'kelly_fraction': ('float', 0.25, 0.75),
            'min_trades': ('int', 5, 20),
            'max_position': ('float', 0.5, 1.5),
            'signal_period': ('int', 5, 20)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        params = self.validate_params(params)
        close = self._get_close(data)
        
        n = len(close)
        
        # Generate base signal (momentum-based)
        momentum = close.pct_change(params['signal_period'])
        base_long_signal = momentum > params['signal_threshold']
        base_short_signal = momentum < -params['signal_threshold']
        
        # Calculate forward returns for win/loss analysis
        forward_return = close.shift(-1) / close - 1
        
        # Calculate rolling win rate and payoff ratio
        # A "win" is when signal direction matches next bar's return direction
        long_correct = (base_long_signal & (forward_return > 0)).astype(float)
        long_incorrect = (base_long_signal & (forward_return <= 0)).astype(float)
        short_correct = (base_short_signal & (forward_return < 0)).astype(float)
        short_incorrect = (base_short_signal & (forward_return >= 0)).astype(float)
        
        # Combine long and short
        wins = long_correct + short_correct
        losses = long_incorrect + short_incorrect
        total_signals = wins + losses
        
        # Rolling sums
        rolling_wins = wins.rolling(params['lookback']).sum()
        rolling_losses = losses.rolling(params['lookback']).sum()
        rolling_total = total_signals.rolling(params['lookback']).sum()
        
        # Win rate
        win_rate = rolling_wins / rolling_total.replace(0, np.nan)
        win_rate = win_rate.fillna(0.5)  # Default to 50% if no data
        
        # Average win and loss sizes
        win_returns = forward_return.where(wins > 0, np.nan).abs()
        loss_returns = forward_return.where(losses > 0, np.nan).abs()
        
        avg_win = win_returns.rolling(params['lookback']).mean().fillna(0.01)
        avg_loss = loss_returns.rolling(params['lookback']).mean().fillna(0.01)
        
        # Payoff ratio (R)
        payoff_ratio = avg_win / avg_loss.replace(0, np.nan)
        payoff_ratio = payoff_ratio.fillna(1.0)
        
        # Kelly Criterion: K = W - (1-W)/R
        kelly = win_rate - (1 - win_rate) / payoff_ratio
        
        # Apply fractional Kelly and cap
        kelly_sized = kelly * params['kelly_fraction']
        kelly_sized = kelly_sized.clip(lower=0, upper=params['max_position'])
        
        # Only recommend sizing when we have enough historical data
        enough_trades = rolling_total >= params['min_trades']
        kelly_sized = kelly_sized.where(enough_trades, 0)
        
        # Entry: Kelly suggests a positive position and base signal is active
        entries = (kelly_sized > 0) & base_long_signal & enough_trades
        short_entries = (kelly_sized > 0) & base_short_signal & enough_trades
        
        # Exit when Kelly goes negative or base signal flips
        exits = (kelly_sized <= 0) | (~base_long_signal & base_long_signal.shift(1))
        short_exits = (kelly_sized <= 0) | (~base_short_signal & base_short_signal.shift(1))
        
        return SignalOutput(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            indicators={
                'kelly_size': kelly_sized,
                'win_rate': win_rate,
                'payoff_ratio': payoff_ratio,
                'raw_kelly': kelly,
                'momentum': momentum,
                'enough_trades': enough_trades.astype(int)
            }
        )


# ============================================================================
# STRATEGY COMBINATION UTILITIES
# ============================================================================

def combine_strategies(
    base_signals: SignalOutput,
    filter_signals: SignalOutput,
    mode: str = 'and'
) -> SignalOutput:
    """
    Combine two strategy signals for composite trading systems.
    
    This utility allows you to layer strategies together:
    - Use a trend strategy as a filter for a mean-reversion strategy
    - Combine multiple confirmation signals
    - Create ensemble strategies
    
    Parameters:
    -----------
    base_signals : SignalOutput
        The primary strategy signals (entries/exits)
    filter_signals : SignalOutput  
        The secondary strategy signals used for filtering
    mode : str
        Combination mode:
        - 'and': Both strategies must agree (most conservative)
        - 'or': Either strategy signals (most aggressive)
        - 'filter': Base signals filtered by filter's trend direction
                   (uses entries as bullish bias, short_entries as bearish)
        - 'confirm': Base entry must occur within N bars of filter entry
    
    Returns:
    --------
    SignalOutput
        Combined signals with merged indicators
    
    Example:
    --------
    >>> macd = MACDStrategy().generate_signals(data, {})
    >>> rsi = RSIStrategy().generate_signals(data, {})
    >>> combined = combine_strategies(macd, rsi, mode='and')
    """
    if mode == 'and':
        # Both must agree - most conservative
        entries = base_signals.entries & filter_signals.entries
        exits = base_signals.exits | filter_signals.exits  # Exit if either says exit
        short_entries = base_signals.short_entries & filter_signals.short_entries
        short_exits = base_signals.short_exits | filter_signals.short_exits
        
    elif mode == 'or':
        # Either signals - most aggressive
        entries = base_signals.entries | filter_signals.entries
        exits = base_signals.exits & filter_signals.exits  # Only exit if both say exit
        short_entries = base_signals.short_entries | filter_signals.short_entries
        short_exits = base_signals.short_exits & filter_signals.short_exits
        
    elif mode == 'filter':
        # Base signals filtered by filter's directional bias
        # filter.entries = True means filter is bullish (allow longs)
        # filter.short_entries = True means filter is bearish (allow shorts)
        
        # Create trend state from filter signals
        # Once entry fires, stay in that bias until opposite fires
        filter_long_bias = filter_signals.entries.astype(int).replace(0, np.nan)
        filter_short_bias = filter_signals.short_entries.astype(int).replace(0, np.nan)
        
        # Forward fill the bias
        is_bullish = filter_long_bias.ffill().fillna(0).astype(bool)
        is_bearish = filter_short_bias.ffill().fillna(0).astype(bool)
        
        # Only allow base longs when filter is bullish, base shorts when filter is bearish
        entries = base_signals.entries & is_bullish
        short_entries = base_signals.short_entries & is_bearish
        exits = base_signals.exits | is_bearish  # Exit longs if filter turns bearish
        short_exits = base_signals.short_exits | is_bullish
        
    elif mode == 'confirm':
        # Base entry must be within N bars of filter entry
        confirm_window = 3  # Could make this a parameter
        
        # Rolling sum to detect recent filter signals
        filter_long_recent = filter_signals.entries.astype(int).rolling(confirm_window).sum() > 0
        filter_short_recent = filter_signals.short_entries.astype(int).rolling(confirm_window).sum() > 0
        
        entries = base_signals.entries & filter_long_recent
        short_entries = base_signals.short_entries & filter_short_recent
        exits = base_signals.exits
        short_exits = base_signals.short_exits
        
    else:
        raise ValueError(f"Unknown combination mode: {mode}. Use 'and', 'or', 'filter', or 'confirm'.")
    
    # Merge indicators from both strategies
    merged_indicators = {}
    if base_signals.indicators:
        for k, v in base_signals.indicators.items():
            merged_indicators[f'base_{k}'] = v
    if filter_signals.indicators:
        for k, v in filter_signals.indicators.items():
            merged_indicators[f'filter_{k}'] = v
    
    return SignalOutput(
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        indicators=merged_indicators
    )


def create_filtered_strategy(
    base_strategy: VectorBTStrategy,
    filter_strategy: VectorBTStrategy,
    mode: str = 'filter'
) -> Callable[[pd.DataFrame, Dict[str, Any]], SignalOutput]:
    """
    Create a new signal function that combines a base strategy with a filter.
    
    Parameters:
    -----------
    base_strategy : VectorBTStrategy
        The primary strategy for entry/exit signals
    filter_strategy : VectorBTStrategy
        The filter/confirmation strategy
    mode : str
        Combination mode (see combine_strategies)
    
    Returns:
    --------
    Callable
        A signal function compatible with create_custom_strategy()
    
    Example:
    --------
    >>> signal_fn = create_filtered_strategy(
    ...     RSIStrategy(),
    ...     MultiTimeframeStrategy(),
    ...     mode='filter'
    ... )
    >>> MyFilteredRSI = create_custom_strategy(
    ...     'FilteredRSI',
    ...     {**RSIStrategy.get_params(), **MultiTimeframeStrategy.get_params()},
    ...     {**RSIStrategy.get_param_space(), **MultiTimeframeStrategy.get_param_space()},
    ...     signal_fn
    ... )
    """
    def combined_signal_fn(data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
        # Generate signals from both strategies
        base_signals = base_strategy.generate_signals(data, params)
        filter_signals = filter_strategy.generate_signals(data, params)
        
        # Combine them
        return combine_strategies(base_signals, filter_signals, mode)
    
    return combined_signal_fn


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

# Register all strategies
STRATEGY_REGISTRY = {
    # Existing
    'EmaCrossStrategy': EMACrossStrategy,
    'RSIStrategy': RSIStrategy,
    'MACDStrategy': MACDStrategy,
    'BollingerBandsStrategy': BollingerBandsStrategy,
    
    # Classic (5)
    'ChannelStrategy': ChannelStrategy,
    'IchimokuStrategy': IchimokuStrategy,
    'MaCrossStrategy': MaCrossStrategy,
    'FernandoStrategy': FernandoStrategy,
    'MACDStrategyX': MACDStrategyX,
    
    # Scalping (5)
    'VWAPStrategy': VWAPStrategy,
    'ScalpRSIStrategy': ScalpRSIStrategy,
    'MomentumBreakoutStrategy': MomentumBreakoutStrategy,
    'StochRSIStrategy': StochRSIStrategy,
    'EMARibbonStrategy': EMARibbonStrategy,
    
    # Derivatives (8)
    'FundingRateArbitrage': FundingRateArbitrage,
    'BasisTradingStrategy': BasisTradingStrategy,
    'OpenInterestDivergence': OpenInterestDivergence,
    'LiquidationHuntStrategy': LiquidationHuntStrategy,
    'GridTradingStrategy': GridTradingStrategy,
    'VolatilityBreakoutStrategy': VolatilityBreakoutStrategy,
    'TrendFollowingATR': TrendFollowingATR,
    'MeanReversionBands': MeanReversionBands,
    
    # Research Paper Strategies (3)
    'CointegrationPairsStrategy': CointegrationPairsStrategy,
    'MACDDivergenceStrategy': MACDDivergenceStrategy,
    'WhaleActivityStrategy': WhaleActivityStrategy,
    
    # Advanced Strategies (4)
    'MultiTimeframeStrategy': MultiTimeframeStrategy,
    'SessionBasedStrategy': SessionBasedStrategy,
    'OrderBookImbalanceStrategy': OrderBookImbalanceStrategy,
    'KellyCriterionOverlay': KellyCriterionOverlay,
}

# Register all strategies with the adapter
for name, cls in STRATEGY_REGISTRY.items():
    StrategyAdapter.register(name, cls)


def create_custom_strategy(
    name: str,
    default_params: Dict[str, Any],
    param_space: Dict[str, Tuple],
    signal_fn: Callable[[pd.DataFrame, Dict[str, Any]], SignalOutput]
) -> Type[VectorBTStrategy]:
    """Factory function to create a custom VectorBT strategy from a signal function."""
    class CustomStrategy(VectorBTStrategy):
        _name = name
        _default_params = default_params
        _param_space = param_space
        _signal_fn = staticmethod(signal_fn)
        
        @staticmethod
        def get_params():
            return CustomStrategy._default_params.copy()
        
        @staticmethod
        def get_param_space():
            return CustomStrategy._param_space.copy()
        
        def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> SignalOutput:
            params = self.validate_params(params)
            return CustomStrategy._signal_fn(data, params)
    
    CustomStrategy.__name__ = name
    return CustomStrategy


class SupertrendStrategy(VectorBTStrategy):
    """
    Supertrend Strategy - ATR-based trend following indicator.
    
    Supertrend = popular indicator that plots above/below price:
    - When price > Supertrend line = Bullish (line is green, below price)
    - When price < Supertrend line = Bearish (line is red, above price)
    
    Calculation:
    - Basic Upper Band = (High + Low) / 2 + Multiplier × ATR
    - Basic Lower Band = (High + Low) / 2 - Multiplier × ATR
    - Supertrend flips when price crosses the band
    
    Very effective for trending markets, reduces whipsaws vs simple MA.
    """
    
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {
            'atr_period': 10,
            'multiplier': 3.0,
            'use_close': True  # Use close for direction, else hl2
        }
    
    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {
            'atr_period': ('int', 7, 21),
            'multiplier': ('float', 2.0, 5.0)
        }
    
    def generate_signals(self, data: pd.DataFrame, params: Dict) -> SignalOutput:
        atr_period = params.get('atr_period', 10)
        multiplier = params.get('multiplier', 3.0)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr
        
        # Initialize supertrend arrays
        supertrend = np.zeros(len(data))
        direction = np.zeros(len(data))  # 1 = up (bullish), -1 = down (bearish)
        
        # First valid index
        first_valid = atr_period
        supertrend[:first_valid] = np.nan
        direction[:first_valid] = 1
        
        # Calculate Supertrend with proper band logic
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        for i in range(first_valid, len(data)):
            # Final upper band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            
            # Final lower band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
            
            # Determine direction
            if direction[i-1] == 1:  # Was bullish
                if close.iloc[i] < final_lower.iloc[i]:
                    direction[i] = -1
                    supertrend[i] = final_upper.iloc[i]
                else:
                    direction[i] = 1
                    supertrend[i] = final_lower.iloc[i]
            else:  # Was bearish
                if close.iloc[i] > final_upper.iloc[i]:
                    direction[i] = 1
                    supertrend[i] = final_lower.iloc[i]
                else:
                    direction[i] = -1
                    supertrend[i] = final_upper.iloc[i]
        
        direction = pd.Series(direction, index=data.index)
        supertrend = pd.Series(supertrend, index=data.index)
        
        # Signals on direction change
        entries = (direction == 1) & (direction.shift(1) == -1)
        exits = (direction == -1) & (direction.shift(1) == 1)
        short_entries = (direction == -1) & (direction.shift(1) == 1)
        short_exits = (direction == 1) & (direction.shift(1) == -1)
        
        return SignalOutput(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
            indicators={
                'supertrend': supertrend,
                'direction': direction,
                'atr': atr,
                'upper_band': final_upper,
                'lower_band': final_lower
            }
        )


# Update registry
STRATEGY_REGISTRY['SupertrendStrategy'] = SupertrendStrategy
