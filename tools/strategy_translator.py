#!/usr/bin/env python3
"""
Strategy Translator - Convert vectorbt strategies to Freqtrade v3 format

Usage:
    python strategy_translator.py OICVDDivergence --output ./backend/strategies/momentum/
    python strategy_translator.py --list  # List available vectorbt strategies
    python strategy_translator.py --all   # Convert all strategies
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime

# Template for Freqtrade v3 strategy
FREQTRADE_TEMPLATE = '''"""
{strategy_name} Strategy - Auto-generated from vectorbt

Generated: {timestamp}
Source: vectorbt strategy from maestro research

Original description:
{description}
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class {class_name}(IStrategy):
    """
    {strategy_name} - Converted from vectorbt backtester
    
    {description}
    """
    
    INTERFACE_VERSION = 3
    
    # Strategy settings
    timeframe = '{timeframe}'
    can_short = True
    
    # Minimal ROI - let strategy decide exits
    minimal_roi = {{
        "0": 0.15,
        "30": 0.10,
        "60": 0.05,
        "120": 0.02
    }}
    
    # Stoploss
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    
    # Parameters (adjust based on optimization)
{parameters}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate indicators needed for the strategy."""
        
{indicators}
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions."""
        
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
{entry_conditions}
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions."""
        
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
{exit_conditions}
        
        return dataframe
'''


# Strategy definitions - maps vectorbt strategy to Freqtrade components
STRATEGY_DEFINITIONS = {
    'OIDivergence': {
        'description': '''OI Divergence Strategy - Detects divergence between price and open interest.
    
    Bullish: Price makes lower low while OI stays elevated (accumulation)
    Bearish: Price makes higher high while OI depressed (distribution)
    
    Requires OI data to be merged with OHLCV.''',
        'timeframe': '1d',
        'parameters': '''    # Lookback period for swing detection
    lookback = IntParameter(10, 20, default=14, space='buy')
    
    # OI position threshold
    oi_bull_threshold = DecimalParameter(0.3, 0.5, default=0.4, space='buy')
    oi_bear_threshold = DecimalParameter(0.5, 0.7, default=0.6, space='sell')''',
        'indicators': '''        # Price swing points
        dataframe['price_low'] = dataframe['close'].rolling(self.lookback.value).min()
        dataframe['price_high'] = dataframe['close'].rolling(self.lookback.value).max()
        
        # OI data (must be merged externally or use proxy)
        # Using OBV as proxy if OI not available
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_low'] = dataframe['obv'].rolling(self.lookback.value).min()
        dataframe['obv_high'] = dataframe['obv'].rolling(self.lookback.value).max()
        
        # OI/OBV position (0-1 range)
        obv_range = dataframe['obv_high'] - dataframe['obv_low']
        dataframe['obv_position'] = (dataframe['obv'] - dataframe['obv_low']) / obv_range.replace(0, np.nan)
        
        # Price at extremes
        dataframe['at_price_low'] = dataframe['close'] <= dataframe['price_low'] * 1.02
        dataframe['at_price_high'] = dataframe['close'] >= dataframe['price_high'] * 0.98''',
        'entry_conditions': '''        # Bullish divergence: price at lows, OI/OBV elevated
        dataframe.loc[
            (dataframe['at_price_low']) &
            (dataframe['obv_position'] > self.oi_bull_threshold.value),
            'enter_long'
        ] = 1
        
        # Bearish divergence: price at highs, OI/OBV depressed
        dataframe.loc[
            (dataframe['at_price_high']) &
            (dataframe['obv_position'] < self.oi_bear_threshold.value),
            'enter_short'
        ] = 1''',
        'exit_conditions': '''        # Exit long when OBV position drops
        dataframe.loc[
            (dataframe['obv_position'] < 0.3),
            'exit_long'
        ] = 1
        
        # Exit short when OBV position rises
        dataframe.loc[
            (dataframe['obv_position'] > 0.7),
            'exit_short'
        ] = 1'''
    },
    
    'OICVDDivergence': {
        'description': '''OI + CVD Divergence Strategy - Detects divergence between OI and cumulative volume delta.
    
    Squeeze Setup (Bullish): OI increasing but CVD declining = shorts being added
    Flush Setup (Bearish): OI increasing but CVD rising = longs being added
    Capitulation (Bullish): OI decreasing sharply with CVD crashing = panic bottom
    
    Best on legacy coins: LTC (+269%), TRX (+83%), BNB (+56%), BTC (+25%)''',
        'timeframe': '1d',
        'parameters': '''    # Lookback for momentum calculation
    lookback = IntParameter(5, 15, default=10, space='buy')
    
    # OI change thresholds
    oi_increase_pct = DecimalParameter(0.03, 0.10, default=0.05, space='buy')
    oi_decrease_pct = DecimalParameter(-0.15, -0.05, default=-0.10, space='buy')''',
        'indicators': '''        # CVD (Cumulative Volume Delta) - approximation using candle position
        range_size = dataframe['high'] - dataframe['low']
        close_position = (dataframe['close'] - dataframe['low']) / range_size.replace(0, np.nan)
        delta_ratio = 2 * close_position.fillna(0.5) - 1  # -1 to +1
        dataframe['volume_delta'] = delta_ratio * dataframe['volume']
        dataframe['cvd'] = dataframe['volume_delta'].cumsum()
        
        # CVD momentum
        dataframe['cvd_ma'] = dataframe['cvd'].rolling(self.lookback.value).mean()
        dataframe['cvd_declining'] = dataframe['cvd'] < dataframe['cvd_ma']
        dataframe['cvd_rising'] = dataframe['cvd'] > dataframe['cvd_ma']
        
        # Volume as OI proxy (or merge real OI data)
        dataframe['vol_ma'] = dataframe['volume'].rolling(self.lookback.value * 2).mean()
        dataframe['vol_ratio'] = dataframe['volume'].rolling(self.lookback.value).mean() / dataframe['vol_ma']
        dataframe['vol_increasing'] = dataframe['vol_ratio'] > 1.2
        dataframe['vol_decreasing'] = dataframe['vol_ratio'] < 0.8
        
        # CVD crash detection
        cvd_change = dataframe['cvd'].diff(self.lookback.value)
        dataframe['cvd_crashing'] = cvd_change < cvd_change.rolling(50).quantile(0.1)''',
        'entry_conditions': '''        # Squeeze setup: volume increasing + CVD declining = short squeeze potential
        dataframe.loc[
            (dataframe['vol_increasing']) &
            (dataframe['cvd_declining']),
            'enter_long'
        ] = 1
        
        # Capitulation: volume decreasing + CVD crashing = panic bottom
        dataframe.loc[
            (dataframe['vol_decreasing']) &
            (dataframe['cvd_crashing']),
            'enter_long'
        ] = 1
        
        # Flush setup: volume increasing + CVD rising = long flush potential
        dataframe.loc[
            (dataframe['vol_increasing']) &
            (dataframe['cvd_rising']) &
            ~(dataframe['cvd_crashing']),  # Don't short capitulation
            'enter_short'
        ] = 1''',
        'exit_conditions': '''        # Exit long when CVD starts rising strongly
        dataframe.loc[
            (dataframe['cvd_rising']) &
            (dataframe['vol_ratio'] < 1.0),
            'exit_long'
        ] = 1
        
        # Exit short when CVD starts declining
        dataframe.loc[
            (dataframe['cvd_declining']),
            'exit_short'
        ] = 1'''
    },
    
    'OISqueeze': {
        'description': '''Short Squeeze Detection via OI + CVD.
    
    Detects when shorts are being added (OI up, CVD down) setting up a squeeze.
    Long-only strategy targeting short squeezes.''',
        'timeframe': '1d',
        'parameters': '''    lookback = IntParameter(5, 15, default=10, space='buy')''',
        'indicators': '''        # Same as OICVDDivergence
        range_size = dataframe['high'] - dataframe['low']
        close_position = (dataframe['close'] - dataframe['low']) / range_size.replace(0, np.nan)
        delta_ratio = 2 * close_position.fillna(0.5) - 1
        dataframe['volume_delta'] = delta_ratio * dataframe['volume']
        dataframe['cvd'] = dataframe['volume_delta'].cumsum()
        dataframe['cvd_ma'] = dataframe['cvd'].rolling(self.lookback.value).mean()
        dataframe['cvd_declining'] = dataframe['cvd'] < dataframe['cvd_ma']
        
        dataframe['vol_ma'] = dataframe['volume'].rolling(self.lookback.value * 2).mean()
        dataframe['vol_increasing'] = dataframe['volume'].rolling(self.lookback.value).mean() > dataframe['vol_ma'] * 1.2''',
        'entry_conditions': '''        # Squeeze setup only
        dataframe.loc[
            (dataframe['vol_increasing']) &
            (dataframe['cvd_declining']),
            'enter_long'
        ] = 1''',
        'exit_conditions': '''        # Exit when squeeze plays out (CVD reverses up)
        dataframe.loc[
            (dataframe['cvd'] > dataframe['cvd_ma'] * 1.1),
            'exit_long'
        ] = 1'''
    },
    
    'OICapitulation': {
        'description': '''Capitulation Bottom Detection via OI + CVD.
    
    Detects panic selling (OI crashing, CVD crashing) for bottom fishing.
    Long-only contrarian strategy.''',
        'timeframe': '1d',
        'parameters': '''    lookback = IntParameter(5, 15, default=10, space='buy')''',
        'indicators': '''        # CVD calculation
        range_size = dataframe['high'] - dataframe['low']
        close_position = (dataframe['close'] - dataframe['low']) / range_size.replace(0, np.nan)
        delta_ratio = 2 * close_position.fillna(0.5) - 1
        dataframe['volume_delta'] = delta_ratio * dataframe['volume']
        dataframe['cvd'] = dataframe['volume_delta'].cumsum()
        
        # Volume decreasing (positions closing)
        dataframe['vol_ma'] = dataframe['volume'].rolling(self.lookback.value * 2).mean()
        dataframe['vol_decreasing'] = dataframe['volume'].rolling(self.lookback.value).mean() < dataframe['vol_ma'] * 0.8
        
        # CVD crash detection
        cvd_change = dataframe['cvd'].diff(self.lookback.value)
        dataframe['cvd_crashing'] = cvd_change < cvd_change.rolling(50).quantile(0.1)''',
        'entry_conditions': '''        # Capitulation: panic exit
        dataframe.loc[
            (dataframe['vol_decreasing']) &
            (dataframe['cvd_crashing']),
            'enter_long'
        ] = 1''',
        'exit_conditions': '''        # Exit when volume normalizes
        dataframe.loc[
            (dataframe['volume'] > dataframe['vol_ma']),
            'exit_long'
        ] = 1'''
    },
    
    'FundingRate': {
        'description': '''Funding Rate Arbitrage Strategy.
    
    Negative funding = overleveraged shorts = long opportunity
    High positive funding = overleveraged longs = short opportunity''',
        'timeframe': '1h',
        'parameters': '''    short_threshold = DecimalParameter(-0.001, -0.0001, default=-0.0003, space='buy')
    long_threshold = DecimalParameter(0.0003, 0.001, default=0.0005, space='sell')''',
        'indicators': '''        # Funding rate proxy using momentum + volume
        returns = dataframe['close'].pct_change(8)
        vol_ratio = dataframe['volume'] / dataframe['volume'].rolling(24).mean()
        dataframe['funding_proxy'] = (returns * vol_ratio * 0.1).rolling(3).mean()''',
        'entry_conditions': '''        # Negative funding = long
        dataframe.loc[
            (dataframe['funding_proxy'] < self.short_threshold.value),
            'enter_long'
        ] = 1
        
        # High positive funding = short
        dataframe.loc[
            (dataframe['funding_proxy'] > self.long_threshold.value),
            'enter_short'
        ] = 1''',
        'exit_conditions': '''        # Exit when funding normalizes
        dataframe.loc[
            (dataframe['funding_proxy'] > 0) &
            (dataframe['funding_proxy'] < self.long_threshold.value / 2),
            'exit_long'
        ] = 1
        
        dataframe.loc[
            (dataframe['funding_proxy'] < 0) &
            (dataframe['funding_proxy'] > self.short_threshold.value / 2),
            'exit_short'
        ] = 1'''
    },
    
    'BollingerBreakout': {
        'description': '''Bollinger Bands Breakout Strategy.
    
    Classic breakout strategy using Bollinger Bands.
    Long on upper band breakout, short on lower band breakout.''',
        'timeframe': '1d',
        'parameters': '''    bb_period = IntParameter(15, 25, default=20, space='buy')
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space='buy')''',
        'indicators': '''        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=self.bb_period.value, nbdevup=self.bb_std.value, nbdevdn=self.bb_std.value)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        
        # Band width for volatility
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']''',
        'entry_conditions': '''        # Breakout above upper band
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upper']) &
            (dataframe['close'].shift(1) <= dataframe['bb_upper'].shift(1)),
            'enter_long'
        ] = 1
        
        # Breakout below lower band
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lower']) &
            (dataframe['close'].shift(1) >= dataframe['bb_lower'].shift(1)),
            'enter_short'
        ] = 1''',
        'exit_conditions': '''        # Exit long at middle band
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_middle']),
            'exit_long'
        ] = 1
        
        # Exit short at middle band
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_middle']),
            'exit_short'
        ] = 1'''
    },
    
    'VolatilityRegime': {
        'description': '''Volatility Regime Filter Strategy.
    
    Adapts trading based on volatility environment:
    - Low vol: Mean reversion
    - High vol: Breakout/momentum''',
        'timeframe': '4h',
        'parameters': '''    atr_period = IntParameter(10, 20, default=14, space='buy')
    regime_lookback = IntParameter(60, 120, default=90, space='buy')''',
        'indicators': '''        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Rolling percentile for regime
        dataframe['vol_percentile'] = dataframe['atr_pct'].rolling(self.regime_lookback.value).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Regime classification
        dataframe['low_vol'] = dataframe['vol_percentile'] < 0.25
        dataframe['high_vol'] = dataframe['vol_percentile'] > 0.75
        
        # Mean reversion indicators
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['zscore'] = (dataframe['close'] - dataframe['sma']) / dataframe['close'].rolling(20).std()
        
        # Breakout indicators
        dataframe['high_20'] = dataframe['high'].rolling(20).max()
        dataframe['low_20'] = dataframe['low'].rolling(20).min()''',
        'entry_conditions': '''        # Low vol: Mean reversion
        dataframe.loc[
            (dataframe['low_vol']) &
            (dataframe['zscore'] < -2),
            'enter_long'
        ] = 1
        
        dataframe.loc[
            (dataframe['low_vol']) &
            (dataframe['zscore'] > 2),
            'enter_short'
        ] = 1
        
        # High vol: Breakout
        dataframe.loc[
            (dataframe['high_vol']) &
            (dataframe['close'] >= dataframe['high_20']),
            'enter_long'
        ] = 1
        
        dataframe.loc[
            (dataframe['high_vol']) &
            (dataframe['close'] <= dataframe['low_20']),
            'enter_short'
        ] = 1''',
        'exit_conditions': '''        # Mean reversion exit at zero
        dataframe.loc[
            (dataframe['low_vol']) &
            (dataframe['zscore'].abs() < 0.5),
            ['exit_long', 'exit_short']
        ] = 1
        
        # Breakout exit on pullback
        dataframe.loc[
            (dataframe['high_vol']) &
            (dataframe['close'] < dataframe['sma']),
            'exit_long'
        ] = 1
        
        dataframe.loc[
            (dataframe['high_vol']) &
            (dataframe['close'] > dataframe['sma']),
            'exit_short'
        ] = 1'''
    },
}


def to_class_name(strategy_name: str) -> str:
    """Convert strategy name to valid Python class name."""
    # Remove special chars, convert to PascalCase
    name = re.sub(r'[^a-zA-Z0-9]', '', strategy_name)
    return name


def translate_strategy(strategy_name: str, output_dir: Path) -> str:
    """Translate a vectorbt strategy to Freqtrade format."""
    
    if strategy_name not in STRATEGY_DEFINITIONS:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_DEFINITIONS.keys())}")
    
    defn = STRATEGY_DEFINITIONS[strategy_name]
    class_name = to_class_name(strategy_name)
    
    code = FREQTRADE_TEMPLATE.format(
        strategy_name=strategy_name,
        class_name=class_name,
        timestamp=datetime.now().isoformat(),
        description=defn['description'],
        timeframe=defn['timeframe'],
        parameters=defn['parameters'],
        indicators=defn['indicators'],
        entry_conditions=defn['entry_conditions'],
        exit_conditions=defn['exit_conditions'],
    )
    
    # Write to file
    output_file = output_dir / f"{strategy_name.lower()}_translated.py"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(code)
    
    return str(output_file)


def list_strategies():
    """List all available strategies for translation."""
    print("Available vectorbt strategies for translation:")
    print("=" * 50)
    for name, defn in STRATEGY_DEFINITIONS.items():
        desc = defn['description'].split('\n')[0][:60]
        print(f"  {name:20} - {desc}")


def main():
    parser = argparse.ArgumentParser(description="Translate vectorbt strategies to Freqtrade")
    parser.add_argument('strategy', nargs='?', help='Strategy name to translate')
    parser.add_argument('--output', '-o', default='./backend/strategies/translated/',
                        help='Output directory for translated strategies')
    parser.add_argument('--list', '-l', action='store_true', help='List available strategies')
    parser.add_argument('--all', '-a', action='store_true', help='Translate all strategies')
    
    args = parser.parse_args()
    
    if args.list:
        list_strategies()
        return
    
    output_dir = Path(args.output)
    
    if args.all:
        print(f"Translating all {len(STRATEGY_DEFINITIONS)} strategies...")
        for name in STRATEGY_DEFINITIONS:
            path = translate_strategy(name, output_dir)
            print(f"  ✅ {name} -> {path}")
        print(f"\nDone! {len(STRATEGY_DEFINITIONS)} strategies translated to {output_dir}")
        return
    
    if not args.strategy:
        parser.print_help()
        return
    
    path = translate_strategy(args.strategy, output_dir)
    print(f"✅ Translated {args.strategy} -> {path}")


if __name__ == '__main__':
    main()
